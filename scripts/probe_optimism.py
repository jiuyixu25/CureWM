"""Optimism probe for a released checkpoint (baseline: Cosmos-Policy-LIBERO-Predict2-2B).

For each failing counterfactual (o_{t0}, a-_{t0:t0+16}) and its nominal control
(o_{t0}, a+):
  1) get_action builds the data_batch and yields the clean latent sequence;
  2) replace_latent_with_action_chunk swaps the action slots for the given actions
     (the model's own injection function);
  3) get_future_state_prediction imagines the future frames and get_value_prediction
     returns the value.
Metrics:
  - value optimism: the distribution of V(o, a-) against V(o, a+); the fraction with
    V(o, a-) > 0.5 is the value-side hallucination rate.
  - visual optimism: compare the L2 distance from the imagined future third-person frame
    to the ground-truth nominal future against the distance to the ground-truth failure
    future; the fraction that looks more like the successful world is the visual-side
    hallucination rate.
Run from the cosmos-policy checkout, e.g.
  cd third_party/cosmos-policy && .venv/bin/python <path>/probe_optimism.py --limit 12
Use a small --limit on the first run to check the API assumptions; the script introspects
the keys get_action returns and prints them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "third_party/cosmos-policy"))

from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    get_model, load_dataset_stats, init_t5_text_embeddings_cache,
    get_future_state_prediction, get_value_prediction, resolve_path,
)
from cosmos_policy.models.policy_text2world_model import replace_latent_with_action_chunk  # noqa: E402
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig as GenerateConfig  # noqa: E402
from cosmos_policy.experiments.robot.cosmos_utils import get_action  # noqa: E402

CKPT = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"


def make_cfg() -> GenerateConfig:
    return GenerateConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=CKPT,
        config_file="cosmos_policy/config/config.py",
        use_wrist_image=True, use_proprio=True, normalize_proprio=True,
        unnormalize_actions=True,
        dataset_stats_path=f"{CKPT}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{CKPT}/libero_t5_embeddings.pkl",
        trained_with_image_aug=True, chunk_size=16, num_open_loop_steps=16,
        task_suite_name="libero_goal", randomize_seed=False, seed=195,
        deterministic=True, use_variance_scale=False,
        flip_images=False,  # our frames are already upright; do not flip twice
        ar_future_prediction=False, ar_value_prediction=False,
        num_denoising_steps_action=5, num_denoising_steps_future_state=1,
        num_denoising_steps_value=1, use_jpeg_compression=False,
    )


def reorder_proprio(p9: np.ndarray) -> np.ndarray:
    """Our storage order (eef_pos3, eef_quat4, grip2) -> the official (grip2, eef_pos3, eef_quat4)."""
    return np.concatenate([p9[7:9], p9[0:3], p9[3:7]])


def normalize_actions(a: np.ndarray, stats: dict) -> np.ndarray:
    lo, hi = np.asarray(stats["actions_min"]), np.asarray(stats["actions_max"])
    return np.clip(2.0 * (a - lo) / np.maximum(hi - lo, 1e-8) - 1.0, -1.0, 1.0)


def to_libero_actions(engine_actions: np.ndarray) -> np.ndarray:
    a = engine_actions.copy()
    a[:, 6] = 1.0 - 2.0 * a[:, 6]  # grip [0 closed, 1 open] -> LIBERO [+1 closed, -1 open]
    return a


def first_divergence(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    diff = np.any(np.abs(a[:n] - b[:n]) > 1e-6, axis=1)
    return int(np.argmax(diff)) if diff.any() else 0


def probe_one(cfg, model, stats, ep_np, t0: int, chunk16_raw: np.ndarray,
              obs: dict, language: str, seed: int) -> dict:
    """Given (obs at t0, a chosen action chunk), return the value and the imagined future
    third-person frame."""
    rd = get_action(cfg, model, stats, obs, language, seed=seed,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=False)
    if probe_one.first:
        print("[introspect] get_action keys:", sorted(rd.keys()), flush=True)
        probe_one.first = False
    db = rd["data_batch"]
    gen_latent = rd.get("generated_latent_with_action", rd.get("generated_latent"))
    orig_clean = rd.get("orig_clean_latent_frames", rd.get("orig_latent_frames"))
    a_norm = normalize_actions(chunk16_raw, stats)
    chunk_t = torch.tensor(a_norm[None], dtype=gen_latent.dtype, device=gen_latent.device)
    act_idx = db["action_latent_idx"]
    injected = replace_latent_with_action_chunk(gen_latent.clone(), chunk_t, act_idx)

    fut = get_future_state_prediction(
        cfg, model, data_batch=db, generated_latent_with_action=injected,
        orig_clean_latent_frames=orig_clean,
        future_proprio_latent_idx=int(db["future_proprio_latent_idx"][0]),
        future_wrist_image_latent_idx=int(db["future_wrist_image_latent_idx"][0]),
        future_wrist_image2_latent_idx=int(db["future_wrist_image2_latent_idx"][0]),
        future_image_latent_idx=int(db["future_image_latent_idx"][0]),
        future_image2_latent_idx=int(db["future_image2_latent_idx"][0]),
        seed=seed, num_denoising_steps_future_state=cfg.num_denoising_steps_future_state)
    val = get_value_prediction(
        cfg, model, data_batch=db,
        future_state_samples_list=fut["future_state_samples_list"],
        seed=seed, num_denoising_steps_value=cfg.num_denoising_steps_value)
    pred = fut.get("future_image_predictions") or {}
    img = pred.get("future_image")
    if isinstance(img, torch.Tensor):
        img = img.float().cpu().numpy()
    return {"value": float(val["value_prediction"]), "future_image": img}


probe_one.first = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12, help="number of failing counterfactuals to process")
    ap.add_argument("--data", default=str(Path(__file__).parent / "data/libero_failure_224_v1"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "data/probe_optimism_v0"))
    ap.add_argument("--ckpt", default=None, help="local checkpoint path; defaults to the released weights on the Hub")
    args = ap.parse_args()
    global DATA
    DATA = Path(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg()
    if args.ckpt:
        cfg.ckpt_path = args.ckpt
    model, _model_config = get_model(cfg)  # get_model returns a (model, config) tuple
    stats = load_dataset_stats(resolve_path(cfg.dataset_stats_path))
    init_t5_text_embeddings_cache(resolve_path(cfg.t5_text_embeddings_path))

    rows = [json.loads(l) for l in open(DATA / "index.jsonl", encoding="utf-8")]
    fails = [r for r in rows if r["family"] != "nominal" and r["outcome"] is False][: args.limit]
    print(f"[probe] {len(fails)} failure counterfactuals")

    results = []
    for r in fails:
        try:
            with np.load(DATA / "episodes" / f"{r['name']}.npz") as z:
                p_act, p_frames, p_wrist, p_prop = (z["actions"], z["frames"],
                                                    z["wrist_frames"], z["proprio"])
            with np.load(DATA / "episodes" / f"{r['pair_of']}.npz") as z:
                n_act, n_frames = z["actions"], z["frames"]
            t0 = max(0, first_divergence(p_act, n_act) - 1)
            t0 = min(t0, len(p_act) - 17, len(n_act) - 17)
            if t0 < 0:
                continue
            obs = {"primary_image": p_frames[t0], "wrist_image": p_wrist[t0],
                   "proprio": reorder_proprio(p_prop[t0])}
            lang = r.get("language") or r.get("task_name", "").replace("_", " ")
            pert = probe_one(cfg, model, stats, r, t0,
                             to_libero_actions(p_act[t0:t0 + 16]), obs, lang, cfg.seed)
            nom = probe_one(cfg, model, stats, r, t0,
                            to_libero_actions(n_act[t0:t0 + 16]), obs, lang, cfg.seed)
            rec = {"name": r["name"], "family": r["family"], "severity": r["severity"],
                   "t0": t0, "value_fail_action": pert["value"],
                   "value_nominal_action": nom["value"]}
            t1 = min(t0 + 16, len(p_frames) - 1, len(n_frames) - 1)
            if pert["future_image"] is not None:
                imag = np.asarray(pert["future_image"]).squeeze()
                if imag.ndim == 3 and imag.shape[0] in (1, 3):
                    imag = np.moveaxis(imag, 0, -1)
                imag = imag.astype(np.float32)
                if imag.max() <= 1.5:
                    imag = imag * 255.0
                gt_fail = p_frames[t1].astype(np.float32)
                gt_succ = n_frames[t1].astype(np.float32)
                if imag.shape == gt_fail.shape:
                    d_fail = float(np.mean((imag - gt_fail) ** 2))
                    d_succ = float(np.mean((imag - gt_succ) ** 2))
                    rec.update({"mse_to_gt_failure": d_fail, "mse_to_gt_success": d_succ,
                                "visually_optimistic": bool(d_succ < d_fail)})
                    np.savez_compressed(out / f"{r['name']}_imag.npz",
                                        imagined=imag.astype(np.uint8),
                                        gt_fail=gt_fail.astype(np.uint8),
                                        gt_succ=gt_succ.astype(np.uint8))
            results.append(rec)
            print(f"[probe] {r['name']}: V(a-)={rec['value_fail_action']:.3f} "
                  f"V(a+)={rec['value_nominal_action']:.3f} "
                  f"vis_opt={rec.get('visually_optimistic')}", flush=True)
        except Exception as e:
            import traceback
            print(f"[probe] {r['name']} ERROR: {e}", flush=True)
            traceback.print_exc()
            if len(results) == 0:
                break  # failing on the first item means the API assumptions are wrong; stop and fix

    (out / "probe_results.jsonl").write_text(
        "\n".join(json.dumps(x) for x in results), encoding="utf-8")
    if results:
        vf = np.array([x["value_fail_action"] for x in results])
        vn = np.array([x["value_nominal_action"] for x in results])
        vo = [x.get("visually_optimistic") for x in results if "visually_optimistic" in x]
        print(json.dumps({
            "n": len(results),
            "mean_value_under_failure_actions": float(vf.mean()),
            "mean_value_under_nominal_actions": float(vn.mean()),
            "value_optimism_rate(V(a-)>0.5)": float((vf > 0.5).mean()),
            "visual_optimism_rate": (float(np.mean(vo)) if vo else None),
        }, indent=2))


if __name__ == "__main__":
    main()
