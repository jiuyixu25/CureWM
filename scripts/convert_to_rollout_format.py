"""Convert frame-bearing engine output into the LIBERO-Cosmos-Policy rollout channel format (the treatment data).

Official rollout format, verified against all_episodes/: one hdf5 per episode, with
  top level: actions (T,7) float64 | proprio (T,9) float64 (grip2, pos3, quat4)
        primary_images_jpeg (T,) vlen-uint8 | wrist_images_jpeg (T,) vlen-uint8
  attrs: success (bool), task_description (str)
Filename: episode_data--suite=<suite>--<stamp>--task=<k>--ep=<n>--success=<Bool>--regen_demo.hdf5
The dataloader reads attrs rather than the filename. The value target is the MC return and
terminal is the success flag.

Input: engine output carrying frames/wrist_frames.  Older batches have no
libero_actions key in the npz and are converted on the fly from the engine's convention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import io

VLEN_U8 = h5py.special_dtype(vlen=np.dtype("uint8"))


def jpeg_bytes(img: np.ndarray, quality: int = 95) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def convert_dir(src: Path, dst: Path, suite: str, stamp: str, only_counterfactual: bool,
                ep_counter: dict) -> int:
    rows = [json.loads(l) for l in open(src / "index.jsonl", encoding="utf-8")]
    n = 0
    for r in rows:
        if only_counterfactual and r["family"] == "nominal":
            continue
        with np.load(src / "episodes" / f"{r['name']}.npz") as z:
            if "frames" not in z or "wrist_frames" not in z:
                continue  # batches without frames do not belong in the rollout channel
            a = np.asarray(z["actions"], dtype=np.float64)
            a[:, 6] = 1.0 - 2.0 * a[:, 6]  # engine grip [0,1] -> LIBERO [+1 closed, -1 open]
            prop = np.asarray(z["proprio"], dtype=np.float64)
            prop = np.concatenate([prop[:, 7:9], prop[:, 0:3], prop[:, 3:7]], axis=1)
            frames, wrist = z["frames"], z["wrist_frames"]
        task_key = r["task"]
        k = ep_counter.setdefault(task_key, 0)
        ep_counter[task_key] = k + 1
        success = bool(r["outcome"])
        name = (f"episode_data--suite={suite}--{stamp}--task={task_key.split('/t')[-1]}"
                f"--ep={k}--success={success}--regen_demo.hdf5")
        with h5py.File(dst / name, "w") as h:
            h.create_dataset("actions", data=a)
            h.create_dataset("proprio", data=prop)
            dj = h.create_dataset("primary_images_jpeg", (len(frames),), dtype=VLEN_U8)
            dw = h.create_dataset("wrist_images_jpeg", (len(wrist),), dtype=VLEN_U8)
            for t in range(len(frames)):
                dj[t] = jpeg_bytes(frames[t])
                dw[t] = jpeg_bytes(wrist[t])
            h.attrs["success"] = success
            h.attrs["task_description"] = r.get("language") or r.get("task_name", "").replace("_", " ")
            h.attrs["failsafe_meta"] = json.dumps(
                {kk: r.get(kk) for kk in ("name", "family", "severity", "pair_of")})
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["data/prod_libero_A", "data/prod_libero_B"])
    ap.add_argument("--dst", default="data/counterfactual_rollouts")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--stamp", default="2026_07_17-failsafe_v1")
    ap.add_argument("--include-nominal", action="store_true")
    args = ap.parse_args()
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    ep_counter: dict = {}
    total = 0
    for s in args.src:
        n = convert_dir(Path(s), dst, args.suite, args.stamp,
                        only_counterfactual=not args.include_nominal, ep_counter=ep_counter)
        print(f"[convert] {s}: {n} episodes")
        total += n
    stats = {"total": total,
             "success": sum(1 for f in dst.glob("*.hdf5") if "success=True" in f.name),
             "failure": sum(1 for f in dst.glob("*.hdf5") if "success=False" in f.name)}
    (dst / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(f"[convert] done: {stats}")


if __name__ == "__main__":
    main()
