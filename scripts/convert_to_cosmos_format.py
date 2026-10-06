"""Convert the engine's training-format output into the official LIBERO-Cosmos-Policy hdf5 layout.

Official layout, verified against the released data: <suite>_regen/<task_name>_demo.hdf5
  data/demo_i/actions      (T,7)  float64  native LIBERO actions (grip: +1 closed, -1 open)
  data/demo_i/robot_states (T,9)  float64  ordered (gripper_qpos2, eef_pos3, eef_quat4)
  data/demo_i/states       (T,D)  float64  full MuJoCo state
  data/demo_i/rewards      (T,)   uint8    all zero, last step 1 only on success
  data/demo_i/dones        (T,)   uint8    last step only

By default only counterfactual (perturbed) trajectories are written.  A failure gets
all-zero rewards; a perturbed replay that happened to succeed gets rewards[-1]=1 from its
verified outcome.  That is where the treatment data's value supervision comes from.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def reorder_proprio_to_official(p: np.ndarray) -> np.ndarray:
    """(pos3, quat4, grip2) -> (grip2, pos3, quat4)"""
    return np.concatenate([p[:, 7:9], p[:, 0:3], p[:, 3:7]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/train_fmt_v1")
    ap.add_argument("--dst", default="data/failure_regen/libero_goal_failure")
    ap.add_argument("--include-nominal", action="store_true",
                    help="also write the nominal replays (by default only counterfactuals)")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(src / "index.jsonl", encoding="utf-8")]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["family"] == "nominal" and not args.include_nominal:
            continue
        by_task[r["task_name"]].append(r)

    manifest = {}
    for task_name, eps in sorted(by_task.items()):
        out_f = dst / f"{task_name}_demo.hdf5"
        n_fail = n_lucky = 0
        with h5py.File(out_f, "w") as h:
            g = h.create_group("data")
            for i, r in enumerate(eps):
                with np.load(src / "episodes" / f"{r['name']}.npz") as z:
                    la = np.asarray(z["libero_actions"], dtype=np.float64)
                    rs = reorder_proprio_to_official(np.asarray(z["proprio"], dtype=np.float64))
                    st = np.asarray(z["sim_states"], dtype=np.float64)
                    rw = np.asarray(z["rewards"], dtype=np.uint8)
                    dn = np.asarray(z["dones"], dtype=np.uint8)
                d = g.create_group(f"demo_{i}")
                d.create_dataset("actions", data=la)
                d.create_dataset("robot_states", data=rs)
                d.create_dataset("states", data=st)
                d.create_dataset("rewards", data=rw)
                d.create_dataset("dones", data=dn)
                d.attrs["source"] = json.dumps(
                    {k: r.get(k) for k in ("name", "family", "severity", "outcome", "pair_of")})
                if r["outcome"]:
                    n_lucky += 1
                else:
                    n_fail += 1
        manifest[task_name] = {"episodes": len(eps), "failures": n_fail, "lucky_success": n_lucky}
        print(f"[convert] {task_name}: {len(eps)} eps ({n_fail} fail / {n_lucky} lucky) -> {out_f.name}")

    (dst / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    tot = sum(m["episodes"] for m in manifest.values())
    print(f"[convert] total {tot} episodes across {len(manifest)} tasks -> {dst}")


if __name__ == "__main__":
    main()
