"""Hover the gripper over each taught cup spot and the plate spot so the operator can re-mark
the table — no recording, no grasping.  Use only if the 08-31 tape marks are gone.

    conda run -n robot python visit_spots.py --task-id T1 [--hover 0.05]

Same stack rules as the recorders: restart_stack first (done here), one RobotEnv per launch.
Keep a hand on the e-stop; the arm moves to each spot on <Enter>.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripted_demo import _goto, spots_file  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", default="T1")
    p.add_argument("--hover", type=float, default=0.05, help="height above the taught grasp point (m)")
    p.add_argument("--plate-only", action="store_true", help="skip the cup spots, hover over the plate point only")
    p.add_argument("--hold-file", default=None,
                   help="non-interactive: hover until this file exists (or --hold-timeout), then go home")
    p.add_argument("--hold-timeout", type=float, default=600.0)
    args = p.parse_args()
    cfg = json.load(open(spots_file(args.task_id)))
    spots, plate, rpy = cfg["spots"], cfg["plate"], np.asarray(cfg["rpy"], float)
    print(f"[visit] {len(spots)} cup spots + plate from {spots_file(args.task_id)}")

    print("[init] restarting the control stack (about 20 s)...")
    if subprocess.run(["bash", str(Path(__file__).parent / "restart_stack.sh")]).returncode != 0:
        raise SystemExit("[init] restart_stack failed")
    from droid.robot_env import RobotEnv
    print("[init] RobotEnv (ROBOT MOVES TO HOME) ...")
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    try:
        env._robot.update_gripper(0, velocity=False, blocking=True)
        env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)
        targets = [] if args.plate_only else \
            [(f"cup spot {i + 1}/{len(spots)}", [s[0], s[1], s[2] + args.hover]) for i, s in enumerate(spots)]
        targets.append(("PLATE spot (release height)", [plate[0], plate[1], plate[2] + 0.06 + args.hover]))
        for name, xyz in targets:
            if args.hold_file is None:
                if input(f"[visit] next: {name} at {np.round(xyz, 3).tolist()}  <Enter>=go  q=quit : ").strip() == "q":
                    break
            _goto(env, xyz, rpy, seconds=4.0)
            if args.hold_file is None:
                input(f"[visit] hovering over {name} — mark the table, then <Enter> ")
            else:
                import time as _t
                print(f"[visit] HOVERING over {name} at {np.round(xyz, 3).tolist()} — waiting for {args.hold_file} "
                      f"(timeout {args.hold_timeout:.0f}s)", flush=True)
                t0 = _t.time()
                while not Path(args.hold_file).exists() and _t.time() - t0 < args.hold_timeout:
                    _t.sleep(0.5)
                print("[visit] release signal received" if Path(args.hold_file).exists() else "[visit] hold timeout",
                      flush=True)
    finally:
        print("[visit] returning home")
        env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)


if __name__ == "__main__":
    main()
