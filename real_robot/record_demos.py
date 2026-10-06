"""CureWM-Real demo recorder — Quest-3 teleop, both D415s, CureWM episode format.

Mirrors the proven loop of scripts/tests/collect_lerobot_dataset.py (grip-gated
recording, tap-to-toggle gripper intent, post-success grace tail) but writes
the CureWM episode layout (common.py) with per-frame JOINT positions — replay
needs them to reset exactly — and locks AE/AWB before the first frame.

Quest bindings (right controller default):
    hold G  = enable robot + record        trigger tap = toggle gripper
    A       = save episode (success)       B           = discard
Robot moves to home on env construction and between episodes.

Usage:
  conda run -n robot python record_demos.py --task-id T1 \
      --task "pick up the paper cup and place it in the bowl" --num 30
  add --val for the held-out split (never enters any training pool).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONTROL_HZ, DATA_ROOT, EpisodeBuffer, configure_cameras,  # noqa: E402
                    episode_name, grab_views, lock_cameras, log_session, recover_camera)

CAM_RECOVER_AFTER = 15   # consecutive dropped-frame ticks (~1 s) before the camera pipeline is restarted

TRIGGER_RISE, TRIGGER_FALL = 0.6, 0.3


def reset_gripper_last(env):
    """Home first, THEN open — env.reset() opens mid-air and drops held objects."""
    env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)
    env._robot.update_gripper(0, velocity=False, blocking=True)


def record_episode(env, controller, serials, out_dir, meta, max_steps, grace_frames=15, auto=False):
    buf = EpisodeBuffer(out_dir=out_dir, meta=meta)
    reset_gripper_last(env)
    controller.reset_state()
    if auto:
        print("[demo] AUTO mode: reposition the object, then HOLD G to start "
              "moving+recording; trigger=toggle gripper; A=save, B=discard", flush=True)
    else:
        input("[demo] place the object (vary position across demos), <Enter> to arm teleop ...")
        print("[demo] hold G to move+record; trigger=toggle gripper; A=save, B=discard")

    period = 1.0 / CONTROL_HZ
    gripper_intent, trigger_high = 0.0, False
    grace_count, grace_ticks = -1, 0
    drop_streak = {serials["wrist"]: 0, serials["external"]: 0}
    locks = meta.get("camera_lock")

    while True:
        loop_start = time.time()
        try:
            controller_info = controller.get_info()
            obs = env.get_observation()
            action, ctrl_info = controller.forward(obs, include_info=True)
        except Exception as e:  # noqa: BLE001
            print(f"[demo] transient read error ({type(e).__name__}: {e}); skip tick")
            _pace(loop_start, period); continue

        # Check camera health on every tick, not only while G is held: otherwise frames dropped
        # between grips would never recover.
        wrist, ext = grab_views(obs, serials)
        for serial, img in ((serials["wrist"], wrist), (serials["external"], ext)):
            if img is None:
                drop_streak[serial] += 1
                if drop_streak[serial] in (1, CAM_RECOVER_AFTER):
                    print(f"[demo] camera {serial} dropping (streak {drop_streak[serial]})")
                if drop_streak[serial] >= CAM_RECOVER_AFTER:
                    try:
                        recover_camera(env, serial, locks)
                    except Exception as e:  # noqa: BLE001
                        print(f"[demo] camera recovery failed: {type(e).__name__}: {e}")
                    drop_streak[serial] = 0
            else:
                drop_streak[serial] = 0

        if controller_info.get("movement_enabled", False):
            try:
                action_info = env.step(action)
            except Exception as e:  # noqa: BLE001
                print(f"[demo] env.step error ({type(e).__name__}: {e}); skip tick")
                _pace(loop_start, period); continue

            raw_trigger = float(ctrl_info.get("target_gripper_position", 0.0))
            if not trigger_high and raw_trigger > TRIGGER_RISE:
                trigger_high = True
                gripper_intent = 1.0 - gripper_intent
                print(f"[demo] gripper_intent -> {gripper_intent:.0f}")
            elif trigger_high and raw_trigger < TRIGGER_FALL:
                trigger_high = False

            if wrist is None or ext is None:
                pass
            else:
                st = obs["robot_state"]
                state7 = np.array([*st["cartesian_position"], st["gripper_position"]],
                                  dtype=np.float32)
                action7 = np.array([*action_info["cartesian_position"], gripper_intent],
                                   dtype=np.float32)
                buf.add(action7, state7, st["joint_positions"], time.time(), True, wrist, ext)
                if grace_count >= 0:
                    grace_count += 1

        if controller_info.get("failure"):
            print(f"[demo] DISCARDED ({len(buf)} frames)")
            return None
        if grace_count >= 0:
            grace_ticks += 1
        if grace_count >= grace_frames or (grace_count >= 0 and grace_ticks >= 3 * grace_frames):
            if len(buf) > 0:
                path = buf.save()
                print(f"[demo] SAVED {path.name} ({len(buf)} frames, ~{len(buf)/CONTROL_HZ:.1f}s)")
                return path
            print("[demo] grace ended with empty buffer; discarding")
            return None
        if grace_count < 0 and controller_info.get("success"):
            if len(buf) == 0:
                print("[demo] A pressed before any teleop; ignoring")
            else:
                grace_count = 0
                print(f"[demo] success armed — {grace_frames} tail frames, hold steady")
        if len(buf) >= max_steps:
            path = buf.save()
            print(f"[demo] SAVED at max_steps ({len(buf)} frames)")
            return path
        _pace(loop_start, period)


def _pace(loop_start, period):
    elapsed = time.time() - loop_start
    if elapsed < period:
        time.sleep(period - elapsed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True, help="short id, e.g. T1 / T2")
    p.add_argument("--task", required=True, help="natural-language task text")
    p.add_argument("--num", type=int, default=30, help="episodes to save this run")
    p.add_argument("--val", action="store_true",
                   help="record into the held-out val/ split (never used for training)")
    p.add_argument("--max-steps", type=int, default=CONTROL_HZ * 90)
    p.add_argument("--left-controller", action="store_true")
    p.add_argument("--auto", action="store_true",
                   help="no keyboard prompts: episodes arm automatically; all control via Quest buttons")
    args = p.parse_args()

    from droid.controllers.oculus_controller import VRPolicy
    from droid.robot_env import RobotEnv

    split = "val" if args.val else "demos"
    task_root = DATA_ROOT / split / args.task_id
    task_root.mkdir(parents=True, exist_ok=True)
    existing = len(list(task_root.glob("demo_*")))
    print(f"[init] split={split} task={args.task_id}; {existing} episodes already on disk")

    print("[init] RobotEnv (connects to NUC; ROBOT WILL MOVE TO HOME) ...")
    env = RobotEnv()                      # teleop drives cartesian_velocity
    serials = configure_cameras(env)
    locks = lock_cameras(env)
    controller = VRPolicy(right_controller=not args.left_controller)

    saved = 0
    try:
        while saved < args.num:
            print(f"\n=== {split}/{args.task_id} episode {saved+1}/{args.num} "
                  f"(disk total will be {existing+saved+1}) ===")
            meta = {
                "kind": "demo", "split": split, "task_id": args.task_id,
                "task": args.task, "control_hz": CONTROL_HZ,
                "cameras": serials, "camera_lock": locks,
            }
            out = task_root / episode_name("demo", args.task_id)
            path = record_episode(env, controller, serials, out, meta, args.max_steps, auto=args.auto)
            if path is not None:
                saved += 1
                log_session({"event": "demo_saved", "path": str(path),
                             "task_id": args.task_id, "split": split})
    except KeyboardInterrupt:
        print("\n[done] interrupted")
    finally:
        print(f"[done] saved {saved} episodes this run "
              f"({existing + saved} total for {split}/{args.task_id})")
        reset_gripper_last(env)


if __name__ == "__main__":
    main()
