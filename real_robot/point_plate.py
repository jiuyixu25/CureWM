"""Physical pointer with a camera closed loop: close the gripper, lower the fingertips to ~1 cm
above the table at a robot-frame (x, y), snapshot the external camera, and hold.  While holding,
a JSON move file {"x":..,"y":..} re-targets the pointer (lift, move, descend, snapshot again).
The hold file releases it (lift, home).  REMOVE THE PLATE FIRST.

    conda run -n robot python point_plate.py --x 0.53 --y 0.24 --out /tmp/point --hold-file /tmp/point/done
Snapshots: <out>/snap_home.jpg (arm at home), <out>/snap_0.jpg, snap_1.jpg ... (arm pointing).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONTROL_HZ, configure_cameras, grab_views  # noqa: E402
from scripted_demo import _goto, spots_file  # noqa: E402


def touch_down(env, x, y, rpy, z_start, z_min=0.02, step=0.0006, err_thresh=0.008):
    """Guarded vertical move: lower the closed fingertips ~9 mm/s until the measured z lags the
    target by err_thresh (contact), then return the measured contact height.  None if z_min reached."""
    z = z_start
    while z > z_min:
        z -= step
        env.step(np.concatenate([[x, y, z], rpy, [1.0]]))
        time.sleep(1.0 / CONTROL_HZ)
        zm = float(env.get_state()[0]["cartesian_position"][2])
        if zm - z > err_thresh:
            return zm
    return None


def snap(env, serials, path, tries=60):
    """Save the external view. The D415 stream needs a moment after start-up/motion, so retry."""
    ext = None
    for _ in range(tries):
        _, ext = grab_views(env.get_observation(), serials)
        if ext is not None:
            break
        time.sleep(0.1)
    if ext is None:
        print(f"[point] snapshot FAILED (no external frame) -> {path}", flush=True)
        return False
    cv2.imwrite(str(path), ext)
    print(f"[point] snapshot -> {path}", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", default="T1")
    p.add_argument("--x", type=float, required=True)
    p.add_argument("--y", type=float, required=True)
    p.add_argument("--z", type=float, default=0.175, help="fingertip height (m); plate-surface point taught at 0.17")
    p.add_argument("--out", required=True)
    p.add_argument("--hold-file", required=True)
    p.add_argument("--move-file", default=None, help="JSON {x,y}; default <out>/move.json")
    p.add_argument("--hold-timeout", type=float, default=1200.0)
    p.add_argument("--touch", action="store_true", help="after reaching --z, descend slowly until contact, then hold --gap above it")
    p.add_argument("--gap", type=float, default=0.007)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    move_file = Path(args.move_file) if args.move_file else out / "move.json"
    rpy = np.asarray(json.load(open(spots_file(args.task_id)))["rpy"], float)

    print("[init] restarting the control stack (about 20 s)...")
    if subprocess.run(["bash", str(Path(__file__).parent / "restart_stack.sh")]).returncode != 0:
        raise SystemExit("[init] restart_stack failed")
    from droid.robot_env import RobotEnv
    print("[init] RobotEnv (ROBOT MOVES TO HOME) ...")
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    serials = configure_cameras(env)
    t_w = time.time()                                    # 3 s camera warm-up (as lock_cameras does before recording)
    while time.time() - t_w < 3.0:
        env.get_observation(); time.sleep(0.1)
    x, y, k = args.x, args.y, 0
    try:
        env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)
        snap(env, serials, out / "snap_home.jpg")
        env._robot.update_gripper(1, velocity=False, blocking=True)          # close
        time.sleep(1.0)

        def point_at(x, y):
            above = [x, y, args.z + 0.10]
            _goto(env, above, rpy, seconds=4.0, grip=1.0)
            _goto(env, [x, y, args.z], rpy, seconds=4.0, grip=1.0)
            time.sleep(1.5)                                       # let the controller settle, then report
            st = np.asarray(env.get_state()[0]["cartesian_position"], float)
            print(f"[point] commanded xyz=({x:.3f},{y:.3f},{args.z:.3f})  reached xyz=({st[0]:.3f},{st[1]:.3f},{st[2]:.3f})"
                  f"  dz={100*(st[2]-args.z):+.1f} cm", flush=True)

        point_at(x, y)
        if args.touch:
            zc = touch_down(env, x, y, rpy, args.z)
            if zc is None:
                print("[point] no contact down to z_min, holding at z_min + gap", flush=True); zc = 0.02
            args.z = zc + args.gap
            _goto(env, [x, y, args.z], rpy, seconds=1.5, grip=1.0)
            time.sleep(1.0)
            st = np.asarray(env.get_state()[0]["cartesian_position"], float)
            print(f"[point] CONTACT at z={zc:.3f}; now holding {100*args.gap:.1f} cm above it: reached xyz=({st[0]:.3f},{st[1]:.3f},{st[2]:.3f})", flush=True)
        snap(env, serials, out / f"snap_{k}.jpg")
        print(f"[point] FINGERTIPS at x={x:.3f} y={y:.3f} z={args.z:.3f}, waiting for move file {move_file} "
              f"or release file {args.hold_file}", flush=True)
        t0 = time.time()
        while time.time() - t0 < args.hold_timeout:
            if Path(args.hold_file).exists():
                print("[point] release signal received", flush=True); break
            if move_file.exists():
                try:
                    mv = json.load(open(move_file)); move_file.unlink()
                    nx, ny = float(mv.get("x", x)), float(mv.get("y", y)); nz = float(mv.get("z", args.z)); k += 1
                    if abs(nx - x) < 1e-4 and abs(ny - y) < 1e-4:
                        _goto(env, [x, y, nz], rpy, seconds=max(2.0, 40 * abs(nz - args.z)), grip=1.0)  # vertical only, slow
                    else:
                        _goto(env, [x, y, args.z + 0.10], rpy, seconds=2.0, grip=1.0)             # lift straight up
                        x, y = nx, ny
                        _goto(env, [x, y, args.z + 0.10], rpy, seconds=4.0, grip=1.0)
                        _goto(env, [x, y, nz], rpy, seconds=4.0, grip=1.0)
                    args.z = nz
                    time.sleep(1.5)
                    st = np.asarray(env.get_state()[0]["cartesian_position"], float)
                    snap(env, serials, out / f"snap_{k}.jpg")
                    print(f"[point] FINGERTIPS commanded ({x:.3f},{y:.3f},{args.z:.3f}) reached ({st[0]:.3f},{st[1]:.3f},{st[2]:.3f})", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[point] bad move file: {e}", flush=True)
            time.sleep(0.5)
        else:
            print("[point] hold timeout", flush=True)
        _goto(env, [x, y, args.z + 0.10], rpy, seconds=3.0, grip=1.0)        # lift straight up first
    finally:
        print("[point] returning home")
        env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)
        env._robot.update_gripper(0, velocity=False, blocking=True)


if __name__ == "__main__":
    main()
