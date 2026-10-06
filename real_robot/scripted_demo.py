"""CureWM-Real scripted demo collection — generated pick-and-place trajectories.

Replaces VR teleop for demo collection: smooth, deterministic, near-100%%-success
trajectories executed at 15 Hz in absolute cartesian space, recorded in the
standard CureWM episode format (same cameras, locks, phase structure). The
perturbation/replay pipeline consumes these identically to teleoped demos.

One interactive session does everything (single RobotEnv, single launch —
respects the one-shot zerorpc rule):

  1. Calibration (first run, or --recal): keyboard-jog the arm to teach spots.
       w/s = +x/-x   a/d = +y/-y   r/f = +z/-z     (robot follows each press)
       1 / 2        = step 1 cm / 2 mm
       g            = save current position as a CUP spot (repeat for several)
       p            = save current position as the PLATE spot
       q            = finish calibration
     Teach 4-6 cup spots spread over the table and one plate spot; put the
     gripper fingertips AT GRASP HEIGHT around an actually-placed cup.
  2. Collection: per episode the script names a cup spot, you place the cup
     there, press Enter, the robot does the rest; afterwards Enter=success/
     save, f=failure/discard, r=redo, q=quit.

Run IN YOUR OWN TERMINAL (interactive):
  bash real_robot/restart_stack.sh      # start a fresh server first
  conda run --no-capture-output -n robot python scripted_demo.py \
      --task-id T1 --task "pick up the paper cup and place it in the bowl" --num 30
  add --val for the held-out split; --recal to redo calibration; --dry to only
  print a generated trajectory's stats (no robot).
"""
from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONTROL_HZ, DATA_ROOT, EpisodeBuffer, annotate_phases,  # noqa: E402
                    configure_cameras, episode_name, grab_views, lock_cameras,
                    log_session)

def spots_file(task_id):
    return DATA_ROOT / f"spots_{task_id}.json"
HOVER_DZ = 0.13          # hover height above grasp/place points (m)
PLACE_DZ = 0.06          # release height above the plate point (m)
GRASP_HOLD = 16          # frames closed & still after grasp (~1 s, phase gate)
OPEN_HOLD = 8            # frames after release before retreat
MAX_STEP_M = 0.04        # generator hard cap on per-frame motion


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def _smooth(n):
    """Cosine ease-in-out fractions for n steps (excluding start point)."""
    t = np.linspace(0, 1, max(int(n), 2) + 1)[1:]
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _seg(p0, p1, seconds, rng, jitter=0.0):
    """Linear-in-space, cosine-in-time segment p0->p1. Optional mid jitter."""
    n = max(3, int(round(seconds * CONTROL_HZ)))
    fr = _smooth(n)
    pts = p0[None, :] + fr[:, None] * (p1 - p0)[None, :]
    if jitter > 0:
        bump = rng.uniform(-jitter, jitter, size=3) * np.sin(np.pi * fr)[:, None]
        pts = pts + bump
    return pts


def _smooth_noise(T, sigma, rng, k=9):
    """Temporally smoothed gaussian noise (T,3) — human-tremor-like drift."""
    n = rng.normal(0, sigma, (T, 3))
    ker = np.ones(k) / k
    for c in range(3):
        n[:, c] = np.convolve(n[:, c], ker, mode="same")
    return n


def gen_trajectory(cup, plate, rpy, rng, home_xyz=None, noise=1.0, speed=1.0):
    """Action stream (T,7) for one pick-and-place demo, with per-episode variety.

    cup/plate: xyz of grasp point and plate centre. rpy: fixed tool orientation.
    home_xyz: episode start (robot home pose); trajectory begins there.
    """
    v = lambda lo, hi: rng.uniform(lo, hi)  # noqa: E731
    cup = np.asarray(cup, float).copy()
    plate = np.asarray(plate, float).copy()
    cup[:2] += rng.uniform(-0.003, 0.003, 2)          # grasp tolerance
    plate[:2] += rng.uniform(-0.015, 0.015, 2)        # place variety
    hover_c = cup + [0, 0, HOVER_DZ + v(-0.02, 0.03)]
    hover_p = plate + [0, 0, HOVER_DZ + v(-0.02, 0.03)]
    place = plate + [0, 0, PLACE_DZ + v(-0.005, 0.01)]
    if home_xyz is None:
        start = hover_c + [v(-0.02, 0.02), v(-0.02, 0.02), v(0.0, 0.04)]
        t_start = v(0.8, 1.3) / speed
    else:
        start = np.asarray(home_xyz, float)
        t_start = max(1.2, np.linalg.norm(hover_c - start) / (0.18 * speed)) * v(0.95, 1.2)

    s_ = speed
    segs = [
        (_seg(start, hover_c, t_start, rng), 0.0),
        (_seg(hover_c, cup, v(1.2, 1.8) / s_, rng), 0.0),           # descend
        (np.repeat(cup[None, :], GRASP_HOLD, 0), 1.0),               # close & settle
        (_seg(cup, hover_c, v(1.0, 1.5) / s_, rng), 1.0),            # lift
        (_seg(hover_c, hover_p, v(1.5, 2.4) / s_, rng, jitter=0.02 * min(noise, 1.0)), 1.0),
        (_seg(hover_p, place, v(1.0, 1.6) / s_, rng), 1.0),          # lower
        (np.repeat(place[None, :], OPEN_HOLD, 0), 0.0),              # open
        (_seg(place, hover_p, v(0.8, 1.2) / s_, rng), 0.0),          # retreat
    ]
    xyz = np.concatenate([s for s, _ in segs])
    grip = np.concatenate([np.full(len(s), g) for s, g in segs])
    T = len(xyz)
    rpy_stream = np.repeat(np.asarray(rpy, float)[None, :], T, axis=0)

    if noise > 0:
        # Per-frame envelope: noise everywhere by default, zero through the grasp window (the end
        # of the descent plus the closed-gripper hold) to keep precision, and halved while placing
        seg_lens = [len(s) for s, _ in segs]
        b = np.cumsum([0] + seg_lens)          # segment boundary frame indices
        w = np.ones(T)
        w[max(0, b[2] - 10):b[3] + 3] = 0.0    # from 10 frames before the grasp to 3 after closing
        w[b[5]:b[7]] = 0.4                     # lowering and opening the gripper
        xyz = xyz + noise * w[:, None] * _smooth_noise(T, 0.004, rng)
        ang = noise * w[:, None] * _smooth_noise(T, np.deg2rad(1.5), rng)
        rpy_stream[:, 1:3] += ang[:, 1:3] * 0.8   # pitch and yaw only; roll is left alone (+/-pi wrap)

    acts = np.zeros((T, 7), dtype=np.float32)
    acts[:, :3] = xyz
    acts[:, 3:6] = rpy_stream
    acts[:, 6] = grip

    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    assert steps.max() <= MAX_STEP_M, f"step {steps.max():.3f}m exceeds cap"
    return acts


# ---------------------------------------------------------------------------
# Calibration (keyboard jog)
# ---------------------------------------------------------------------------

def _getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def calibrate(env, start_xyz, rpy, out_file):
    print(__doc__.split("1. Calibration", 1)[1].split("2. Collection")[0])
    print("  extra keys: u = delete the last cup spot   r = retrieval grasp height for the target spot (used by auto reset)")
    pos = np.asarray(start_xyz, float).copy()
    step = 0.01
    spots, plate = [], None
    retrieve = None
    if Path(out_file).exists():
        _old = json.load(open(out_file))
        spots = [list(s) for s in _old.get("spots", [])]
        plate = _old.get("plate")
        retrieve = _old.get("plate_grasp")
        print(f"[cal] preloaded {out_file}: {len(spots)} spots" + (" + plate" if plate else ""))
    _goto(env, pos, rpy, seconds=2.0)
    keymap = {"w": (0, +1), "s": (0, -1), "a": (1, +1), "d": (1, -1),
              "r": (2, +1), "f": (2, -1)}
    while True:
        print(f"\r pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) step={step*1000:.0f}mm "
              f"spots={len(spots)} plate={'SET' if plate is not None else '-'}   ",
              end="", flush=True)
        ch = _getch()
        if ch in keymap:
            ax, sgn = keymap[ch]
            pos[ax] += sgn * step
            _goto(env, pos, rpy, seconds=max(0.15, step / 0.05))
        elif ch == "1":
            step = 0.01
        elif ch == "2":
            step = 0.002
        elif ch == "g":
            spots.append(pos.tolist())
            print(f"\n[cal] cup spot {len(spots)} saved: {np.round(pos,3).tolist()}")
        elif ch == "u":
            if spots:
                rm = spots.pop()
                print(f"\n[cal] removed cup spot {np.round(rm,3).tolist()} ({len(spots)} left)")
        elif ch == "r":
            retrieve = pos.tolist()
            print(f"\n[cal] retrieval grasp point saved: {np.round(pos,3).tolist()}")
        elif ch == "p":
            plate = pos.tolist()
            print(f"\n[cal] plate saved: {np.round(pos,3).tolist()}")
        elif ch in ("q", "\x03"):
            if spots and plate is not None:
                break
            print("\n[cal] not enough saved yet (need at least one cup spot plus the plate). Press q again to abandon, any other key to keep calibrating")
            if _getch() in ("q", "\x03"):
                raise SystemExit("[cal] aborted — rerun with --recal")
    data = {"spots": spots, "plate": plate, "rpy": list(map(float, rpy)),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if retrieve is not None:
        data["plate_grasp"] = retrieve
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[cal] wrote {out_file} ({len(spots)} cup spots)")
    return data


def _goto(env, xyz, rpy, seconds=1.0, grip=0.0):
    """Stream a smooth move to (xyz,rpy) at 15 Hz (non-blocking steps)."""
    cur = np.asarray(env.get_state()[0]["cartesian_position"], float)
    p0 = cur[:3]
    for f in _smooth(int(seconds * CONTROL_HZ)):
        target = p0 + f * (np.asarray(xyz) - p0)
        env.step(np.concatenate([target, rpy, [grip]]))
        time.sleep(1.0 / CONTROL_HZ)


# ---------------------------------------------------------------------------
# Episode execution
# ---------------------------------------------------------------------------

def _record_tick(env, serials, buf, action, moving):
    obs = env.get_observation()
    env.step(action)
    wrist, ext = grab_views(obs, serials)
    if wrist is None or ext is None:
        return 1
    st = obs["robot_state"]
    state7 = np.array([*st["cartesian_position"], st["gripper_position"]],
                      dtype=np.float32)
    buf.add(np.asarray(action, np.float32), state7, st["joint_positions"],
            time.time(), moving, wrist, ext)
    return 0


def run_episode(env, serials, locks, acts, out_dir, meta, tail_cap_s=60):
    """Stream the trajectory, then KEEP RECORDING (arm holding still) until the
    operator answers the outcome prompt — the recording window is Enter-to-verdict,
    and the settled tail doubles as the ground-truth ending for the probes."""
    buf = EpisodeBuffer(out_dir=out_dir, meta=meta)
    period = 1.0 / CONTROL_HZ
    drops = 0
    for t in range(len(acts)):
        t0 = time.time()
        try:
            drops += _record_tick(env, serials, buf, acts[t], True)
        except Exception as e:  # noqa: BLE001
            print(f"[run] tick {t} error ({type(e).__name__}: {e})")
        el = time.time() - t0
        if el < period:
            time.sleep(period - el)

    print("    outcome? <Enter>=success/save  f=fail/discard : ", end="", flush=True)
    hold = acts[-1].copy()
    t_end = time.time() + tail_cap_s
    verdict = ""
    while True:
        t0 = time.time()
        if select.select([sys.stdin], [], [], 0)[0]:
            verdict = sys.stdin.readline().strip()
            break
        if time.time() < t_end:
            try:
                drops += _record_tick(env, serials, buf, hold, False)
            except Exception:  # noqa: BLE001
                pass
        el = time.time() - t0
        if el < period:
            time.sleep(period - el)
    return buf, drops, verdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", default="T1")
    p.add_argument("--task", default="pick up the paper cup and place it in the bowl")
    p.add_argument("--num", type=int, default=30)
    p.add_argument("--val", action="store_true")
    p.add_argument("--recal", action="store_true")
    p.add_argument("--seed", type=int, default=None,
                   help="rng seed base (default: derived from time)")
    p.add_argument("--split-name", default=None,
                   help="override split dir (e.g. icra_extra); default demos/val by --val")
    p.add_argument("--only-spot", type=int, default=None,
                   help="collect only this spot (1-indexed, as displayed)")
    p.add_argument("--per-spot", type=int, default=5,
                   help="consecutive episodes per cup spot before moving on")
    p.add_argument("--objects", default=None,
                   help="comma-separated object colors, e.g. red,green,blue.  In a multi-object scene each "
                        "demonstration rotates the target color and records layout/target_color for the wrong_target family")
    p.add_argument("--speed", type=float, default=1.0,
                   help="global speed factor; below 1 is slower and steadier")
    p.add_argument("--noise", type=float, default=1.0,
                   help="within-trajectory noise scale (0=off, 1=default ~4mm RMS)")
    p.add_argument("--dry", action="store_true", help="print a generated trajectory, no robot")
    p.add_argument("--no-hover", action="store_true",
                   help="skip the pre-episode hover over the cup spot (default: hover so the cup can be placed under the gripper)")
    p.add_argument("--hover-dz", type=float, default=0.10,
                   help="pre-episode hover height above the taught grasp point (m)")
    args = p.parse_args()

    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    rng = np.random.default_rng(seed)

    if args.dry:
        _sf = spots_file(args.task_id)
        cfg = json.load(open(_sf)) if _sf.exists() else \
            {"spots": [[0.5, 0.05, 0.20]], "plate": [0.55, -0.12, 0.20], "rpy": [np.pi, 0, 0]}
        acts = gen_trajectory(cfg["spots"][0], cfg["plate"], cfg["rpy"], rng,
                              home_xyz=[0.31, 0.0, 0.49], noise=args.noise, speed=args.speed)
        ph = annotate_phases(acts)
        steps = np.linalg.norm(np.diff(acts[:, :3], axis=0), axis=1)
        print(f"frames={len(acts)} (~{len(acts)/CONTROL_HZ:.1f}s) phases={ph}")
        print(f"max step={steps.max()*1000:.1f}mm  z range=[{acts[:,2].min():.3f},{acts[:,2].max():.3f}]")
        return

    import subprocess
    print("[init] restarting the control stack (about 20 s)...")
    rc = subprocess.run(["bash", str(Path(__file__).parent / "restart_stack.sh")]).returncode
    if rc != 0:
        raise SystemExit("[init] restart_stack failed -- check the control-box connection and retry")

    from droid.robot_env import RobotEnv

    split = args.split_name or ("val" if args.val else "demos")
    task_root = DATA_ROOT / split / args.task_id
    task_root.mkdir(parents=True, exist_ok=True)
    existing = len(list(task_root.glob("*demo_*")))
    print(f"[init] split={split}; {existing} episodes on disk; seed base={seed}")
    print("[init] RobotEnv (ROBOT MOVES TO HOME) ...")
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    serials = configure_cameras(env)
    locks = lock_cameras(env)

    home_state = env.get_state()[0]["cartesian_position"]
    rpy_default = np.asarray(home_state[3:6], float)

    SPOTS = spots_file(args.task_id)
    if SPOTS.exists() and not args.recal:
        cfg = json.load(open(SPOTS))
        print(f"[cal] loaded {SPOTS} ({len(cfg['spots'])} cup spots)")
    else:
        start = [0.50, 0.02, 0.30]
        cfg = calibrate(env, start, rpy_default, spots_file(args.task_id))
    rpy = np.asarray(cfg["rpy"], float)
    spots, plate = cfg["spots"], cfg["plate"]

    objects = [c.strip() for c in args.objects.split(",")] if args.objects else None
    saved = 0
    spot_order = list(rng.permutation(len(spots)))
    skip_off = 0
    try:
        while saved < args.num:
            if args.only_spot is not None:
                k = args.only_spot - 1
                assert 0 <= k < len(spots), f"--only-spot must be in 1..{len(spots)}"
            else:
                k = spot_order[(saved // args.per_spot + skip_off) % len(spots)]
            layout = target_color = None
            if objects:
                if len(spots) < len(objects):
                    raise SystemExit(f"[obj] need at least {len(objects)} spots, only {len(spots)} are calibrated")
                perm = list(np.random.default_rng(args.seed + 977 * saved).permutation(len(spots)))
                layout = {c: int(perm[i]) for i, c in enumerate(objects)}
                target_color = objects[saved % len(objects)]
                k = layout[target_color]
            env._robot.update_gripper(0, velocity=False, blocking=True)
            env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)
            home_xyz = np.asarray(env.get_state()[0]["cartesian_position"], float)[:3]
            acts = gen_trajectory(spots[k], plate, rpy, rng, home_xyz=home_xyz,
                                  noise=args.noise, speed=args.speed)
            if objects:
                place = "  ".join(f"{c} -> spot {layout[c]+1}" for c in objects)
                print(f"\n[obj] layout: {place}   |   target for this episode = {target_color}")
            n_here = saved % args.per_spot + 1
            print(f"\n=== {split}/{args.task_id} {saved+1}/{args.num} — "
                  f"CUP on spot {k+1}/{len(spots)} ({np.round(spots[k][:2],3).tolist()}) "
                  f"[episode {n_here}/{args.per_spot}] ===")
            hovered = False
            if not args.no_hover:
                # Show the operator where the spot is: open gripper hovers over the taught grasp
                # point; the cup goes directly under the fingertips.  Not recorded.
                _goto(env, [spots[k][0], spots[k][1], spots[k][2] + args.hover_dz], rpy, seconds=3.0)
                hovered = True
                print(f"    the gripper is hovering {100*args.hover_dz:.0f} cm above cup spot {k+1} -- place the cup directly under the fingertips")
            ans = input("    <Enter>=go   s=next spot   q=quit : ").strip()
            if hovered:
                env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)   # back home before recording
            if ans == "q":
                break
            if ans == "s":
                skip_off += 1
                continue
            ep_task = (f"pick up the {target_color} cube and place it on the plate"
                       if objects else args.task)
            meta = {"kind": "demo", "source": "scripted", "split": split,
                    "task_id": args.task_id, "task": ep_task,
                    "control_hz": CONTROL_HZ, "cameras": serials,
                    "camera_lock": locks, "spot": int(k), "traj_seed": int(seed)}
            if objects:
                meta.update({"objects": objects, "layout": layout,
                             "target_color": target_color})
            out = task_root / episode_name("demo", args.task_id)
            buf, drops, lab = run_episode(env, serials, locks, acts, out, meta)
            if drops:
                print(f"[run] {drops} camera drops")
            if lab == "" and len(buf) > 0:
                meta["outcome"] = "success"
                path = buf.save()
                saved += 1
                print(f"[run] SAVED {path.name} ({len(buf)} frames)")
                log_session({"event": "scripted_demo_saved", "path": str(path),
                             "spot": int(k), "split": split})
            else:
                print("[run] discarded")
    finally:
        print(f"\n[done] saved {saved} episodes this run ({existing + saved} total)")
        env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)
        env._robot.update_gripper(0, velocity=False, blocking=True)


if __name__ == "__main__":
    main()
