"""Run a served VLA policy on the FR3 for the cube-pickup evaluation (runs in the `robot` env).

The policy is served by `vqb serve` (VLAQuantBench) on the laptop, full precision or any quantized
configuration, so every condition is the same checkpoint behind the same HTTP interface. This script owns the
robot side: layout (robot shuffles the cubes), the 15 Hz observe -> query -> step loop, the telemetry success
verdict (cube held through a lift), put-back, and a per-episode record identical in layout to the demos.

    python policy_client.py --server http://127.0.0.1:8010 --task-id T1C --cond fp \
        --episodes 10 --colors red,green,blue --wording "pick up the {c} cube" --auto-shuffle
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))
import auto_collect as AC  # noqa: E402
from common import (CONTROL_HZ, DATA_ROOT, EpisodeBuffer, configure_cameras, grab_views,  # noqa: E402
                    lock_cameras, log_session)
from common import SafetyEnvelope  # noqa: E402

MAX_VIOLATIONS = 30       # a policy that keeps leaving the demo envelope is off-distribution: abort the episode


def build_envelope(demo_root: Path, pattern: str = "pickup_demos_*") -> SafetyEnvelope:
    """Workspace box = union of every collected demo's end-effector path (+10 cm xy, -0.5 cm below the lowest grasp),
    per-step jump capped at 6 cm. The policy's absolute pose commands never get to leave it."""
    import glob
    xyz = []
    for d in sorted(glob.glob(str(demo_root / pattern))):
        try:
            xyz.append(np.load(f"{d}/traj.npz")["states"][:, :3])
        except Exception:
            continue
    if not xyz:
        raise SystemExit(f"no demos under {demo_root} to derive the safety envelope from")
    env = SafetyEnvelope(np.concatenate(xyz))
    print(f"safety envelope from {len(xyz)} demos: x[{env.lo[0]:.3f},{env.hi[0]:.3f}] y[{env.lo[1]:.3f},{env.hi[1]:.3f}] "
          f"z[{env.lo[2]:.3f},{env.hi[2]:.3f}] max step {env.max_step_m*100:.0f} cm")
    return env


def cube_colour_in_gripper(wrist_bgr: np.ndarray) -> str:
    """Dominant saturated colour in the gripper region of the wrist view (lower-centre of the frame).
    The wooden table (hue ~10-20) is excluded from red by the hue bound; returns 'unknown' when no cube dominates."""
    import cv2
    h, w = wrist_bgr.shape[:2]
    roi = wrist_bgr[int(0.35 * h):, int(0.25 * w):int(0.95 * w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (S > 120) & (V > 70)
    counts = {"red": int((((H < 6) | (H > 172)) & (S > 150) & (V > 80)).sum()),
              "green": int(((H > 45) & (H < 85) & sat).sum()),
              "blue": int(((H > 95) & (H < 130) & sat).sum())}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0.02 * roi.shape[0] * roi.shape[1] else "unknown"


def guard_orientation(action7: np.ndarray, prev7: np.ndarray | None) -> tuple[np.ndarray, bool]:
    """The demos keep the gripper pointing down (roll ~ +-pi, pitch ~ 0, yaw in [-0.85, -0.75]). A decoded pose
    outside that band (e.g. a roll that wandered towards 0 = a 180-degree flip) keeps the previous orientation."""
    r, pt, y = action7[3], action7[4], action7[5]
    ok = abs(r) >= np.pi - 0.5 and -0.3 <= pt <= 0.3 and -1.3 <= y <= -0.3
    if ok or prev7 is None:
        return action7, ok
    a = action7.copy(); a[3:6] = prev7[3:6]
    return a, False

LIFT_DZ = 0.06            # target counts as lifted when the gripper is closed and z rose this much above the grasp
HOLD_FRAMES = 15          # ... for ~1 s


def enc(a: np.ndarray) -> dict:
    buf = io.BytesIO(); np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
    return {"__ndarray__": base64.b64encode(buf.getvalue()).decode("ascii")}


def dec(o):
    if isinstance(o, dict) and "__ndarray__" in o:
        return np.load(io.BytesIO(base64.b64decode(o["__ndarray__"])), allow_pickle=False)
    return o


class Policy:
    def __init__(self, server: str, timeout: float = 120.0):
        self.server, self.timeout = server.rstrip("/"), timeout
        self.info = self.post("/info", {})

    def post(self, path: str, payload: dict):
        r = requests.post(self.server + path, json=payload, timeout=self.timeout); r.raise_for_status()
        return r.json()

    def reset(self, task: dict):
        self.post("/reset", {"task": task})

    def act(self, ext_rgb, wrist_rgb, state7, instruction, step, task) -> np.ndarray:
        out = self.post("/act", {"task": task, "obs": {"images": {"exterior": enc(ext_rgb), "wrist": enc(wrist_rgb)},
                                                        "instruction": instruction, "state": enc(np.asarray(state7, np.float32)),
                                                        "step": int(step), "raw": None}})
        return np.asarray(dec(out["action"]), dtype=np.float64).reshape(-1)


def run_episode(env, serials, policy, task, instruction, cubes, target, cfg, out_dir, meta, max_steps, envelope):
    import cv2
    buf = EpisodeBuffer(out_dir=out_dir, meta=meta)
    policy.reset(task)
    period = 1.0 / CONTROL_HZ; z0 = None; lifted_frames = 0; lifted_at = None; drops = 0
    violations = 0; prev_cmd = None; envelope.violations = 0; envelope._prev = None
    grasp_xy = None; lifted_colour = None
    for t in range(max_steps):
        t0 = time.time()
        obs = env.get_observation()
        wrist, ext = grab_views(obs, serials)
        if wrist is None or ext is None:
            drops += 1; time.sleep(period); continue
        st = obs["robot_state"]
        state7 = np.array([*st["cartesian_position"], st["gripper_position"]], dtype=np.float32)
        if z0 is None:
            z0 = float(state7[2])
        raw_action = policy.act(cv2.cvtColor(ext, cv2.COLOR_BGR2RGB), cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB), state7, instruction, t, task)
        v0 = envelope.violations
        action = envelope.filter(raw_action)
        action, ori_ok = guard_orientation(action, prev_cmd)
        violations += (envelope.violations - v0) + (0 if ori_ok else 1)
        prev_cmd = action
        if violations > MAX_VIOLATIONS:
            print(f"  !! {violations} envelope violations at step {t}: policy is off-distribution, aborting the episode")
            break
        env.step(action)
        buf.add(action.astype(np.float32), state7, st["joint_positions"], time.time(), True, wrist, ext)
        holding = AC.GRIP_HELD_MIN < state7[6] < AC.GRIP_HELD_MAX      # the demo SOP's "cube in the fingers" band
        z_grasp = AC.grasp_z(cfg, cubes[target]) if cubes else AC.grasp_z(cfg, state7[:2])
        if holding and grasp_xy is None:
            grasp_xy = state7[:2].copy()                                  # where the fingers closed: put-back target
        if holding and state7[2] > z_grasp + LIFT_DZ:
            lifted_frames += 1
            if lifted_frames >= HOLD_FRAMES and lifted_at is None:
                lifted_at = t; lifted_colour = cube_colour_in_gripper(wrist); break
        else:
            lifted_frames = 0
        el = time.time() - t0
        if el < period:
            time.sleep(period - el)
    return buf, lifted_at, drops, violations, grasp_xy, lifted_colour


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:8010")
    p.add_argument("--task-id", default="T1C")
    p.add_argument("--cond", required=True, help="condition label, e.g. fp | uniW4A6 | pgjd")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--colors", default="red,green,blue")
    p.add_argument("--wording", default="pick up the {c} cube", help="instruction template; {c} = target colour")
    p.add_argument("--targets", default=None, help="comma-separated target colours per episode (default: cycle)")
    p.add_argument("--max-steps", type=int, default=300, help="20 s at 15 Hz")
    p.add_argument("--speed", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--auto-shuffle", action="store_true")
    p.add_argument("--out-subdir", default="eval")
    p.add_argument("--manual-layout", action="store_true",
                   help="the person places the cubes by hand: no guided placement, no shuffling, no prompts; "
                        "the lifted cube's colour is read from the wrist camera and it is put back where it was picked")
    p.add_argument("--no-restart-stack", action="store_true", help="skip restart_stack.sh when the NUC server is already up")
    args = p.parse_args()

    colors = [c.strip() for c in args.colors.split(",")]
    cfg = json.load(open(DATA_ROOT / f"spots_{args.task_id}.json"))
    lo, hi = AC.workspace(cfg); rng = np.random.default_rng(args.seed)
    policy = Policy(args.server); print("[policy] server:", json.dumps(policy.info)[:300])
    envelope = build_envelope(DATA_ROOT / "demos" / args.task_id)

    from common import check_port
    robot_host = os.environ.get("CUREWM_ROBOT_HOST", "").rpartition("@")[2]
    robot_port = int(os.environ.get("CUREWM_ROBOT_PORT", "4242"))
    if args.no_restart_stack and robot_host and check_port(robot_host, robot_port):
        print("[init] NUC server already up, not restarting the stack")
    else:
        print("[init] restarting the control stack ...")
        if subprocess.run(["bash", str(Path(__file__).parent / "restart_stack.sh")]).returncode != 0:
            raise SystemExit("[init] restart_stack failed")
    from droid.robot_env import RobotEnv
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    serials = configure_cameras(env); locks = lock_cameras(env); AC.go_home(env)
    if args.manual_layout:
        cubes = {}                                    # unknown layout: the person placed the cubes
    else:
        start = AC.sample_positions(lo, hi, len(colors), rng); cubes = {c: start[i] for i, c in enumerate(colors)}
        for c in colors:
            AC.place_guided(env, serials, cubes[c], cfg, c)
        AC.go_home(env)

    targets = (args.targets.split(",") if args.targets else [colors[i % len(colors)] for i in range(args.episodes)])
    tally = {"success": 0, "wrong_cube": 0, "no_lift": 0, "lifted_unknown": 0}
    for ep in range(args.episodes):
        if ep > 0 and args.manual_layout:
            pass                                      # the person re-arranges between runs
        elif ep > 0:
            if args.auto_shuffle:
                ok, why = AC.shuffle_cubes(env, serials, cfg, cubes, lo, hi, rng, args.speed)
                if not ok:
                    print(f"[shuffle] failed: {why}; stopping"); break
            else:
                AC.place_all(env, serials, cfg, colors, cubes, lo, hi, rng)
        target = targets[ep]; instruction = args.wording.format(c=target)
        task = {"benchmark": "fr3", "suite": args.task_id, "task_id": colors.index(target), "task_name": f"pick_{target}",
                "instruction": instruction, "max_steps": args.max_steps,
                "extra": {"fr3_hz": CONTROL_HZ}}  # the server holds each predicted waypoint for round(0.133 s * hz) ticks
        stamp = time.strftime("%m%d_%H%M%S")
        out = DATA_ROOT / "eval" / args.task_id / args.cond / f"{args.out_subdir}_{args.cond}_{stamp}"
        meta = {"kind": "eval", "cond": args.cond, "server": args.server, "server_info": policy.info, "task_id": args.task_id,
                "task": instruction, "wording_template": args.wording, "target_color": target, "objects": colors,
                "cubes": [{"color": c, "xy": [float(v[0]), float(v[1])]} for c, v in cubes.items()],  # [] for a manual layout
                "control_hz": CONTROL_HZ, "cameras": serials, "camera_lock": locks, "max_steps": args.max_steps}
        print(f"\n=== [{args.cond}] episode {ep+1}/{args.episodes}  target={target}  '{instruction}'  " +
              ("  ".join(f"{c}({cubes[c][0]:.2f},{cubes[c][1]:.2f})" for c in colors) if cubes else "manual layout") + " ===", flush=True)
        buf, lifted_at, drops, violations, grasp_xy, lifted_colour = run_episode(env, serials, policy, task, instruction, cubes, target, cfg, out, meta, args.max_steps, envelope)
        # which cube (if any) moved: compare the final gripper xy with the layout
        st = env.get_observation()["robot_state"]; xy = np.asarray(st["cartesian_position"][:2])
        if cubes:
            nearest = min(colors, key=lambda c: np.linalg.norm(np.asarray(cubes[c]) - xy))
        else:
            nearest = lifted_colour or "unknown"      # manual layout: colour seen between the fingers at the lift
        if lifted_at is not None:
            outcome = "success" if nearest == target else ("wrong_cube" if nearest in colors else "lifted_unknown")
        else:
            outcome = "no_lift"
        tally[outcome] += 1
        meta.update({"outcome": outcome, "lifted_at_step": lifted_at, "steps": len(buf), "camera_drops": drops,
                     "nearest_cube_at_end": nearest, "envelope_violations": int(violations),
                     "lifted_colour": lifted_colour, "grasp_xy": None if grasp_xy is None else [float(v) for v in grasp_xy]})
        print(f"    {outcome} (steps {len(buf)}, lifted_at {lifted_at}, nearest {nearest}, envelope violations {violations})", flush=True)
        if lifted_at is not None:        # put the lifted cube back where it was, never go_home while holding
            back_xy = cubes[nearest] if cubes else grasp_xy
            AC.put_back(env, serials, back_xy, cfg, rng, args.speed)
        else:
            AC.go_home(env)
        buf.save()
        log_session({"event": "eval_episode", "cond": args.cond, "target": target, "outcome": outcome, "path": str(out)})
    print(f"[eval] {args.cond}: {tally}")


if __name__ == "__main__":
    main()
