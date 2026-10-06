"""CureWM-Real replay executor: nominal (R0 fidelity gate) and perturbed replays.

For each source demo, resets to the demo's recorded initial JOINT configuration,
then streams the (optionally perturbed) absolute cartesian+gripper targets at
15 Hz through RobotEnv(action_space="cartesian_position"), recording both
cameras + full state each tick. The operator labels the physical outcome at
episode end — that label is the ground truth of the whole experiment.

Modes:
  R0 gate      python replay_perturbed.py --episode <demo_dir> --nominal --repeat 3
  single       python replay_perturbed.py --episode <demo_dir> --family wrist_tilt \
                   --severity 0.8 --seed 0
  batch        python replay_perturbed.py --plan plan.json          (resumable)

Safety: xyz stays on the demo path by construction (gripper-only families) or
changes orientation only (wrist_tilt <= 15 deg); a SafetyEnvelope additionally
clamps to the demo's bounding box and limits per-step jumps. Keep a hand on
the e-stop; the operator confirms every replay before the arm moves.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONTROL_HZ, DATA_ROOT, EpisodeBuffer, SafetyEnvelope,  # noqa: E402
                    configure_cameras, grab_views, load_episode, lock_cameras,
                    log_session)
from perturb import SkipEpisode, perturb  # noqa: E402

TAIL_FRAMES = 15          # keep recording ~1 s after the last action (GT ending)
R0_REPORT = DATA_ROOT / "r0_report.jsonl"

FAM_ABBR = {"insufficient_grip": "ig", "premature_release": "pr",
            "carry_slip": "cs", "wrist_tilt": "wt", "wrong_target": "mt",
            "nominal": "nom"}


def goto_start(env, joints0):
    env._robot.update_gripper(0, velocity=False, blocking=True)
    env._robot.update_joints(joints0, velocity=False, blocking=True)
    time.sleep(0.5)


def go_home(env):
    env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)
    env._robot.update_gripper(0, velocity=False, blocking=True)


def run_replay(env, serials, demo_dir: Path, family: str, severity: float,
               seed: int, out_dir: Path, locks, ctx: dict | None = None) -> dict | None:
    """Execute one replay. Returns the saved meta dict, or None if skipped."""
    demo_meta, demo_traj = load_episode(demo_dir)
    demo_actions = demo_traj["actions"]

    if family == "nominal":
        actions, pinfo = demo_actions.copy(), {"family": "nominal"}
    else:
        try:
            actions, pinfo = perturb(demo_actions, family, severity, seed, ctx)
        except SkipEpisode as e:
            print(f"[replay] SKIP {demo_dir.name} {family}: {e}")
            log_session({"event": "replay_skip", "demo": demo_dir.name,
                         "family": family, "severity": severity, "reason": str(e)})
            return None

    spot = demo_meta.get("spot")
    layout = demo_meta.get("layout")
    print(f"\n[replay] {demo_dir.name}  family={family} sev={severity} seed={seed} "
          f"frames={len(actions)}")
    if layout:
        place = "   ".join(f"{c} -> spot {s+1}" for c, s in layout.items())
        print(f"[replay] ******  layout: {place}  ******")
        print(f"[replay]        target = {demo_meta.get('target_color')} (spot {spot+1})")
        if family == "wrong_target":
            d = pinfo.get("detail", {})
            print(f"[replay]        this episode will grasp >> {d.get('distractor')} "
                  f"(spot {(d.get('distractor_spot') or 0)+1}) << "
                  f"= the wrong target, {d.get('dist_m', 0)*100:.0f} cm away")
            print("[replay]        rule: wrong color on the plate -> f;  target color on the plate -> the redirect did not take, press d")
    else:
        spot_txt = f"cup spot {spot+1}" if isinstance(spot, int) else "see the reference image"
        print(f"[replay] ******  place the cup at >> {spot_txt} <<  ******")
    print(f"[replay] (reference image: {demo_dir/'init_scene.jpg'})")
    if isinstance(spot, int) and not layout:
        # Show the operator the spot: open gripper hovers 10 cm above the taught grasp point of the
        # source demo; the cup goes directly under the fingertips.  goto_start() re-homes afterwards.
        try:
            cfg = json.load(open(DATA_ROOT / f"spots_{demo_meta['task_id']}.json"))
            sp, rpy = cfg["spots"][spot], np.asarray(cfg["rpy"], float)
            from scripted_demo import _goto
            _goto(env, [sp[0], sp[1], sp[2] + 0.10], rpy, seconds=3.0)
            print(f"[replay] the gripper is hovering 10 cm above cup spot {spot+1} -- place the cup directly under the fingertips")
        except Exception as e:  # noqa: BLE001
            print(f"[replay] hover skipped: {e}")
    ans = input("[replay] place object, clear the arm path, <Enter>=go  s=skip  q=quit: ").strip()
    if ans == "s":
        return None
    if ans == "q":
        raise KeyboardInterrupt

    envl = SafetyEnvelope(demo_actions, extra_xyz=actions)
    goto_start(env, demo_traj["joints"][0])

    meta = {
        "kind": "replay", "source_demo": str(demo_dir), "task_id": demo_meta["task_id"],
        "task": demo_meta["task"], "split": demo_meta.get("split", "demos"),
        "control_hz": CONTROL_HZ, "cameras": serials, "camera_lock": locks,
        "perturbation": pinfo,
    }
    buf = EpisodeBuffer(out_dir=out_dir, meta=meta)
    period = 1.0 / CONTROL_HZ
    drops = 0

    for t in range(len(actions) + TAIL_FRAMES):
        loop_start = time.time()
        a = actions[min(t, len(actions) - 1)]
        a = envl.filter(np.asarray(a, dtype=np.float64))
        try:
            obs = env.get_observation()
            env.step(a)
        except Exception as e:  # noqa: BLE001
            print(f"[replay] tick {t} error ({type(e).__name__}: {e}); re-issuing next tick")
            _pace(loop_start, period)
            continue
        wrist, ext = grab_views(obs, serials)
        if wrist is None or ext is None:
            drops += 1
        else:
            st = obs["robot_state"]
            state7 = np.array([*st["cartesian_position"], st["gripper_position"]],
                              dtype=np.float32)
            buf.add(a.astype(np.float32), state7, st["joint_positions"],
                    time.time(), t < len(actions), wrist, ext)
        _pace(loop_start, period)

    if envl.violations:
        print(f"[replay] WARNING: safety envelope clamped {envl.violations} commands "
              "(should be 0 — inspect before trusting this episode)")
    if drops:
        print(f"[replay] {drops} dropped camera ticks")

    while True:
        lab = input("[replay] outcome?  s=success  f=failure  d=discard  q=quit: ").strip().lower()
        if lab in ("s", "f", "d", "q"):
            break
    if lab == "q":
        raise KeyboardInterrupt
    if lab == "d":
        print("[replay] discarded")
        log_session({"event": "replay_discard", "demo": demo_dir.name, "family": family,
                     "severity": severity, "seed": seed})
        return None

    meta["outcome"] = "success" if lab == "s" else "failure"
    note = input("[replay] note (optional): ").strip()
    if note:
        meta["note"] = note
    meta["safety_violations"] = envl.violations
    meta["camera_drops"] = drops

    # Nominal replays: tracking fidelity vs. the demo (the R0 metric).
    if family == "nominal":
        n = min(len(buf.states), len(demo_traj["states"]))
        drift = np.linalg.norm(
            np.stack(buf.states)[:n, :3] - demo_traj["states"][:n, :3], axis=1)
        meta["tracking"] = {"mean_m": float(drift.mean()), "max_m": float(drift.max()),
                            "final_m": float(drift[-1])}
        print(f"[replay] tracking drift mean={drift.mean()*100:.2f}cm "
              f"max={drift.max()*100:.2f}cm final={drift[-1]*100:.2f}cm")
        with open(R0_REPORT, "a") as f:
            f.write(json.dumps({"demo": demo_dir.name, "outcome": meta["outcome"],
                                **meta["tracking"]}) + "\n")

    path = buf.save()
    print(f"[replay] SAVED {path}")
    log_session({"event": "replay_saved", "path": str(path), "demo": demo_dir.name,
                 "family": family, "severity": severity, "seed": seed,
                 "outcome": meta["outcome"]})

    input("[replay] remove/reset the object; <Enter> to send robot home: ")
    go_home(env)
    return meta


def _pace(loop_start, period):
    elapsed = time.time() - loop_start
    if elapsed < period:
        time.sleep(period - elapsed)


def out_dir_for(demo_dir: Path, family: str, severity: float, seed: int) -> Path:
    split = "replays_val" if demo_dir.parent.parent.name == "val" else "replays"
    task_id = demo_dir.parent.name
    name = f"rep_{demo_dir.name}_{FAM_ABBR[family]}_s{severity:g}_k{seed}"
    return DATA_ROOT / split / task_id / name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", help="source demo dir (single mode)")
    p.add_argument("--family", choices=list(FAM_ABBR), default=None)
    p.add_argument("--nominal", action="store_true", help="alias for --family nominal")
    p.add_argument("--severity", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1, help="repetitions (R0 gate: 3)")
    p.add_argument("--plan", help="plan.json from make_plan.py (batch mode, resumable)")
    args = p.parse_args()

    import subprocess
    print("[init] restarting the control stack...")
    if subprocess.run(["bash", str(Path(__file__).parent / "restart_stack.sh")]).returncode != 0:
        raise SystemExit("[init] restart_stack failed")

    from droid.robot_env import RobotEnv

    jobs = []
    if args.plan:
        with open(args.plan) as f:
            plan = json.load(f)
        done = {e["out"] for e in plan.get("entries", []) if Path(e["out"]).exists()}
        for e in plan["entries"]:
            if e["out"] not in done:
                jobs.append(e)
        print(f"[plan] {len(plan['entries'])} entries, {len(done)} already done, "
              f"{len(jobs)} to run")
    else:
        assert args.episode, "--episode or --plan required"
        family = "nominal" if (args.nominal or args.family is None) else args.family
        demo = Path(args.episode)
        for k in range(args.repeat):
            seed = args.seed + k
            suffix = f"_r{k}" if args.repeat > 1 else ""
            out = out_dir_for(demo, family, args.severity, seed)
            jobs.append({"demo": str(demo), "family": family, "severity": args.severity,
                         "seed": seed, "out": str(out) + suffix})

    print("[init] RobotEnv cartesian_position (connects to NUC; ROBOT MOVES TO HOME) ...")
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    serials = configure_cameras(env)
    locks = lock_cameras(env)

    done_n = 0
    try:
        for e in jobs:
            out = Path(e["out"])
            if out.exists():
                continue
            run_replay(env, serials, Path(e["demo"]), e["family"],
                       float(e["severity"]), int(e["seed"]), out, locks, e.get("ctx"))
            done_n += 1
    except KeyboardInterrupt:
        print("\n[done] interrupted")
    finally:
        print(f"[done] {done_n} replays executed this session")
        try:
            go_home(env)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
