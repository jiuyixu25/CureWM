"""Hands-off collection of cube counterfactual pairs (T1C) — the operator
supervises quality, the script does the labelling and the scene reset.

Task: `pick up the {red|green|blue} cube` — a pure grasp (lift and hold), the
form the neighbouring Franka effort validated over 180 episodes. It buys three
properties that make unattended running sound:

  * the verdict follows from telemetry, not judgement: a 2 cm cube holds the
    fingers at ~0.70 where an empty close reads ~1.0 (threshold 0.88). A thin
    paper cup reads 0.97 against 1.00, which is why the cup task stayed manual;
  * the episode ends with the cube in the gripper, so the reset is "put it back
    where it came from" — no search, no vision;
  * cube positions are known to the millimetre because the robot itself placed
    them (guided placement at the start, then autonomous shuffles).

Each scene yields a matched group recorded back to back, so the counterfactual
pairs share a physically identical initial state rather than an approximate one:

    nominal grasp of the target        -> success
    re-target to the near distractor   -> failure (wrong cube lifted)
    re-target to the far distractor    -> failure
    insufficient_grip on the target    -> failure (nothing lifted, no reset)

Safety: every trajectory passes SafetyEnvelope; anything unexpected pauses for
the human instead of guessing; two consecutive anomalies abort the run.

  python auto_collect.py --task-id T1C --scenes 12 --confirm
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONTROL_HZ, DATA_ROOT, EpisodeBuffer, SafetyEnvelope,  # noqa: E402
                    annotate_phases, configure_cameras, lock_cameras, log_session)
from perturb import SkipEpisode, perturb, wrong_target_ctx_xy  # noqa: E402
from scripted_demo import HOVER_DZ, _record_tick, gen_trajectory  # noqa: E402

HOME_XYZ = [0.31, 0.0, 0.49]
TAIL_FRAMES = 20             # ~1.3 s of settled ending, uniform across episodes
GRIP_HELD_MAX = 0.88         # a 2 cm cube reads ~0.70; an empty close reads ~1.0
GRIP_HELD_MIN = 0.15         # fully open reads ~0.0 — "held" is a band, not a threshold
HOLD_FRAC_MIN = 0.60
MIN_SEP_M = 0.09             # keep cubes apart so "which one" is unambiguous
PLACE_HOVER_DZ = 0.05        # low hover for guided placement: a precise visual guide
INSET_M = 0.03               # shrink the taught workspace before sampling
MAX_ANOMALIES = 2


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def grasp_z(cfg, xy=None):
    """Grasp height at (x, y), bilinear over the four taught corners.

    The corners of this rig differ by ~20 mm in z, which a single median would
    spread as +/-10 mm of error — the entire tolerance of a 2 cm cube, and enough
    to drive the fingers into the table at the high corner. Interpolating keeps
    every sampled point at the height it was actually taught at.
    """
    sp = np.asarray(cfg["spots"], float)
    if xy is None or len(sp) != 4:
        return float(np.median(sp[:, 2]))
    lo, hi = sp[:, :2].min(0), sp[:, :2].max(0)
    ctr = (lo + hi) / 2.0
    corner = {}
    for s in sp:                                   # label each taught point by quadrant
        corner[(s[0] > ctr[0], s[1] > ctr[1])] = float(s[2])
    if len(corner) != 4:
        return float(np.median(sp[:, 2]))
    u = float(np.clip((xy[0] - lo[0]) / max(hi[0] - lo[0], 1e-6), 0, 1))
    v = float(np.clip((xy[1] - lo[1]) / max(hi[1] - lo[1], 1e-6), 0, 1))
    z = ((1 - u) * (1 - v) * corner[(False, False)] + u * (1 - v) * corner[(True, False)]
         + u * v * corner[(True, True)] + (1 - u) * v * corner[(False, True)])
    return float(max(z, sp[:, 2].min() - 0.005))   # never dive below the lowest taught height


def workspace(cfg):
    """Axis-aligned sampling box from the taught points, inset for margin."""
    xy = np.asarray([s[:2] for s in cfg["spots"]], float)
    lo, hi = xy.min(0) + INSET_M, xy.max(0) - INSET_M
    if np.any(hi <= lo):
        raise SystemExit(f"[cal] the taught region is empty after a {INSET_M*100:.0f} cm inset -- re-teach the four corners")
    return lo, hi


def sample_positions(lo, hi, n, rng, tries=8000):
    for _ in range(tries):
        p = rng.uniform(lo, hi, size=(n, 2))
        d = [np.linalg.norm(p[i] - p[j]) for i in range(n) for j in range(i + 1, n)]
        if min(d) >= MIN_SEP_M:
            return p
    raise SystemExit(f"[cal] the inset {(hi-lo)[0]*100:.0f}x{(hi-lo)[1]*100:.0f} cm region cannot hold {n} cubes "
                     f"{MIN_SEP_M*100:.0f} cm apart -- teach the four corners further out")


# --------------------------------------------------------------------------
# Motion primitives
# --------------------------------------------------------------------------

def stream(env, serials, acts, buf=None, tail=0):
    envl = SafetyEnvelope(acts, extra_xyz=acts)
    period, drops = 1.0 / CONTROL_HZ, 0
    for t in range(len(acts) + tail):
        t0 = time.time()
        a = envl.filter(np.asarray(acts[min(t, len(acts) - 1)], float))
        try:
            if buf is not None:
                drops += _record_tick(env, serials, buf, a.astype(np.float32), t < len(acts))
            else:
                env.get_observation(); env.step(a)
        except Exception as e:  # noqa: BLE001
            print(f"    tick {t} error ({type(e).__name__}: {e})")
        el = time.time() - t0
        if el < period:
            time.sleep(period - el)
    return drops, envl.violations


def go_home(env):
    env._robot.update_joints(env.reset_joints, velocity=False, blocking=True)
    env._robot.update_gripper(0, velocity=False, blocking=True)


def grasp_traj(cube_xy, cfg, rng, noise, speed):
    """home -> hover -> descend -> close & settle -> lift, cut before any traverse."""
    zg = grasp_z(cfg, cube_xy)
    grasp = np.array([cube_xy[0], cube_xy[1], zg], float)
    lo, hi = workspace(cfg)
    far = np.array([*((lo + hi) / 2.0), zg], float)           # only shapes the (cut) traverse
    full = gen_trajectory(grasp, far, np.asarray(cfg["rpy"], float), rng,
                          home_xyz=HOME_XYZ, noise=noise, speed=speed)
    ph = annotate_phases(full)
    moved = np.linalg.norm(full[:, :2] - grasp[:2], axis=1) > 0.012
    after = np.where(moved & (np.arange(len(full)) > ph["carry_start"]))[0]
    cut = int(after[0]) if len(after) else len(full)
    return full[:cut]


def held(states, phase_ref):
    """Did the fingers hold a cube through the settle+lift? (telemetry verdict)

    Phases come from the nominal action stream: insufficient_grip caps the close
    command below the intent threshold, so annotating the perturbed actions
    would leave an empty window.
    """
    ph = annotate_phases(phase_ref)
    seg = np.asarray(states)[ph["carry_start"]:ph["t_release"], 6]
    if len(seg) == 0:
        return False, 0.0, float("nan")
    frac = float(((seg > GRIP_HELD_MIN) & (seg < GRIP_HELD_MAX)).mean())
    return frac >= HOLD_FRAC_MIN, frac, float(np.median(seg))


def put_back(env, serials, xy, cfg, rng, speed):
    """Cube is in the gripper above `xy`: descend, open, retreat. Returns ok."""
    z_g = grasp_z(cfg, xy); z_h = z_g + HOVER_DZ
    rpy = list(cfg["rpy"])
    cur = np.asarray(env.get_observation()["robot_state"]["cartesian_position"], float)
    approach = [[*(cur[:3] + (np.array([xy[0], xy[1], z_h]) - cur[:3]) * f), *rpy, 1.0]
                for f in np.linspace(0, 1, 10)]        # ease over from wherever the lift ended
    seq = (approach + [[xy[0], xy[1], z_h, *rpy, 1.0]] * 3
           + [[xy[0], xy[1], z_g + (z_h - z_g) * f, *rpy, 1.0] for f in np.linspace(1, 0, 14)]
           + [[xy[0], xy[1], z_g, *rpy, 0.0]] * 6
           + [[xy[0], xy[1], z_g + (z_h - z_g) * f, *rpy, 0.0] for f in np.linspace(0, 1, 12)])
    stream(env, serials, np.asarray(seq, float))
    go_home(env)
    return True


def move_cube(env, serials, src_xy, dst_xy, cfg, rng, speed):
    """Pick the cube at src and set it down ON THE TABLE at dst. Telemetry confirms the pick.

    The cup->plate trajectory used before released at PLACE_DZ above the target with +-1.5 cm xy jitter, so
    cubes were dropped and landed off their recorded positions. Now: pick (same scripted grasp as a demo),
    carry at hover height, lower to the grasp height of dst (the cube's resting height), settle, open, retreat.
    """
    pick = grasp_traj(src_xy, cfg, rng, noise=0.0, speed=speed)        # home -> hover -> descend -> close -> lift
    buf = EpisodeBuffer(out_dir=Path("/tmp/_unused"), meta={})
    stream(env, serials, pick, buf=buf)
    ok, frac, med = held(buf.states, pick)
    if not ok:
        go_home(env)
        return ok, frac, med
    z_g = grasp_z(cfg, dst_xy); z_h = z_g + HOVER_DZ
    rpy = list(cfg["rpy"])
    cur = np.asarray(env.get_observation()["robot_state"]["cartesian_position"], float)
    hover = np.array([dst_xy[0], dst_xy[1], z_h], float)
    seq = ([[*(cur[:3] + (hover - cur[:3]) * f), *rpy, 1.0] for f in np.linspace(0, 1, 18)]      # carry to hover
           + [[*hover, *rpy, 1.0]] * 3
           + [[dst_xy[0], dst_xy[1], z_g + (z_h - z_g) * f, *rpy, 1.0] for f in np.linspace(1, 0, 16)]  # lower to the table
           + [[dst_xy[0], dst_xy[1], z_g, *rpy, 1.0]] * 4                                             # settle
           + [[dst_xy[0], dst_xy[1], z_g, *rpy, 0.0]] * 6                                             # open on the table
           + [[dst_xy[0], dst_xy[1], z_g + (z_h - z_g) * f, *rpy, 0.0] for f in np.linspace(0, 1, 12)])  # retreat
    stream(env, serials, np.asarray(seq, float))
    go_home(env)
    return ok, frac, med


def place_all(env, serials, cfg, colors, cubes, lo, hi, rng):
    """Sample a fresh separated layout and let the operator place each cube under
    the hovering gripper — positions stay known to the millimetre."""
    pos = sample_positions(lo, hi, len(colors), rng)
    for i, c in enumerate(colors):
        cubes[c] = pos[i]
        place_guided(env, serials, cubes[c], cfg, c)
    go_home(env)
    d = [np.linalg.norm(cubes[a] - cubes[b]) * 100
         for i, a in enumerate(colors) for b in colors[i + 1:]]
    print("    separation " + " / ".join(f"{x:.0f}" for x in d) + " cm")
    return cubes


def place_guided(env, serials, xy, cfg, color):
    """Hover the open gripper over the sampled point so the operator can drop the
    cube exactly there — the position is then known to the millimetre."""
    z_h = grasp_z(cfg, xy) + PLACE_HOVER_DZ
    hold = np.array([xy[0], xy[1], z_h, *cfg["rpy"], 0.0], float)
    stream(env, serials, np.repeat(hold[None, :], 20, axis=0))
    input(f"    place the {color} cube directly under the open gripper ({xy[0]:.3f},{xy[1]:.3f}), then press Enter : ")


def shuffle_cubes(env, serials, cfg, cubes, lo, hi, rng, speed):
    """Robot re-arranges the scene itself; positions stay known from memory."""
    for c in sorted(cubes, key=lambda _: rng.random()):
        others = [cubes[k] for k in cubes if k != c]
        p = None
        for _ in range(3000):
            q = rng.uniform(lo, hi)
            if all(np.linalg.norm(q - o) >= MIN_SEP_M for o in others):
                p = q
                break
        if p is None:
            return False, "no free position for the shuffle"
        ok, frac, _ = move_cube(env, serials, cubes[c], p, cfg, rng, speed)
        if not ok:
            return False, f"lost the {c} cube while shuffling (hold {frac:.2f})"
        cubes[c] = p
    return True, ""


# --------------------------------------------------------------------------
# One matched scene: nominal + two re-targets + one grip failure
# --------------------------------------------------------------------------
# Verdicts are determined by construction, so telemetry decides them and any
# other grasp state is an anomaly that gets discarded and flagged, never labelled.

def episode(env, serials, cfg, cubes, target, job, base_acts, rng, args, place_xy=None):
    from replay_perturbed import out_dir_for
    scene = [{"color": c, "xy": [float(v[0]), float(v[1])]} for c, v in cubes.items()]
    common_meta = {"task_id": args.task_id, "control_hz": CONTROL_HZ,
                   "task": f"pick up the {target} cube", "objects": list(cubes),
                   "cubes": scene, "target_color": target,
                   "target_xy": [float(cubes[target][0]), float(cubes[target][1])],
                   "scene_id": job["scene_id"], "cameras": job["serials"],
                   "camera_lock": job["locks"]}
    lifted = target
    if job["kind"] == "demo":
        acts, pinfo = base_acts, None
        out = DATA_ROOT / "demos" / args.task_id / job["name"]
        meta = {**common_meta, "kind": "demo", "source": "scripted-auto", "split": "demos"}
    else:
        ctx = None
        if job["family"] == "wrong_target":
            ctx = wrong_target_ctx_xy(scene, target, job["severity"])
            lifted = ctx["distractor"]
        try:
            acts, pinfo = perturb(base_acts, job["family"], job["severity"], job["seed"], ctx)
        except SkipEpisode as e:
            return "skip", str(e), None
        out = Path(out_dir_for(job["demo_dir"], job["family"], job["severity"], job["seed"]))
        meta = {**common_meta, "kind": "replay", "source_demo": str(job["demo_dir"]),
                "split": "demos", "perturbation": pinfo}
    if out.exists():
        return "skip", "already recorded", None

    if args.confirm and input(f"    record this one? <Enter>=go  s=skip  q=quit : ").strip() in ("s", "q"):
        return "skip", "operator skipped", None
    buf = EpisodeBuffer(out_dir=out, meta=meta)
    drops, viol = stream(env, serials, acts, buf=buf, tail=TAIL_FRAMES)
    is_held, frac, med = held(buf.states, base_acts)

    if job["kind"] == "demo":
        outcome = "success" if is_held else "discard"
    elif job["family"] == "wrong_target":
        outcome = "failure" if is_held else "discard"          # wrong cube lifted
    else:                                                       # insufficient_grip
        outcome = "discard" if is_held else "failure"           # nothing lifted
    meta.update({"outcome": outcome, "camera_drops": int(drops), "safety_violations": int(viol),
                 "auto_verdict": {"held": bool(is_held), "hold_frac": round(frac, 3),
                                  "median_grip": round(med, 3),
                                  "thresholds": [GRIP_HELD_MAX, HOLD_FRAC_MIN]}})
    # NEVER go_home while holding: it opens the gripper and would drop the cube.
    # place_xy relocates the lifted cube instead of returning it: the robot is already holding it, so one
    # scene change costs no extra pick-and-place (the caller must then update its own layout bookkeeping).
    if is_held:
        put_back(env, serials, cubes[lifted] if place_xy is None else place_xy, cfg, rng, args.speed)
    else:
        go_home(env)
    note = f"{outcome}  (hold {frac:.2f}, grip {med:.2f}, drops {drops})"

    if args.confirm:
        ans = input(f"    auto: {note}  <Enter>=accept  d=discard  q=quit : ").strip().lower()
        if ans == "q":
            raise KeyboardInterrupt
        if ans == "d":
            outcome = "discard"
            meta["outcome"] = outcome
            meta["auto_verdict"]["operator_override"] = True

    saved = None
    if outcome == "discard":
        status = "anomaly" if not args.confirm else "skip"
    else:
        saved = buf.save()
        status = "ok"
        log_session({"event": "auto_saved", "path": str(saved), "kind": job["kind"],
                     "family": job.get("family"), "outcome": outcome,
                     "hold_frac": round(frac, 3), "scene": job["scene_id"]})
    return status, note, saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", default="T1C")
    p.add_argument("--scenes", type=int, default=12, help="number of scenes; each yields 1 nominal and 3 counterfactuals")
    p.add_argument("--colors", default="red,green,blue")
    p.add_argument("--speed", type=float, default=0.6)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--confirm", action="store_true", help="ask before each episode and allow a veto before it is written")
    p.add_argument("--auto-shuffle", action="store_true",
                   help="let the robot re-arrange between scenes; the default instead hovers over each position for the operator to place")
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()

    colors = [c.strip() for c in args.colors.split(",")]
    cfg = json.load(open(DATA_ROOT / f"spots_{args.task_id}.json"))
    lo, hi = workspace(cfg)
    rng = np.random.default_rng(args.seed)
    print(f"[auto] sampling region x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] "
          f"({(hi[0]-lo[0])*100:.0f}×{(hi[1]-lo[1])*100:.0f} cm)")
    zs = [grasp_z(cfg, c) for c in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))]
    print(f"[auto] grasp height is bilinear over the four corners: {[round(z,3) for z in zs]} (spread {(max(zs)-min(zs))*1000:.0f} mm)")
    sample_positions(lo, hi, len(colors), np.random.default_rng(0))     # fail fast
    print(f"[auto] {args.scenes} scenes -> {args.scenes} nominal + {args.scenes*2} redirect + "
          f"{args.scenes} insufficient-grip = {args.scenes*4} episodes")
    if args.dry:
        for i in range(min(3, args.scenes)):
            pos = sample_positions(lo, hi, len(colors), rng)
            d = [np.linalg.norm(pos[a]-pos[b])*100 for a in range(3) for b in range(a+1,3)]
            print(f"    scene {i} target={colors[i%len(colors)]} separation {['%.0f'%x for x in d]} cm")
        return

    print("[init] restarting the control stack ...")
    if subprocess.run(["bash", str(Path(__file__).parent / "restart_stack.sh")]).returncode != 0:
        raise SystemExit("[init] restart_stack failed")
    from droid.robot_env import RobotEnv
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    serials = configure_cameras(env)
    locks = lock_cameras(env)
    go_home(env)

    start = sample_positions(lo, hi, len(colors), rng)
    cubes = {c: start[i] for i, c in enumerate(colors)}
    print("\n[place] initial layout: the gripper hovers over each of the three positions in turn; place the matching cube directly beneath it")
    for c in colors:
        place_guided(env, serials, cubes[c], cfg, c)
    go_home(env)

    tally = {"ok": 0, "skip": 0, "anomaly": 0}
    streak = 0
    try:
        for s in range(args.scenes):
            if s > 0:
                if args.auto_shuffle:
                    print(f"\n[shuffle] scene {s+1}: the robot re-arranges the three cubes ...")
                    ok, why = shuffle_cubes(env, serials, cfg, cubes, lo, hi, rng, args.speed)
                    if not ok:
                        print(f"[shuffle] failed: {why} -- stopping for the operator")
                        break
                else:
                    print(f"\n[place] scene {s+1} layout: the gripper hovers in turn, you place the cubes")
                    place_all(env, serials, cfg, colors, cubes, lo, hi, rng)
            target = colors[s % len(colors)]
            stamp = time.strftime("%m%d_%H%M%S")
            print(f"\n=== scene {s+1}/{args.scenes}  target={target}  " +
                  "  ".join(f"{c}({cubes[c][0]:.2f},{cubes[c][1]:.2f})" for c in colors) + " ===")
            base = grasp_traj(cubes[target], cfg, rng, args.noise, args.speed)
            demo_dir = DATA_ROOT / "demos" / args.task_id / f"demo_{args.task_id}_{stamp}"
            group = [{"kind": "demo", "name": demo_dir.name}] + [
                {"kind": "replay", "family": "wrong_target", "severity": sev,
                 "seed": int(rng.integers(0, 2**31 - 1)), "demo_dir": demo_dir}
                for sev in (0.6, 1.0)] + [
                {"kind": "replay", "family": "insufficient_grip", "severity": 1.0,
                 "seed": int(rng.integers(0, 2**31 - 1)), "demo_dir": demo_dir}]
            for job in group:
                job.update({"scene_id": s, "serials": serials, "locks": locks})
                tag = job.get("family", "nominal")
                print(f"  [{tag} {job.get('severity','')}]", flush=True)
                status, note, _ = episode(env, serials, cfg, cubes, target, job, base, rng, args)
                tally[status] += 1
                print(f"    {status}: {note}", flush=True)
                if status == "anomaly":
                    streak += 1
                    log_session({"event": "auto_anomaly", "scene": s, "family": tag, "note": note})
                    if streak >= MAX_ANOMALIES:
                        raise SystemExit(f"[auto] {streak} anomalies in a row -- stopping for the operator; "
                                         f"fix the scene and rerun this command, already-recorded episodes are skipped")
                else:
                    streak = 0
                if job["kind"] == "demo" and status != "ok":
                    print("    the nominal demonstration did not succeed, skipping this scene's counterfactuals"); break
    except KeyboardInterrupt:
        print("\n[auto] interrupted")
    finally:
        print(f"\n[auto] recorded {tally['ok']} / skipped {tally['skip']} / anomalies {tally['anomaly']}")
        try:
            go_home(env)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
