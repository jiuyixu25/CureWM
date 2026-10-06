"""Scripted language-labelled demonstrations for the cube-pickup task (training data for pi0.5 / X-VLA).

Reuses auto_collect.py end to end (workspace from the taught spots, cube detection-free placement with the
robot's own shuffle, scripted grasp with noise, telemetry verdict, automatic put-back) but records ONLY
nominal demonstrations: every scene has all colours present, the target cycles through the colours so the
dataset is balanced, and the stored instruction is drawn from --wordings so the policy sees more than one
phrasing per colour (held-out phrasings are kept for evaluation).

    python collect_demos.py --task-id T1C --scenes 90 --colors red,green,blue --auto-shuffle \
        --wordings "pick up the {c} cube|grab the {c} cube"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import auto_collect as AC  # noqa: E402
from common import DATA_ROOT, configure_cameras, lock_cameras, log_session  # noqa: E402


def perturb_path(base: np.ndarray, cube_xy, rng, lo, hi, detour=(0.015, 0.04), descent=(0.005, 0.015),
                 corr_frames=(12, 18)) -> tuple[np.ndarray, dict]:
    """Detour-then-correct variant of a clean scripted pick (DART-style corrective data).

    The clean path is home -> hover above the cube -> descend -> close -> lift. The policy trained on such
    noise-free demos never sees itself off the path, so a 3 cm miss is never corrected. Here the approach
    ends `detour` metres beside the cube, a short correction segment slides back over it at hover height,
    and the descent carries a small in-and-out lateral bump that is gone before the fingers close. The grasp
    pose and everything from the close command on are identical to the clean path, so the demo stays valid.
    """
    from common import annotate_phases
    base = np.asarray(base, float)
    fc = annotate_phases(base)["first_close"]
    d = np.linalg.norm(base[:fc, :2] - np.asarray(cube_xy, float), axis=1)
    above = np.where(d < 0.006)[0]
    if len(above) == 0:
        raise ValueError("clean path never hovers above the cube")
    hover = int(above[0])                              # first frame above the cube = end of the approach
    acts = base.copy()
    # (1) approach detour: ramps in from the home pose so the start is unchanged
    ang = rng.uniform(0, 2 * np.pi); r = float(rng.uniform(*detour))
    delta = np.array([np.cos(ang), np.sin(ang)]) * r
    target_hover = np.clip(acts[hover, :2] + delta, lo[:2] + 0.005, hi[:2] - 0.005)
    delta = target_hover - acts[hover, :2]
    ramp = np.linspace(0.0, 1.0, hover + 1)[:, None]
    acts[: hover + 1, :2] += ramp * delta
    # (2) correction at hover height: smooth slide from beside the cube back over it, gripper open
    K = int(rng.integers(corr_frames[0], corr_frames[1] + 1))
    sm = (1 - np.cos(np.linspace(0, np.pi, K + 2)[1:-1])) / 2.0      # smoothstep, endpoints excluded
    corr = np.repeat(acts[hover][None, :], K, axis=0)
    corr[:, :2] = base[hover, :2] + (1.0 - sm)[:, None] * delta
    # (3) descent bump: in-and-out lateral offset, zero again before the close command
    desc = acts[hover + 1: fc].copy()
    ang2 = rng.uniform(0, 2 * np.pi); r2 = float(rng.uniform(*descent))
    delta2 = np.array([np.cos(ang2), np.sin(ang2)]) * r2
    bump = np.sin(np.pi * np.linspace(0, 1, len(desc) + 2)[1:-1])
    desc[:, :2] += bump[:, None] * delta2
    out = np.concatenate([acts[: hover + 1], corr, desc, base[fc:]], axis=0)
    info = {"detour_xy_m": [round(float(v), 4) for v in delta], "detour_r_m": round(float(np.linalg.norm(delta)), 4),
            "correction_frames": K, "descent_bump_xy_m": [round(float(v), 4) for v in delta2],
            "hover_frame": hover, "first_close_clean": int(fc), "first_close": int(fc + K)}
    return out, info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", default="T1C")
    p.add_argument("--scenes", type=int, default=90, help="episodes to record (one demo per scene)")
    p.add_argument("--colors", default="red,green,blue")
    p.add_argument("--speed", type=float, default=0.6)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2027, help="layout RNG; use a fresh value for every session")
    p.add_argument("--min-move", type=float, default=0.06, help="the relocated cube must land at least this far (m) from where it was")
    p.add_argument("--wordings", default="pick up the {c} cube",
                   help="'|'-separated instruction templates; {c} is the colour. One wording in training; "
                        "every other phrasing is held out for the paraphrase evaluation.")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--auto-shuffle", action="store_true", help="kept for compatibility; the scene now changes by relocating the lifted cube")
    p.add_argument("--no-relocate", dest="relocate", action="store_false", help="put every cube back where it was (static layout)")
    p.add_argument("--out-subdir", default="pickup_demos")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--perturb", action="store_true",
                   help="detour-then-correct demos: approach ends beside the cube and slides back over it, with a small "
                        "lateral bump during the descent (corrective data; the grasp itself is unchanged)")
    p.add_argument("--perturb-frac", type=float, default=1.0, help="fraction of scenes recorded with the perturbation")
    p.add_argument("--detour", type=float, nargs=2, default=(0.015, 0.04), metavar=("MIN", "MAX"), help="approach detour radius (m)")
    p.add_argument("--descent-bump", type=float, nargs=2, default=(0.005, 0.015), metavar=("MIN", "MAX"), help="descent bump radius (m)")
    args = p.parse_args()
    args.task_id = args.task_id   # episode() reads args.task_id / args.confirm / args.speed

    colors = [c.strip() for c in args.colors.split(",")]
    wordings = [w.strip() for w in args.wordings.split("|") if w.strip()]
    cfg = json.load(open(DATA_ROOT / f"spots_{args.task_id}.json"))
    lo, hi = AC.workspace(cfg)
    rng = np.random.default_rng(args.seed)
    print(f"[demos] {args.scenes} scenes, colours {colors}, wordings {wordings}, seed {args.seed}")

    def relocate(cur_xy, others):
        """A free spot for the cube just lifted: clear of the other cubes and a real move from its own place."""
        for _ in range(4000):
            q = rng.uniform(lo, hi)
            if all(np.linalg.norm(q - o) >= AC.MIN_SEP_M for o in others) and np.linalg.norm(q - np.asarray(cur_xy)) >= args.min_move:
                return q
        return None
    if args.dry:
        for s in range(min(5, args.scenes)):
            pos = AC.sample_positions(lo, hi, len(colors), rng)
            c = colors[s % len(colors)]; w = wordings[int(rng.integers(len(wordings)))]
            print(f"  scene {s}: target={c} instruction={w.format(c=c)!r} positions={np.round(pos, 3).tolist()}")
        return

    print("[init] restarting the control stack ...")
    if subprocess.run(["bash", str(Path(__file__).parent / "restart_stack.sh")]).returncode != 0:
        raise SystemExit("[init] restart_stack failed")
    from droid.robot_env import RobotEnv
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    serials = configure_cameras(env)
    locks = lock_cameras(env)
    AC.go_home(env)

    start = AC.sample_positions(lo, hi, len(colors), rng)
    cubes = {c: start[i] for i, c in enumerate(colors)}
    print("\n[place] initial layout: the gripper hovers over each spot, put the matching cube under it")
    for c in colors:
        AC.place_guided(env, serials, cubes[c], cfg, c)
    AC.go_home(env)

    out_root = DATA_ROOT / "demos" / args.task_id
    tally = {"ok": 0, "skip": 0, "anomaly": 0}; streak = 0
    try:
        for s in range(args.scenes):
            target = colors[s % len(colors)]
            wording = wordings[int(rng.integers(len(wordings)))]
            stamp = time.strftime("%m%d_%H%M%S")
            print(f"\n=== scene {s+1}/{args.scenes}  target={target}  " +
                  "  ".join(f"{c}({cubes[c][0]:.2f},{cubes[c][1]:.2f})" for c in colors) + " ===")
            base = AC.grasp_traj(cubes[target], cfg, rng, args.noise, args.speed)
            pinfo = None
            if args.perturb and rng.uniform() < args.perturb_frac:
                base, pinfo = perturb_path(base, cubes[target], rng, lo, hi, tuple(args.detour), tuple(args.descent_bump))
                print(f"    perturbed: detour {pinfo['detour_r_m']*100:.1f} cm, correction {pinfo['correction_frames']} frames, "
                      f"descent bump {np.linalg.norm(pinfo['descent_bump_xy_m'])*100:.1f} cm")
            job = {"kind": "demo", "name": f"{args.out_subdir}_{args.task_id}_{stamp}", "scene_id": s,
                   "serials": serials, "locks": locks}
            # the scene changes by relocating ONLY the cube this episode lifts. The robot is holding it anyway
            new_xy = relocate(cubes[target], [cubes[c] for c in colors if c != target]) if args.relocate else None
            status, note, saved = AC.episode(env, serials, cfg, cubes, target, job, base, rng, args, place_xy=new_xy)
            if saved is not None and pinfo is not None:      # sidecar note: which demos carry the perturbation
                mp = Path(saved) / "meta.json"; m = json.load(open(mp)); m["perturbation"] = {"kind": "detour_then_correct", **pinfo}
                m["source"] = "scripted-auto-perturbed"; json.dump(m, open(mp, "w"), indent=1)
            if new_xy is not None and status == "ok":
                cubes[target] = new_xy          # the put-back left it there
            tally[status] += 1
            print(f"    {status}: {note}", flush=True)
            if saved is not None:
                mp = Path(saved) / "meta.json"; m = json.load(open(mp))
                m["task_canonical"] = m.get("task"); m["task"] = wording.format(c=target)
                m["wording_template"] = wording; m["dataset"] = args.out_subdir
                json.dump(m, open(mp, "w"), indent=2, ensure_ascii=False)
            if status == "anomaly":
                streak += 1
                if streak >= AC.MAX_ANOMALIES:
                    raise SystemExit(f"[demos] {streak} anomalies in a row; stopping for a human")
            else:
                streak = 0
    except KeyboardInterrupt:
        print("\n[demos] interrupted")
    finally:
        try:
            AC.go_home(env)
        except Exception:
            pass
    print(f"[demos] done: {tally}  ->  {out_root}")
    log_session({"event": "collect_demos_done", "task_id": args.task_id, "tally": tally})


if __name__ == "__main__":
    main()
