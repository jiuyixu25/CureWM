"""CureWM-Real perturbation families, the hardware mirror of
curewm.perturbations, operating on DROID absolute-action
streams [x, y, z, roll, pitch, yaw, gripper(0=open,1=closed)].

Design rules carried over from the sim engine:
  * severity in [0,1]; 0 degrades to (approximately) the nominal trajectory;
  * all randomness through an explicit rng so (demo, family, severity, seed)
    reproduces exactly;
  * perturbations touch only their target phases (annotate_phases windows).

Hardware deltas vs. sim (documented for the paper's protocol appendix):
  * carry_slip uses one contiguous open window (Franka hand actuation latency
    ~0.3 s makes sim's single-frame pulses no-ops) and drops sim's lateral
    action spikes (arm must stay on the recorded path for safety);
  * wrist_tilt is an absolute orientation offset composed in the EE frame,
    ramped in over `RAMP_FRAMES` and held to episode end (absolute-position
    replay would otherwise "un-tilt" after the window, unlike velocity-space
    sim where offsets integrate);
  * contact_oscillation / approach_overshoot are excluded on hardware
    (actuator wear / collision risk).

Dry-run mode (no robot): python perturb.py --episode <demo_dir> --family X \
    --severity 0.8 --seed 0  → prints phase windows + per-channel diffs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ALL_FAMILIES, DATA_ROOT, FAMILIES, SEMANTIC_FAMILIES,  # noqa: E402
                    annotate_phases, load_episode)  # noqa: E402

RAMP_FRAMES = 8                     # wrist_tilt ramp-in (~0.5 s @15 Hz)
TILT_MAX_DEG = 15.0                 # hard safety ceiling
SLIP_FRAMES = {0.6: 3, 0.8: 5, 1.0: 7}   # carry_slip open-window length


class SkipEpisode(Exception):
    """Perturbation not applicable to this demo (e.g. no carry phase)."""


def _euler_to_R(rpy):
    # droid.misc.transformations uses scipy xyz-extrinsic euler throughout
    # (quat_to_euler == R.from_quat(...).as_euler("xyz")). Match it.
    return R.from_euler("xyz", rpy)


def _R_to_euler(rot):
    return rot.as_euler("xyz")


def insufficient_grip(actions, ph, severity, rng):
    """Cap the close command from first_close on: interpolate toward open by
    0.9*severity (sim formula). Franka-hand mechanics make the cure literal:
    capped command <= 0.5 also switches update_gripper to force=0 mode."""
    if ph["first_close"] >= ph["T"]:
        raise SkipEpisode("demo never closes the gripper")
    a = actions.copy()
    m = np.arange(ph["T"]) >= ph["first_close"]
    a[m, 6] = a[m, 6] * (1.0 - 0.9 * severity)
    return a, {"cap": float(1.0 - 0.9 * severity), "from_frame": ph["first_close"]}


def premature_release(actions, ph, severity, rng):
    """Open at t' = carry_start + (1-severity)*(t_release - carry_start)."""
    if ph["carry_start"] >= ph["t_release"]:
        raise SkipEpisode("no carry phase")
    t_new = int(round(ph["carry_start"] +
                      (1.0 - severity) * (ph["t_release"] - ph["carry_start"])))
    a = actions.copy()
    a[t_new:, 6] = 0.0
    return a, {"release_frame": t_new, "nominal_release": ph["t_release"]}


def carry_slip(actions, ph, severity, rng):
    """Contiguous open window centred on the carry midpoint. Then restore."""
    n = SLIP_FRAMES[min(SLIP_FRAMES, key=lambda k: abs(k - severity))]
    carry_len = ph["t_release"] - ph["carry_start"]
    if carry_len < n + 4:
        raise SkipEpisode(f"carry too short ({carry_len} frames) for slip window {n}")
    mid = ph["carry_start"] + carry_len // 2
    lo, hi = mid - n // 2, mid - n // 2 + n
    a = actions.copy()
    a[lo:hi, 6] = 0.0
    return a, {"window": [int(lo), int(hi)], "n_frames": int(n)}


def wrist_tilt(actions, ph, severity, rng):
    """severity*15 deg about a horizontal EE axis (x or y, rng-chosen), ramped
    in over RAMP_FRAMES starting one third into carry, held to episode end."""
    carry_len = ph["t_release"] - ph["carry_start"]
    if carry_len < RAMP_FRAMES + 2:
        raise SkipEpisode(f"carry too short ({carry_len} frames) for tilt ramp")
    angle = np.deg2rad(min(severity * TILT_MAX_DEG, TILT_MAX_DEG))
    axis = np.zeros(3)
    axis[int(rng.integers(0, 2))] = 1.0            # EE-frame x or y
    onset = ph["carry_start"] + carry_len // 3
    a = actions.copy()
    for t in range(onset, ph["T"]):
        frac = min(1.0, (t - onset + 1) / RAMP_FRAMES)
        r_new = _euler_to_R(a[t, 3:6]) * R.from_rotvec(frac * angle * axis)
        a[t, 3:6] = _R_to_euler(r_new)
    return a, {"onset": int(onset), "axis": axis.tolist(),
               "angle_deg": float(np.rad2deg(angle))}


MT_RAMP_FRAMES = 25          # ~1.7 s @15 Hz to converge back onto the demo path
MT_MAX_STEP_M = 0.05         # stay under SafetyEnvelope.max_step_m so it never has to clamp


def wrong_target(actions, ph, severity, rng, ctx=None):
    """Semantic family: re-aim the reach at a distractor object, then converge
    back onto the demo path during the lift so carry and place are unchanged.

    Execution stays clean, with a smooth trajectory, a firm grasp and the object
    delivered to the plate. The task fails only because the wrong object was taken.
    Unlike the mechanical families this leaves no kinematic cue of failure, so
    a prediction that still shows the target object arriving is evidence that
    the model is not following the action at all. Severity selects the
    distractor (<=0.8 nearest, else farthest). See wrong_target_ctx.
    """
    if ctx is None or "delta_m" not in ctx:
        raise SkipEpisode("wrong_target needs a scene context (delta_m)")
    d = np.asarray(ctx["delta_m"], dtype=float)
    if d.shape != (3,):
        raise SkipEpisode("delta_m must be xyz")
    if ph["first_close"] >= ph["T"]:
        raise SkipEpisode("demo never closes the gripper")
    if ph["first_close"] < 8:
        raise SkipEpisode("approach too short to ease the re-target in")
    t_in = max(1, ph["first_close"] // 2)          # fully offset by mid-approach
    t0 = ph["carry_start"]                         # grasp settled
    pure_grasp = ph["t_release"] >= ph["T"]        # never releases: lift-and-hold task
    ramp_end = ph["T"] if pure_grasp else min(t0 + MT_RAMP_FRAMES, ph["T"])
    if not pure_grasp and ph["t_release"] <= ramp_end + 2:
        raise SkipEpisode("carry too short to converge back onto the path")
    w = np.zeros(ph["T"])
    s_in = np.linspace(0.0, 1.0, t_in, endpoint=False)
    w[:t_in] = 0.5 * (1.0 - np.cos(np.pi * s_in))          # 0 -> 1, cosine ease
    if pure_grasp:
        w[t_in:] = 1.0                                     # lift the wrong cube from its own spot
    else:
        w[t_in:t0] = 1.0
        s_out = np.linspace(0.0, 1.0, ramp_end - t0, endpoint=False)
        w[t0:ramp_end] = 0.5 * (1.0 + np.cos(np.pi * s_out))   # 1 -> 0
    a = actions.copy()
    a[:, :3] += w[:, None] * d
    step = float(np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1).max())
    if step > MT_MAX_STEP_M:
        raise SkipEpisode(f"re-target needs {step*100:.1f} cm/tick "
                          f"(> {MT_MAX_STEP_M*100:.0f}); approach too short for this distance")
    # from_frame=0: the paths separate from the very first tick, so the probe's
    # t0 (= injection frame) is the episode start and its horizon reads the reach.
    return a, {"delta_m": d.tolist(), "dist_m": float(np.linalg.norm(d[:2])),
               "max_step_m": step, "from_frame": 0,
               "distractor": ctx.get("distractor"), "distractor_spot": ctx.get("distractor_spot"),
               "target_color": ctx.get("target_color"), "target_spot": ctx.get("target_spot"),
               "pure_grasp": bool(pure_grasp),
               "ramp_in": [0, int(t_in)], "ramp_out": [int(t0), int(ramp_end)]}


def wrong_target_ctx_xy(cubes, target_color: str, severity: float, z=None) -> dict:
    """Context from recorded object positions (no taught-spot table needed).

    `cubes` is [{"color": str, "xy": [x, y]}, ...] as logged by the collector. Severity <= 0.8 picks the nearest distractor, otherwise the farthest.
    """
    by = {c["color"]: np.asarray(c["xy"], dtype=float) for c in cubes}
    if target_color not in by:
        raise SkipEpisode(f"target colour {target_color} not in the recorded scene")
    others = [(c, p) for c, p in by.items() if c != target_color]
    if not others:
        raise SkipEpisode("scene has no distractor")
    ranked = sorted(others, key=lambda cp: float(np.linalg.norm(cp[1] - by[target_color])))
    color, p = ranked[0] if severity <= 0.8 else ranked[-1]
    d = p - by[target_color]
    return {"delta_m": [float(d[0]), float(d[1]), 0.0], "distractor": color,
            "target_color": target_color, "distractor_xy": p.tolist(),
            "target_xy": by[target_color].tolist()}


def wrong_target_ctx(demo_meta: dict, spots, severity: float) -> dict:
    """Build the wrong_target context from a demo's recorded multi-object layout.

    severity <= 0.8 picks the nearest distractor, otherwise the farthest, so the
    family carries a real dose axis (how far the action visibly departs from the
    correct object) while every level fails by construction.
    """
    layout = demo_meta.get("layout")
    target = demo_meta.get("target_color")
    if not layout or target not in layout:
        raise SkipEpisode("demo has no multi-object layout")
    ts = int(layout[target])
    tp = np.asarray(spots[ts], dtype=float)
    others = [(c, int(s)) for c, s in layout.items() if c != target]
    if not others:
        raise SkipEpisode("layout has no distractor")
    ranked = sorted(others, key=lambda cs: float(
        np.linalg.norm(np.asarray(spots[cs[1]], dtype=float)[:2] - tp[:2])))
    color, sp = ranked[0] if severity <= 0.8 else ranked[-1]
    return {"delta_m": (np.asarray(spots[sp], dtype=float) - tp).tolist(),
            "distractor": color, "distractor_spot": sp,
            "target_color": target, "target_spot": ts}


FAMILY_FNS = {
    "insufficient_grip": insufficient_grip,
    "premature_release": premature_release,
    "carry_slip": carry_slip,
    "wrist_tilt": wrist_tilt,
    "wrong_target": wrong_target,
}
assert set(FAMILY_FNS) == set(ALL_FAMILIES)


def perturb(actions: np.ndarray, family: str, severity: float, seed: int, ctx=None):
    """Returns (perturbed_actions, info). Raises SkipEpisode when inapplicable.

    `ctx` carries scene information for the semantic families (wrong_target).
    """
    ph = annotate_phases(actions)
    rng = np.random.default_rng(seed)
    if family in SEMANTIC_FAMILIES:
        pert, detail = FAMILY_FNS[family](actions, ph, severity, rng, ctx)
    else:
        pert, detail = FAMILY_FNS[family](actions, ph, severity, rng)
    info = {"family": family, "severity": float(severity), "seed": int(seed),
            "phases": ph, "detail": detail}
    return pert, info


# ----------------------------- dry run -------------------------------------

def main():
    p = argparse.ArgumentParser(description="Dry-run a perturbation on a recorded demo (no robot).")
    p.add_argument("--episode", required=True)
    p.add_argument("--family", required=True, choices=ALL_FAMILIES)
    p.add_argument("--delta", default=None, help="wrong_target dry run: dx,dy,dz in metres")
    p.add_argument("--severity", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta, traj = load_episode(args.episode)
    actions = traj["actions"]
    ctx = None
    if args.family in SEMANTIC_FAMILIES:
        if args.delta:
            ctx = {"delta_m": [float(v) for v in args.delta.split(",")], "distractor": "dry-run"}
        else:
            spots_file = DATA_ROOT / f"spots_{meta.get('task_id')}.json"
            if not spots_file.exists():
                print(f"SKIP: need --delta or {spots_file}"); return
            ctx = wrong_target_ctx(meta, json.load(open(spots_file))["spots"], args.severity)
            print(f"ctx: target={ctx['target_color']}@spot{ctx['target_spot']+1} -> "
                  f"distractor={ctx['distractor']}@spot{ctx['distractor_spot']+1}")
    try:
        pert, info = perturb(actions, args.family, args.severity, args.seed, ctx)
    except SkipEpisode as e:
        print(f"SKIP: {e}")
        return
    ph = info["phases"]
    print(f"episode {meta.get('task_id')} frames={ph['T']}  "
          f"approach[0:{ph['first_close']}) grasp[{ph['first_close']}:{ph['carry_start']}) "
          f"carry[{ph['carry_start']}:{ph['t_release']}) place[{ph['t_release']}:{ph['T']})")
    print("detail:", info["detail"])
    d = pert - actions
    for name, sl in [("xyz(m)", slice(0, 3)), ("rpy(rad)", slice(3, 6)), ("grip", slice(6, 7))]:
        dd = np.abs(d[:, sl])
        print(f"  Δ{name}: max={dd.max():.4f} @frame {int(np.unravel_index(dd.argmax(), dd.shape)[0])}, "
              f"changed frames={int((dd.sum(axis=1) > 1e-9).sum())}")


if __name__ == "__main__":
    main()
