"""Offline invariant tests for the CureWM-Real engine. No robot, no cameras.

Builds a synthetic demo (approach 30 / grasp 15 / carry 45 / place 15 frames),
saves it in episode format, then checks every family x severity:
  * phase windows land where constructed;
  * gripper-only families leave xyz+rpy bit-identical;
  * wrist_tilt leaves xyz identical, bounds orientation delta by severity*15deg,
    and touches nothing before its onset;
  * severity monotonicity where it is analytic (release timing, grip cap,
    slip window length);
  * determinism: same (family, severity, seed) twice -> identical arrays;
  * plan builder: correct counts, per-family balance, stable under --append.
Run: conda run -n robot python test_offline.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import annotate_phases  # noqa: E402
from perturb import FAMILY_FNS, SkipEpisode, perturb, wrong_target_ctx  # noqa: E402

APPROACH, GRASP, CARRY, PLACE = 30, common.GRASP_SETTLE_FRAMES, 45, 15
T = APPROACH + GRASP + CARRY + PLACE
checks = [0, 0]


def check(ok, msg):
    checks[0] += ok
    checks[1] += 1
    if not ok:
        print(f"  FAIL: {msg}")


def synthetic_actions() -> np.ndarray:
    t = np.linspace(0, 1, T)
    a = np.zeros((T, 7), dtype=np.float32)
    a[:, 0] = 0.45 + 0.15 * np.sin(2 * np.pi * t)          # x
    a[:, 1] = -0.10 + 0.20 * t                              # y
    a[:, 2] = 0.30 - 0.18 * np.sin(np.pi * t)               # z dips to grasp
    a[:, 3:6] = np.array([np.pi, 0.05, 0.1])                # top-down-ish
    a[APPROACH:APPROACH + GRASP + CARRY, 6] = 1.0           # closed until release
    return a


def test_wrong_target(acts, ph):
    """Semantic family: clean execution, wrong object. Must leave the gripper and
    orientation untouched, start and end on the demo path, hold the full offset
    through the grasp, and never demand more than the per-step safety limit."""
    spots = [[0.60, -0.20, 0.11], [0.60, -0.08, 0.11], [0.60, 0.16, 0.11]]   # 12 cm / 36 cm away
    meta = {"layout": {"red": 0, "green": 1, "blue": 2}, "target_color": "red"}
    dists = {}
    for sev in common.MT_SEVERITIES:
        ctx = wrong_target_ctx(meta, spots, sev)
        p1, info = perturb(acts, "wrong_target", sev, seed=7, ctx=ctx)
        p2, _ = perturb(acts, "wrong_target", sev, seed=7, ctx=ctx)
        det = info["detail"]; dists[sev] = det["dist_m"]
        d = p1[:, :3] - acts[:, :3]
        delta = np.asarray(ctx["delta_m"])
        t_in, (r0, r1) = max(1, ph["first_close"] // 2), det["ramp_out"]
        check(np.array_equal(p1, p2), f"wrong_target s{sev} non-deterministic")
        check(np.array_equal(p1[:, 3:], acts[:, 3:]),
              f"wrong_target s{sev} must not touch orientation or gripper")
        check(np.allclose(d[0], 0, atol=1e-6), f"wrong_target s{sev} must start on the demo path")
        check(np.allclose(d[t_in:ph["carry_start"]], delta, atol=1e-6),
              f"wrong_target s{sev} offset not held through the grasp")
        check(np.allclose(d[r1:], 0, atol=1e-6), f"wrong_target s{sev} must rejoin the demo path")
        check(det["max_step_m"] <= 0.05 + 1e-9, f"wrong_target s{sev} step too large")
        w = (d @ delta) / float(delta @ delta)
        check(np.all(np.diff(w[:t_in]) >= -1e-9), f"wrong_target s{sev} ease-in not monotone")
        check(np.all(np.diff(w[r0:r1]) <= 1e-9), f"wrong_target s{sev} ease-out not monotone")
        check(np.isclose(w[t_in:ph["carry_start"]].min(), 1.0, atol=1e-6),
              f"wrong_target s{sev} weight not saturated at the grasp")
    check(dists[0.6] < dists[1.0], "severity must select the near distractor first")
    try:
        perturb(acts, "wrong_target", 1.0, seed=7, ctx=None)
        check(False, "wrong_target without ctx must raise SkipEpisode")
    except SkipEpisode:
        check(True, "")


def main():
    acts = synthetic_actions()
    ph = annotate_phases(acts)
    check(ph["first_close"] == APPROACH, f"first_close {ph['first_close']} != {APPROACH}")
    check(ph["carry_start"] == APPROACH + GRASP, "carry_start")
    check(ph["t_release"] == APPROACH + GRASP + CARRY, "t_release")

    test_wrong_target(acts, ph)

    for fam in common.FAMILIES:
        prev_metric = None
        for sev in common.SEVERITIES:
            p1, i1 = perturb(acts, fam, sev, seed=7)
            p2, _ = perturb(acts, fam, sev, seed=7)
            check(np.array_equal(p1, p2), f"{fam} s{sev} non-deterministic")
            check(p1.shape == acts.shape, f"{fam} shape")

            if fam in ("insufficient_grip", "premature_release", "carry_slip"):
                check(np.array_equal(p1[:, :6], acts[:, :6]),
                      f"{fam} s{sev} must not touch the arm channels")
            if fam == "insufficient_grip":
                cap = i1["detail"]["cap"]
                check(np.isclose(p1[APPROACH:, 6].max(), cap, atol=1e-6),
                      f"grip cap {p1[APPROACH:,6].max():.3f} != {cap:.3f}")
                metric = -cap                       # higher severity -> lower cap
            if fam == "premature_release":
                rel = i1["detail"]["release_frame"]
                check(ph["carry_start"] <= rel <= ph["t_release"], "release in carry")
                check((p1[rel:, 6] == 0).all(), "released stays open")
                metric = -rel                       # higher severity -> earlier
            if fam == "carry_slip":
                lo, hi = i1["detail"]["window"]
                check(ph["carry_start"] <= lo and hi <= ph["t_release"], "slip window in carry")
                check((p1[lo:hi, 6] == 0).all() and p1[hi, 6] == 1.0, "slip then re-close")
                metric = hi - lo                    # higher severity -> longer
            if fam == "wrist_tilt":
                check(np.array_equal(p1[:, [0, 1, 2, 6]], acts[:, [0, 1, 2, 6]]),
                      f"wrist_tilt s{sev} must only touch orientation")
                onset = i1["detail"]["onset"]
                check(np.array_equal(p1[:onset], acts[:onset]), "untouched before onset")
                dmax = max(
                    (R.from_euler("xyz", acts[t2, 3:6]).inv() *
                     R.from_euler("xyz", p1[t2, 3:6])).magnitude()
                    for t2 in range(onset, T))
                check(dmax <= np.deg2rad(15.0) + 1e-6,
                      f"tilt {np.rad2deg(dmax):.1f}deg exceeds 15deg")
                check(abs(np.rad2deg(dmax) - sev * 15) < 0.5,
                      f"tilt magnitude {np.rad2deg(dmax):.1f} != {sev*15:.1f}")
                metric = dmax
            if prev_metric is not None:
                check(metric >= prev_metric - 1e-9,
                      f"{fam} severity metric not monotone at s{sev}")
            prev_metric = metric

    # SkipEpisode on a no-grasp demo
    flat = synthetic_actions(); flat[:, 6] = 0.0
    for fam in FAMILY_FNS:
        try:
            perturb(flat, fam, 1.0, 0)
            check(False, f"{fam} should skip a no-grasp demo")
        except SkipEpisode:
            check(True, "")

    # Plan builder on a fake data root
    tmp = Path(tempfile.mkdtemp())
    try:
        common_orig = common.DATA_ROOT
        import make_plan
        import replay_perturbed
        for mod in (common, make_plan, replay_perturbed):
            mod.DATA_ROOT = tmp
        for i in range(6):
            d = tmp / "demos" / "T1" / f"demo_T1_0000_{i:02d}"
            d.mkdir(parents=True)
            (d / "meta.json").write_text("{}")
        for i in range(2):
            d = tmp / "val" / "T1" / f"demo_T1_1111_{i:02d}"
            d.mkdir(parents=True)
            (d / "meta.json").write_text("{}")
        rng = np.random.default_rng(2027)
        two = lambda r: r.choice(common.SEVERITIES, size=2, replace=False)  # noqa: E731
        e_train = make_plan.assign(sorted((tmp / "demos" / "T1").glob("demo_*")), rng, 2, two)
        e_val = make_plan.assign(sorted((tmp / "val" / "T1").glob("demo_*")), rng, 2, [1.0])
        check(len(e_train) == 6 * 4, f"train plan {len(e_train)} != 24")
        check(len(e_val) == 2 * 2, f"val plan {len(e_val)} != 4")
        fam_counts = {}
        for e in e_train:
            fam_counts[e["family"]] = fam_counts.get(e["family"], 0) + 1
        check(max(fam_counts.values()) - min(fam_counts.values()) <= 4,
              f"family balance off: {fam_counts}")
        check(all("replays_val" in e["out"] for e in e_val), "val replays route to replays_val")
        check(all("/replays/" in e["out"] for e in e_train), "train replays route to replays")
        for mod in (common, make_plan, replay_perturbed):
            mod.DATA_ROOT = common_orig
    finally:
        shutil.rmtree(tmp)

    print(f"\n{checks[0]}/{checks[1]} checks passed"
          + ("  ALL GREEN" if checks[0] == checks[1] else "  FAILURES ABOVE"))
    sys.exit(0 if checks[0] == checks[1] else 1)


if __name__ == "__main__":
    main()
