"""Engine contract tests that need no simulator.

A stub backend stands in for MuJoCo: it executes an action sequence against a toy model
of a pick-and-place task where the object is dropped if the gripper opens while carrying,
or if the grip command is too weak to hold it.  That is enough to exercise everything the
engine is responsible for, namely phase annotation, where each family is allowed to act,
the severity parameterization and the pairing protocol. It also asserts the property the
paper calls Gate 1: a family's failure rate must rise with severity.

    python3 -m pytest tests/ -q          (or: python3 tests/test_engine.py)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curewm.perturbations import (  # noqa: E402
    GRIP_CLOSE_THRESH, SEVERITY_GRID, Phase, Trajectory,
    annotate_phases, families_for, generate_pairs, sanity_report,
)

HOLD_THRESH = 0.35      # a grip command above this is too weak to hold the object
T_GRASP, T_LIFT, T_PLACE = 10, 16, 34
SETTLE = 3              # steps between the close command and sustained contact


def nominal_actions(T: int = 40) -> np.ndarray:
    """Approach with the gripper open, close on it, carry, then release over the target."""
    a = np.zeros((T, 7), dtype=np.float64)
    a[:, 2] = -0.02                       # descend
    a[T_GRASP:, 2] = 0.0
    a[T_LIFT:T_PLACE, 0] = 0.03           # carry sideways
    a[T_LIFT:T_PLACE, 2] = 0.01           # and up
    a[:, 6] = 1.0                         # open
    a[T_GRASP:T_PLACE, 6] = 0.0           # closed through grasp and carry
    a[T_PLACE:, 6] = 1.0                  # release
    return a


class StubBackend:
    """Toy pick-and-place. Holds the object while the grip command stays firm. The task
    succeeds only if the object is still held when it arrives over the target."""

    def reset_to(self, init_state: dict) -> None:
        self._held = False

    def rollout(self, actions: np.ndarray) -> dict:
        T = len(actions)
        contacts = np.zeros(T, dtype=bool)
        held, dropped_at = False, None
        x = 0.0
        for t in range(T):
            g = actions[t, 6]
            if (not held and g < GRIP_CLOSE_THRESH and dropped_at is None
                    and t >= T_GRASP + SETTLE):      # contact settles a few steps after closing
                held = True
            if held and g > HOLD_THRESH:              # opened, or never gripped firmly
                held, dropped_at = False, t
            contacts[t] = held
            if held:
                x += actions[t, 0]
        carried = x > 0.25                            # reached the target region
        released_over_target = dropped_at is not None and dropped_at >= T_PLACE
        success = bool(carried and (held or released_over_target))
        return {
            "frames": np.zeros((T, 2, 2, 3), dtype=np.uint8),
            "contacts": contacts,
            "success": success,
            "obj_states": np.zeros((T, 1)),
            "ee_states": np.zeros((T, 1)),
        }


def make_demo() -> Trajectory:
    tr = Trajectory(actions=nominal_actions(), init_state={"seed": 0}, task_id="StubPick")
    tr.contacts = StubBackend().rollout(tr.actions)["contacts"]
    tr.phases = annotate_phases(tr)
    return tr


def test_nominal_succeeds():
    assert StubBackend().rollout(nominal_actions())["success"], \
        "the unperturbed demonstration must succeed, or nothing downstream means anything"


def test_phases_cover_the_trajectory():
    ph = make_demo().phases
    seen = {Phase(v).name for v in np.unique(ph)}
    assert {"APPROACH", "GRASP", "CARRY"} <= seen, f"phase annotation lost a phase: {seen}"


def test_strength_increases_with_severity():
    """Over the grid that is actually used, a family's departure from the demonstration
    must not shrink as severity rises.  Severity 0 is deliberately not tested: the
    unperturbed trajectory comes from replaying the demonstration, not from apply(.., 0)."""
    demo = make_demo()
    for fam in families_for(demo.task_id):
        mags = []
        for s in SEVERITY_GRID:
            # common random numbers, as generate_pairs uses: the seed excludes severity
            out = fam.apply(demo, s, np.random.default_rng(11))
            assert out.shape == demo.actions.shape
            mags.append(float(np.abs(out - demo.actions).sum()))
        assert mags[-1] >= mags[0], \
            f"{type(fam).__name__}: departure fell from {mags[0]:.3f} to {mags[-1]:.3f} across the grid"
        assert mags[-1] > 0, f"{type(fam).__name__} never modifies the trajectory"


def test_families_only_touch_their_own_phases():
    demo = make_demo()
    rng = np.random.default_rng(0)
    for fam in families_for(demo.task_id):
        changed = np.any(fam.apply(demo, 1.0, rng) != demo.actions, axis=1)
        allowed = np.isin(demo.phases, [p.value for p in fam.target_phases])
        stray = int(np.sum(changed & ~allowed))
        assert stray == 0, f"{type(fam).__name__} modified {stray} steps outside its phases"


def test_randomness_is_reproducible():
    """A given (demo, family, severity, seed) must reproduce exactly."""
    demo = make_demo()
    for fam in families_for(demo.task_id):
        a = fam.apply(demo, 0.6, np.random.default_rng(7))
        b = fam.apply(demo, 0.6, np.random.default_rng(7))
        assert np.array_equal(a, b), f"{type(fam).__name__} is not reproducible under a fixed seed"


def test_generate_pairs_labels_outcomes_by_replay():
    """The pairing protocol writes one nominal plus every (family, severity, seed) cell,
    and every outcome comes from a replay rather than an assumption."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "pairs"
        index = generate_pairs([make_demo()], StubBackend(), out, seeds_per_cell=1)
        rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]

    n_fam = len(families_for("StubPick"))
    assert len(rows) == 1 + n_fam * len(SEVERITY_GRID), \
        f"expected 1 nominal + {n_fam}x{len(SEVERITY_GRID)} cells, got {len(rows)}"
    assert sum(r["family"] == "nominal" for r in rows) == 1
    assert all(isinstance(r["outcome"], bool) for r in rows), "every outcome must be a replayed label"
    assert any(r["outcome"] is False for r in rows), "no family ever failed; the stub task is too forgiving"
    assert any(r["outcome"] is True and r["family"] != "nominal" for r in rows), \
        "every perturbation failed; a severity grid that never survives cannot build graded partners"


def test_failure_rate_rises_with_severity():
    """Gate 1 of the paper: within a family, failures must become more frequent as
    severity rises.  Checked on the families the stub task can actually express."""
    with tempfile.TemporaryDirectory() as d:
        index = generate_pairs([make_demo()], StubBackend(), Path(d) / "pairs", seeds_per_cell=2)
        report = sanity_report(index)
        rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]

    rates = {}
    for r in rows:
        if r["family"] != "nominal":
            rates.setdefault(r["family"], {}).setdefault(r["severity"], []).append(not r["outcome"])

    expressive = {f: s for f, s in rates.items()
                  if any(any(v) for v in s.values()) and not all(all(v) for v in s.values())}
    assert expressive, "the stub task expressed no graded family; the test would be vacuous"
    for fam, by_sev in expressive.items():
        sevs = sorted(by_sev)
        fr = [float(np.mean(by_sev[s])) for s in sevs]
        assert fr[-1] >= fr[0], f"{fam}: failure rate fell from {fr[0]:.2f} to {fr[-1]:.2f} as severity rose"
    assert "families" in report or report, "sanity_report returned nothing"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}\n      {e}")
    print(f"\n{'all tests passed' if not fails else f'{fails} test(s) failed'}")
    sys.exit(1 if fails else 0)
