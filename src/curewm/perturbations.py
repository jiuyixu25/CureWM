"""Failure data engine: the six perturbation families and the pairing protocol.

Follows the failure-inducing families of MiraBench (arXiv 2605.29360) to generate paired
trajectories for failure-aware world-model post-training:

    (o_0, a+_{1:T}, o+_{1:T}, outcome=success)   nominal demonstration
    (o_0, a-_{1:T}, o-_{1:T}, outcome=?)         same initial state, perturbed replay

Design constraints:
- Perturbations act on the end-effector action sequence, aligned with the relative-EE
  action space, and are decoupled from the simulator: a backend only has to satisfy the
  SimBackend protocol.
- A perturbed trajectory is always replayed and labelled by the task's own success
  predicate.  A perturbation is never assumed to fail.  Gate 1 requires each family's
  failure rate to rise monotonically with severity.
- A perturbation is injected only into its target phases, which come from the gripper
  command and the simulated contact events.
- All randomness goes through an explicit rng, so a given (demo, family, severity, seed)
  reproduces exactly.

Importing this module gives the families and generate_pairs(). A backend and a demo loader
supply the rest.  generate_pairs writes <out_dir>/index.jsonl and <out_dir>/episodes/*.npz.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

# ---------------------------------------------------------------- core types

class Phase(Enum):
    APPROACH = 0   # moving toward the target, gripper open
    GRASP = 1      # from the closing command to the first sustained contact
    CARRY = 2      # moving while holding the object
    PLACE = 3      # placing and releasing
    OTHER = 4


@dataclass
class Trajectory:
    """One end-effector trajectory.

    actions[t] = [dx, dy, dz, drx, dry, drz, grip] with grip in [0 closed, 1 open].
    """

    actions: np.ndarray                 # (T, 7)
    init_state: dict                    # full initial state the simulator can reset to
    task_id: str
    contacts: np.ndarray | None = None  # (T,) bool, filled in by the simulator
    phases: np.ndarray | None = None    # (T,) Phase.value, filled in by annotate_phases
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.actions)


class SimBackend(Protocol):
    """Simulator adapter.  Implemented for LIBERO (MuJoCo), RoboCasa and ManiSkill3."""

    def reset_to(self, init_state: dict) -> None: ...

    def rollout(self, actions: np.ndarray) -> dict:
        """Execute the action sequence and return

        {"frames": (T,H,W,3) uint8, "contacts": (T,) bool,
         "success": bool, "obj_states": (T, ...), "ee_states": (T, ...)}
        """
        ...


# ---------------------------------------------------------------- phase annotation

GRIP_CLOSE_THRESH = 0.5  # a grip command below this counts as intent to close


def annotate_phases(traj: Trajectory) -> np.ndarray:
    """Split a trajectory into phases from the gripper command and the contact events.

    Requires traj.contacts, filled in by one nominal replay.  The rules are deliberately
    conservative: an ambiguous step is labelled OTHER and no perturbation touches it.
    """
    assert traj.contacts is not None, "run one nominal replay first to fill in contacts"
    g = traj.actions[:, 6]
    phases = np.full(traj.T, Phase.OTHER.value, dtype=np.int8)

    closing = g < GRIP_CLOSE_THRESH
    first_close = int(np.argmax(closing)) if closing.any() else traj.T
    contact_after_close = traj.contacts.copy()
    contact_after_close[:first_close] = False
    first_stable = int(np.argmax(contact_after_close)) if contact_after_close.any() else traj.T
    reopen = closing.copy()
    reopen[: first_stable + 1] = True  # opening before the grasp is stable is not a release
    first_release = int(np.argmax(~reopen)) if (~reopen).any() else traj.T

    phases[:first_close] = Phase.APPROACH.value
    phases[first_close:first_stable] = Phase.GRASP.value
    phases[first_stable:first_release] = Phase.CARRY.value
    phases[first_release:] = Phase.PLACE.value
    traj.phases = phases
    return phases


# ---------------------------------------------------------------- perturbation families

class Perturbation(ABC):
    """A perturbation family, parameterized by a severity in [0, 1].

    Strength increases with severity, and the grid actually used is SEVERITY_GRID, which
    starts at 0.2.  Severity 0 is not part of it: the unperturbed trajectory is obtained
    by replaying the demonstration itself, not by calling apply(.., 0.0), so no family is
    required to be the exact identity there.  Five of the six are. Carry_slip still emits
    one pulse because its pulse count floors at one.
    """

    family: str
    target_phases: tuple[Phase, ...]

    @abstractmethod
    def apply(self, traj: Trajectory, severity: float, rng: np.random.Generator) -> np.ndarray:
        """Return a perturbed copy of actions. Never modify in place."""

    def _mask(self, traj: Trajectory) -> np.ndarray:
        assert traj.phases is not None, "call annotate_phases first"
        return np.isin(traj.phases, [p.value for p in self.target_phases])


class InsufficientGrip(Perturbation):
    """1/6 insufficient grip: pull the grip command back toward open from the closing phase on."""
    family = "insufficient_grip"
    target_phases = (Phase.GRASP, Phase.CARRY)

    def apply(self, traj, severity, rng):
        a = traj.actions.copy()
        m = self._mask(traj)
        a[m, 6] = a[m, 6] + severity * (1.0 - a[m, 6]) * 0.9
        return a


class PrematureRelease(Perturbation):
    """2/6 premature release: move the release moment linearly back from place into carry."""
    family = "premature_release"
    target_phases = (Phase.CARRY, Phase.PLACE)

    def apply(self, traj, severity, rng):
        a = traj.actions.copy()
        assert traj.phases is not None
        carry_idx = np.where(traj.phases == Phase.CARRY.value)[0]
        if len(carry_idx) == 0:
            return a
        t_nominal_release = int(carry_idx[-1]) + 1
        t_new = int(round(carry_idx[0] + (1.0 - severity) * (t_nominal_release - carry_idx[0])))
        a[t_new:, 6] = 1.0  # stay open from the new release moment on
        return a


class CarrySlip(Perturbation):
    """3/6 carry slip: brief open pulses plus lateral spikes during the carry phase."""
    family = "carry_slip"
    target_phases = (Phase.CARRY,)

    def apply(self, traj, severity, rng):
        a = traj.actions.copy()
        idx = np.where(self._mask(traj))[0]
        if len(idx) < 4:
            return a
        n_pulse = max(1, int(round(severity * 3)))
        for t in rng.choice(idx[1:-1], size=min(n_pulse, len(idx) - 2), replace=False):
            a[t, 6] = min(1.0, a[t, 6] + 0.5 + 0.5 * severity)      # momentary release
            a[t, 0:2] += rng.normal(0, 0.5 * severity, size=2)       # lateral spike, normalised action units
        return a


class ContactOscillation(Perturbation):
    """4/6 contact oscillation: a sinusoid along the approach axis during grasp and carry.

    The grasp window is often only a few steps and PickCube-style demonstrations have no
    place phase, so this family has to cover carry as well.
    """
    family = "contact_oscillation"
    target_phases = (Phase.GRASP, Phase.CARRY)

    def apply(self, traj, severity, rng):
        a = traj.actions.copy()
        idx = np.where(self._mask(traj))[0]
        if len(idx) == 0:
            return a
        amp = 0.7 * severity                         # normalised action units, +/-1 is full scale
        omega = 2 * np.pi / max(4, len(idx) // 3)
        a[idx, 2] += amp * np.sin(omega * np.arange(len(idx)))  # z is the approach axis by default
        return a


class WristTilt(Perturbation):
    """5/6 wrist tilt: a sustained rotation about a horizontal axis once the object is held."""
    family = "wrist_tilt"
    target_phases = (Phase.CARRY,)

    def apply(self, traj, severity, rng):
        a = traj.actions.copy()
        idx = np.where(self._mask(traj))[0]
        if len(idx) < 3:
            return a
        axis = int(rng.integers(3, 5))               # drx or dry
        third = len(idx) // 3
        window = idx[third: third + max(2, third)]   # one contiguous window in the middle of carry
        a[window, axis] += 0.8 * severity            # normalised action units, sustained tilt
        return a


class ApproachOvershoot(Perturbation):
    """6/6 approach overshoot: extend the final approach along the motion direction, into the target."""
    family = "approach_overshoot"
    target_phases = (Phase.APPROACH,)

    def apply(self, traj, severity, rng):
        a = traj.actions.copy()
        idx = np.where(self._mask(traj))[0]
        if len(idx) < 2:
            return a
        tail = idx[-max(2, len(idx) // 4):]
        # Additive push along the nominal direction of the last steps.  A multiplicative
        # gain does almost nothing there, because the approach is already decelerating.
        d = a[tail, 0:3]
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        mean_dir = d.sum(0) / max(np.linalg.norm(d.sum(0)), 1e-6)
        a[tail, 0:3] = d + (0.6 * severity) * np.where(norms > 1e-4, d / np.maximum(norms, 1e-6), mean_dir)
        return a


ALL_FAMILIES: list[Perturbation] = [
    InsufficientGrip(), PrematureRelease(), CarrySlip(),
    ContactOscillation(), WristTilt(), ApproachOvershoot(),
]

# Family-by-task compatibility.  Some mechanisms cannot physically cause a failure on a
# given task, and turning up the amplitude only manufactures a false signal.  PickCube, for
# instance, judges success from the object position alone, which a wrist tilt does not change.
TASK_FAMILY_EXCLUDE: dict[str, set[str]] = {
    "PickCube-v1": {"wrist_tilt"},
}

# Per-task severity scaling.  High-precision tasks such as peg insertion saturate across the
# whole grid and lose the informative part of the dose-response curve.  The scale applies to
# the applied strength only. The nominal severity is what gets recorded.
TASK_SEVERITY_SCALE: dict[str, float] = {
    "PegInsertionSide-v1": 0.35,
}

# Finer grain: an extra scale for one (task, family) pair, where the task is otherwise
# well calibrated but a single family saturates.
TASK_FAMILY_SEVERITY_SCALE: dict[tuple[str, str], float] = {
    ("LiftPegUpright-v1", "approach_overshoot"): 0.3,
}


def families_for(task_id: str) -> list[Perturbation]:
    excl = TASK_FAMILY_EXCLUDE.get(task_id, set())
    return [f for f in ALL_FAMILIES if f.family not in excl]

# ---------------------------------------------------------------- generation pipeline

SEVERITY_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)


def generate_pairs(demos: list[Trajectory], backend: SimBackend, out_dir: Path,
                   severity_grid=SEVERITY_GRID, seeds_per_cell: int = 2) -> Path:
    """Replay each demonstration, then every (family, severity, seed) cell, and label the outcome.

    The nominal replay fills in contacts and phases first.  Writes out_dir/index.jsonl and
    out_dir/episodes/*.npz and returns the index path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(exist_ok=True)
    index = open(out_dir / "index.jsonl", "a", encoding="utf-8")
    n = 0
    for di, demo in enumerate(demos):
        backend.reset_to(demo.init_state)
        nominal = backend.rollout(demo.actions)
        if not nominal["success"]:
            continue  # keep only demonstrations that replay successfully
        demo.contacts = nominal["contacts"]
        annotate_phases(demo)
        extra = {k: v for k, v in demo.meta.items() if k in ("task_name", "language", "demo")}
        _dump(out_dir, f"d{di:04d}_nominal", demo.actions, nominal, index,
              dict(task=demo.task_id, family="nominal", severity=0.0, outcome=True, **extra))
        task_scale = TASK_SEVERITY_SCALE.get(demo.task_id, 1.0)
        for fam in families_for(demo.task_id):
            scale = task_scale * TASK_FAMILY_SEVERITY_SCALE.get((demo.task_id, fam.family), 1.0)
            for s in severity_grid:
                for k in range(seeds_per_cell):
                    # Common random numbers: the seed excludes severity, so the random
                    # pulse positions and axis choices are identical along the whole
                    # severity axis for one (demo, k).  Otherwise the positional
                    # randomness swamps the strength effect and breaks monotonicity.
                    rng = np.random.default_rng(hash((di, fam.family, k)) % 2**32)
                    pa = fam.apply(demo, s * scale, rng)
                    backend.reset_to(demo.init_state)
                    res = backend.rollout(pa)
                    _dump(out_dir, f"d{di:04d}_{fam.family}_s{s:.1f}_k{k}", pa, res, index,
                          dict(task=demo.task_id, family=fam.family, severity=s,
                               outcome=bool(res["success"]), pair_of=f"d{di:04d}_nominal", **extra))
                    n += 1
    index.close()
    print(f"[curewm] wrote {n} perturbed trajectories -> {out_dir}")
    return out_dir / "index.jsonl"


def _dump(out_dir: Path, name: str, actions: np.ndarray, res: dict, index, meta: dict):
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    # Uncompressed savez: at 224 squared with two cameras, zlib is the throughput
    # bottleneck and costs a measured 3-4x.  Trade disk for time.
    np.savez(out_dir / "episodes" / f"{name}.npz", actions=actions, **arrays)
    index.write(json.dumps({"name": name, **meta}, ensure_ascii=False) + "\n")
    index.flush()


# ---------------------------------------------------------------- Gate 1 check

def sanity_report(index_path: Path, tol: float = 0.1) -> dict:
    """Gate 1: each family's failure rate must rise monotonically with severity.

    A non-monotonic family is mis-calibrated and must not enter post-training.
    tol is the dip allowed for sampling noise: 0.1 for an n=10 smoke test, where a single
    flip is 0.1, and 0.05 for a full dataset with n >= 50.
    """
    rows = [json.loads(l) for l in open(index_path, encoding="utf-8")]
    report: dict = {}
    for fam in {r["family"] for r in rows if r["family"] != "nominal"}:
        by_s: dict[float, list[bool]] = {}
        for r in rows:
            if r["family"] == fam:
                by_s.setdefault(r["severity"], []).append(r["outcome"])
        curve = {s: 1.0 - float(np.mean(v)) for s, v in sorted(by_s.items())}
        vals = list(curve.values())
        report[fam] = {"failure_rate_by_severity": curve,
                       "monotonic": bool(all(b >= a - tol for a, b in zip(vals, vals[1:])))}
    return report


if __name__ == "__main__":
    # This module is a library: a simulator backend and a demo loader drive it.
    # Running it directly only prints the protocol above.
    print(__doc__)
