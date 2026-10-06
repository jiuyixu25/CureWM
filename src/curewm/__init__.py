"""CureWM: repairing failure insensitivity in robot world models with verified
counterfactual replay.

The engine lives in `curewm.perturbations`: six perturbation families parameterized by a
severity in [0, 1], a phase annotator that decides where each family may act, and
`generate_pairs`, which replays every (demonstration, family, severity, seed) cell and
labels the outcome with the task's own success predicate.  A perturbation is never
assumed to fail.

Simulator bindings are in `curewm.backends`. Each only has to satisfy the `SimBackend`
protocol declared in `curewm.perturbations`.
"""

from curewm.perturbations import (
    ApproachOvershoot,
    CarrySlip,
    ContactOscillation,
    InsufficientGrip,
    Perturbation,
    Phase,
    PrematureRelease,
    SimBackend,
    Trajectory,
    WristTilt,
    annotate_phases,
    families_for,
    generate_pairs,
    sanity_report,
)

__all__ = [
    "ApproachOvershoot", "CarrySlip", "ContactOscillation", "InsufficientGrip",
    "Perturbation", "Phase", "PrematureRelease", "SimBackend", "Trajectory", "WristTilt",
    "annotate_phases", "families_for", "generate_pairs", "sanity_report",
]
