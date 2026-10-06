"""Simulator bindings.

Each backend satisfies the `SimBackend` protocol in `curewm.perturbations` and converts
between the simulator's action convention and the engine's, which is
`grip in [0 closed, 1 open]`.  The conversion happens only at the backend boundary, so
nothing above it has to know a simulator's sign convention.

The imports are deliberately lazy: `curewm.backends.libero` needs a LIBERO checkout and
`curewm.backends.maniskill` needs ManiSkill3, and neither should be required to use the
other or to use the engine alone.
"""
