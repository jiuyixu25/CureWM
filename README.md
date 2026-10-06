# CureWM

Robot world models are increasingly used as evaluation and training infrastructure, a role
that assumes their predictions turn pessimistic when an action would fail. Released models
often do not: they score failing actions as confidently as succeeding ones and render
success under actions that break the task. **CureWM repairs this with data alone** —
physically verified counterfactual replays, mixed into ordinary fine-tuning. No
architecture change, no loss change.

The core idea is one function. Take a successful demonstration, perturb its action
sequence in a way whose strength you control, replay it, and **let the task's own success
predicate decide the label**. A perturbation is never assumed to fail. What survives
becomes a graded successful partner; what fails becomes a verified counterfactual. Train on
the pair.

```python
from curewm import generate_pairs, families_for

index = generate_pairs(demos, backend, out_dir)   # one nominal + (family x severity x seed)
```

## Install

```bash
git clone <this repo> && cd curewm
pip install -e .                 # the engine: numpy only
python3 tests/test_engine.py     # 7 checks, no simulator needed
```

Simulators and the world-model stack are separate installs, because almost nobody needs
all of them. See `requirements/` for the tiers: `core`, `libero`, `maniskill`,
`real_robot`, `probes`.

## What is here

```
src/curewm/            the engine
  perturbations.py       six families, phase annotation, generate_pairs, sanity_report
  backends/              LIBERO and ManiSkill3 bindings (SimBackend protocol)
scripts/               command-line entry points
  generate_libero.py     demonstrations -> counterfactual pairs, in LIBERO
  generate_maniskill.py  the same, in ManiSkill3
  collect_demos.py       motion-planned demonstration collection
  convert_to_*.py        pairs -> the world model's training formats
  probe_optimism.py      measure value and visual optimism on a checkpoint
analysis/              reproduce the paper's numbers, with the probe outputs included
real_robot/            the hardware pipeline for a Franka arm
cluster/               post-training launcher and the experiment-config block
tests/                 engine contract tests, no simulator required
```

## Reproducing the reported numbers

```bash
cd analysis && python3 check.py
```

Standard library only, about ten seconds, no GPU and no network. It runs the three
analysis scripts, compares twenty reported values against what they recompute, prints
PASS or FAIL for each, and exits non-zero if any row fails.

| Script | Reproduces |
|---|---|
| `cw_paired_ci.py` | Paired held-out margin on the video world model, control minus CureWM |
| `sevauroc_ci.py`  | Family x severity stratified macro-AUROC |
| `grid44.py`       | The LIBERO table: 4 suites x 4 metrics x 3 arms, with paired contrasts |

The probe outputs these read are included as inputs. Regenerating *them* needs the
fine-tuned checkpoints and a GPU, which this repository does not carry; see below.

## The pipeline

**1. Build counterfactuals.** Each family acts only on the phases it targets, with
strength set by a severity in `SEVERITY_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)`:

| Family | Acts on | At severity s |
|---|---|---|
| `insufficient_grip`  | grasp, carry | grip command moved toward open by `0.9s`, so closure is capped at `1 - 0.9s` |
| `premature_release`  | carry, place | release frame interpolated toward the start of the carry; at `s=1` it releases immediately |
| `carry_slip`         | carry        | `max(1, round(3s))` brief open pulses of `0.5 + 0.5s`, plus lateral spikes |
| `contact_oscillation`| grasp, carry | oscillation of amplitude `0.7s` about the contact axis |
| `wrist_tilt`         | carry        | sustained tilt of `0.8s` on one wrist axis |
| `approach_overshoot` | approach     | the approach is extended by `0.6s` along its own direction |

```bash
PYTHONPATH=src python3 scripts/generate_libero.py --suite libero_goal --out data/pairs
```

Every cell is replayed and labelled. `sanity_report` checks the property the paper calls
Gate 1: within a family, the failure rate must rise with severity. The random seed excludes
severity on purpose, so along one severity axis the pulse positions and axis choices are
identical and the strength effect is not swamped by positional randomness.

**2. Convert.** `scripts/convert_to_rollout_format.py` writes the rollout channel the world
model trains on; `convert_to_cosmos_format.py` writes the demonstration layout. Failures
get all-zero rewards; a perturbed replay that happened to succeed gets its verified
outcome. That is where the value supervision comes from.

**3. Post-train.** `cluster/train.sh`, pointed at a rollout mixture. Every arm in the paper
is the same launcher with a different mixture and nothing else changed — same
initialization, schedule, seed, batch size and sampling ratios. The experiment config is
applied as described in `cluster/experiment_config_additions.py`.

**4. Probe and analyse.** `scripts/probe_optimism.py` against a checkpoint, then the
scripts in `analysis/`.

## Hardware

`real_robot/` runs the same protocol on a Franka arm through DROID: scripted demonstration
collection, action-by-action replay, perturbation at a chosen severity, and outcome
labelling from gripper telemetry with an operator veto.

Configure by environment, never by editing the scripts:

```bash
export CUREWM_ROBOT_HOST=user@control-box     # key-based SSH; no passwords anywhere
export CUREWM_DATA_ROOT=~/curewm_data
bash real_robot/restart_stack.sh              # fresh server before every run
```

`restart_stack.sh` needs passwordless sudo on the control box for two `pkill` lines; it is
documented in the script's header.

## What this repository does not contain

- **Model weights and checkpoints.** Obtain the released world models from their own
  sources; fine-tuned checkpoints are too large to distribute.
- **Datasets.** LIBERO, RoboCasa and the hardware recordings are not included. The probe
  outputs under `analysis/` are the exception, because the reported statistics are computed
  from them.
- **Any third-party source.** In particular the upstream Cosmos-Policy experiment config
  carries an NVIDIA proprietary notice, so `cluster/experiment_config_additions.py` ships
  only the block we append, with instructions. See `NOTICE`.

## Citation

```bibtex
@inproceedings{curewm,
  title     = {CureWM},
  author    = {},
  booktitle = {},
  year      = {}
}
```

## License

MIT, see `LICENSE`. Third-party components keep their own licenses; see `NOTICE`.
