<div align="center">

<h1>World Models Dream of Success</h1>
<h3>Diagnosing and Repairing Failure Insensitivity in Robot World Models</h3>

<p>
  Jiuyi Xu<sup>1</sup> &nbsp;&nbsp;
  Xiao Hu<sup>2</sup> &nbsp;&nbsp;
  Meida Chen<sup>3</sup> &nbsp;&nbsp;
  Peng Gao<sup>4</sup> &nbsp;&nbsp;
  Yang Ye<sup>2</sup> &nbsp;&nbsp;
  Yangming Shi<sup>1</sup>
</p>

<p>
  <sup>1</sup>Colorado School of Mines &nbsp;&nbsp;&nbsp;
  <sup>2</sup>Northeastern University <br>
  <sup>3</sup>Institute for Creative Technologies, University of Southern California &nbsp;&nbsp;&nbsp;
  <sup>4</sup>North Carolina State University
</p>

<p>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square" alt="Python 3.10+"></a>
</p>

<img src="docs/overview.png" width="96%" alt="Perturb a successful demonstration across a severity grid, verify every outcome by execution, and post-train on the verified failures and the surviving successes.">

</div>

---

Released robot world models are used to evaluate policies, score action candidates and
generate training data. All three uses assume the model turns pessimistic when an action
would fail. Across four released checkpoints from two architecture families, it often does
not: failing actions keep high values and success-like futures.

**CureWM repairs a released checkpoint with data alone** — no architecture change, no loss
change. Take a successful demonstration, perturb its actions across a severity grid,
**execute each variant and let the task decide the outcome**, then fine-tune on the
verified failures and the surviving successes alongside the original data.

The point is the contrast: the same starting context with two different actions and two
different outcomes. A single action per context lets a model fit outcomes without ever
learning what distinguishes a good action from a bad one.

## Install

```bash
git clone https://github.com/jiuyixu25/CureWM.git && cd CureWM
pip install -e .          # the engine needs numpy and nothing else
```

Check it works — no simulator, no GPU, about two seconds:

```bash
python3 tests/test_engine.py
```

```
PASS  test_failure_rate_rises_with_severity
PASS  test_families_only_touch_their_own_phases
PASS  test_generate_pairs_labels_outcomes_by_replay
PASS  test_nominal_succeeds
PASS  test_phases_cover_the_trajectory
PASS  test_randomness_is_reproducible
PASS  test_strength_increases_with_severity
```

Simulators and the world-model stack are separate installs; see [`requirements/`](requirements/).

## Run it on your own setup

### 1. Give the engine a simulator

Anything that can reset to a state and execute an action sequence works. That is the whole
interface:

```python
class SimBackend(Protocol):
    def reset_to(self, init_state: dict) -> None: ...
    def rollout(self, actions: np.ndarray) -> dict:
        """-> {"frames", "contacts", "success", "obj_states", "ee_states"}"""
```

LIBERO and ManiSkill3 bindings ship in [`src/curewm/backends/`](src/curewm/backends). Use
them as the template for your own simulator — each is about 130 lines, and most of that is
converting between the simulator's gripper sign convention and the engine's
(`grip in [0 closed, 1 open]`).

### 2. Build verified counterfactuals

```bash
PYTHONPATH=src python3 scripts/generate_libero.py \
    --suite libero_goal --demos-per-task 10 --out data/pairs
```

Each demonstration produces one nominal replay plus one episode per
*(family, severity, seed)* cell. Every episode is **executed and labelled by the task's own
success predicate** — a perturbation is never assumed to fail, and the ones that survive
are kept as graded successful partners rather than discarded.

```python
from curewm import generate_pairs, sanity_report

index  = generate_pairs(demos, backend, Path("data/pairs"))
report = sanity_report(index)     # failure rate must rise with severity, per family
```

`sanity_report` enforces the screening condition from the paper: a family enters training
only if its empirical failure rate is non-decreasing in severity. A family that fails that
check is miscalibrated for your task — rescale it in `TASK_SEVERITY_SCALE` rather than
shipping it.

### 3. Convert to your model's training format

```bash
python3 scripts/convert_to_rollout_format.py --src data/pairs --out data/rollout_cure
```

Failures get zero value targets; surviving replays keep the original labelling rule. For
video-only models, supervision is the frames recorded during execution.

### 4. Post-train

```bash
export CUREWM_ROOT=/path/to/workspace
export CUREWM_ROLLOUT_DIR=data/rollout_cure
export CUREWM_INIT_PT=/path/to/released_weights.pt
bash cluster/train.sh
```

Every arm in the paper is this same launcher with a different mixture — the step-matched
control, the own-failure ablation, the other-demonstration ablation. Same initialization,
schedule, seed, batch size and sampling ratios; only the data differs.

### 5. Measure what changed

```bash
python3 scripts/probe_optimism.py --ckpt <checkpoint> --limit 100
```

Reports **value optimism** (the rate at which a failing action still scores above 0.5) and
**visual optimism** (the rate at which the imagined future looks more like the success
ending than the failure ending).

> Report optimism together with a discrimination measure. Optimism alone can fall simply
> because a model became globally pessimistic, which is not a repair. The paper uses
> $\Delta_{\mathrm{SF}}$, the success-failure value gap, and AUROC alongside it.

## The six perturbation families

Each acts only on the phases it targets, with strength set by a severity in
`SEVERITY_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)`.

| Family | Acts on | At severity `s` |
|---|---|---|
| `insufficient_grip`   | grasp, carry | grip command moved toward open by `0.9s`, capping closure at `1 - 0.9s` |
| `premature_release`   | carry, place | release frame interpolated toward the start of the carry; at `s=1` it releases immediately |
| `carry_slip`          | carry        | `max(1, round(3s))` brief open pulses of `0.5 + 0.5s`, plus lateral spikes |
| `contact_oscillation` | grasp, carry | oscillation of amplitude `0.7s` about the contact axis |
| `wrist_tilt`          | carry        | sustained tilt of `0.8s` on one wrist axis |
| `approach_overshoot`  | approach     | the approach extended by `0.6s` along its own direction |

Phases come from the gripper command and the recorded contact events; an ambiguous step is
labelled `OTHER` and no family touches it. The random seed deliberately **excludes
severity**, so along one severity axis the pulse positions and axis choices are identical
and the strength effect is not swamped by positional randomness.

Adding a family means subclassing `Perturbation` with a `family` name, `target_phases`, and
an `apply(traj, severity, rng)`.

## On hardware

[`real_robot/`](real_robot/) runs the same protocol on a Franka arm through
[DROID](https://github.com/droid-dataset/droid): scripted demonstration collection,
action-by-action replay, perturbation at a chosen severity, and outcome labelling from
gripper telemetry with an operator veto. Everything is configured by environment variable;
see [`real_robot/README.md`](real_robot/README.md).

## Results

Full tables and analysis are in the paper. Headline numbers:

| | Released | Fine-tuned on official data | **CureWM** |
|---|---|---|---|
| LIBERO-Goal, value optimism on 484 held-out failures | 79.1% | 79.6% | **30.2%** |
| LIBERO-Goal, success-failure value gap $\Delta_{\mathrm{SF}}$ | +0.002 | +0.003 | **+0.282** |
| LIBERO mean over four suites, AUROC | 0.490 | 0.492 | **0.613** |
| Franka cup task, optimism (second evaluation) | 15/15 | 12/15 | **4/15** |

Closed-loop task success is preserved: averaged over the four LIBERO suites, 96.6% for
CureWM against 96.8% for the step-matched control.

[`analysis/`](analysis/) contains the probe outputs behind these tables and the scripts
that compute them, for anyone who wants to check the statistics rather than rerun the
training.

## Repository layout

```
src/curewm/          the engine: perturbation families, phase annotation, generate_pairs
  backends/            LIBERO and ManiSkill3 bindings
scripts/             command-line entry points for each pipeline stage
real_robot/          hardware pipeline for a Franka arm
cluster/             post-training launcher and the experiment-config block
analysis/            probe outputs and the statistics behind the paper's tables
tests/               engine contract tests, no simulator required
```

## What is not here

Model weights, fine-tuned checkpoints and the simulator datasets are not redistributed, and
neither is any third-party source. In particular the upstream Cosmos-Policy experiment
config carries an NVIDIA proprietary notice, so
[`cluster/experiment_config_additions.py`](cluster/experiment_config_additions.py) ships
only the block we append to it, with instructions. See [`NOTICE`](NOTICE).

## License

MIT, see [`LICENSE`](LICENSE). Third-party components keep their own licenses; see
[`NOTICE`](NOTICE).
