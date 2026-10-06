# Statistics behind the paper's tables

These are the probe outputs and the scripts that turn them into the reported numbers. They
exist so the statistics can be checked without rerunning any training, and they are not
needed to use CureWM. For that, start from the top-level README.

```bash
python3 check.py        # runs all three scripts, compares 20 reported values, PASS/FAIL
```

Standard library only: no install, no GPU, no network, about ten seconds.

| Script | Produces |
|---|---|
| `grid44.py`       | Table 1: four LIBERO suites x four metrics x three arms, with the macro-average row and the paired contrasts |
| `cw_paired_ci.py` | Table 3: the paired full-pool and held-out margins on the video world model |
| `sevauroc_ci.py`  | Table 8: AUROC at fixed perturbation type and severity, macro-averaged |

## Inputs

`probe_<arm>fa_<shard>.jsonl` are the LIBERO probe outputs, one file per arm and evaluation
shard. `<arm>` is the checkpoint probed: `pre` and `base` are the released checkpoint,
`ctrl` the step-matched control, `post` CureWM. `<shard>` names the suite and the split.
`ho_A`, `ho_B`, `ho2_A`, `ho2_B` are the four Goal held-out shards, and `spatial_*`,
`object_*`, `10_*` the three transfer suites. `visual2_{ctrl,treat}.jsonl` are the
Ctrl-World probe outputs.

Each record carries the probed value on the nominal action and on the perturbed action, the
verified outcome, and the source demonstration in its `name` field.

These are the survivor-inclusive probes: they keep the perturbed replays whose outcome
stayed successful. Those survivors are the negative class that AUROC and the false-positive
rate need.

## Statistics

Intervals are a cluster bootstrap that resamples **source demonstrations**, not individual
counterfactuals: replays of one demonstration share its scene, object placement and
trajectory, so treating them as independent would understate the interval. 20,000
resamples, seed 0, percentile intervals. `*` in the grid output marks an interval that
excludes zero.

Arms are compared pairwise, matched record by record on the episode identifier, so a
difference is a mean of within-episode differences rather than a difference of means.

Results do not depend on the interpreter's hash seed: cluster keys come from
insertion-ordered dictionaries or are sorted explicitly, and each bootstrap uses its own
`random.Random(0)` rather than the global generator.

## One rounding note

The macro-average AUROC of the released checkpoint is exactly 0.4895. The paper's Table 1
rounds half-up and prints `0.490`, while Python's `%.3f` rounds half-to-even and prints
`0.489`.
The underlying value is the same. Nothing else in these tables sits on a rounding boundary.
