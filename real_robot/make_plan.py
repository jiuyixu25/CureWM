"""Build the randomized replay plan for CureWM-Real.

Train demos: 2 families x 2 severities = 4 replays each, family assignment
cycled through a shuffled deck so all four families get equal coverage.
Val demos (held-out split): 2 families x severity 1.0 = 2 replays each,
maximizing failure odds for the probe pairs.

Deterministic under --global-seed; --append keeps existing entries stable and
only adds newly recorded demos, so the plan can grow across collection days.

Usage:
  python make_plan.py --task-id T1                  # fresh plan
  python make_plan.py --task-id T1 --append         # after recording more demos
Output: $CUREWM_DATA_ROOT/plan_<task>.json  (consumed by replay_perturbed.py --plan)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_ROOT, FAMILIES, MT_SEVERITIES, SEVERITIES  # noqa: E402
from perturb import SkipEpisode, wrong_target_ctx  # noqa: E402
from replay_perturbed import out_dir_for  # noqa: E402


def assign(demos: list[Path], rng, n_families: int, severities_per_family) -> list[dict]:
    entries, deck = [], []
    for demo in demos:
        if len(deck) < n_families:
            deck += list(rng.permutation(FAMILIES))
        fams = [deck.pop(0) for _ in range(n_families)]
        for fam in fams:
            if callable(severities_per_family):
                sevs = severities_per_family(rng)
            else:
                sevs = severities_per_family
            for sev in sevs:
                seed = int(rng.integers(0, 2**31 - 1))
                entries.append({
                    "demo": str(demo), "family": fam, "severity": float(sev),
                    "seed": seed, "out": str(out_dir_for(demo, fam, float(sev), seed)),
                })
    return entries


def assign_fixed(demos: list[Path], rng, family: str, sevs) -> list[dict]:
    """Every demo gets the same mechanical family — for the cube task that is
    insufficient_grip, the only family whose object never leaves its spot (so the
    scene needs no reset) and the exact path-preserving contrast to wrong_target."""
    entries = []
    for demo in demos:
        for sev in sevs:
            seed = int(rng.integers(0, 2**31 - 1))
            entries.append({"demo": str(demo), "family": family, "severity": float(sev),
                            "seed": seed,
                            "out": str(out_dir_for(demo, family, float(sev), seed))})
    return entries


def assign_wrong_target(demos: list[Path], rng, spots) -> list[dict]:
    """One entry per (demo, near/far distractor); ctx is frozen into the plan so
    the replay is fully specified and auditable."""
    entries = []
    for demo in demos:
        meta = json.load(open(demo / "meta.json"))
        for sev in MT_SEVERITIES:
            try:
                ctx = wrong_target_ctx(meta, spots, sev)
            except SkipEpisode as e:
                print(f"[plan] skip wrong_target for {demo.name}: {e}")
                break
            seed = int(rng.integers(0, 2**31 - 1))
            entries.append({
                "demo": str(demo), "family": "wrong_target", "severity": float(sev),
                "seed": seed, "ctx": ctx,
                "out": str(out_dir_for(demo, "wrong_target", float(sev), seed)),
            })
    return entries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--global-seed", type=int, default=2027)
    p.add_argument("--append", action="store_true")
    p.add_argument("--profile", choices=["default", "cube", "graded"], default="default",
                   help="cube: one mechanical family at s=1.0 plus a near and a far wrong_target "
                        "distractor per demonstration.  graded: a low-severity sweep that gives each failure a surviving partner from the same family")
    p.add_argument("--families", default="insufficient_grip,premature_release",
                   help="families for the graded profile, comma separated; only families whose severity is continuous")
    p.add_argument("--train-severities", default="0.2,0.4,0.6")
    p.add_argument("--val-severities", default="0.4,0.6")
    args = p.parse_args()

    plan_path = DATA_ROOT / f"plan_{args.task_id}.json"
    old_entries, known_demos = [], set()
    if args.append and plan_path.exists():
        with open(plan_path) as f:
            old_entries = json.load(f)["entries"]
        known_demos = {e["demo"] for e in old_entries}

    train_demos = sorted((DATA_ROOT / "demos" / args.task_id).glob("demo_*"))
    val_demos = sorted((DATA_ROOT / "val" / args.task_id).glob("demo_*"))
    new_train = [d for d in train_demos if str(d) not in known_demos]
    new_val = [d for d in val_demos if str(d) not in known_demos]
    print(f"[plan] train demos: {len(train_demos)} ({len(new_train)} new); "
          f"val demos: {len(val_demos)} ({len(new_val)} new)")

    rng = np.random.default_rng(args.global_seed + len(old_entries))
    if args.profile == "cube":
        spots_file = DATA_ROOT / f"spots_{args.task_id}.json"
        spots = json.load(open(spots_file))["spots"]
        entries = old_entries \
            + assign_fixed(new_train + new_val, rng, "insufficient_grip", [1.0]) \
            + assign_wrong_target(new_train + new_val, rng, spots)
    elif args.profile == "graded":
        # Every demonstration is replayed by every listed family across the severity sweep, so a
        # family's failures and its surviving replays come from the same demonstration: the
        # same-family graded partners the hardware pool has never had.
        fams = [f.strip() for f in args.families.split(",") if f.strip()]
        tr_sevs = [float(x) for x in args.train_severities.split(",")]
        va_sevs = [float(x) for x in args.val_severities.split(",")]
        entries = old_entries
        for fam in fams:
            entries = entries + assign_fixed(new_train, rng, fam, tr_sevs) \
                              + assign_fixed(new_val, rng, fam, va_sevs)
    else:
        two_sevs = lambda r: r.choice(SEVERITIES, size=2, replace=False)  # noqa: E731
        entries = old_entries \
            + assign(new_train, rng, n_families=2, severities_per_family=two_sevs) \
            + assign(new_val, rng, n_families=2, severities_per_family=[1.0])

    counts = {}
    for e in entries:
        key = (e["family"], e["severity"])
        counts[key] = counts.get(key, 0) + 1
    print("[plan] family x severity counts:")
    for (fam, sev), n in sorted(counts.items()):
        print(f"    {fam:<20} s={sev:<4} n={n}")
    print(f"[plan] total replays: {len(entries)}")

    with open(plan_path, "w") as f:
        json.dump({"task_id": args.task_id, "global_seed": args.global_seed,
                   "entries": entries}, f, indent=2)
    print(f"[plan] wrote {plan_path}")


if __name__ == "__main__":
    main()
