"""Live QC for the second held-out cup draw (09-18): counts and per-episode sanity for every
val demo / val replay whose name carries today's date stamp.  Read-only.

    conda run -n robot python draw2_qc.py [--date 0918]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CUREWM_DATA_ROOT", Path.home() / "curewm_data"))


def episodes(split, date):
    return sorted(p for p in (ROOT / split / "T1").glob(f"*demo_T1_{date}_*") if (p / "meta.json").exists())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--date", default="0918"); a = ap.parse_args()
    demos = episodes("val", a.date); reps = episodes("replays_val", a.date)
    print(f"== draw-2 val demos: {len(demos)}   replays: {len(reps)}")
    bad = []
    spots = collections.Counter()
    for d in demos:
        m = json.load(open(d / "meta.json")); t = np.load(d / "traj.npz")
        n = len(t["actions"]); spots[m.get("spot")] += 1
        flag = "" if (150 <= n <= 230 and m.get("outcome") == "success") else "  <-- CHECK"
        if flag: bad.append(d.name)
        print(f"  {d.name}  spot {m.get('spot')}  frames {n}  outcome {m.get('outcome')}{flag}")
    print(f"  spots used: {dict(sorted(spots.items()))}")
    fam = collections.Counter(); out = collections.Counter()
    for r in reps:
        m = json.load(open(r / "meta.json")); f = m["perturbation"]["family"]; o = m.get("outcome")
        fam[(f, o)] += 1; out[o] += 1
        flag = "" if (m.get("camera_drops", 0) == 0 and m.get("safety_violations", 0) == 0) else "  <-- drops/safety"
        if flag: bad.append(r.name)
        print(f"  {r.name}  {f:18s} s{m['perturbation']['severity']}  {o:7s}  frames {m.get('frames')}  "
              f"drops {m.get('camera_drops', 0)} safety {m.get('safety_violations', 0)}{flag}")
    print(f"  replay outcomes: {dict(out)}")
    for (f, o), n in sorted(fam.items()): print(f"    {f:18s} {o:8s} {n}")
    done_demos = {json.load(open(r / 'meta.json'))['source_demo'].split('/')[-1] for r in reps}
    print(f"  demos with >=1 replay: {len(done_demos)}/{len(demos)}   verified failures so far: {out.get('failure', 0)}")
    print(f"== flagged: {bad if bad else 'none'}")


if __name__ == "__main__":
    main()
