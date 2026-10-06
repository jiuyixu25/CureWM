"""Pinned family-severity macro AUROC contrast for the held-out survivor-inclusive probes.

Configuration is fixed in this file so the reported interval is reproducible by running it.
  inputs      probe_{ctrl,post}fa_{ho_A,ho_B,ho2_A,ho2_B}.jsonl  (survivor-inclusive probes)
  score       value_fail_action; a stratum's AUROC is P(V survivor > V failure) with ties at 1/2
  strata      (family, severity) pairs holding at least three failures and three survivors
              in the observed data; this set is fixed once and reused in every resample
  statistic   macro average over strata, CureWM minus control
  cluster     (shard, source demonstration), the key used by libero_arms_verdict.py
  bootstrap   resample clusters with replacement; a resample is excluded when any fixed
              stratum loses either outcome in it
  seed        0            fixed in advance, as in every other analysis here
  B           20000        attempted resamples
  interval    percentile, 2.5 and 97.5, over the retained resamples
"""
import json, re, random, collections, pathlib

D = pathlib.Path(__file__).parent
SEED, B, SHARDS = 0, 20000, ('ho_A', 'ho_B', 'ho2_A', 'ho2_B')
demo = lambda r: (r['_shard'], re.match(r'(d\d+)', r['name']).group(1))  # cluster key as in libero_arms_verdict.py

def load(tag):
    out = []
    for sh in SHARDS:
        for line in (D / f'probe_{tag}fa_{sh}.jsonl').read_text().splitlines():
            if line.strip():
                r = json.loads(line); r['_shard'] = sh; out.append(r)
    return out

def auroc(pos, neg):                       # pos = survivors, neg = failures
    if not pos or not neg: return None
    s = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return s / (len(pos) * len(neg))

def macro(records, strata):
    by = collections.defaultdict(lambda: ([], []))
    for r in records:
        by[(r['family'], r['severity'])][0 if r['outcome'] else 1].append(r['value_fail_action'])
    vals = []
    for st in strata:
        pos, neg = by[st]
        a = auroc(pos, neg)
        if a is None: return None
        vals.append(a)
    return sum(vals) / len(vals)

C, T = load('ctrl'), load('post')
key = lambda r: (r['_shard'], r['name'])
Cm, Tm = {key(r): r for r in C}, {key(r): r for r in T}
common = sorted(set(Cm) & set(Tm))
counts = collections.defaultdict(lambda: [0, 0])
for k in common:
    r = Cm[k]; counts[(r['family'], r['severity'])][0 if r['outcome'] else 1] += 1
strata = sorted(st for st, (ns, nf) in counts.items() if ns >= 3 and nf >= 3)
print(f'records per arm {len(common)}   fixed strata {len(strata)}')
mc, mt = macro([Cm[k] for k in common], strata), macro([Tm[k] for k in common], strata)
print(f'macro AUROC   control {mc:.4f}   CureWM {mt:.4f}   difference {mt-mc:+.4f}')

clusters = collections.defaultdict(list)
for k in common: clusters[demo(Cm[k])].append(k)
ks = list(clusters); rng = random.Random(SEED)
draws, dropped = [], 0
for _ in range(B):
    picked = []
    for _ in range(len(ks)): picked += clusters[ks[rng.randrange(len(ks))]]
    a, b = macro([Cm[k] for k in picked], strata), macro([Tm[k] for k in picked], strata)
    if a is None or b is None: dropped += 1; continue
    draws.append(b - a)
draws.sort(); n = len(draws)
print(f'clusters {len(ks)}   attempted {B}   retained {n}   excluded {dropped}')
print(f'95% percentile CI [{draws[int(.025*n)]:+.4f}, {draws[int(.975*n)-1]:+.4f}]')
print(f'rounded: {mt-mc:+.3f}  [{draws[int(.025*n)]:+.3f}, {draws[int(.975*n)-1]:+.3f}]')
