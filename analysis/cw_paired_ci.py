"""Pinned paired statistic for the Ctrl-World video probe.

Fixed configuration, so the reported intervals are reproducible by running this file:
  score      s(a-) = mse_gen_minus_to_failure - mse_gen_minus_to_success   (Eq. 1 at a-)
  contrast   control - CureWM, paired by evaluation pair
  cluster    source demonstration (the dXXXX prefix of the pair name)
  bootstrap  resample clusters, pool the drawn clusters' pair values, take the mean
  seed       0          (the seed used by every other analysis in this project)
  B          20000
Seed 0 is fixed in advance and not selected to match any previously reported value.
"""
import json, re, random, collections, pathlib
D = pathlib.Path(__file__).parent
SEED, B = 0, 20000
score = lambda r: r['mse_gen_minus_to_failure'] - r['mse_gen_minus_to_success']
demo  = lambda p: re.search(r'(d\d+)', p).group(1)

def load(name):
    return {json.loads(l)['pair']: json.loads(l) for l in (D / name).read_text().splitlines() if l.strip()}

def paired_ci(A, Bm, keys):
    g = collections.defaultdict(list)
    for p in keys:
        g[demo(p)].append(score(A[p]) - score(Bm[p]))
    ks = list(g); rng = random.Random(SEED)
    point = sum(x for k in ks for x in g[k]) / len(keys)
    draws = []
    for _ in range(B):
        vals = []
        for _ in range(len(ks)):
            vals += g[ks[rng.randrange(len(ks))]]
        draws.append(sum(vals) / len(vals))
    draws.sort()
    return point, draws[int(.025 * B)], draws[int(.975 * B) - 1], len(keys), len(ks)

if __name__ == '__main__':
    C, T = load('visual2_ctrl.jsonl'), load('visual2_treat.jsonl')
    full = [p for p in C if p in T]
    held = [p for p in full if C[p].get('held_out')]
    for keys, label in ((full, 'full pool'), (held, 'held-out subset')):
        pt, lo, hi, n, k = paired_ci(C, T, keys)
        print('%-18s n=%2d clusters=%2d   control-CureWM %+0.4f   95%% CI [%+0.4f, %+0.4f]'
              % (label, n, k, pt, lo, hi))
        print('%-18s rounded to three decimals: %+0.3f  [%+0.3f, %+0.3f]' % ('', pt, lo, hi))
