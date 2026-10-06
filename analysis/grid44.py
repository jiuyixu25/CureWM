"""The 4x4 LIBERO grid: 4 suites x {optimism, dSF, within-family AUROC, false-alarm}
x 3 arms, all from the fa probe files (which include surviving perturbed replays).

Cluster bootstrap over source demonstrations, 20,000 resamples, seed 0, the same
protocol as the pinned scripts.  Pure stdlib so it runs on the login node.
"""
import json, os, re, random

D = os.path.dirname(os.path.abspath(__file__))
SUITES = [("Goal", ["ho_A", "ho_B", "ho2_A", "ho2_B"], "pre"),
          ("Spatial", ["spatial_A", "spatial_B"], "base"),
          ("Object", ["object_A", "object_B"], "base"),
          ("LIBERO-10", ["10_A", "10_B"], "base")]
N_BOOT = 20000
DPAT = re.compile(r"(d\d+)")

def load(arm, shard):
    p = os.path.join(D, "probe_%sfa_%s.jsonl" % (arm, shard))
    if not os.path.exists(p):
        return None
    out = []
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            r["_shard"] = shard
            out.append(r)
    return out

def demo(r):
    m = DPAT.match(r["name"])
    return "%s/%s" % (r.get("_shard", ""), m.group(1) if m else r["name"])

def auroc(fails, survs):
    """P(V(a-) ranks a failure below a survivor). Ties 0.5.  Rank-based so this
    stays linearithmic, and the pools run to a few thousand items."""
    if not fails or not survs:
        return None
    allv = sorted([(v, 1) for v in survs] + [(v, 0) for v in fails])
    i = 0.0
    rsum = 0.0
    k = 0
    while k < len(allv):
        j = k
        while j + 1 < len(allv) and allv[j + 1][0] == allv[k][0]:
            j += 1
        avg = (k + j) / 2.0 + 1.0
        for t in range(k, j + 1):
            if allv[t][1] == 1:
                rsum += avg
        k = j + 1
    n1, n0 = len(survs), len(fails)
    return (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def metrics(rs):
    fail = [r for r in rs if r.get("outcome") is False]
    surv = [r for r in rs if r.get("outcome") is True]
    n, ns = len(fail), len(surv)
    if not n or not ns:
        # a probe still running has written only one outcome class so far
        return dict(partial=True, n=n, ns=ns)
    vf = [r["value_fail_action"] for r in fail]
    vn = [r["value_nominal_action"] for r in fail]
    d = dict(n=n, ns=ns,
             opt=100.0 * sum(1 for v in vf if v > 0.5) / n,
             dsf=sum(vn) / n - sum(vf) / n,
             mvn=sum(vn) / n, mvf=sum(vf) / n)
    if ns:
        d["auroc"] = auroc(vf, [r["value_fail_action"] for r in surv])
        d["fa"] = 100.0 * sum(1 for r in surv if r["value_fail_action"] <= 0.5) / ns
    return d

def paired_ci(a_rs, b_rs, fn):
    """fn maps a record -> scalar. Cluster = (shard, source demo)."""
    A = {(r["_shard"], r["name"]): r for r in a_rs}
    B = {(r["_shard"], r["name"]): r for r in b_rs}
    keys = sorted(set(A) & set(B))
    if not keys:
        return None
    g = {}
    for k in keys:
        g.setdefault(demo(A[k]), []).append(fn(A[k]) - fn(B[k]))
    ks = list(g)
    if len(ks) < 5:
        return None
    flat = [x for k in ks for x in g[k]]
    d0 = sum(flat) / len(flat)
    rng = random.Random(0)
    bs = []
    for _ in range(N_BOOT):
        s = []
        for _ in ks:
            s.extend(g[ks[rng.randrange(len(ks))]])
        bs.append(sum(s) / len(s))
    bs.sort()
    return d0, bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT) - 1], len(ks)

print("%-11s %-9s %6s %6s %8s %8s %8s %8s %8s" %
      ("suite", "arm", "nFail", "nSurv", "optim%", "dSF", "AUROC", "falseAl%", "meanV-"))
summary = {}
for name, shards, relarm in SUITES:
    pooled = {}
    for arm, tag in [("released", relarm), ("control", "ctrl"), ("CureWM", "post")]:
        rs = []
        miss = False
        for sh in shards:
            got = load(tag, sh)
            if got is None:
                miss = True
            else:
                rs.extend(got)
        if not rs:
            print("%-11s %-9s  (no fa probe yet%s)" % (name, arm, ", partial" if miss else ""))
            continue
        pooled[arm] = rs
        m = metrics(rs)
        if not m:
            continue
        if m.get("partial"):
            print("%-11s %-9s  (probe still running: %d fail / %d surv written)"
                  % (name, arm, m["n"], m["ns"]))
            continue
        print("%-11s %-9s %6d %6d %8.2f %+8.4f %8s %8s %8.3f" %
              (name, arm, m["n"], m["ns"], m["opt"], m["dsf"],
               ("%.3f" % m["auroc"]) if m.get("auroc") is not None else "-",
               ("%.1f" % m["fa"]) if m.get("fa") is not None else "-", m["mvf"]))
        summary.setdefault(name, {})[arm] = m
    if "CureWM" in pooled and "control" in pooled:
        for lab, fn in [("optimism pts", lambda r: 100.0 * (r["value_fail_action"] > 0.5)),
                        ("dSF", lambda r: r["value_nominal_action"] - r["value_fail_action"])]:
            fa = [r for r in pooled["CureWM"] if r.get("outcome") is False]
            fb = [r for r in pooled["control"] if r.get("outcome") is False]
            res = paired_ci(fa, fb, fn)
            if res:
                d, lo, hi, k = res
                star = "*" if (lo > 0 or hi < 0) else " "
                print("    CureWM-control %-13s %+8.3f  CI[%+.3f,%+.3f]%s (%d demos)" %
                      (lab, d, lo, hi, star, k))
    print()

print("=== macro-average over the suites that have each metric ===")
for key, lab in [("opt", "Value optimism %"), ("dsf", "dSF"),
                 ("auroc", "AUROC"), ("fa", "False-alarm %")]:
    row = []
    for arm in ["released", "control", "CureWM"]:
        vals = [summary[s][arm][key] for s in summary
                if arm in summary[s] and summary[s][arm].get(key) is not None]
        row.append("%s (%d suites)" % (("%.3f" % (sum(vals)/len(vals))) if vals else "-", len(vals)))
    print("  %-18s released=%-18s control=%-18s CureWM=%s" % (lab, row[0], row[1], row[2]))
