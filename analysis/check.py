"""One command, one answer: do the shipped scripts still produce the paper's numbers?

Runs the three analysis scripts and compares the values the paper reports against what
they print now.  Exit status is 0 only if every row passes.  Standard library only.

    python3 check.py
"""
import re, subprocess, sys

# (script, label, regex capturing the value, value in the paper, tolerance)
CHECKS = [
    ("cw_paired_ci.py", "Ctrl-World full pool, margin",
     r"full pool.*control-CureWM\s+([-+][\d.]+)", 0.0533, 5e-4),
    ("cw_paired_ci.py", "Ctrl-World full pool, CI low",
     r"full pool.*95% CI \[([-+][\d.]+)", 0.0284, 5e-4),
    ("cw_paired_ci.py", "Ctrl-World full pool, CI high",
     r"full pool.*95% CI \[[-+][\d.]+, ([-+][\d.]+)\]", 0.0758, 5e-4),
    ("cw_paired_ci.py", "Ctrl-World held-out, margin",
     r"held-out subset.*control-CureWM\s+([-+][\d.]+)", 0.0657, 5e-4),
    ("cw_paired_ci.py", "Ctrl-World held-out, CI low",
     r"held-out subset.*95% CI \[([-+][\d.]+)", 0.0207, 5e-4),
    ("cw_paired_ci.py", "Ctrl-World held-out, CI high",
     r"held-out subset.*95% CI \[[-+][\d.]+, ([-+][\d.]+)\]", 0.1020, 5e-4),

    ("sevauroc_ci.py", "Stratified macro-AUROC, control",
     r"macro AUROC\s+control\s+([\d.]+)", 0.5368, 5e-4),
    ("sevauroc_ci.py", "Stratified macro-AUROC, CureWM",
     r"CureWM\s+([\d.]+)\s+difference", 0.7144, 5e-4),
    ("sevauroc_ci.py", "Stratified macro-AUROC, difference",
     r"difference\s+([-+][\d.]+)", 0.1777, 5e-4),
    ("sevauroc_ci.py", "Stratified macro-AUROC, CI low",
     r"95% percentile CI \[([-+][\d.]+)", 0.1100, 5e-4),
    ("sevauroc_ci.py", "Stratified macro-AUROC, CI high",
     r"95% percentile CI \[[-+][\d.]+, ([-+][\d.]+)\]", 0.2428, 5e-4),

    ("grid44.py", "Goal optimism, released",
     r"Goal\s+released\s+\d+\s+\d+\s+([\d.]+)", 79.13, 5e-3),
    ("grid44.py", "Goal optimism, control",
     r"Goal\s+control\s+\d+\s+\d+\s+([\d.]+)", 79.55, 5e-3),
    ("grid44.py", "Goal optimism, CureWM",
     r"Goal\s+CureWM\s+\d+\s+\d+\s+([\d.]+)", 30.17, 5e-3),
    ("grid44.py", "Spatial optimism, released",
     r"Spatial\s+released\s+\d+\s+\d+\s+([\d.]+)", 65.67, 5e-3),
    ("grid44.py", "Object optimism, CureWM",
     r"Object\s+CureWM\s+\d+\s+\d+\s+([\d.]+)", 37.41, 5e-3),
    ("grid44.py", "LIBERO-10 optimism, CureWM",
     r"LIBERO-10\s+CureWM\s+\d+\s+\d+\s+([\d.]+)", 2.51, 5e-3),
    ("grid44.py", "Macro AUROC, released",
     r"AUROC\s+released=([\d.]+)", 0.489, 5e-4),
    ("grid44.py", "Macro AUROC, control",
     r"AUROC\s+released=[\d.]+ \(4 suites\)\s+control=([\d.]+)", 0.492, 5e-4),
    ("grid44.py", "Macro AUROC, CureWM",
     r"AUROC\s+released=.*CureWM=([\d.]+)", 0.613, 5e-4),
]

outs, fails = {}, 0
for script in dict.fromkeys(c[0] for c in CHECKS):
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"*FAIL* {script} exited {r.returncode}\n{r.stderr.strip()[:400]}")
        fails += 1
    outs[script] = r.stdout

width = max(len(c[1]) for c in CHECKS)
for script, label, pattern, expected, tol in CHECKS:
    m = None
    for line in outs.get(script, "").splitlines():
        m = re.search(pattern, line)
        if m:
            break
    if not m:
        print(f"*FAIL* {label:<{width}}  value not found in {script} output")
        fails += 1
        continue
    got = float(m.group(1))
    ok = abs(got - expected) <= tol
    fails += not ok
    print(f"{'PASS ' if ok else '*FAIL*'} {label:<{width}}  paper {expected:>8.4f}   recomputed {got:>8.4f}")

n = len(CHECKS)
print(f"\n{n - fails}/{n} checks passed." if not fails else
      f"\n{n - fails}/{n} checks passed, {fails} FAILED.")
sys.exit(1 if fails else 0)
