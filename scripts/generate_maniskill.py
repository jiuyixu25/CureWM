"""Run the failure data engine end to end: load demonstrations -> six perturbation
families x the severity grid -> sanity report."""
import argparse
import json
from pathlib import Path

from curewm.perturbations import generate_pairs, sanity_report
from curewm.backends.maniskill import ManiSkill3Backend, load_ms_demos

p = argparse.ArgumentParser()
p.add_argument("--h5", required=True, help="trajectory h5, already converted to pd_ee_delta_pose")
p.add_argument("--env-id", required=True)
p.add_argument("--out", default="data/failure_v0")
p.add_argument("--max-demos", type=int, default=10)
p.add_argument("--seeds-per-cell", type=int, default=1)
args = p.parse_args()

demos = load_ms_demos(args.h5, env_id=args.env_id, max_demos=args.max_demos)
backend = ManiSkill3Backend(args.env_id)
index = generate_pairs(demos, backend, Path(args.out), seeds_per_cell=args.seeds_per_cell)
report = sanity_report(index)
print(json.dumps(report, indent=2, ensure_ascii=False))
Path(args.out, "sanity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
