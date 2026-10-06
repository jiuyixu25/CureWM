"""Entry point for the failure engine in LIBERO: official demonstrations -> six
perturbation families -> sanity report.  Demonstrations are processed grouped by task,
because the backend swaps environments lazily."""
import argparse
import json
from pathlib import Path

from curewm.perturbations import generate_pairs, sanity_report
from curewm.backends.libero import LiberoBackend, load_libero_demos

p = argparse.ArgumentParser()
p.add_argument("--suite", default="libero_goal")
p.add_argument("--demos-per-task", type=int, default=2)
p.add_argument("--max-tasks", type=int, default=5)
p.add_argument("--out", default="data/libero_failure_v0")
p.add_argument("--seeds-per-cell", type=int, default=1)
p.add_argument("--severities", type=float, nargs="+", default=None,
               help="override the severity grid, e.g. --severities 0.6 1.0 for a fast probe batch")
p.add_argument("--tasks", default=None, help="task shard, e.g. 0-4 or 5,7,9; takes precedence over --max-tasks")
p.add_argument("--no-render", action="store_true", help="training-format mode: skip rendering, an order of magnitude faster")
args = p.parse_args()

task_ids = None
if args.tasks:
    task_ids = (list(range(int(args.tasks.split("-")[0]), int(args.tasks.split("-")[1]) + 1))
                if "-" in args.tasks else [int(x) for x in args.tasks.split(",")])
demos = load_libero_demos(args.suite, args.demos_per_task, args.max_tasks, task_ids=task_ids)
backend = LiberoBackend(args.suite, render=not args.no_render)
kw = {"severity_grid": tuple(args.severities)} if args.severities else {}
try:
    index = generate_pairs(demos, backend, Path(args.out), seeds_per_cell=args.seeds_per_cell, **kw)
finally:
    backend.close()
report = sanity_report(index)
print(json.dumps(report, indent=2, ensure_ascii=False))
Path(args.out, "sanity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
