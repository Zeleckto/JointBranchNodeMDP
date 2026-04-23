"""
results_generator.py — Generate analysis tables and figures from evaluation results.

Usage:
  python results_generator.py --problem setcover
  python results_generator.py --all
"""

import os
import json
import argparse
import numpy as np

import config as cfg
from utils.metrics import shifted_geometric_mean, win_rate, wilcoxon_p_value

POLICY_LABELS = {
    "scip_default":  "SCIP default",
    "pi1_only":      "π1 (Branch Ranking)",
    "pi1_scip_uct":  "π1 + SCIP UCT",
    "pi1_pi2":       "π1 + π2 (proposed)",
}


def load_results(problem, difficulty):
    path = os.path.join(cfg.RESULTS_DIR, f"{problem}_{difficulty}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_table(problem):
    print(f"\n{'═'*72}")
    print(f"  {problem.upper()}")
    print(f"{'═'*72}")

    header = f"{'Policy':<28}"
    for diff in ["easy", "medium", "hard"]:
        header += f"  {diff:>12}"
    print(header)
    print(f"{'─'*72}")

    # Collect SGM times
    for policy_key, label in POLICY_LABELS.items():
        row = f"{label:<28}"
        for diff in ["easy", "medium", "hard"]:
            data = load_results(problem, diff)
            if data and policy_key in data["results"]:
                times = data["results"][policy_key]["times"]
                sgm   = shifted_geometric_mean(times)
                row  += f"  {sgm:>12.2f}"
            else:
                row  += f"  {'—':>12}"
        print(row)

    print(f"{'─'*72}")

    # GO/NO GO per difficulty
    for diff in ["easy", "medium", "hard"]:
        data = load_results(problem, diff)
        if data:
            wr  = data.get("win_rate", float('nan'))
            pv  = data.get("p_value",  float('nan'))
            go  = data.get("go",       False)
            print(f"  {diff:<8}  win_rate={wr:.2%}  p={pv:.4f}  "
                  f"{'GO ✓' if go else 'NO GO ✗'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", default=None)
    p.add_argument("--all",     action="store_true")
    args = p.parse_args()

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    problems = cfg.PROBLEM_TYPES if args.all else [args.problem or cfg.PROBLEM_TYPE]

    for problem in problems:
        print_table(problem)

    print(f"\nResults files in {cfg.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
