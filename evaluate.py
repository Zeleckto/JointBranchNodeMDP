"""
evaluate.py — Four-way comparison evaluation.

Policy 1: SCIP default (relpcost + hybridestim)
Policy 2: π1 only (GCN + hybridestim)          ← Branch Ranking
Policy 3: π1 + SCIP UCT                         ← ablation
Policy 4: π1 + π2 (BCNodeSelector)             ← proposed

Usage:
  python evaluate.py --problem setcover --difficulty easy --n-instances 20 --n-seeds 5
"""

import os
import json
import argparse
import numpy as np
import torch

import config as cfg
from data.instance_generator import generate_instances
from models.gcn import build_gcn
from models.node_mlp import NodeChildMLP
from branching.branch_rule import LearnedBranchRule
from node_selection.node_selector import BCNodeSelector
from utils.metrics import solve_instance, shifted_geometric_mean, win_rate, wilcoxon_p_value


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--problem",      default=cfg.PROBLEM_TYPE, choices=cfg.PROBLEM_TYPES)
    p.add_argument("--difficulty",   default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--n-instances",  type=int, default=20)
    p.add_argument("--n-seeds",      type=int, default=5)
    p.add_argument("--time-limit",   type=float, default=3600.0)
    p.add_argument("--device",       default=cfg.DEVICE)
    p.add_argument("--no-pi2",       action="store_true", help="Ablation: skip π2")
    return p.parse_args()


def load_gcn(problem, device):
    path = os.path.join(cfg.CHECKPOINT_DIR, f"{problem}_gcn_best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"GCN checkpoint not found: {path}")
    gcn  = build_gcn(cfg).to(device)
    ckpt = torch.load(path, map_location=device)
    gcn.load_state_dict(ckpt['model_state'])
    gcn.eval()
    print(f"  Loaded π1  (val_loss={ckpt.get('val_loss',float('nan')):.4f}, "
          f"epoch={ckpt.get('epoch','?')})")
    return gcn


def load_pi2(problem, device):
    path      = os.path.join(cfg.CHECKPOINT_DIR, f"{problem}_pi2.pt")
    norm_path = path.replace('.pt', '_norm.npy')

    if not os.path.exists(path):
        print(f"  π2 checkpoint not found: {path}. Running without π2.")
        return None, None

    pi2  = NodeChildMLP()
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        pi2.load_state_dict(ckpt['model_state'])
    else:
        pi2.load_state_dict(ckpt)
    pi2.eval()

    norm = np.load(norm_path) if os.path.exists(norm_path) else None
    print(f"  Loaded π2  (norm_stats={'found' if norm is not None else 'MISSING'})")
    return pi2, norm


def run_policy(instance_paths, policy_name, branch_rule, node_selector,
               n_seeds, time_limit, device):
    """Run a policy over all instances × seeds, return list of solve_time."""
    times  = []
    nodes  = []

    for path in instance_paths:
        inst_times = []
        inst_nodes = []
        for seed in range(n_seeds):
            result = solve_instance(
                path,
                branch_rule   = branch_rule,
                node_selector = node_selector,
                time_limit    = time_limit,
                paper_settings= True,
                seed          = seed,
            )
            inst_times.append(result["solve_time"])
            inst_nodes.append(result["n_nodes"])

        # Median over seeds
        t = float(np.nanmedian(inst_times)) if any(np.isfinite(inst_times)) else float('nan')
        n = int(np.nanmedian(inst_nodes))
        times.append(t)
        nodes.append(n)

    return times, nodes


def main():
    args   = parse_args()
    pt     = args.problem
    diff   = args.difficulty
    device = args.device

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # ── Load test instances ────────────────────────────────────────────────────
    inst_dir = os.path.join(cfg.INSTANCE_DIR, pt, diff)
    if not os.path.exists(inst_dir) or len(os.listdir(inst_dir)) == 0:
        print(f"Generating {diff} test instances for {pt}...")
        generate_instances(pt, diff, args.n_instances, inst_dir, seed=9999)

    all_paths = sorted([os.path.join(inst_dir, f)
                        for f in os.listdir(inst_dir) if f.endswith('.lp')])
    paths = all_paths[:args.n_instances]

    if not paths:
        print(f"ERROR: no .lp files found in {inst_dir}")
        return

    print(f"\nEvaluating {pt} {diff}: {len(paths)} instances × {args.n_seeds} seeds")

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading models...")
    gcn     = load_gcn(pt, device)
    pi2, pi2_norm = load_pi2(pt, device) if not args.no_pi2 else (None, None)

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Policy 1 — SCIP default
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/4] SCIP default...")
    t1, n1 = run_policy(paths, "scip_default", None, None,
                        args.n_seeds, args.time_limit, device)
    results["scip_default"] = {"times": t1, "nodes": n1}
    print(f"      SGM time = {shifted_geometric_mean(t1):.2f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # Policy 2 — π1 only (Branch Ranking baseline)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/4] π1 only (GCN + hybridestim)...")
    br2 = LearnedBranchRule(gcn, pi2=None, node_selector=None, device=device)
    t2, n2 = run_policy(paths, "pi1_only", br2, None,
                        args.n_seeds, args.time_limit, device)
    results["pi1_only"] = {"times": t2, "nodes": n2}
    print(f"      SGM time = {shifted_geometric_mean(t2):.2f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # Policy 3 — π1 + SCIP UCT (ablation)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/4] π1 + SCIP UCT (ablation)...")
    # Use SCIP's built-in UCT node selector by enabling it via param
    # We use a dummy branch rule with no π2, let SCIP handle nodes via params
    t3, n3 = [], []
    for path in paths:
        inst_times = []
        inst_nodes = []
        for seed in range(args.n_seeds):
            from pyscipopt import Model
            import time as _time
            m = Model()
            m.hideOutput(True)
            m.setParam("limits/time", args.time_limit)
            m.setParam("separating/maxroundsroot", -1)
            m.setParam("separating/maxrounds", 0)
            m.setParam("presolving/maxrestarts", 0)
            m.setParam("randomization/randomseedshift", seed)
            m.setParam("nodeselection/uct/stdpriority", 1000000)
            m.readProblem(path)

            br3 = LearnedBranchRule(gcn, pi2=None, node_selector=None, device=device)
            m.includeBranchrule(br3, "learned", "π1 branch rule",
                                priority=10_000_000, maxdepth=-1, maxbounddist=1.0)

            t0 = _time.time()
            m.optimize()
            elapsed = _time.time() - t0

            status = m.getStatus()
            inst_times.append(elapsed if status in ('optimal', 'timelimit') else float('nan'))
            inst_nodes.append(m.getNNodes())

        t3.append(float(np.nanmedian(inst_times)))
        n3.append(int(np.nanmedian(inst_nodes)))
    results["pi1_scip_uct"] = {"times": t3, "nodes": n3}
    print(f"      SGM time = {shifted_geometric_mean(t3):.2f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # Policy 4 — π1 + π2 (proposed)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4/4] π1 + π2 BCNodeSelector (proposed)...")
    node_sel4 = BCNodeSelector()
    br4       = LearnedBranchRule(gcn, pi2=pi2, pi2_norm=pi2_norm,
                                   node_selector=node_sel4, device=device)

    t4_list, n4_list = [], []
    for path in paths:
        inst_times = []
        inst_nodes = []
        for seed in range(args.n_seeds):
            from pyscipopt import Model
            import time as _time
            m = Model()
            m.hideOutput(True)
            m.setParam("limits/time", args.time_limit)
            m.setParam("separating/maxroundsroot", -1)
            m.setParam("separating/maxrounds", 0)
            m.setParam("presolving/maxrestarts", 0)
            m.setParam("randomization/randomseedshift", seed)
            m.readProblem(path)

            # Fresh selector per solve (shared state must be reset)
            ns = BCNodeSelector()
            br = LearnedBranchRule(gcn, pi2=pi2, pi2_norm=pi2_norm,
                                    node_selector=ns, device=device)
            m.includeBranchrule(br, "learned", "π1 branch rule",
                                priority=10_000_000, maxdepth=-1, maxbounddist=1.0)
            m.includeNodesel(ns, "bc_nodesel", "π2 node selector",
                             stdpriority=1_000_000, memsavepriority=500_000)

            t0 = _time.time()
            m.optimize()
            elapsed = _time.time() - t0

            status = m.getStatus()
            inst_times.append(elapsed if status in ('optimal', 'timelimit') else float('nan'))
            inst_nodes.append(m.getNNodes())

        t4_list.append(float(np.nanmedian(inst_times)))
        n4_list.append(int(np.nanmedian(inst_nodes)))

    results["pi1_pi2"] = {"times": t4_list, "nodes": n4_list}
    print(f"      SGM time = {shifted_geometric_mean(t4_list):.2f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"RESULTS: {pt.upper()} {diff.upper()} | {len(paths)} instances")
    print(f"{'─'*60}")
    print(f"{'Policy':<30} {'SGM time':>12} {'SGM nodes':>12}")
    print(f"{'─'*60}")
    for name, label in [
        ("scip_default",  "1. SCIP default"),
        ("pi1_only",      "2. π1 (Branch Ranking)"),
        ("pi1_scip_uct",  "3. π1 + SCIP UCT"),
        ("pi1_pi2",       "4. π1 + π2 (proposed)"),
    ]:
        d = results[name]
        sgm_t = shifted_geometric_mean(d["times"])
        sgm_n = shifted_geometric_mean(d["nodes"])
        print(f"{label:<30} {sgm_t:>12.2f} {sgm_n:>12.0f}")
    print(f"{'─'*60}")

    # ── GO/NO GO ──────────────────────────────────────────────────────────────
    wr  = win_rate(results["pi1_pi2"]["times"], results["pi1_only"]["times"])
    pv  = wilcoxon_p_value(results["pi1_pi2"]["times"], results["pi1_only"]["times"])
    print(f"\nGO/NO GO (Policy 4 vs Policy 2):")
    print(f"  Win rate : {wr:.2%}  (threshold ≥ 60%)")
    print(f"  p-value  : {pv:.4f}  (threshold < 0.05)")
    go = wr >= 0.60 and pv < 0.05
    print(f"  Decision : {'✓ GO' if go else '✗ NO GO'}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(cfg.RESULTS_DIR, f"{pt}_{diff}_results.json")
    with open(out_path, 'w') as f:
        json.dump({
            "problem": pt, "difficulty": diff,
            "n_instances": len(paths), "n_seeds": args.n_seeds,
            "results": {k: {"times": v["times"], "nodes": v["nodes"]}
                        for k, v in results.items()},
            "win_rate": wr, "p_value": pv, "go": go,
        }, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
