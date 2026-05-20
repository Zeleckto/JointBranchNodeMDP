"""
problem_classifier.py

CLI tool that:
  1. Takes a .lp file as argument
  2. Classifies it into one of 4 benchmark types (setcover, auction, facility, indset)
  3. Shows classification confidence and structural evidence
  4. Offers to load the corresponding trained π1 + π2 policy
  5. Optionally runs a quick solve with the loaded policy

Usage:
    python problem_classifier.py path/to/problem.lp
    python problem_classifier.py path/to/problem.lp --solve
    python problem_classifier.py path/to/problem.lp --checkpoint-dir checkpoints/
"""

import os
import re
import sys
import argparse
import math
from collections import defaultdict

# ── Optional rich for pretty output ───────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich import box
    from rich.text import Text
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None


# ══════════════════════════════════════════════════════════════════════════════
# LP FILE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_lp_file(path):
    """
    Parse a .lp format file and extract structural statistics.
    Returns a dict of features used for classification.
    """
    with open(path, 'r') as f:
        content = f.read()

    lines = [l.strip() for l in content.split('\n')]

    obj_sense      = None   # 'minimize' or 'maximize'
    n_constraints  = 0
    rhs_values     = []
    constraint_ops = []     # list of '>=' or '<='
    vars_per_con   = []     # how many vars in each constraint
    n_binary       = 0
    n_continuous   = 0
    n_integer      = 0
    obj_terms      = 0

    section        = None

    for line in lines:
        ll = line.lower().strip()
        if not ll or ll.startswith('\\'):
            continue

        # Detect section headers
        if ll in ('minimize', 'min', 'minimize'):
            obj_sense = 'minimize'; section = 'obj'; continue
        if ll in ('maximize', 'max', 'maximize'):
            obj_sense = 'maximize'; section = 'obj'; continue
        if ll.startswith('subject to') or ll.startswith('s.t.') or ll == 'st':
            section = 'constraints'; continue
        if ll.startswith('bound'):
            section = 'bounds'; continue
        if ll.startswith('general') or ll.startswith('generals'):
            section = 'generals'; continue
        if ll.startswith('binary') or ll.startswith('binaries'):
            section = 'binary'; continue
        if ll == 'end':
            break

        if section == 'obj':
            terms = re.findall(r'[+-]?\s*[\d.e+-]*\s*\w+', line)
            obj_terms += len(terms)

        elif section == 'constraints':
            # Count constraint and parse RHS + operator
            if '>=' in line or '<=' in line or '=' in line:
                n_constraints += 1
                # operator
                if '>=' in line:
                    constraint_ops.append('>=')
                elif '<=' in line:
                    constraint_ops.append('<=')
                else:
                    constraint_ops.append('=')
                # RHS value
                rhs_match = re.search(r'[<>]?=\s*([-\d.e+]+)\s*$', line)
                if rhs_match:
                    try:
                        rhs_values.append(float(rhs_match.group(1)))
                    except Exception:
                        pass
                # Count variables (words that look like variable names after colon)
                lhs = line.split(':')[-1] if ':' in line else line
                lhs = re.split(r'[<>]?=', lhs)[0]
                var_count = len(re.findall(r'\b[a-zA-Z_]\w*\b', lhs))
                vars_per_con.append(var_count)

        elif section == 'binary':
            vars_in_line = re.findall(r'\b\w+\b', line)
            n_binary += len(vars_in_line)

        elif section == 'generals':
            vars_in_line = re.findall(r'\b\w+\b', line)
            n_integer += len(vars_in_line)

        elif section == 'bounds':
            # Continuous variables appear in bounds section without binary/general
            pass

    return {
        'obj_sense':       obj_sense,
        'n_constraints':   n_constraints,
        'n_binary':        n_binary,
        'n_continuous':    n_continuous,
        'n_integer':       n_integer,
        'rhs_values':      rhs_values,
        'constraint_ops':  constraint_ops,
        'vars_per_con':    vars_per_con,
        'obj_terms':       obj_terms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify(features):
    """
    Rule-based classifier. Returns list of (problem_type, score, evidence_dict)
    sorted by score descending.
    """
    ops    = features['constraint_ops']
    rhs    = features['rhs_values']
    vpc    = features['vars_per_con']
    sense  = features['obj_sense']
    n_bin  = features['n_binary']
    n_con  = features['n_continuous']
    n_int  = features['n_integer']
    n_cns  = features['n_constraints']

    def frac(lst, cond): return sum(1 for x in lst if cond(x)) / max(len(lst), 1)

    # Derived fractions
    frac_ge         = frac(ops, lambda x: x == '>=')
    frac_le         = frac(ops, lambda x: x == '<=')
    frac_rhs_one    = frac(rhs, lambda x: abs(x - 1.0) < 0.01)
    frac_rhs_large  = frac(rhs, lambda x: abs(x) > 1.5)
    frac_two_vars   = frac(vpc, lambda x: x == 2)
    frac_few_vars   = frac(vpc, lambda x: x <= 4)
    frac_many_vars  = frac(vpc, lambda x: x > 4)
    has_continuous  = n_con > 0 or n_int > 0
    is_minimize     = sense == 'minimize'
    is_maximize     = sense == 'maximize'

    scores    = {}
    evidence  = {}

    # ── Set Covering ──────────────────────────────────────────────────────────
    sc = 0
    ev = {}
    if is_minimize:             sc += 30; ev['Minimize'] = '✓'
    if frac_ge > 0.85:          sc += 25; ev[f'>= constraints ({frac_ge:.0%})'] = '✓'
    if frac_rhs_one > 0.85:     sc += 25; ev[f'RHS=1 ({frac_rhs_one:.0%})'] = '✓'
    if not has_continuous:      sc += 10; ev['Binary only'] = '✓'
    if frac_many_vars > 0.5:    sc += 10; ev[f'Many vars/constraint ({frac_many_vars:.0%})'] = '✓'
    scores['setcover'] = sc; evidence['setcover'] = ev

    # ── Combinatorial Auction ─────────────────────────────────────────────────
    sc = 0
    ev = {}
    if is_maximize:             sc += 30; ev['Maximize'] = '✓'
    if frac_le > 0.85:          sc += 25; ev[f'<= constraints ({frac_le:.0%})'] = '✓'
    if frac_rhs_one > 0.80:     sc += 25; ev[f'RHS=1 ({frac_rhs_one:.0%})'] = '✓'
    if not has_continuous:      sc += 10; ev['Binary only'] = '✓'
    if frac_few_vars > 0.4:     sc += 10; ev[f'Few vars/constraint ({frac_few_vars:.0%})'] = '✓'
    scores['auction'] = sc; evidence['auction'] = ev

    # ── Capacitated Facility Location ─────────────────────────────────────────
    sc = 0
    ev = {}
    if is_minimize:             sc += 25; ev['Minimize'] = '✓'
    if has_continuous:          sc += 30; ev['Binary + continuous vars'] = '✓'
    if frac_rhs_large > 0.2:    sc += 25; ev[f'Large RHS values ({frac_rhs_large:.0%})'] = '✓'
    if frac_le > 0.5:           sc += 10; ev[f'<= constraints ({frac_le:.0%})'] = '✓'
    if n_cns > 0:               sc += 10; ev[f'{n_cns} constraints'] = '✓'
    scores['facility'] = sc; evidence['facility'] = ev

    # ── Maximum Independent Set ───────────────────────────────────────────────
    sc = 0
    ev = {}
    if is_maximize:             sc += 30; ev['Maximize'] = '✓'
    if frac_two_vars > 0.70:    sc += 35; ev[f'2-var edge constraints ({frac_two_vars:.0%})'] = '✓'
    if frac_rhs_one > 0.90:     sc += 20; ev[f'RHS=1 ({frac_rhs_one:.0%})'] = '✓'
    if not has_continuous:      sc += 10; ev['Binary only'] = '✓'
    if frac_le > 0.85:          sc += 5;  ev[f'<= constraints ({frac_le:.0%})'] = '✓'
    scores['indset'] = sc; evidence['indset'] = ev

    # Normalise to 0-100
    total = sum(scores.values())
    if total > 0:
        scores = {k: round(100 * v / total) for k, v in scores.items()}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked, evidence


# ══════════════════════════════════════════════════════════════════════════════
# POLICY LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_policy(problem_type, checkpoint_dir, device):
    """Load π1 GCN and π2 NodeChildMLP for the given problem type."""
    import torch
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    from models.gcn import build_gcn
    from models.node_mlp import NodeChildMLP

    gcn_path  = os.path.join(checkpoint_dir, f"{problem_type}_gcn_best.pt")
    pi2_path  = os.path.join(checkpoint_dir, f"{problem_type}_pi2.pt")
    norm_path = os.path.join(checkpoint_dir, f"{problem_type}_pi2_norm.npy")

    result = {'gcn': None, 'pi2': None, 'norm': None, 'errors': []}

    # Load GCN
    if os.path.exists(gcn_path):
        try:
            gcn  = build_gcn(cfg).to(device)
            ckpt = torch.load(gcn_path, map_location=device)
            gcn.load_state_dict(ckpt['model_state'])
            gcn.eval()
            result['gcn']       = gcn
            result['gcn_info']  = {'val_loss': ckpt.get('val_loss', '?'), 'epoch': ckpt.get('epoch', '?')}
        except Exception as e:
            result['errors'].append(f"GCN load failed: {e}")
    else:
        result['errors'].append(f"GCN checkpoint not found: {gcn_path}")

    # Load π2
    if os.path.exists(pi2_path):
        try:
            pi2  = NodeChildMLP()
            ckpt = torch.load(pi2_path, map_location=device)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                pi2.load_state_dict(ckpt['model_state'])
            else:
                pi2.load_state_dict(ckpt)
            pi2.eval()
            result['pi2'] = pi2
        except Exception as e:
            result['errors'].append(f"π2 load failed: {e}")
    else:
        result['errors'].append(f"π2 checkpoint not found: {pi2_path}")

    if os.path.exists(norm_path):
        try:
            result['norm'] = np.load(norm_path)
        except Exception as e:
            result['errors'].append(f"Norm stats load failed: {e}")
    else:
        result['errors'].append(f"π2 norm not found: {norm_path}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SOLVE WITH POLICY
# ══════════════════════════════════════════════════════════════════════════════

def solve_with_policy(lp_path, gcn, pi2, norm, device, time_limit=60):
    """Run a SCIP solve with loaded policy on the .lp file."""
    import time
    from pyscipopt import Model
    from branching.branch_rule import LearnedBranchRule
    from node_selection.node_selector import BCNodeSelector

    m = Model()
    m.hideOutput(False)
    m.setParam("limits/time", time_limit)
    m.readProblem(lp_path)

    ns = BCNodeSelector()
    br = LearnedBranchRule(gcn, pi2=pi2, pi2_norm=norm, node_selector=ns, device=device)

    m.includeBranchrule(br, "learned", "π1+π2", priority=10_000_000, maxdepth=-1, maxbounddist=1.0)
    m.includeNodesel(ns, "bc_nodesel", "BCNodeSelector", stdpriority=1_000_000, memsavepriority=500_000)

    t0 = time.time()
    m.optimize()
    elapsed = time.time() - t0

    return {
        'status':     m.getStatus(),
        'n_nodes':    m.getNNodes(),
        'solve_time': elapsed,
        'obj_val':    m.getObjVal() if m.getStatus() in ('optimal', 'bestsol') else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

LABELS = {'setcover': 'Set Covering', 'auction': 'Combinatorial Auction',
          'facility': 'Facility Location', 'indset': 'Max Independent Set'}
COLORS = {'setcover': 'blue', 'auction': 'green', 'facility': 'magenta', 'indset': 'yellow'}


def display_classification_rich(lp_path, features, ranked, evidence):
    top_type, top_score = ranked[0]
    color = COLORS.get(top_type, 'white')

    console.print()
    console.print(Panel(
        f"[bold {color}]{LABELS[top_type]}[/bold {color}]  ({top_score}% confidence)",
        title=f"[bold]Classification: {os.path.basename(lp_path)}[/bold]",
        subtitle="Based on LP structure analysis",
        border_style=color
    ))

    # Score table
    t = Table(title="All Scores", box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Problem Type", width=28)
    t.add_column("Score", width=8, justify="right")
    t.add_column("Confidence", width=20)
    bar_chars = "█░"
    for pt, sc in ranked:
        bar = "█" * (sc // 5) + "░" * (20 - sc // 5)
        col = COLORS.get(pt, 'white')
        mark = " ◀ best" if pt == top_type else ""
        t.add_row(f"[{col}]{LABELS[pt]}[/{col}]", str(sc), f"[{col}]{bar}[/{col}]{mark}")
    console.print(t)

    # Evidence
    ev = evidence[top_type]
    if ev:
        e = Table(title=f"Evidence for {LABELS[top_type]}", box=box.SIMPLE, header_style="bold")
        e.add_column("Feature", width=40)
        e.add_column("", width=4)
        for k, v in ev.items():
            e.add_row(k, f"[green]{v}[/green]")
        console.print(e)

    # Structure summary
    console.print(Panel(
        f"Objective: [bold]{features['obj_sense']}[/bold]  |  "
        f"Constraints: [bold]{features['n_constraints']}[/bold]  |  "
        f"Binary vars: [bold]{features['n_binary']}[/bold]  |  "
        f"Continuous/Integer: [bold]{features['n_continuous'] + features['n_integer']}[/bold]",
        title="LP Structure", border_style="dim"
    ))


def display_classification_plain(lp_path, features, ranked, evidence):
    print(f"\n{'='*60}")
    print(f"FILE: {os.path.basename(lp_path)}")
    print(f"{'='*60}")
    top_type, top_score = ranked[0]
    print(f"\nBEST MATCH: {LABELS[top_type]}  ({top_score}% confidence)\n")
    print("ALL SCORES:")
    for pt, sc in ranked:
        bar = "█" * (sc // 5) + "░" * (20 - sc // 5)
        mark = " <-- best" if pt == top_type else ""
        print(f"  {LABELS[pt]:30s}  {sc:3d}%  {bar}{mark}")
    print(f"\nEVIDENCE:")
    for k in evidence[top_type]:
        print(f"  • {k}")
    print(f"\nSTRUCTURE:")
    print(f"  Objective      : {features['obj_sense']}")
    print(f"  Constraints    : {features['n_constraints']}")
    print(f"  Binary vars    : {features['n_binary']}")
    print(f"  Continuous/Int : {features['n_continuous'] + features['n_integer']}")
    print(f"{'='*60}\n")


def ask_load_policy_rich(ranked, checkpoint_dir):
    console.print("\n[bold]Available policy options:[/bold]")
    choices = {}
    i = 1
    for pt, sc in ranked:
        gcn_exists  = os.path.exists(os.path.join(checkpoint_dir, f"{pt}_gcn_best.pt"))
        pi2_exists  = os.path.exists(os.path.join(checkpoint_dir, f"{pt}_pi2.pt"))
        status = "[green]✓ ready[/green]" if gcn_exists and pi2_exists else "[red]✗ not trained[/red]"
        console.print(f"  [{i}] {LABELS[pt]:30s}  {status}")
        choices[str(i)] = pt
        i += 1
    console.print(f"  [0] Skip — do not load any policy\n")

    choice = Prompt.ask("Load policy", choices=[str(j) for j in range(i)], default="1")
    if choice == "0":
        return None
    return choices.get(choice)


def ask_load_policy_plain(ranked, checkpoint_dir):
    print("Available policy options:")
    choices = {}
    i = 1
    for pt, sc in ranked:
        gcn_exists = os.path.exists(os.path.join(checkpoint_dir, f"{pt}_gcn_best.pt"))
        status = "ready" if gcn_exists else "not trained"
        print(f"  [{i}] {LABELS[pt]:30s}  ({status})")
        choices[str(i)] = pt
        i += 1
    print("  [0] Skip")
    choice = input("Load policy [1]: ").strip() or "1"
    if choice == "0":
        return None
    return choices.get(choice)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Classify a .lp MIP instance and load its trained policy.")
    parser.add_argument("lp_file",
                        help="Path to the .lp problem file")
    parser.add_argument("--checkpoint-dir", "-c",
                        default="checkpoints",
                        help="Directory containing trained checkpoints (default: checkpoints/)")
    parser.add_argument("--device", "-d",
                        default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for policy inference (default: cpu)")
    parser.add_argument("--solve", "-s",
                        action="store_true",
                        help="Run a SCIP solve with the loaded policy after classifying")
    parser.add_argument("--time-limit", "-t",
                        type=float, default=60.0,
                        help="Time limit for solve in seconds (default: 60)")
    parser.add_argument("--auto",  "-a",
                        action="store_true",
                        help="Auto-load best matching policy without prompting")
    args = parser.parse_args()

    # ── Validate file ─────────────────────────────────────────────────────────
    if not os.path.exists(args.lp_file):
        print(f"ERROR: File not found: {args.lp_file}")
        sys.exit(1)
    if not args.lp_file.endswith('.lp'):
        print(f"WARNING: File does not have .lp extension: {args.lp_file}")

    # ── Parse and classify ────────────────────────────────────────────────────
    print(f"\nParsing {args.lp_file} ...")
    features = parse_lp_file(args.lp_file)
    ranked, evidence = classify(features)

    if RICH:
        display_classification_rich(args.lp_file, features, ranked, evidence)
    else:
        display_classification_plain(args.lp_file, features, ranked, evidence)

    # ── Ask which policy to load ──────────────────────────────────────────────
    if args.auto:
        chosen = ranked[0][0]
        print(f"Auto-loading: {LABELS[chosen]}")
    else:
        if RICH:
            chosen = ask_load_policy_rich(ranked, args.checkpoint_dir)
        else:
            chosen = ask_load_policy_plain(ranked, args.checkpoint_dir)

    if chosen is None:
        print("No policy loaded. Exiting.")
        return

    # ── Load policy ───────────────────────────────────────────────────────────
    print(f"\nLoading {LABELS[chosen]} policy from {args.checkpoint_dir}/ ...")
    policy = load_policy(chosen, args.checkpoint_dir, args.device)

    if policy['errors']:
        print("\nIssues encountered:")
        for e in policy['errors']:
            print(f"  ✗ {e}")

    loaded = []
    if policy['gcn'] is not None:
        info = policy.get('gcn_info', {})
        loaded.append(f"π1 GCN  (val_loss={info.get('val_loss','?'):.4f}, epoch={info.get('epoch','?')})"
                       if isinstance(info.get('val_loss'), float) else "π1 GCN")
    if policy['pi2'] is not None:
        loaded.append("π2 NodeChildMLP (880 params)")
    if policy['norm'] is not None:
        loaded.append("π2 normalisation stats")

    if loaded:
        print("\nLoaded successfully:")
        for l in loaded:
            print(f"  ✓ {l}")
    else:
        print("\nNo policy components loaded. Check checkpoint directory and that training has been run.")
        return

    # ── Optional solve ────────────────────────────────────────────────────────
    do_solve = args.solve
    if not do_solve and not args.auto:
        if RICH:
            do_solve = Confirm.ask(f"\nRun solve on this instance with loaded policy? (time limit: {args.time_limit}s)", default=False)
        else:
            ans = input(f"\nRun solve with policy? [{args.time_limit}s limit] (y/N): ").strip().lower()
            do_solve = ans in ('y', 'yes')

    if do_solve:
        print(f"\nSolving {os.path.basename(args.lp_file)} with π1+π2 policy ...")
        print(f"Time limit: {args.time_limit}s\n")
        try:
            result = solve_with_policy(
                args.lp_file,
                policy['gcn'],
                policy['pi2'],
                policy['norm'],
                args.device,
                time_limit=args.time_limit
            )
            print(f"\n{'─'*40}")
            print(f"Status     : {result['status']}")
            print(f"Nodes      : {result['n_nodes']}")
            print(f"Solve time : {result['solve_time']:.2f}s")
            if result['obj_val'] is not None:
                print(f"Objective  : {result['obj_val']:.6f}")
            print(f"{'─'*40}")
        except Exception as e:
            print(f"Solve failed: {e}")
            print("Make sure SCIP is installed and pyscipopt is working.")


if __name__ == "__main__":
    main()
