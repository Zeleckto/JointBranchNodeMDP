"""
visualize_dataset.py

Tabular viewer for collected_data/*.pkl files.
Shows K samples in a clean formatted table with all key fields.

Usage:
    python visualize_dataset.py
    python visualize_dataset.py --problem auction --k 50
    python visualize_dataset.py --k 200 --filter pi2
    python visualize_dataset.py --k 100 --filter reward --export


Examples commands : 
# Default: setcover, 100 rows, all samples
python visualize_dataset.py

# 200 rows for auction
python visualize_dataset.py --problem auction --k 200

# Only rows with valid L/R labels (π2 training data)
python visualize_dataset.py --k 150 --filter pi2

# Only reward=1 rows (what π1 trains on)
python visualize_dataset.py --filter reward

# Only samples where LEFT was preferred
python visualize_dataset.py --filter left

# Export filtered rows to CSV
python visualize_dataset.py --k 200 --filter pi2 --export

# Skip summary stats, just show the table
python visualize_dataset.py --k 50 --no-stats
"""

import os
import sys
import pickle
import argparse
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    RICH = True
    console = Console()
except ImportError:
    RICH = False


FEAT_NAMES = ['frac', 'frac_sym', 'floor', 'ceil',
              'rc_norm', 'obj_norm', 'at_lb', 'n_cons', 'avg_coef']


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load(problem, data_dir="collected_data"):
    path = os.path.join(data_dir, f"{problem}_data.pkl")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run train.py first.")
        sys.exit(1)
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['long_term_groups'], d['sb_samples']


def flatten(lt_groups, sb_samples):
    rows = []
    for gidx, group in enumerate(lt_groups):
        for s in group:
            rows.append({'src': 'LT', 'grp': gidx, 's': s})
    for s in sb_samples:
        rows.append({'src': 'SB', 'grp': -1, 's': s})
    return rows


def apply_filter(rows, filt):
    if filt == 'pi2':
        return [r for r in rows if r['s'].y_LR is not None and r['s'].feat_pi2 is not None]
    if filt == 'reward':
        return [r for r in rows if r['s'].reward == 1.0]
    if filt == 'left':
        return [r for r in rows if r['s'].y_LR == 1]
    if filt == 'right':
        return [r for r in rows if r['s'].y_LR == 0]
    if filt == 'lt':
        return [r for r in rows if r['src'] == 'LT']
    if filt == 'sb':
        return [r for r in rows if r['src'] == 'SB']
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATS
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(lt_groups, sb_samples, all_rows, filtered, filt, k, problem):
    lt_flat   = [r for r in all_rows if r['src'] == 'LT']
    pi2_valid = [r for r in all_rows if r['s'].y_LR is not None]
    r1        = [r for r in all_rows if r['s'].reward == 1.0]
    y_vals    = [r['s'].y_LR for r in pi2_valid]
    rt_vals   = [r['s'].trajectory_return for r in lt_flat if r['s'].trajectory_return is not None]
    feat_ok   = [r for r in lt_flat if r['s'].feat_pi2 is not None]

    if RICH:
        lines = [
            f"[bold]Problem:[/bold] {problem.upper()}",
            f"[bold]B&B nodes:[/bold] {len(lt_groups)}",
            f"[bold]LT samples (K×nodes):[/bold] {len(lt_flat)}",
            f"[bold]SB samples:[/bold] {len(sb_samples)}",
            f"[bold]Total:[/bold] {len(all_rows)}",
            "",
            f"[bold]π2 valid y_LR:[/bold] {len(pi2_valid)} ({100*len(pi2_valid)/max(len(all_rows),1):.1f}%)",
            f"[bold]  y=1 (prefer L):[/bold] {sum(y_vals)} ({100*sum(y_vals)/max(len(y_vals),1):.1f}%)" if y_vals else "",
            f"[bold]  y=0 (prefer R):[/bold] {len(y_vals)-sum(y_vals)} ({100*(len(y_vals)-sum(y_vals))/max(len(y_vals),1):.1f}%)" if y_vals else "",
            f"[bold]reward=1:[/bold] {len(r1)} ({100*len(r1)/max(len(all_rows),1):.1f}%)",
            f"[bold]feat_pi2 populated:[/bold] {len(feat_ok)} ({100*len(feat_ok)/max(len(lt_flat),1):.1f}% of LT)",
        ]
        if rt_vals:
            lines.append(f"[bold]R_t range:[/bold] {min(rt_vals):.0f} to {max(rt_vals):.0f}  (mean {np.mean(rt_vals):.1f})")

        filt_str = f"filter=[yellow]{filt}[/yellow]" if filt != 'all' else "no filter"
        lines.append("")
        lines.append(f"[bold]Showing:[/bold] {min(k, len(filtered))} of {len(filtered)} ({filt_str})")

        console.print(Panel("\n".join(l for l in lines if l is not None),
                            title="[bold]Dataset Summary[/bold]", border_style="blue"))
    else:
        print(f"\n{'═'*60}")
        print(f"  DATASET SUMMARY — {problem.upper()}")
        print(f"{'═'*60}")
        print(f"  B&B nodes            : {len(lt_groups)}")
        print(f"  LT samples           : {len(lt_flat)}")
        print(f"  SB samples           : {len(sb_samples)}")
        print(f"  Total                : {len(all_rows)}")
        print(f"  π2 valid             : {len(pi2_valid)}  ({100*len(pi2_valid)/max(len(all_rows),1):.1f}%)")
        if y_vals:
            print(f"  y=1 (prefer L)       : {sum(y_vals)}  ({100*sum(y_vals)/len(y_vals):.1f}%)")
            print(f"  y=0 (prefer R)       : {len(y_vals)-sum(y_vals)}  ({100*(len(y_vals)-sum(y_vals))/len(y_vals):.1f}%)")
        print(f"  reward=1             : {len(r1)}  ({100*len(r1)/max(len(all_rows),1):.1f}%)")
        if rt_vals:
            print(f"  R_t range            : {min(rt_vals):.0f} to {max(rt_vals):.0f}")
        print(f"\n  Showing {min(k,len(filtered))} of {len(filtered)} (filter={filt})")
        print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE STATS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_stats(all_rows):
    feat_rows = [r for r in all_rows if r['s'].feat_pi2 is not None]
    if not feat_rows:
        return
    F = np.array([r['s'].feat_pi2 for r in feat_rows])

    if RICH:
        t = Table(title="π2 Feature Statistics (9-dim)", box=box.SIMPLE_HEAVY, header_style="bold cyan")
        t.add_column("#",         width=3,  justify="right")
        t.add_column("Feature",   width=12)
        t.add_column("Mean",      width=9,  justify="right")
        t.add_column("Std",       width=9,  justify="right")
        t.add_column("Min",       width=9,  justify="right")
        t.add_column("Max",       width=9,  justify="right")
        t.add_column("Distribution", width=22)
        for j, name in enumerate(FEAT_NAMES):
            col = F[:, j]
            lo, hi = col.min(), col.max()
            rng = hi - lo + 1e-8
            buckets = 10
            hist = [0] * buckets
            for v in col:
                b = min(int((v - lo) / rng * buckets), buckets - 1)
                hist[b] += 1
            bar = ''.join('█' if h > max(hist) * 0.5 else ('▄' if h > max(hist) * 0.2 else '░') for h in hist)
            t.add_row(str(j), name,
                      f"{col.mean():.4f}", f"{col.std():.4f}",
                      f"{lo:.4f}", f"{hi:.4f}", bar)
        console.print(t)
    else:
        print(f"\n{'π2 Feature Statistics':}")
        print(f"  {'#':2s}  {'Feature':12s}  {'Mean':9s}  {'Std':9s}  {'Min':9s}  {'Max':9s}")
        print(f"  {'─'*60}")
        for j, name in enumerate(FEAT_NAMES):
            col = F[:, j]
            print(f"  {j:2d}  {name:12s}  {col.mean():9.4f}  {col.std():9.4f}  {col.min():9.4f}  {col.max():9.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_table_rich(rows, k):
    t = Table(
        title=f"Dataset Samples (showing {min(k, len(rows))} of {len(rows)})",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white on dark_blue",
        row_styles=["", "dim"],
        expand=True,
    )
    t.add_column("#",      width=5,  justify="right")
    t.add_column("src",    width=4)
    t.add_column("grp",    width=5,  justify="right")
    t.add_column("R_t",    width=7,  justify="right")
    t.add_column("n_L",    width=6,  justify="right")
    t.add_column("n_R",    width=6,  justify="right")
    t.add_column("y_LR",   width=7,  justify="center")
    t.add_column("rew",    width=4,  justify="center")
    t.add_column("frac",   width=6,  justify="right")
    t.add_column("f_sym",  width=6,  justify="right")
    t.add_column("rc",     width=8,  justify="right")
    t.add_column("obj",    width=8,  justify="right")
    t.add_column("n_cons", width=7,  justify="right")
    t.add_column("sb_sc",  width=7,  justify="right")
    t.add_column("col_idx",width=7,  justify="right")

    for i, row in enumerate(rows[:k]):
        s = row['s']
        f = s.feat_pi2

        ylr = s.y_LR
        if ylr == 1:   ylr_str = "[green bold]L[/green bold]"
        elif ylr == 0: ylr_str = "[red bold]R[/red bold]"
        else:          ylr_str = "[dim]tie[/dim]"

        rew_str = "[green]1[/green]" if s.reward == 1.0 else "[dim]0[/dim]"
        src_str = f"[blue]{row['src']}[/blue]" if row['src'] == 'LT' else f"[yellow]{row['src']}[/yellow]"

        t.add_row(
            str(i + 1),
            src_str,
            str(row['grp']) if row['grp'] >= 0 else "—",
            f"{s.trajectory_return:.0f}" if s.trajectory_return is not None else "—",
            str(s.n_left)  if s.n_left  is not None else "[dim]—[/dim]",
            str(s.n_right) if s.n_right is not None else "[dim]—[/dim]",
            ylr_str,
            rew_str,
            f"{f[0]:.4f}" if f is not None else "—",
            f"{f[1]:.4f}" if f is not None else "—",
            f"{f[4]:.4f}" if f is not None else "—",
            f"{f[5]:.4f}" if f is not None else "—",
            f"{f[7]:.4f}" if f is not None else "—",
            f"{s.sb_score:.4f}" if s.sb_score is not None else "—",
            str(s.action_col_idx) if s.action_col_idx >= 0 else "—",
        )
    console.print(t)

    # Legend
    console.print(
        "[dim]Columns: # = row, src = LT/SB, grp = B&B node group, "
        "R_t = trajectory return, n_L/n_R = left/right sub-solve nodes, "
        "y_LR = L/R preference (L=green, R=red, tie=dim), rew = reward, "
        "frac/f_sym/rc/obj/n_cons = π2 features, sb_sc = SB proxy score, col_idx = variable column[/dim]"
    )


def print_table_plain(rows, k):
    header = (f"{'#':>4}  {'src':3}  {'grp':4}  {'R_t':>7}  "
              f"{'n_L':>6}  {'n_R':>6}  {'yLR':>4}  {'rew':>3}  "
              f"{'frac':>6}  {'f_sym':>6}  {'rc':>8}  {'obj':>8}  {'n_cons':>6}")
    print(header)
    print('─' * len(header))
    for i, row in enumerate(rows[:k]):
        s = row['s']
        f = s.feat_pi2
        ylr = {1: 'L', 0: 'R', None: 'tie'}.get(s.y_LR, '?')
        print(
            f"{i+1:>4}  {row['src']:3}  {row['grp']:4}  "
            f"{s.trajectory_return:7.0f}  " if s.trajectory_return else f"{'—':>4}  {'—':3}  {row['grp']:4}  {'—':>7}  ",
            end=""
        )
        print(
            f"{str(s.n_left):>6}  {str(s.n_right):>6}  {ylr:>4}  "
            f"{s.reward:>3.0f}  "
            f"{f[0]:6.4f}  {f[1]:6.4f}  {f[4]:8.4f}  {f[5]:8.4f}  {f[7]:6.4f}"
            if f is not None else f"{'—':>6}  {'—':>6}  {ylr:>4}  {s.reward:>3.0f}  —"
        )


def plain_table_row(i, row):
    s = row['s']
    f = s.feat_pi2
    ylr = {1: 'L', 0: 'R', None: 'tie'}.get(s.y_LR, '?')
    rt  = f"{s.trajectory_return:7.0f}" if s.trajectory_return is not None else "      —"
    nl  = f"{s.n_left:6}"  if s.n_left  is not None else "     —"
    nr  = f"{s.n_right:6}" if s.n_right is not None else "     —"
    if f is not None:
        feat_str = f"{f[0]:6.4f}  {f[1]:6.4f}  {f[4]:8.4f}  {f[5]:8.4f}  {f[7]:6.4f}"
    else:
        feat_str = "     —       —         —         —       —"
    print(f"{i+1:>4}  {row['src']:3}  {str(row['grp']) if row['grp']>=0 else '—':>4}  "
          f"{rt}  {nl}  {nr}  {ylr:>4}  {s.reward:>3.0f}  {feat_str}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT CSV
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(rows, problem, filt):
    import csv
    os.makedirs("results", exist_ok=True)
    out = f"results/{problem}_dataset_{filt}.csv"
    fields = ['idx', 'src', 'group', 'trajectory_return', 'n_left', 'n_right',
              'y_LR', 'reward', 'action_col_idx', 'sb_score'] + FEAT_NAMES + \
             ['n_rows', 'n_cols', 'n_edges', 'n_candidates']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows):
            s = row['s']
            feat = s.feat_pi2
            g    = s.state_graph
            d = {
                'idx': i+1, 'src': row['src'], 'group': row['grp'],
                'trajectory_return': s.trajectory_return,
                'n_left': s.n_left, 'n_right': s.n_right,
                'y_LR': s.y_LR, 'reward': s.reward,
                'action_col_idx': s.action_col_idx,
                'sb_score': s.sb_score,
                'n_rows':       g['n_rows']                if g else None,
                'n_cols':       g['n_cols']                if g else None,
                'n_edges':      g['edge_index'].shape[1]   if g else None,
                'n_candidates': int(g['cand_mask'].sum())  if g else None,
            }
            for j, name in enumerate(FEAT_NAMES):
                d[name] = float(feat[j]) if feat is not None else None
            w.writerow(d)
    print(f"\nExported {len(rows)} rows → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Visualise collected dataset pkl file")
    parser.add_argument("--problem",  default="setcover",
                        choices=["setcover", "auction", "facility", "indset"])
    parser.add_argument("--k",        type=int, default=100,
                        help="Number of rows to display (default 100)")
    parser.add_argument("--filter",   default="all",
                        choices=["all", "lt", "sb", "pi2", "reward", "left", "right"],
                        help=("Filter rows: all=no filter, lt=LT only, sb=SB only, "
                              "pi2=has y_LR label, reward=reward=1 only, "
                              "left=y_LR=1, right=y_LR=0"))
    parser.add_argument("--no-stats", action="store_true",
                        help="Skip summary stats and feature statistics tables")
    parser.add_argument("--export",   action="store_true",
                        help="Export ALL (not just displayed K) rows to CSV")
    parser.add_argument("--data-dir", default="collected_data")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    lt_groups, sb_samples = load(args.problem, args.data_dir)
    all_rows  = flatten(lt_groups, sb_samples)
    filtered  = apply_filter(all_rows, args.filter)

    if not filtered:
        print(f"No rows match filter '{args.filter}'. Try a different filter.")
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    if not args.no_stats:
        print_summary(lt_groups, sb_samples, all_rows, filtered, args.filter, args.k, args.problem)
        print_feature_stats(all_rows)

    # ── Main table ────────────────────────────────────────────────────────────
    if RICH:
        print_table_rich(filtered, args.k)
    else:
        header = (f"{'#':>4}  {'src':3}  {'grp':4}  {'R_t':>7}  "
                  f"{'n_L':>6}  {'n_R':>6}  {'yLR':>4}  {'rew':>3}  "
                  f"{'frac':>6}  {'f_sym':>6}  {'rc':>8}  {'obj':>8}  {'n_cons':>6}")
        print(header)
        print('─' * len(header))
        for i, row in enumerate(filtered[:args.k]):
            plain_table_row(i, row)

    # ── Export ────────────────────────────────────────────────────────────────
    if args.export:
        export_csv(filtered, args.problem, args.filter)


if __name__ == "__main__":
    main()
