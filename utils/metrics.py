"""
utils/metrics.py

Evaluation metrics and instance solver utilities.
"""

import time
import numpy as np
from pyscipopt import Model


def solve_instance(instance_path, branch_rule=None, node_selector=None,
                   time_limit=3600, paper_settings=True, seed=0):
    """
    Solve a MIP instance and return metrics.

    Returns dict with:
        solve_time : float  (wall-clock seconds, NaN if not solved)
        n_nodes    : int    (B&B nodes processed)
        status     : str    ('optimal', 'infeasible', 'timelimit', ...)
        gap        : float  (primal-dual gap at termination)
    """
    m = Model()
    m.hideOutput(True)
    m.setParam("limits/time", time_limit)
    m.setParam("randomization/randomseedshift", seed)

    if paper_settings:
        m.setParam("separating/maxroundsroot", -1)
        m.setParam("separating/maxrounds",      0)
        m.setParam("presolving/maxrestarts",    0)

    m.readProblem(instance_path)

    if branch_rule is not None:
        m.includeBranchrule(
            branchrule   = branch_rule,
            name         = "learned",
            desc         = "Learned branch rule",
            priority     = 10_000_000,
            maxdepth     = -1,
            maxbounddist = 1.0,
        )

    if node_selector is not None:
        m.includeNodesel(
            nodesel      = node_selector,
            name         = "bc_nodesel",
            desc         = "BC node selector",
            stdpriority  = 1_000_000,
            memsavepriority = 500_000,
        )

    t0 = time.time()
    m.optimize()
    elapsed = time.time() - t0

    status   = m.getStatus()
    n_nodes  = m.getNNodes()
    gap      = m.getGap() if status not in ('infeasible', 'inforunbd') else float('nan')

    solve_time = elapsed if status in ('optimal', 'timelimit') else float('nan')

    return {
        "solve_time": solve_time,
        "n_nodes":    n_nodes,
        "status":     status,
        "gap":        gap,
    }


def shifted_geometric_mean(values, shift=1.0):
    """
    Shifted geometric mean — standard metric for solver benchmarking.
    SGM(x) = exp( mean( log(x_i + shift) ) ) - shift
    """
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float('nan')
    return float(np.exp(np.mean(np.log(arr + shift))) - shift)


def win_rate(times_a, times_b):
    """Fraction of instances where policy A is faster than policy B."""
    wins = sum(a < b for a, b in zip(times_a, times_b) if np.isfinite(a) and np.isfinite(b))
    total = sum(1 for a, b in zip(times_a, times_b) if np.isfinite(a) and np.isfinite(b))
    return wins / max(total, 1)


def wilcoxon_p_value(times_a, times_b):
    """Wilcoxon signed-rank p-value for paired comparison."""
    try:
        from scipy.stats import wilcoxon
        pairs = [(a, b) for a, b in zip(times_a, times_b)
                 if np.isfinite(a) and np.isfinite(b) and a != b]
        if len(pairs) < 5:
            return float('nan')
        a_arr = [p[0] for p in pairs]
        b_arr = [p[1] for p in pairs]
        _, p = wilcoxon(a_arr, b_arr, alternative='less')
        return float(p)
    except Exception:
        return float('nan')
