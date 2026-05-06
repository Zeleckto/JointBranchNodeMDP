"""
training/data_collector.py

Hybrid search data collection — extends Branch Ranking with L/R labels for π2.

For each B&B node during collection:
  - K=30 sub-solves to estimate trajectory returns  (π1 data)
  - The committed variable x* has n_left, n_right stored separately → y_LR  (π2 data)

All data is crash-safe: checkpointed every COLLECT_EVERY instances.
"""

import os
import math
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from pyscipopt import Model, Branchrule, SCIP_RESULT

import config as cfg
from data.feature_extractor import extract_bipartite_graph, extract_pi2_features


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeSample:
    # ── π1 fields (Branch Ranking, unchanged) ────────────────────────────────
    state_graph:       Optional[dict]  = None
    action_col_idx:    int             = -1
    trajectory_return: Optional[float] = None
    sb_score:          Optional[float] = None
    is_long_term:      bool            = False
    is_short_term:     bool            = False
    reward:            float           = 0.0

    # ── π2 fields (new) ──────────────────────────────────────────────────────
    n_left:            Optional[int]   = None   # nodes in left sub-solve
    n_right:           Optional[int]   = None   # nodes in right sub-solve
    y_LR:              Optional[int]   = None   # 1=LEFT, 0=RIGHT, None=tie/unknown
    feat_pi2:          Optional[np.ndarray] = None  # 9-dim feature vector of x*


# ─────────────────────────────────────────────────────────────────────────────
# Sub-problem solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_subproblem(instance_path, tight_bounds, extra_branch, time_limit=60):
    """
    Solve a restricted sub-B&B with current tight bounds + one extra branch.
    Returns (n_nodes, is_optimal).
    getVarByName does not exist in PySCIPOpt 6.x — use {v.name: v for v in getVars()}.
    """
    try:
        m = Model()
        m.hideOutput(True)
        m.setParam("limits/time",                  time_limit)
        m.setParam("limits/nodes",                 5000)
        m.setParam("separating/maxroundsroot",     -1)
        m.setParam("separating/maxrounds",          0)
        m.setParam("presolving/maxrestarts",        0)
        m.readProblem(instance_path)

        var_dict = {v.name: v for v in m.getVars()}

        for var_name, side, bound in list(tight_bounds) + [extra_branch]:
            var = var_dict.get(var_name)
            if var is None:
                continue
            try:
                if side == 'lb':
                    m.chgVarLb(var, float(bound))
                else:
                    m.chgVarUb(var, float(bound))
            except Exception:
                pass

        m.optimize()
        return m.getNNodes(), m.getStatus() == 'optimal'

    except Exception:
        return 5000, False


def get_current_var_bounds(model):
    """Snapshot tightened variable bounds at current B&B node."""
    tight_bounds = []
    try:
        for var in model.getVars(transformed=True):
            try:
                lb_local  = var.getLbLocal()
                ub_local  = var.getUbLocal()
                lb_global = var.getLbGlobal()
                ub_global = var.getUbGlobal()
                if lb_local > lb_global + 1e-8:
                    tight_bounds.append((var.name, 'lb', lb_local))
                if ub_local < ub_global - 1e-8:
                    tight_bounds.append((var.name, 'ub', ub_local))
            except Exception:
                continue
    except Exception:
        pass
    return tight_bounds


# ─────────────────────────────────────────────────────────────────────────────
# Data collection branch rule
# ─────────────────────────────────────────────────────────────────────────────

class DataCollectionBranchRule(Branchrule):
    """
    Branch rule that runs hybrid search to collect offline data.

    At each node:
      1. Snapshot LP state as bipartite graph
      2. Compute SB scores for all candidates
      3. Sample K variables, run sub-solves → trajectory returns
      4. Commit best variable (highest return)
      5. Store (graph, var, return, n_left, n_right, y_LR, feat_pi2) per sample
    """

    def __init__(self, instance_path: str, use_long_term: bool = True):
        super().__init__()
        self.instance_path  = instance_path
        self.use_long_term  = use_long_term
        self.long_term_groups: List[List[NodeSample]] = []
        self.sb_samples:       List[NodeSample]       = []

    def branchexeclp(self, allowaddcons):
        model = self.model

        # ── Extract bipartite graph ───────────────────────────────────────────
        graph = None
        try:
            graph = extract_bipartite_graph(model)
        except Exception as ex:
            print(f"    Feature extraction failed: {ex}")

        if graph is None or not np.any(graph["cand_mask"]):
            return {"result": SCIP_RESULT.DIDNOTRUN}

        cand_mask = graph["cand_mask"]

        # ── Get candidate variables ───────────────────────────────────────────
        try:
            candidates = [v for v in model.getVars(transformed=True)
                          if v.getCol() is not None and
                             1e-6 < v.getLPSol() - math.floor(v.getLPSol()) < 1 - 1e-6]
        except Exception:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if not candidates:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # Map candidate variable names to column indices in the graph
        cols        = model.getLPColsData()
        col_name_to_idx = {cols[j].getVar().name: j
                           for j in range(len(cols))
                           if cols[j].getVar() is not None}

        # Filter to candidates that appear in graph
        valid_cands = [v for v in candidates if v.name in col_name_to_idx]
        if not valid_cands:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # ── SB proxy scores ───────────────────────────────────────────────────
        sb_scores = {}
        for var in valid_cands:
            lp_val = var.getLPSol()
            frac   = lp_val - math.floor(lp_val)
            sb_scores[var.name] = min(frac, 1.0 - frac) ** 2

        best_sb_name = max(sb_scores, key=sb_scores.get)

        # ── Long-term hybrid search ───────────────────────────────────────────
        tight_bounds   = get_current_var_bounds(model)
        k              = min(cfg.K_EXPLORE, len(valid_cands))
        exp_vars       = np.random.choice(valid_cands, size=k, replace=False).tolist()

        rollout_results = {}  # var_name → dict with n_left, n_right, return
        lt_group        = []

        for var in exp_vars:
            lp_val   = var.getLPSol()
            floor_bd = math.floor(lp_val)
            ceil_bd  = math.ceil(lp_val)

            n_left, _  = solve_subproblem(self.instance_path, tight_bounds,
                                           (var.name, 'ub', floor_bd),
                                           time_limit=cfg.SCIP_TIME_LIMIT)
            n_right, _ = solve_subproblem(self.instance_path, tight_bounds,
                                           (var.name, 'lb', ceil_bd),
                                           time_limit=cfg.SCIP_TIME_LIMIT)

            traj_return = -min(n_left, n_right)
            col_idx     = col_name_to_idx.get(var.name, -1)

            rollout_results[var.name] = {
                'n_left':  n_left,
                'n_right': n_right,
                'return':  traj_return,
            }

            sample = NodeSample(
                state_graph       = graph,
                action_col_idx    = col_idx,
                trajectory_return = traj_return,
            )
            lt_group.append((var.name, sample))

        if not lt_group:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # ── Commit best variable ──────────────────────────────────────────────
        best_name  = max(rollout_results, key=lambda n: rollout_results[n]['return'])
        best_var   = next(v for v in exp_vars if v.name == best_name)
        best_data  = rollout_results[best_name]

        # ── π2 labels for committed variable ─────────────────────────────────
        n_left_best  = best_data['n_left']
        n_right_best = best_data['n_right']
        if n_left_best != n_right_best:
            y_LR = 1 if n_left_best < n_right_best else 0
        else:
            y_LR = None   # tie — excluded from D_π2

        try:
            feat_pi2 = extract_pi2_features(best_var, model)
        except Exception:
            feat_pi2 = None

        # ── Finalise long-term samples ────────────────────────────────────────
        # Store y_LR for ALL K sampled variables — not just committed one.
        # n_left and n_right are already computed for every variable in rollout_results.
        # This gives ~K× more π2 training data at zero additional computation cost.
        lt_samples = []
        exp_var_map = {v.name: v for v in exp_vars}
        for name, sample in lt_group:
            r = rollout_results[name]
            sample.trajectory_return = r['return']
            nl = r['n_left']
            nr = r['n_right']
            sample.n_left  = nl
            sample.n_right = nr
            sample.y_LR    = (1 if nl < nr else 0) if nl != nr else None
            try:
                var_obj = exp_var_map.get(name)
                if var_obj is not None:
                    sample.feat_pi2 = extract_pi2_features(var_obj, model)
            except Exception:
                sample.feat_pi2 = None
            lt_samples.append(sample)

        if self.use_long_term:
            self.long_term_groups.append(lt_samples)

        # ── Short-term SB sample ──────────────────────────────────────────────
        best_sb_var = next((v for v in valid_cands if v.name == best_sb_name), None)
        if best_sb_var is not None:
            sb_col_idx = col_name_to_idx.get(best_sb_name, -1)
            sb_feat_pi2 = None
            sb_y_LR     = None
            sb_n_left   = None
            sb_n_right  = None

            # If SB var was also explored in K rollouts, use its L/R data
            if best_sb_name in rollout_results:
                sb_data    = rollout_results[best_sb_name]
                sb_n_left  = sb_data['n_left']
                sb_n_right = sb_data['n_right']
                if sb_n_left != sb_n_right:
                    sb_y_LR = 1 if sb_n_left < sb_n_right else 0
                try:
                    sb_feat_pi2 = extract_pi2_features(best_sb_var, model)
                except Exception:
                    pass

            sb_sample = NodeSample(
                state_graph    = graph,
                action_col_idx = sb_col_idx,
                sb_score       = sb_scores[best_sb_name],
                n_left         = sb_n_left,
                n_right        = sb_n_right,
                y_LR           = sb_y_LR,
                feat_pi2       = sb_feat_pi2,
            )
            self.sb_samples.append(sb_sample)

        # ── Execute branch ────────────────────────────────────────────────────
        model.branchVar(best_var)
        return {"result": SCIP_RESULT.BRANCHED}


# ─────────────────────────────────────────────────────────────────────────────
# Collection driver
# ─────────────────────────────────────────────────────────────────────────────

def collect_data_from_instance(instance_path, use_long_term=True):
    """Run one instance through hybrid search data collection."""
    m = Model()
    m.hideOutput(True)
    m.setParam("limits/time",              cfg.SCIP_TIME_LIMIT)
    m.setParam("separating/maxroundsroot", -1)
    m.setParam("separating/maxrounds",      0)
    m.setParam("presolving/maxrestarts",    0)
    m.readProblem(instance_path)

    br = DataCollectionBranchRule(instance_path, use_long_term=use_long_term)
    m.includeBranchrule(br, "data_collection", "Data collection branching rule",
                        priority=1_000_000, maxdepth=-1, maxbounddist=1.0)
    m.optimize()
    return br.long_term_groups, br.sb_samples


def collect_dataset(instance_paths, out_path, use_long_term=True,
                    checkpoint_every=None):
    """
    Collect data from multiple instances with crash-safe checkpointing.
    Resumes automatically from partial checkpoint if power cut off mid-run.
    """
    checkpoint_every = checkpoint_every or cfg.COLLECT_EVERY
    partial_path     = out_path.replace('.pkl', '_partial.pkl')
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)

    all_lt  = []
    all_sb  = []
    start   = 0

    if os.path.exists(partial_path):
        try:
            with open(partial_path, 'rb') as f:
                saved = pickle.load(f)
            all_lt = saved.get('long_term_groups', [])
            all_sb = saved.get('sb_samples', [])
            start  = saved.get('completed', 0)
            print(f"  Resuming from instance {start + 1}/{len(instance_paths)}"
                  f"  ({len(all_sb)} SB samples loaded)")
        except Exception as e:
            print(f"  Could not load partial progress ({e}), starting fresh")

    for i, path in enumerate(instance_paths):
        if i < start:
            continue
        print(f"  Collecting from instance {i+1}/{len(instance_paths)}: "
              f"{os.path.basename(path)}")
        try:
            lt, sb = collect_data_from_instance(path, use_long_term)
            all_lt.extend(lt)
            all_sb.extend(sb)
        except Exception as e:
            print(f"    Warning: failed on {path}: {e}")

        if (i + 1) % checkpoint_every == 0:
            with open(partial_path, 'wb') as f:
                pickle.dump({'long_term_groups': all_lt, 'sb_samples': all_sb,
                             'completed': i + 1}, f)
            print(f"    [Checkpoint saved at instance {i+1}]")

    with open(out_path, 'wb') as f:
        pickle.dump({'long_term_groups': all_lt, 'sb_samples': all_sb}, f)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    lt_total = sum(len(g) for g in all_lt)
    print(f"  Saved {len(all_sb)} SB samples, {lt_total} LT samples → {out_path}")
    return all_lt, all_sb


def load_dataset(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['long_term_groups'], d['sb_samples']