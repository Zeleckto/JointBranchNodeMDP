"""
data/feature_extractor.py

Two responsibilities:
  1. extract_bipartite_graph(model)   → dict of numpy arrays for GCN (π1)
  2. extract_pi2_features(var, model) → np.float32[9] for NodeChildMLP (π2)
  3. get_prenorm_stats(graphs)        → mean/std dicts for GCN prenorm init
"""

import math
import numpy as np

# Feature dimensions (must match config.py)
CONSTRAINT_FEAT_DIM = 5
EDGE_FEAT_DIM       = 1
VARIABLE_FEAT_DIM   = 14


# ─────────────────────────────────────────────────────────────────────────────
# π1 — Bipartite graph features  (Gasse et al. NeurIPS 2019 encoding)
# ─────────────────────────────────────────────────────────────────────────────

def extract_bipartite_graph(model):
    """
    Extract bipartite graph representation of the current LP state.

    Returns dict with keys:
        con_feats   : (n_rows, 5)   float32
        edge_index  : (2, n_edges)  int32   [row_idx, col_idx]
        edge_feats  : (n_edges, 1)  float32
        var_feats   : (n_cols, 14)  float32
        cand_mask   : (n_cols,)     bool    — True for priority branch candidates
        n_rows      : int
        n_cols      : int
        node_number : int           — current SCIP node number
    """
    rows = model.getLPRowsData()
    cols = model.getLPColsData()

    n_rows = len(rows)
    n_cols = len(cols)

    if n_rows == 0 or n_cols == 0:
        return None

    # ── Objective value for normalisation ────────────────────────────────────
    try:
        obj_val = abs(model.getLPObjVal())
    except Exception:
        obj_val = 1.0
    if obj_val < 1.0:
        obj_val = 1.0

    n_lps = max(model.getNLPs(), 1)

    # ── Constraint features (n_rows × 5) ─────────────────────────────────────
    con_feats = np.zeros((n_rows, CONSTRAINT_FEAT_DIM), dtype=np.float32)

    for i, row in enumerate(rows):
        lhs      = row.getLhs()
        rhs      = row.getRhs()
        dualsol  = row.getDualsol()
        row_norm = row.getNorm()
        if row_norm < 1e-8:
            row_norm = 1.0

        # 0: rhs normalised
        con_feats[i, 0] = np.clip(rhs / row_norm, -10.0, 10.0) if not np.isinf(rhs) else 0.0

        # 1: is_tight — compute activity manually
        activity = row.getConstant()
        try:
            for col_obj, coef in zip(row.getCols(), row.getVals()):
                try:
                    lp_v = col_obj.getPrimsol()
                except AttributeError:
                    try:
                        lp_v = col_obj.getVar().getLPSol()
                    except Exception:
                        lp_v = 0.0
                activity += coef * lp_v
        except Exception:
            activity = 0.0

        if not np.isinf(rhs):
            con_feats[i, 1] = float(abs(activity - rhs) < 1e-6)
        elif not np.isinf(lhs):
            con_feats[i, 1] = float(abs(activity - lhs) < 1e-6)
        else:
            try:
                bstat = row.getBasisStatus()
                con_feats[i, 1] = float(bstat != 'basic')
            except Exception:
                con_feats[i, 1] = 0.0

        # 2: dual solution normalised
        con_feats[i, 2] = np.clip(dualsol / obj_val, -10.0, 10.0)

        # 3: LP age normalised
        try:
            con_feats[i, 3] = float(row.getAge()) / n_lps
        except AttributeError:
            con_feats[i, 3] = 0.0

        # 4: objective cosine similarity
        try:
            con_feats[i, 4] = row.getObjParallelism()
        except AttributeError:
            con_feats[i, 4] = 0.0

    # ── Variable features + edges ─────────────────────────────────────────────
    var_feats  = np.zeros((n_cols, VARIABLE_FEAT_DIM), dtype=np.float32)
    edge_rows, edge_cols, edge_vals = [], [], []
    cand_mask  = np.zeros(n_cols, dtype=bool)

    # Build col → index map
    col_to_idx = {}
    for j, col in enumerate(cols):
        col_to_idx[col.getLPPos()] = j

    # Build priority candidate set (npriocands)
    try:
        # getPrioChildren may not exist in all PySCIPOpt versions
        result = model.getPrioChildren()
        prio_vars = {v.name for v in result[0]} if result else set()
    except (AttributeError, Exception):
        # Fallback: use all fractional variables as candidates
        prio_vars = set()

    for j, col in enumerate(cols):
        var = col.getVar()
        if var is None:
            continue

        lp_sol   = col.getPrimsol()
        lb       = col.getLb()
        ub       = col.getUb()
        obj_c    = col.getObjCoeff()
        # getRedcost() not available on Column in PySCIPOpt 6.x
        rc = 0.0
        try:
            rc = var.getRedcost()
        except AttributeError:
            try:
                rc = model.getVarRedcost(var)
            except Exception:
                rc = 0.0
        except Exception:
            rc = 0.0
        basis    = col.getBasisStatus()

        vtype = var.vtype()
        var_feats[j, 0] = float(vtype == 'BINARY')
        var_feats[j, 1] = float(vtype == 'INTEGER')
        var_feats[j, 2] = float(vtype == 'CONTINUOUS')
        var_feats[j, 3] = float(vtype == 'IMPLINT')

        var_feats[j, 4] = np.clip(obj_c / obj_val, -10.0, 10.0)

        var_feats[j, 5] = float(not np.isinf(lb))
        var_feats[j, 6] = float(not np.isinf(ub))

        var_feats[j, 7] = float(abs(lp_sol - lb) < 1e-6) if not np.isinf(lb) else 0.0
        var_feats[j, 8] = float(abs(lp_sol - ub) < 1e-6) if not np.isinf(ub) else 0.0

        # fractionality
        frac = lp_sol - math.floor(lp_sol)
        var_feats[j, 9] = np.clip(frac, 0.0, 1.0)

        # basis status one-hot
        var_feats[j, 10] = float(basis == 'lower')
        var_feats[j, 11] = float(basis == 'basic')
        var_feats[j, 12] = float(basis == 'upper')

        # reduced cost normalised
        var_feats[j, 13] = np.clip(rc / obj_val, -10.0, 10.0)

        # candidate mask
        if 1e-6 < frac < 1 - 1e-6:
            if not prio_vars or var.name in prio_vars:
                cand_mask[j] = True

        # edges
        try:
            col_rows = col.getRows()
            col_vals = col.getVals()
        except Exception:
            col_rows, col_vals = [], []

        for row_obj, coef in zip(col_rows, col_vals):
            r_pos = row_obj.getLPPos()
            if 0 <= r_pos < n_rows:
                rn = row_obj.getNorm()
                norm_coef = coef / (rn if rn > 1e-8 else 1.0)
                edge_rows.append(r_pos)
                edge_cols.append(j)
                edge_vals.append(np.clip(norm_coef, -10.0, 10.0))

    if not np.any(cand_mask):
        # Fallback: all fractional variables
        for j in range(n_cols):
            if 1e-6 < var_feats[j, 9] < 1 - 1e-6:
                cand_mask[j] = True

    edge_index = np.array([edge_rows, edge_cols], dtype=np.int32) if edge_rows else np.zeros((2, 0), dtype=np.int32)
    edge_feats = np.array(edge_vals, dtype=np.float32).reshape(-1, 1) if edge_vals else np.zeros((0, 1), dtype=np.float32)

    try:
        node_number = model.getCurrentNode().getNumber()
    except Exception:
        node_number = -1

    return {
        "con_feats":   con_feats,
        "edge_index":  edge_index,
        "edge_feats":  edge_feats,
        "var_feats":   var_feats,
        "cand_mask":   cand_mask,
        "n_rows":      n_rows,
        "n_cols":      n_cols,
        "node_number": node_number,
    }


# ─────────────────────────────────────────────────────────────────────────────
# π2 — 9-dim feature vector for chosen variable x*
# ─────────────────────────────────────────────────────────────────────────────

def extract_pi2_features(var, model):
    """
    Extract 9 hand-crafted features from chosen variable x* at current LP state.
    Called inside branchexeclp() after π1 selects x*.

    Features:
      0  frac(x*)         fractionality ∈ [0,1]
      1  frac_sym(x*)     min(frac, 1-frac) ∈ [0, 0.5]
      2  floor_bd         ⌊x̄⌋
      3  ceil_bd          ⌈x̄⌉
      4  rc_norm          reduced cost / |obj_val|, clipped [-10,10]
      5  obj_norm         obj coeff / |obj_val|, clipped [-10,10]
      6  at_lower_bound   binary: is x* at its lower bound?
      7  n_cons_norm      fraction of LP rows x* appears in
      8  avg_coeff_norm   mean |A[i,x*]| / ||row_i||

    Returns: np.float32[9]
    """
    try:
        lp_val = var.getLPSol()
    except Exception:
        lp_val = 0.5

    frac      = lp_val - math.floor(lp_val)
    frac_sym  = min(frac, 1.0 - frac)
    floor_bd  = math.floor(lp_val)
    ceil_bd   = math.ceil(lp_val)

    try:
        rc = var.getRedcost()
    except AttributeError:
        try:
            rc = model.getVarRedcost(var)
        except Exception:
            rc = 0.0
    except Exception:
        rc = 0.0

    try:
        obj_c = var.getObj()
    except Exception:
        obj_c = 0.0

    try:
        obj_val = abs(model.getLPObjVal())
    except Exception:
        obj_val = 1.0
    if obj_val < 1.0:
        obj_val = 1.0

    rc_norm  = float(np.clip(rc / obj_val, -10.0, 10.0))
    obj_norm = float(np.clip(obj_c / obj_val, -10.0, 10.0))

    try:
        lb_local = var.getLbLocal()
        at_lb    = float(abs(lp_val - lb_local) < 1e-6)
    except Exception:
        at_lb = 0.0

    # Constraint coupling
    try:
        col       = var.getCol()
        col_rows  = col.getRows()
        col_vals  = col.getVals()
        n_cons    = len(col_rows)
        m         = max(model.getNLPRows(), 1)
        norms     = [r.getNorm() for r in col_rows]
        avg_coeff = float(np.mean([abs(v) / (n if n > 1e-8 else 1.0)
                                   for v, n in zip(col_vals, norms)])) if col_vals else 0.0
        avg_coeff = float(np.clip(avg_coeff, -10.0, 10.0))
    except Exception:
        n_cons, m, avg_coeff = 1, 1, 0.0

    return np.array([
        frac,
        frac_sym,
        float(floor_bd),
        float(ceil_bd),
        rc_norm,
        obj_norm,
        at_lb,
        float(n_cons) / float(m),
        avg_coeff,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Prenorm statistics for GCN initialisation
# ─────────────────────────────────────────────────────────────────────────────

def get_prenorm_stats(feature_dicts):
    """
    Compute mean/std prenorm stats from a list of bipartite graph dicts.
    Also stores sample_graphs key so GCN can run forward-pass hooks to
    initialise internal conv-layer prenorm_c / prenorm_v layers.
    """
    def clean(arr):
        arr = np.array(arr, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.clip(arr, -1e6, 1e6)
        return arr.astype(np.float32)

    all_con = clean(np.concatenate([d["con_feats"] for d in feature_dicts], axis=0))
    all_var = clean(np.concatenate([d["var_feats"] for d in feature_dicts], axis=0))
    edge_list = [d["edge_feats"] for d in feature_dicts if len(d["edge_feats"]) > 0]
    if edge_list:
        all_edg = clean(np.concatenate(edge_list, axis=0))
        edg_mean = all_edg.mean(0).astype(np.float32)
        edg_std  = np.maximum(all_edg.std(0), 1e-4).astype(np.float32)
    else:
        edg_mean = np.zeros(1, dtype=np.float32)
        edg_std  = np.ones(1,  dtype=np.float32)

    return {
        "con_mean":      all_con.mean(0).astype(np.float32),
        "con_std":       np.maximum(all_con.std(0), 1e-4).astype(np.float32),
        "var_mean":      all_var.mean(0).astype(np.float32),
        "var_std":       np.maximum(all_var.std(0), 1e-4).astype(np.float32),
        "edg_mean":      edg_mean,
        "edg_std":       edg_std,
        "sample_graphs": feature_dicts,
    }