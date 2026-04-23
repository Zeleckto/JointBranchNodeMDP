"""
branching/branch_rule.py

LearnedBranchRule — joint π1 + π2 inference inside SCIP's branchexeclp().

Decision flow at each node:
  1. π1 (GCN): pick variable x* from candidates
  2. π2 (NodeChildMLP): predict L or R preference from x*'s features
  3. Set node_selector.preferred_child so nodeselect() returns that child first
"""

import math
import numpy as np
import torch
from pyscipopt import Branchrule, SCIP_RESULT

from data.feature_extractor import extract_bipartite_graph, extract_pi2_features


class LearnedBranchRule(Branchrule):
    """
    Inference-time branch rule.

    Parameters
    ----------
    gcn          : BranchingGCN model (π1), in eval mode
    pi2          : NodeChildMLP model (π2), in eval mode. None = ablation mode.
    pi2_norm     : (2, 9) numpy array [mean, std] for π2 feature normalisation
    node_selector: BCNodeSelector instance (shared reference)
    device       : 'cpu' or 'cuda'
    """

    def __init__(self, gcn, pi2=None, pi2_norm=None, node_selector=None, device='cpu'):
        super().__init__()
        self.gcn           = gcn
        self.pi2           = pi2
        self.pi2_norm      = pi2_norm    # shape (2, 9): [mean, std]
        self.node_selector = node_selector
        self.device        = device

        if gcn is not None:
            self.gcn.eval()
        if pi2 is not None:
            self.pi2.eval()

    def branchexeclp(self, allowaddcons):
        model = self.model

        # ── Extract LP state ──────────────────────────────────────────────────
        try:
            graph = extract_bipartite_graph(model)
        except Exception:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        if graph is None or not np.any(graph["cand_mask"]):
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # ── Build candidate variable list in cand_mask order ──────────────────
        cols      = model.getLPColsData()
        cand_mask = graph["cand_mask"]
        candidates = []
        for j, flag in enumerate(cand_mask):
            if flag and j < len(cols) and cols[j].getVar() is not None:
                candidates.append(cols[j].getVar())

        if not candidates:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # ── π1: branch head ───────────────────────────────────────────────────
        try:
            con  = torch.from_numpy(graph["con_feats"]).float().to(self.device)
            ei   = torch.from_numpy(graph["edge_index"]).long().to(self.device)
            ef   = torch.from_numpy(graph["edge_feats"]).float().to(self.device)
            vf   = torch.from_numpy(graph["var_feats"]).float().to(self.device)
            mask = torch.from_numpy(graph["cand_mask"]).bool().to(self.device)

            with torch.no_grad():
                logits, _ = self.gcn(con, ei, ef, vf, mask)

            best_local = logits.argmax().item()
            if best_local >= len(candidates):
                best_local = 0
            x_star = candidates[best_local]

        except Exception:
            # Fallback: most fractional variable
            x_star = max(candidates, key=lambda v: min(
                v.getLPSol() - math.floor(v.getLPSol()),
                1.0 - (v.getLPSol() - math.floor(v.getLPSol()))
            ))

        # ── π2: node head ─────────────────────────────────────────────────────
        prefer_left = True   # default if π2 unavailable

        if self.pi2 is not None and self.node_selector is not None:
            try:
                feat     = extract_pi2_features(x_star, model)          # (9,)
                if self.pi2_norm is not None:
                    feat = (feat - self.pi2_norm[0]) / self.pi2_norm[1]  # normalise
                prefer_left = self.pi2.predict(feat)
            except Exception:
                prefer_left = True

        # ── Branch ────────────────────────────────────────────────────────────
        model.branchVar(x_star)

        # ── Signal preferred child to node selector ───────────────────────────
        if self.node_selector is not None:
            children = model.getChildren()
            if len(children) >= 2:
                # SCIP convention for branchVar():
                #   children[0] = down-branch  (x* ≤ ⌊x*⌋)  = LEFT
                #   children[1] = up-branch    (x* ≥ ⌈x*⌉)  = RIGHT
                preferred = children[0] if prefer_left else children[1]
                self.node_selector.preferred_child = preferred.getNumber()
            elif len(children) == 1:
                self.node_selector.preferred_child = children[0].getNumber()

        return {"result": SCIP_RESULT.BRANCHED}
