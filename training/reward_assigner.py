"""
training/reward_assigner.py

Reward assignment for π1 (Branch Ranking Definitions 1 and 2).
π2 labels (y_LR) are already attached to NodeSamples during collection.
"""

import numpy as np
from typing import List, Tuple
import config as cfg
from training.data_collector import NodeSample


# ─────────────────────────────────────────────────────────────────────────────
# π1 reward assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_long_term_rewards(node_groups: List[List[NodeSample]],
                              top_p: float = None) -> List[NodeSample]:
    """
    Definition 1 — Long-Term Promising:
    For each group of K samples at one node, label top-p% by trajectory_return as r=1.
    """
    top_p = top_p or cfg.TOP_P
    flat  = []

    for group in node_groups:
        valid = [s for s in group if s.trajectory_return is not None]
        if not valid:
            flat.extend(group)
            continue

        returns   = np.array([s.trajectory_return for s in valid])
        threshold = np.percentile(returns, 100 * (1 - top_p))

        for s in valid:
            if s.trajectory_return >= threshold:
                s.is_long_term = True
                s.reward       = 1.0

        flat.extend(group)

    return flat


def assign_short_term_rewards(samples: List[NodeSample]) -> List[NodeSample]:
    """
    Definition 2 — Short-Term Promising:
    For each SB sample, mark it as r=1 (only one per node, the highest SB score).
    """
    for s in samples:
        if s.sb_score is not None:
            s.is_short_term = True
            s.reward        = 1.0
    return samples


def build_training_dataset(lt_flat: List[NodeSample],
                            sb_samples: List[NodeSample],
                            h: float = None
                            ) -> Tuple[List[NodeSample], List[dict], List[float]]:
    """
    Combine long-term and short-term promising samples with mixing ratio h.

    h = proportion of short-term samples in final dataset.
    Returns (combined_samples, graphs, rewards).
    """
    h = h if h is not None else cfg.SB_PROPORTION

    lt_promising = [s for s in lt_flat    if s.is_long_term]
    sb_promising = [s for s in sb_samples if s.is_short_term]

    n_sb_target = int(len(lt_promising) * h / (1 - h + 1e-8)) if h < 1.0 else len(sb_promising)
    n_sb_target = min(n_sb_target, len(sb_promising))

    rng         = np.random.RandomState(cfg.SEED)
    sb_selected = rng.choice(sb_promising, size=n_sb_target, replace=False).tolist() \
                  if n_sb_target < len(sb_promising) else sb_promising

    combined = lt_promising + sb_selected
    rng.shuffle(combined)

    graphs  = [s.state_graph for s in combined]
    rewards = [s.reward      for s in combined]

    return combined, graphs, rewards


def build_pi2_dataset(lt_flat: List[NodeSample],
                       sb_samples: List[NodeSample]
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build D_π2 for training π2.

    Uses reward=1 samples that have y_LR defined (not a tie) and feat_pi2 present.
    Returns (feats, labels) as numpy arrays.
    """
    feats  = []
    labels = []

    for s in lt_flat:
        if s.reward == 1.0 and s.y_LR is not None and s.feat_pi2 is not None:
            feats.append(s.feat_pi2)
            labels.append(float(s.y_LR))

    for s in sb_samples:
        if s.reward == 1.0 and s.y_LR is not None and s.feat_pi2 is not None:
            feats.append(s.feat_pi2)
            labels.append(float(s.y_LR))

    if not feats:
        return np.zeros((0, cfg.PI2_INPUT_DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)

    return (np.array(feats,  dtype=np.float32),
            np.array(labels, dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Loss function for π1 training
# ─────────────────────────────────────────────────────────────────────────────

def compute_weighted_ce_loss(logits_list, target_indices, rewards):
    """
    Weighted cross-entropy: L = -Σ r(s,a) * log π(a|s).
    logits_list   : list of (n_cands,) tensors
    target_indices: list of int (local index into candidates)
    rewards       : list of float (0 or 1)
    """
    import torch
    import torch.nn.functional as F

    total = torch.tensor(0.0, requires_grad=True)
    n     = 0

    for logits, idx, r in zip(logits_list, target_indices, rewards):
        if r == 0.0 or idx < 0 or idx >= logits.shape[0]:
            continue
        log_probs = F.log_softmax(logits, dim=0)
        total     = total - r * log_probs[idx]
        n        += 1

    return total / max(n, 1)
