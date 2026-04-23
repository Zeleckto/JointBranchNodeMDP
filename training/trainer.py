"""
training/trainer.py

GCNTrainer — trains π1 (Branch Ranking GCN) with crash-safe checkpointing.
initialize_prenorms — fits all prenorm layers before training.
train_pi2 — Phase 2: trains π2 (NodeChildMLP) on L/R labels.
"""

import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from models.gcn import BranchingGCN, build_gcn
from models.node_mlp import NodeChildMLP, NodeChildTrainer
from data.feature_extractor import get_prenorm_stats
from training.reward_assigner import compute_weighted_ce_loss


# ─────────────────────────────────────────────────────────────────────────────
# Prenorm initialisation
# ─────────────────────────────────────────────────────────────────────────────

def initialize_prenorms(gcn: BranchingGCN, graphs):
    """
    Compute statistics from training graphs and initialise ALL prenorm layers:
      - Input prenorms (prenorm_con, prenorm_var, prenorm_edg) from raw feature stats
      - Internal conv prenorm_c / prenorm_v from forward-pass aggregation stats
    Must be called once before training begins.
    """
    print("Initializing prenorm layers from training data stats...")
    stats = get_prenorm_stats(graphs)
    gcn.initialize_prenorms(stats)
    print("  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# GCN Trainer (π1)
# ─────────────────────────────────────────────────────────────────────────────

class GCNTrainer:
    def __init__(self, model: BranchingGCN, device: str = 'cpu'):
        self.model          = model.to(device)
        self.device         = device
        self.opt            = torch.optim.Adam(
            model.parameters(), lr=cfg.GCN_LR, weight_decay=cfg.GCN_WEIGHT_DECAY)
        self.scheduler      = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=5, factor=0.5, verbose=True)
        self.best_val_loss  = float('inf')
        self.patience_count = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _graph_to_tensors(self, graph):
        dev = self.device
        con  = torch.from_numpy(graph["con_feats"]).float().to(dev)
        ei   = torch.from_numpy(graph["edge_index"]).long().to(dev)
        ef   = torch.from_numpy(graph["edge_feats"]).float().to(dev)
        vf   = torch.from_numpy(graph["var_feats"]).float().to(dev)
        mask = torch.from_numpy(graph["cand_mask"]).bool().to(dev)
        return con, ei, ef, vf, mask

    def _get_local_idx(self, sample) -> int:
        """Map global column index → local index in cand_mask."""
        cand_mask = sample.state_graph["cand_mask"]
        col_idx   = sample.action_col_idx
        cand_positions = np.where(cand_mask)[0]
        matches = np.where(cand_positions == col_idx)[0]
        return int(matches[0]) if len(matches) > 0 else -1

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, samples, graphs, rewards) -> float:
        self.model.train()
        idx   = np.random.permutation(len(samples))
        total = 0.0
        n     = 0

        for batch_start in range(0, len(idx), cfg.GCN_BATCH_SIZE):
            batch = idx[batch_start: batch_start + cfg.GCN_BATCH_SIZE]
            self.opt.zero_grad()

            batch_logits  = []
            batch_targets = []
            batch_rewards = []

            for bi in batch:
                sample = samples[bi]
                if rewards[bi] == 0:
                    continue
                try:
                    con, ei, ef, vf, mask = self._graph_to_tensors(graphs[bi])
                    logits, _             = self.model(con, ei, ef, vf, mask)
                    target                = self._get_local_idx(sample)
                    if target < 0:
                        continue
                    batch_logits.append(logits)
                    batch_targets.append(target)
                    batch_rewards.append(rewards[bi])
                except Exception:
                    continue

            if not batch_logits:
                continue

            loss = compute_weighted_ce_loss(batch_logits, batch_targets, batch_rewards)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total += loss.item()
            n     += 1

        return total / max(n, 1)

    def evaluate(self, samples, graphs, rewards) -> float:
        self.model.eval()
        total = 0.0
        n     = 0
        with torch.no_grad():
            for sample, graph, reward in zip(samples, graphs, rewards):
                if reward == 0:
                    continue
                try:
                    con, ei, ef, vf, mask = self._graph_to_tensors(graph)
                    logits, _             = self.model(con, ei, ef, vf, mask)
                    target                = self._get_local_idx(sample)
                    if target < 0:
                        continue
                    loss = compute_weighted_ce_loss([logits], [target], [reward])
                    total += loss.item()
                    n     += 1
                except Exception:
                    continue
        return total / max(n, 1)

    # ── Full training loop ────────────────────────────────────────────────────

    def fit(self, train_data, val_data=None, checkpoint_dir=None):
        """
        Train with crash-safe checkpointing.
        Saves gcn_latest.pt every 10 epochs, gcn_best.pt on val_loss improvement.
        Resumes automatically from gcn_latest.pt if present.
        """
        train_samples, train_graphs, train_rewards = train_data
        print(f"GCN training: {len(train_graphs)} samples, device={self.device}")

        # ── Resume ────────────────────────────────────────────────────────────
        start_epoch = 0
        if checkpoint_dir:
            latest_path = os.path.join(checkpoint_dir, "gcn_latest.pt")
            best_path   = os.path.join(checkpoint_dir, "gcn_best.pt")
            resume      = latest_path if os.path.exists(latest_path) else \
                          best_path   if os.path.exists(best_path)   else None
            if resume:
                try:
                    ckpt = torch.load(resume, map_location=self.device)
                    self.model.load_state_dict(ckpt['model_state'])
                    self.opt.load_state_dict(ckpt['opt_state'])
                    self.best_val_loss  = ckpt.get('val_loss', float('inf'))
                    start_epoch         = ckpt.get('epoch', 0) + 1
                    self.patience_count = ckpt.get('patience', 0)
                    print(f"  Resumed from epoch {start_epoch} "
                          f"(best val={self.best_val_loss:.4f})")
                except Exception as e:
                    print(f"  Resume failed ({e}), starting fresh")

        for epoch in range(start_epoch, cfg.GCN_MAX_EPOCHS):
            train_loss = self.train_epoch(train_samples, train_graphs, train_rewards)

            val_str = ""
            if val_data is not None:
                vs, vg, vr = val_data
                val_loss   = self.evaluate(vs, vg, vr)
                val_str    = f"  val={val_loss:.4f}"
                self.scheduler.step(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss   = val_loss
                    self.patience_count  = 0
                    if checkpoint_dir:
                        self.save(os.path.join(checkpoint_dir, "gcn_best.pt"), epoch=epoch)
                else:
                    self.patience_count += 1
                    if self.patience_count >= cfg.GCN_STOP_PATIENCE:
                        print(f"Early stop at epoch {epoch}")
                        break

            if checkpoint_dir and epoch % 10 == 0:
                self.save(os.path.join(checkpoint_dir, "gcn_latest.pt"),
                          epoch=epoch, patience=self.patience_count)

            if epoch % 5 == 0:
                print(f"Epoch {epoch:4d}  train={train_loss:.4f}{val_str}")

        print("GCN training complete.")

    def save(self, path, epoch=0, patience=0):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'opt_state':   self.opt.state_dict(),
            'val_loss':    self.best_val_loss,
            'epoch':       epoch,
            'patience':    patience,
        }, path)
        print(f"  Saved checkpoint → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])


# ─────────────────────────────────────────────────────────────────────────────
# π2 training helper
# ─────────────────────────────────────────────────────────────────────────────

def train_pi2(feats: np.ndarray, labels: np.ndarray, checkpoint_path: str,
              device: str = 'cpu') -> tuple:
    """
    Phase 2: train NodeChildMLP (π2) on L/R labels.

    Normalises features, saves norm stats alongside checkpoint.
    Returns (model, mean, std).
    """
    if len(feats) == 0:
        print("  WARNING: no π2 training samples. Skipping.")
        return None, None, None

    pos_rate = labels.mean()
    print(f"  π2 dataset: {len(labels)} samples  pos_rate={pos_rate:.2%}  "
          f"baseline={max(pos_rate, 1-pos_rate):.2%}")

    # Normalise
    mean = feats.mean(0)
    std  = np.maximum(feats.std(0), 1e-8)
    feats_norm = (feats - mean) / std

    # Save norm stats
    norm_path = checkpoint_path.replace('.pt', '_norm.npy')
    np.save(norm_path, np.stack([mean, std]))
    print(f"  Saved π2 norm stats → {norm_path}")

    # Train
    split   = int(0.9 * len(feats_norm))
    model   = NodeChildMLP()
    trainer = NodeChildTrainer(model, device=device)
    trainer.fit(
        feats_norm[:split],  labels[:split],
        feats_norm[split:],  labels[split:],
        checkpoint_path=checkpoint_path,
    )
    trainer.save(checkpoint_path)
    print(f"  Saved π2 → {checkpoint_path}")

    return model, mean, std
