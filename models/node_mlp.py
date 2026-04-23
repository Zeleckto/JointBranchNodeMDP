"""
models/node_mlp.py

NodeChildMLP — π2 policy for child node preference.

Inputs:  9 hand-crafted features of the chosen variable x*
         [frac, frac_sym, floor_bd, ceil_bd, rc_norm, obj_norm,
          at_lb, n_cons_norm, avg_coeff_norm]

Output:  scalar logit → sigmoid → P(prefer LEFT child)
         > 0.5 → go LEFT first (x* ≤ floor)
         ≤ 0.5 → go RIGHT first (x* ≥ ceil)

Training: binary cross-entropy (imitation learning on L/R labels)
          label y=1 if n_L(x*) < n_R(x*), else y=0
"""

import os
import torch
import torch.nn as nn
import numpy as np
import config as cfg


class NodeChildMLP(nn.Module):
    """
    Small MLP for π2.  Input dim = 9, output = 1 logit (no sigmoid — use BCEWithLogitsLoss).
    Call predict() at inference for the boolean L/R preference.
    """

    def __init__(self, input_dim=None, hidden_dims=None):
        super().__init__()
        input_dim   = input_dim   or cfg.PI2_INPUT_DIM
        hidden_dims = hidden_dims or cfg.PI2_HIDDEN_DIMS

        layers = []
        prev   = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (batch, input_dim) → (batch,) raw logits"""
        return self.net(x).squeeze(-1)

    def predict(self, feat_np: np.ndarray) -> bool:
        """
        Inference convenience.
        feat_np: (9,) or (1,9) numpy array — already normalised.
        Returns True = prefer LEFT child.
        """
        self.eval()
        with torch.no_grad():
            t     = torch.from_numpy(feat_np.reshape(1, -1)).float()
            logit = self.forward(t)
            return torch.sigmoid(logit).item() > 0.5


class NodeChildTrainer:
    """Trains NodeChildMLP via binary cross-entropy with crash-safe checkpointing."""

    def __init__(self, model: NodeChildMLP, device: str = 'cpu'):
        self.model     = model.to(device)
        self.device    = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.PI2_LR, weight_decay=cfg.PI2_WEIGHT_DECAY)
        self.criterion = nn.BCEWithLogitsLoss()

    def _eval_loss_and_acc(self, feats, labels):
        self.model.eval()
        with torch.no_grad():
            x      = torch.from_numpy(feats).float().to(self.device)
            y      = torch.from_numpy(labels).float().to(self.device)
            logits = self.model(x)
            loss   = self.criterion(logits, y).item()
            preds  = (torch.sigmoid(logits) > 0.5).float()
            acc    = (preds == y).float().mean().item()
        return loss, acc

    def fit(self, feats: np.ndarray, labels: np.ndarray,
            val_feats: np.ndarray = None, val_labels: np.ndarray = None,
            checkpoint_path: str = None):
        """
        Full training loop.
        feats:  (N, 9) float32 — normalised features
        labels: (N,)   float32 — 0 or 1
        """
        best_val   = float('inf')
        patience   = 0
        start      = 0

        # ── Resume if checkpoint exists ───────────────────────────────────────
        if checkpoint_path:
            latest = checkpoint_path.replace('.pt', '_latest.pt')
            resume = latest if os.path.exists(latest) else \
                     checkpoint_path if os.path.exists(checkpoint_path) else None
            if resume:
                try:
                    ckpt = torch.load(resume, map_location=self.device)
                    if isinstance(ckpt, dict) and 'model_state' in ckpt:
                        self.model.load_state_dict(ckpt['model_state'])
                        self.optimizer.load_state_dict(ckpt['opt_state'])
                        best_val = ckpt.get('val_loss', float('inf'))
                        start    = ckpt.get('epoch', 0) + 1
                        patience = ckpt.get('patience', 0)
                    else:
                        self.model.load_state_dict(ckpt)
                    print(f"  π2 resumed from epoch {start}")
                except Exception as e:
                    print(f"  π2 resume failed ({e}), starting fresh")

        for epoch in range(start, cfg.PI2_MAX_EPOCHS):
            self.model.train()
            idx   = np.random.permutation(len(feats))
            total = 0.0
            n     = 0

            for s in range(0, len(feats), cfg.PI2_BATCH_SIZE):
                bi = idx[s: s + cfg.PI2_BATCH_SIZE]
                x  = torch.from_numpy(feats[bi]).float().to(self.device)
                y  = torch.from_numpy(labels[bi]).float().to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()
                total += loss.item()
                n     += 1

            val_str = ""
            if val_feats is not None:
                vl, acc = self._eval_loss_and_acc(val_feats, val_labels)
                val_str = f"  val={vl:.4f}  acc={acc:.2%}"
                if vl < best_val:
                    best_val = vl
                    patience = 0
                    if checkpoint_path:
                        self._save_full(checkpoint_path, epoch, patience, best_val)
                else:
                    patience += 1
                    if patience >= cfg.PI2_PATIENCE:
                        print(f"  Early stop at epoch {epoch}")
                        break

            if checkpoint_path and epoch % 10 == 0:
                self._save_full(
                    checkpoint_path.replace('.pt', '_latest.pt'),
                    epoch, patience, best_val)

            if epoch % 10 == 0:
                print(f"  π2 epoch {epoch:3d}  train={total/max(n,1):.4f}{val_str}")

    def _save_full(self, path, epoch, patience, val_loss):
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'opt_state':   self.optimizer.state_dict(),
            'epoch':       epoch,
            'patience':    patience,
            'val_loss':    val_loss,
        }, path)

    def save(self, path: str):
        """Save weights only (for lightweight deployment)."""
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            self.model.load_state_dict(ckpt['model_state'])
        else:
            self.model.load_state_dict(ckpt)
