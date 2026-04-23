"""
models/gcn.py

Bipartite Graph Convolutional Network for variable selection (π1).
Architecture follows Gasse et al. NeurIPS 2019 exactly:
  - PrenormLayer  (fixed affine normalisation, fitted once from data)
  - BipartiteConvLayer  (un-normalised sum aggregation, V→C then C→V)
  - output_mlp  (shared scorer: 64→64→1)

The internal prenorm_c / prenorm_v layers inside each BipartiteConvLayer
are initialised via forward-pass hooks (not from raw feature stats) to
normalise aggregation sums whose scale depends on graph degree.
"""

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class PrenormLayer(nn.Module):
    """Fixed affine normalisation: x ← clip((x - β) / σ, -10, 10). Frozen after init."""

    def __init__(self, n_features):
        super().__init__()
        self.register_buffer('beta',  torch.zeros(n_features))
        self.register_buffer('sigma', torch.ones(n_features))
        self._initialized = False

    def initialize(self, mean: np.ndarray, std: np.ndarray):
        mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        std  = np.where(np.isfinite(std) & (std > 1e-4), std, 1.0).astype(np.float32)
        self.beta.copy_(torch.from_numpy(mean))
        self.sigma.copy_(torch.from_numpy(std))
        self._initialized = True

    def forward(self, x):
        return torch.clamp((x - self.beta) / self.sigma, -10.0, 10.0)


class BipartiteConvLayer(nn.Module):
    """
    One bipartite message-passing round:
      Pass 1  V → C:  c ← fC(c, prenorm_c(Σ gC(c_i, v_j, e_ij)))
      Pass 2  C → V:  v ← fV(v, prenorm_v(Σ gV(c_i, v_j, e_ij)))
    Un-normalised sum (not mean).
    """

    def __init__(self, emb_dim, edge_dim):
        super().__init__()
        self.emb_dim  = emb_dim
        self.edge_dim = edge_dim

        self.gC = MLP(2 * emb_dim + edge_dim, emb_dim, emb_dim)
        self.gV = MLP(2 * emb_dim + edge_dim, emb_dim, emb_dim)
        self.fC = MLP(2 * emb_dim, emb_dim, emb_dim)
        self.fV = MLP(2 * emb_dim, emb_dim, emb_dim)

        self.prenorm_c = PrenormLayer(emb_dim)
        self.prenorm_v = PrenormLayer(emb_dim)

    def forward(self, c, v, edge_index, e):
        row_idx = edge_index[0]
        col_idx = edge_index[1]

        # Pass 1: V → C
        msgs_c = self.gC(torch.cat([c[row_idx], v[col_idx], e], dim=-1))
        agg_c  = torch.zeros(c.shape[0], self.emb_dim, device=c.device)
        agg_c.scatter_add_(0, row_idx.unsqueeze(-1).expand_as(msgs_c), msgs_c)
        c_new  = self.fC(torch.cat([c, self.prenorm_c(agg_c)], dim=-1))

        # Pass 2: C → V
        msgs_v = self.gV(torch.cat([c_new[row_idx], v[col_idx], e], dim=-1))
        agg_v  = torch.zeros(v.shape[0], self.emb_dim, device=v.device)
        agg_v.scatter_add_(0, col_idx.unsqueeze(-1).expand_as(msgs_v), msgs_v)
        v_new  = self.fV(torch.cat([v, self.prenorm_v(agg_v)], dim=-1))

        return c_new, v_new


class BranchingGCN(nn.Module):
    """
    Full bipartite GCN for π1 (variable selection).

    forward() returns:
        logits         : (n_cands,)      scores for each candidate variable
        var_embeddings : (n_cols, emb_dim)  all variable embeddings
    """

    def __init__(self, con_dim=5, edge_dim=1, var_dim=14, emb_dim=64, n_layers=1):
        super().__init__()
        self.emb_dim  = emb_dim
        self.n_layers = n_layers

        self.prenorm_con = PrenormLayer(con_dim)
        self.prenorm_var = PrenormLayer(var_dim)
        self.prenorm_edg = PrenormLayer(edge_dim)

        self.con_embed = MLP(con_dim,  emb_dim, emb_dim)
        self.var_embed = MLP(var_dim,  emb_dim, emb_dim)

        self.conv_layers = nn.ModuleList([
            BipartiteConvLayer(emb_dim, edge_dim) for _ in range(n_layers)
        ])

        self.output_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

    def initialize_prenorms(self, stats: dict):
        """
        Initialise ALL prenorm layers:
          - input prenorms from raw feature statistics
          - internal conv prenorm_c / prenorm_v from forward-pass hooks
        """
        self.prenorm_con.initialize(stats["con_mean"], stats["con_std"])
        self.prenorm_var.initialize(stats["var_mean"], stats["var_std"])
        self.prenorm_edg.initialize(stats["edg_mean"], stats["edg_std"])

        if not self.conv_layers:
            return

        agg_c_samples = [[] for _ in self.conv_layers]
        agg_v_samples = [[] for _ in self.conv_layers]

        def make_hooks(idx):
            def hook_c(module, inp, out):
                agg_c_samples[idx].append(inp[0].detach().cpu().float())
            def hook_v(module, inp, out):
                agg_v_samples[idx].append(inp[0].detach().cpu().float())
            return hook_c, hook_v

        handles = []
        for i, layer in enumerate(self.conv_layers):
            hc, hv = make_hooks(i)
            handles.append(layer.prenorm_c.register_forward_hook(hc))
            handles.append(layer.prenorm_v.register_forward_hook(hv))

        self.eval()
        device  = next(self.parameters()).device
        graphs  = stats.get("sample_graphs", [])

        with torch.no_grad():
            for graph in graphs[:min(50, len(graphs))]:
                try:
                    con = torch.from_numpy(graph["con_feats"]).float().to(device)
                    ei  = torch.from_numpy(graph["edge_index"]).long().to(device)
                    ef  = torch.from_numpy(graph["edge_feats"]).float().to(device)
                    var = torch.from_numpy(graph["var_feats"]).float().to(device)
                    msk = torch.from_numpy(graph["cand_mask"]).bool().to(device)
                    if msk.any():
                        self.forward(con, ei, ef, var, msk)
                except Exception:
                    continue

        for h in handles:
            h.remove()

        for i, layer in enumerate(self.conv_layers):
            if agg_c_samples[i]:
                arr = torch.cat(agg_c_samples[i], dim=0).numpy()
                arr = np.where(np.isfinite(arr), arr, 0.0)
                layer.prenorm_c.initialize(
                    arr.mean(0).astype(np.float32),
                    np.maximum(arr.std(0), 1e-4).astype(np.float32))
            if agg_v_samples[i]:
                arr = torch.cat(agg_v_samples[i], dim=0).numpy()
                arr = np.where(np.isfinite(arr), arr, 0.0)
                layer.prenorm_v.initialize(
                    arr.mean(0).astype(np.float32),
                    np.maximum(arr.std(0), 1e-4).astype(np.float32))

    def forward(self, con_feats, edge_index, edge_feats, var_feats, cand_mask):
        c = self.prenorm_con(con_feats)
        v = self.prenorm_var(var_feats)
        e = self.prenorm_edg(edge_feats)

        c = self.con_embed(c)
        v = self.var_embed(v)

        for layer in self.conv_layers:
            c, v = layer(c, v, edge_index, e)

        var_embeddings  = v
        cand_embeddings = v[cand_mask]
        logits          = self.output_mlp(cand_embeddings).squeeze(-1)

        return logits, var_embeddings


def build_gcn(cfg) -> BranchingGCN:
    return BranchingGCN(
        con_dim  = cfg.CONSTRAINT_FEAT_DIM,
        edge_dim = cfg.EDGE_FEAT_DIM,
        var_dim  = cfg.VARIABLE_FEAT_DIM,
        emb_dim  = cfg.EMBEDDING_DIM,
        n_layers = cfg.GCN_LAYERS,
    )
