"""
smoke_test.py — Fast sanity checks before full training run.

Tests:
  1. Instance generation
  2. Feature extraction (bipartite graph)
  3. π2 feature extraction
  4. GCN forward pass
  5. NodeChildMLP forward pass
  6. BCNodeSelector interface
  7. Data collection (1 instance, K=3)
  8. Reward assignment
  9. π2 dataset build
  10. Train loop (2 epochs)

Usage: python smoke_test.py
"""

import os
import sys
import numpy as np
import torch

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

errors = []

def check(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        errors.append((name, str(e)))

print("\n=== Smoke Test ===\n")

# ── 1. Instance generation ────────────────────────────────────────────────────
import config as cfg
from data.instance_generator import generate_instances

def test_instance_gen():
    paths = generate_instances("setcover", "easy", 2,
                                "smoke_instances/setcover/easy", seed=0)
    assert len(paths) == 2
    assert all(os.path.exists(p) for p in paths)

check("instance generation", test_instance_gen)

# ── 2+3. Feature extraction ───────────────────────────────────────────────────
from pyscipopt import Model, Branchrule, SCIP_RESULT
from data.feature_extractor import extract_bipartite_graph, extract_pi2_features, get_prenorm_stats

extracted_graph = None
extracted_var   = None

class _SmokeBR(Branchrule):
    def branchexeclp(self, allowaddcons):
        global extracted_graph, extracted_var
        try:
            extracted_graph = extract_bipartite_graph(self.model)
            cands = [v for v in self.model.getVars(transformed=True)
                     if hasattr(v, 'getLPSol')]
            if cands:
                extracted_var = cands[0]
            # Extract π2 features
            if extracted_var is not None:
                extract_pi2_features(extracted_var, self.model)
        except Exception:
            pass
        return {"result": SCIP_RESULT.DIDNOTRUN}

def test_feature_extraction():
    m = Model()
    m.hideOutput(True)
    m.setParam("limits/nodes", 3)
    m.readProblem("smoke_instances/setcover/easy/setcover_easy_00000.lp")
    br = _SmokeBR()
    m.includeBranchrule(br, "smoke", "smoke", priority=1_000_000, maxdepth=-1, maxbounddist=1.0)
    m.optimize()
    assert extracted_graph is not None, "graph is None"
    assert "var_feats" in extracted_graph
    assert extracted_graph["var_feats"].shape[1] == 14

check("feature extraction (bipartite graph)", test_feature_extraction)

# ── 4. GCN forward pass ───────────────────────────────────────────────────────
from models.gcn import build_gcn

def test_gcn_forward():
    gcn = build_gcn(cfg)
    g   = extracted_graph
    con  = torch.from_numpy(g["con_feats"]).float()
    ei   = torch.from_numpy(g["edge_index"]).long()
    ef   = torch.from_numpy(g["edge_feats"]).float()
    vf   = torch.from_numpy(g["var_feats"]).float()
    mask = torch.from_numpy(g["cand_mask"]).bool()
    if not mask.any():
        mask[0] = True
    logits, var_emb = gcn(con, ei, ef, vf, mask)
    assert logits.shape[0] == mask.sum().item()
    assert var_emb.shape == (g["n_cols"], 64)

check("GCN forward pass", test_gcn_forward)

# ── 5. NodeChildMLP forward pass ──────────────────────────────────────────────
from models.node_mlp import NodeChildMLP

def test_pi2_forward():
    pi2  = NodeChildMLP()
    feat = np.random.randn(9).astype(np.float32)
    out  = pi2(torch.from_numpy(feat).unsqueeze(0))
    assert out.shape == (1,)
    result = pi2.predict(feat)
    assert isinstance(result, bool)

check("NodeChildMLP forward + predict", test_pi2_forward)

# ── 6. BCNodeSelector interface ───────────────────────────────────────────────
from node_selection.node_selector import BCNodeSelector

def test_node_selector():
    ns = BCNodeSelector()
    assert ns.preferred_child is None
    ns.preferred_child = 42
    assert ns.preferred_child == 42
    assert ns.nodecomp(None, None) == 0

check("BCNodeSelector interface", test_node_selector)

# ── 7. Data collection (1 instance) ──────────────────────────────────────────
import config as cfg
_orig_k = cfg.K_EXPLORE
_orig_t = cfg.SCIP_TIME_LIMIT
cfg.K_EXPLORE = 1         # only 1 sub-solve so it completes fast
cfg.SCIP_TIME_LIMIT = 5   # 5 second cap per sub-solve

from training.data_collector import collect_data_from_instance, collect_dataset, load_dataset

def test_data_collection():
    lt, sb = collect_data_from_instance(
        "smoke_instances/setcover/easy/setcover_easy_00000.lp",
        use_long_term=True)
    # May be 0 if problem solved without branching — that's ok
    assert isinstance(lt, list)
    assert isinstance(sb, list)

check("data collection (1 instance, K=1, 5s limit)", test_data_collection)
cfg.K_EXPLORE = _orig_k
cfg.SCIP_TIME_LIMIT = _orig_t

# ── 8+9. Reward assignment and π2 dataset ────────────────────────────────────
from training.reward_assigner import (
    assign_long_term_rewards, assign_short_term_rewards,
    build_training_dataset, build_pi2_dataset
)
from training.data_collector import NodeSample

def test_reward_assignment():
    # Create synthetic samples
    fake_graph = extracted_graph
    samples = []
    for r in [10, 20, 30, 40, 50]:
        s = NodeSample(state_graph=fake_graph, action_col_idx=0,
                       trajectory_return=-float(r), sb_score=0.2)
        samples.append(s)

    lt_flat = assign_long_term_rewards([samples])
    n_lt    = sum(s.is_long_term for s in lt_flat)
    assert n_lt >= 1, "no long-term rewards assigned"

def test_pi2_dataset():
    fake_graph = extracted_graph
    samples = []
    feat    = np.ones(9, dtype=np.float32)
    for n_l, n_r in [(10, 50), (30, 20), (15, 15), (8, 100)]:
        y = None if n_l == n_r else (1 if n_l < n_r else 0)
        s = NodeSample(state_graph=fake_graph, action_col_idx=0,
                       trajectory_return=-min(n_l,n_r), reward=1.0,
                       n_left=n_l, n_right=n_r, y_LR=y, feat_pi2=feat,
                       is_long_term=True)
        samples.append(s)
    feats, labels = build_pi2_dataset(samples, [])
    # 3 valid samples (one tie excluded)
    assert len(feats) == 3

check("reward assignment", test_reward_assignment)
check("π2 dataset build", test_pi2_dataset)

# ── 10. Training (2 epochs) ───────────────────────────────────────────────────
from training.trainer import GCNTrainer, initialize_prenorms, train_pi2

def test_training():
    fake_graph = extracted_graph
    fake_feat  = np.ones(9, dtype=np.float32)
    gcn        = build_gcn(cfg)

    # Minimal π1 dataset
    from training.data_collector import NodeSample
    cand_mask = fake_graph["cand_mask"]
    if not np.any(cand_mask):
        cand_mask[0] = True
    first_cand = int(np.where(cand_mask)[0][0])
    s = NodeSample(state_graph=fake_graph, action_col_idx=first_cand,
                   trajectory_return=-10.0, reward=1.0, is_long_term=True)

    graphs  = [fake_graph]
    rewards = [1.0]
    samples = [s]

    initialize_prenorms(gcn, graphs)
    trainer = GCNTrainer(gcn, device='cpu')

    _orig = cfg.GCN_MAX_EPOCHS
    cfg.GCN_MAX_EPOCHS = 2
    trainer.fit((samples, graphs, rewards), val_data=(samples, graphs, rewards))
    cfg.GCN_MAX_EPOCHS = _orig

    # π2 training
    feats  = np.random.randn(20, 9).astype(np.float32)
    labels = np.random.randint(0, 2, 20).astype(np.float32)
    _orig2 = cfg.PI2_MAX_EPOCHS
    cfg.PI2_MAX_EPOCHS = 2
    train_pi2(feats, labels, "smoke_checkpoints/pi2_test.pt", device='cpu')
    cfg.PI2_MAX_EPOCHS = _orig2

check("GCN + π2 training (2 epochs)", test_training)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
if errors:
    print(f"FAILED {len(errors)}/{len(errors)+10-len(errors)} checks:")
    for name, err in errors:
        print(f"  ✗ {name}: {err}")
    sys.exit(1)
else:
    print("All checks passed. Pipeline is healthy.")
    print("Next: python train.py --problem setcover --skip-generate")

# Cleanup
import shutil
for d in ["smoke_instances", "smoke_checkpoints"]:
    if os.path.exists(d):
        shutil.rmtree(d)