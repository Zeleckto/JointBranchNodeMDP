"""
config.py — All hyperparameters for the Joint Branch+Node MDP project.

π1 = Branch Ranking GCN      (variable selection, offline RL)
π2 = NodeChildMLP             (L/R child preference, imitation learning)
"""

import torch

# ── Problem ───────────────────────────────────────────────────────────────────
PROBLEM_TYPE  = "setcover"
PROBLEM_TYPES = ["setcover", "auction", "facility", "indset"]
TRAIN_SIZE    = "easy"
SEED          = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
INSTANCE_DIR  = "instances"
DATA_DIR      = "collected_data"
CHECKPOINT_DIR= "checkpoints"
RESULTS_DIR   = "results"

# ── Data Collection ───────────────────────────────────────────────────────────
K_EXPLORE        = 3          # sub-solves per node (paper: 30, fast test: 3)
SCIP_TIME_LIMIT  = 15      # seconds per sub-solve (paper: 3600, fast: 120)
N_TRAIN_SAMPLES  = 5_000      # max training samples (paper: 50000, fast: 5000)
COLLECT_EVERY    = 5          # checkpoint every N instances during collection

# ── Reward Assignment ─────────────────────────────────────────────────────────
TOP_P            = 0.10       # top-p% trajectory returns → r=1 (paper: 10%)
SB_PROPORTION    = 0.70       # default h (mixing ratio); tuned per benchmark

# Benchmark-specific h values from paper Table 2
SB_PROPORTION_MAP = {
    "setcover": 0.70,
    "auction":  0.90,
    "facility": 0.95,
    "indset":   0.90,
}

# ── GCN Architecture (π1) ─────────────────────────────────────────────────────
CONSTRAINT_FEAT_DIM = 5
EDGE_FEAT_DIM       = 1
VARIABLE_FEAT_DIM   = 14
EMBEDDING_DIM       = 64
GCN_LAYERS          = 1

# ── GCN Training (π1) ────────────────────────────────────────────────────────
GCN_LR            = 1e-3
GCN_WEIGHT_DECAY  = 1e-4
GCN_BATCH_SIZE    = 32
GCN_MAX_EPOCHS    = 10        # paper: 1000
GCN_STOP_PATIENCE = 3        # paper: 20

# ── NodeChildMLP (π2) ─────────────────────────────────────────────────────────
PI2_INPUT_DIM    = 9          # 9 hand-crafted features of x*
PI2_HIDDEN_DIMS  = [32, 16]   # MLP hidden layers
PI2_LR           = 1e-3
PI2_WEIGHT_DECAY = 1e-4
PI2_BATCH_SIZE   = 512
PI2_MAX_EPOCHS   = 100
PI2_PATIENCE     = 10

# ── Inference ─────────────────────────────────────────────────────────────────
USE_PI2          = True       # False = π1 only (ablation baseline)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
