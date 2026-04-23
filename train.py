"""
train.py — Main training script.

Phase 1: Train π1 (Branch Ranking GCN)
  Step 1: Generate instances
  Step 2: Collect offline data (hybrid search) — with L/R labels for π2
  Step 3: Assign rewards (Definitions 1 & 2)
  Step 4: Train GCN (offline RL, Branch Ranking Eq. 3)

Phase 2: Train π2 (NodeChildMLP)
  Step 5: Extract D_π2 from collected data
  Step 6: Train NodeChildMLP on L/R labels (BCE, imitation learning)

Usage:
  python train.py [--problem setcover] [--skip-generate] [--skip-collect]
                  [--skip-gcn] [--skip-pi2] [--device cpu]
"""

import os
import argparse
import shutil
import torch
import numpy as np

import config as cfg
from data.instance_generator import generate_instances
from training.data_collector import collect_dataset, load_dataset
from training.reward_assigner import (
    assign_long_term_rewards, assign_short_term_rewards,
    build_training_dataset, build_pi2_dataset,
)
from training.trainer import GCNTrainer, initialize_prenorms, train_pi2
from models.gcn import build_gcn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--problem",        default=cfg.PROBLEM_TYPE, choices=cfg.PROBLEM_TYPES)
    p.add_argument("--skip-generate",  action="store_true")
    p.add_argument("--skip-collect",   action="store_true")
    p.add_argument("--skip-gcn",       action="store_true")
    p.add_argument("--skip-pi2",       action="store_true")
    p.add_argument("--device",         default=cfg.DEVICE)
    return p.parse_args()


def main():
    args   = parse_args()
    pt     = args.problem
    device = args.device

    os.makedirs(cfg.INSTANCE_DIR,    exist_ok=True)
    os.makedirs(cfg.DATA_DIR,        exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR,  exist_ok=True)

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 1: Generate instances ─────────────────────────────────────────────
    if not args.skip_generate:
        print(f"\n[1] Generating {pt} instances...")
        train_dir = os.path.join(cfg.INSTANCE_DIR, pt, cfg.TRAIN_SIZE)
        val_dir   = os.path.join(cfg.INSTANCE_DIR, pt, "val")
        train_paths = generate_instances(pt, cfg.TRAIN_SIZE, 200, train_dir, seed=cfg.SEED)
        val_paths   = generate_instances(pt, cfg.TRAIN_SIZE,  50, val_dir,   seed=cfg.SEED+1)
        print(f"  {len(train_paths)} train, {len(val_paths)} val instances")
    else:
        train_dir   = os.path.join(cfg.INSTANCE_DIR, pt, cfg.TRAIN_SIZE)
        val_dir     = os.path.join(cfg.INSTANCE_DIR, pt, "val")
        train_paths = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir)
                               if f.endswith('.lp')]) if os.path.exists(train_dir) else []
        val_paths   = sorted([os.path.join(val_dir, f)   for f in os.listdir(val_dir)
                               if f.endswith('.lp')]) if os.path.exists(val_dir) else []
        print(f"  Found {len(train_paths)} train, {len(val_paths)} val instances")

    # Early exit
    if args.skip_collect and args.skip_gcn and args.skip_pi2:
        print("\nInstance generation complete.")
        return

    # ── Step 2: Collect data ───────────────────────────────────────────────────
    data_path = os.path.join(cfg.DATA_DIR, f"{pt}_data.pkl")
    if not args.skip_collect:
        print(f"\n[2] Collecting offline data  (K={cfg.K_EXPLORE})...")
        collect_dataset(train_paths[:50], data_path, use_long_term=True)
    else:
        if not os.path.exists(data_path):
            print(f"\nERROR: {data_path} not found. Run without --skip-collect first.")
            return
        print(f"  Loaded data from {data_path}")

    # ── Step 3: Reward assignment ──────────────────────────────────────────────
    if not args.skip_gcn or not args.skip_pi2:
        print(f"\n[3] Assigning rewards...")
        lt_groups, sb_samples = load_dataset(data_path)

        lt_flat    = assign_long_term_rewards(lt_groups, top_p=cfg.TOP_P)
        sb_samples = assign_short_term_rewards(sb_samples)

        h = cfg.SB_PROPORTION_MAP.get(pt, cfg.SB_PROPORTION)
        combined, train_graphs, train_rewards = build_training_dataset(lt_flat, sb_samples, h=h)

        n_lt = sum(s.is_long_term  for s in combined)
        n_sb = sum(s.is_short_term for s in combined)
        print(f"  π1 training samples: {len(train_graphs)}  (LT={n_lt}, SB={n_sb})")

        if len(train_graphs) == 0:
            print("  WARNING: 0 training samples. Check data collection.")
            return

    # ── Step 4: Train GCN (π1) ─────────────────────────────────────────────────
    gcn_path = os.path.join(cfg.CHECKPOINT_DIR, f"{pt}_gcn_best.pt")

    if not args.skip_gcn:
        print(f"\n[4] Training GCN (π1) on {device}...")
        gcn = build_gcn(cfg)
        initialize_prenorms(gcn, train_graphs)

        trainer = GCNTrainer(gcn, device=device)

        split      = int(0.9 * len(train_graphs))
        train_data = (combined[:split], train_graphs[:split], train_rewards[:split])
        val_data   = (combined[split:], train_graphs[split:], train_rewards[split:])

        trainer.fit(train_data, val_data, checkpoint_dir=cfg.CHECKPOINT_DIR)

        # Copy best checkpoint with problem-specific name
        best = os.path.join(cfg.CHECKPOINT_DIR, "gcn_best.pt")
        if os.path.exists(best):
            shutil.copy(best, gcn_path)
            print(f"  Saved π1 → {gcn_path}")
    else:
        print(f"  Skipping GCN training")

    # ── Step 5+6: Train NodeChildMLP (π2) ──────────────────────────────────────
    pi2_path = os.path.join(cfg.CHECKPOINT_DIR, f"{pt}_pi2.pt")

    if not args.skip_pi2:
        print(f"\n[5] Building π2 dataset from collected data...")
        # Reload if we skipped GCN training (combined may not be defined)
        if args.skip_gcn and (args.skip_collect):
            lt_groups, sb_samples = load_dataset(data_path)
            lt_flat    = assign_long_term_rewards(lt_groups, top_p=cfg.TOP_P)
            sb_samples = assign_short_term_rewards(sb_samples)

        feats, labels = build_pi2_dataset(lt_flat, sb_samples)

        if len(feats) == 0:
            print("  WARNING: 0 π2 training samples. Check that y_LR is populated.")
            print("  This happens if K_EXPLORE=0 or all sub-solves tied.")
        else:
            print(f"\n[6] Training NodeChildMLP (π2) on {device}...")
            train_pi2(feats, labels, pi2_path, device=device)
    else:
        print("  Skipping π2 training")

    print(f"\nTraining complete. Checkpoints in {cfg.CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
