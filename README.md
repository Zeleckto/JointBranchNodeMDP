# Joint Branch-and-Node MDP for MIP Solving

Extends **Branch Ranking** (Huang et al., ECML-PKDD 2022) from a single-decision MDP
(variable selection only) to a **joint action MDP** where both variable selection and
node child preference are learned by independent policies.

## Architecture

```
π(a_branch, a_node | s) = π1(a_branch | s) · π2(a_node | s, a_branch)

π1 = Branch Ranking GCN          — offline RL, unchanged from paper
π2 = NodeChildMLP (9→32→16→1)   — imitation learning on L/R labels
```

Both policies act for a single agent at each B&B node.
`π2` is structurally limited to `{L, R}` — the two children of the most recent branch.
All other open nodes are handled by SCIP's hybridestim as normal.

## Project Structure

```
branch_ranking_mip/
├── config.py                   # all hyperparameters
├── train.py                    # Phase 1 (π1) + Phase 2 (π2) training
├── evaluate.py                 # 4-way comparison evaluation
├── smoke_test.py               # fast sanity check
├── results_generator.py        # print result tables
├── requirements.txt
├── data/
│   ├── feature_extractor.py    # bipartite graph + π2 features
│   └── instance_generator.py  # synthetic MIP benchmarks
├── models/
│   ├── gcn.py                  # BranchingGCN (π1)
│   └── node_mlp.py             # NodeChildMLP (π2)
├── training/
│   ├── data_collector.py       # hybrid search + L/R label collection
│   ├── reward_assigner.py      # Definitions 1 & 2 + π2 dataset builder
│   └── trainer.py              # GCNTrainer + train_pi2
├── branching/
│   └── branch_rule.py          # LearnedBranchRule (π1 + π2 at inference)
├── node_selection/
│   └── node_selector.py        # BCNodeSelector (preferred-child mechanism)
└── utils/
    └── metrics.py              # SGM, win_rate, Wilcoxon
```

## Quick Start

### 1. Install dependencies
```bash
pip install torch numpy pyscipopt scipy
# SCIP 10.x must be installed separately: https://www.scipopt.org/
```

### 2. Fast test (K=3, ~30 minutes end-to-end)
```bash
# config.py: K_EXPLORE=3, SCIP_TIME_LIMIT=120, GCN_MAX_EPOCHS=30

python smoke_test.py                                          # sanity check
python train.py --problem setcover                            # collect + train
python evaluate.py --problem setcover --n-instances 5 --n-seeds 2
```

### 3. Paper-scale run (K=30, overnight)
```bash
# config.py: K_EXPLORE=30, SCIP_TIME_LIMIT=3600, GCN_MAX_EPOCHS=1000

python train.py --problem setcover
python evaluate.py --problem setcover --difficulty easy   --n-instances 20 --n-seeds 5
python evaluate.py --problem setcover --difficulty medium --n-instances 20 --n-seeds 3
python evaluate.py --problem setcover --difficulty hard   --n-instances 20 --n-seeds 3
```

## Crash Recovery

All long-running steps checkpoint automatically:

| Step | Checkpoint | Resume command |
|---|---|---|
| Data collection | `*_data_partial.pkl` every 5 instances | Same command — auto-detects partial |
| GCN training | `gcn_latest.pt` every 10 epochs | Same command — auto-resumes |
| π2 training | `*_pi2_latest.pt` every 10 epochs | Same command — auto-resumes |

## Four-Way Comparison

| Policy | Branch rule | Node selector |
|---|---|---|
| 1. SCIP default | relpcost | hybridestim |
| 2. π1 only | GCN | hybridestim |
| 3. π1 + SCIP UCT | GCN | SCIP UCT |
| 4. π1 + π2 (proposed) | GCN | BCNodeSelector |

**GO criterion**: Policy 4 vs Policy 2 — win rate ≥ 60% AND Wilcoxon p < 0.05.

## References

- Huang et al. (2022). *Branch Ranking for Efficient Mixed-Integer Programming via Offline Ranking-based Policy Learning*. ECML-PKDD.
- Gasse et al. (2019). *Exact Combinatorial Optimization with Graph Convolutional Neural Networks*. NeurIPS.
