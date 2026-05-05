# JointBranchNodeMDP — Codebase Reference
### For Recreation and Explanation

---

## What This Project Does

Extends **Branch Ranking** (Huang et al., ECML-PKDD 2022) from a single-decision MDP (variable selection only) to a **joint action MDP** where both:
1. **π1** — which variable to branch on (GCN, offline RL, unchanged from paper)
2. **π2** — which child (L or R) to expand first (small MLP, imitation learning, new)

Both run inside **SCIP** via native plugin callbacks. No new GCN parameters.

---

## File-by-File Summary

### `config.py`
Single source of truth for all hyperparameters.
- `K_EXPLORE` — how many sub-solves per node during data collection (3=fast test, 30=paper)
- `SCIP_TIME_LIMIT` — cap per sub-solve in seconds
- `GCN_*` — π1 training settings
- `PI2_*` — π2 training settings
- `SB_PROPORTION_MAP` — per-benchmark mixing ratio h (from paper)

**Edit this file to switch between fast test and paper scale.**

---

### `data/feature_extractor.py`

Two functions:

**`extract_bipartite_graph(model)`**
- Called inside `branchexeclp()` at each B&B node
- Extracts constraint features (5-dim), variable features (14-dim), edge features (1-dim) from current SCIP LP state
- Returns dict with `con_feats`, `edge_index`, `edge_feats`, `var_feats`, `cand_mask`
- `cand_mask` marks fractional integer variables that are branch candidates

**`extract_pi2_features(var, model)`**
- Called after π1 picks x* at each node
- Returns 9-dim numpy float32 vector: fractionality, reduced cost, objective coefficient, constraint coupling statistics of x*
- These are standard LP statistics — available in any LP-based solver

**`get_prenorm_stats(graphs)`**
- Computes mean/std from training graphs for prenorm layer initialisation
- Returns dict including `sample_graphs` key (used for forward-pass hook initialisation of internal conv prenorms)

**Key API fix**: `Column.getRedcost()` does not exist in PySCIPOpt 6.x — use `var.getRedcost()` with try/except fallback to `model.getVarRedcost(var)`.

---

### `data/instance_generator.py`

Generates synthetic MIP benchmark instances for 4 problem classes:
- `setcover` — set covering: minimize cost s.t. each element covered
- `auction` — combinatorial auction: maximize revenue
- `facility` — capacitated facility location
- `indset` — maximum independent set on random graphs

Each instance written as `.lp` file. **Reusable across all training runs** — generator skips existing files.

Storage: `instances/{problem}/{difficulty}/{problem}_{difficulty}_{idx:05d}.lp`

---

### `models/gcn.py`

The π1 GCN. **Identical architecture to Branch Ranking / Gasse et al.**

**`PrenormLayer`** — fixed affine normalisation `clip((x - β) / σ, -10, 10)`. Fitted once from training data, then frozen. Critical for stable training (see below).

**`BipartiteConvLayer`** — one round of bipartite message passing:
- Pass 1: variables → constraints (sum aggregation, not mean)
- Pass 2: updated constraints → variables
- Each pass uses two 2-layer MLPs (gC, gV for messages; fC, fV for updates)
- Internal `prenorm_c` and `prenorm_v` normalise aggregation sums (scale ~56 for setcover)

**`BranchingGCN`** — full model:
- Prenorm → embed MLPs → N bipartite conv layers → `output_mlp: Linear(64→64)→ReLU→Linear(64→1)`
- `forward()` returns `(logits, var_embeddings)` where logits shape = (n_candidates,)

**`initialize_prenorms(stats)`** — MUST be called before training. Uses forward-pass hooks to fit internal prenorm layers from actual aggregation statistics. Without this, aggregation sums (~56× scale) dominate gradients and training fails.

**`build_gcn(cfg)`** — factory function.

---

### `models/node_mlp.py`

The π2 MLP. **New component, 880 parameters total.**

**`NodeChildMLP`** — `Linear(9,32)→ReLU→Linear(32,16)→ReLU→Linear(16,1)`. Outputs raw logit (no sigmoid — sigmoid applied in loss).

**`predict(feat_np)`** — inference method. Returns `True` = prefer LEFT child. Uses `>= 0.5` threshold (default LEFT on tie when logit=0).

**`NodeChildTrainer`** — trains with BCEWithLogitsLoss. Has crash-safe checkpointing: saves `*_latest.pt` every 10 epochs, `*_best.pt` on val loss improvement. Auto-resumes on restart.

---

### `training/data_collector.py`

The most important file. Implements Branch Ranking's **hybrid search** data collection.

**`NodeSample`** dataclass — what gets stored per branching decision:
```
π1 fields (unchanged from Branch Ranking):
  state_graph         bipartite graph features dict
  action_col_idx      column index of variable x
  trajectory_return   R_t(x) = -min(n_L, n_R)
  sb_score            SB proxy score (SB samples only)
  is_long_term        True if top-p% return
  is_short_term       True if best SB score at node
  reward              0.0 or 1.0

π2 fields (new — stored for ALL K explored variables):
  n_left              B&B nodes in left sub-solve
  n_right             B&B nodes in right sub-solve
  y_LR                1=prefer L, 0=prefer R, None=tie
  feat_pi2            9-dim feature vector of variable x
```

**`DataCollectionBranchRule`** — SCIP Branchrule plugin:
1. Extract bipartite graph G_t
2. Compute SB proxy score for all candidates; store best → D_SB
3. Sample K variables uniformly from candidates
4. For each: run 2 sub-solves (left branch, right branch) → get n_L, n_R
5. Compute R_t(x) = -min(n_L, n_R) and y_LR for ALL K variables
6. Commit best (highest R_t) to real tree
7. `branchVar(x*)` — SCIP creates L and R children

**`solve_subproblem()`** — creates fresh SCIP model, applies tight bounds + one extra branch, calls `m.optimize()`. SCIP runs internally with relpcost + hybridestim. Returns node count.

**`collect_dataset()`** — crash-safe: saves `_partial.pkl` every 5 instances. On restart detects partial file and resumes.

> **Key insight**: n_L and n_R are already computed for ALL K variables (they define R_t(x) = -min(n_L, n_R)). Storing y_LR for all K costs nothing extra and gives ~K× more π2 training data.

---

### `training/reward_assigner.py`

Post-processing step run after data collection. **Does not touch SCIP.**

**`assign_long_term_rewards(node_groups, top_p=0.10)`** — for each group of K samples at one node, label top-10% by trajectory_return as `reward=1`. With K=30: exactly 3 samples per node get r=1.

**`assign_short_term_rewards(samples)`** — labels each SB sample `reward=1` (already one per node).

**`build_training_dataset(lt_flat, sb_samples, h)`** — mixes long-term and short-term promising samples with ratio h. Returns (samples, graphs, rewards) for π1 training.

**`build_pi2_dataset(lt_flat, sb_samples)`** — extracts π2 training data. Uses **all** long-term samples with valid y_LR (not filtered by reward=1). L/R labels are meaningful for any variable regardless of trajectory quality.

**`compute_weighted_ce_loss(logits_list, targets, rewards)`** — π1 training loss: -Σ r·log π(a|s). Only reward=1 samples contribute gradient.

---

### `training/trainer.py`

**`initialize_prenorms(gcn, graphs)`** — fits all prenorm layers before π1 training. Calls `gcn.initialize_prenorms(stats)` which uses forward-pass hooks.

**`GCNTrainer`** — trains π1:
- Adam optimiser, ReduceLROnPlateau scheduler
- Crash-safe: `gcn_latest.pt` every 10 epochs, `gcn_best.pt` on val loss improvement
- Auto-resumes from `gcn_latest.pt` on restart

**`train_pi2(feats, labels, checkpoint_path, device)`** — trains π2:
- Normalises features, saves `*_norm.npy` alongside checkpoint
- Returns (model, mean, std)

---

### `branching/branch_rule.py`

Inference-time SCIP Branchrule plugin. **Joint π1 + π2 decision at each node:**

1. `extract_bipartite_graph()` → G_t
2. GCN forward pass → `(logits, var_emb)` → `x* = argmax(logits)`
3. `extract_pi2_features(x*, model)` → normalise → `π2.predict()` → prefer L or R
4. `model.branchVar(x*)`
5. `node_selector.preferred_child = chosen_child.getNumber()`

If π2 is None (ablation mode): defaults to prefer_left=True (left always first).

Fallback if GCN fails: most fractional variable.

---

### `node_selection/node_selector.py`

**`BCNodeSelector`** — SCIP Nodesel plugin. Single responsibility:

```
If preferred_child is set:
    Search getChildren() for matching node number
    If found: return it immediately (one-shot, then clear)
    If not found (child was pruned): clear and fall through
Fall through: return model.getBestLeaf()  ← SCIP handles the rest
```

π2 **never scores other open nodes**. Its action space is always exactly {L, R} of the most recent branch. All backtracking and further navigation is SCIP's hybridestim.

---

### `train.py`

Main entry point. Six steps in sequence:

```
[1] Generate instances  →  instances/{problem}/easy/
[2] Collect offline data  →  collected_data/{problem}_data.pkl
[3] Assign rewards  (in memory)
[4] Train π1 GCN  →  checkpoints/{problem}_gcn_best.pt
[5] Build D_π2 from collected data
[6] Train π2 NodeChildMLP  →  checkpoints/{problem}_pi2.pt
                             checkpoints/{problem}_pi2_norm.npy
```

Flags: `--skip-generate`, `--skip-collect`, `--skip-gcn`, `--skip-pi2`, `--device`

---

### `evaluate.py`

Runs 4-way comparison on test instances:
1. SCIP default (relpcost + hybridestim)
2. π1 only (GCN + hybridestim)
3. π1 + SCIP UCT (ablation)
4. π1 + π2 (BCNodeSelector) ← proposed

For each policy × instance × seed: full SCIP solve, record time and nodes.
Computes SGM, win rate (policy 4 vs 2), Wilcoxon p-value, GO/NO GO.
Saves `results/{problem}_{difficulty}_results.json`.

---

### `smoke_test.py`

10-point sanity check. Run first, always. Tests: instance generation, feature extraction, GCN forward pass, NodeChildMLP predict, BCNodeSelector interface, data collection (K=1, 5s), reward assignment, π2 dataset build, 2-epoch training loop.

---

## Data Flow Summary

```
instances/*.lp
    ↓  (DataCollectionBranchRule — SCIP runs hybrid search)
collected_data/{problem}_data.pkl
  └─ long_term_groups: List[List[NodeSample]]  ← K samples per node
  └─ sb_samples: List[NodeSample]              ← 1 sample per node
    ↓  (reward_assigner.py — in memory, no SCIP)
samples with reward ∈ {0,1}  +  D_π2 with y_LR labels
    ↓  (trainer.py)
checkpoints/{problem}_gcn_best.pt     ← π1 weights
checkpoints/{problem}_pi2.pt          ← π2 weights
checkpoints/{problem}_pi2_norm.npy    ← π2 feature normalisation
    ↓  (evaluate.py — SCIP + branch_rule + node_selector)
results/{problem}_{difficulty}_results.json
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| π2 uses 9 hand-crafted features, not GCN embedding | Independence: π2 trains/deploys without touching GCN |
| y_LR stored for ALL K variables, not just x* | ~K× more π2 training data at zero extra SCIP cost |
| BCNodeSelector returns immediately on preferred child | Structural masking: π2 never scores other open nodes |
| All LT samples used for D_π2 (not just reward=1) | L/R preference valid for any variable; maximises |D_π2| |
| `>= 0.5` threshold in predict() | Default LEFT on tie (logit=0, completely uncertain π2) |
| Prenorm fitted via forward-pass hooks | Aggregation sums scale ~56× vs embeddings ~1×; without this training diverges |
| No entropy coefficient | Offline RL — no online exploration needed |
| Policy deterministic at inference | argmax for π1; >= 0.5 threshold for π2 |

---

## What Is NOT in This Codebase

- **No true Strong Branching** — SB proxy (fractionality²) used instead
- **No ecole dependency** — runs on Windows with SCIP plugins natively  
- **No entropy regularisation** — offline RL does not need it
- **No shared weights between π1 and π2** — fully independent parameters
- **No global open-node scoring** — π2 only controls {L, R} of fresh children
