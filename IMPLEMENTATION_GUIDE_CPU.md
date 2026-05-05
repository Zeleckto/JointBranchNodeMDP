# Implementation Guide for CPU Machines
### Reproducing JointBranchNodeMDP

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or 3.11 | |
| SCIP Optimizer | 10.x | **Install before pyscipopt** |
| PySCIPOpt | 6.x | Python wrapper for SCIP |
| PyTorch | Any recent CPU build | No GPU needed |
| NumPy | >= 1.24 | |
| SciPy | >= 1.10 | For Wilcoxon test in evaluation |

**Install SCIP first**: Download from [scipopt.org](https://www.scipopt.org/index.php?page=download) for your OS. Run the installer — it sets environment variables that pyscipopt needs.

---

## Setup

```bash
# 1. Clone repo
git clone https://github.com/Zeleckto/JointBranchNodeMDP
cd JointBranchNodeMDP

# 2. Create virtual environment
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install PyTorch (CPU build — no CUDA needed)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install numpy scipy pyscipopt

# 5. Verify SCIP works
python -c "from pyscipopt import Model; m = Model(); print('SCIP OK')"

# 6. Verify PyTorch (CPU is fine)
python -c "import torch; print('PyTorch:', torch.__version__)"
```

---

## Sanity Check First

```bash
python smoke_test.py
```

Expected output:
```
=== Smoke Test ===
  ✓ instance generation
  ✓ feature extraction (bipartite graph)
  ✓ GCN forward pass
  ✓ NodeChildMLP forward + predict
  ✓ BCNodeSelector interface
  ✓ data collection (1 instance, K=1, 5s limit)
  ✓ reward assignment
  ✓ π2 dataset build
  ✓ GCN + π2 training (2 epochs)

All checks passed. Pipeline is healthy.
```

If any check fails, fix before proceeding.

---

## config.py — Set This Before Running

Open `config.py` and set device to CPU:

```python
DEVICE = "cpu"   # change from "cuda" if needed; auto-detected but force here
```

For a fast end-to-end test (30-60 minutes total on CPU):

```python
K_EXPLORE         = 3      # sub-solves per node
SCIP_TIME_LIMIT   = 15     # seconds per sub-solve
GCN_MAX_EPOCHS    = 30
GCN_STOP_PATIENCE = 10
PI2_MAX_EPOCHS    = 20
```

For meaningful results (4-8 hours on CPU):

```python
K_EXPLORE         = 10
SCIP_TIME_LIMIT   = 60
GCN_MAX_EPOCHS    = 500
GCN_STOP_PATIENCE = 15
PI2_MAX_EPOCHS    = 50
```

---

## Running Training

```bash
# Full pipeline in one command (generates instances + collects data + trains π1 + trains π2)
python train.py --problem setcover --device cpu

# If instances already exist (skip regeneration):
python train.py --problem setcover --skip-generate --device cpu
```

### Step-by-step (for debugging):

```bash
# Step 1: Generate instances only (~2 min)
python train.py --problem setcover --skip-collect --skip-gcn --skip-pi2 --device cpu

# Step 2: Data collection only (slow — this is the bottleneck)
python train.py --problem setcover --skip-gcn --skip-pi2 --device cpu

# Step 3: Train π1 GCN (after data collected)
python train.py --problem setcover --skip-collect --skip-pi2 --device cpu

# Step 4: Train π2 (after π1 trained and data collected)
python train.py --problem setcover --skip-generate --skip-collect --skip-gcn --device cpu
```

### Crash recovery:

If anything is interrupted, **just rerun the same command**. Every step checkpoints:
- Data collection: resumes from last 5-instance checkpoint
- GCN training: resumes from last 10-epoch checkpoint  
- π2 training: resumes from last 10-epoch checkpoint

---

## Verifying Data Collection Worked

```bash
python -c "
import pickle
d = pickle.load(open('collected_data/setcover_data.pkl', 'rb'))
lt = d['long_term_groups']
sb = d['sb_samples']
print('LT groups:', len(lt))
print('SB samples:', len(sb))
pi2 = [s for g in lt for s in g if s.y_LR is not None]
print('π2 samples:', len(pi2))
import numpy as np
if pi2:
    y = [s.y_LR for s in pi2]
    print('y=1 rate:', f'{sum(y)/len(y):.2%}')
"
```

You need: LT groups > 0, SB samples > 0, π2 samples > 0.

---

## Running Evaluation

```bash
# Quick test (5 instances, 30s timeout — verify pipeline runs)
python evaluate.py --problem setcover --n-instances 5 --n-seeds 2 --time-limit 30 --device cpu

# Paper-scale (20 instances, all difficulties)
python evaluate.py --problem setcover --difficulty easy   --n-instances 20 --n-seeds 5 --time-limit 3600 --device cpu
python evaluate.py --problem setcover --difficulty medium --n-instances 20 --n-seeds 3 --time-limit 3600 --device cpu
python evaluate.py --problem setcover --difficulty hard   --n-instances 20 --n-seeds 3 --time-limit 3600 --device cpu
```

Results saved to `results/setcover_easy_results.json` etc.

---

## Viewing Results

```bash
python results_generator.py --problem setcover
```

---

## Reusing the Dataset

If you have access to a collected dataset (pkl files) shared via Google Drive:

1. Create the `collected_data/` directory
2. Place `setcover_data.pkl` (and other benchmarks) in it
3. Skip data collection entirely:

```bash
python train.py --problem setcover --skip-collect --device cpu
```

The instances directory is also reusable — place the `.lp` files in `instances/setcover/easy/` and run with `--skip-generate`.

---

## Time Estimates on CPU

| Phase | K=3, 15s | K=10, 60s | Notes |
|---|---|---|---|
| Instance generation | 5 min | 5 min | One-time, reusable |
| Data collection (setcover) | 30-60 min | 4-8 hr | CPU-bound, cannot use GPU |
| GCN training (π1) | 5-15 min | 30-60 min | Benefits from GPU but runs on CPU |
| π2 training | 2-5 min | 5-10 min | Fast regardless |
| Evaluation (easy, 5 inst) | 5-15 min | 20-40 min | CPU-bound |

> **Note**: GPU only accelerates GCN training and inference. Data collection (SCIP sub-solves) is always CPU-bound — SCIP is single-threaded by design.

---

## Parallelising Data Collection (4× speedup)

Open 4 terminals, activate venv in each, run:

```bash
# Terminal 1
python -c "
import config as cfg; cfg.K_EXPLORE=10; cfg.SCIP_TIME_LIMIT=60
import glob
from training.data_collector import collect_dataset
paths = sorted(glob.glob('instances/setcover/easy/*.lp'))[:13]
collect_dataset(paths, 'collected_data/setcover_data_p1.pkl')
"

# Terminal 2 (instances 13-25)
# Terminal 3 (instances 26-38)
# Terminal 4 (instances 39-49)
# (same pattern, different slice indices)
```

Then merge:
```bash
python -c "
import pickle
all_lt, all_sb = [], []
for i in range(1, 5):
    d = pickle.load(open(f'collected_data/setcover_data_p{i}.pkl','rb'))
    all_lt.extend(d['long_term_groups'])
    all_sb.extend(d['sb_samples'])
pickle.dump({'long_term_groups': all_lt, 'sb_samples': all_sb},
            open('collected_data/setcover_data.pkl','wb'))
print('Merged:', sum(len(g) for g in all_lt), 'LT samples')
"
```

---

## Common Issues

| Error | Fix |
|---|---|
| `SCIP not found` / pyscipopt import error | Install SCIP 10.x first, then `pip install pyscipopt` |
| `Column has no attribute getRedcost` | Already handled in feature_extractor.py via try/except |
| All eval policies show same SGM time | They hit the `--time-limit`. Increase it or use easier instances |
| `π2 samples: 0` | Very rare — all sub-solves tied. Increase K or check SCIP install |
| Collection stuck with 0% CPU | SCIP crashed silently. Kill, check instance files, rerun |

---

## Repo Structure

```
JointBranchNodeMDP/
├── config.py                   ← edit this to change settings
├── train.py                    ← main training entry point
├── evaluate.py                 ← 4-way comparison
├── smoke_test.py               ← run first
├── results_generator.py        ← print result tables
├── data/
│   ├── feature_extractor.py    ← LP → bipartite graph; π2 features
│   └── instance_generator.py  ← generate .lp files
├── models/
│   ├── gcn.py                  ← π1 GCN (Branch Ranking architecture)
│   └── node_mlp.py             ← π2 NodeChildMLP (880 params)
├── training/
│   ├── data_collector.py       ← hybrid search + L/R label collection
│   ├── reward_assigner.py      ← top-p% ranking + D_π2 builder
│   └── trainer.py              ← GCNTrainer + train_pi2()
├── branching/
│   └── branch_rule.py          ← SCIP Branchrule: π1 + π2 at inference
├── node_selection/
│   └── node_selector.py        ← SCIP Nodesel: preferred-child mechanism
└── utils/
    └── metrics.py              ← SGM, win_rate, Wilcoxon
```
