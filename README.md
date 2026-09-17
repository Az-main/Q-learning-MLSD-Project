# Inventory Restocking Agent — DS-4491 MLSD Project

A tabular Q-learning agent that decides how much stock to reorder each day,
built as a reproducible DVC pipeline. Bonus: River online learning, ADWIN drift
detection with an automatic retraining response, and an optional Feast feature store.

## Problem and Dataset
A shop must decide every day how many units to order. Ordering costs money today,
leftover stock costs holding fees, and running out loses sales.

- **Dataset:** Kaggle *Store Item Demand Forecasting Challenge*, store 1 / item 1 —
  1,826 daily rows (2013–2017). No Kaggle account? `tools/get_data.py` generates a synthetic series.
- **Split (chronological, never shuffled):** train 2013–2015, stream 2016 (online stage),
  test 2017 (evaluation).
- **Simulated drift:** from 2016-07-01, stream demand is multiplied by `drift_factor` (2.0).
  The test year is left untouched.

## ML Model
Tabular **Q-learning** with ε-greedy exploration.
- **State (168):** stock bucket (8) × day of week (7) × demand trend (3)
- **Actions (4):** order 0, 10, 20 or 40 units (delivered overnight)
- **Reward:** `5·sold − 2·ordered − 0.3·leftover − 2·unmet`
- **Update:** `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]`, α=0.1, γ=0.95, 600 episodes

## Project Structure
```
├── data/raw/sales.csv(.dvc)   raw data (DVC-tracked)
├── data/processed/            prepare outputs (DVC-tracked)
├── src/                       prepare, env, agent, train, evaluate, online_drift, feast_serve, utils
├── feature_repo/              Feast definitions (optional)
├── tools/get_data.py          creates data/raw/sales.csv
├── tests/smoke_test.py        sanity checks
├── models/q_table.json        trained policy (DVC-tracked)
├── metrics/                   train/eval/drift JSON + plot CSVs
├── results/                   figures
├── params.yaml  dvc.yaml  dvc.lock  requirements.txt  .gitattributes
```

## DVC Pipeline
```
sales.csv.dvc → prepare → train → evaluate
                   └──────────┴──→ online_drift
```
| Stage | Input | Output |
|---|---|---|
| prepare | sales.csv, `prepare` params | train/stream/test parquet |
| train | train.parquet, `env`,`train` params | q_table.json, training curve |
| evaluate | q_table, test.parquet | eval.json, policy_comparison.png |
| online_drift | q_table, stream.parquet, `online`,`drift` params | drift.json, online_drift.png |

## How to Run
Requires **Python 3.11** (Windows / PowerShell shown).
```powershell
git clone https://github.com/Az-main/Q-learning-MLSD-Project.git
cd Q-learning-MLSD-Project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
dvc pull          # fetch data + model from the DVC remote
dvc status        # everything should be up to date
dvc repro         # rebuilds only what changed
dvc dag
dvc metrics show
dvc plots show    # then open dvc_plots/index.html
```
The DVC remote is a local folder (`D:\dvcstore`). Without access to it, create the data
yourself and rebuild: `python tools/get_data.py`, then `dvc repro`.

## Results

| Policy (2017 test year) | Total reward | Service level | Stockout days | Avg leftover |
|---|---|---|---|---|
| **Q-learning** | **18,197** | 0.86 | 0.47 | 3.9 |
| Order recent average | 16,589 | 0.89 | 0.34 | 26.5 |
| Always order 20 | 15,258 | 0.87 | 0.38 | 27.8 |
| Random | 12,980 | 0.81 | 0.27 | 27.3 |

- The agent keeps the shelf lean (3.9 units left per night vs ~27), accepting small
  shortfalls because holding stock costs money.
- With γ = 0 the agent never orders: total reward −16,054, service level 0.2%.
- Drift: concept drift was detected 4 days after the demand jump. Average daily reward
  after drift was 51.5 with the retrain response and 13.7 without it.

![policy comparison](results/policy_comparison.png)
![online drift](results/online_drift.png)