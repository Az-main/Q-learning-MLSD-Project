# Build Guide — MLSD Inventory RL Project (Windows / PowerShell)

Work top to bottom. Steps 1–7 cover every **mandatory** PDF requirement.
Steps 8–9 are **bonus**. Don't start the bonus until step 7 is pushed.

---

## Step 1 — Environment (≈30 min)

```powershell
cd mlsd-inventory-rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"   # must point inside .venv
python -m pip install --upgrade pip
pip install -r requirements.txt
dvc --version
```

- Use Python **3.11**.
- If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- Use `Activate.ps1`, never `activate.bat`, inside PowerShell.

## Step 2 — Git + DVC init (≈15 min)  → `dvc init`

```powershell
git init
dvc init
git status              # see what dvc init created: .dvc/config, .dvc/.gitignore, .dvcignore
git add .
git commit -m "Project skeleton + dvc init"
```

## Step 3 — Data (≈20 min)  → `dvc add`

```powershell
# real data (Kaggle train.csv downloaded somewhere):
python tools/get_data.py --kaggle C:\path\to\train.csv
# OR synthetic data:
python tools/get_data.py

dvc add data/raw/sales.csv
type data\raw\sales.csv.dvc      # md5 hash, size, path
type data\raw\.gitignore         # DVC keeps the CSV out of Git
git add data/raw/sales.csv.dvc data/raw/.gitignore
git commit -m "Track raw data with DVC"
```

**Inspect:** open `.dvc\cache\files\md5\<first 2 chars of the hash>\` — your CSV is
stored there, renamed to its own hash. That is content-addressable storage.

## Step 4 — Remote (≈15 min)  → `dvc push` / `dvc pull`

```powershell
dvc remote add -d localremote D:\dvcstore     # use C:\dvcstore if you have no D: drive
dvc push
git add .dvc/config
git commit -m "Add DVC remote"
```

Test it:

```powershell
Remove-Item data\raw\sales.csv
dvc status          # reports the file as deleted
dvc pull            # file comes back from the remote
```

## Step 5 — First stage (≈30 min)  → `dvc repro`, `dvc status`

```powershell
dvc repro prepare
python tests/smoke_test.py
dvc status          # "Data and pipelines are up to date"
dvc repro prepare   # skipped — nothing changed
```

**Inspect:** open `dvc.lock`. It stores a hash for every dep, param and output of each stage.

## Step 6 — Full pipeline (≈1–2 h incl. reading the code)  → `dvc dag`

```powershell
dvc repro
dvc dag
dvc metrics show
dvc plots show      # open dvc_plots\index.html in a browser
```

Open `models\q_table.json` and look at a few entries (4 numbers = value of ordering 0/10/20/40).

```powershell
git add .
git commit -m "Complete DVC pipeline"
dvc push
```

## Step 7 — GitHub + README (≈1 h)

1. Create an empty repo on GitHub named `mlsd-inventory-rl`.
2. Push:
   ```powershell
   git branch -M main
   git remote add origin https://github.com/<you>/mlsd-inventory-rl.git
   git push -u origin main
   ```
3. Replace the numbers in the README "Results" table with yours (`dvc metrics show`).
4. Fresh-clone test in another folder:
   ```powershell
   git clone https://github.com/<you>/mlsd-inventory-rl.git fresh-test
   cd fresh-test
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   dvc pull
   dvc repro        # should say everything is up to date
   ```

✅ **All mandatory requirements are now done.**

## Step 8 — Bonus: River + drift (already in the pipeline, ≈1.5–2 h to understand)

Read `src/online_drift.py`, then run the comparison:

```powershell
# in params.yaml set   drift: respond: false
dvc status          # only online_drift is stale
dvc repro
dvc metrics diff    # reward after drift drops a lot
# set respond back to true
dvc repro
```

Screenshot `results\online_drift.png` in both modes.

## Step 9 — Bonus: Feast (optional, ≈1.5–2 h)

```powershell
pip install -r requirements-feast.txt
```

- If that fails, skip Feast. River + drift is enough bonus.
- If it works, open `dvc.yaml`, delete only the leading `#` on each line of the `feast_serve` block, then:

```powershell
dvc repro
type metrics\feast.json
cd feature_repo
feast feature-views list
cd ..
```

---

## Live demo script for the viva (≈7 min)

| # | Do | Say |
|---|---|---|
| 1 | `dvc dag` | "Four stages. Prepare feeds training and the online branch." |
| 2 | `dvc status` | "Up to date — DVC compared current hashes with dvc.lock." |
| 3 | set `gamma: 0.0`, `dvc status` | "DVC noticed the param change. Only train and the stages after it are stale." |
| 4 | `dvc repro` | "Prepare is skipped because it doesn't depend on gamma." |
| 5 | `dvc metrics diff` | "With gamma 0 the agent never orders, since ordering only pays off tomorrow." |
| 6 | set `gamma: 0.95`, `dvc repro` | "Train is restored from DVC's run-cache instead of retraining." |
| 7 | delete `data\raw\sales.csv`, `dvc status`, `dvc pull` | "Git keeps the pointer, DVC keeps the data." |
| 8 | open `results\online_drift.png` | "Demand doubled here, ADWIN caught it, retraining brought reward back." |

If something breaks live: `dvc repro --force`.

---

## Viva answers (PDF section 4.1)

**What does `dvc repro` do?**
It reads `dvc.yaml`, checks each stage's deps/params/outs against the hashes in `dvc.lock`,
and reruns only the stages that changed, plus everything downstream of them.

**Why DVC?**
Git can't handle data and model files well. DVC versions them by hash and stores the bytes in a cache/remote.
It also makes the whole pipeline reproducible with one command.

**What is `dvc.yaml`?**
The pipeline definition. It lists each stage's command, dependencies, params, outputs, metrics and plots.

**What is `params.yaml`?**
All hyperparameters in one place. DVC tracks the specific keys each stage lists,
so changing one reruns only the stages that use it.

**How does DVC know which stage to rerun?**
It compares MD5 hashes of the current deps, params and outs with those stored in `dvc.lock`.

**What happens if the dataset changes?**
`dvc status` shows `sales.csv` as modified. `dvc repro` reruns prepare and every stage after it.
Then `dvc add`/`git commit`/`dvc push` records the new version.

**What happens if a parameter changes?**
Only stages that list that parameter, and their downstream stages, rerun.
Example: `gamma` reruns train, evaluate and online_drift, but not prepare.

**Git vs DVC?**
Git tracks code and small text files, including the `.dvc` pointer files and `dvc.lock`.
DVC tracks large data and model files, storing them in `.dvc/cache` and the remote, addressed by hash.

**`dvc push` / `dvc pull`?**
Push uploads cached data and models to the remote storage. Pull downloads the versions referenced by the current Git commit.

**How can another person reproduce your experiment?**
`git clone`, then `pip install -r requirements.txt`, then `dvc pull`, then `dvc repro`.
Pinned versions, fixed seeds and the chronological split make the results repeatable.

## Bonus answers (PDF section 5)

**Online learning vs batch learning.**
Batch learning trains once on the full dataset. Online learning updates the model one sample at a time as data arrives.
Here, the Q-table and the River regressor both update every day of the stream.

**River.**
A Python library for streaming ML: `learn_one` / `predict_one` on one sample at a time.
We use `LinearRegression` as a demand forecaster and `ADWIN` as the drift detector.
Evaluation is "prequential": predict first, then learn.

**Data drift vs concept drift.**
- Data drift: the input distribution changes. We run ADWIN on the demand values.
- Concept drift: the input→output relationship changes, so the old model becomes wrong. We run ADWIN on the forecaster's error.

**Response to drift.**
Retrain the agent on the last 14 days, replayed 50 times in the simulator.
Evidence: after the drift, average daily reward is about 51 with the response vs about 14 without it.

**Feast.**
- The offline store holds full feature history (Parquet); `get_historical_features` gives point-in-time-correct training rows.
- The online store (SQLite) holds only the latest values for fast serving; `get_online_features` provides them.
- `materialize` copies offline → online.
- Purpose: the same feature definitions are used for training and serving, which avoids training/serving skew.

## Know these limitations (say them before you're asked)

- The environment is a simplified simulator (overnight delivery, fixed prices).
- The drift is simulated (demand ×2), so we know exactly when it happened and can check the detector.
- ADWIN also fires on natural seasonal changes; that is real data drift, not an error.
- Results vary a little with the random seed.
- The local DVC remote only exists on this PC. For other people you would use a shared remote such as DagsHub or S3.

---

## Time estimate

| Task | Hours |
|---|---|
| Steps 1–4: environment, Git/DVC, data, remote | 1.5–2 |
| Steps 5–6: run pipeline, read and understand each file | 2–3 |
| Step 7: README, GitHub, fresh-clone test | 1–1.5 |
| Step 8: River + drift understanding and demo | 1.5–2 |
| Step 9: Feast (optional) | 1.5–2 |
| Viva practice | 2 |
| **Total** | **~8–10 h without Feast, ~10–12 h with Feast** |
