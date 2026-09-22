# MLSD DVC Project - Progress and Remaining Plan

Last updated: 2026-09-22

## 1. Project Summary

This project is an inventory restocking system built for the DS-4491 Machine
Learning Systems Design course. A tabular Q-learning agent decides how many
units a shop should order each day. The complete workflow is managed by DVC so
that the data, model, parameters, metrics, plots, and pipeline can be reproduced.

Project repository:

- GitHub: `https://github.com/Az-main/Q-learning-MLSD-Project`
- DagsHub: `https://dagshub.com/Az-main/Q-learning-MLSD-Project`
- Original development folder: `D:\DVC Project`
- Friend's laptop clone: `D:\MLSD\Q-learning-MLSD-Project`

## 2. Dataset and Model

### Dataset

- Source: Kaggle Store Item Demand Forecasting Challenge
- Selected data: store 1 and item 1
- Size: 1,826 daily rows from 2013 through 2017
- Training period: 2013-2015
- Online stream period: 2016
- Test period: 2017
- A synthetic-data generator is also available in `tools/get_data.py`.

### Model

- Model type: tabular Q-learning
- State space: 168 states
- Actions: order 0, 10, 20, or 40 units
- Main output: `models/q_table.json`
- Evaluation compares Q-learning with simple ordering policies.

### Bonus Components Already Implemented

- River incremental learning
- ADWIN data-drift detection
- ADWIN concept-drift detection
- Automatic retraining response after drift
- Simulated demand drift in the 2016 stream

Feast files exist in the project, but the Feast stage is still optional and has
not been enabled as a completed feature.

## 3. DVC Pipeline

The project contains the required reproducible pipeline:

```text
data/raw/sales.csv.dvc
          |
          v
       prepare
       /     \
      v       v
    train  online_drift
      |
      v
   evaluate
```

Actual dependency behavior also connects the trained Q-table to the online
drift stage.

| Stage | Purpose | Main outputs |
|---|---|---|
| `prepare` | Cleans and splits the data chronologically | train, stream, and test Parquet files |
| `train` | Trains the Q-learning policy | Q-table, training metrics, and training plot |
| `evaluate` | Tests and compares policies | evaluation metrics and comparison plot |
| `online_drift` | Runs streaming learning and drift response | drift metrics and online-drift plot |

The pipeline can be inspected and reproduced with:

```powershell
dvc status
dvc dag
dvc repro
dvc metrics show
dvc plots show
```

## 4. Work Completed

### Environment and Repository Setup

- Created a Python 3.11 virtual environment.
- Installed the packages from `requirements.txt`.
- Initialized Git and DVC.
- Added `.gitignore`, `.dvcignore`, DVC configuration, and the project files.
- Pushed the complete code repository to GitHub.
- Added a README containing the problem, dataset, model, structure, pipeline,
  setup instructions, collaboration workflow, and results.

### Data Versioning

- Added `data/raw/sales.csv` to DVC.
- Confirmed that Git tracks the small `sales.csv.dvc` pointer instead of the
  large CSV file.
- Confirmed that DVC stores files by content hash in `.dvc/cache`.
- Created the original local DVC remote at `D:\dvcstore` as a backup.
- Connected the project to a shared DagsHub DVC remote.
- Made the DagsHub remote named `origin` the default DVC remote.
- Pushed all required DVC-tracked data and model artifacts to DagsHub.

Current remote arrangement:

```text
localremote -> D:\dvcstore
origin      -> DagsHub DVC storage (default)
```

### Pipeline and Results

- Ran the `prepare`, `train`, `evaluate`, and `online_drift` stages.
- Generated `dvc.lock` with hashes of dependencies, parameters, and outputs.
- Verified that unchanged stages are skipped by `dvc repro`.
- Ran the smoke test successfully.
- Generated the metrics and plots used in the README.

Current important results:

| Policy | Total reward | Service level | Stockout days | Average leftover |
|---|---:|---:|---:|---:|
| Q-learning | 18,197 | 0.86 | 0.47 | 3.9 |
| Recent-average order | 16,589 | 0.89 | 0.34 | 26.5 |
| Always order 20 | 15,258 | 0.87 | 0.38 | 27.8 |
| Random policy | 12,980 | 0.81 | 0.27 | 27.3 |

The drift experiment detected the simulated demand change and showed an
average daily reward after drift of about 51.5 with retraining, compared with
about 13.7 without the response.

### Shared Collaboration Setup

- Connected the GitHub repository to DagsHub.
- Added the friend as a collaborator on GitHub and DagsHub.
- Installed Python 3.11, Git, DVC, and the project dependencies on the friend's
  laptop.
- Cloned the GitHub repository on the friend's laptop.
- Configured the friend's own DagsHub username and access token locally.
- Confirmed that `.dvc/config.local` is ignored by Git.
- Successfully ran `dvc pull` on the friend's laptop.
- Confirmed that the dataset and model were downloaded on the laptop.
- Successfully ran `dvc status` and `python tests\smoke_test.py` on the laptop.
- Successfully created and pushed the test branch
  `collaboration/access-check`, proving that the friend has GitHub write access.

Access tokens are never stored in Git. Every collaborator must use their own
DagsHub token in their local `.dvc/config.local` file.

## 5. Course Requirement Status

| Requirement from the PDF | Status | Evidence |
|---|---|---|
| Data preparation stage | Complete | `prepare` in `dvc.yaml` |
| Model training stage | Complete | `train` in `dvc.yaml` |
| Model evaluation stage | Complete | `evaluate` in `dvc.yaml` |
| Suitable dataset and model | Complete | Kaggle demand data and Q-learning agent |
| DVC command knowledge | Implemented; viva practice remains | Commands are documented and tested |
| Clean GitHub repository | Complete | Shared GitHub repository is working |
| Complete README | Complete | `README.md` contains all required sections |
| Reproducibility on another machine | Complete | Fresh setup and DVC pull passed on the laptop |
| River online learning bonus | Complete | Used in `online_drift` |
| Drift detection bonus | Complete | ADWIN data and concept drift detection |
| Response to drift bonus | Complete | Automatic retraining response |
| Feast bonus | Not completed | Optional stage remains disabled |

All mandatory PDF requirements have been completed. The main remaining work is
to improve the evidence for the bonus experiment, optionally finish Feast, and
prepare the final demonstration.

## 6. Remaining Plan

### Task 1 - Preserve Both Drift Comparison Results

Status: next task

The response-enabled drift plot currently exists, but both modes should be
saved separately so the faculty can directly compare them.

Planned work:

1. Create a feature branch from an updated `main` branch.
2. Save the current response-enabled plot with a clear filename.
3. Change `drift.respond` in `params.yaml` from `true` to `false`.
4. Run the `online_drift` stage again.
5. Save the no-response plot with a different filename.
6. Record the metrics difference.
7. Restore `drift.respond: true` as the final project configuration.
8. Add both plots and a short comparison to the README.
9. Push the branch and merge it through a pull request.

Expected comparison:

- With response: post-drift daily reward is about 51.5.
- Without response: post-drift daily reward is about 13.7.

### Task 2 - Decide Whether to Complete Feast

Status: optional bonus

If time permits:

1. Install `requirements-feast.txt` in Python 3.11.
2. Enable only the existing `feast_serve` block in `dvc.yaml`.
3. Run the stage and verify `metrics/feast.json`.
4. Test offline and online feature retrieval.
5. Document what Feast does and why it is useful.

If Feast causes package or setup problems, it can be skipped. River and drift
detection already provide meaningful bonus work.

### Task 3 - Final Reproducibility Check

Run these checks on the original PC and the friend's laptop:

```powershell
git switch main
git pull
dvc pull
dvc status
dvc repro
python tests\smoke_test.py
git status
```

Expected final state:

- DVC reports that data and pipelines are up to date.
- The smoke test passes.
- Git reports a clean working tree.

### Task 4 - Prepare the Viva Demonstration

Practice explaining and running:

```powershell
dvc status
dvc repro
dvc dag
dvc metrics show
dvc push
dvc pull
```

Be ready to explain:

- The difference between Git and DVC
- The purpose of `dvc.yaml`, `dvc.lock`, and `params.yaml`
- How DVC decides which stage must rerun
- What happens when data or a parameter changes
- How another person reproduces the project
- Batch learning compared with online learning
- How River and ADWIN are used
- Data drift compared with concept drift
- How the system responds after drift is detected
- Project limitations and why the drift is simulated

### Task 5 - Final Presentation Material

- Capture clear screenshots of `dvc dag`, metrics, and both drift modes.
- Prepare a short architecture/pipeline slide.
- Show the GitHub repository and DagsHub data storage.
- Keep the live demonstration short and repeatable.
- Perform one complete practice run before the presentation.

## 7. Normal Collaboration Workflow

Before starting a task:

```powershell
git switch main
git pull
dvc pull
git switch -c username/short-task-name
```

After completing a task:

```powershell
dvc repro
dvc push
git status
git add .
git commit -m "Describe the change"
git push -u origin HEAD
```

Then open a GitHub pull request, review it, merge it into `main`, and delete the
completed branch. Avoid editing the same files on both computers at the same
time.

## 8. Current Overall Status

- Mandatory project requirements: complete
- GitHub and shared DagsHub setup: complete
- Reproduction on the friend's laptop: complete
- Multi-user Git/DVC access test: complete
- River and drift bonus implementation: complete
- Separate drift comparison evidence: remaining
- Feast bonus: optional and remaining
- Viva and presentation practice: remaining

The project is already in a reproducible and presentable state. The next best
step is to preserve the two drift experiment modes and document their
comparison before deciding whether to add Feast.
