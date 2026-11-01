# Code and Running Instructions

## 1. Requirements

The code uses Python, NumPy, PyTorch, and Matplotlib. All required data and trained checkpoints are included in the submission.

The recorded environment used Python 3.14.6, NumPy 2.5.2, PyTorch 2.13.0+cu130, and Matplotlib 3.11.1. Minimum package versions are listed in `requirements.txt`.

Open a terminal in the project folder containing `run.py`.

### Windows PowerShell

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

If activation is blocked, use `./.venv/Scripts/python.exe` instead of `python` in every command. For example:

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe check_search.py
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Check the Included Results

Run these commands in order:

```bash
python check_search.py
python verify_results.py
python report_results.py
```

- `check_search.py`: checks search determinism, candidate uniqueness, and budget accounting using synthetic fitness.
- `verify_results.py`: verifies saved files, replays searches, reloads checkpoints, and recomputes metrics. A successful run prints `"status": "passed"`.
- `report_results.py`: generates result summaries and comparison figures using the saved results. It requires successful verification and does not retrain models.

The default result directory is `results/development_v1/`. Generated files include:

```text
results/development_v1/
  verification.json
  results_table.csv
  RESULTS_ZH.md
  figures/
    audit_comparison.png
    audit_comparison.pdf
```

## 3. Run the Complete Experiment

Use a new output directory:

```bash
python run.py --stage all --output results/reproduction
python verify_results.py --output results/reproduction
python report_results.py --output results/reproduction
```

Alternatively, run fitting and evaluation separately:

```bash
python run.py --stage fit --output results/reproduction
python run.py --stage audit --output results/reproduction
python verify_results.py --output results/reproduction
python report_results.py --output results/reproduction
```

Use either workflow for a given new directory. Keep the same `--output` path throughout that workflow.

- `fit`: prepares data, runs configuration searches, trains final models, and freezes selected checkpoints.
- `audit`: evaluates the frozen models and saves predictions and metrics.
- `all`: runs both stages.

Completed output directories are protected against retraining. For another run, choose a new path, such as `results/reproduction_02`.

### Device Selection

The default is `--device auto`, which uses an available GPU and otherwise the CPU. To choose explicitly:

```bash
python run.py --stage all --output results/reproduction_cpu --device cpu
python run.py --stage all --output results/reproduction_gpu --device cuda
```

These are alternative runs. CUDA execution requires a compatible GPU-enabled PyTorch installation. For separate stages, use the same device setting for fitting and evaluation.

`verify_results.py` selects its device automatically and has no `--device` argument. Its numerical checks are strict; different hardware or package versions can cause verification differences.

## 4. Source Code Structure

```text
run.py                         Main training and evaluation entry point
check_search.py                Search-logic checks
verify_results.py              Checkpoint and result verification
report_results.py              Result summaries and comparison plots
requirements.txt              Python dependencies
data/                         Included telemetry and split files
src/
  energy_config/
    core.py                   Search space, original searches, and loss
    refined_search.py         Seed verification and mutation variant
    data.py                   Data windows and preprocessing
    metrics.py                Power and energy metrics
    training.py               Final training and early stopping
    progressive.py            Search-stage training and continuation
  world_model/
    data.py                   Telemetry feature construction
    models/rssm.py            Recurrent state-space model
results/development_v1/        Included checkpoints and saved results
```

Training settings are defined in `run.py`, `src/energy_config/core.py`, `refined_search.py`, and `training.py`. They are not exposed as command-line options. Modified code must use a fresh output directory because existing runs check frozen source hashes.

Within a result directory, `search_training/` and `full_training/` contain checkpoints and training histories; `search_runs/` contains search traces. Predictions and metrics are saved as `audit_predictions.npz`, `audit_metrics.csv`, `audit_per_flight.csv`, and `audit_summary.json`.

## 5. Troubleshooting

- **Missing module:** run `python -m pip install -r requirements.txt` using the same interpreter used to run the scripts.
- **Frozen output protected:** select a new `--output` directory.
- **Audit already exists:** evaluation has already finished; run `verify_results.py` to check it.
- **Missing summary or verification file:** run fitting, audit, verification, and report generation in that order.
- **Hash mismatch:** source files or frozen artifacts differ from those recorded for that run. Use unchanged files to verify the included results.
- **Help:** run `python run.py --help`, `python verify_results.py --help`, or `python report_results.py --help`.
