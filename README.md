# TS-prep-vis-forecast

This repository is meant to be a jumping off point for time series forecasting projects: data ingestion + preprocessing, diagnostics, visualizations, forecasting pipelines, and evaluation utilities. Includes example jobs using the Rossmann sales dataset from Kaggle.

**Quick overview**
- **Purpose:** Provide reusable building blocks for time series projects (preprocessing, transformation, modeling, plotting, and evaluation).
- **Language:** Python 3.8+
- **Layout:** See the `src` package for modules and example jobs.

**Repository structure**
- `src/config/` — project configuration and settings
- `src/ts_forecasting/` — main package containing data, jobs, RAG tools, and time-series pipeline modules
	- `data/kaggle_rossmann/` — sample Rossmann dataset CSVs (train/test/store/sample_submission)
	- `jobs/` — example ingestion and preparation jobs (`rossmann_ingest_job.py`, `rossmann_prep_job.py`)
	- `time_series_pipeline/` — preprocessing, diagnostics, plotting, forecasting and evaluation helpers
	- `rag/` — retrieval-augmented generation helpers, chat CLI, and indexing utilities for developer assistance

Getting started
- Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
pip install -r requirements.txt
```

- Run from the repository root; ensure the project root is on `PYTHONPATH` when running modules. Example (PowerShell):

```powershell
$env:PYTHONPATH = '.'
python -m ts_forecasting.jobs.rossmann_prep_job
```

What’s included
- Data ingestion examples: `src/ts_forecasting/jobs/rossmann_ingest_job.py`
- Preprocessing pipeline: `src/ts_forecasting/time_series_pipeline/ts0_preprocessing.py` and helpers
- Visualizations & diagnostics: `ts1_diagnostics.py`, `ts2_plots.py`
- Forecasting & evaluation: `ts3_forecasting.py`, `ts4_forecast_evaluation.py`
- RAG helper: CLI and indexing utilities to query project docs and code snippets (`src/ts_forecasting/rag`)

Usage notes
- Data: Place your datasets under `src/ts_forecasting/data/` (subfolders per dataset). Example Rossmann CSVs are already present for experimentation.
- Config: Adjust runtime settings in `src/config/settings.py`.
- Notebooks & experimentation: Add notebook files at the repo root or a `notebooks/` folder and set `PYTHONPATH='.'` before running.

Next steps and TODOs
- Flesh out example notebooks demonstrating a full train→forecast→evaluate workflow.
- Add CI checks and tests for pipeline utilities.

