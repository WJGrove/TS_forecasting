# TS-prep-vis-forecast

This repository is meant to be a jumping off point for time series forecasting projects: data ingestion + preprocessing, diagnostics, visualizations, forecasting pipelines, and evaluation utilities. Includes example jobs using the Rossmann sales dataset from Kaggle.

**Quick overview**
- **Purpose:** Provide reusable building blocks for time series projects (preprocessing, transformation, modeling, plotting, and evaluation).
- **Language:** Python 3.8+
- **Layout:** See the `src` package for modules and example jobs.

**Repository structure**
- 'src/' — main source folder
	- `src/config/` — project configuration and settings
	- `src/ts_forecasting/` — main package containing data, jobs, RAG tools, and time-series pipeline modules
		- `data/kaggle_rossmann/` — placeholder for Rossmann CSVs (dataset files are not included in this repository)
		- `jobs/` — example ingestion and preparation jobs (`rossmann_ingest_job.py`, `rossmann_prep_job.py`)
		- `time_series_pipeline/` — preprocessing, diagnostics, plotting, forecasting and evaluation helpers
		- `rag/` — retrieval-augmented generation helpers, chat CLI, and indexing utilities for developer assistance

Getting started

Create a virtual environment and install the project in editable mode:

```powershell
python -m venv .venv_tsf
.\.venv_tsf\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify imports and settings:

```powershell
python -c "import ts_forecasting, config; print('ok')"
python -m ts_forecasting.env_info
```

Dependencies are managed in `pyproject.toml`. Editable installs will create local build metadata like `forecasting.egg-info/`—it’s ignored by Git.

Running modules

With the editable install you do not need to set PYTHONPATH:

```powershell
# Example: Rossmann prep job
python -m ts_forecasting.jobs.rossmann_prep_job
```

(If you choose not to install in editable mode, set `PYTHONPATH=./src` before running modules.)

Configuration

Copy `.env.example` → `.env` and fill in any needed values.

Project settings live in `src/config/settings.py` (Pydantic). Typical fields include paths, logging level, and API keys.

Data

Dataset files are not tracked. For the Rossmann example, place:

```
src/ts_forecasting/data/kaggle_rossmann/
	train.csv
	test.csv
	store.csv
	sample_submission.csv
```

Artifacts like RAG indices (e.g., `.rag_index.pkl`) are generated locally and ignored.

What’s included

- Jobs: src/ts_forecasting/jobs/rossmann_ingest_job.py, rossmann_prep_job.py
- Pipeline utilities: preprocessing, diagnostics, plotting, forecasting, evaluation under time_series_pipeline/
- RAG helpers: developer-oriented code/doc search tools in rag/
- CLI sample: `ts_forecasting.env_info` prints selected settings for a quick health-check

Dev notes

Common local folders/files ignored by Git: virtualenvs, data, Spark warehouse, build metadata, caches, RAG indices.

VS Code (optional): set the interpreter to `.venv_tsf` and, for IntelliSense,

```json
// .vscode/settings.json
{
	"python.defaultInterpreterPath": "${workspaceFolder}\\.venv_tsf\\Scripts\\python.exe",
	"python.analysis.extraPaths": ["./src"]
}
```

Acceptance checks

Fresh venv → `python -m pip install -e .` succeeds.

```powershell
python -m ts_forecasting.env_info   # prints your environment/log level/DB URL summary
python -m ts_forecasting.jobs.rossmann_prep_job   # runs from the repo root without setting PYTHONPATH
```

Git status shows no tracked venvs, data files, .pkl artifacts, or egg-info metadata.

