from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .ts_forecast_utils import (
    compute_wape,
    compute_mae,
    compute_rmse,
    compute_mape,
)


@dataclass
class ForecastEvalConfig:
    """
    Configuration for forecast evaluation.

    This is intentionally decoupled from TSForecastConfig so that:
    - You can evaluate different forecast variants (different cols).
    - You can reuse the same evaluation logic for future models.

    The default names mirror your current pipeline, but everything is
    overrideable.
    """

    group_col: str = "time_series_id"
    date_col: str = "ds"
    target_col: str = "y_clean_int"
    forecast_col: str = "y_hat"

    # Optional: baseline forecast column for comparison
    baseline_col: Optional[str] = None

    # Optional: flag that marks short series (for “modeled vs full” logic)
    short_flag_col: str = "is_short_series"

    # Optional: forecast interval columns for coverage checks
    forecast_lower_col: Optional[str] = "y_hat_lower"
    forecast_upper_col: Optional[str] = "y_hat_upper"


def build_aligned_panel(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    cfg: ForecastEvalConfig,
) -> pd.DataFrame:
    """
    Align actuals and forecasts on (group_col, date_col).

    Returns a single DataFrame with at least:
    - cfg.group_col
    - cfg.date_col
    - cfg.target_col            (actuals)
    - cfg.forecast_col          (model forecast)
    - cfg.baseline_col          (optional baseline)
    - cfg.short_flag_col        (if present in actuals or forecasts)
    - cfg.forecast_lower_col    (if present in forecasts)
    - cfg.forecast_upper_col    (if present in forecasts)

    This is the core "joined" panel that all other evaluation functions use.
    """
    g = cfg.group_col
    d = cfg.date_col

    required_actual = {g, d, cfg.target_col}
    missing_actual = required_actual - set(df_actual.columns)
    if missing_actual:
        raise ValueError(
            f"df_actual is missing required columns: {missing_actual}. "
            f"Got columns: {list(df_actual.columns)}"
        )

    required_forecast = {g, d, cfg.forecast_col}
    missing_forecast = required_forecast - set(df_forecast.columns)
    if missing_forecast:
        raise ValueError(
            f"df_forecast is missing required columns: {missing_forecast}. "
            f"Got columns: {list(df_forecast.columns)}"
        )

    # Build the actual side (we keep only what we need)
    actual_cols = [g, d, cfg.target_col]
    if cfg.short_flag_col in df_actual.columns:
        actual_cols.append(cfg.short_flag_col)

    df_a = df_actual[actual_cols].copy()

    # Build the forecast side
    forecast_cols = [g, d, cfg.forecast_col]
    if cfg.baseline_col and cfg.baseline_col in df_forecast.columns:
        forecast_cols.append(cfg.baseline_col)
    if cfg.forecast_lower_col and cfg.forecast_lower_col in df_forecast.columns:
        forecast_cols.append(cfg.forecast_lower_col)
    if cfg.forecast_upper_col and cfg.forecast_upper_col in df_forecast.columns:
        forecast_cols.append(cfg.forecast_upper_col)

    df_f = df_forecast[forecast_cols].copy()

    # Align on (group, date)
    aligned = df_a.merge(df_f, on=[g, d], how="inner")

    if aligned.empty:
        raise ValueError(
            "No overlapping rows between actuals and forecasts after join on "
            f"({g}, {d}); cannot evaluate."
        )

    # Ensure datetime type for date_col
    aligned[d] = pd.to_datetime(aligned[d])

    return aligned


def compute_panel_metrics(
    aligned: pd.DataFrame,
    cfg: ForecastEvalConfig,
    *,
    forecast_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute panel-level metrics (WAPE, MAE, RMSE, MAPE) using the joined
    aligned DataFrame.

    Parameters
    ----------
    aligned : pd.DataFrame
        Output of build_aligned_panel(...).
    forecast_col : str, optional
        Which forecast column to evaluate. Defaults to cfg.forecast_col.

    Returns
    -------
    dict
        {
          "wape": float,
          "mae": float,
          "rmse": float,
          "mape": float,
          "n_rows": int,
          "n_series": int,
        }
    """
    g = cfg.group_col
    d = cfg.date_col
    y = cfg.target_col
    f = forecast_col or cfg.forecast_col

    if f not in aligned.columns:
        raise ValueError(
            f"Forecast column '{f}' not found in aligned columns: {list(aligned.columns)}"
        )

    df_a = aligned[[g, d, y]].copy()
    df_f = aligned[[g, d, f]].copy()

    metrics = {
        "wape": compute_wape(df_a, df_f, g, d, y, f),
        "mae": compute_mae(df_a, df_f, g, d, y, f),
        "rmse": compute_rmse(df_a, df_f, g, d, y, f),
        "mape": compute_mape(df_a, df_f, g, d, y, f),
        "n_rows": float(len(aligned)),
        "n_series": float(aligned[g].nunique()),
    }
    return metrics


def compute_series_level_metrics(
    aligned: pd.DataFrame,
    cfg: ForecastEvalConfig,
    *,
    forecast_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute per-series error metrics from the aligned panel.

    Each row of the returned DataFrame corresponds to one series (group_col)
    and contains:

    - group_col
    - n_obs
    - series_volume   = sum(|y|)
    - wape            = sum(|y - y_hat|) / sum(|y|)
    - mae             = mean(|y - y_hat|)
    - rmse            = sqrt(mean((y - y_hat)^2))
    - mape            = mean(|y - y_hat| / |y|) over non-zero y
    - is_short_series (if cfg.short_flag_col present)
    """
    g = cfg.group_col
    y_col = cfg.target_col
    f = forecast_col or cfg.forecast_col

    if f not in aligned.columns:
        raise ValueError(
            f"Forecast column '{f}' not found in aligned columns: {list(aligned.columns)}"
        )

    rows = []

    for series_id, pdf in aligned.groupby(g):
        y = pdf[y_col].astype("float64")
        y_hat = pdf[f].astype("float64")

        # Basic stats
        n_obs = len(pdf)
        series_volume = y.abs().sum()

        if n_obs == 0:
            # Should not happen in practice, but guard anyway
            continue

        abs_err = (y - y_hat).abs()
        sq_err = (y - y_hat) ** 2

        # WAPE
        denom = y.abs().sum()
        if denom == 0:
            wape = np.nan
        else:
            wape = abs_err.sum() / denom

        # MAE
        mae = abs_err.mean()

        # RMSE
        rmse = np.sqrt(sq_err.mean())

        # MAPE (only where |y| > 0)
        mask = y != 0
        if mask.any():
            mape = (abs_err[mask] / y[mask].abs()).mean()
        else:
            mape = np.nan

        row = {
            g: series_id,
            "n_obs": float(n_obs),
            "series_volume": float(series_volume),
            "wape": float(wape) if np.isfinite(wape) else float("nan"),
            "mae": float(mae) if np.isfinite(mae) else float("nan"),
            "rmse": float(rmse) if np.isfinite(rmse) else float("nan"),
            "mape": float(mape) if np.isfinite(mape) else float("nan"),
        }

        # If short_flag_col exists, attach a representative value
        if cfg.short_flag_col in aligned.columns:
            # Use the most common value in this series
            series_flag = pdf[cfg.short_flag_col].mode(dropna=True)
            if not series_flag.empty:
                row[cfg.short_flag_col] = bool(series_flag.iloc[0])
            else:
                row[cfg.short_flag_col] = None

        rows.append(row)

    return pd.DataFrame(rows)


def compute_dual_panel_metrics(
    aligned: pd.DataFrame,
    cfg: ForecastEvalConfig,
    *,
    forecast_col: Optional[str] = None,
    modeled_mask: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Compute both:

    - Panel metrics over the FULL aligned panel (operational view).
    - Panel metrics over a subset of rows considered "modeled" (model-quality view).

    If modeled_mask is None, we fallback to:
        modeled_mask = (aligned[cfg.short_flag_col] == False)
    when that column is available. Otherwise, all rows are treated as modeled.

    Returns a nested dict:
    {
      "panel":   {...metrics...},
      "modeled": {...metrics...},
      "modeled_volume_pct": float,  # fraction of total |y| in modeled subset
    }
    """
    y_col = cfg.target_col
    f = forecast_col or cfg.forecast_col

    # Full-panel metrics
    panel_metrics = compute_panel_metrics(aligned, cfg, forecast_col=f)

    # Default modeled_mask if not provided
    if modeled_mask is None:
        if cfg.short_flag_col in aligned.columns:
            modeled_mask = aligned[cfg.short_flag_col] == False
        else:
            modeled_mask = pd.Series(True, index=aligned.index)

    modeled = aligned[modeled_mask].copy()

    if modeled.empty:
        modeled_metrics: Dict[str, float] = {
            "wape": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "n_rows": 0.0,
            "n_series": 0.0,
        }
        modeled_volume_pct = float("nan")
    else:
        modeled_metrics = compute_panel_metrics(modeled, cfg, forecast_col=f)

        total_vol = aligned[y_col].astype("float64").abs().sum()
        modeled_vol = modeled[y_col].astype("float64").abs().sum()
        if total_vol > 0:
            modeled_volume_pct = float(modeled_vol / total_vol)
        else:
            modeled_volume_pct = float("nan")

    return {
        "panel": panel_metrics,
        "modeled": modeled_metrics,
        "modeled_volume_pct": modeled_volume_pct,
    }


def compute_ci_coverage(
    aligned: pd.DataFrame,
    cfg: ForecastEvalConfig,
) -> Dict[str, float]:
    """
    Compute coverage statistics for forecast intervals:

    - inside_pct:   fraction of actuals within [lower, upper]
    - below_pct:    fraction of actuals < lower
    - above_pct:    fraction of actuals > upper
    - n_evaluated:  number of rows where all of (y, lower, upper) are finite

    This is useful both as a sanity check on CI plumbing and as a diagnostic
    for whether intervals are too tight or too wide.
    """
    lower_col = cfg.forecast_lower_col
    upper_col = cfg.forecast_upper_col

    if not lower_col or not upper_col:
        raise ValueError(
            "forecast_lower_col and forecast_upper_col must be set on ForecastEvalConfig "
            "to compute CI coverage."
        )

    for col in [cfg.target_col, lower_col, upper_col]:
        if col not in aligned.columns:
            raise ValueError(
                f"Column '{col}' not found in aligned DataFrame; cannot compute CI coverage."
            )

    y = aligned[cfg.target_col].astype("float64").to_numpy()
    lower = aligned[lower_col].astype("float64").to_numpy()
    upper = aligned[upper_col].astype("float64").to_numpy()

    # Only evaluate rows where all three are finite
    mask_valid = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    n = int(mask_valid.sum())

    if n == 0:
        return {
            "inside_pct": float("nan"),
            "below_pct": float("nan"),
            "above_pct": float("nan"),
            "n_evaluated": 0.0,
        }

    yv = y[mask_valid]
    lv = lower[mask_valid]
    uv = upper[mask_valid]

    inside = (yv >= lv) & (yv <= uv)
    below = yv < lv
    above = yv > uv

    inside_pct = float(inside.mean())
    below_pct = float(below.mean())
    above_pct = float(above.mean())

    return {
        "inside_pct": inside_pct,
        "below_pct": below_pct,
        "above_pct": above_pct,
        "n_evaluated": float(n),
    }
