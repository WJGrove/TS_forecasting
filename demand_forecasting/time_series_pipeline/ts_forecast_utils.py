import pandas as pd

def compute_wape(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    group_col: str,
    date_col: str,
    target_col: str,
    forecast_col: str,
) -> float:
    """
    Compute WAPE (Weighted Absolute Percentage Error) across all series and dates.

    WAPE = sum(|y - y_hat|) / sum(y)

    Both dataframes must have (group_col, date_col) keys.
    This computes a *global* WAPE (not per-series), which is usually what you
    want for panel-level model comparison.
    """
    # Keep only the columns we need, and avoid accidental column name collisions.
    actual_trim = df_actual[[group_col, date_col, target_col]].copy()
    forecast_trim = df_forecast[[group_col, date_col, forecast_col]].copy()

    merged = actual_trim.merge(
        forecast_trim,
        on=[group_col, date_col],
        how="inner",
        suffixes=("_actual", "_forecast"),
    )

    if merged.empty:
        raise ValueError(
            "No overlapping rows between actuals and forecasts; "
            "cannot compute WAPE."
        )

    abs_err = (
        merged[f"{target_col}_actual"] - merged[f"{target_col}_forecast"]
    ).abs()
    total_abs_err = abs_err.sum()
    total_actual = merged[f"{target_col}_actual"].abs().sum()

    if total_actual == 0:
        # Degenerate case: no volume at all → WAPE is undefined in a strict sense.
        # Returning NaN here is safer than dividing by zero.
        return float("nan")

    return float(total_abs_err / total_actual)


# Evaluation layer (metrics + interpretation)
# we'll add compute_mape, compute_rmse, and compute_mae later

# Option A – metrics only on modeled series
    # Compute WAPE/MAE/etc. only on the subset you actually forecasted.
    # Then separately report:

    # short_series_volume_pct

    # % of total test-set volume covered by modeled series.

# Option B – metrics on full panel (uglier but “real-world”)
    # If you set unmodeled short-series forecasts to zero, they’ll blow up WAPE/WMAE when short-series volume is big. This is “truthful” from a total-system point of view, but can obscure how well your model works where you actually applied it.

# In practice, I’d do both:

    # Panel metrics (all series) → operational view.

    # Modeled metrics (long + maybe some short strategy) → model-quality view.
    # And explicitly track “short-series volume % in test window” as a diagnostic.