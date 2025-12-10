import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F

def train_test_split_panel(
    df: SparkDataFrame,
    *,
    date_col: str,
    test_set_length: int,
    time_granularity: str = "week",
) -> tuple[SparkDataFrame, SparkDataFrame]:
    """
    Split a preprocessed Spark panel into train and test sets based on time.

    - Uses a *global* cutoff date based on the maximum date in the panel.
    - Test set = all rows with date_col in (test_start_date, max_date]
    - Train set = all rows with date_col <= test_start_date

    test_set_length is interpreted as a number of periods:
    - 'week'  -> weeks
    - 'month' -> months
    """

    if date_col not in df.columns:
        raise ValueError(
            f"date_col '{date_col}' not found in DataFrame columns: {df.columns}"
        )

    max_date_row = df.agg(F.max(date_col).alias("max_date")).collect()[0]
    max_date = max_date_row["max_date"]

    if max_date is None:
        raise ValueError(
            f"All values in date_col '{date_col}' are null; cannot perform time-based split."
        )

    gran = time_granularity.lower()

    if test_set_length <= 0:
        raise ValueError("test_set_length must be a positive integer")

    if gran == "week":
        from datetime import timedelta

        test_start_date = max_date - timedelta(weeks=test_set_length)
    elif gran == "month":
        try:
            from dateutil.relativedelta import relativedelta
        except ImportError as e:
            raise ImportError(
                "dateutil is required for monthly train/test splits. "
                "Install via 'pip install python-dateutil'."
            ) from e

        test_start_date = max_date - relativedelta(months=test_set_length)
    else:
        raise ValueError(
            f"Unsupported time_granularity '{time_granularity}'. "
            "Expected 'week' or 'month'."
        )

    test_df = df.filter(
        (F.col(date_col) > F.lit(test_start_date)) & (F.col(date_col) <= F.lit(max_date))
    )
    train_df = df.filter(F.col(date_col) <= F.lit(test_start_date))

    return train_df, test_df

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