import pandas as pd
import numpy as np
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


def _align_actual_forecast(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    group_col: str,
    date_col: str,
    target_col: str,
    forecast_col: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Align actuals and forecasts on (group_col, date_col) and return
    (y, y_hat) as float64 Series.

    This is the common core for all panel-level metrics.
    """
    if group_col not in df_actual.columns or date_col not in df_actual.columns:
        raise ValueError(
            f"df_actual must contain '{group_col}' and '{date_col}'. "
            f"Got columns: {list(df_actual.columns)}"
        )
    if group_col not in df_forecast.columns or date_col not in df_forecast.columns:
        raise ValueError(
            f"df_forecast must contain '{group_col}' and '{date_col}'. "
            f"Got columns: {list(df_forecast.columns)}"
        )
    if target_col not in df_actual.columns:
        raise ValueError(
            f"target_col '{target_col}' not found in df_actual columns: {list(df_actual.columns)}"
        )
    if forecast_col not in df_forecast.columns:
        raise ValueError(
            f"forecast_col '{forecast_col}' not found in df_forecast columns: {list(df_forecast.columns)}"
        )

    actual_trim = df_actual[[group_col, date_col, target_col]].copy()
    forecast_trim = df_forecast[[group_col, date_col, forecast_col]].copy()

    merged = actual_trim.merge(
        forecast_trim,
        on=[group_col, date_col],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No overlapping rows between actuals and forecasts; "
            "cannot compute metrics."
        )

    y = merged[target_col].astype("float64")
    y_hat = merged[forecast_col].astype("float64")

    return y, y_hat

def compute_wape(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    group_col: str,
    date_col: str,
    target_col: str,
    forecast_col: str = "y_hat",
) -> float:
    """
    Compute WAPE (Weighted Absolute Percentage Error) across all series and dates.

    WAPE = sum(|y - y_hat|) / sum(|y|)

    This is a *global* panel-level metric (not per-series).
    """
    y, y_hat = _align_actual_forecast(
        df_actual=df_actual,
        df_forecast=df_forecast,
        group_col=group_col,
        date_col=date_col,
        target_col=target_col,
        forecast_col=forecast_col,
    )

    abs_err = (y - y_hat).abs()
    total_abs_err = abs_err.sum()
    total_actual = y.abs().sum()

    if total_actual == 0:
        # Degenerate case: no volume at all → WAPE undefined.
        return float("nan")

    return float(total_abs_err / total_actual)


def compute_mae(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    group_col: str,
    date_col: str,
    target_col: str,
    forecast_col: str = "y_hat",
) -> float:
    """
    Compute MAE (Mean Absolute Error) across all aligned rows.

    MAE = mean(|y - y_hat|)
    """
    y, y_hat = _align_actual_forecast(
        df_actual=df_actual,
        df_forecast=df_forecast,
        group_col=group_col,
        date_col=date_col,
        target_col=target_col,
        forecast_col=forecast_col,
    )

    abs_err = (y - y_hat).abs()
    return float(abs_err.mean())


def compute_rmse(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    group_col: str,
    date_col: str,
    target_col: str,
    forecast_col: str = "y_hat",
) -> float:
    """
    Compute RMSE (Root Mean Squared Error) across all aligned rows.

    RMSE = sqrt(mean((y - y_hat)^2))
    """
    y, y_hat = _align_actual_forecast(
        df_actual=df_actual,
        df_forecast=df_forecast,
        group_col=group_col,
        date_col=date_col,
        target_col=target_col,
        forecast_col=forecast_col,
    )

    sq_err = (y - y_hat) ** 2
    return float(np.sqrt(sq_err.mean()))


def compute_mape(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    group_col: str,
    date_col: str,
    target_col: str,
    forecast_col: str = "y_hat",
) -> float:
    """
    Compute MAPE (Mean Absolute Percentage Error) across all aligned rows.

    MAPE = mean( |y - y_hat| / |y| ) over rows where |y| > 0

    Returns a value in [0, +inf). Typically interpreted as a *fraction*;
    multiply by 100 for %.
    """
    y, y_hat = _align_actual_forecast(
        df_actual=df_actual,
        df_forecast=df_forecast,
        group_col=group_col,
        date_col=date_col,
        target_col=target_col,
        forecast_col=forecast_col,
    )

    # Avoid division by zero: only use rows with non-zero actuals
    mask = y != 0
    if not mask.any():
        return float("nan")

    pct_err = ((y[mask] - y_hat[mask]).abs() / y[mask].abs())
    return float(pct_err.mean())




# Evaluation layer (metrics + interpretation)
# add at least compute_mape, compute_rmse, and compute_mae functions here

# A – metrics only on modeled series - this is used for model development and tuning.
    # Compute WAPE/MAE/etc. only on the subset you actually forecasted.
    # Then separately report:

    # short_series_volume_pct

    # % of total test-set volume covered by modeled series.

# B – metrics on full panel (uglier but “real-world”)
    # Set unmodeled short-series forecasts to zero (pretty much everything should be predicted somehow), let them blow up WAPE/WMAE when short-series volume is big. From my experience in business, I think this is a must-report pretty much at all times - because overall forecast performance is the most important thing.

# We're going to do both:

    # Panel metrics (all series) → operational view.

    # Modeled metrics (long + maybe some short strategy) → model-quality view.
    # And explicitly track “short-series volume % in test window” as a diagnostic.