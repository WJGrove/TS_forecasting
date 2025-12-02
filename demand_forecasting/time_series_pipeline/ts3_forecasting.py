import pandas as pd
from dataclasses import dataclass, field
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

from pyspark.sql import DataFrame as SparkDataFrame
from typing import Literal, Optional


TimeGranularity = Literal["week", "month"]
BaselineMethod = Literal["seasonal_naive"]  # can extend later
ShortSeriesStrategy = Literal["naive", "seasonal_naive", "comp_based"]

@dataclass
class TSForecastConfig:
    horizon: int = 26  # number of periods ahead
    time_granularity: TimeGranularity = "week"
    seasonal_period: int = 52
    min_history_for_exp_smoothing: int = 2 * 52  # e.g. 2 years of weekly data

    # train/test split configuration
    test_set_length: int = 26  # in periods; used for validation
    test_end_anchor: Literal["calendar", "max_ds"] = "max_ds" # this defines how we pick test set end date

    stats_alpha: float = 0.05 # probability of type 1 error ("false positive") in statistical tests. This gives 95% confidence level.
    smoothing_alpha: float | None = None  # smoothing param; if None, estimate

    target_col: str = "y_clean_int"
    transformed_target_col: str | None = "y_clean_int_transformed"
    transformation_constant_col: str | None = "series_lambda"
    group_col: str = "time_series_id"
    date_col: str = "ds"
    short_flag_col: str = "is_short_series"

    # configurable columns for exogenous variables
    holiday_indicator_cols: list[str] = field(default_factory=list)
    exogenous_cols: list[str] = field(default_factory=list)

    forecast_transformed_target: bool = False

    # short series/comparison group config
    short_series_strategy: ShortSeriesStrategy = "naive" # this can be naive, seasonal_naive, or comp_based; if comp_based, uses comp_group_cols    
    comp_group_default_yoy: float = 0.02  # default for groups with no history
    comp_group_cols: list[str] = field(default_factory=list)  # columns defining comparison groups (e.g., [parent_company, item_id, market])

    # evaluation/QC
    baseline_method: BaselineMethod = "seasonal_naive"
    fc_ci_outlier_threshold: float = 3.0 # to handle extreme outliers in forecast confidence intervals

    def __post_init__(self):
        # ensure positive values where needed
        if self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if self.seasonal_period <= 0:
            raise ValueError("seasonal_period must be a positive integer")
        if self.min_history_for_exp_smoothing <= 0:
            raise ValueError("min_history_for_exp_smoothing must be a positive integer")
        if not (0 < self.stats_alpha < 1):
            raise ValueError("stats_alpha must be between 0 and 1")
        if self.smoothing_alpha is not None and not (0 < self.smoothing_alpha < 1):
            raise ValueError("smoothing_alpha must be between 0 and 1 if specified")
        
        # validate string literals
        if self.time_granularity not in {"week", "month"}:
            raise ValueError("time_granularity must be 'week' or 'month'")
        if self.test_end_anchor not in {"calendar", "max_ds"}:
            raise ValueError("test_end_anchor must be 'calendar' or 'max_ds'")
        if self.short_series_strategy not in {"naive", "seasonal_naive", "comp_based"}:
            raise ValueError("short_series_strategy must be 'naive', 'seasonal_naive', or 'comp_based'")


class TSForecaster:
    """
    Orchestrates panel forecasting:
    - splits short vs "long" (non-short) series,
    - applies exponential smoothing for long series,
    - simpler fallback for short series,
    - optional baseline + metrics.
    """

    def __init__(self, config: TSForecastConfig) -> None:
        self.config = config

    # ---------- Public API ----------

    def forecast_panel_pandas(self, df_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full forecasting pipeline on a preprocessed panel in pandas form.

        Expected columns (by default):
        - config.group_col
        - config.date_col
        - config.target_col
        - config.short_flag_col

        Returns a DataFrame of forecasts with one row per (group, forecast_date).
        """
        c = self.config

        if not {c.group_col, c.date_col, c.target_col, c.short_flag_col}.issubset(
            df_panel.columns
        ):
            missing = {c.group_col, c.date_col, c.target_col, c.short_flag_col} - set(
                df_panel.columns
            )
            raise ValueError(f"Input panel is missing required columns: {missing}")

        # Split short vs non-short
        df_long = df_panel[df_panel[c.short_flag_col] == False].copy()
        df_short = df_panel[df_panel[c.short_flag_col] == True].copy()

        # Forecast both groups
        long_fcst = self._forecast_long_series_panel(df_long)
        short_fcst = self._forecast_short_series_panel(df_short)

        # Combine
        all_fcst = pd.concat([long_fcst, short_fcst], ignore_index=True)

        # attach baseline for later evaluation
        baseline = self._compute_baseline_panel(df_panel)
        # You can join / align baseline later when you do metrics.

        return all_fcst

    # ---------- Internal: panel-level ----------

    def _compute_baseline_panel(self, df_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a seasonal-naive baseline forecast for all series
        for comparison purposes.

        For now, this is just a stub: we’ll implement once the main
        forecasting path is in place.
        """
        # e.g., we will:
        # - for each series, take the last `seasonal_period` actuals,
        # - align them as the horizon forecasts.
        return pd.DataFrame()

    def _forecast_long_series_panel(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Loop over long series and forecast each one.
        For now, run sequentially; later we can add joblib for parallelism.
        """
        c = self.config
        results: list[pd.DataFrame] = []

        for series_id, pdf in df_long.groupby(c.group_col):
            fcst = self._forecast_long_one_series(series_id, pdf)
            results.append(fcst)

        if not results:
            return pd.DataFrame(columns=[c.group_col, c.date_col, c.target_col])

        return pd.concat(results, ignore_index=True)

    def _forecast_short_series_panel(self, df_short: pd.DataFrame) -> pd.DataFrame:
        """
        Loop over short series and apply a simpler fallback forecast.
        """
        c = self.config
        results: list[pd.DataFrame] = []

        for series_id, pdf in df_short.groupby(c.group_col):
            fcst = self._forecast_short_one_series(series_id, pdf)
            results.append(fcst)

        if not results:
            return pd.DataFrame(columns=[c.group_col, c.date_col, c.target_col])

        return pd.concat(results, ignore_index=True)

    # ---------- Internal: per-series hooks (we'll plug your old code here) ----------

    def _forecast_long_one_series(self, series_id: str, pdf: pd.DataFrame) -> pd.DataFrame:
    c = self.config

    pdf = pdf.sort_values(c.date_col)
    y_col = c.transformed_target_col if c.forecast_transformed_target and c.transformed_target_col in pdf.columns else c.target_col

    y = pdf[y_col].to_numpy(dtype="float64")

    # STL decomposition with configurable seasonal_period
    stl = STL(y, period=c.seasonal_period, seasonal=13, robust=True)
    result = stl.fit()
    seasonal, trend, resid = result.seasonal, result.trend, result.resid

    # ETS on trend + resid; optionally pass smoothing_alpha if not None
    model = ExponentialSmoothing(
        trend + resid,
        trend="additive",
        seasonal=None,
        initialization_method="estimated",
    )
    model_fit = model.fit(
        optimized=(c.smoothing_alpha is None),
        smoothing_level=c.smoothing_alpha,
    )

    # Forecast horizon periods in Box-Cox or original space (depending on y_col)
    fcst_core = model_fit.forecast(c.horizon)
    # Add last seasonal pattern back
    seasonal_tail = seasonal[-c.seasonal_period:]
    # etc...

    def _forecast_short_one_series(self, series_id: str, pdf: pd.DataFrame) -> pd.DataFrame:
    if self.config.short_series_strategy == "comp_based":
        return self._forecast_short_comp_based(series_id, pdf)
    elif self.config.short_series_strategy == "seasonal_naive":
        return self._forecast_short_seasonal_naive(series_id, pdf)
    else:
        return self._forecast_short_naive(series_id, pdf)


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

    Both dataframes should have (group_col, date_col) keys; this function will
    join them on those keys and compute a single global WAPE.
    """
    merged = df_actual[[group_col, date_col, target_col]].merge(
        df_forecast[[group_col, date_col, forecast_col]],
        on=[group_col, date_col],
        how="inner",
        suffixes=("_actual", "_forecast"),
    )

    if merged.empty:
        raise ValueError(
            "No overlapping rows between actuals and forecasts to compute WAPE."
        )

    num = (
        (merged[f"{target_col}_actual"] - merged[f"{target_col}_forecast"])
        .abs()
        .sum()
    )
    denom = merged[f"{target_col}_actual"].abs().sum()

    if denom == 0:
        # Degenerate case: no volume at all
        return float("nan")

    return float(num / denom)


def train_test_split_panel(
    df: pd.DataFrame,
    config: TSForecastConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the preprocessed panel into train and test sets by time.

    By default:
    - test set = last `test_set_length` periods before the max date in the data
    - train set = everything earlier.

    (This is slightly more general and less calendar-specific than your "last Sunday"
     approach, but we can add a "calendar mode" later if needed.)
    """
    c = config
    if c.date_col not in df.columns:
        raise ValueError(f"date_col '{c.date_col}' not found in panel.")

    # Sort by date just for sanity
    df_sorted = df.sort_values(c.date_col)

    max_date = df_sorted[c.date_col].max()
    # test set starts `test_set_length` periods before max_date
    # (for weekly/monthly panel this is just "last N rows in time")
    # For now, we can approximate by using ranks or rolling, but we’ll
    # implement it in code when we get there.

    ...
