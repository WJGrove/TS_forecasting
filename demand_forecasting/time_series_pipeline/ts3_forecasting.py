import pandas as pd
from dataclasses import dataclass, field
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from datetime import timedelta
from typing import Literal, Optional, Tuple


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


def train_test_split_panel(
    df: SparkDataFrame,
    config: TSForecastConfig,
) -> tuple[SparkDataFrame, SparkDataFrame]:
    """
    Split a preprocessed Spark panel into train and test sets based on time.

    Current behavior:
    - Uses a *global* cutoff date based on the maximum date in the panel.
    - Test set = all rows with date_col in (test_start_date, max_date]
    - Train set = all rows with date_col <= test_start_date

    Where:
    - max_date is the maximum non-null value in config.date_col.
    - test_start_date is computed as:
        * week  : max_date - test_set_length * 1 week
        * month : max_date - test_set_length * 1 month

    This matches the spirit of your original notebook, but anchors to the data
    itself (max ds) instead of wall-clock "today" / "last Sunday".

    Parameters
    ----------
    df : SparkDataFrame
        Preprocessed panel with at least config.date_col.
    config : TSForecastConfig
        Forecast configuration (uses date_col, test_set_length, time_granularity).

    Returns
    -------
    (train_df, test_df) : tuple[SparkDataFrame, SparkDataFrame]
        Two DataFrames with the same schema as df.
    """
    c = config

    if c.date_col not in df.columns:
        raise ValueError(
            f"date_col '{c.date_col}' not found in DataFrame columns: {df.columns}"
        )

    # Get the maximum date from the panel
    max_date_row = df.agg(F.max(c.date_col).alias("max_date")).collect()[0]
    max_date = max_date_row["max_date"]

    if max_date is None:
        raise ValueError(
            f"All values in date_col '{c.date_col}' are null; cannot perform time-based split."
        )

    gran = c.time_granularity.lower()

    # Compute the start of the test window based on the configured granularity
    if gran == "week":
        # test_set_length is interpreted as number of weeks
        test_start_date = max_date - timedelta(weeks=c.test_set_length)
    elif gran == "month":
        # For monthly data, use relativedelta to subtract whole months.
        try:
            from dateutil.relativedelta import relativedelta
        except ImportError as e:
            raise ImportError(
                "dateutil is required for monthly train/test splits. "
                "Install via 'pip install python-dateutil'."
            ) from e

        test_start_date = max_date - relativedelta(months=c.test_set_length)
    else:
        # Should be prevented by TSPreprocessingConfig/TSForecastConfig validation,
        # but keep a defensive check in case of future changes.
        raise ValueError(
            f"Unsupported time_granularity '{c.time_granularity}'. "
            "Expected 'week' or 'month'."
        )

    # Build train/test filters
    # Test = dates strictly greater than test_start_date, up to and including max_date
    test_df = df.filter(
        (F.col(c.date_col) > F.lit(test_start_date))
        & (F.col(c.date_col) <= F.lit(max_date))
    )

    # Train = dates less than or equal to test_start_date
    train_df = df.filter(F.col(c.date_col) <= F.lit(test_start_date))

    return train_df, test_df



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
