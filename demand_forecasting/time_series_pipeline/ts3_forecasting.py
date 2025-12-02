import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

from dataclasses import dataclass, field
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from datetime import timedelta
from typing import Literal, Optional, Tuple

from ts_forecast_utils import train_test_split_panel

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

    def train_test_split(
        self, df_panel_spark: SparkDataFrame
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        """
        Split the input panel DataFrame into train and test sets
        based on the configuration.

        Returns (train_df, test_df) as Spark DataFrames.
        """
        c = self.config

        train_df, test_df = train_test_split_panel(
            df=df_panel_spark,
            date_col=c.date_col,
            time_granularity=c.time_granularity,
            test_set_length=c.test_set_length,
        )

        return train_df, test_df

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
        Forecast all 'long' series in a preprocessed panel (pandas).

        Assumes df_long contains at least:
        - config.group_col
        - config.date_col
        - config.target_col (and optionally transformed_target_col)

        Returns a DataFrame with one row per (group, forecast_date):
        - group_col
        - date_col (future dates)
        - 'y_hat' (point forecast)

        Short/insufficient series should generally have been filtered out
        *before* calling this function, but we still guard against edge cases.
        """
        c = self.config

        required_cols = {c.group_col, c.date_col, c.target_col}
        missing = required_cols - set(df_long.columns)
        if missing:
            raise ValueError(
                f"df_long is missing required columns for long-series forecasting: {missing}"
            )

        # Just in case someone passes an empty panel
        if df_long.empty:
            return pd.DataFrame(
                columns=[c.group_col, c.date_col, "y_hat"],
                dtype="float64",
            )

        results: list[pd.DataFrame] = []

        # Group by series and forecast each one
        for series_id, pdf in df_long.groupby(c.group_col):
            try:
                fcst = self._forecast_long_one_series(series_id, pdf)
            except Exception as exc:
                # Defensive: if a single series fails, we don't want the whole
                # pipeline to crash. You can log this or re-raise depending on
                # your appetite.
                print(
                    f"[WARN] Failed to forecast long series '{series_id}': {exc!r}. "
                    "Falling back to naive forecast for this series."
                )
                fcst = self._forecast_naive_one_series(series_id, pdf)

            results.append(fcst)

        return pd.concat(results, ignore_index=True)

    def _forecast_long_one_series(self, series_id: str, pdf: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast a single 'long' series using STL decomposition + ExponentialSmoothing.

        Steps:
        - Sort by date.
        - Choose the modeling target:
            * transformed_target_col if forecast_transformed_target=True and present
            * else target_col
        - Perform STL decomposition with seasonal_period from config.
        - Fit ETS on (trend + resid).
        - Forecast `horizon` steps ahead.
        - Add back the seasonal component by repeating the last seasonal period.
        - Generate future dates based on time_granularity (week or month).

        Returns a DataFrame with:
        - group_col
        - date_col (future dates)
        - 'y_hat' (point forecast in the same space as the modeling target)
        """
        c = self.config

        # Defensive: copy and sort
        pdf = pdf.copy()
        pdf = pdf.sort_values(c.date_col)

        if pdf.empty:
            raise ValueError(f"Series '{series_id}' has no data; cannot forecast.")

        # Decide which column to model on
        if (
            c.forecast_transformed_target
            and c.transformed_target_col is not None
            and c.transformed_target_col in pdf.columns
        ):
            y_col = c.transformed_target_col
        else:
            y_col = c.target_col

        if y_col not in pdf.columns:
            raise ValueError(
                f"Target column '{y_col}' not found for series '{series_id}'. "
                "Check TSForecastConfig.target_col / transformed_target_col."
            )

        # Extract the time series as a float64 numpy array
        y = pdf[y_col].astype("float64").to_numpy()

        # Basic validation: length and finite values
        n_obs = len(y)
        if n_obs < c.min_history_for_exp_smoothing:
            raise ValueError(
                f"Series '{series_id}' has length {n_obs}, which is less than "
                f"min_history_for_exp_smoothing={c.min_history_for_exp_smoothing}."
            )

        # If all values are NaN or non-finite, fall back
        if not np.any(np.isfinite(y)):
            raise ValueError(
                f"Series '{series_id}' has no finite values in '{y_col}'."
            )

        # If the series is constant (or almost), ETS isn't very meaningful;
        # fall back to naive constant forecast.
        if np.nanmax(y) - np.nanmin(y) < 1e-8:
            return self._forecast_naive_one_series(series_id, pdf)

        # STL decomposition
        # Note: STL expects a 1D array/Series; we use seasonal_period from config.
        stl = STL(y, period=c.seasonal_period, seasonal=13, robust=True)
        stl_res = stl.fit()

        seasonal = stl_res.seasonal
        trend = stl_res.trend
        resid = stl_res.resid

        # Deseasonalized series = trend + resid
        deseasonalized = trend + resid

        # ETS model on deseasonalized series
        # We allow smoothing_alpha to override the automatic optimization.
        model = ExponentialSmoothing(
            deseasonalized,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )

        if c.smoothing_alpha is None:
            # Let statsmodels choose the smoothing parameter
            model_fit = model.fit(optimized=True)
        else:
            model_fit = model.fit(
                optimized=False,
                smoothing_level=c.smoothing_alpha,
            )

        # Forecast on the deseasonalized scale
        h = c.horizon
        fcst_core = model_fit.forecast(h)

        # Build seasonal pattern for the forecast horizon by repeating the
        # last full seasonal period and truncating to horizon.
        season_len = c.seasonal_period
        if season_len <= 0:
            raise ValueError("seasonal_period must be positive.")

        # Use the last 'season_len' seasonal values as the template
        seasonal_tail = seasonal[-season_len:]
        # Tile enough times to cover the horizon
        reps = int(np.ceil(h / season_len))
        seasonal_future = np.tile(seasonal_tail, reps)[:h]

        y_hat = fcst_core + seasonal_future

        # Build future dates based on last observed date and time granularity
        last_date = pd.to_datetime(pdf[c.date_col].max())

        if c.time_granularity.lower() == "week":
            # Weekly: step by 1 week
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(weeks=1),
                periods=h,
                freq="W",  # weekly; you could tweak to "W-SUN" / "W-MON" if desired
            ).normalize()
        elif c.time_granularity.lower() == "month":
            # Monthly: step by 1 month, align on month start
            future_dates = pd.date_range(
                start=(last_date + pd.DateOffset(months=1)).replace(day=1),
                periods=h,
                freq="MS",  # Month Start
            )
        else:
            raise ValueError(
                f"Unsupported time_granularity '{c.time_granularity}'. "
                "Expected 'week' or 'month'."
            )

        # Assemble forecast DataFrame
        out = pd.DataFrame(
            {
                c.group_col: [series_id] * h,
                c.date_col: future_dates,
                "y_hat": y_hat,
            }
        )

        return out

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

    def _forecast_short_one_series(self, series_id: str, pdf: pd.DataFrame) -> pd.DataFrame:
        if self.config.short_series_strategy == "comp_based":
            return self._forecast_short_comp_based(series_id, pdf)
        elif self.config.short_series_strategy == "seasonal_naive":
            return self._forecast_short_seasonal_naive(series_id, pdf)
        else:
            return self._forecast_short_naive(series_id, pdf)
        
    def _forecast_naive_one_series(self, series_id: str, pdf: pd.DataFrame) -> pd.DataFrame:
        """
        Very simple fallback forecaster for a single series:

        - Uses the last observed target value as the forecast for all horizon steps.
        - Works on the *original* target_col (not transformed).

        This is used as a safety net when STL/ETS fails or when the series
        is effectively constant.
        """
        c = self.config

        pdf = pdf.copy()
        pdf = pdf.sort_values(c.date_col)

        if pdf.empty:
            raise ValueError(f"Series '{series_id}' has no data; cannot forecast.")

        if c.target_col not in pdf.columns:
            raise ValueError(
                f"Target column '{c.target_col}' not found for series '{series_id}'."
            )

        last_val = float(pdf[c.target_col].iloc[-1])
        h = c.horizon

        last_date = pd.to_datetime(pdf[c.date_col].max())

        if c.time_granularity.lower() == "week":
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(weeks=1),
                periods=h,
                freq="W",
            ).normalize()
        elif c.time_granularity.lower() == "month":
            future_dates = pd.date_range(
                start=(last_date + pd.DateOffset(months=1)).replace(day=1),
                periods=h,
                freq="MS",
            )
        else:
            raise ValueError(
                f"Unsupported time_granularity '{c.time_granularity}'. "
                "Expected 'week' or 'month'."
            )

        out = pd.DataFrame(
            {
                c.group_col: [series_id] * h,
                c.date_col: future_dates,
                "y_hat": np.full(h, last_val, dtype="float64"),
            }
        )

        return out
