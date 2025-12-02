import pandas as pd
from dataclasses import dataclass

from pyspark.sql import DataFrame as SparkDataFrame
from typing import Literal, Optional

TimeGranularity = Literal["week", "month"]
BaselineMethod = Literal["seasonal_naive"]  # can extend later


@dataclass
class TSForecastConfig:
    horizon: int = 26  # number of periods ahead
    time_granularity: TimeGranularity = "week"
    seasonal_period: int = 52
    min_history_for_exp_smoothing: int = 2 * 52  # e.g. 2 years of weekly data

    target_col: str = "y_clean_int"
    transformed_target_col: str | None = "y_clean_int_transformed"
    transformation_constant_col: str | None = "series_lambda"
    group_col: str = "time_series_id"
    date_col: str = "ds"
    short_flag_col: str = "is_short_series"

    forecast_transformed_target: bool = False

    baseline_method: BaselineMethod = "seasonal_naive"

    def __post_init__(self):
        # ensure positive integers where needed
        if self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if self.seasonal_period <= 0:
            raise ValueError("seasonal_period must be a positive integer")
        if self.min_history_for_exp_smoothing <= 0:
            raise ValueError("min_history_for_exp_smoothing must be a positive integer")


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

    def _forecast_long_one_series(
        self, series_id: str, pdf: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Forecast one 'long' series using exponential smoothing.

        This is the place we’ll paste/refactor your old ETS logic.
        For now, keep it as a stub or a very simple example.
        """
        raise NotImplementedError

    def _forecast_short_one_series(
        self, series_id: str, pdf: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Forecast one 'short' series using a simpler strategy
        (e.g., naive/seasonal-naive/mean of last N).
        """
        raise NotImplementedError

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
