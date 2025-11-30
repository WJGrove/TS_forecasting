# ts2_plots.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .ts0_preprocessing import TSPreprocessingConfig
from .ts1_diagnostics import TSDiagnostics


@dataclass
class TSPlotConfig:
    """
    Configuration for plotting defaults.
    """
    default_figsize: tuple[float, float] = (12.0, 6.0)
    # Max number of categories to show in bar charts, etc.
    max_categories: int = 20
    # Day-of-year tick spacing for year/day plots
    year_day_xtick_step: int = 30


class TSPlotter:
    """
    Layer 3 plotting utilities for time-series diagnostics and EDA.

    This class expects:
    - TSPreprocessingConfig (for column names, dims, etc.)
    - A TSPlotConfig (optional, for plotting defaults)

    Most methods accept:
    - A TSDiagnostics instance (to get Spark aggregations)
    - Or a Spark DataFrame directly (for dimension-specific analyses)
    """

    def __init__(
        self,
        config: TSPreprocessingConfig,
        plot_config: Optional[TSPlotConfig] = None,
    ) -> None:
        self.config = config
        self.plot_config = plot_config or TSPlotConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spark_to_pandas(
        self,
        df: DataFrame,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Convert a (usually aggregated) Spark DataFrame to pandas.

        The optional limit is a safety valve if someone passes a non-aggregated DF.
        """
        if limit is not None:
            df = df.limit(limit)
        return df.toPandas()

    # ------------------------------------------------------------------
    # 1) Distribution plots: lengths & volumes
    # ------------------------------------------------------------------

    def plot_series_length_histogram(
        self,
        diagnostics: TSDiagnostics,
        *,
        short_only: bool = False,
        bins: int = 20,
    ) -> None:
        """
        Histogram of series lengths (in periods).

        - If short_only=True, restricts to short series.
        - Uses the output of TSDiagnostics.series_length_distribution().
        """
        lengths_df = diagnostics.series_length_distribution()
        pdf = self._spark_to_pandas(lengths_df)

        if short_only and "is_short_series" in pdf.columns:
            pdf = pdf[pdf["is_short_series"] == True]

        if pdf.empty:
            print("No data available for series length histogram.")
            return

        plt.figure(figsize=self.plot_config.default_figsize)
        plt.hist(pdf["series_length"], bins=bins, alpha=0.75)
        plt.xlabel("Length of Time Series")
        plt.ylabel("Frequency")

        if short_only:
            plt.title(
                f"Histogram of Short Series Lengths "
                f"(< {self.config.short_series_threshold} periods)"
            )
        else:
            plt.title("Histogram of All Series Lengths")

        plt.grid(True)
        plt.show()

    def plot_series_volume_histogram(
        self,
        diagnostics: TSDiagnostics,
        *,
        value_col: str = "y_clean",
        bins: int = 20,
    ) -> None:
        """
        Histogram of total volume per series.

        Uses TSDiagnostics.series_volume_distribution().
        """
        volumes_df = diagnostics.series_volume_distribution(value_col=value_col)
        pdf = self._spark_to_pandas(volumes_df)

        if pdf.empty:
            print("No data available for series volume histogram.")
            return

        plt.figure(figsize=self.plot_config.default_figsize)
        plt.hist(pdf["series_volume"], bins=bins, alpha=0.75)
        plt.xlabel(f"Total {value_col} per Series")
        plt.ylabel("Frequency")
        plt.title("Histogram of All Series Volumes")
        plt.grid(True)
        plt.show()

    def plot_volume_vs_length_scatter(
        self,
        diagnostics: TSDiagnostics,
        *,
        value_col: str = "y_clean",
    ) -> None:
        """
        Scatter plot: total volume vs series length.

        This is useful for spotting short but high-volume series that may
        deserve more modeling effort than a naive short-series rule.
        """
        c = self.config

        lengths_df = diagnostics.series_length_distribution()
        volumes_df = diagnostics.series_volume_distribution(value_col=value_col)

        combined_df = lengths_df.join(volumes_df, on=c.group_col, how="inner")
        pdf = self._spark_to_pandas(combined_df)

        if pdf.empty:
            print("No data available for volume vs length scatter.")
            return

        plt.figure(figsize=self.plot_config.default_figsize)
        plt.scatter(pdf["series_length"], pdf["series_volume"], alpha=0.75)
        plt.xlabel("Length of Time Series (periods)")
        plt.ylabel(f"Total {value_col} per Series")
        plt.title("Series Volume vs. Length")
        plt.grid(True)
        plt.show()

    # ------------------------------------------------------------------
    # 2) Volume & short-series share by dimension
    # ------------------------------------------------------------------

    def _dimension_short_series_summary(
        self,
        diagnostics: TSDiagnostics,
        dim_col: str,
        *,
        value_col: str = "y_clean",
    ) -> DataFrame:
        """
        Compute per-dimension short-series stats in Spark.

        Returns a Spark DataFrame with:
        - dim_col
        - total_series
        - short_series
        - short_series_pct
        - total_volume
        - short_series_volume
        - short_series_volume_pct
        """
        c = self.config
        df = diagnostics.df

        if dim_col not in df.columns:
            raise ValueError(f"Dimension column '{dim_col}' not found in DataFrame.")

        # Series-level summary with is_short_series and volume
        series_summary = diagnostics.series_level_summary(value_col=value_col)
        # Attach dimension to series
        dim_map = (
            df.select(c.group_col, dim_col)
            .dropDuplicates([c.group_col, dim_col])
        )

        joined = series_summary.join(dim_map, on=c.group_col, how="left")

        # Aggregate by dimension
        agg_df = (
            joined.groupBy(dim_col)
            .agg(
                F.countDistinct(c.group_col).alias("total_series"),
                F.sum(
                    F.when(F.col("is_short_series") == True, 1).otherwise(0)
                ).alias("short_series"),
                F.sum(F.col("series_volume")).alias("total_volume"),
                F.sum(
                    F.when(F.col("is_short_series") == True, F.col("series_volume"))
                    .otherwise(0.0)
                ).alias("short_series_volume"),
            )
        )

        agg_df = agg_df.withColumn(
            "short_series_pct",
            F.when(
                F.col("total_series") > 0,
                F.col("short_series") / F.col("total_series"),
            ).otherwise(F.lit(0.0)),
        )

        agg_df = agg_df.withColumn(
            "short_series_volume_pct",
            F.when(
                F.col("total_volume") > 0,
                F.col("short_series_volume") / F.col("total_volume"),
            ).otherwise(F.lit(0.0)),
        )

        return agg_df

    def plot_short_series_by_dimension(
        self,
        diagnostics: TSDiagnostics,
        dim_col: str,
        *,
        value_col: str = "y_clean",
        top_n: Optional[int] = None,
    ) -> None:
        """
        Bar chart of short-series percentage by a dimension column.

        - dim_col should be one of your customer_/product_ columns from config.
        - Shows only top categories by total volume (or by total_series).
        """
        top_n = top_n or self.plot_config.max_categories

        dim_df = self._dimension_short_series_summary(
            diagnostics, dim_col=dim_col, value_col=value_col
        )

        pdf = self._spark_to_pandas(dim_df)
        if pdf.empty:
            print(f"No data available for {dim_col} short-series plot.")
            return

        # Focus on the categories with highest total volume
        pdf = pdf.sort_values("total_volume", ascending=False).head(top_n)

        # Convert percentages to 0–100 for plotting
        pdf["short_series_pct_100"] = pdf["short_series_pct"] * 100.0
        pdf["short_series_volume_pct_100"] = pdf["short_series_volume_pct"] * 100.0

        fig, ax = plt.subplots(figsize=self.plot_config.default_figsize)
        ax.bar(
            pdf[dim_col].astype(str),
            pdf["short_series_pct_100"],
            alpha=0.75,
        )
        ax.set_xlabel(dim_col)
        ax.set_ylabel("Short Series (% of series)")
        ax.set_title(
            f"Short-Series Share by {dim_col} "
            f"(top {top_n} by total volume)"
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(axis="y")
        plt.show()

        # Optional second plot: short-series volume share
        fig, ax = plt.subplots(figsize=self.plot_config.default_figsize)
        ax.bar(
            pdf[dim_col].astype(str),
            pdf["short_series_volume_pct_100"],
            alpha=0.75,
        )
        ax.set_xlabel(dim_col)
        ax.set_ylabel("Short-Series Volume (% of volume)")
        ax.set_title(
            f"Short-Series Volume Share by {dim_col} "
            f"(top {top_n} by total volume)"
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(axis="y")
        plt.show()

    def plot_short_series_by_all_dims(
        self,
        diagnostics: TSDiagnostics,
        *,
        value_col: str = "y_clean",
        prefixes: Sequence[str] = ("customer_", "product_"),
    ) -> None:
        """
        Convenience method: loop over all dim_cols in config that start with the
        given prefixes (e.g., customer_*, product_*) and plot short-series stats.

        This may generate many plots if you have many dimensions; intended for
        notebook-style EDA, not dashboards.
        """
        dim_cols = [
            col
            for col in self.config.dim_cols
            if any(col.startswith(p) for p in prefixes)
        ]

        for dim_col in dim_cols:
            print(f"\n=== Short-series diagnostics by {dim_col} ===")
            self.plot_short_series_by_dimension(
                diagnostics,
                dim_col=dim_col,
                value_col=value_col,
            )

    # ------------------------------------------------------------------
    # 3) Volume by dimension (e.g., "T52W volume by bottle type")
    # ------------------------------------------------------------------

    def plot_volume_by_dimension(
        self,
        df: DataFrame,
        dim_col: str,
        *,
        value_col: str = "y_clean",
        last_n_weeks: Optional[int] = None,
    ) -> None:
        """
        Bar chart: total volume by a dimension over an optional trailing window.

        - df should be the preprocessed Spark DataFrame (or a filtered subset).
        - last_n_weeks: if provided, restrict to the last N weeks based on ds.
          (We use max(ds) in df as the anchor; callers can pre-filter if they
          need "last Sunday before today" semantics.)
        """
        c = self.config
        if dim_col not in df.columns:
            raise ValueError(f"Dimension column '{dim_col}' not found in DataFrame.")
        if c.date_col not in df.columns:
            raise ValueError(f"Date column '{c.date_col}' not found in DataFrame.")
        if value_col not in df.columns:
            raise ValueError(f"value_col '{value_col}' not found in DataFrame.")

        work_df = df

        if last_n_weeks is not None:
            # Use the latest date in this DF as the anchor.
            max_date = work_df.agg(F.max(F.col(c.date_col)).alias("max_ds")).collect()[0][
                "max_ds"
            ]
            if max_date is not None:
                # Compute cutoff date inside Spark
                cutoff = max_date - pd.Timedelta(weeks=last_n_weeks)
                work_df = work_df.filter(F.col(c.date_col) >= F.lit(cutoff))

        agg_df = (
            work_df.groupBy(dim_col)
            .agg(F.sum(F.col(value_col)).alias("total_volume"))
            .orderBy(F.desc("total_volume"))
        )

        pdf = self._spark_to_pandas(agg_df)
        if pdf.empty:
            print(f"No data available for volume-by-{dim_col} plot.")
            return

        top_n = self.plot_config.max_categories
        pdf = pdf.head(top_n)

        plt.figure(figsize=self.plot_config.default_figsize)
        plt.bar(pdf[dim_col].astype(str), pdf["total_volume"], alpha=0.75)
        plt.xlabel(dim_col)
        plt.ylabel(f"Total {value_col}")
        if last_n_weeks is not None:
            plt.title(f"Volume by {dim_col} (last {last_n_weeks} periods)")
        else:
            plt.title(f"Volume by {dim_col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(axis="y")
        plt.show()

    # ------------------------------------------------------------------
    # 4) Year vs Day-of-Year plots (3D + 2D template)
    # ------------------------------------------------------------------

    def prepare_year_day_aggregation(
        self,
        df: DataFrame,
        dim_col: str,
        *,
        value_col: str = "y_clean",
    ) -> DataFrame:
        """
        Aggregate to (dim_col, year, day_of_year) for year-over-year seasonal plots.

        This matches the logic you had for the 3D line plots.
        """
        c = self.config

        if dim_col not in df.columns:
            raise ValueError(f"Dimension column '{dim_col}' not found in DataFrame.")
        if c.date_col not in df.columns:
            raise ValueError(f"Date column '{c.date_col}' not found in DataFrame.")
        if value_col not in df.columns:
            raise ValueError(f"value_col '{value_col}' not found in DataFrame.")

        agg_df = (
            df.groupBy(
                dim_col,
                F.year(F.col(c.date_col)).alias("year"),
                F.dayofyear(F.col(c.date_col)).alias("day_of_year"),
            )
            .agg(F.sum(F.col(value_col)).alias("sum_value"))
        )

        return agg_df

    def plot_year_day_lines_for_dimension(
        self,
        df_year_day: DataFrame,
        dim_col: str,
        dim_value: str,
        *,
        value_label: str = "Sum of value",
        use_3d: bool = False,
    ) -> None:
        """
        Plot year-over-year profiles for one dimension value across day_of_year:

        - If use_3d=True: 3D "stacked" lines, one per year.
        - If use_3d=False: standard 2D multi-line plot.

        df_year_day should be the output of prepare_year_day_aggregation().
        """
        pdf = self._spark_to_pandas(
            df_year_day.filter(F.col(dim_col) == dim_value)
        )

        if pdf.empty:
            print(f"No data for {dim_col} = {dim_value} in year/day aggregation.")
            return

        pivot_df = pdf.pivot(
            index="day_of_year", columns="year", values="sum_value"
        ).sort_index()

        # Fill missing days forward to smooth plot (optional, as you did)
        pivot_df = pivot_df.ffill(axis=0) 

        if use_3d:
            # 3D stacked-line template
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection="3d")

            years = pivot_df.columns.values
            year_indices = np.arange(len(years))
            days = pivot_df.index.values

            for i, year in enumerate(years):
                zs = pivot_df[year].values
                ax.plot(
                    np.full(days.shape, i),
                    days,
                    zs,
                    label=str(year),
                )

            ax.set_xticks(year_indices)
            ax.set_xticklabels(years)
            ax.set_xlabel("Year")
            ax.set_ylabel("Day of Year")
            ax.set_zlabel(value_label + f" ({dim_value})")
            ax.view_init(elev=0, azim=0)
            ax.legend(title="Year", bbox_to_anchor=(1, 1), loc="upper left")
            plt.title(
                f"3D Year-over-Year Profile for {dim_col} = {dim_value}"
            )
            plt.show()
        else:
            # 2D multi-line template
            plt.figure(figsize=(10, 7))
            for year in pivot_df.columns:
                plt.plot(
                    pivot_df.index,
                    pivot_df[year],
                    label=str(year),
                )
            plt.xlabel("Day of Year")
            plt.ylabel(value_label + f" ({dim_value})")
            plt.title(
                f"Year-over-Year Profile for {dim_col} = {dim_value}"
            )
            plt.legend(title="Year")
            plt.grid(True)
            plt.xticks(
                np.arange(
                    0,
                    367,
                    step=self.plot_config.year_day_xtick_step,
                )
            )
            plt.tight_layout()
            plt.show()
