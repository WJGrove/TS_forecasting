from __future__ import annotations

import time
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from demand_forecasting.time_series_pipeline.ts0_preprocessing import (
    TSPreprocessingConfig,
    TSPreprocessor,
)
from demand_forecasting.time_series_pipeline.ts1_diagnostics import TSDiagnostics
from demand_forecasting.time_series_pipeline.ts2_plots import TSPlotter, TSPlotConfig


# --------------------------------------------------------------------------------------
# Local SparkSession for testing/debugging (Databricks would provide spark for us)
# --------------------------------------------------------------------------------------
spark = (
    SparkSession.builder.appName("ts-preprocessing-job")
    .master("local[*]")  # use all local cores
    .getOrCreate()
)


DATA_DIR = Path("demand_forecasting/data/kaggle_rossmann")
RAW_PANEL_PARQUET = DATA_DIR / "rossmann_panel.parquet"
PREP_PANEL_PARQUET = DATA_DIR / "rossmann_panel_preprocessed.parquet"


def main(run_plots: bool = False, use_boxcox: bool = True) -> None:
    t_start_all = time.time()

    # Make sure a database exists for outputs (local dev)
    spark.sql("CREATE DATABASE IF NOT EXISTS forecast_local")

    # Read the Rossmann panel parquet and expose as a temp view
    df_panel = spark.read.parquet(str(RAW_PANEL_PARQUET))
    df_panel.createOrReplaceTempView("rossmann_panel")

    # ----------------------------------------------------------------------------------
    # Config and preprocessor
    # ----------------------------------------------------------------------------------
    config = TSPreprocessingConfig(
        source_table="rossmann_panel",
        # for local Spark, "default" is usually fine
        output_catalog="forecast_local",
        output_table_name="rossmann_panel_preprocessed",
        raw_date_col="ds",
        raw_value_col="y",
        group_col="time_series_id",
        facility_col="time_series_id",  # here, facility == store id , which is same as time_series_id
        facility_dim_col1="StoreType",
        facility_dim_col2="Assortment",
        time_granularity="week",
        interpolation_method="linear",
        seasonal_period=52,
        short_series_threshold=52 * 2,  # 2 years of weekly data
        inactive_threshold=3,
        insufficient_data_threshold=1,
        outlier_threshold=3.0,
    )

    pre = TSPreprocessor(spark, config)

    # ----------------------------------------------------------------------------------
    # Layer 1: preprocessing
    # ----------------------------------------------------------------------------------
    t0 = time.time()
    df_final = pre.run(with_boxcox=use_boxcox)
    t1 = time.time()

    # Write as a Spark table (in the local metastore)
    pre.write_output_table(df_final)
    t2 = time.time()

    # Also write a parquet so you can inspect it easily with pandas if you want
    df_final.write.mode("overwrite").parquet(str(PREP_PANEL_PARQUET))

    print("\n=== Layer 1: preprocessing ===")
    print(
        f"  Transform runtime (TSPreprocessor.run, boxcox={use_boxcox}): {t1 - t0:.2f} sec"
    )
    print(
        f"  Write runtime (write_output_table):                         {t2 - t1:.2f} sec"
    )
    print(
        f"  Total Layer 1 runtime:                                      {t2 - t0:.2f} sec"
    )
    print(
        f"  Preprocessed parquet written to:                             {PREP_PANEL_PARQUET}"
    )

    # ----------------------------------------------------------------------------------
    # Layer 2: diagnostics
    # ----------------------------------------------------------------------------------
    t_diag_start = time.time()
    diagnostics = TSDiagnostics(spark, config, df_final)

    short_stats = diagnostics.compute_short_series_stats(value_col="y_clean")
    nulls_df = diagnostics.column_null_counts()
    series_summary_df = diagnostics.series_level_summary(value_col="y_clean")
    t_diag_end = time.time()

    print("\n=== Layer 2: diagnostics ===")
    print(f"  Runtime: {t_diag_end - t_diag_start:.2f} sec")

    print("\n--- Short-series diagnostics ---")
    print(f"Total series: {short_stats.total_series}")
    print(f"Short series: {short_stats.short_series}")
    print(f"Short series ratio: {short_stats.short_series_ratio:.3f}")
    print(
        f"Short-series volume share: "
        f"{short_stats.short_series_volume_pct:.2f}% "
        f"(window total volume = {short_stats.total_volume_window:.1f})"
    )
    print(f"Warning level: {short_stats.warn_level}")
    print(short_stats.warn_message)

    print("\n--- Null counts by column (top 50) ---")
    nulls_df.show(50, truncate=False)

    print("\n--- Example of series-level summary ---")
    series_summary_df.show(10, truncate=False)

    # ----------------------------------------------------------------------------------
    # Layer 3: plots (optional)
    # ----------------------------------------------------------------------------------
    if run_plots:
        t_plot_start = time.time()

        plot_config = TSPlotConfig(default_figsize=(12.0, 6.0), max_categories=15)
        plotter = TSPlotter(config, plot_config)

        # 3.1 Distributions
        print("\n=== Layer 3: plotting (distributions) ===")
        plotter.plot_series_length_histogram(diagnostics, short_only=False)
        plotter.plot_series_length_histogram(diagnostics, short_only=True)
        plotter.plot_series_volume_histogram(diagnostics, value_col="y_clean")
        plotter.plot_volume_vs_length_scatter(diagnostics, value_col="y_clean")

        # 3.2 Short series by dims (customer_*, product_*, facility_*)
        print("\n=== Layer 3: plotting (short series by dimension) ===")
        plotter.plot_short_series_by_all_dims(
            diagnostics,
            value_col="y_clean",
            prefixes=("customer_", "product_", "facility_"),
        )

        # Build list of dim columns present in df_final
        dim_cols_for_customer_product = [
            col
            for col in config.dim_cols
            if (
                col.startswith("customer_")
                or col.startswith("product_")
                or col.startswith("facility_")
            )
            and col in df_final.columns
        ]

        # 3.3 Volume by dimension (T52) for these dims
        print(
            "\n=== Layer 3: plotting (T52 volume by customer_/product_/facility_ dims) ==="
        )
        for dim_col in dim_cols_for_customer_product:
            print(f"\n--- Volume by {dim_col} (last 52 periods) ---")
            plotter.plot_volume_by_dimension(
                df_final,
                dim_col=dim_col,
                value_col="y_clean",
                last_n_weeks=52,
            )

        # 3.4 Year/day-of-year profiles for top value of each dim
        print(
            "\n=== Layer 3: plotting (year vs day-of-year profiles) for leader of each dimension ==="
        )
        for dim_col in dim_cols_for_customer_product:
            top_val_row = (
                df_final.groupBy(dim_col)
                .agg(F.sum("y_clean").alias("total_volume"))
                .orderBy(F.desc("total_volume"))
                .limit(1)
                .collect()
            )
            if not top_val_row:
                continue

            dim_value = top_val_row[0][0]
            print(f"\n--- Year/day-of-year profile for {dim_col} = {dim_value} ---")

            year_day_df = plotter.prepare_year_day_aggregation(
                df_final,
                dim_col=dim_col,
                value_col="y_clean",
            )

            plotter.plot_year_day_lines_for_dimension(
                year_day_df,
                dim_col=dim_col,
                dim_value=dim_value,
                value_label="Sum of y_clean",
                use_3d=False,
            )

        t_plot_end = time.time()
        print("\n=== Layer 3: plotting ===")
        print(f"  Runtime: {t_plot_end - t_plot_start:.2f} sec")

    # ----------------------------------------------------------------------------------
    # Overall
    # ----------------------------------------------------------------------------------
    t_overall_end = time.time()
    print("\n=== Overall runtime ===")
    if run_plots:
        print("  (including Layer 3: plotting)")
    else:
        print("  (excluding Layer 3: plotting)")
    print(f"  Total preprocessing: {t_overall_end - t_start_all:.2f} sec")


if __name__ == "__main__":
    main(run_plots=True, use_boxcox=True)
