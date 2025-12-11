# The first ten lines of this file are only needed for testing and debugging locally, outside of Databricks.
from pyspark.sql import SparkSession, functions as F

# Create a local SparkSession
spark = (
    SparkSession.builder.appName("ts-preprocessing-job")
    .master("local[*]")  # use all local cores
    .getOrCreate()
)
# Everything above this line is only needed for testing and debugging locally, outside of Databricks.

import time

from pyspark.sql import functions as F

from demand_forecasting.time_series_pipeline.ts0_preprocessing import TSPreprocessingConfig, TSPreprocessor
from demand_forecasting.time_series_pipeline.ts1_diagnostics import TSDiagnostics
from demand_forecasting.time_series_pipeline.ts2_plots import TSPlotter, TSPlotConfig


def main(run_plots: bool = False, use_boxcox: bool = True) -> None:
    t_start_all = time.time()

    # -----------------------------
    # Config and preprocessor
    # -----------------------------
    config = TSPreprocessingConfig(
        source_table="forecast_dev.data_science.ts_source_view",
        output_catalog="forecast_dev.data_science",
        output_table_name="ts_preprocessed",
        group_key_cols=["parent_company_fc", "item_id_fc"],
    )

    pre = TSPreprocessor(spark, config)

    # -----------------------------
    # Layer 1: preprocessing
    # -----------------------------
    t0 = time.time()
    df_final = pre.run(with_boxcox=use_boxcox)   # <-- toggle Box-Cox here
    t1 = time.time()
    pre.write_output_table(df_final)
    t2 = time.time()

    print("\n=== Layer 1: preprocessing ===")
    print(f"  Transform runtime (TSPreprocessor.run, boxcox={use_boxcox}): {t1 - t0:.2f} sec")
    print(f"  Write runtime (write_output_table):                         {t2 - t1:.2f} sec")
    print(f"  Total Layer 1 runtime:                                      {t2 - t0:.2f} sec")

    # -----------------------------
    # Layer 2: diagnostics
    # -----------------------------
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

    # -----------------------------
    # Layer 3: plots (optional)
    # -----------------------------
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

        # 3.2 Short series by dims (customer_*, product_*)
        print("\n=== Layer 3: plotting (short series by dimension) ===")
        plotter.plot_short_series_by_all_dims(
            diagnostics,
            value_col="y_clean",
            prefixes=("customer_", "product_"),
        )

        # Build the list of dim columns we’ll use for 3.3 and 3.4:
        #   - only dims that start with "customer_" or "product_"
        #   - and are actually present in df_final
        dim_cols_for_customer_product = [
            col
            for col in config.dim_cols
            if (col.startswith("customer_") or col.startswith("product_"))
            and col in df_final.columns
        ]

        # 3.3 Volume by dimension (T52) for ALL such dims
        print("\n=== Layer 3: plotting (T52 volume by customer_/product_ dims) ===")
        for dim_col in dim_cols_for_customer_product:
            print(f"\n--- Volume by {dim_col} (last 52 periods) ---")
            plotter.plot_volume_by_dimension(
                df_final,
                dim_col=dim_col,
                value_col="y_clean",
                last_n_weeks=52,
            )

        # 3.4 Year/day-of-year profiles for ALL such dims
        print("\n=== Layer 3: plotting (year vs day-of-year profiles) for Leader of each dimension ===")
        for dim_col in dim_cols_for_customer_product:
            # Find the top value of this dim by total volume (overall) to use as our representative
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

            # One plot per dim: multi-year lines for the single top category
            plotter.plot_year_day_lines_for_dimension(
                year_day_df,
                dim_col=dim_col,
                dim_value=dim_value,
                value_label="Sum of y_clean",
                use_3d=False,  # flip to True if you want the 3D view
            )

        t_plot_end = time.time()
        print("\n=== Layer 3: plotting ===")
        print(f"  Runtime: {t_plot_end - t_plot_start:.2f} sec")

    # -----------------------------
    # Overall
    # -----------------------------
    t_overall_end = time.time()
    print("\n=== Overall runtime ===")
    print(f"  Total (Layers 1–3): {t_overall_end - t_start_all:.2f} sec")


if __name__ == "__main__":
    # Example usage:
    #   - run_plots=False for “just ETL + diagnostics” in production jobs
    #   - use_boxcox=False if you want a quick run without transformation
    main(run_plots=False, use_boxcox=True)

