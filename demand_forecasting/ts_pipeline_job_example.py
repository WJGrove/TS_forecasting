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

from demand_forecasting.ts0_preprocessing import TSPreprocessingConfig, TSPreprocessor
from demand_forecasting.ts1_diagnostics import TSDiagnostics
from demand_forecasting.ts2_plots import TSPlotter, TSPlotConfig


def main(run_plots: bool = True) -> None:
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
    df_final = pre.run(with_boxcox=True)          # 1A: transformations
    t1 = time.time()
    pre.write_output_table(df_final)              # 1B: write/I-O
    t2 = time.time()

    print("\n=== Layer 1: preprocessing ===")
    print(f"  Transform runtime (TSPreprocessor.run): {t1 - t0:.2f} sec")
    print(f"  Write runtime (write_output_table):     {t2 - t1:.2f} sec")
    print(f"  Total Layer 1 runtime:                   {t2 - t0:.2f} sec")

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

        # 3.3 Example: volume by a single dimension over the last 52 periods
        example_dim_for_volume = config.product_dim_col1 or config.customer_parent_company_col
        if example_dim_for_volume and example_dim_for_volume in df_final.columns:
            print(f"\n=== Volume by {example_dim_for_volume} (last 52 periods) ===")
            plotter.plot_volume_by_dimension(
                df_final,
                dim_col=example_dim_for_volume,
                value_col="y_clean",
                last_n_weeks=52,
            )

        # 3.4 Example: year/day-of-year profile for one dim value
        example_dim_for_year_day = config.product_dim_col1 or config.customer_parent_company_col
        if example_dim_for_year_day and example_dim_for_year_day in df_final.columns:
            first_val_row = (
                df_final.select(example_dim_for_year_day)
                .where(F.col(example_dim_for_year_day).isNotNull())
                .limit(1)
                .collect()
            )
            if first_val_row:
                example_value = first_val_row[0][0]
                print(
                    f"\n=== Year/day-of-year profile for "
                    f"{example_dim_for_year_day} = {example_value} ==="
                )

                year_day_df = plotter.prepare_year_day_aggregation(
                    df_final,
                    dim_col=example_dim_for_year_day,
                    value_col="y_clean",
                )

                plotter.plot_year_day_lines_for_dimension(
                    year_day_df,
                    dim_col=example_dim_for_year_day,
                    dim_value=example_value,
                    value_label="Sum of y_clean",
                    use_3d=False,
                )

        t_plot_end = time.time()
        print("\n=== Layer 3: plotting ===")
        print(f"  Runtime: {t_plot_end - t_plot_start:.2f} sec")

    # -----------------------------
    # Overall
    # -----------------------------
    t_overall_end = time.time()
    print("\n=== Overall runtime ===")
    print(f"  Total (Layers 1–3): {t_overall_end - t0:.2f} sec")