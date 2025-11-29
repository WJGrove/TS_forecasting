from pyspark.sql import SparkSession

# Create a local SparkSession
spark = (
    SparkSession.builder.appName("ts-preprocessing-job")
    .master("local[*]")  # use all local cores
    .getOrCreate()
)

# Everything above this line is only needed for local testing and debugging locally, outside of Databricks.

from ts_preprocessing import TSPreprocessingConfig, TSPreprocessor
import time


config = TSPreprocessingConfig(
    source_table="forecast_dev.data_science.ts_source_view",
    output_catalog="forecast_dev.data_science",
    output_table_name="ts_preprocessed",
)

pre = TSPreprocessor(spark, config)

t0 = time.time()
df_final = pre.run(with_boxcox=True)
pre.write_output_table(df_final)
t1 = time.time()

print(f"Runtime (sec): {t1 - t0}")
print(f"Runtime (min): {(t1 - t0) / 60}")
