from ts_preprocessing import TSPreprocessingConfig, TSPreprocessor
import time
from pyspark.sql import SparkSession

config = TSPreprocessingConfig(
    source_table="forecast_dev.data_science.sales_forecast_source_view",
    output_catalog="forecast_dev.data_science",
    output_table_name="sales_preprocessed",
)

pre = TSPreprocessor(spark, config)

t0 = time.time()
df_final = pre.run(with_boxcox=True)
pre.write_output_table(df_final)
t1 = time.time()

print(f"Runtime (sec): {t1 - t0}")
print(f"Runtime (min): {(t1 - t0) / 60}")