# Databricks notebook source
# Confirm Mount to Azure Storage Account

# dbutils.fs.mounts()

# COMMAND ----------

# MAGIC %md
# MAGIC #Installs, Functions, and Inputs

# COMMAND ----------

# MAGIC %md
# MAGIC ##Installs/Imports

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Data Science/Project Prophet v3/General_Project_Imports"

# COMMAND ----------

# MAGIC %md
# MAGIC ##Functions

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Data Science/Project Prophet v3/General_Project_Functions"

# COMMAND ----------

# MAGIC %md
# MAGIC ##Variables

# COMMAND ----------

FiscalYear = '2024'
FiscalPeriod = 'P3'


date_col = 'ds' # origin: 'ReqDelFWstart'
seasonal_period = 52
group_col = 'TimeSeriesID' # origin: concat(col("CustomerNumber"), col("ItemNumber"))


# COMMAND ----------

# MAGIC %md
# MAGIC #Retrieve Data

# COMMAND ----------

# Grab sales data from the 'Data_Pre-Prep_Sales' notebook.
transformed_nonconstant_df = spark.sql("SELECT * FROM global_temp.transformed_nonconstant_df_temp_view")
transformed_nonconstant_df.show()
print(f"'transformed_nonconstant_df' length = {transformed_nonconstant_df.count()}")


# COMMAND ----------

# Stop here for SARIMAX models.
dbutils.notebook.exit("stop")

# COMMAND ----------

# MAGIC %md
# MAGIC # Seasonality and Stationarity Prep (move to its own notebook)

# COMMAND ----------

# MAGIC %md
# MAGIC >##One way to achieve stationarity in a large group of time series (when assuming a seasonal pattern) is to use STL to remove the seasonality from the series and subsequently test for stationarity and difference as needed. This is the method I'm going to use - initially - for the series that aren't long enough for autofitting a SARIMAX model.

# COMMAND ----------

# MAGIC %md
# MAGIC ##STL

# COMMAND ----------

def get_stl_udf(value_col, seasonal_period=52, date_col='ds', group_col='TimeSeriesID'):
    # Define the output schema of the Pandas UDF
    schema = StructType([
        StructField("TimeSeriesID", StringType(), True),
        StructField(date_col, DateType(), True),
        StructField(f"{value_col}_seasonal", DoubleType(), True),
        StructField(f"{value_col}_trend", DoubleType(), True),
        StructField(f"{value_col}_resid", DoubleType(), True),
    ])
    
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def stl_decompose(pdf):
        # Ensure the DataFrame is sorted by the specified date column
        pdf = pdf.sort_values(by=date_col)
        
        # Perform STL decomposition
        stl = STL(pdf[value_col], period=seasonal_period, robust=True)
        result = stl.fit()
        
        # Prepare the result as a new DataFrame
        return pd.DataFrame({
            group_col: pdf[group_col],
            date_col: pdf[date_col],
            f"{value_col}_seasonal": result.seasonal,
            f"{value_col}_trend": result.trend,
            f"{value_col}_resid": result.resid,
        })
    
    return stl_decompose

# COMMAND ----------

# Example usage with custom date column name
value_col = 'y_out_rem_transformed'  # The column to decompose

# Create the UDF with specified parameters
stl_udf = get_stl_udf(value_col, seasonal_period, date_col)
# Apply the STL decomposition UDF to each TimeSeriesID group
stl_results_df = transformed_df.groupBy("TimeSeriesID").apply(stl_udf)

# Show the results
stl_results_df.show()
print(f"'stl_results_df' length = {stl_results_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##ADF Test

# COMMAND ----------

# I don't think this function ('test_stationarity_and_augment_df') will work for this purpose (forecasting) because it's limited to the driver node
# due to using a Pandas df. Need a Grouped Map Pandas UDF (I think there are three types of Pandas UDFs: Scalar, Grouped Map, and Grouped Aggregate.)
def test_stationarity_and_augment_df(spark_df, group_col, date_col, value_col):
    """
    This function tests each time series in a Spark DataFrame for stationarity using the Augmented Dickey-Fuller (ADF) test.
    It converts the Spark DataFrame to a Pandas DataFrame to apply the test, then augments the original DataFrame with
    the results: an 'is_stationary' flag, the ADF test statistic ('adf_result'), and the p-value ('adf_p_value').
    
    Parameters:
    - spark_df: The input Spark DataFrame containing the time series data.
    - group_col: The name of the column that identifies each time series group.
    - date_col: The name of the column that represents the date or time index.
    - value_col: The name of the column containing the time series values to test.
    
    Returns:
    - A Spark DataFrame identical to the input DataFrame but with added columns for stationarity testing results.
    """
    
    # Convert the Spark DataFrame to a Pandas DataFrame for processing
    station_tested_df = spark_df.toPandas()
    
    # Ensure the DataFrame is sorted by group_col and date_col
    station_tested_df.sort_values(by=[group_col, date_col], inplace=True)
    
    # Initialize columns for ADF test results
    station_tested_df['is_stationary'] = False
    station_tested_df['adf_result'] = None
    station_tested_df['adf_p_value'] = None
    
    # Perform the ADF test for each group and update the DataFrame
    for group_name, group_df in station_tested_df.groupby(group_col):
        # Perform the ADF test
        adf_result = adfuller(group_df[value_col], autolag='AIC')
        
        # Update the DataFrame with the ADF test results
        station_tested_df.loc[group_df.index, 'adf_result'] = adf_result[0]  # Test statistic
        station_tested_df.loc[group_df.index, 'adf_p_value'] = adf_result[1]  # p-value
        station_tested_df.loc[group_df.index, 'is_stationary'] = adf_result[1] < 0.05  # Stationarity flag
    
    # Convert the Pandas DataFrame back to a Spark DataFrame
    station_tested_df = spark.createDataFrame(station_tested_df)
    
    return station_tested_df


# COMMAND ----------

# Pandas UDF for Augmented Dickey-Fuller test:
# (Using Pandas UDFs to be able to utilize Pandas and parallelization simultaneously is key to this process.)
# Rewrite this so it's wrapped in another function to accept variable parameters (see other Pandas UDF in this project).


# Define the schema of the UDF output to include both TimeSeriesID and p-value
schema = StructType([
    StructField("TimeSeriesID", StringType(), True),
    StructField("ADF_p_value", DoubleType(), True),
])

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def adf_test_UDF(group: pd.DataFrame) -> pd.DataFrame:
    # Ensure the group is sorted by date
    group = group.sort_values(by=date_col)
    # Perform the Augmented Dickey-Fuller test on the 'value' column
    result = adfuller(group[value_col], autolag='AIC')
    # Return a DataFrame with TimeSeriesID and the p-value
    return pd.DataFrame({
        "TimeSeriesID": [group['TimeSeriesID'].iloc[0]],  # Assuming TimeSeriesID is constant within each group
        "ADF_p_value": [result[1]]
    })

# COMMAND ----------

# Apply the ADF stationarity test.
value_col = 'y_out_rem_transformed_resid'
resid_ADF_test_results = stl_results_df.groupBy("TimeSeriesID").apply(adf_test_UDF)
print(f'ADF_test_results:')
resid_ADF_test_results.show()

# Plot histogram of p-values
# Collect 'ADF_p_value' into a Pandas DataFrame
p_values_df = resid_ADF_test_results.select('ADF_p_value').toPandas()
# Plot
plt.figure(figsize=(15, 6))
sns.histplot(p_values_df['ADF_p_value'], bins=30, kde=False, label='ADF_p_value')
plt.xlabel('ADF p-value')
plt.ylabel('Frequency')
plt.title('Histogram of ADF p-values for Remainder Component')
plt.legend()
plt.grid(True)
plt.show()

# Count the non-stationary series.
nonstationary_count = resid_ADF_test_results.agg(
    F.count(F.when(F.col('ADF_p_value') >= 0.05, 1)).alias('ADF_nonstationary_series'))
nonstationary_count.show()

# Attach the results (p-values) back to stl_results_df and add a boolean to interpret the p-value for differencing.
ADF_tested_df = stl_results_df.join(resid_ADF_test_results, on='TimeSeriesID', how='left')

# The following is for using differencing on the residual of some series.
# , but not necessarily. (Run the ADF again to see.)
# Add a boolean column 'should_difference' based on the p-value condition
# ADF_tested_df = ADF_tested_df.withColumn('should_difference', F.col('ADF_p_value') >= 0.05)
ADF_tested_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##KPSS Test

# COMMAND ----------

def get_kpss_test_udf(value_col, group_col, date_col):
    # Define the schema for the output of the Pandas UDF
    schema = StructType([
        StructField(group_col, StringType(), True),
        StructField("kpss_stat", DoubleType(), True),
        StructField("kpss_p_value", DoubleType(), True),
        StructField("lags", DoubleType(), True),
        StructField("stationarity", StringType(), True),  # Based on p-value, for example
    ])

    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def kpss_test_UDF(pdf):
        # Ensure the DataFrame is sorted by date_col
        pdf = pdf.sort_values(by=date_col)
        
        # Perform KPSS test
        try:
            stat, p_value, lags, critical_values = kpss(pdf[value_col], nlags="auto")
            # Interpret the p-value
            stationarity = 'stationary' if p_value >= 0.05 else 'non-stationary'
        except ValueError as e:  # Handle potential errors from KPSS test
            stat, p_value, lags, stationarity = None, None, None, 'test not applicable'
        
        return pd.DataFrame({
            group_col: [pdf[group_col].iloc[0]],
            "kpss_stat": [stat],
            "kpss_p_value": [p_value],
            "lags": [lags],
            "stationarity": [stationarity],
        })

    return kpss_test_UDF

# COMMAND ----------

# Apply the KPSS stationarity test.

# define value_col
value_col = 'y_out_rem_transformed_resid'
# Create the UDF with specified parameters.
# (Wrapping a Pandas UDF in another function allows us to use variable parameters like normal.)
kpss_udf = get_kpss_test_udf(value_col, group_col, date_col)
# Now, apply the UDF to the DataFrame
resid_KPSS_test_results = stl_results_df.groupBy(group_col).apply(kpss_udf)

print(f'KPSS_test_results:')
resid_KPSS_test_results.show()

# COMMAND ----------

dbutils.notebook.exit("stop")

# COMMAND ----------

# Count the non-stationary series.
KPSS_nonstationary_count = resid_ADF_test_results.agg(
    F.count(F.when(F.col('kpss_p_value') < 0.05, 1)).alias('KPSS_nonstationary_series'))
KPSS_nonstationary_count.show()

# COMMAND ----------

dbutils.notebook.exit("stop")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Differencing

# COMMAND ----------

# difference the data based on stationarity test above

# Define the window specification
windowSpec = Window.partitionBy(group_col).orderBy(date_col)

# Apply differencing conditionally, based on 'should_difference'
# For rows that aren't differenced, retain the original 'y_transformed' value.
differenced_df = ADF_tested_df.withColumn('y_differenced',
                                          F.when(F.col('should_difference'),
                                                 F.col('y_transformed') - F.lag('y_transformed', 1).over(windowSpec))
                                          .otherwise(F.col('y_transformed')))

# Handle the nulls for the first row in each group.
differenced_df = differenced_df.na.fill({'y_differenced': 0})  # Or use another appropriate fill method
differenced_df.show()
print(f"'differenced_df' length = {differenced_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Store Data

# COMMAND ----------

# Check the column names, data types, size, etc. of the data.

# Are there NULLS? (Check the series individually for NULLS as the series are different lengths.)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Global Temporary View

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ##Write to Storage

# COMMAND ----------

# change this
# expanded_df.write.format("parquet").save(f"{sales_file_path}Expanded_Sales")