# Databricks notebook source
# MAGIC %md
# MAGIC ##Run Sales Data Pre-Processing Notebook

# COMMAND ----------

# %run "/Workspace/Users/sgrove@drinkmilos.com/Data Science/Project Prophet v3/Pre-Processing_Internal"

# COMMAND ----------

# MAGIC %md
# MAGIC #Imports, Functions, and Variables

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Data Science/Project Prophet v3/General_Project_Imports"

# COMMAND ----------

# Easter has to be handled separately due to the nature of it's changing date
# , so we have to define a custom holiday class.
class CustomUSHoliday(UnitedStates):
    def _populate(self, year):
        # Make sure to call the super()._populate to inherit standard US holidays
        super()._populate(year)
        
        # Adding Easter to the holiday list for the specified year
        self[easter(year)] = "Easter"

# sauce: https://pypi.org/project/holidays/

# COMMAND ----------

# MAGIC %md
# MAGIC ##Variables

# COMMAND ----------

holiday_year_range = range(2020, 2027) 

# COMMAND ----------

# MAGIC %md
# MAGIC #Retrieve Transformed Sales Data

# COMMAND ----------

spark.sql("SHOW TABLES IN global_temp").show()


# COMMAND ----------

# Grab sales data from the 'Data_Pre-Prep_Sales' notebook.
SARIMA_training_df = spark.sql("SELECT * FROM global_temp.SARIMA_training_df_temp_view")
SARIMA_training_df.show()
print(f"'SARIMA_training_df' length = {SARIMA_training_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC >**WARNING:** The target value's column is being renamed to 'y' to better align with process norms. Naming from this point forward will not necessarily be consistent with naming prior to this point.

# COMMAND ----------

# trim columns unnecessary for SARIMAX and rename target value column.
cols_to_drop = ['CustomerCompany', 'CustomerNumber', 'ItemNumber', 'GrossPrice', 'BottleClass', 'ItemDescription', 'ItemGroup', 'y', 'y_out_rem', 'y_is_outlier']
trimmed_df = SARIMA_training_df.drop(*cols_to_drop)
trimmed_df = trimmed_df.withColumnRenamed('y_out_rem_transformed', 'y')
trimmed_df = trimmed_df.orderBy(F.asc('TimeSeriesID'), F.asc('ds'))
trimmed_df.show()
print(f"'trimmed_df' length = {trimmed_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Retrieve and Prepare Holiday Data

# COMMAND ----------

# Define year range
holiday_year_range = holiday_year_range
# Initialize custom holiday class
custom_holidays = CustomUSHoliday()
# Generate the custom holiday list
holiday_list = []
for year in holiday_year_range:
    # Update the year for the custom_holidays instance
    custom_holidays._populate(year)  # Populate holidays for the specified year if needed
    for date, name in sorted(custom_holidays.items()):
        if date.year == year:  # Ensure only holidays for the current year in the loop are added
            holiday_list.append((date, name))

# Print the holiday list
for date, name in holiday_list:
    print(date, name)
print()

# Convert the list to a Spark DataFrame
holiday_df_2020_to_2027 = spark.createDataFrame(holiday_list, schema=["date", "holiday_name"])
# Add 'day_of_year' column
holiday_df = holiday_df_2020_to_2027.withColumn("day_of_year", F.dayofyear(F.col("date")))
# Our data is aggregated to the weekly level and attached to Sundays (the beginning of the fiscal week)
# , so we need to attach the holidays to the Sunday immediately preceding the holiday (if it doesn't fall on a Sunday already).
# If it falls on a Sunday, we just use the original date.
# Add 'leading_Sun_date' column.
holiday_df = holiday_df.withColumn(
    "leading_Sun_date",
    F.expr("date_sub(date, (dayofweek(date) - 1) % 7)")
)

# Show the updated DataFrame to verify the 'leading_Sun_date' column
print(f"holiday_df: ")
holiday_df.show()
print(f"'holiday_df' length = {holiday_df.count()}")
print()
print(f"All holidays: ")
# Select the unique holiday names and collect them
unique_holidays = holiday_df.select("holiday_name").distinct().collect()
# Iterate over the results and print each unique holiday name
for row in unique_holidays:
    print(row.holiday_name)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Choose Holidays

# COMMAND ----------

# List holidays to keep
specified_holidays = [
    "Easter",
    "Independence Day",
    "Memorial Day",
    "Thanksgiving",
    "Christmas Day",
    "Labor Day",
    "New Year's Day",
]
# Filter the DataFrame to keep only the rows where 'holiday_name' matches one of the specified holidays
filtered_holiday_df = holiday_df.filter(F.col("holiday_name").isin(specified_holidays))

filtered_holiday_df.show()
print(f"'filtered_holiday_df' length = {filtered_holiday_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Add One-Hot Encoded Holidays 

# COMMAND ----------

# Join the DataFrames
joined_df = trimmed_df.join(filtered_holiday_df, trimmed_df.ds == filtered_holiday_df.leading_Sun_date, "left")

# Get a list of unique holiday names
unique_holidays = [row['holiday_name'] for row in filtered_holiday_df.select('holiday_name').distinct().collect()]
print(unique_holidays)
# Add a one-hot encoded column for each holiday
# Initialize the DataFrame before the loop to accumulate the one-hot encoded columns
one_hot_encoded_df = joined_df

for holiday in unique_holidays:
    # Perform the special character replacement operations outside the f-string
    clean_holiday_name = holiday.replace(' ', '_').replace("'", "").replace('(', '').replace(')', '')
    column_name = f"exog_{clean_holiday_name}"
    
    # Now, use the cleaned holiday name to create a one-hot encoded column
    one_hot_encoded_df = one_hot_encoded_df.withColumn(column_name, F.when(F.col("holiday_name") == holiday, 1).otherwise(0))

# drop columns that are no longer needed
one_hot_encoded_df = one_hot_encoded_df.drop('day_of_year', 'holiday_name', 'date', 'leading_Sun_date')

one_hot_encoded_df.show()
print(f"'one_hot_encoded_df' length = {one_hot_encoded_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Add Back-Shifted Holidays (Lead Features)

# COMMAND ----------

# Define the window specification, ordered by date
windowSpec = Window.partitionBy("TimeSeriesID").orderBy("ds")
# Assuming one_hot_encoded_df is your DataFrame and windowSpec is already defined
for holiday in unique_holidays:
    clean_holiday_name = holiday.replace(' ', '_').replace("'", "").replace('(', '').replace(')', '')
    column_name = f"exog_{clean_holiday_name}"
    
    # Use coalesce to fill NULLs resulting from lead with 0
    one_hot_encoded_df = one_hot_encoded_df.withColumn(
        f"{column_name}_1w_ahead", 
        F.coalesce(F.lead(one_hot_encoded_df[column_name], 1).over(windowSpec), F.lit(0))
    )
    one_hot_encoded_df = one_hot_encoded_df.withColumn(
        f"{column_name}_2w_ahead", 
        F.coalesce(F.lead(one_hot_encoded_df[column_name], 2).over(windowSpec), F.lit(0))
    )

# Show the DataFrame to verify the new columns and that NULLs are filled with 0
one_hot_encoded_df.show()
print(f"'one_hot_encoded_df' length = {one_hot_encoded_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Trim Some Features

# COMMAND ----------

# For now, I want to drop the lead features for the "holiday observed" dates to reduce redundancy.
# Create a list of columns to drop based on the specified conditions
feature_cols_to_drop = [
    column for column in one_hot_encoded_df.columns 
    if "Observed" in column and ("1w" in column or "2w" in column)
]
# Drop the specified columns from one_hot_encoded_df
trimmed_hol_ohe_df = one_hot_encoded_df.drop(*feature_cols_to_drop)
# trimmed_hol_ohe_df.show()
print("Column names:")
print(trimmed_hol_ohe_df.columns)
print()
print(f"'trimmed_hol_ohe_df' length = {trimmed_hol_ohe_df.count()}")

# number of series
series_count = trimmed_hol_ohe_df.select("TimeSeriesID").distinct().count()
print(f"No. of series: {series_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Check for NULLs

# COMMAND ----------

# Aggregate NULL counts for each column and collect the results
null_counts_row = trimmed_hol_ohe_df.select([F.count(F.when(F.col(c).isNull(), True)).alias(c) for c in trimmed_hol_ohe_df.columns]).collect()[0]
# Convert the Row object into a list of tuples (column_name, null_count)
null_counts_list = [(c, null_counts_row[c]) for c in null_counts_row.__fields__]
print("Number of NULLs in each column: ")
for column, count in null_counts_list:
    print(column, count)

# Print rows containing NULLs in any column
# This creates a condition that is true if any column in a row has a NULL value
has_null_condition = F.expr(' OR '.join([f'({col} IS NULL)' for col in trimmed_hol_ohe_df.columns]))
# Filter to rows that satisfy the 'has_null_condition' and show the result
# print("Rows with NULLs: ")
rows_w_NULLs_df = trimmed_hol_ohe_df.filter(has_null_condition)
# rows_w_NULLs_df.show()
print()
print(f"'rows_w_NULLs_df' length = {rows_w_NULLs_df.count()}")

# COMMAND ----------

# dbutils.notebook.exit("stop")

# COMMAND ----------

# MAGIC %md
# MAGIC #Autofit SARIMAX

# COMMAND ----------

# Exogenous columns list, adjust as needed
exog_columns = [column for column in trimmed_hol_ohe_df.columns if "exog" in column]

@pandas_udf('TimeSeriesID string, best_params string', PandasUDFType.GROUPED_MAP)
def autofit_arima(series_pd: pd.DataFrame) -> pd.DataFrame:
    # Ensure the data is in time order
    series_pd = series_pd.sort_values(by='ds')
    
    # Extract target and exogenous variables
    y = series_pd['y'].values
    exog = series_pd[exog_columns].values if exog_columns else None
    
    # Fit the model using auto_arima
    auto_model = auto_arima(y, exogenous=exog, seasonal=True, m=52, 
                        alpha=0.05, trend='ct',
                        max_D=1,  # Allow auto_arima to consider up to 1 order of seasonal differencing
                        trace=True, error_action='warn', # 'ignore', 'warn', and 'raise' are the error_action options.
                        suppress_warnings=False, stepwise=True)

    
    # You can adjust auto_arima parameters as needed for your specific use case
    
    # Extract model parameters or other relevant information
    best_params = auto_model.get_params()
    
    # Return a DataFrame with the ID and the best parameters as a JSON string
    return pd.DataFrame({
        'TimeSeriesID': [series_pd['TimeSeriesID'].iloc[0]],
        'best_params': [str(best_params)]  # Convert the params dict to string for easy representation
    })

# COMMAND ----------

# timestamp
t0 = time.time()
print("start time: ", t0)

# Group by 'ID_col' and apply the Pandas UDF
model_params_df = trimmed_hol_ohe_df.groupBy('TimeSeriesID').apply(autofit_arima)
model_params_df.show(truncate=False)

t1 = time.time()
print("end time: ", t1)
print("Runtime (sec): ", (t1 - t0))
print("Runtime (min): ", (t1 - t0) / 60)

# COMMAND ----------

dbutils.notebook.exit("stop")

# COMMAND ----------

# The following version of the Pandas UDF has print statements for exceptions.

# COMMAND ----------

@pandas_udf('TimeSeriesID string, best_params string', PandasUDFType.GROUPED_MAP)
def auto_fit_arima(series_pd: pd.DataFrame) -> pd.DataFrame:
    try:
        # Ensure the data is in time order
        series_pd = series_pd.sort_values(by='ds')
        
        # Extract target and exogenous variables
        y = series_pd['y'].values
        exog = series_pd[exog_columns].values if exog_columns else None
        
        # Fit the model using auto_arima
        auto_model = auto_arima(y, exogenous=exog, seasonal=True, m=52, alpah=0.05, trend='ct',
                                max_D=1,  # Allow auto_arima to consider up to 1 order of seasonal differencing
                                trace=True, error_action='warn', # 'ignore', 'warn', and 'raise' are the error_action options.
                                suppress_warnings=False, stepwise=True)
        
        # Extract model parameters or other relevant information
        best_params = auto_model.get_params()
        
        # Format the parameters as a string for the DataFrame output
        best_params_str = str(best_params)
        
    except Exception as e:
        # Handle any exceptions that occur during model fitting
        print(f"Error fitting series {series_pd['TimeSeriesID'].iloc[0] if not series_pd.empty else 'Unknown'}: {str(e)}")
        traceback.print_exc()
        
        # Use a placeholder to indicate a fitting error
        best_params_str = 'Error fitting model'
        
    # Return a DataFrame with the ID and the best parameters (or error indication)
    return pd.DataFrame({
        'TimeSeriesID': [series_pd['TimeSeriesID'].iloc[0] if not series_pd.empty else 'Unknown'],
        'best_params': [best_params_str]
    })


# COMMAND ----------

# timestamp
t0 = time.time()
print("start time: ", t0)

# Group by 'ID_col' and apply the Pandas UDF
model_params_df = trimmed_hol_ohe_df.groupBy('TimeSeriesID').apply(auto_fit_arima)
model_params_df.show(truncate=False)

t1 = time.time()
print("end time: ", t1)
print("Runtime (sec): ", (t1 - t0))
print("Runtime (min): ", (t1 - t0) / 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Count and separate series with p, d, q, P, D, and Q equal to zero.

# COMMAND ----------

# MAGIC %md
# MAGIC >###First, check that the 'best_params' column contains only dictionaries, as expected.

# COMMAND ----------

@pandas_udf("boolean")
def is_acceptable_format(params_series: pd.Series) -> pd.Series:
    def check_format(params_str):
        try:
            params = ast.literal_eval(params_str)
            # Add more specific checks here if needed, e.g., checking types or key existence
            required_keys = ['maxiter', 'method', 'order', 'out_of_sample_size', 'scoring', 'seasonal_order', 'trend', 'with_intercept']
            return all(key in params for key in required_keys)
        except:
            return False  # Return False if parsing fails or format is not as expected
    
    return params_series.apply(check_format)

# COMMAND ----------

model_params_df = model_params_df.withColumn("is_acceptable_format", is_acceptable_format(F.col("best_params")))

# COMMAND ----------



# COMMAND ----------

# Define the Pandas UDF
@pandas_udf(BooleanType())
def check_zeros_pandas_udf(params_series: pd.Series) -> pd.Series:
    def check_zeros(params_str):
        try:
            params = ast.literal_eval(params_str)
            order, seasonal_order = params['order'], params['seasonal_order']
            return order == (0, 0, 0) and seasonal_order[:3] == (0, 0, 0)
        except:
            return False  # Ensure a boolean return even in case of parsing failure
    
    # Apply the function vectorized across the Series
    return params_series.apply(check_zeros)

# COMMAND ----------

# Apply the Pandas UDF to the DataFrame
model_params_df = model_params_df.withColumn("is_all_zeros", check_zeros_pandas_udf("best_params"))

# Filter the DataFrame to only include series with all zeros
all_zeros_df = model_params_df.filter(model_params_df["is_all_zeros"])

# Count the number of series with all zeros
num_series_all_zeros = all_zeros_df.count()
print(f"Number of series with all zeros for parameters: {num_series_all_zeros}")

# COMMAND ----------

# Define the Pandas UDF using the 'apply' method for a Series
@pandas_udf(BooleanType())
def check_zeros_pandas_udf(params_series: pd.Series) -> pd.Series:
    def check_zeros(params_str):
        try:
            # Assuming the string is in JSON format; otherwise, adjust the parsing method accordingly
            params = json.loads(params_str)
            order = params.get('order', [])
            seasonal_order = params.get('seasonal_order', [])
            return order[:3] == [0, 0, 0] and seasonal_order[:3] == [0, 0, 0]
        except Exception as e:
            # Log the error or handle it as appropriate
            print(f"Error parsing params: {e}")
            return False
    
    return params_series.apply(check_zeros)

# COMMAND ----------

# Apply the Pandas UDF to the DataFrame
model_params_df = model_params_df.withColumn("is_all_zeros", check_zeros_pandas_udf("best_params"))

# Filter the DataFrame to only include series with all zeros
all_zeros_df = model_params_df.filter(model_params_df["is_all_zeros"])

# Count the number of series with all zeros
num_series_all_zeros = all_zeros_df.count()
print(f"Number of series with all zeros for parameters: {num_series_all_zeros}")

# COMMAND ----------

# Filter the trimmed_hol_ohe_df (the training set)

# apz_training_df is "all-params-zero training df" and is the df to be used
#   for forecasting these series using a different method.
apz_training_df = trimmed_hol_ohe_df.join(model_params_df.filter("is_all_zeros"), ["TimeSeriesID"], "inner")