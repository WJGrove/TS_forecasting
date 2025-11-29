# Databricks notebook source
# MAGIC %md
# MAGIC # Imports, Functions, and Variables

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Forecast Files/demand_forecasting/general_forecasting_imports"

# COMMAND ----------

t0 = time.time()
print("start time: ", t0)

# COMMAND ----------

# MAGIC %run "/Workspace/Users/sgrove@drinkmilos.com/Forecast Files/demand_forecasting/general_forecasting_functions"

# COMMAND ----------

# DEBUGGING 

def find_duplicate_dates(df, time_series_col, date_col, index_col=None):
    """
    Identifies duplicate dates in a DataFrame grouped by a time series identifier
    and optionally an index column. Prints the count of duplicates per series and
    the rows where the duplicates occur.

    Parameters:
        df (DataFrame): The input Spark DataFrame.
        time_series_col (str): The column name representing the time series identifier.
        date_col (str): The column name representing the date field.
        index_col (str, optional): The column name representing the index field.
                                   If None, the index field is ignored.
    """
    # Group by time series and date, optionally including the index field
    group_cols = [time_series_col, date_col]
    if index_col:
        group_cols.append(index_col)
    
    # Count occurrences of each (time_series_id, ds, [index_col]) grouping
    duplicate_counts = df.groupBy(*group_cols).count().filter(F.col("count") > 1)
    
    # Count total number of series in the input DataFrame
    total_series = df.select(time_series_col).distinct().count()
    print(f"Total number of series in the DataFrame: {total_series}")
    
    # Count the number of series that have duplicates
    series_with_duplicates = duplicate_counts.select(time_series_col).distinct().count()
    print(f"Number of series with duplicate dates: {series_with_duplicates}")

    # Count duplicates per series
    duplicates_per_series = duplicate_counts.groupBy(time_series_col).agg(
        F.count("*").alias("duplicate_count")
    )
    
    # Show the counts of duplicates per series
    print("Count of duplicate dates per series:")
    duplicates_per_series.show(truncate=False)
    
    # Find and display the rows with duplicate dates
    if index_col:
        print("Rows with duplicate dates (including index):")
        duplicates = df.join(
            duplicate_counts.select(time_series_col, date_col, index_col),
            on=group_cols,
            how="inner"
        )
    else:
        print("Rows with duplicate dates (excluding index):")
        duplicates = df.join(
            duplicate_counts.select(time_series_col, date_col),
            on=group_cols[:2],  # Only time_series_col and date_col
            how="inner"
        )
    
    duplicates = duplicates.orderBy(time_series_col, date_col)
    duplicates.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Variables/Inputs

# COMMAND ----------



# COMMAND ----------

fiscal_period = "p12" # add a join to a calendar in the query that creates the source view and get rid of this.
fiscal_year = "2024" # add a join to a calendar in the query that creates the source view and get rid of this.

forecasting_files_save_location = "forecast_dev.data_science"
local_timezone = pytz.timezone("America/Chicago")

group_col = 'time_series_id' # origin: soldto_id + item_id_fc I.e., distribution center and item.
value_col = "y_clean_int_transformed" # expanded, outliers removed, interpolated, and transformed
date_col = 'ds' # origin: 'req_del_fw_start_date'

# Define comparable series (comps):
# When the Nielsen data is ingested, we will integrate location/market data to further improve the process.
comp_def_col_1 = "parent_company_fc"
comp_def_col_2 = "item_id_fc"
# comp_def_col_3 = "[SOME KIND OF LOCATION/MARKET DATA]" 
# A FEW OTHER CHANGES WILL NEED TO BE MADE TO THE CELLS RELATED TO COMPS WHEN THIS DATA IS ADDED.

# COMMAND ----------

forecast_horizon = 104 # weeks
comp_group_default_yoy = 0.03 # This is used to forecast more than a year out when there aren't any comps for a series.
fc_ci_outlier_threshold = 3 # standard deviations
short_series_threshold = 52 # weeks
test_set_length = 26 # weeks

# COMMAND ----------

# MAGIC %md
# MAGIC #Retrieve Data

# COMMAND ----------

# Grab transformed sales data
sales_preprocessed_df = spark.sql(
    "SELECT * FROM forecast_dev.data_science.sales_preprocessed"
)
sales_preprocessed_df.show(5)
print(f"'sales_preprocessed_df' length = {sales_preprocessed_df.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split

# COMMAND ----------

# MAGIC %md
# MAGIC >### Test

# COMMAND ----------

# Define the current date and calculate last Sunday
today = datetime.now(local_timezone).date()
last_sunday = today - timedelta(days=today.weekday() + 1)
test_start_date = last_sunday - timedelta(weeks=test_set_length)  # Adjust weeks if needed
print(f"today's date: {today}")
print(f"last Sunday: {last_sunday}")
print(f"test_set_length: {test_set_length} weeks")
print(f"test set start date: {test_start_date}")
print()

# Filter for test and train sets
test_df = sales_preprocessed_df.filter(
    (F.col("ds") >= F.lit(test_start_date)) & (F.col("ds") < F.lit(last_sunday))
)
print(f"'test_df' length = {test_df.count()}")
print()

# COMMAND ----------

# MAGIC %md
# MAGIC >### Train

# COMMAND ----------

train_df = sales_preprocessed_df.filter(F.col("ds") < F.lit(test_start_date))
print(f"'train_df' length = {train_df.count()}")

# count the lengths of the training series.
training_series_lengths_all = train_df.groupBy(group_col).agg(
    F.count('ds').alias('training_series_length')
)
# find the max date for the training series
training_series_max_dates = train_df.groupBy(group_col).agg(
    F.max('ds').alias('training_series_max_date')
)
# add columns for the length and max date to the train_df
train_df = train_df.join(
    training_series_lengths_all,
    on=group_col,
    how='left'
    ).join(
        training_series_max_dates,
        on=group_col,
        how='left'
)

# add a boolean for 'is_short_training_series'
train_df = train_df.withColumn(
    'is_short_training_series',
    F.when(train_df['training_series_length'] < short_series_threshold, True).otherwise(False)
)

# check
# print(f"test_start_date: {test_start_date}")
# train_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #Separate by Length

# COMMAND ----------

# MAGIC %md
# MAGIC >##Exponential smoothing doesn't always work well with very short series, so we need to forecast them separately.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Short Training Series

# COMMAND ----------

# # Count all of the time series
# total_series_count = training_series_lengths_all.count()
# print(f"Total number of training series: {total_series_count}")

# Isolate the short series' data
strain_df = train_df.filter(F.col("is_short_training_series") == True)

# Add columns to hold the respective totals of 'y_clean_int' and 'y_clean_int_transformed' for each 'time_series_id'.
short_training_series_totals = strain_df.groupBy(group_col).agg(
    F.sum("y_clean_int").alias("series_total_cases"),
    F.sum(value_col).alias("sum_of_transformed_series")
)
strain_df = strain_df.join(short_training_series_totals, on=group_col, how="left")

# # Count the short series.
# short_training_series_count = strain_df.select(group_col).distinct().count()
# print(f"Number of short training series: {short_training_series_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Long Training Series

# COMMAND ----------

ltrain_df = train_df.join(strain_df.select(group_col).distinct(), on=group_col, how="left_anti")

# # Count the long series.
# long_training_series_count = ltrain_df.select(group_col).distinct().count()
# print(f"Number of long training series: {long_training_series_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Prepare Comp Data

# COMMAND ----------

# MAGIC %md
# MAGIC >We just want the data for each comp's first year, so we will filter out series having first date equal to the minimum date of the entire set. This will ensure that we're looking at the first year of the series and not just the first year that appears in this data set; some series go back further than this data.

# COMMAND ----------

# Remove all series that did not start after the minimum date of the entire set.
# This will ensure that we aren't assuming that the series being present during the earliest week of the set
#   means it began on that week.
minimum_lts_date = ltrain_df.agg(F.min("ds")).collect()[0][0]
comps_data = ltrain_df.filter(F.col("series_min_date") > F.lit(minimum_lts_date))

# Filter to just the first year of each comp
comp_first_year_data = comps_data.filter(
    (F.datediff("ds", "series_min_date") >= 0) & (F.datediff("ds", "series_min_date") < 365)
) # we already know they're at least a year long

# add a col to hold the totals for the series' first years
# Calculate the total
sum_first_year_df = comp_first_year_data.groupBy(group_col).agg(
    F.sum("y_clean_int").alias("sum_y_clean_int_first_year")
)
# Join the sum for the first year back to the original DataFrame
comp_first_year_df = comp_first_year_data.join(
    sum_first_year_df
    , on=group_col
    , how="inner"
)

# Create the volume-normalized value column
comp_first_year_df = comp_first_year_df.withColumn(
    "normalized_y_clean_int", F.col("y_clean_int") / F.col("sum_y_clean_int_first_year")
)

# Add a numerical index by 'time_series_id', so we can reindex to align series with different start dates.
# Define a window specification for ordering and indexing
window_spec = Window.partitionBy(group_col).orderBy("ds")
# add index column
comp_first_year_df = comp_first_year_df.withColumn(
    "cmp_series_first_year_index", F.row_number().over(window_spec)
)

# Calculate the average of 'normalized_y_clean_int' for each week by comp group.
# This gives us the average proportion of the first year's volume represented by each of the first 52 weeks.
# I.e., It gives us the average shape of the first year's volume distribution for a given item and company.
average_shape_df = comp_first_year_df.groupBy(comp_def_col_1, comp_def_col_2, "cmp_series_first_year_index").agg(
    F.avg("normalized_y_clean_int").alias("cmpgrp_avg_normalized_y_clean_int"),
    F.expr("percentile_approx(normalized_y_clean_int, 0.5)").alias("cmpgrp_med_normalized_y_clean_int")
)

# Join the average back to the original DataFrame
comp_first_year_df = comp_first_year_df.join(
    average_shape_df, on=[comp_def_col_1, comp_def_col_2, "cmp_series_first_year_index"], how="left"
)



# #DEBUGGING


# print(f"comp_first_year_df length: {comp_first_year_df.count()}")
# no_of_comps = comp_first_year_df.select(group_col).distinct().count()
# print(f"Number of unique comps: {no_of_comps}")
# avg_len_of_comp_index = comp_first_year_df.count()/no_of_comps # should be 52.000000000
# print(avg_len_of_comp_index)
# comp_first_year_df.show(60)

# #-------------------------------------------------------
# # Filter for a specific time_series_id and select required columns
# filtered_df = comp_first_year_df.filter(F.col("time_series_id") == "ADUS079460000").select(
#     "time_series_id", "cmp_series_first_year_index", "ds"
# )
# # Show all records for the filtered time series
# filtered_df.show(filtered_df.count(), truncate=False)
# #-------------------------------------------------------

# Calculate the max index for each series
series_max_index = comp_first_year_df.groupBy("time_series_id").agg(
    F.max("cmp_series_first_year_index").alias("max_index")
)

# Count the number of series with max index = 52
count_52 = series_max_index.filter(F.col("max_index") == 52).count()
print(f"Number of series with max index = 52: {count_52}")

# Count the number of series with max index > 52
count_53_or_more = series_max_index.filter(F.col("max_index") > 52).count()
print(f"Number of series with max index > 52: {count_53_or_more}")

# Count the number of series with max index > 53
count_54_or_more = series_max_index.filter(F.col("max_index") > 53).count()
print(f"Number of series with max index > 53: {count_54_or_more}")

# Get the IDs of the series with max index > 52
ids_with_more_than_52 = series_max_index.filter(F.col("max_index") > 52).select("time_series_id")
print("IDs of series with max index > 52:")
ids_with_more_than_52.show(150, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #Visualize Short Series and Comps

# COMMAND ----------

# Convert the entire strain_df to a pandas DataFrame
strain_df_pd = strain_df.toPandas()

# Plot each series separately
plt.figure(figsize=(12, 8))
for key, grp in strain_df_pd.groupby("time_series_id"):
    plt.plot(grp["ds"], grp["y_clean_int"], label=key)
plt.xlabel("Date")
plt.ylabel("Cases")
plt.title(f"Recently Added (<{short_series_threshold} wks) PODs")
plt.grid(True)
plt.show()

# COMMAND ----------

# Group by comp_def_col_1 and comp_def_col_2 to get the count of unique time_series_id for the comp group
ss_freq_by_compgrp = strain_df.groupBy(comp_def_col_2, comp_def_col_1).agg(
    F.countDistinct(group_col).alias("company_item_freq")
)
# Add the frequency column to the df
short_series_with_freq_df = strain_df.join(
    ss_freq_by_compgrp, on=[comp_def_col_2, comp_def_col_1], how="left"
)


# COMMAND ----------

# MAGIC %md
# MAGIC >3-D plot of short series frequency by comp group:

# COMMAND ----------

# Convert the short series df with the frequencies by comp group to a Pandas df.
short_series_freq_df_pandas = short_series_with_freq_df.toPandas()

# Get the unique values for comp_def_col_1 and item_id_fcs
customer_companies = sorted(
    short_series_freq_df_pandas[comp_def_col_1].unique(), reverse=True
)
item_numbers = sorted(short_series_freq_df_pandas[comp_def_col_2].unique(), key=int)

# Map the comp_def_col_1 and item_id_fc to integer indices
customer_company_map = {company: idx for idx, company in enumerate(customer_companies)}
item_number_map = {item: idx for idx, item in enumerate(item_numbers)}

# Create a frequency matrix initialized to zeros
frequency_matrix = np.zeros((len(customer_companies), len(item_numbers)))

# Fill the frequency matrix with the number of unique time_series_id
for _, row in short_series_freq_df_pandas.iterrows():
    company_idx = customer_company_map[row[comp_def_col_1]]
    item_idx = item_number_map[str(row[comp_def_col_2])]
    frequency_matrix[company_idx, item_idx] = row["company_item_freq"]

# Create the 3D bar plot
fig = plt.figure(figsize=(24, 12))
ax = fig.add_subplot(111, projection="3d")

# Create a mesh grid
x, y = np.meshgrid(np.arange(len(customer_companies)), np.arange(len(item_numbers)))
x = x.flatten()
y = y.flatten()
z = frequency_matrix.T.flatten()  # Transpose to match the orientation

# Create a colormap for item numbers
cmap = plt.cm.get_cmap("tab20", len(item_numbers))

# Plot the bars
dx = dy = 0.8  # Width and depth of the bars
dz = z

for i in range(len(z)):
    ax.bar3d(
        x[i], y[i], 0, dx, dy, dz[i], color=cmap(y[i] / len(item_numbers)), shade=True
    )

# Set the labels
ax.set_xticks(np.arange(len(customer_companies)))
ax.set_xticklabels(customer_companies, rotation=270)
ax.set_yticks(np.arange(len(item_numbers)))
ax.set_yticklabels(item_numbers, rotation=45)
ax.set_zlabel("Number of Series")
ax.set_title(f"Frequency of Short Series by {comp_def_col_1} and {comp_def_col_2}")

# Create a custom legend
handles = [
    plt.Line2D([0], [0], color=cmap(i / len(item_numbers)), lw=4)
    for i in range(len(item_numbers))
]
labels = item_numbers
ax.legend(
    handles, labels, title="Item Number", bbox_to_anchor=(0.9, 1), loc="upper left"
)

# Adjust the perspective
ax.view_init(elev=15, azim=60)

# Adjust layout
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC >3-D plot of comp frequency by comp group:

# COMMAND ----------

# Convert the necessary short-series columns to lists for filtering the long-series data 
strain_item_numbers_list = (
    short_series_with_freq_df.select("item_id_fc")
    .distinct()
    .rdd.flatMap(lambda x: x)
    .collect()
)
strain_customer_companies_list = (
    short_series_with_freq_df.select("parent_company_fc")
    .distinct()
    .rdd.flatMap(lambda x: x)
    .collect()
)
# strain_market/area/region_list = (
#     short_series_with_freq_df.select("market/area/region column name goes here")
#     .distinct()
#     .rdd.flatMap(lambda x: x)
#     .collect()
# )

# Filter comp_first_year_df based on the item_id_fc and parent_company_fc present in the short series
comps_first_year_df = comp_first_year_df.filter(
    (comp_first_year_df["item_id_fc"].isin(strain_item_numbers_list))
    & (comp_first_year_df["parent_company_fc"].isin(strain_customer_companies_list))
)

# Group by 'item_id_fc' and 'parent_company_fc' to get the count of unique time_series_id in the comp groups
grouped_comps_first_year_df = comps_first_year_df.groupBy(
    "item_id_fc", "parent_company_fc"
).agg(F.countDistinct("time_series_id").alias("company_item_freq"))

# Convert the DataFrame to Pandas
filtered_df_pandas = grouped_comps_first_year_df.toPandas()

# Get the unique values for parent_company_fc and sorted item_id_fc
customer_companies = sorted(
    filtered_df_pandas["parent_company_fc"].unique(), reverse=True
)
# item_numbers = sorted(filtered_df_pandas["item_id_fc"].unique(), key=int)
# This is commented out because it should only include those item numbers found in the short series.
# This means we can use the list created in the cell plotting the short series freq by comp groups (above).

# Map the parent_company_fc and item_id_fc to integer indices
customer_company_map = {company: idx for idx, company in enumerate(customer_companies)}
item_number_map = {item: idx for idx, item in enumerate(item_numbers)}

# Create a frequency matrix initialized to zeros
frequency_matrix = np.zeros((len(customer_companies), len(item_numbers)))
# Fill the frequency matrix with the number of unique time_series_id
for _, row in filtered_df_pandas.iterrows():
    company_idx = customer_company_map[row["parent_company_fc"]]
    item_idx = item_number_map[str(row["item_id_fc"])]
    frequency_matrix[company_idx, item_idx] = row["company_item_freq"]

# Create the 3D bar plot
fig = plt.figure(figsize=(24, 12))
ax = fig.add_subplot(111, projection="3d")
# Create a mesh grid
x, y = np.meshgrid(np.arange(len(customer_companies)), np.arange(len(item_numbers)))
x = x.flatten()
y = y.flatten()
z = frequency_matrix.T.flatten()  # Transpose to match the orientation

# Create a colormap for item numbers
cmap = plt.cm.get_cmap("tab20", len(item_numbers))
# Plot the bars
dx = dy = 0.8  # Width and depth of the bars
dz = z
for i in range(len(z)):
    ax.bar3d(
        x[i], y[i], 0, dx, dy, dz[i], color=cmap(y[i] / len(item_numbers)), shade=True
)
# Set the labels
ax.set_xticks(np.arange(len(customer_companies)))
ax.set_xticklabels(customer_companies, rotation=270)
ax.set_yticks(np.arange(len(item_numbers)))
ax.set_yticklabels(item_numbers, rotation=45)
ax.set_zlabel("Number of Series")
ax.set_title("Frequency of Short Series' Comps")

# Create a custom legend
handles = [
    plt.Line2D([0], [0], color=cmap(i / len(item_numbers)), lw=4)
    for i in range(len(item_numbers))
]
labels = item_numbers
ax.legend(
    handles, labels, title="Item ID", bbox_to_anchor=(0.9, 1), loc="upper left"
)

# Adjust the perspective
ax.view_init(elev=15, azim=60)

# Adjust layout
plt.show()

# COMMAND ----------

# Plot histogram of series lengths
plt.figure(figsize=(8, 6))
plt.hist(strain_df_pd["series_length"], bins=30, alpha=0.75)
plt.xlabel("Length of Time Series")
plt.ylabel("Frequency")
plt.title("Histogram of Short Series Lengths")
plt.grid(True)
plt.show()

# Plot histogram of series volumes
plt.figure(figsize=(8, 6))
plt.hist(strain_df_pd["series_total_cases"], bins=30, alpha=0.75)
plt.xlabel("Total Volume of Time Series")
plt.ylabel("Frequency")
plt.title("Histogram of Short Series Volumes")
plt.grid(True)
plt.show()

# Plotting the relationship between length and volume
plt.figure(figsize=(8, 6))
plt.scatter(strain_df_pd["series_length"], strain_df_pd["series_total_cases"], alpha=0.75)
plt.xlabel("Length of Time Series")
plt.ylabel("Total Volume of Time Series")
plt.title("Series Volume vs. Length")
plt.grid(True)
plt.show()

# Distribution among different 'bottle_type' values
plt.figure(figsize=(8, 6))
customer_company_counts = strain_df_pd.groupby("bottle_type")[
    "time_series_id"
].nunique()
customer_company_counts.plot(kind="bar", alpha=0.75)
plt.xlabel("Bottle Type")
plt.ylabel("Number of Unique Time Series")
plt.title("Frequency of Short Series by Bottle Size")
plt.grid(True)
plt.show()

# This visual is commented out because the next one is more useful.
# Distribution among different 'item_id_fc' values
plt.figure(figsize=(8, 6))
item_number_counts = strain_df_pd.groupby("item_id_fc")[
    "time_series_id"
].nunique()
item_number_counts.plot(kind="bar", alpha=0.75)
plt.xlabel("Item Number")
plt.ylabel("Number of Unique Time Series")
plt.title("Frequency of Short Series by Item Number")
plt.grid(True)
plt.show()

# Distribution among different 'item_desc' values
# Get counts for 'item_desc' but ensure they are ordered by 'item_id_fc'
#   ; this will have the affect of grouping the bottle sizes together because of how we assign item numbers.

# item_counts = strain_df_pd.groupby(["item_id_fc", "item_desc"])["time_series_id"].nunique().reset_index(name="count")
# # Sort by 'item_id_fc'
# item_counts_sorted = item_counts.sort_values(by="item_id_fc")
# # Set 'item_desc' as the index
# item_counts_sorted = item_counts_sorted.set_index("item_desc")
# # Distribution among different 'item_desc' values ordered by 'item_id_fc'
# plt.figure(figsize=(8, 6))
# item_counts_sorted["count"].plot(kind="bar", alpha=0.75)
# plt.xlabel("Item Description")
# # plt.xticks(rotation=45)
# plt.ylabel("Number of Unique Time Series")
# plt.title("Frequency of Short Series by Item Description")
# plt.grid(True)
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Define a Sample of Short Series for Exploration

# COMMAND ----------

# here's a list of options for defining samples to look at
print(f"Customer companies appearing in the short series data:\n\n{strain_customer_companies_list} \
      \n\nItem numbers appearing in the short series data:\n\n{strain_item_numbers_list}") #\
      # \n\nmarkets/areas/regions appearing in the short series data:\n\n{strain_market/area/region_list}")

# COMMAND ----------


sample_parent_companies = ["KROGER", "AWG", "C&S WHOLESALE GROC"]  # strain_customer_companies_list
sample_item_ids = ["60020", "61005"]  # strain_item_numbers_list
# sample_market/area/region = [] # strain_market/area/region_list

# COMMAND ----------

# MAGIC %md
# MAGIC >Sample Info

# COMMAND ----------

# Filter the DataFrame
short_series_sample_df = strain_df.filter(
    (strain_df["parent_company_fc"].isin(sample_parent_companies))
    & (strain_df["item_id_fc"].isin(sample_item_ids))
)

# Display distinct time series IDs and product descriptions
sample_series_information = short_series_sample_df.select(
    "time_series_id", "bottle_type", "training_series_length", "series_min_date" 
).distinct()
sample_series_information = sample_series_information.orderBy("time_series_id")
sample_series_information.show(50, truncate=False)
print(f"No. of short series in sample: {sample_series_information.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC >###Sample Visualizations

# COMMAND ----------

# Convert the DataFrame to Pandas
short_series_sample_df_pd = short_series_sample_df.toPandas()

# Plot each series separately
plt.figure(figsize=(12, 6))
for time_series_id in short_series_sample_df_pd["time_series_id"].unique():
    series_data = short_series_sample_df_pd[
        short_series_sample_df_pd["time_series_id"] == time_series_id
    ]
    plt.plot(series_data["ds"], series_data["y_clean_int"], label=time_series_id)

plt.title('Short Series Sample')
plt.xlabel("ds")
plt.ylabel("Cases Ordered")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC > Color coded by comp_def_col_1:

# COMMAND ----------

# select a color coding column
sample_color_grouping_column = comp_def_col_1 
# identify unique groups based on the sample_color_grouping_column
unique_color_groups = short_series_sample_df.select(sample_color_grouping_column).distinct().toPandas()[sample_color_grouping_column].tolist()


# Generate color map
colors = plt.cm.get_cmap('tab20', len(unique_color_groups)) 
group_color_map = {group: colors(i) for i, group in enumerate(unique_color_groups)}


# plot color coded by group
plt.figure(figsize=(14, 6))

for group in unique_color_groups:
    # Filter data for the current group
    group_data = short_series_sample_df_pd[short_series_sample_df_pd[sample_color_grouping_column] == group]
    
    # Plot each time series within the group separately to avoid connecting lines between different series
    for time_series_id in group_data["time_series_id"].unique():
        series_data = group_data[group_data["time_series_id"] == time_series_id]
        plt.plot(series_data["ds"], series_data["y_clean_int"], 
                 label=f"{group} - {time_series_id}", color=group_color_map[group], alpha=0.7)
    
# Adding labels, title, and legend
plt.xlabel("Date")
plt.ylabel("Transformed Value")
plt.title(f"Short Series Sample - Color Coded by {sample_color_grouping_column}")
plt.legend(title=sample_color_grouping_column, loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC > Color coded by comp_def_col_2:

# COMMAND ----------

# select a color coding column
sample_color_grouping_column = comp_def_col_2
# identify unique groups based on the sample_color_grouping_column
unique_color_groups = short_series_sample_df.select(sample_color_grouping_column).distinct().toPandas()[sample_color_grouping_column].tolist()


# Generate a color map
colors = plt.cm.get_cmap('tab20', len(unique_color_groups)) 
group_color_map = {group: colors(i) for i, group in enumerate(unique_color_groups)}


# plot color coded by group
plt.figure(figsize=(12, 6))

for group in unique_color_groups:
    # Filter data for the current group
    group_data = short_series_sample_df_pd[short_series_sample_df_pd[sample_color_grouping_column] == group]
    
    # Plot each time series within the group separately to avoid connecting lines between different series
    for time_series_id in group_data["time_series_id"].unique():
        series_data = group_data[group_data["time_series_id"] == time_series_id]
        plt.plot(series_data["ds"], series_data["y_clean_int"], 
                 label=f"{group} - {time_series_id}", color=group_color_map[group], alpha=0.7)
    
# Adding labels, title, and legend
plt.xlabel("Date")
plt.ylabel("Transformed Value")
plt.title(f"Short Series Sample - Color Coded by {sample_color_grouping_column}")
plt.legend(title=sample_color_grouping_column, loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC > Color coded by comp_def_col_3:

# COMMAND ----------

# # select a color coding column
# sample_color_grouping_column = comp_def_col_3
# # identify unique groups based on the sample_color_grouping_column
# unique_color_groups = short_series_sample_df.select(sample_color_grouping_column).distinct().toPandas()[sample_color_grouping_column].tolist()


# # Generate a color map
# colors = plt.cm.get_cmap('tab20', len(unique_color_groups)) 
# group_color_map = {group: colors(i) for i, group in enumerate(unique_color_groups)}


# # plot color coded by group
# plt.figure(figsize=(12, 6))

# for group in unique_color_groups:
#     # Filter data for the current group
#     group_data = short_series_sample_df_pd[short_series_sample_df_pd[sample_color_grouping_column] == group]
    
#     # Plot each time series within the group separately to avoid connecting lines between different series
#     for time_series_id in group_data["time_series_id"].unique():
#         series_data = group_data[group_data["time_series_id"] == time_series_id]
#         plt.plot(series_data["ds"], series_data["y_clean_int"], 
#                  label=f"{group} - {time_series_id}", color=group_color_map[group], alpha=0.7)
    
# # Adding labels, title, and legend
# plt.xlabel("Date")
# plt.ylabel("Transformed Value")
# plt.title(f"Short Series Sample - Color Coded by {sample_color_grouping_column}")
# plt.legend(title=sample_color_grouping_column, loc="upper left", bbox_to_anchor=(1, 1))
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC >Plot the sample's comps' first 52 weeks aligned in time to look for patterns.
# MAGIC
# MAGIC >We'll use the normalized values allow us to focus on the shapes of the comps' first years. These groups of comps are used to create the forecasts for the short series.  

# COMMAND ----------

# filter the comp data to match the sample definition 
sample_comp_first_year_df = comp_first_year_df.filter(
    (comp_first_year_df["parent_company_fc"].isin(sample_parent_companies))
    & (comp_first_year_df["item_id_fc"].isin(sample_item_ids))
    # & (comp_first_year_df[##market/area/region##].isin(sample_markets/areas/regions))
)
sample_comp_first_year_df = sample_comp_first_year_df.orderBy(group_col, date_col)
sample_comp_first_year_df_pd = sample_comp_first_year_df.toPandas()

# Define color grouping columns and identify unique groups
color_grouping_columns = ['parent_company_fc', 'item_id_fc']# ['parent_company_fc', 'item_id_fc', 'market/area/region'] this is the future list, market/area/region data not available yet.
unique_color_groups = sample_comp_first_year_df.select(color_grouping_columns).distinct().toPandas()

# Create a color map for each unique (parent_company_fc, item_id_fc) combination
colors = plt.cm.get_cmap('tab20', len(unique_color_groups))
group_color_map = {
    tuple(row): colors(i) for i, row in enumerate(unique_color_groups[color_grouping_columns].itertuples(index=False))
}

# COMMAND ----------

# Plot each series
plt.figure(figsize=(12, 6))
for time_series_id in sample_comp_first_year_df_pd["time_series_id"].unique():
    # Filter the data for the specific time series
    series_data = sample_comp_first_year_df_pd[
        sample_comp_first_year_df_pd["time_series_id"] == time_series_id
    ]
    
    # Identify the group (tuple of `parent_company_fc` and `item_id_fc`) and its color
    group = tuple(series_data[color_grouping_columns].iloc[0])  # Convert to tuple for dictionary lookup
    color = group_color_map[group]
    
    # Plot the series with the color based on its group
    plt.plot(
        series_data["cmp_series_first_year_index"],
        series_data["y_clean_int"],
        label=f"{group[0]} - {group[1]} (ID: {time_series_id})",
        color=color,
    )

# Set y-axis limits
# plt.ylim(-0.05, 0.15)

# Add titles and labels
plt.title("First-Year Plot of Sample Comps")
plt.xlabel("First 52 Weeks")
plt.ylabel("Cases Ordered")

# Create custom handles for legend (using only unique groups)
handles = [
    plt.Line2D([0], [0], color=group_color_map[tuple(row)], lw=4) 
    for row in unique_color_groups[color_grouping_columns].itertuples(index=False)
]
labels = [f"{row[0]} - {row[1]}" for row in unique_color_groups[color_grouping_columns].itertuples(index=False)]

# Display legend with group names
plt.legend(handles, labels, title="Company - Item", loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()  # Adjust layout to fit legend outside plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC >The plot below is most useful when the sample is limited to one or two groups, especially groups with many comps.

# COMMAND ----------

# Plot each series
plt.figure(figsize=(12, 6))
for time_series_id in sample_comp_first_year_df_pd["time_series_id"].unique():
    # Filter the data for the specific time series
    series_data = sample_comp_first_year_df_pd[
        sample_comp_first_year_df_pd["time_series_id"] == time_series_id
    ]
    
    # Identify the group (tuple of `parent_company_fc` and `item_id_fc`) and its color
    group = tuple(series_data[color_grouping_columns].iloc[0])  # Convert to tuple for dictionary lookup
    color = group_color_map[group]
    
    # Plot the series with the color based on its group
    plt.plot(
        series_data["cmp_series_first_year_index"],
        series_data["normalized_y_clean_int"],
        label=f"{group[0]} - {group[1]} (ID: {time_series_id})",
        color=color,
    )

# Set y-axis limits
plt.ylim(-0.05, 0.15)

# Add titles and labels
plt.title("First-Year Plot of Sample Comps (Normalized)")
plt.xlabel("First 52 Weeks")
plt.ylabel("% of First Year")

# Create custom handles for legend (using only unique groups)
handles = [
    plt.Line2D([0], [0], color=group_color_map[tuple(row)], lw=4) 
    for row in unique_color_groups[color_grouping_columns].itertuples(index=False)
]
labels = [f"{row[0]} - {row[1]}" for row in unique_color_groups[color_grouping_columns].itertuples(index=False)]

# Display legend with group names
plt.legend(handles, labels, title="Company - Item", loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()  # Adjust layout to fit legend outside plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Forecast Short Series

# COMMAND ----------

# MAGIC %md
# MAGIC >## To forecast a short series:
# MAGIC
# MAGIC > We will use the mean/median shape of its comps in combination with the series' total volume to-date to calculate an estimated total for the series' first year.
# MAGIC
# MAGIC > The difference between the first-year estimated total and the series' volume to-date will be distributed throughout the remaining weeks of the series' first year (using the shape of the comps); this creates a forecast for the remainder of the first year. Now this forecast needs to be extended to a length of 104 weeks.
# MAGIC
# MAGIC > Over a two-year period, it is reasonable to expect some additional growth on top of the initial ramp, so - since we are building the short series' forecasts manually - we must include some type of growth rate in the back half of that two-year period. This growth rate will be calculated from the series' comps. If a series doesn't have comps, a default rate will be assigned (defined in the 'Inputs' section at the top of this notebook). This growth rate will be applied to the first year's est. total prior to being distributed across the second year of the forecast.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Calculate comp growth rates and growth rate stats.

# COMMAND ----------

# MAGIC %md
# MAGIC >First, calculate each series' growth rate for all series at least two years long.

# COMMAND ----------

# Filter long training series to those at least 104 weeks long
two_year_min_series_lengths = training_series_lengths_all.filter(F.col('training_series_length') >= 104)
two_year_min_df = ltrain_df.join(two_year_min_series_lengths, on=group_col, how='inner')

# calculate and add series max date col
max_dates_df = two_year_min_df.groupBy(group_col).agg(F.max(date_col).alias("max_date"))
two_year_min_df = two_year_min_df.join(max_dates_df, on=group_col)

# calculate the sums for the recent and previous 52-week periods
recent_52_df = two_year_min_df.filter(F.col(date_col) >= (F.col("max_date") - F.expr("INTERVAL 52 WEEKS"))) \
    .groupBy(group_col) \
    .agg(F.sum("y_clean_int").alias("sum_recent_52"))

previous_52_df = two_year_min_df.filter(
    (F.col(date_col) < (F.col("max_date") - F.expr("INTERVAL 52 WEEKS"))) &
    (F.col(date_col) >= (F.col("max_date") - F.expr("INTERVAL 104 WEEKS")))
).groupBy(group_col) \
    .agg(F.sum("y_clean_int").alias("sum_previous_52"))

# join recent and previous 52-week period sums and calculate and add series YoY growth rate col
yoy_df = recent_52_df.join(previous_52_df, on=group_col, how="inner") \
    .withColumn("series_yoy_growth_rate", ((F.col("sum_recent_52") - F.col("sum_previous_52")) / F.col("sum_previous_52")))

two_year_min_df = two_year_min_df.join(yoy_df.select(group_col, "series_yoy_growth_rate"), on=group_col, how="left")
# two_year_min_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC >Group long training series into comp groups and calculate YoY growth rate stats for each group. Then, create a df to hold all comp groups and their growth rate stats.

# COMMAND ----------

comp_group_yoy_stats = two_year_min_df.groupBy(comp_def_col_1, comp_def_col_2).agg(
    F.avg('series_yoy_growth_rate').alias('cmpgrp_avg_yoy_growth_rate'),
    F.expr('percentile_approx(series_yoy_growth_rate, 0.5)').alias('cmpgrp_median_yoy_growth_rate'),
    F.count('series_yoy_growth_rate').alias('cmpgrp_count_yoy_growth_rate'),
    F.stddev('series_yoy_growth_rate').alias('cmpgrp_stddev_yoy_growth_rate'),
    F.min('series_yoy_growth_rate').alias('cmpgrp_min_yoy_growth_rate'),
    F.max('series_yoy_growth_rate').alias('cmpgrp_max_yoy_growth_rate'),
    F.expr('percentile_approx(series_yoy_growth_rate, 0.25)').alias('cmpgrp_25th_percentile_yoy_growth_rate'),
    F.expr('percentile_approx(series_yoy_growth_rate, 0.75)').alias('cmpgrp_75th_percentile_yoy_growth_rate')
)

# define the set of all possible comp groups and add the calculated rates for the groups with historicals
# This is necessary because the stats df only contains series that are at least 2 years long
# , and we want to fill in the average and median values with the default value.
all_comp_groups = ltrain_df.select(comp_def_col_1, comp_def_col_2).distinct()
# Join the average and median YoY growth rates with the universal set of comp groups
comp_group_growth_rate_df = all_comp_groups.join(
    comp_group_yoy_stats,
    on=[comp_def_col_1, comp_def_col_2],
    how='left'
)

# Fill missing values with the default YoY growth rate (defined in the "Variables" section at the top of the notebook)
comp_group_growth_rate_df = comp_group_growth_rate_df.fillna({
    'cmpgrp_avg_yoy_growth_rate': comp_group_default_yoy,
    'cmpgrp_median_yoy_growth_rate': comp_group_default_yoy
}).orderBy("parent_company_fc", ascending=False)

# print("comp_group_growth_rate_df:")
# comp_group_growth_rate_df.show(50)
# print()
# print(f"comp_group_growth_rate_df length: {comp_group_growth_rate_df.count()}")

# COMMAND ----------

# THIS IS THE POINT IN THE PROCESS IN WHICH THE PRO FORMA ESTIMATES WOULD BE DISTRIBUTED ACCORDING TO COMP GROUP SHAPE AND EST GROWTH RATE. CREATE A SEPARATE SECTION TO WORK IN UNTIL THE AGGREGATION STEP.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Attach the necessary comp data and calculate the estimated total cases ordered for each series' first year.

# COMMAND ----------

# MAGIC %md
# MAGIC >First, add a series-specific index column to the short training series data and grab the comp data to make the estimate.

# COMMAND ----------

# add an index to each series
# Create a window specification to generate row numbers for each time_series_id
window_spec_index = Window.partitionBy(group_col).orderBy(date_col)
# Add the series index
strain_df = strain_df.withColumn('short_series_index', F.row_number().over(window_spec_index)) # index begins with 1

# COMMAND ----------

# This is the average shape of the comp group's first year.
strain_with_comp_stats = strain_df.join(
    comp_first_year_df.select('cmp_series_first_year_index', 'cmpgrp_avg_normalized_y_clean_int', 'cmpgrp_med_normalized_y_clean_int'),
    on=(strain_df.parent_company_fc == comp_first_year_df.parent_company_fc) &
        (strain_df.item_id_fc == comp_first_year_df.item_id_fc) &
        (strain_df.short_series_index == comp_first_year_df.cmp_series_first_year_index),
    how='left'
)
strain_with_comp_stats = strain_with_comp_stats.join(
    comp_group_growth_rate_df.select('cmpgrp_avg_yoy_growth_rate', 'cmpgrp_median_yoy_growth_rate'),
    on=(strain_with_comp_stats.parent_company_fc == comp_group_growth_rate_df.parent_company_fc) &
        (strain_with_comp_stats.item_id_fc == comp_group_growth_rate_df.item_id_fc),
    how='left'
)

# Calculate first_year_est_total_cases

# Calculate the sum of the first n average and median normalized_y_clean_int values for each series, where n = length of the series.
sum_of_n_avg_df = strain_with_comp_stats.groupBy(group_col).agg(
    F.sum('cmpgrp_avg_normalized_y_clean_int').alias('sum_of_n_avgs')
)
sum_of_n_med_df = strain_with_comp_stats.groupBy(group_col).agg(
    F.sum('cmpgrp_med_normalized_y_clean_int').alias('sum_of_n_meds')
)
# Join this sum back to the strain_with_comp_stats
strain_with_comp_stats = strain_with_comp_stats.join(sum_of_n_avg_df, on=group_col, how='left')
strain_with_comp_stats = strain_with_comp_stats.join(
    sum_of_n_med_df.select(group_col, 'sum_of_n_meds'),
    on=group_col,
    how='left'
)

# Calculate first_year_est_total_cases using the sum of the first n average and median normalized_y_clean_int values
strain_with_comp_stats = strain_with_comp_stats.withColumn(
    'first_year_est_with_comp_avg',
    F.col('series_total_cases') / F.col('sum_of_n_avgs')
)
strain_with_comp_stats = strain_with_comp_stats.withColumn(
    'first_year_est_with_comp_med',
    F.col('series_total_cases') / F.col('sum_of_n_meds')) \
        .orderBy(group_col, date_col)

# Multiplying this number by the cmpgrp_avg_normalized_y_clean_int values for the remaining (52 - n) weeks of the first year
# will give a forecasted value for those weeks. This is because the average and median normalized_y_clean_int values are estimates of the percentage of the year's volume ordered on that week.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a df to hold the forecasts

# COMMAND ----------

# MAGIC %md
# MAGIC >Create a df with all the training series IDs, their max dates, and their lengths from the training set.

# COMMAND ----------

series_to_forecast = strain_with_comp_stats.select(group_col, "training_series_max_date", "training_series_length").distinct()

# COMMAND ----------

find_duplicate_dates(strain_with_comp_stats, group_col, date_col)

# COMMAND ----------

# MAGIC %md
# MAGIC >Generate weekly forecast dates for each series.

# COMMAND ----------

# Create a list of date offsets (in weeks) for the forecast horizon
date_offsets = [(i + 1) for i in range(forecast_horizon)]
# Explode these offsets to create a row per forecast week
ss_forecast_df = series_to_forecast.withColumn("forecast_week", F.explode(F.array([F.lit(i) for i in date_offsets])))
# Add `forecast_date` by adding offset weeks to `series_max_date`
ss_forecast_df = ss_forecast_df.withColumn("forecast_date", 
    F.date_add(F.col("training_series_max_date"), F.col("forecast_week") * 7))
    # E.g., week 2 of a series' forecast is 14 days after the max date of the series.(The weeks represent the orders for leading week, not the following week.)
ss_forecast_df = ss_forecast_df.drop("training_series_max_date")

# COMMAND ----------

# MAGIC %md
# MAGIC >add placeholders for `y_hat`, `y_hat_upper`, and `y_hat_lower` and check.

# COMMAND ----------


ss_forecast_df = ss_forecast_df.withColumn("y_hat", F.lit(None).cast("double")) \
    .withColumn("y_hat_upper", F.lit(None).cast("double")) \
    .withColumn("y_hat_lower", F.lit(None).cast("double")) \
    .orderBy(group_col, "forecast_date")

# # inspect the df
# ss_forecast_df.show(110)

# # print the length
# print(f"ss_forecast_df length: {ss_forecast_df.count()}")

# # print number of unique time series
# print(f"number of short training series: {series_to_forecast.count()}")

# Calculate and print the average number of rows (forecast weeks) per forecast.
avg_forecast_weeks_per_series = ss_forecast_df.count() / series_to_forecast.count()
# print(f"Average number of forecast dates per series: {avg_forecast_weeks_per_series:.2f}")

# # By construction, this should be equal to the forecast_horizon.
# print(forecast_horizon)

if avg_forecast_weeks_per_series != forecast_horizon:
    print(f"""
        {Back.RED + Style.BRIGHT}
        *************************************************************************************
        The average number of forecast weeks per series is NOT equal to the forecast horizon.
        *************************************************************************************
        {Style.RESET_ALL}
        """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Short Series Forecasts

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import broadcast

# COMMAND ----------

t1 = time.time()
print("end time: ", t1)
print("Runtime (sec): ", (t1 - t0))
print("Runtime (min): ", (t1 - t0) / 60)

# COMMAND ----------

dbutils.notebook.exit("stop")

# COMMAND ----------

# Step 1:
##############################
# join the forecast df with the training data, be sure to align the forecast weeks with the comp data properly!
##############################
# Add forecast alignment index
ss_forecast_df = ss_forecast_df.withColumn(
    "forecast_alignment_index",
    F.col("forecast_week") + F.col("training_series_length")
)
ss_forecast_df = ss_forecast_df.drop("training_series_length") # to avoid ambiguity error
# print("ss_forecast_df:")
# ss_forecast_df.show(5)
# print()

# select relevant cols from strain_with_comp_stats
sseries_comp_stats = strain_with_comp_stats.select(
        "time_series_id",
        "training_series_length",
        "training_series_max_date",        
        "cmp_series_first_year_index",
        "cmpgrp_avg_normalized_y_clean_int",
        "cmpgrp_med_normalized_y_clean_int",
        "cmpgrp_avg_yoy_growth_rate",
        "cmpgrp_median_yoy_growth_rate",
        "first_year_est_with_comp_avg",
        "first_year_est_with_comp_med"
    )
# Repartition and sort by join keys
ss_forecast_df = ss_forecast_df.repartition(group_col).sortWithinPartitions("forecast_alignment_index")
sseries_comp_stats = sseries_comp_stats.repartition(group_col).sortWithinPartitions("cmp_series_first_year_index")

# broadcast the sseries_comp_stats to save time (it's sufficiently small)
sseries_comp_stats = broadcast(sseries_comp_stats)

ss_forecast_df = ss_forecast_df.join(
    sseries_comp_stats,
    (ss_forecast_df[group_col] == sseries_comp_stats[group_col]) &
    (ss_forecast_df["forecast_alignment_index"] == sseries_comp_stats["cmp_series_first_year_index"]),
    how="left"
)

ss_forecast_df.show(60)

# COMMAND ----------

t2 = time.time()
print("end time: ", t2)
print("Runtime (sec): ", (t2 - t1))
print("Runtime (min): ", (t2 - t1) / 60)

# COMMAND ----------

# Step 2:
# Define the growth rate scalars (1 + growth rate)
ss_forecast_df = ss_forecast_df.withColumn(
    "gr_s_1", F.col("cmpgrp_avg_yoy_growth_rate") + F.lit(1)  # For _aa1, _am1, _ma1, _mm1
).withColumn(
    "gr_s_2", F.col("cmpgrp_median_yoy_growth_rate") + F.lit(1)  # For _aa2, _am2, _ma2, _mm2
)

# Step 3:
# Calculate y_hat_aa1
ss_forecast_df = ss_forecast_df.withColumn(
    "y_hat_aa1",
    F.when(
        F.col("forecast_week") <= (52 - F.col("training_series_length")),
        F.col("first_year_est_with_comp_avg") * F.col("cmpgrp_avg_normalized_y_clean_int")
    ).when(
        (F.col("forecast_week") > (52 - F.col("training_series_length"))) &
        (F.col("forecast_week") <= ((2*52) - F.col("training_series_length"))),
        F.col("first_year_est_with_comp_avg") * F.col("gr_s_1") * F.col("cmpgrp_avg_normalized_y_clean_int")
    ).otherwise(
        F.col("first_year_est_with_comp_avg") * (F.col("gr_s_1") ** 2) * F.col("cmpgrp_avg_normalized_y_clean_int")
    )
)

# Calculate y_hat_aa2
ss_forecast_df = ss_forecast_df.withColumn(
    "y_hat_aa2",
    F.when(
        F.col("forecast_week") <= (52 - F.col("training_series_length")),
        F.col("first_year_est_with_comp_avg") * F.col("cmpgrp_avg_normalized_y_clean_int")
    ).when(
        (F.col("forecast_week") > (52 - F.col("training_series_length"))) &
        (F.col("forecast_week") <= ((2*52) - F.col("training_series_length"))),
        F.col("first_year_est_with_comp_avg") * F.col("gr_s_2") * F.col("cmpgrp_avg_normalized_y_clean_int")
    ).otherwise(
        F.col("first_year_est_with_comp_avg") * (F.col("gr_s_2") ** 2) * F.col("cmpgrp_avg_normalized_y_clean_int")
    )
)


# Calculate y_hat_am1 and y_hat_am2 based on forecast week ranges

# Calculate y_hat_ma1 and y_hat_ma2 based on forecast week ranges

# Calculate y_hat_mm1 and y_hat_mm2 based on forecast week ranges






# Step 4: Drop intermediate columns
ss_forecast_df = ss_forecast_df.drop("gr_s_1", "gr_s_2")

# Inspect the resulting DataFrame
ss_forecast_df.show(100)


# COMMAND ----------

dbutils.notebook.exit("stop")

# COMMAND ----------



# COMMAND ----------

## Command 1
```python
# Step 5: generate future dates and forecast_df

# Calculate the last date of each series
last_date_df = strain_with_comp_stats.groupBy('time_series_id').agg(F.max('ds').alias('last_date'))

# Generate future dates for each series
future_dates_rdd = last_date_df.rdd.flatMap(lambda row: [(row['time_series_id'], row['last_date'] + timedelta(weeks=i)) for i in range(1, 105)])
future_dates_schema = T.StructType([
    T.StructField('time_series_id', T.StringType(), True),
    T.StructField('ds', T.DateType(), True)
])
forecast_df = spark.createDataFrame(future_dates_rdd, future_dates_schema)

# Join the necessary columns to forecast_df
forecast_df = forecast_df.join(
    strain_with_comp_stats.select('time_series_id', 'first_year_est_total_cases', 'cmpgrp_avg_normalized_y_clean_int', 'avg_yoy_growth_rate'),
    on='time_series_id',
    how='left'
)

# Create a window specification to handle the rolling forecast values
window_spec_roll = Window.partitionBy('time_series_id').orderBy('ds')

# Calculate the forecast values for 104 weeks
forecast_df = forecast_df.withColumn(
    'week_number',
    F.row_number().over(window_spec_roll)
)

forecast_df = forecast_df.withColumn(
    'y_hat',
    F.col('first_year_est_total_cases') * F.col('cmpgrp_avg_normalized_y_clean_int') * F.expr('pow(1 + avg_yoy_growth_rate, floor(week_number / 52))')
)
# This needs to be fixed, it appears that the growth rate is applied on the 53rd week of the forecast, not on the 53rd week of the series.

forecast_df = forecast_df.withColumn("fc_fiscal_year", F.lit(fiscal_year)) \
    .withColumn("fc_fiscal_period", F.lit(fiscal_period)) \
    .withColumn("fc_creation_date", F.current_date())

# Select the final columns in the required order
forecast_result_df = forecast_df.select('time_series_id', 'ds', 'y_hat', 'fc_fiscal_year', 'fc_fiscal_period', 'fc_creation_date')

# Show the final forecast DataFrame
forecast_result_df.show(50)


# COMMAND ----------

# MAGIC %md
# MAGIC #Forecast Long Series

# COMMAND ----------

## Command 1
```python
# Join the filtered series lengths to isolate series of the appropriate length
long_series_df = transformed_nonconstant_series_df.join(
    filtered_series_lengths,
    on='time_series_id',
    how='inner'
)

# COMMAND ----------

## Defining the Exponential Smoothing UDF
```python
def stl_w_smoothing_ts_forecast_ff(group_col, value_col, date_col):
    # Define the schema of the DataFrame to be returned by the UDF
    schema = StructType(
        [
            StructField(group_col, StringType(), True),
            StructField(date_col, DateType(), True),
            StructField("y_hat", DoubleType(), True),
        ]
    )

    # The actual UDF function defined with the correct parameters
    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def forecast(pdf):

        pdf = pdf.sort_values(date_col)

        group_value = pdf[group_col].iloc[0]

        stl = STL(pdf[value_col], period=52, seasonal=13, robust=True)
        result = stl.fit()
        seasonal, trend, resid = result.seasonal, result.trend, result.resid

        model = ExponentialSmoothing(
            trend + resid,
            trend="additive",
            seasonal=None,
            initialization_method="estimated",
        )
        model_fit = model.fit(optimized=True)

        forecast = model_fit.forecast(52)
        forecast += seasonal[-52:].values

        future_dates = pd.date_range(
            start=pdf[date_col].max() + pd.DateOffset(weeks=1), periods=52, freq="W"
        ).normalize()

        return pd.DataFrame(
            {group_col: [group_value] * 52, date_col: future_dates, "y_hat": forecast}
        )

    return forecast


# COMMAND ----------

## Creating Specific Exponential Smoothing Forecast UDF
```python
# Create specific exponential smoothing forecast UDF
exp_smooth_forecast_udf = stl_w_smoothing_ts_forecast_ff(group_col, value_col, date_col)

# Applying the UDF
long_series_forecast_df = long_series_df.groupBy(
    group_col
).apply(exp_smooth_forecast_udf)
long_series_forecast_df.show()


# COMMAND ----------

## Command 1
```python
long_series_forecast_df.show()
print(f"'long_series_forecast_df' length = {long_series_forecast_df.count()}")

long_series_forecast_count = (
    long_series_forecast_df.select(group_col).distinct().count()
)
print(f"'forecast_count' = {long_series_forecast_count}")


# COMMAND ----------

long_series_forecast_df_w_ci = long_series_forecast_df.groupBy("time_series_id").apply(
    calculate_confidence_intervals
)

long_series_forecast_df_w_ci.show()
print(f"'long_series_forecast_df_w_ci' length = {long_series_forecast_df_w_ci.count()}")


# COMMAND ----------

## Define Schema and Transformation Function
```python
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType

# Define the schema of the output DataFrame
schema = StructType(
    [
        StructField("time_series_id", StringType(), True),
        StructField("ds", DateType(), True),
        StructField("y_hat", DoubleType(), True),
        StructField("y_hat_upper", DoubleType(), True),
        StructField("y_hat_lower", DoubleType(), True),
    ]
)

# Define the inverse Box-Cox transformation function
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def inverse_boxcox_transform(pdf):
    time_series_id = pdf["time_series_id"].iloc[0]  # Assuming time_series_id is uniform within each group
    lam_info = lambda_values_dict.get(time_series_id, {"fitted_lambda": None, "is_transformed": False})

    if lam_info["is_transformed"]:
        lam = lam_info["fitted_lambda"]
        pdf["y_hat"] = pdf["y_hat"].apply(lambda x: inv_boxcox(x, lam) if pd.notnull(x) else x)
        pdf["y_hat_upper"] = pdf["y_hat_upper"].apply(lambda x: inv_boxcox(x, lam) if pd.notnull(x) else x)
        pdf["y_hat_lower"] = pdf["y_hat_lower"].apply(lambda x: inv_boxcox(x, lam) if pd.notnull(x) else x)

    return pdf


# COMMAND ----------

## Inverse Transform Application
```python
# Apply the inverse-transform UDF after grouping
lsfc_inverse_transformed_df = long_series_forecast_df_w_ci.groupBy(group_col).apply(
    inverse_boxcox_transform
)

lsfc_inverse_transformed_df.show()
print(f"'lsfc_inv_trans_df' length = {lsfc_inverse_transformed_df.count()}")

# COMMAND ----------

## Removing Outliers
```python
cols_for_outlier_removal = ["y_hat", "y_hat_lower", "y_hat_upper"]

# Remove the outliers (replacing with NULLs).
lsfc_out_rem = spark_remove_outliers(
    lsfc_inverse_transformed_df,
    cols_for_outlier_removal,
    outlier_threshold=fc_ci_outlier_threshold,
)
# lsfc_out_rem.show()
print(f"'lsfc_out_rem' length = {lsfc_out_rem.count()}")

lsfc_out_rem = lsfc_out_rem.drop(
    "y_hat",
    "y_hat_upper",
    "y_hat_lower",
    "y_hat_is_outlier",
    "y_hat_lower_is_outlier",
    "y_hat_upper_is_outlier",
)

print()
print()
print("after dropping a few columns:")
lsfc_out_rem.show()


# COMMAND ----------

## Replacing Extreme Values with Nulls
```python
def replace_extremes_with_null(df, columns):
    """
    Replace extreme values in specified columns with NULL.
    Extreme values are defined as less than 1 or equal to infinity.
    """
    for column in columns:
        df = df.withColumn(
            column,
            F.when(
                (F.col(column) < 1) | (F.col(column) == float("inf")), None
            ).otherwise(F.col(column)),
        )
    return df

columns_to_alter = ["y_hat", "y_hat_upper", "y_hat_lower"]
lsfc_extreme_val_rem_df = replace_extremes_with_null(
    lsfc_inverse_transformed_df, columns_to_alter
)

# lsfc_extreme_val_rem_df.show(150)


# COMMAND ----------

## Command 3
```python
# Remove extreme values
lsfc_extreme_val_rem_df = lsfc_df.filter(
    (F.abs(F.col("y_hat") - F.col("y")) < fc_ci_outlier_threshold) &
    (F.abs(F.col("y_hat_upper") - F.col("y")) < fc_ci_outlier_threshold) &
    (F.abs(F.col("y_hat_lower") - F.col("y")) < fc_ci_outlier_threshold)
)

lsfc_extreme_val_rem_df.show()

# COMMAND ----------

## Interpolating Columns
```python
cols_to_interpolate = ["y_hat_out_rem", "y_hat_lower_out_rem", "y_hat_upper_out_rem"]
interpolation_method = "linear"
interpolated_lsfc_df = spark_pandas_interpolate_convert(
    lsfc_out_rem, group_col, cols_to_interpolate, date_col, interpolation_method
)

interpolated_lsfc_df.show()
print(f"'interpolated_lsfc_df' length = {interpolated_lsfc_df.count()}")


# COMMAND ----------

# Combine short and long series' forecast dfs

# COMMAND ----------

## Aggregate Forecasts by date for visualization and testing
```python
aggregated_forecasts_df = interpolated_lsfc_df.groupBy("ds").agg(
    F.sum("y_hat_out_rem").alias("agg_y_hat"),
    F.sum("y_hat_lower_out_rem").alias("agg_y_hat_lower"),
    F.sum("y_hat_upper_out_rem").alias("agg_y_hat_upper"),
)
aggregated_forecasts_df.show()


# COMMAND ----------

# visualize total-vol forecast with actuals

# COMMAND ----------

# shift by a week?

# COMMAND ----------

# test total-vol forecast

# COMMAND ----------

# Save unaggregated forecasts to a table

# automatically enable/disable code to save the forecasts as a table using 'test_set_length'?? (ex: if test_set_length = 0 save a forecast to the relevent location, else skip)

# In the testing notebook, write something such that if the test_set_length is not 0, the test_set_length, the start/end dates, error stats, sample size, etc. are saved to a testing table.