# Introduction 
The forecasting work done by the Data Team can be thought of in four main parts: Project Prophet v1, v2, & v3 and the replacement of the "base/calculated layer" of the current forecast.

The goal of Project Prophet is to replace/automate nearly the entire process of creating our demand forecast. Versions 1 & 2 were a POC and an MVP and predate our adoption of Databricks. The goal of v3 is to utilize Databricks to implement a much more sophisticated forecast that would align more with industry standards. The initial phase involves comparing many Prophet, SARIMAX, and ML models to see which approaches best fit our data. This work has been paused multiple times.

Replacing the base forecast means replacing the forecast calculation (in the Forecast Team's current process) prior to any adjustments/assumptions being added in. This consists of rewriting a modified version of Project Prophet v2 with some additional bells and whistles. For example, it makes sense to go ahead and integrate the pro forma volume estimates (and the requisite comp analysis) to save them time during the monthly iteration.

# Using the Code
## Imports and Functions
Most of the notebooks in the Data Team's **'demand_forecasting'** repo run the **'General_Project_Imports'** and the **'General_Project_Functions'** notebooks (located in the Project Prophet v3 folder) remotely using the %run command, so libraries and function definitions not immediately visable can be found there.
## Sales Preprocessing Notebook
Currently, while we are migrating data to Databricks, the data is being sourced from milosdata via a Databricks mount (/mnt/MilosAI_Storage_Container/) to a container in an Azure storage account (**'container01' in 'milostestblob'** - these were named by IT Security). The **'Sales_Preprocessing'** notebook pulls the data, processes it, and creates multiple views and tables. The views are temporary and for accessing the prepared data from other notebooks and are session-specific. As for the tables, one is for the constant series that can be forecasted with a simple naive forecast and the other is for the non-constant series that will be forecasted with smoothing. 
## Project Prophet Notebooks
[TBD]
## Base Forecast Replacement Notebook
**Note: Much of the code was lost in a recent event that shall not be described here. This notebook is currently being recreated/rewritten (we recovered some of the code using a text file from the cache and ChatGPT), so more is known about this code and process than can be demonstrated at the moment.**

**To test forecasts with different hyperparameter values, ensure the code to save the forecasts is disabled** then adjust the hyperparameter values, set the **'test_set_length'**, and run the notebook. The test wil be done at the bottom of the notebook. Adjust hyperparameters and repeat as needed. 

**To create the base forecast, ensure the code to save the forecasts is enabled and the 'test_set_length' is set to zero**, input the fiscal period, fiscal year, file save location, etc. in the cell labeled for such at the top, then run the notebook. Many visualizations as well as the forecast will be created. The forecast will then be saved to the designated path. 

**More Information:** 
The base forecast layer in the Forecast Team's current process is calculated using a 20-week average of YoY growth rates for each series (we're working at the item and DC level). The replacement will utilize a Box-Cox transformation to correct non-constant variance where necessary before using exponential smoothing to create the forecasts for series of at least a certain length. Series that aren't long enough for smoothing will have a custom forecast created using the estimated distribution of their first year's volume (based on comps) and their total volume to date. So, a helpful way to think about the organization of the forecasting process is by series length/age. Some series have no historical data (just a pro forma volume estimate for the first year), some series have between 0 and 52 weeks of historical data, and some series have at least 52 weeks of data.  
