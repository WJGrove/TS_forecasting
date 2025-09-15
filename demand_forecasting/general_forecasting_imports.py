# Databricks notebook source
# %pip install colorama
# I installed this^^ module on the 'data Science Cluster'

from colorama import *
import psutil
import io
import os
import time
import datetime
import pytz
from datetime import timedelta
from datetime import datetime
import ast
import json
import holidays
from holidays.countries.united_states import UnitedStates
from dateutil.easter import easter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import BooleanType, StringType, IntegerType, DoubleType, DateType, StructType, StructField, DecimalType, TimestampType
from pyspark.sql import DataFrame
import warnings
import traceback
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from pmdarima.arima import auto_arima
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.stats.diagnostic import het_breuschpagan

from mpl_toolkits.mplot3d import Axes3D
# may be redundant 
from pyspark.sql.functions import pandas_udf, PandasUDFType