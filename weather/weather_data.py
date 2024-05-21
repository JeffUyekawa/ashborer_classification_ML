#%%
import numpy as np
import pandas as pd
import os

weather = pd.read_csv(r"C:\Users\jeffu\Downloads\72375003103.csv")
recording_path = r"C:\Users\jeffu\Documents\Recordings"

# %%
weather['DATE'] = pd.to_datetime(weather['DATE'])

# Define the start and end dates
start_date = pd.to_datetime('2024-05-16')
end_date = pd.to_datetime('2024-05-20')

# Filter the dataframe to keep only the entries between the start and end dates
filtered_weather = weather[(weather['DATE'] >= start_date) & (weather['DATE'] <= end_date)]
# %%
