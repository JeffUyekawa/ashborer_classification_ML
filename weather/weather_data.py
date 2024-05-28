#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Point, Hourly, Stations

'''
This file reads in a folder of new recordings, and uses meteostat to
gather weather data and join it to file names in a csv

To gather weather for a new batch, do the following:
Set recording_path to the folder with new batch of recordings.
Set end date to include date of last record.
Change path in full_df.to_csv to desired location for weather csv.

'''

recording_path = r"C:\Users\jeffu\Documents\Recordings\05_24_2024"

stations = Stations()
stations = stations.nearby(35.1983,-111.6513)
station = stations.fetch(1)
start = datetime(2024, 4, 16)
end = datetime(2024,5,24)

data = Hourly(station, start, end)
data = data.fetch()

data = data.loc[:,['temp','prcp','wspd']]
data['date'] = pd.to_datetime(data.index).date
data['hour'] = pd.to_datetime(data.index).hour
# %%
file_names= []
datetimes = []
for file_name in os.listdir(recording_path):
    if file_name.endswith('.wav') :
        datetime_object = datetime.strptime(file_name[:-4], '%Y-%m-%d_%H_%M_%S') 

        file_names.append(file_name)
        datetimes.append(datetime_object)
dates = [d.date() for d in datetimes]
hours = [d.hour for d in datetimes]
wav_data = {'File': file_names, 'date':dates, 'hour':hours}
wav_df = pd.DataFrame(wav_data)

full_df = pd.merge(wav_df,data,how = 'inner', on = ['date','hour'])  


full_df.to_csv(r"C:\Users\jeffu\Documents\Recordings\Weather\weather_05_24_2024.csv", index = False)
# %%
wind_df = full_df[full_df.wspd >= 37.0]
prcp_df = full_df[full_df.prcp!=0]
# %%
