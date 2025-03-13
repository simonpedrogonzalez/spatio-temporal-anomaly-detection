import xarray as xr
import pandas as pd
from constants import RESULTS_PATH
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

def slice_ds(ds, lats, longs, dates):
    y_min, y_max = min(lats), max(lats)
    x_min, x_max = min(longs), max(longs)
    # transform dates from 'yyyy-mm-dd' to datetime64 numbers
    dates = pd.to_datetime(dates, format='%Y-%m-%d')
    dates = (dates[0].value, dates[1].value)
    date_min, date_max = min(dates), max(dates)

    def is_ascending(arr):
        return arr[0] < arr[1]

    dims = ['x', 'y', 'date']
    values = [(x_min, x_max), (y_min, y_max), (date_min, date_max)]

    for i, dim in enumerate(dims):
        asc = is_ascending(ds[dim])
        v1, v2 = values[i]
        if not asc:
            v1, v2 = v2, v1
        ds = ds.sel({dim: slice(v1, v2)})
    return ds
        

# def discretize(v):
#     kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
#     v = v.values.reshape(-1, 1)
#     v = kb.fit_transform(v)
#     return v

def add_location(ds):
    y_flat = ds.y.values.ravel()
    x_flat = ds.x.values.ravel()
    location_keys = pd.Series([f"{y}_{x}" for y, x in zip(y_flat, x_flat)])
    location_ids = location_keys.astype('category').cat.codes + 1  # Start from 1 instead of 0
    ds['location'] = location_ids
    return ds

ds = xr.open_dataset(f"{RESULTS_PATH}/pm_2016.nc")

# Define the box
lats = (40.529, 41.401)
longs = (-112.661, -111.726)
dates = ('2016-01-01', '2016-01-31')

ds = slice_ds(ds, lats, longs, dates)

# write the ds to a nc file in case we want to analyze with pyscan / python
ds.to_netcdf(f"{RESULTS_PATH}/pm_2016_small.nc")

df = ds.to_dataframe().reset_index()

# date from datetime64 ns to 'yyyy-mm-dd'
df['date'] = pd.to_datetime(df['date'], unit='ns').dt.strftime('%Y-%m-%d')
df.drop(columns=['spatial_ref'], inplace=True)

# round the values to int, dunno if this is necessary
df['value'] = df['value'].round(0).astype(int)

# add location column
df = add_location(df)

df.to_csv(f"{RESULTS_PATH}/satscan.cas", sep=" ", index=False, header=True)
print('done')
