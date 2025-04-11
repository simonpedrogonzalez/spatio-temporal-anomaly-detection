import xarray as xr
import pandas as pd
from constants import RESULTS_PATH
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

def slice_ds(ds, lats=None, longs=None, dates=None):
    if lats is None:
        lats = (min(ds.y.values), max(ds.y.values))
    if longs is None:
        longs = (min(ds.x.values), max(ds.x.values))
    if dates is None:
        dates = (min(ds.date.values), max(ds.date.values))
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
        

def aggregate_on_dates(ds):
    # just remove the date dimension by calculating the mean over it
    ds = ds.mean(dim='date')
    return ds


def add_location(ds):
    y_flat = ds.y.values.ravel()
    x_flat = ds.x.values.ravel()
    location_keys = pd.Series([f"{y}_{x}" for y, x in zip(y_flat, x_flat)])
    location_ids = location_keys.astype('category').cat.codes + 1  # Start from 1 instead of 0
    ds['location'] = location_ids
    return ds

ds = xr.open_dataset(f"{RESULTS_PATH}/pm_2016.nc")

# Take for example month of January 2016 and October 2016 in only Salt Lake City area

# lats = (40.529, 41.401)
# longs = (-112.661, -111.726)
january = ('2016-01-01', '2016-01-31')
october = ('2016-10-01', '2016-10-31')

ds_january = slice_ds(ds, None, None, january)
ds_october = slice_ds(ds, None, None, october)

# now aggregate monthly
ds_january = aggregate_on_dates(ds_january)
ds_october = aggregate_on_dates(ds_october)

# write to nc files
ds_january.to_netcdf(f"{RESULTS_PATH}/pm_utah_2016_january.nc")
ds_october.to_netcdf(f"{RESULTS_PATH}/pm_utah_2016_october.nc")


# # create a shp file with squares for each pixel
# import geopandas as gpd
# import shapely

# def create_geodataframe(ds):
#     y_flat = ds.y.values.ravel()
#     x_flat = ds.x.values.ravel()
#     xy_diff = (x_flat[1] - x_flat[0], y_flat[1] - y_flat[0])
#     xy_diff_2 = (xy_diff[0]/2, xy_diff[1]/2)
#     squares = []
#     for y in y_flat:
#         for x in x_flat:
#             ll = (x-xy_diff_2[0], y-xy_diff_2[1])
#             ul = (x-xy_diff_2[0], y+xy_diff_2[1])
#             ur = (x+xy_diff_2[0], y+xy_diff_2[1])
#             lr = (x+xy_diff_2[0], y-xy_diff_2[1])
#             square = shapely.geometry.Polygon([ll, ul, ur, lr])
#             squares.append({'geometry': square})
#     gdf = gpd.GeoDataFrame(squares)
#     return gdf

# gdf = create_geodataframe(ds_january)
# gdf.to_file(f"{RESULTS_PATH}/pm_slc.shp")

