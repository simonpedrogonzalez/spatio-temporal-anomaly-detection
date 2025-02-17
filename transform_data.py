from constants import RESULTS_PATH, PM2_GRID_PATH, PM2_2016_FILE_PATHS, TMP_PATH
from rasterize_grid import rasterize
import xarray as xr
import datetime
import tqdm
import numpy as np
import os

def extract_date(file_path):
    file_name = file_path.split("/")[-1]
    year = file_name[:4]
    month = file_name[4:6]
    day = file_name[6:8]
    return f"{year}-{month}-{day}"

def to_float32(ds):
    # Here I'm also converting date from datetime64 to float32
    # Which may not be a good idea for some reason
    for var in ds.data_vars:
        if ds[var].dtype == np.float64:
            ds[var] = ds[var].astype(np.float32)
    for coor in ds.coords:
        if ds[coor].dtype == np.float64:
            ds[coor] = ds[coor].astype(np.float32)
    return ds

def get_encodings(ds):
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {'zlib': True, 'complevel': 5}
    return encoding 
    

def date_as_dim(ds):
    unique_dates = np.unique(ds['date'].values[~np.isnan(ds['date'].values)])
    assert len(unique_dates) == 1, "Dates are not unique"
    single_date = unique_dates[0]
    ds = ds.drop_vars('date')
    ds = ds.expand_dims(date=[single_date])
    return ds


def generate_net_cdf_files():
    for file in tqdm.tqdm(PM2_2016_FILE_PATHS):
        ds = rasterize(f"{file}", PM2_GRID_PATH)
        ds = date_as_dim(ds)
        ds = to_float32(ds)
        encodings = get_encodings(ds)
        
        file_name = file.split("/")[-1]
        ds.to_netcdf(f"{TMP_PATH}/{file_name}.nc", encoding=encodings)
    print("Done generating per_date netcdf files")
        

def concatenate_net_cdf_files():
    # warning dont put .nc files you dont want to concatenate in the TMP_PATH
    print("Concatenating netcdf files")
    ds = xr.open_mfdataset(
        f'{TMP_PATH}/*.nc',
        combine = 'nested',
        concat_dim = ['date'])
    ds.to_netcdf(f'{RESULTS_PATH}/pm_2016.nc') # Export netcdf file
    print("Done concatenating netcdf files")

def test_file():
    foile = f"{RESULTS_PATH}/pm_2016.nc"
    ds = xr.open_dataset(foile)
    print(ds)


# generate_net_cdf_files()
# concatenate_net_cdf_files()
test_file()