from constants import RESULTS_PATH, PM2_GRID_PATH, PM2_2016_FILE_PATHS
from rasterize_grid import rasterize
import xarray as xr
import datetime

def extract_date(file_path):
    file_name = file_path.split("/")[-1]
    year = file_name[:4]
    month = file_name[4:6]
    day = file_name[6:8]
    return f"{year}-{month}-{day}"


for file in PM2_2016_FILE_PATHS:
    print(f"Processing {file}")
    
    ggm = rasterize(f"{PM2_2016_DATA_PATH}/{file}", PM2_GRID_PATH)
    ggm.rio.to_raster(f"{RESULTS_PATH}/{file}.tif")
    print(f"Saved {RESULTS_PATH}/{file}.tif")
    print("Done")

