import xarray as xr
import rioxarray
import numpy as np

from constants import RESULTS_PATH, POP_DATA_PATH, MY_CRS

# Load the pollution dataset
ds = xr.open_dataset(f"{RESULTS_PATH}/pm_2016_utah_urban.nc")
pm_data = ds['value'].isel(date=0)  # Take one time slice just to match the grid

# Load population raster as xarray
pop_data = rioxarray.open_rasterio(POP_DATA_PATH, masked=True).squeeze()

# Assign CRS if missing
if not pop_data.rio.crs:
    pop_data = pop_data.rio.write_crs(MY_CRS)

# Reproject and align to pollution dataset grid
pop_regridded = pop_data.rio.reproject_match(pm_data)

# Mask to the extent of the pollution dataset
pop_masked = pop_regridded.where(~np.isnan(pm_data))

# Save the masked + regridded population raster
output_path = f"{RESULTS_PATH}/pop_urban.tif"
pop_masked.rio.to_raster(output_path)

print("Saved to:", output_path)
