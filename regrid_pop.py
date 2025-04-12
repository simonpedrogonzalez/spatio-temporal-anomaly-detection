import xarray as xr
import rioxarray
import numpy as np
import matplotlib.pyplot as plt

from constants import RESULTS_PATH, POP_DATA_PATH, MY_CRS

# Load the pollution dataset
ds = xr.open_dataset(f"{RESULTS_PATH}/pm_2016_utah_urban.nc")
pm_data = ds['value'].isel(date=0)  # Take one time slice just to match the grid
print("Shape")
print(pm_data.shape)
print("firstpoint coordinates")
print(pm_data.x[0].values, pm_data.y[0].values)

pm_data.plot()
plt.title("PM2.5 Data")
plt.savefig("pm.png")
plt.clf()

# Load population raster as xarray
pop_data = rioxarray.open_rasterio(POP_DATA_PATH, masked=True).squeeze()

# Assign CRS if missing
if not pop_data.rio.crs:
    pop_data = pop_data.rio.write_crs(MY_CRS)

# Reproject and align to pollution dataset grid
pop_regridded = pop_data.rio.reproject_match(pm_data)
# pop_regridded.plot()
# plt.show()

# Mask to the extent of the pollution dataset
pop_masked = pop_regridded.where(~np.isnan(pm_data))


# Save the masked + regridded population raster
output_path = f"{RESULTS_PATH}/pop_urban.tif"
pop_masked.rio.set_crs(MY_CRS, inplace=True)

pop_masked.plot()
plt.savefig("pop.png")

print("Shape after masking")
print(pop_masked.shape)
print("firstpoint coordinates after masking")
print(pop_masked.x[0].values, pop_masked.y[0].values)


pop_masked.rio.to_raster(output_path)

print("Saved to:", output_path)
