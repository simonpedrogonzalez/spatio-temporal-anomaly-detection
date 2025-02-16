import pyreadr
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.affinity import rotate

resolution_meters = 1000  # 1 km resolution


# Load the RDS file
result = pyreadr.read_r("data/aq_data/PM_2016/20161231.rds")  # Reads the RDS file
dat = result[None].T  # Extracts the DataFrame

dat.columns = ["value"]  # Rename for consistency

# Load the PM25Grid CSV file
grid = pd.read_csv("data/aq_data/PM25Grid.csv")[['lat', 'long']]
assert grid.shape[0] == dat.shape[0], "Data and grid dimensions do not match!"

df = pd.concat([grid, dat], axis=1)

lat, lon, value = df["lat"], df["long"], df["value"]

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["long"], df["lat"]))
gdf.set_crs(epsg=3567, inplace=True)
fig, ax = plt.subplots(figsize=(50, 50))
gdf.plot(column="value", cmap="viridis", legend=True, markersize=1, ax=ax)
plt.savefig("test.png")
print('done')

# Define raster bounds
min_lon, max_lon = lon.min(), lon.max()
min_lat, max_lat = lat.min(), lat.max()

# Compute the number of rows and columns
cols = int((max_lon - min_lon) / lon_diff) + 1
rows = int((max_lat - min_lat) / lat_diff) + 1

# Create an empty raster grid
raster_array = np.full((rows, cols), np.nan)  # Initialize with NaNs

# Assign PM2.5 values to their respective centroid locations in the raster
for i in range(len(lon)):
    col = int((lon[i] - min_lon) / lon_diff)
    row = int((max_lat - lat[i]) / lat_diff)  # Invert row index for raster format
    raster_array[row, col] = pm25[i]

# Define transform using centroid-based grid
transform = from_origin(min_lon, max_lat, lon_diff, lat_diff)

# Save as GeoTIFF
with rasterio.open(
    "PM25_20161231.tif",
    "w",
    driver="GTiff",
    height=rows,
    width=cols,
    count=1,
    dtype=raster_array.dtype,
    crs="EPSG:4326",  # Adjust CRS if necessary
    transform=transform,
) as dst:
    dst.write(raster_array, 1)

print("Raster file 'PM25_20161231.tif' created successfully!")
