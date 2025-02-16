import pyreadr
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import AffineTransformer
import geopandas as gpd
import matplotlib.pyplot as plt
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from functools import partial
from shapely.affinity import affine_transform
from rasterio.transform import Affine
from xarray import Dataset

# 1 km resolution
resolution_meters = 1000

def get_dat(path=None):
    path = "data/aq_data/PM_2016/20161231.rds"
    result = pyreadr.read_r(path)
    dat = result[None].T
    dat.columns = ["value"]
    return dat

def get_grid():
    # Load the PM25Grid CSV file
    grid = pd.read_csv("data/aq_data/PM25Grid.csv")[['lat', 'long']]
    return grid

def create_gdf(dat, grid):
    assert grid.shape[0] == dat.shape[0], "Data and grid dimensions do not match!"

    df = pd.concat([grid, dat], axis=1)
    lat, lon = df["lat"], df["long"]
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lon, lat))
    gdf.set_crs(epsg=4326, inplace=True)
    # transform to UTM
    gdf_utm = gdf.to_crs(epsg=32612)
    return gdf_utm

def get_map_geometry():
    return gpd.read_file("data/aq_data/shapefiles/Counties.shp").to_crs(epsg=32612).geometry.union_all()

def get_rotated_map_geometry():
    gdf = gpd.read_file("data/aq_data/shapefiles/Counties.shp").to_crs(epsg=32612)
    centroid = gdf.union_all().centroid
    angle = 14.42
    geom = gdf.rotate(angle, origin=centroid).geometry
    gdf.geometry = geom
    return gdf.geometry.union_all()

def rotate_gdf(gdf):
    angle = 14.42
    gdf2 = gdf.copy()
    gdf2.set_crs(epsg=32612, inplace=True)
    gdf2.drop(columns=['lat', 'long'], inplace=True)
    centroid = gdf2.union_all().centroid
    geom = gdf2.rotate(angle, origin=centroid).geometry
    gdf2.geometry = geom
    return gdf2, angle, centroid

# def get_raster_transform(centroid, affine_src):
#     angle = 14.42
#     # shear_transform = Affine.rotation(shear_angle)  # Y-shear applied
#     # shear_transform *= Affine.translation(centroid.x, centroid.y)
#     # affine_src = Affine.from_gdal(*affine)
#     # affine_dst = affine_src * affine_src.rotation(angle, (centroid.x, centroid.y))
#     affine_dst = Affine.rotation(angle, (centroid.x, centroid.y))
#     return affine_dst

import rasterio
from rasterio.transform import Affine
import numpy as np

def cos_sin_deg(angle):
    """Compute cos and sin for a given angle in degrees"""
    theta = np.radians(angle)
    return np.cos(theta), np.sin(theta)

def get_raster_transform(centroid, affine_src):
    angle = 14.42
    centroid = (0,0)
    return Affine.rotation(angle, centroid) * Affine.translation(centroid[0], centroid[1])

# from rasterio.transform import Affine


dat = get_dat()
grid = get_grid()
gdf_utm = create_gdf(dat, grid)

# Rotate to follow the point cloud arrangement
# gdf_utm, angle, centroid = rotate_gdf(gdf_utm)

geo_grid = make_geocube(
    vector_data=gdf_utm,
    measurements=["value"],
    resolution=(-resolution_meters, resolution_meters),
    fill=np.nan,
    rasterize_function=rasterize_points_griddata,
    interpolate_na_method="nearest", # try to preserve the original values
)

# mask_geometry = get_rotated_map_geometry()
mask_geometry = get_map_geometry()
# export mask_geometry to a shapefile
# mask_geometry.to_file("mask_geometry.shp")

ggm = geo_grid.rio.clip([mask_geometry], geo_grid.rio.crs, drop=True)
# centroid to EPSG:4326
# gdf_utm = gdf_utm.to_crs(epsg=4326)
# centroid = gdf_utm.union_all().centroid
# where is centroid in terms of raster coordinates?
# height, width = ggm.rio.height, ggm.rio.width
# centroid = (width // 2, height // 2)

# tr = get_raster_transform(centroid, ggm.rio.transform())
# print(centroid)

ggm = ggm.rio.reproject(
    "EPSG:4326",
    # ggm.rio.crs,
    # transform=tr)
)

# ggm.value.plot()
# plt.savefig('map4.png')
ggm.rio.to_raster("test_4326_rot.tif")
# read file with rasterio and apply transform

