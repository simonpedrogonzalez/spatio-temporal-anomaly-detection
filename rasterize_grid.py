import pyreadr
import pandas as pd
import numpy as np
import geopandas as gpd
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from rasterio.transform import Affine
from constants import RESULTS_PATH, PM2_GRID_PATH, COUNTIES_SHP_PATH

RESOLUTION_IN_METERS = 1000 # 1 km
MAP_GEOMETRY = None
GRID_DF = None
GRID_FILE_PATH = None

def get_map_geometry():
    global MAP_GEOMETRY
    if MAP_GEOMETRY is None:
        MAP_GEOMETRY = gpd.read_file(COUNTIES_SHP_PATH).to_crs(epsg=32612).geometry.union_all()
    return MAP_GEOMETRY

def extract_date(file_path):
    file_name = file_path.split("/")[-1]
    year = file_name[:4]
    month = file_name[4:6]
    day = file_name[6:8]
    return f"{year}-{month}-{day}"

def read_rds_data(path):
    result = pyreadr.read_r(path)
    dat = result[None].T
    dat.columns = ["value"]
    date = extract_date(path)
    dat["date"] = pd.to_datetime(date)
    return dat

def get_grid(path):
    global GRID_DF
    if GRID_DF is None or GRID_FILE_PATH != path:
        GRID_DF = pd.read_csv(path)[['lat', 'long']]
        GRID_FILE_PATH = path
    return GRID_DF

def export_test_geojson(gdf):
    gdf.to_crs(epsg=4326, inplace=True)
    gdf.to_file(f"{RESULTS_PATH}/test.geojson", driver="GeoJSON")


def create_gdf(dat, grid):
    assert grid.shape[0] == dat.shape[0], "Data and grid dimensions do not match!"

    df = pd.concat([grid, dat], axis=1)
    lat, lon = df["lat"], df["long"]
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(lon, lat))
    gdf.set_crs(epsg=4326, inplace=True)
    # transform to UTM
    gdf_utm = gdf.to_crs(epsg=32612)
    return gdf_utm

def rasterize(data_file_path, grid_file_path):
    dat = read_rds_data(data_file_path)
    grid = get_grid(grid_file_path)
    gdf_utm = create_gdf(dat, grid)

    geo_grid = make_geocube(
        vector_data=gdf_utm,
        measurements=["value", "date"],
        datetime_measurements=["date"],
        resolution=(-RESOLUTION_IN_METERS, RESOLUTION_IN_METERS),
        fill=np.nan,
        rasterize_function=rasterize_points_griddata,
        interpolate_na_method="nearest", # try to preserve the original values
    )
    geo_grid.date.rio.write_nodata(np.nan, inplace=True)
    mask_geometry = get_map_geometry()
    geo_grid = geo_grid.rio.clip([mask_geometry], geo_grid.rio.crs, drop=True)

    geo_grid = geo_grid.rio.reproject("EPSG:4326")
    # # set all nodata in date column to Nan
    # ggm['date'] = ggm['date'].where(ggm['date']!= 0)
    return geo_grid

# TODO: [low priority]
# Untested experiment for more precision when rasterizing
# rotate both the raster and the geometry by about
# 14.42 degrees to align better with the point cloud arrangement
# clip the raster and rotate back. 

def get_rotated_map_geometry():
    # works OK
    gdf = gpd.read_file(COUNTIES_SHP_PATH).to_crs(epsg=32612)
    centroid = gdf.union_all().centroid
    angle = 14.42
    geom = gdf.rotate(angle, origin=centroid).geometry
    gdf.geometry = geom
    return gdf.geometry.union_all()

def rotate_gdf(gdf):
    # works OK
    angle = 14.42
    gdf2 = gdf.copy()
    gdf2.set_crs(epsg=32612, inplace=True)
    gdf2.drop(columns=['lat', 'long'], inplace=True)
    centroid = gdf2.union_all().centroid
    geom = gdf2.rotate(angle, origin=centroid).geometry
    gdf2.geometry = geom
    return gdf2, angle, centroid

def get_raster_transform(centroid, affine_src):
    # does not work correctly: it seems to flip the raster
    # in unexpected ways
    angle = 14.42
    return Affine.rotation(angle, centroid) * affine_src

def rasterize_with_rotation(data_file_path, grid_file_path):
    # does not work correctly
    dat = read_rds_data(data_file_path)
    grid = get_grid(grid_file_path)
    gdf_utm = create_gdf(dat, grid)

    # Rotate to follow the point cloud arrangement
    gdf_utm, angle, centroid = rotate_gdf(gdf_utm)

    geo_grid = make_geocube(
        vector_data=gdf_utm,
        measurements=["value", "date"],
        datetime_measurements=["date"],
        resolution=(-RESOLUTION_IN_METERS, RESOLUTION_IN_METERS),
        fill=np.nan,
        rasterize_function=rasterize_points_griddata,
        interpolate_na_method="nearest", # try to preserve the original values
    )

    mask_geometry = get_rotated_map_geometry()

    ggm = geo_grid.rio.clip([mask_geometry], geo_grid.rio.crs, drop=True)
    
    # centroid to EPSG:4326
    # gdf_utm = gdf_utm.to_crs(epsg=4326)
    # centroid = gdf_utm.union_all().centroid
    # height, width = ggm.rio.height, ggm.rio.width
    # centroid = (width // 2, height // 2)

    tr = get_raster_transform(centroid, ggm.rio.transform())

    ggm = ggm.rio.reproject(
        "EPSG:4326",
        transform=tr
    )

    # ggm.value.plot()
    # plt.savefig('map4.png')
    # ggm.rio.to_raster("test_4326_rot.tif")
    # read file with rasterio and apply transform

