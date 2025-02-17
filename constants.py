import os

RESULTS_PATH = "results"
PM2_GRID_PATH = "data/aq_data/PM25Grid.csv"
PM2_2016_DATA_PATH = "data/aq_data/PM_2016"
PM2_2016_FILE_PATHS = sorted([
    f"{PM2_2016_DATA_PATH}/{f}" for f in os.listdir(PM2_2016_DATA_PATH) if os.path.isfile(os.path.join(PM2_2016_DATA_PATH, f))
])
COUNTIES_SHP_PATH = "data/aq_data/shapefiles/Counties.shp"
