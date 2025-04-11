import os

RESULTS_PATH = "results"
TMP_PATH = "tmp"
PM2_GRID_PATH = "data/aq_data/PM25Grid.csv"
PM2_2016_DATA_PATH = "data/aq_data/PM_2016"
PM2_2016_FILE_PATHS = sorted([
    f"{PM2_2016_DATA_PATH}/{f}" for f in os.listdir(PM2_2016_DATA_PATH) if os.path.isfile(os.path.join(PM2_2016_DATA_PATH, f))
])
COUNTIES_SHP_PATH = "data/aq_data/shapefiles/Counties.shp"
DATA_PATH = "data"
POP_DATA_PATH = "data/50_US_states_1km_2016.tif"

INTERESTING_TOWNS = [
    "Logan",
    # "Wellsville",
    "Hyrum",
    # "Tremonton",
    "Brigham City",
    "Willard",
    "Ogden",
    "Layton",
    # "Syracuse",
    "Eden",
    "Morgan",
    "Bountiful",
    "Salt Lake City",
    "North Salt Lake",
    # "South Salt Lake",
    "Millcreek",
    "Murray",
    "Midvale",
    # "Sandy",
    "South Jordan",
    "Herriman",
    # "Park City",
    # "Heber City",
    # "Tooele",
    # "Grantsville",
    # "Stansbury Park",
    "Eagle Mountain",
    "Lehi",
    "Orem",
    "Provo",
    "Spanish Fork",
    # "Salem",
    "Payson",
    "Santaquin"
]

MY_CRS = "EPSG:4326"