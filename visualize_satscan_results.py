from constants import RESULTS_PATH, COUNTIES_SHP_PATH, DATA_PATH, INTERESTING_TOWNS
import geopandas as gpd
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import patches
from matplotlib import pyplot as plt
import xarray as xr
from datetime import datetime
import pandas as pd
from sklearn.decomposition import PCA

ds = xr.open_dataset(f"{RESULTS_PATH}/pm_2016_utah_urban.nc")
# aggregate over space
ds = ds.mean(dim='x').mean(dim='y')

df2 = ds.to_dataframe().reset_index()

df2['date'] = pd.to_datetime(df2['date'], unit='ns').dt.strftime('%Y-%m-%d')
df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')
df2.drop(columns=['spatial_ref'], inplace=True)

# Step 1: Read raw file spatio temporal
df = pd.read_csv(f"{RESULTS_PATH}/satscan_utah/results.col.txt",
                 delim_whitespace=True,
                 header=0,
                 dtype=str,  # Read as string first so we can clean
                 encoding='utf-8')

# Step 2: Replace commas in floats and convert numeric columns
for col in df.columns:
    if col not in ['LOC_ID', 'START_DATE', 'END_DATE', 'GINI_CLUST']:
        df[col] = df[col].str.replace(",", ".").astype(float)

# Step 3: Parse dates
df["START_DATE"] = pd.to_datetime(df["START_DATE"], format="%Y/%m/%d")
df["END_DATE"] = pd.to_datetime(df["END_DATE"], format="%Y/%m/%d")
df = df[df.P_VALUE == df.P_VALUE.min()]

df['DURATION'] = (df['END_DATE'] - df['START_DATE']).dt.days
df['DURATION'] = df['DURATION'].astype(int)

# Transform LATITUDE and LONGITUDE to be on km projection
import geopandas as gpd
from shapely.geometry import Point

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['LONGITUDE'], df['LATITUDE']), crs="EPSG:4326")
gdf = gdf.to_crs(epsg=32612)  # UTM zone 12N
utm_coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))

towns = gpd.read_file(f"{RESULTS_PATH}/CitiesTownsLocationsUrban.shp")
# keep only towns with population > 500
towns = towns[towns['POPULATION'] > 500]
towns = towns[towns['NAME'].isin(INTERESTING_TOWNS)]

towns = towns.to_crs(epsg=32612)  # UTM zone 12N
towns_coords = np.column_stack((towns.geometry.x, towns.geometry.y))

#join the two coords
all_coords = np.concatenate((utm_coords, towns_coords), axis=0)

# Compute PCA on LATITUDE and LONGITUDE
pca = PCA(n_components=1)
pca.fit(all_coords)
all_proj = pca.transform(all_coords)[:, 0]
min_proj = all_proj.min()
max_proj = all_proj.max()
proj_range = max_proj - min_proj

# Cluster and town proj
s_proj = pca.transform(utm_coords)[:, 0]
town_proj = pca.transform(towns_coords)[:, 0]
towns['PROJ_CENTER'] = town_proj
pop_min = towns['POPULATION'].min()
pop_max = towns['POPULATION'].max()
towns['NORM_POP'] = (towns['POPULATION'] - pop_min) / (pop_max - pop_min)

# Project radius as interval in meters
radius_in_m = df['RADIUS'] * 1000
s_min = s_proj - radius_in_m
s_max = s_proj + radius_in_m
s_range = s_max - s_min

# add the new columns to the cluster dataframe
df['PROJ_CENTER'] = s_proj
df['PROJ_MIN'] = s_min * 0.9
df['PROJ_MAX'] = s_max * 0.9

# sort by PROJ_CENTER
df = df.sort_values(by='PROJ_CENTER', ascending=True)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime

import locale
locale.setlocale(locale.LC_ALL, 'es_ES.utf8')

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(20, 15))


# Set limits
start_min = df["START_DATE"].min()
start_date = mdates.date2num(start_min)
end_max = df["END_DATE"].max()
ax.set_xlim(start_min, end_max)


ax.set_ylim(min_proj - 0.05 * proj_range, max_proj + 0.05 * proj_range)




# Add vertical grid every week
# Still draw weekly vertical grid lines
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
ax.grid(which='minor', axis='x', linestyle='--', color='gray', alpha=0.5)

# Set ticks at the beginning of each month and use month name
ax.xaxis.set_major_locator(mdates.MonthLocator())

from matplotlib.ticker import FuncFormatter




ax.grid(which='major', axis='x', linestyle='--', color='gray', alpha=0.9)

# Format ticks to show only month numbers
fig.autofmt_xdate()

# Add seasonal background colors
def paint_season(ax, year):
    seasons = {
        "Winter":  [(1, 1), (3, 20)],
        "Spring":  [(3, 21), (6, 20)],
        "Summer":  [(6, 21), (9, 22)],
        "Fall":    [(9, 23), (12, 20)],
        "Winter2": [(12, 21), (12, 31)]
    }

    season_colors = {
        "Winter": '#CCE5FF',
        "Winter2": '#CCE5FF',
        "Spring": '#E6FFCC',
        "Summer": '#FFFACD',
        "Fall":   '#FFDAB9'
    }

    for season, ((start_m, start_d), (end_m, end_d)) in seasons.items():
        start_date = datetime(year, start_m, start_d)
        end_date = datetime(year, end_m, end_d)
        ax.axvspan(start_date, end_date, color=season_colors[season], alpha=0.8)

# Paint all relevant years
years = sorted(set(df['START_DATE'].dt.year.tolist() + df['END_DATE'].dt.year.tolist()))
for y in years:
    paint_season(ax, y)


ax.set_xlabel("Month")
ax.set_ylabel("Location")




# draw clusters
for _, row in df.iterrows():
    start = mdates.date2num(row["START_DATE"])
    end = mdates.date2num(row["END_DATE"])
    duration = end - start

    y = row["PROJ_CENTER"]
    y_min_v = row["PROJ_MIN"]
    y_max_v = row["PROJ_MAX"]

    height = y_max_v - y_min_v

    rect = patches.Rectangle(
        (start, y), width=duration, height=height, color='steelblue', alpha=0.8
    )
    ax.add_patch(rect)


# draw towns in y axis
for _, row in towns.iterrows():
    y = row["PROJ_CENTER"]
    radius = row["NORM_POP"]

    # Draw a horizontal dotted line for each town
    ax.hlines(y, start_min, end_max, color='gray', linestyle='dotted')

# Replace numeric Y-axis with town names
ax.set_yticks(towns["PROJ_CENTER"])
ax.set_yticklabels(towns["NAME"])


# First turn off all grid lines
ax.grid(False)

# Then re-enable vertical grid lines only
ax.grid(axis='x', linestyle='--', color='gray', alpha=0.5)


# Create a secondary Y axis
ax2 = ax.twinx()
# Plot pollution on that second axis
ax2.plot(df2["date"], df2["value"], color="tomato", linewidth=1.5, label="Mean Pollution")
# Label it
ax2.set_ylabel("Pollution", color="tomato")
ax2.tick_params(axis='y', labelcolor="tomato")
# Optional: Add to legend
lines_labels = [*ax.get_legend_handles_labels(), *ax2.get_legend_handles_labels()]
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper right")


def english_month_formatter(x, pos=None):
    dt = mdates.num2date(x)
    m = dt.strftime('%b')
    # print(m)
    mapper = { 'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
              'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec' }
    m = m.lower()
    return mapper.get(m, m)

ax.xaxis.set_major_formatter(FuncFormatter(english_month_formatter))

# Final touches

# ax.set_yticks(df["LLR"].tolist())
#print data of the first cluster
print(df.iloc[0])


ax.set_title("SaTScan Spatio-Temporal Clusters in Projected Space")
fig.tight_layout()
plt.show()
fig.savefig("clusters_pca.png")