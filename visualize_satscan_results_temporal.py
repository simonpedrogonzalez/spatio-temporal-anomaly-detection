from constants import RESULTS_PATH, COUNTIES_SHP_PATH
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

ds = xr.open_dataset(f"{RESULTS_PATH}/pm_2016_utah_urban.nc")
# aggregate over space
ds = ds.mean(dim='x').mean(dim='y')

df2 = ds.to_dataframe().reset_index()

df2['date'] = pd.to_datetime(df2['date'], unit='ns').dt.strftime('%Y-%m-%d')
df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')
df2.drop(columns=['spatial_ref'], inplace=True)


# Step 1: Read raw file
df = pd.read_csv(f"{RESULTS_PATH}/satscan_utah_temporal/results.col.txt",
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

# sort by LLR
df = df.sort_values(by='LLR', ascending=False)


# sns.histplot(df['DURATION'], bins=30)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime

import locale
locale.setlocale(locale.LC_ALL, 'es_ES.utf8')

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 7))


# Set limits
start_min = df["START_DATE"].min()
end_max = df["END_DATE"].max()
ax.set_xlim(start_min, end_max)
ax.set_ylim(df["LLR"].min() - 5, df["LLR"].max() + df["LLR"].max() * 0.2)



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

# Plot rectangles for each cluster
# for i, row in df.iterrows():
#     start = mdates.date2num(row["START_DATE"])
#     end = mdates.date2num(row["END_DATE"])
#     duration = end - start

#     rect = patches.Rectangle(
#         (start, i), width=duration, height=0.8, color='steelblue'
#     )
#     ax.add_patch(rect)

ax.set_xlabel("Month")
ax.set_ylabel("LLR")
ax.set_yscale("log")

for _, row in df.iterrows():
    start = mdates.date2num(row["START_DATE"])
    end = mdates.date2num(row["END_DATE"])
    duration = end - start

    y = row["LLR"]
    log_height_factor = 1.2
    height = y * (log_height_factor - 1)

    rect = patches.Rectangle(
        (start, y), width=duration, height=height, color='steelblue', alpha=0.8
    )
    ax.add_patch(rect)


# ax2 = ax.twinx()
# Label Y-axis

# ax.set_yticklabels([])  # Or optionally: df["CLUSTER"] directly if you want just numbers




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

ax.set_title("SaTScan Temporal Clusters with Weekly Grid and Seasons")
fig.tight_layout()
plt.show()
fig.savefig("temporal_clusters_seasonal.png")