import xarray as xr
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from sklearn.linear_model import LinearRegression
import tqdm

from constants import RESULTS_PATH, POP_DATA_PATH, MY_CRS

# Load the pollution dataset
ds = xr.open_dataset(f"{RESULTS_PATH}/pm_2016_utah_urban.nc")


pop = rioxarray.open_rasterio(f"{RESULTS_PATH}/pop_urban.tif", masked=True).squeeze()

pop_flat = pop.values.flatten()
valid_pop_mask = ~np.isnan(pop_flat)

residuals = []

# Loop over each date
for date in tqdm.tqdm(ds['date'].values):
    pm = ds.sel(date=date)['value']
    pm_flat = pm.values.flatten()

    valid_mask = valid_pop_mask & ~np.isnan(pm_flat)

    if valid_mask.sum() == 0:
        residuals.append(np.full_like(pm, np.nan))
        continue

    X = pop_flat[valid_mask].reshape(-1, 1)
    y = pm_flat[valid_mask]

    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Residuals
    residual = np.full_like(pm_flat, np.nan)
    residual[valid_mask] = y - y_pred
    residual_2d = residual.reshape(pm.shape)
    residuals.append(residual_2d)


# Standarize the residuals
residuals = np.array(residuals)
mean_residual = np.nanmean(residuals, axis=0)
std_residual = np.nanstd(residuals, axis=0)
residuals = (residuals - mean_residual) / std_residual


# Stack and save
residuals_stack = xr.DataArray(
    data=np.stack(residuals),
    dims=['date', 'y', 'x'],
    coords={'date': ds['date'], 'y': ds['y'], 'x': ds['x']},
    name='pm_residual'
)

# get o slice
residuals_stack.isel(date=0).plot.imshow(cmap='viridis', add_colorbar=True)
plt.title("PM2.5 Residuals")
plt.savefig("pm_residual.png")
plt.clf()

# Save to NetCDF
residuals_stack.to_netcdf(f"{RESULTS_PATH}/pm_2016_adjusted_for_pop.nc")

ds_mean = ds['value'].mean(dim='date').values.flatten()[valid_pop_mask]
pop_flat = pop_flat[valid_pop_mask]

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



lr = LinearRegression()
lr.fit(pop_flat.reshape(-1, 1), ds_mean)
pred = lr.predict(pop_flat.reshape(-1, 1))

mse1 = mean_squared_error(ds_mean, pred)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(pop_flat.reshape(-1, 1))
poly.fit(pop_flat.reshape(-1, 1), ds_mean)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, ds_mean)
pred_poly = poly_reg.predict(X_poly)

mse2 = mean_squared_error(ds_mean, pred_poly)
print(f"Linear Regression MSE: {mse1}")
print(f"Polynomial Regression MSE: {mse2}")

# Plot a mean density plot for pollution
sns.kdeplot(ds_mean, color='blue', label='PM2.5 Density')
plt.xlabel("PM2.5 Values")
plt.ylabel("Density")
plt.title("PM2.5 Density Plot")
plt.savefig("pm_density.png")
plt.clf()




plt.figure(figsize=(10, 6))
sns.scatterplot(x=pop_flat, y=ds_mean, alpha=0.1)
plt.plot(pop_flat, pred, color='red', linewidth=2)
plt.plot(pop_flat, pred_poly, color='blue', linewidth=2)
plt.xlabel("Population")
plt.ylabel("PM2.5 Values")
plt.title("PM2.5 vs Population")
plt.savefig("pop_vs_pm.png")
plt.clf()



