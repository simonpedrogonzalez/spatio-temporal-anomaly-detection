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


# FIT A GAUSSIAN MIXTURE AND REMOVE OUTLIERS
from sklearn.mixture import GaussianMixture
import numpy as np
# Flatten data
pollution_flat = ds.value.values.flatten()
valid_mask = ~np.isnan(pollution_flat)
pollution_valid = pollution_flat[valid_mask].reshape(-1, 1)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(pollution_valid)
log_probs = gmm.score_samples(pollution_valid)
threshold = np.percentile(log_probs, 0.99)  # Adjust this threshold as needed
outlier_mask = log_probs < threshold
without_outliers = pollution_valid[log_probs > threshold]
print(f"N outliers: {np.sum(outlier_mask)}")


# PLOT

import numpy as np
import matplotlib.pyplot as plt

# pollution_flat: 1D pollution data (with NaNs possibly)
# outlier_mask: boolean mask of same shape (True = outlier)
# gmm: fitted GaussianMixture

# Filter valid data (for histogram and outlier indexing)
valid_mask = ~np.isnan(pollution_flat)
X = pollution_flat[valid_mask]
X = X[~outlier_mask[valid_mask]]  # remove outliers from X
# outliers = X[outlier_mask[valid_mask]]  # just the outliers (valid only)

# Range for GMM PDF
x_plot = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
logprob = gmm.score_samples(x_plot)
responsibilities = gmm.predict_proba(x_plot)
pdf = np.exp(logprob)
pdf_individual = responsibilities * pdf[:, np.newaxis]

# Plot
fig, ax = plt.subplots(figsize=(15, 3))
ax.hist(X, bins=50, density=True, alpha=0.4, color='steelblue', label='Pollution Data')
ax.plot(x_plot, pdf, '-k', label='Total GMM')
ax.plot(x_plot, pdf_individual, '--k', alpha=0.6, label='GMM Components')

# Plot red markers for outliers
# ax.scatter(outliers, np.zeros_like(outliers), color='red', s=10, label='Outliers', zorder=5)

# Labels
ax.set_xlabel("Pollution")
ax.set_ylabel("Density")
ax.set_title("Pollution Distribution with GMM")
ax.legend()
plt.tight_layout()
plt.savefig("gmm_outliers.png")
plt.clf()


# END OF GMM PLOTTING


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
# filter outliers from ds_mean
log_probs_mean = gmm.score_samples(ds_mean.reshape(-1, 1))
gmm_mean = GaussianMixture(n_components=3, random_state=42)
gmm_mean.fit(ds_mean.reshape(-1, 1))
log_probs_mean = gmm_mean.score_samples(ds_mean.reshape(-1, 1))
threshold = np.percentile(log_probs_mean, 0.99)  # Adjust this threshold as needed


valid_mean_mask = log_probs_mean > threshold
print(f"N outliers in mean: {np.sum(~valid_mean_mask)}")
ds_mean = ds_mean[valid_mean_mask]

pop_flat = pop_flat[valid_pop_mask]



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



