"""
Multiple linear regression to predict evapotranspiration over land,
computed per-gridpoint.

Δ operator := difference between perturbed and default members

Predictand:
    ΔET   = annually averaged ET
Predictors:
    ΔT    = annually averaged air temperature        (TSA, K)
    ΔLAI  = annual growing-season averaged LAI       (TLAI, m2 m-2)
    ΔP    = annually cumulative precipitation        (RAIN_FROM_ATM, mm yr-1)
    ΔSWin = annually averaged incoming shortwave     (FSDS, W m-2)
    

Mirroring (but across the PPE, not time):
----------
Forzieri et al. (2020). Increased control of vegetation on global
terrestrial energy fluxes. Nature Climate Change, 10(4), 356-362.
https://doi.org/10.1038/s41558-020-0717-0
"""

from pathlib import Path
import numpy as np
import xarray as xr
import xclimate as xclim

OUTDIR = Path("/glade/derecho/scratch/bbuchovecky/derived/mlr")
OUTNAME = "mlr_ppe_gridpoint_masked_onlyTLAI_1950-2014.nc"

# Variables to load
VARIABLES = [
    "EFLX_LH_TOT_month_1",
    # "TSA_month_1",
    "TLAI_month_1",
    # "RAIN_FROM_ATM_month_1",
    # "FSDS_month_1",
]

PREDICTORS = [
    "TLAI_month_1",
    # "TSA_month_1",
    # "RAIN_FROM_ATM_month_1",
    # "FSDS_month_1",
]
COEFF_NAMES = ["intercept"] + [p.replace("_month_1", "") for p in PREDICTORS]

NORMALIZE_TARGET_BY_MEAN_TEMP = False

USE_DASK_CLUSTER = False

# Time period for analysis
START_TIME = "1950-01"
END_TIME = "2014-12"

# Minimum average growing-season LAI, 0.15 is from Forzieri et al. (2020)
GRSN_LAI_MIN_THRESHOLD = 0.15


#################################
#################################


CLIENT_CLUSTER = None
if USE_DASK_CLUSTER:
    print("Creating Dask cluster...")
    CLIENT_CLUSTER = xclim.create_dask_cluster(
        account='UWAS0155',
        nworkers=4,
        ncores=1,
        nmem='10GB',
        walltime='02:00:00',
    )

# Load grid
grid = xclim.load_fhist_ppe_grid().compute()

print(f"Loading variables from {START_TIME} to {END_TIME}...")
data = {}

for v in VARIABLES:
    print(f"  Loading {v}...")
    name = "_".join(v.split("_")[:-2])

    data[v] = (
        xclim.load_fhist(v, keep_var_only=True)[name]
        .sel(lat=slice(-60, None), time=slice(START_TIME, END_TIME))
        .reindex_like(grid, method="nearest", tolerance=1e-3)
    )

# Mask out glaciated gridpoints without any plants (PCT_GLAC>80 and LAI=0)
min_lai = data['TLAI_month_1'].min(dim=['member', 'time'])
mask = xr.where((min_lai == 0) & (grid.PCT_GLC >= 80), 0, 1)

# Weight by days in month
time_coord = data[list(data.keys())[0]].time
dpm = time_coord.dt.days_in_month
weights = dpm.groupby('time.year') / dpm.groupby('time.year').sum()

# Get annual values
data_annual = {}

# Compute annual mean ET
print("Computing annual mean ET...")
data_annual['EFLX_LH_TOT_month_1'] = (data['EFLX_LH_TOT_month_1'] * weights).groupby('time.year').sum()

if "FSDS_month_1" in PREDICTORS:
    # Compute annual mean FSDS
    print("Computing annual mean FSDS...")
    data_annual['FSDS_month_1'] = (data['FSDS_month_1'] * weights).groupby('time.year').sum()

if "TSA_month_1" in PREDICTORS:
    # Compute annual mean TSA
    print("Computing annual mean TSA...")
    data_annual['TSA_month_1'] = (data['TSA_month_1'] * weights).groupby('time.year').sum()

if "RAIN_FROM_ATM_month_1" in PREDICTORS:
    # Compute annual cumulative RAIN_FROM_ATM (mm/s -> mm/year)
    print("Computing annual cumulative RAIN_FROM_ATM...")
    days_per_month = data['RAIN_FROM_ATM_month_1'].time.dt.days_in_month
    data_annual['RAIN_FROM_ATM_month_1'] = (data['RAIN_FROM_ATM_month_1'] * days_per_month * 86400).groupby('time.year').sum()

if "TLAI_month_1" in PREDICTORS:
    # Compute annual growing-season mean TLAI
    print("Computing growing-season mean TLAI...")
    lai = data['TLAI_month_1']
    climlai = lai.groupby('time.month').mean()
    climlai_max = climlai.max(dim='month')
    climlai_rng = climlai.max(dim='month') - climlai.min(dim='month')
    threshlai = climlai_max - 0.5 * climlai_rng
    grsn_mask = climlai >= threshlai

    grsn_lai = lai.assign_coords(month=('time', lai.time.dt.month.values))
    grsn_lai_ym = (
        grsn_lai.assign_coords(year=('time', lai.time.dt.year.values))
        .set_index(time=['year', 'month'])
        .unstack('time')
        .transpose('member', 'year', 'month', 'lat', 'lon')
    )
    grsn_lai_masked = grsn_lai_ym.where(grsn_mask).transpose('member', 'year', 'month', 'lat', 'lon')
    data_annual['TLAI_month_1'] = grsn_lai_masked.mean(dim='month', skipna=True).transpose('member', 'year', 'lat', 'lon')

    # Mask out low growing-season LAI
    print("Masking out low growing-season LAI and computing time mean values...")
    grsn_lai_tmean = data_annual['TLAI_month_1'].mean(dim='year').compute()
    grsn_lai_min_mask = grsn_lai_tmean > GRSN_LAI_MIN_THRESHOLD
    for v, da in data_annual.items():
        data_annual[v] = data_annual[v].where(grsn_lai_min_mask).mean(dim='year').compute()

# Extract target
target = data_annual["EFLX_LH_TOT_month_1"]

if NORMALIZE_TARGET_BY_MEAN_TEMP:
    target = target / data_annual["TSA_month_1"]

n_coeffs = len(PREDICTORS) + 1

# Calculate difference relative to default member
# dims: (lat, lon, member)
# shape: (n_lat, n_lon, n_member-1)
print("\nCalculating difference relative to default member...")
target_diff = target.sel(member=slice(1, None)) - target.sel(member=0)
predictor_diff = {
    p: data_annual[p].sel(member=slice(1, None)) - data_annual[p].sel(member=0) for p in PREDICTORS
}

# Retrieve coordinate metadata for output reconstruction
lat   = target_diff.lat
lon   = target_diff.lon

n_lat     = len(lat)
n_lon     = len(lon)
n_pred    = len(PREDICTORS)
n_coeffs  = n_pred + 1  # predictors + intercept

# Stack (lat, lon) -> gridpoint for vectorized regression
# target_diff shape after stack: (n_gridpoints)
print("Stacking spatial dimensions...")
target_stacked = (
    target_diff
    .stack(gridpoint=('lat', 'lon'))
    .transpose('gridpoint', 'member')
    .values  # (n_gridpoints, n_members)
)

predictor_stacked = np.stack(
    [
        predictor_diff[p]
        .stack(gridpoint=('lat', 'lon'))
        .transpose('gridpoint', 'member')
        .values
        for p in PREDICTORS
    ],
    axis=-1,
)  # shape: (n_gridpoints, n_members, n_pred)

n_gridpoints = target_stacked.shape[0]

# Output array: (n_gridpoints, n_coeffs)
coeffs_out = np.full((n_gridpoints, n_coeffs), np.nan)
std_coeffs_out = np.full((n_gridpoints, n_coeffs), np.nan)

METRIC_NAMES = ["R2", "RMSE", "MAE"]
n_metrics = len(METRIC_NAMES)
metrics_out = np.full((n_gridpoints, n_metrics), np.nan)

print("Running per-gridpoint OLS regression...")
# y: (n_gridpoints, n_members)
y = target_stacked

# X_raw: (n_gridpoints, n_members, n_pred)
X_raw = predictor_stacked

for g in range(n_gridpoints):
    y_g = y[g]        # (n_members,)
    X_g = X_raw[g]    # (n_members, n_pred)

    # NaN mask: drop any member where target or any predictor is NaN
    valid = ~(np.isnan(y_g) | np.any(np.isnan(X_g), axis=1))
    n_valid = valid.sum()

    # Need at least n_coeffs samples to fit
    if n_valid < n_coeffs:
        continue

    y_v = y_g[valid]          # (n_valid,)
    X_v = X_raw[g][valid]     # (n_valid, n_pred)

    # Augment X with a leading column of ones for the intercept
    # X_design: (n_valid, n_coeffs)  where col 0 = 1 (intercept)
    X_design = np.column_stack([np.ones(n_valid), X_v])

    # OLS via least squares: minimizes ||y - X_design @ beta||^2
    # beta: (n_coeffs,)  = [intercept, beta_1, ..., beta_n_pred]
    beta, _, _, _ = np.linalg.lstsq(X_design, y_v, rcond=None)
    coeffs_out[g, :] = beta

    # Compute the standardized coefficients
    X_std = X_v.std(axis=0)   # (n_pred,)  std of each predictor over valid years
    y_std = y_v.std()         # scalar     std of target over valid years
    # beta[0] is the intercept — standardization is undefined, store NaN
    std_coeffs = np.concatenate([[np.nan], beta[1:] * X_std / y_std])  # (n_coeffs,)
    std_coeffs_out[g, :] = std_coeffs

    # Compute OLS statistics
    y_pred = X_design @ beta
    residuals = y_v - y_pred
    ss_res = np.dot(residuals, residuals)
    ss_tot = np.dot(y_v - y_v.mean(), y_v - y_v.mean())
    r2   = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / n_valid)
    mae  = np.mean(np.abs(residuals))
    metrics_out[g, :] = [r2, rmse, mae]

# Reshape output arrays back to (n_coeffs, n_lat, n_lon)
coeffs_out = coeffs_out.reshape(n_lat, n_lon, n_coeffs).transpose(2, 0, 1)
std_coeffs_out = std_coeffs_out.reshape(n_lat, n_lon, n_coeffs).transpose(2, 0, 1)
metrics_out = metrics_out.reshape(n_lat, n_lon, n_metrics).transpose(2, 0, 1)

# Build output xarray.Dataset with one variable per coefficient,
# dims: (coefficient, member, lat, lon)
print("Building output Dataset...")
coeffs_da = xr.DataArray(
    data=coeffs_out,
    dims=["coefficient", "lat", "lon"],
    coords={
        "coefficient": COEFF_NAMES,
        "lat":         lat,
        "lon":         lon,
    },
    name="beta",
    attrs={
        "description": "OLS regression coefficients: ΔET ~ intercept + Σ beta_i * ΔX_i",
        "target":      "EFLX_LH_TOT (W m-2), difference relative to the default member",
        "predictors":  ", ".join(PREDICTORS),
        "time_period": f"{START_TIME} to {END_TIME}",
        "units_note":  "intercept in W m-2; predictor coefficients in W m-2 per predictor unit",
        "masks":       "LAI > 0 and PCT_GLC < 80",
    },
)

std_coeffs_da = xr.DataArray(
    data=std_coeffs_out,
    dims=["coefficient", "lat", "lon"],
    coords={
        "coefficient": COEFF_NAMES,
        "lat":             lat,
        "lon":             lon,
    },
    name="std_beta",
    attrs={
        "description": "standardized OLS regression coefficients: beta_i * (s_X_i / s_ET)",
        "target":      "EFLX_LH_TOT (W m-2), difference relative to the default member",
        "predictors":  ", ".join(PREDICTORS),
        "time_period": f"{START_TIME} to {END_TIME}",
        "units_note":  "intercept is NaN; predictor standardized coefficients are unitless",
        "masks":       "LAI > 0 and PCT_GLC < 80",
    },
)

metrics_da = xr.DataArray(
    data=metrics_out,
    dims=["metric", "lat", "lon"],
    coords={
        "metric": METRIC_NAMES,
        "lat":    lat,
        "lon":    lon,
    },
    name="regression_metrics",
    attrs={
        "description": "OLS goodness-of-fit metrics per gridpoint",
        "R2":   "coefficient of determination, dimensionless",
        "RMSE": "root mean squared error, W m-2",
        "MAE":  "mean absolute error, W m-2",
    },
)

ds_out = xr.Dataset({"beta": coeffs_da, "std_beta": std_coeffs_da, "regression_metrics": metrics_da})

out_file = OUTDIR / OUTNAME
ds_out.to_netcdf(out_file)
print(f"Coefficients and metrics saved to {out_file}")

# Close Dask cluster
if CLIENT_CLUSTER is not None:
    print("\nClosing Dask cluster...")
    xclim.close_dask_cluster(CLIENT_CLUSTER)

print("Done!")
