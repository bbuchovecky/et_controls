"""
Variance Inflation Factor (VIF) for MLR predictors of evapotranspiration,
computed per-gridpoint across the PPE.
 
For each predictor j, VIF_j = 1 / (1 - R^2_j), where R^2_j is the
coefficient of determination from regressing predictor j on all remaining
predictors.
 
The predictor differences (relative to the default member) used here are
identical to those in mlr_ppe_gridpoint_masked_1950-2014.nc, so VIF
results are directly interpretable against those regression coefficients.

Δ operator := difference between perturbed and default members
 
Inputs mirror multiple_regression_across_ens.py exactly:
    ΔT    = annually averaged air temperature        (TSA, K)
    ΔLAI  = growing-season averaged LAI              (TLAI, m2 m-2)
    ΔP    = annually cumulative precipitation        (RAIN_FROM_ATM, mm yr-1)
    ΔSWin = annually averaged incoming shortwave     (FSDS, W m-2)
 
Output:
    vif_ppe_gridpoint_masked_1950-2014.nc
    Variables:
        vif      (predictor, lat, lon)  -- VIF per predictor per gridpoint
        r2_aux   (predictor, lat, lon)  -- R^2 of each auxiliary regression
"""

from pathlib import Path
import numpy as np
import xarray as xr
import xclimate as xclim

OUTDIR = Path("/glade/derecho/scratch/bbuchovecky/derived/mlr")
OUTNAME = "vif_ppe_gridpoint_masked_noFSDS_1950-2014.nc"

# Variables to load
VARIABLES = [
    "EFLX_LH_TOT_month_1",
    "TSA_month_1",
    "TLAI_month_1",
    "RAIN_FROM_ATM_month_1",
    # "FSDS_month_1",
]

PREDICTORS = [
    "TLAI_month_1",
    "TSA_month_1",
    "RAIN_FROM_ATM_month_1",
    # "FSDS_month_1",
]
PREDICTOR_NAMES = [p.replace("_month_1", "") for p in PREDICTORS]

# Time period for analysis
START_TIME = "1950-01"
END_TIME = "2014-12"

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

# Minimum average growing-season LAI, 0.15 is from Forzieri et al. (2020)
GRSN_LAI_MIN_THRESHOLD = 0.15

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

# # Compute annual mean FSDS
# print("Computing annual mean FSDS...")
# data_annual['FSDS_month_1'] = (data['FSDS_month_1'] * weights).groupby('time.year').sum()

# Compute annual mean TSA
print("Computing annual mean TSA...")
data_annual['TSA_month_1'] = (data['TSA_month_1'] * weights).groupby('time.year').sum()

# Compute annual cumulative RAIN_FROM_ATM (mm/s -> mm/year)
print("Computing annual cumulative RAIN_FROM_ATM...")
days_per_month = data['RAIN_FROM_ATM_month_1'].time.dt.days_in_month
data_annual['RAIN_FROM_ATM_month_1'] = (data['RAIN_FROM_ATM_month_1'] * days_per_month * 86400).groupby('time.year').sum()

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

# Calculate difference relative to default member
# dims: (lat, lon, member)
# shape: (n_lat, n_lon, n_member-1)
print("\nCalculating difference relative to default member...")
predictor_diff = {
    p: data_annual[p].sel(member=slice(1, None)) - data_annual[p].sel(member=0) for p in PREDICTORS
}

lat = predictor_diff[PREDICTORS[0]].lat
lon = predictor_diff[PREDICTORS[0]].lon
n_lat = len(lat)
n_lon = len(lon)
n_pred = len(PREDICTORS)

# Stack (lat, lon) -> gridpoint for vectorized regression
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

n_gridpoints = predictor_stacked.shape[0]

# Output arrays: (n_gridpoints, n_pred)
vif_out  = np.full((n_gridpoints, n_pred), np.nan)
r2aux_out = np.full((n_gridpoints, n_pred), np.nan)

print("Running per-gridpoint auxiliary OLS regressions for VIF...")
for g in range(n_gridpoints):
    X_g = predictor_stacked[g]  # (n_members, n_pred)
 
    for j in range(n_pred):
        # Target for auxiliary regression: predictor j
        y_j = X_g[:, j]  # (n_members,)
 
        # Remaining predictors as regressors
        X_aux = np.delete(X_g, j, axis=1)  # (n_members, n_pred-1)
 
        # NaN mask: any member with NaN in y_j or any column of X_aux
        valid = ~(np.isnan(y_j) | np.any(np.isnan(X_aux), axis=1))
        n_valid = valid.sum()
 
        # Need at least n_pred samples (n_pred-1 regressors + intercept)
        if n_valid < n_pred:
            continue
 
        y_v = y_j[valid]
        X_v = X_aux[valid]
 
        # Augment with intercept column
        X_design = np.column_stack([np.ones(n_valid), X_v])
 
        beta, _, _, _ = np.linalg.lstsq(X_design, y_v, rcond=None)
 
        y_pred    = X_design @ beta
        residuals = y_v - y_pred
        ss_res    = np.dot(residuals, residuals)
        ss_tot    = np.dot(y_v - y_v.mean(), y_v - y_v.mean())
 
        # Guard against degenerate cases (zero variance in target)
        if ss_tot == 0.0:
            continue
 
        r2_j = 1.0 - ss_res / ss_tot
        r2aux_out[g, j] = r2_j
 
        # Clamp R^2 to [0, 1) to avoid division by zero or negative VIF
        r2_j_clamped = np.clip(r2_j, 0.0, 1.0 - 1e-10)
        vif_out[g, j] = 1.0 / (1.0 - r2_j_clamped)

# Reshape to (n_pred, n_lat, n_lon)
vif_out   = vif_out.reshape(n_lat, n_lon, n_pred).transpose(2, 0, 1)
r2aux_out = r2aux_out.reshape(n_lat, n_lon, n_pred).transpose(2, 0, 1)

print("Building output Dataset...")
vif_da = xr.DataArray(
    data=vif_out,
    dims=["predictor", "lat", "lon"],
    coords={
        "predictor": PREDICTOR_NAMES,
        "lat":       lat,
        "lon":       lon,
    },
    name="vif",
    attrs={
        "description":      "Variance Inflation Factor per predictor per gridpoint",
        "formula":          "VIF_j = 1 / (1 - R2_j), where R2_j is from regressing predictor j on all remaining predictors",
        "predictors":       ", ".join(PREDICTORS),
        "time_period":      f"{START_TIME} to {END_TIME}",
        "differencing":     "Predictors are differences relative to the default PPE member (member=0)",
        "masks":            "growing-season LAI > 0.15 and PCT_GLC < 80",
    },
)
 
r2aux_da = xr.DataArray(
    data=r2aux_out,
    dims=["predictor", "lat", "lon"],
    coords={
        "predictor": PREDICTOR_NAMES,
        "lat":       lat,
        "lon":       lon,
    },
    name="r2_aux",
    attrs={
        "description":  "R^2 of auxiliary regression used to compute VIF",
        "formula":      "R2_j from regressing predictor j on all remaining predictors",
        "predictors":   ", ".join(PREDICTORS),
        "time_period":  f"{START_TIME} to {END_TIME}",
        "differencing": "Predictors are differences relative to the default PPE member (member=0)",
        "masks":        "growing-season LAI > 0.15 and PCT_GLC < 80",
    },
)
 
ds_out = xr.Dataset({"vif": vif_da, "r2_aux": r2aux_da})
 
out_file = OUTDIR / OUTNAME
ds_out.to_netcdf(out_file)
print(f"VIF and auxiliary R^2 saved to {out_file}")
 
print("\nClosing Dask cluster...")
if CLIENT_CLUSTER is not None:
    xclim.close_dask_cluster(CLIENT_CLUSTER)
 
print("Done!")
