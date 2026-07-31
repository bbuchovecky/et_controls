"""
Compute ET trends in observational products.
"""

import gc
import argparse
from pathlib import Path
from distributed import wait

import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import t as t_dist
import regionmask as regmask
from ILAMB import ilamblib
import xclimate as xclim
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANALYSIS_CHUNKS = {
    "time": 120,
    "lat": 48,
    "lon": 96,
}

ROOT = Path("/glade/campaign/univ/uwas0155/obs")
OBS_PRODUCTS = {
    "et": [
        "FLUXCOM",
        "DOLCE",
        "CLASS",
        "WECANN",
        "GLEAMv3.3a",
        "MODIS",
        "MOD16A2",
    ],
    "lai": [
        "AVHRR",
        "AVH15C1",
        "MODIS",
        "GIMMS_LAI4g",
    ]
}

DSC_DIM = "year"
REGIONS = regmask.defined_regions.giorgi

DEFAULT_OUTPUT_DIR = {
    "et": Path("/glade/work/bbuchovecky/et_unc/obs/et/trends"),
    "lai": Path("/glade/work/bbuchovecky/et_unc/obs/lai/trends"),
}


# ---------------------------------------------------------------------------
# Ordinary least squares regression
# ---------------------------------------------------------------------------

def ols_over_time(
        y: xr.DataArray,
        dim: str = "year",
) -> xr.Dataset:
    """
    Vectorized OLS regression of a DataArray along a specified dimension.

    Parameters
    ----------
    y  : xr.DataArray with at least one dimension `dim`
    dim : dimension to regress over (the "time" axis)

    Returns
    -------
    xr.Dataset with variables:
        slope     : regression slope (units of y per unit of dim coordinate)
        intercept : regression intercept
        r2        : coefficient of determination
        p_value   : two-tailed p-value for slope (H0: slope = 0), t-distribution
        std_err   : standard error of the slope estimate
    """
    coord = y[dim].values.astype(float)
    n = len(coord)
    
    x = coord
    ss_x = ((x - x.mean()) ** 2).sum()

    # Broadcast x along all non-time dimensions via xr.DataArray
    x_da = xr.DataArray(x, coords={dim: y[dim]}, dims=[dim])
    x_mean = x_da.mean(dim=dim)

    y_mean = y.mean(dim=dim)
    slope = ((y - y_mean) * (x_da - x_mean)).sum(dim=dim) / ss_x
    intercept = y_mean - slope * x.mean()

    y_hat = slope * x_da + intercept
    ss_res = ((y - y_hat) ** 2).sum(dim=dim)   # residual sum of squares
    ss_tot = ((y - y_mean) ** 2).sum(dim=dim)  # total sum of squares

    # Mask out ss_tot == 0
    ss_tot_safe = ss_tot.where(ss_tot > 0, other=1.0)
    degenerate = ss_tot <= 0

    # Coefficient of determination
    r2 = 1.0 - ss_res / ss_tot_safe
    
    # Sample Pearson correlation coefficient
    r = ((y - y_mean) * (x_da - x_mean)).sum(dim=dim) / np.sqrt(((x_da - x_mean) ** 2).sum(dim=dim) * ss_tot_safe)

    # Mask out where ss_tot == 0
    r2 = r2.where(~degenerate)
    r = r.where(~degenerate)

    # Standard error of slope: s² = SS_res / (n-2); SE_slope = sqrt(s² / SS_x)
    s2 = ss_res / (n - 2)
    std_err = np.sqrt(s2 / ss_x)

    # t-statistic and two-tailed p-value (t-distribution, df = n-2)
    t_stat = slope / std_err
    p_value = xr.apply_ufunc(
        lambda t: 2.0 * t_dist.sf(np.abs(t), df=n - 2),
        t_stat,
        dask="parallelized",
        output_dtypes=[float],
    )

    return xr.Dataset(
        data_vars={
            "slope":        slope,
            "intercept":    intercept,
            "r":            r,
            "r2":           r2,
            "p_value":      p_value,
            "slope_stderr": std_err,
            "n":            xr.full_like(slope, n),
        },
        attrs={
            "description": "grid-cell OLS regression",
        }
    )


def compute_regional_ols(
    da: xr.DataArray,
    dsc_dim: str,
    weights: xr.DataArray,
    regions: regmask.Regions = REGIONS,
) -> xr.Dataset:
    """
    ...
    Parameters
    ----------
    weights : xr.DataArray, dims (lat, lon)
        Grid-cell land area weights. Normalization
        is applied within each regional mask.
    ...
    """
    ols_dim = {"month": "decimal_year", "year": "year"}[dsc_dim]
    mask = regions.mask(da)

    results = []
    for region in regions:
        region_mask = mask == region.number
        da_masked   = da.where(region_mask)

        w       = weights.where(region_mask)
        da_mean = (da_masked * w).sum(["lat", "lon"]) / w.sum(["lat", "lon"])

        dsc = deseasonalize(da_mean, dsc_dim)
        ds  = ols_over_time(dsc, ols_dim)
        results.append(ds)

    ds_out = xr.concat(results, dim="region")
    ds_out = ds_out.assign_coords(
        region = ("region", np.arange(len(regions.regions))),
        abbrev = ("region", list(regions.abbrevs)),
        name   = ("region", list(regions.names)),
    )
    ds_out.attrs["description"] = f"regional OLS regression - {regions.name}"
    return ds_out


# ---------------------------------------------------------------------------
# Aridity
# ---------------------------------------------------------------------------

def compute_aridity_index(
    precip: xr.DataArray,
    pet: xr.DataArray,
    clip: bool = False,
) -> xr.DataArray:
    """Compute the annual mean aridity index."""
    if clip:
        pet = pet.clip(min=0)
    ai = precip / pet
    ai = ai.rename("AI")
    ai.attrs = {
        "long_name": "aridity index (P/PET)",
        "description": "computed from annual mean P and PET{' (PET floor at zero)' if clip else ''}",
    }
    return ai


def compute_inverse_aridity_index(
    precip: xr.DataArray,
    pet: xr.DataArray,
    clip: bool = False,
) -> xr.DataArray:
    """Compute the annual mean inverse aridity index."""
    if clip:
        pet = pet.clip(min=0)
    inv_ai = pet / precip
    inv_ai = inv_ai.rename("AI")
    inv_ai.attrs = {
        "long_name": "inverse aridity index (PET/P)",
        "description": f"computed from annual mean P and PET{' (PET floor at zero)' if clip else ''}",
    }
    return inv_ai


# ---------------------------------------------------------------------------
# Deseasonalize
# ---------------------------------------------------------------------------

def compute_annual_mean(da):
    days_in_month = da.time.dt.days_in_month
    weights = days_in_month.groupby('time.year') / days_in_month.groupby('time.year').sum()
    return (da * weights).groupby('time.year').sum()


def deseasonalize(
        da: xr.DataArray,
        dim: str,
    ) -> xr.DataArray:
    """Remove the seasonal cycle or the long-term mean from a time series.

    Parameters
    ----------
    da : xr.DataArray
        Must have a "time" dimension with a datetime-like coordinate
        (e.g., cftime or np.datetime64).
    dim : {"month", "year"}
        "month" : Subtract the climatological monthly mean from `da`,
            removing the seasonal cycle. The "time" dimension is
            preserved, but a new "decimal_year" coordinate is attached
            and set as the index used for selection and alignment
            (continuous time in fractional years, e.g., 2015.50 ≈
            July 2015). The original "time" coordinate (cftime/
            datetime64 values) is retained as a non-indexed coordinate
            for inspection.
        "year" : Subtract the long-term mean from an annual-mean series
            (computed via `compute_annual_mean`), removing the
            interannual mean. Output is indexed by integer "year".

    Returns
    -------
    xr.DataArray
        Anomalies relative to the monthly climatology ("month") or the
        long-term annual mean ("year"). Same units as `da`.
    """

    if "time" not in da.dims:
        raise ValueError('"time" must be a dimension of `da`')

    if dim == "month":
        clim = da.groupby('time.month')
        dsc = clim - clim.mean()
        dsc = dsc.assign_coords(
            decimal_year=dsc.time.dt.year + (dsc.time.dt.month - 1) / 12
        )
        dsc = dsc.swap_dims(time='decimal_year')
        return dsc
    
    elif dim == "year":
        annual = compute_annual_mean(da)
        dsc = annual - annual.mean(dim='year')
        return dsc
    
    else:
        raise ValueError(f"Unknown dim '{dim}'.  Choose 'month' or 'year'.")


def to_yyyymm(time) -> str:
    time_raw = time.values
    if isinstance(time_raw, np.datetime64):
        return pd.Timestamp(time_raw).strftime("%Y%m")
    elif isinstance(time_raw, np.ndarray):
        return time.item().strftime("%Y%m")
    raise TypeError(f"Unsupported type {type(time)!r} for time")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ilamb_obs(variable: str, dsname: str, time_start: str | None, time_stop: str | None):
    """Load observations downloaded from ILAMB and compute cell area."""

    # Load the land mask from the Natural Earth dataset
    land = regmask.defined_regions.natural_earth_v5_1_2.land_50

    ds = xr.open_dataset(ROOT / variable / f"{variable}_{dsname}.nc")

    if dsname == "WECANN":
        ds = ds.sortby("lat", ascending=True)  # flip the lat dimension for WECANN

    obs = ds[variable].sel(time=slice(time_start, time_stop))
    print(f"{dsname:11}: {obs.dims} {obs.shape} {to_yyyymm(obs.time[0])}-{to_yyyymm(obs.time[-1])}", end=" ")

    if "lat_bounds" in ds and "lon_bounds" in ds:
        print("using bounds")
        method = "bounds"
        lat_bounds = ds["lat_bounds"].values
        lon_bounds = ds["lon_bounds"].values
    else:
        print("could not find bounds, using coordinates instead")
        method = "coords"
        lat_bounds = None
        lon_bounds = None

    # Compute the cell areas using the ILAMB function
    area = ilamblib.CellAreas(
        lat=ds["lat"].values,
        lon=ds["lon"].values,
        lat_bnds=lat_bounds,
        lon_bnds=lon_bounds,
    )
    area = xr.DataArray(area, dims=["lat", "lon"], coords={"lat": ds["lat"], "lon": ds["lon"]})
    area.attrs["units"] = "m2"
    area.attrs["long_name"] = "grid cell area"
    area.attrs["method"] = method
    
    # Create a land mask for the dataset using the Natural Earth land mask
    mask = xr.where(np.isnan(land.mask(lon_or_obj=area.lon, lat=area.lat)), 0, 1)
    la = area * mask
    la.attrs["units"] = "m2"
    la.attrs["long_name"] = "land grid cell area"
    la.attrs["method"] = method

    return obs, area, la


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_ols_dataset(
    da: xr.DataArray,
    dsc_dim: str,
    region: regmask._OneRegion | None = None,
    chunks: dict[str, int] = ANALYSIS_CHUNKS,
) -> xr.Dataset:
    """Compute the OLS regression."""

    ols_dim = {
        "month": "decimal_year",
        "year": "year",
    }

    da_chunked = da.chunk(chunks)    
    dsc = deseasonalize(da_chunked, dsc_dim)
    ds = ols_over_time(dsc, ols_dim[dsc_dim])

    if dsc_dim == "month":
        ds.attrs["anomaly_method"] = "subtract the climatological monthly mean from the monthly timeseries"
    elif dsc_dim == "year":
        ds.attrs["anomaly_method"] = "subtract the long-term mean from the annual mean timeseries"
    
    if region is not None:
        ds.attrs["anomaly_method"] = "subtract the climatological monthly mean from the monthly timeseries"
    
    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute temporal trends in observational products.",
    )
    parser.add_argument(
        "variable",
        choices=["et", "lai"],
        type=str,
        help="Variable name.",
    )
    parser.add_argument(
        "--dataset",
        nargs='*',
        default=[],
        help=("Which dataset to load."),
    )
    parser.add_argument(
        "--time-start",
        default=None,
        help=(
            "Start of the analysis period, format YYYY-MM or YYYY-MM-DD. "
            f"Default: length of each observational product"
        ),
    )
    parser.add_argument(
        "--time-stop",
        default=None,
        help=(
            "End of the analysis period, format YYYY-MM or YYYY-MM-DD. "
            f"Default: length of each observational product"
        ),
    )
    parser.add_argument(
        "--dsc",
        default=DSC_DIM,
        help=(
            "Method for removing the seasonal cycle. "
            "Remove the climatological monthly mean (month) or "
            "subtract the long-term mean from the annual mean (year). "
            f"Default: {DSC_DIM}"
        )
    )
    parser.add_argument(
        "--regions",
        action="store_true",
        help=(
            "Compute OLS on the area-weighted regional mean for each region "
            "instead of at the gridcell level. Output has a 'region' dimension."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--dask-cluster",
        action="store_true",
        help="Spin up a Dask cluster via xclim.create_dask_cluster.",
    )
    parser.add_argument(
        "--dask-workers",
        default=2,
        type=int,
        help=(
            "Number of Dask workers (equal to ncores). "
            "Default: 2"
        )
    )
    parser.add_argument(
        "--dask-memory",
        default='16GB',
        type=str,
        help=(
            "Amount of memory for Dask cluster. "
            "Default: '16GB'"
        )
    )
    parser.add_argument(
        "--dask-walltime",
        default="01:00:00",
        type=str,
        help=(
            "Walltime for Dask cluster. "
            "Default: '01:00:00'"
        )
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    var = args.variable
    if len(args.dataset) == 0:
        datasets = OBS_PRODUCTS[var]
    else:
        datasets = args.dataset
        for dsname in datasets:
            assert dsname in OBS_PRODUCTS[var]
    
    print(f"Variable: {var}")
    print(f"Datasets: {datasets}")
    print("="*20)
    
    client_cluster = None
    if args.dask_cluster:
        client_cluster = xclim.create_dask_cluster(
            account="UWAS0155",
            nworkers=args.dask_workers,
            ncores=args.dask_workers,
            nmem=args.dask_memory,
            walltime=args.dask_walltime,
        )
        client_cluster[0].wait_for_workers(args.dask_workers)

    try:
        for dsname in datasets:

            # ------------------------------------------------------------------
            # 1. Handle paths
            # ------------------------------------------------------------------
            if args.output is None:
                ols_path = DEFAULT_OUTPUT_DIR[var]
            elif args.output.is_dir():
                ols_path = args.output                
            else:
                raise ValueError(f"Output path must be a directory.")

            ols_path.parent.mkdir(parents=True, exist_ok=True)

            # ------------------------------------------------------------------
            # 2. Load variable
            # ------------------------------------------------------------------
            da, _, weights = load_ilamb_obs(var, dsname, args.time_start, args.time_stop)

            # Materialize da on workers
            if client_cluster is not None:
                da = client_cluster[0].persist(da.chunk(ANALYSIS_CHUNKS))
                wait(da)  # block until all chunks are resident on workers

            # ------------------------------------------------------------------
            # 3. Compute OLS — gridcell or regional
            # ------------------------------------------------------------------
            print("Computing OLS...")
            if args.regions:
                # Spatial average first, then trend — mathematically equivalent to
                # averaging gridcell slopes for OLS, but cleaner for inference.
                ols_ds = compute_regional_ols(da, args.dsc, weights=weights, regions=REGIONS)
                region_tag = "giorgi_regions"
            else:
                ols_ds = build_ols_dataset(da, args.dsc)
                region_tag = None
            
            # Generate filename
            if args.time_start is None:
                start_str = to_yyyymm(da.time[0])
            else:
                start_str = args.time_start.replace('-', '')
            
            if args.time_stop is None:
                stop_str = to_yyyymm(da.time[-1])
            else:
                stop_str = args.time_stop.replace('-', '')
            
            suffix = f"_{region_tag}" if region_tag else ""
            fname = (
                f"{var}_{dsname}_{args.dsc}_dsc{suffix}_ols_stats"
                f"_{start_str}-{stop_str}.nc"
            )

            ols_ds.to_netcdf(ols_path / fname)
            print(f"Wrote {ols_path / fname}")

            del ols_ds
            gc.collect()
            if client_cluster is not None:
                client_cluster[0].run(gc.collect)
        
    finally:
        if client_cluster is not None:
            xclim.close_dask_cluster(client_cluster)

    
if __name__ == "__main__":
    main()
