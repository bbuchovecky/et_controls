"""
Compute linear trends in variables at the grid-cell level.
"""

from __future__ import annotations

import gc
import argparse
from pathlib import Path
from distributed import wait

import numpy as np
import xarray as xr
from scipy.stats import t as t_dist
import regionmask as regmask

import xclimate as xclim


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DSC_DIM = "year"
DEFAULT_TIME_START = "1950-01"
DEFAULT_TIME_STOP = "2014-12"
DEFAULT_ANALYSIS_CHUNKS = {
    "time": 120,
    "member": 1,
    "lat": 48,
    "lon": 96,
}
DEFAULT_REGIONS = regmask.defined_regions.giorgi


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
    regions: regmask.Regions = DEFAULT_REGIONS,
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
# Data loading
# ---------------------------------------------------------------------------


_BAD_SCALAR_COORDS = {"landunit", "column", "pft"}


def _strip_bad_scalar_coords(da: xr.DataArray) -> xr.DataArray:
    """Reconstruct a DataArray without scalar coords that alias a dimension name.
    
    Methods like drop_vars() internally call _to_temp_dataset(), which raises
    ValueError if a scalar coordinate name matches a dimension name. This avoids
    that path entirely by constructing a new DataArray from the raw Variable.
    """
    clean_coords = {k: v for k, v in da.coords.items() if k not in _BAD_SCALAR_COORDS}
    return xr.DataArray(da.variable, coords=clean_coords, name=da.name, attrs=da.attrs)


def load_fhist_variable(
    variable: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    members: list[int] | None = None,
) -> xr.DataArray:
    """Load FHIST data and align it to the PPE grid."""

    time_slice = slice(time_start, time_stop)
    grid = xclim.load_fhist_ppe_grid()

    vv = "_".join(variable.split("_")[:-2])
    da = (
        xclim.load_fhist(variable, keep_var_only=True)[vv]
        .sel(time=time_slice)
        .reindex_like(grid, method="nearest", tolerance=1e-3)
        .where(grid.LANDFRAC > 0)
    )

    if members is not None:
        da = da.sel(member=members)

    return da


def load_lens2_variable(
    variable: str,
    stream: str,
    gcomp: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    members: list[int] | None = None,
) -> xr.DataArray:
    """Load LENS2 data and align it to the LENS2 grid."""

    time_slice = slice(time_start, time_stop)
    grid = xclim.load_cesm2le_grid()

    vv = "_".join(variable.split("_")[:-2])
    freq = "_".join(variable.split("_")[-2:])
    da = (
        xclim.load_cesm2le(vv, gcomp, freq, stream)[vv]
        .sel(time=time_slice)
        .reindex_like(grid, method="nearest", tolerance=1e-3)
        .where(grid.LANDFRAC > 0)
    )

    if members is not None:
        da = da.sel(member=members)

    return _strip_bad_scalar_coords(da)


def load_goga2_variable(
    variable: str,
    stream: str,
    gcomp: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    members: list[int] | None = None,
) -> xr.DataArray:
    """Load GOGA2 data and align it to the GOGA2 grid."""

    time_slice = slice(time_start, time_stop)
    grid = xclim.load_goga2_grid()

    vv = "_".join(variable.split("_")[:-2])
    freq = "_".join(variable.split("_")[-2:])
    da = (
        xclim.load_goga2(vv, gcomp, freq, stream)[vv]
        .sel(time=time_slice)
        .reindex_like(grid, method="nearest", tolerance=1e-3)
        .where(grid.LANDFRAC > 0)
    )

    if members is not None:
        da = da.sel(member=members)

    return da


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_variable(
    dataset: str,
    variable: str,
    gcomp: str,
    stream: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    members: list[int] | None = None,
) -> xr.DataArray:
    """Route to the appropriate dataset loader.

    Parameters
    ----------
    dataset : {"fhist", "lens2", "goga2"}
        Which ensemble to load.
    variable : str
        Variable name; interpretation is dataset-specific (see individual
        loader docstrings).
    time_start, time_stop : str
        ISO time bounds.
    members : list of int or None
        Member indices to load; None loads all.

    Returns
    -------
    xr.DataArray
        Dimensions: (member, time, lat, lon).
    """
    if dataset == "fhist":
        return load_fhist_variable(
            variable=variable,
            time_start=time_start,
            time_stop=time_stop,
            members=members,
        )
    elif dataset == "lens2":
        return load_lens2_variable(
            variable=variable,
            gcomp=gcomp,
            stream=stream,
            time_start=time_start,
            time_stop=time_stop,
            members=members,
        )
    elif dataset == "goga2":
        return load_goga2_variable(
            variable=variable,
            gcomp=gcomp,
            stream=stream,
            time_start=time_start,
            time_stop=time_stop,
            members=members,
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.  Choose 'fhist', 'lens2', or 'goga2'.")


def load_area_weights(dataset: str) -> xr.DataArray:
    """
    Load grid-cell land area weights for the specified dataset.

    Parameters
    ----------
    dataset : {"fhist", "lens2", "goga2"}

    Returns
    -------
    xr.DataArray, dims (lat, lon)
        Land area weights in whatever units the source provides;
        normalization is handled downstream in compute_regional_ols.
    """
    if dataset == "fhist":
        return xclim.load_fhist_ppe_grid().LANDAREA
    elif dataset == "lens2":
        return xclim.load_cesm2le_grid().LANDAREA
    elif dataset == "goga2":
        return xclim.load_goga2_grid().LANDAREA
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'fhist', 'lens2', or 'goga2'.")


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


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_ols_dataset(
    da: xr.DataArray,
    dsc_dim: str,
    region: regmask._OneRegion | None = None,
    chunks: dict[str, int] = DEFAULT_ANALYSIS_CHUNKS,
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
        description=(
            "Compute temporal trends in either FHIST PPE, LENS2, or GOGA2."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["fhist", "lens2", "goga2"],
        default="fhist",
        help=(
            "Which ensemble to load.  "
            "'fhist' uses xclim.load_fhist; "
            "'lens2' uses xclim.load_cesm2le; "
            "'goga2' uses xclim.load_goga2. "
            "Default: fhist"
        ),
    )
    parser.add_argument(
        "--variable",
        required=True,
        nargs="+",
        type=str,
        help=(
            "Variable name(s), including frequency suffix (e.g. FCTR_month_1). "
        ),
    )
    parser.add_argument(
        "--gcomp",
        choices=["lnd", "atm"],
        default="lnd",
        help=(
            "General model component for history output."
            "For use with GOGA2 and LENS2;"
            "{'lnd' or 'atm'}."
        ),
    )
    parser.add_argument(
        "--stream",
        default="h0",
        type=str,
        help=(
            "Stream for history output."
            "For use with GOGA2 and LENS2;"
            "(e.g., h0 or h6)."
        ),
    )
    parser.add_argument(
        "--time-start",
        default=DEFAULT_TIME_START,
        help=(
            "Start of the analysis period, format YYYY-MM or YYYY-MM-DD. "
            f"Default: {DEFAULT_TIME_START}"
        ),
    )
    parser.add_argument(
        "--time-stop",
        default=DEFAULT_TIME_STOP,
        help=(
            "End of the analysis period, format YYYY-MM or YYYY-MM-DD. "
            f"Default: {DEFAULT_TIME_STOP}"
        ),
    )
    parser.add_argument(
        "--members",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of member indices to process",
    )
    parser.add_argument(
        "--dsc",
        default=DEFAULT_DSC_DIM,
        help=(
            "Method for removing the seasonal cycle. "
            "Remove the climatological monthly mean (month) or "
            "subtract the long-term mean from the annual mean (year). "
            f"Default: {DEFAULT_DSC_DIM}"
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
        default=Path(__file__).with_name("ols_regression_stats.nc"),
        help="Output NetCDF path",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client_cluster = None
    if args.dask_cluster:
        client_cluster = xclim.create_dask_cluster(
            account='UWAS0155',
            nworkers=args.dask_workers,
            ncores=args.dask_workers,
            nmem=args.dask_memory,
            walltime='02:00:00',
        )
        client_cluster[0].wait_for_workers(args.dask_workers)
    
    try:
        for var in args.variable:
            print(var, args.gcomp, args.stream)

            # ------------------------------------------------------------------
            # 1. Handle paths
            # ------------------------------------------------------------------
            ols_path = args.output

            var_no_time = "_".join(var.split("_")[:-2])
            fname = (
                f"{args.dataset.upper()}.{var_no_time}.{args.dsc}_dsc.ols_stats"
                f".{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}.nc"
            )

            if len(args.variable) > 1:
                if args.output.is_dir():
                    ols_path = args.output
                else:
                    raise ValueError(f"Output path must be a directory for len(variable) > 1.")
            else:
                if args.output.is_dir():
                    ols_path = args.output            
                
            ols_path.parent.mkdir(parents=True, exist_ok=True)

            # ------------------------------------------------------------------
            # 2. Load variable
            # ------------------------------------------------------------------
            print(f"Loading {args.dataset.upper()}...")
            da = load_variable(
                dataset=args.dataset,
                variable=var,
                gcomp=args.gcomp,
                stream=args.stream,
                time_start=args.time_start,
                time_stop=args.time_stop,
                members=args.members,
            )

            # Materialize da on workers
            if client_cluster is not None:
                da = client_cluster[0].persist(da.chunk(DEFAULT_ANALYSIS_CHUNKS))
                wait(da)  # block until all chunks are resident on workers

            # ------------------------------------------------------------------
            # 3. Compute OLS — gridcell or regional
            # ------------------------------------------------------------------
            print("Computing OLS...")
            if args.regions:
                # Spatial average first, then trend — mathematically equivalent to
                # averaging gridcell slopes for OLS, but cleaner for inference.
                weights = load_area_weights(args.dataset)
                ols_ds = compute_regional_ols(da, args.dsc, weights=weights, regions=DEFAULT_REGIONS)
                region_tag = "giorgi_regions"
            else:
                ols_ds = build_ols_dataset(da, args.dsc)
                region_tag = None
            
            # Update filename
            suffix = f".{region_tag}" if region_tag else ""
            fname = (
                f"{args.dataset.upper()}.{var_no_time}.{args.dsc}_dsc"
                f"{suffix}.ols_stats"
                f".{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}.nc"
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
