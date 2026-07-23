"""
Calculate temperature distribution statistics.

Arguments:
    --dataset
    --start-time
    --stop-time
    --window-days


Pool: maximum 30-day running mean

Load daily temperature variable
For each gridcell, select 30 days with climatological maximum mean temperature
Pool 30 days across full time period
Compute per-member anomalies relative to the climatological day-of-year value
Compute distribution statistics for anomalies
    - mean, median, standard deviation, skewness, kurtosis
    - 90p, 95p, 99p
Compute Gaussian kernel density estimate - for raw values and anomalies
    - sklearn.neighbors.KernelDensity
    - fit to data, then score_samples with X=np.linspace(x0, x1, n)
Build dataset
    - data_vars: quantile, mean, median, stddev
    - attrs: n

Plot map of 30-day window center
Plot per-member Gaussian KDE
"""

from __future__ import annotations

import os
import time
import logging
import argparse
from pathlib import Path
from distributed import wait

import numpy as np
import xarray as xr
from scipy.stats import kurtosis, skew, gaussian_kde

import xclimate as xclim
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VARIABLE = "TREFHT_day_1"
DEFAULT_TIME_START = "1985-01-01"
DEFAULT_TIME_STOP = "2014-12-31"
DEFAULT_WINDOW_DAYS = 30

DEFAULT_GRID = {
    "fhist": xclim.load_fhist_ppe_grid(),
    "lens2": xclim.load_cesm2le_grid(),
    "goga2": xclim.load_goga2_grid(),
}

DEFAULT_ANALYSIS_CHUNKS = {
    "time": -1,
    "member": 1,
    "lat": 48,
    "lon": 96,
}
DEFAULT_ANALYSIS_CHUNKS_NOTIME = {
    "member": 1,
    "lat": 48,
    "lon": 96,
}
DEFAULT_OUTPUT_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/proc/dist")
DEFAULT_LOG_PATH = Path("/glade/u/home/bbuchovecky/projects/et_controls/temp_extremes/logs")


# ---------------------------------------------------------------------------
# Setup timing
# ---------------------------------------------------------------------------

class Caliper:
    def __init__(self, logger):
        self.t0 = time.perf_counter()
        self.t_last = time.perf_counter()
        self.logger = logger

    @staticmethod
    def _format_hms(seconds: float) -> str:
        """Format a duration in seconds as H:MM:SS.ss (hours unbounded, not wrapped at 24)."""
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"

    def lap(self, label):
        t_now = time.perf_counter()
        dt0 = t_now - self.t0
        dt = t_now - self.t_last
        self.t_last = t_now
        self.logger.info(
            f"{label}: Lap = {self._format_hms(dt)}, Total = {self._format_hms(dt0)}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_time_string(time_str: str) -> tuple[int, ...]:
    """
    Parse time string with format 'year-month-day', 'year-month', or 'year'
    to a tuple of integers with format (year, month, day). If month or day are
    not included, sets them to 1 by default.
    """

    time_str_split = time_str.split("-")
    if len(time_str_split) == 2:
        time_str_split += ["01"]
    elif len(time_str_split) == 1:
        time_str_split += ["01", "01"]
    elif (len(time_str_split) > 3) or (len(time_str_split) == 0):
        raise ValueError("Time string must contain year, month, or day.")
    return tuple(int(s) for s in time_str_split)


def to_datetime(time_str: str) -> datetime:
    """
    Convert a time string 'YYYY-MM-DD', 'YYYY-MM', or 'YYYY' to a
    datetime.datetime at midnight (00:00:00). Useful for subtracting to
    obtain timedeltas.
    """
    year, month, day = parse_time_string(time_str)
    return datetime(year, month, day)


# def to_datetime(date_string: str, format_mask: str = "%Y-%m-%d") -> datetime:
#     """
#     Convert a time string to a datetime.datetime for subtracting to obtain timedeltas.
#     """
#     return datetime.strptime(date_string, format_mask)


def format_timedelta_string(td: timedelta) -> str:
    """Generate formatted timedelta string."""
    years, days_remainder = divmod(td.days, 365)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{years:02d}y:{days_remainder:02d}d:{hours:02d}h:{minutes:02d}m:{seconds:02d}s"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_max_window_doy(x: xr.DataArray, ndays: int = DEFAULT_WINDOW_DAYS) -> xr.DataArray:
    """
    Identify the ndays-day window of the annual cycle with the maximum
    mean, defined as the contiguous run of ndays days ending on the day
    that maximizes the trailing ndays-day rolling mean. Works for any
    ndays (odd or even) since it avoids the ambiguity of a "centered"
    window for even lengths.

    Parameters
    ----------
    x : xr.DataArray
        Climatological daily data with a 'dayofyear' dimension (plus
        arbitrary other dims, e.g. lat, lon, member).
    ndays : int, optional
        Window length in days.

    Returns
    -------
    xr.DataArray
        1-indexed dayofyear labels of the window, dims
        (*x.dims_excluding_dayofyear, 'window_day'), wrap-aware across
        the Dec-Jan boundary. Use with .sel(dayofyear=...) to subset
        other variables sharing the same dayofyear coordinate convention.
    """
    n_doy = x.sizes["dayofyear"]
    if ndays > n_doy:
        raise ValueError(f"ndays ({ndays}) exceeds available days ({n_doy})")

    pad = ndays - 1
    x_padded = xr.concat(
        [x.isel(dayofyear=slice(-pad, None)), x],
        dim="dayofyear",
    )

    trailing_mean = (
        x_padded.rolling(dayofyear=ndays, center=False)
        .mean()
        .isel(dayofyear=slice(pad, None))  # restrict candidates to original year
        .fillna(-np.inf)
    )

    end_iday = trailing_mean.argmax(dim="dayofyear").compute()  # 0-indexed, within original year

    offsets = np.arange(-(ndays - 1), 1)  # e.g. [-29, ..., 0] for ndays=30
    doy_labels = np.mod(end_iday.values[..., None] + offsets, n_doy) + 1  # 1-indexed

    return xr.DataArray(
        doy_labels,
        dims=end_iday.dims + ("window_day",),
        coords={**end_iday.coords, "window_day": np.arange(ndays)},
        name="max_window_doy",
        attrs={"long_name": f"{ndays}-day window of maximum {x.name}, day-of-year labels"},
    )


def select_window_multi_year(other: xr.DataArray, doy_labels: xr.DataArray) -> xr.DataArray:
    """
    For each year in `other`, select the days-of-year given by
    doy_labels, producing dims (year, window_day, *other_dims_minus_time).

    Assumes each year in `other` has the same calendar convention as
    doy_labels (e.g. both noleap, or both accounting for leap years
    consistently) and that each year is complete (no missing days).
    """
    idx = (doy_labels - 1).astype(int)  # 0-indexed offsets within a single year

    yearly = []
    for yr, year_slice in other.groupby("time.year"):
        n_doy_yr = year_slice.sizes["time"]
        if int(idx.max()) >= n_doy_yr:
            raise ValueError(f"doy_labels exceed year {yr}'s length ({n_doy_yr} days); leap-year mismatch?")
        sel = year_slice.isel(time=idx).assign_coords(window_day=doy_labels.window_day)
        sel = sel.expand_dims(year=[yr])
        yearly.append(sel.drop_vars("time", errors="ignore"))

    return xr.concat(yearly, dim="year")


def lag1_autocorr_pooled(
    x: xr.DataArray,
    year_dim: str = "year",
    day_dim: str = "window_day",
) -> xr.DataArray:
    """
    Compute lag-1 autocorrelation pooled across years, using only
    within-year (within-window) consecutive day pairs. Assumes x is
    already a dayofyear anomaly (climatological seasonal cycle
    already removed), so only the sample mean of the pooled window is
    subtracted here -- not a second climatological correction.

    r1 = sum_{y,t} [(x_{y,t} - m)(x_{y,t+1} - m)] / sum_{y,t} [(x_{y,t} - m)^2]

    where y indexes year, t indexes window_day, and m is the sample
    mean of x over the pooled (year, window_day) selection (expected
    to be near zero, but not exactly zero in a finite subsample of an
    anomaly field -- see Note below).

    Parameters
    ----------
    x : xr.DataArray
        Dayofyear-anomaly data with dims (year_dim, day_dim, ...),
        e.g. output of select_window_multi_year applied to an
        already-anomalized field.
    year_dim : str, optional
        Name of the year dimension.
    day_dim : str, optional
        Name of the within-window day dimension (chronologically
        ordered within each year; must NOT span across years).

    Returns
    -------
    xr.DataArray
        Pooled lag-1 autocorrelation, dims = x.dims minus
        (year_dim, day_dim).

    Notes
    -----
    Subtracting the pooled sample mean here (rather than assuming
    x's mean is exactly zero) matters because this 30-day window was
    itself selected via max_window_doy as the mean-maximizing window
    -- i.e., it's a biased subsample of the anomaly field by
    construction. Its sample mean over this specific window/years
    subset need not be zero even though x is a global anomaly.
    Skipping the demeaning step and using raw x directly would treat
    that selection-induced offset as part of the "signal" in the
    autocorrelation numerator, which is not what you want.
    """
    def _lag1_pooled(arr):
        # arr shape (..., n_year, n_day) per apply_ufunc core-dim convention
        m = np.nanmean(arr, axis=(-2, -1), keepdims=True)
        anom = arr - m
        num = np.nansum(anom[..., :, :-1] * anom[..., :, 1:], axis=(-2, -1))
        den = np.nansum(anom**2, axis=(-2, -1))
        return num / den

    return xr.apply_ufunc(
        _lag1_pooled,
        x,
        input_core_dims=[[year_dim, day_dim]],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[float],
    ).rename(f"lag1_autocorr_pooled_{x.name}" if x.name else "lag1_autocorr_pooled")


def descriptive_stats(x: xr.DataArray, dim: str, q_frac: list | np.ndarray) -> xr.Dataset:
    """Compute descriptive statistics."""
    x_n = x[dim].size
    x_sigma = x.std(dim=dim)
    x_mu = x.mean(dim=dim)
    x_skew = skewness_along_dim(x, dim=dim)
    x_kurt = kurtosis_along_dim(x, dim=dim).assign_attrs({"excess": "True"})
    x_min = x.min(dim=dim)
    x_max = x.max(dim=dim)
    x_q = x.quantile(q_frac, dim="pool")

    return xr.Dataset(
        data_vars=dict(
            sigma=x_sigma,
            mu=x_mu,
            skew=x_skew,
            kurt=x_kurt,
            min=x_min,
            max=x_max,
            quantiles=x_q,
        ),
        attrs={
            "variable": x.name,
            "n": x_n,
            "stats_dim": dim,
            **x.attrs,
        }
    )


def skewness_along_dim(
    da: xr.DataArray,
    dim: str = "time",
    bias: bool = True,
    nan_policy: str = "propagate",
) -> xr.DataArray:
    """Third standardised moment along *dim* via scipy.stats.skew."""

    return xr.apply_ufunc(
        skew,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={
            "axis": -1,
            "bias": bias,
            "nan_policy": nan_policy,
        },
    )


def kurtosis_along_dim(
    da: xr.DataArray,
    dim: str = "time",
    fisher: bool = True,
    bias: bool = True,
    nan_policy: str = "propagate",
) -> xr.DataArray:
    """Fourth standardised moment along *dim* via scipy.stats.kurtosis.

    Parameters
    ----------
    fisher : bool
        If True, Fisher's definition (excess kurtosis; Normal → 0).
        If False, Pearson's definition (Normal → 3).
    """

    return xr.apply_ufunc(
        kurtosis,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={
            "axis": -1,
            "fisher": fisher,
            "bias": bias,
            "nan_policy": nan_policy,
        },
    )


def kde_pdf(
        x: xr.DataArray,
        dim: str = "window_day",
        eval_points: np.ndarray | None = None,
        bw_method: str = "scott",
)-> xr.DataArray:
    """
    Gaussian KDE of x along `dim`, evaluated at `eval_points`.

    Parameters
    ----------
    x : xr.DataArray
        Data with dimension `dim` containing the samples to fit the
        KDE to (e.g. pooled window_day x year values).
    dim : str, optional
        Dimension containing samples.
    eval_points : np.ndarray, optional
        Points at which to evaluate the estimated density. Defaults to
        200 points spanning the sample's observed range.
    bw_method : str or float, optional
        Passed to scipy.stats.gaussian_kde ('scott', 'silverman', or a
        scalar). Defaults to scipy's default ('scott').

    Returns
    -------
    xr.DataArray
        Estimated density, dims = (*x.dims_minus_dim, 'eval_point').

    Notes
    -----
    Does not account for serial correlation among samples along `dim`
    -- gaussian_kde treats all input points as i.i.d. draws. Given the
    lag-1 autocorrelation work upstream in this pipeline, the
    effective sample size feeding this KDE is likely well below the
    raw n; this will make the KDE appear more confident (less noisy)
    than is actually justified. Consider subsampling to
    ~n_eff independent points, or treat the KDE as descriptive only,
    not for density-based inference (e.g. don't derive CIs from it
    without a block-bootstrap or similar correction).
    """
    if eval_points is None:
        lo, hi = float(x.min()), float(x.max())
        pad = 0.1 * (hi - lo)
        eval_points = np.linspace(lo - pad, hi + pad, 200)

    def _kde(arr):
        arr = arr[~np.isnan(arr)]
        if arr.size < 2:
            return np.full(len(eval_points), np.nan)
        kde = gaussian_kde(arr, bw_method=bw_method)
        return kde(eval_points)

    result = xr.apply_ufunc(
        _kde,
        x,
        input_core_dims=[[dim]],
        output_core_dims=[["eval_point"]],
        exclude_dims={dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"eval_point": len(eval_points)}},
    )
    result = result.rename("kde").assign_coords(eval_point=eval_points)
    result.attrs = {
        "long_name": "Gaussian KDE",
        "bw_factor": bw_method,
        "method": "scipy.stats.gaussian_kde",
    }

    return result


# ---------------------------------------------------------------------------
# Load datasets
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
    grid: xr.DataArray | xr.Dataset = DEFAULT_GRID["fhist"],
) -> xr.DataArray:
    """Load FHIST data and align it to the PPE grid."""
    time_slice = slice(time_start, time_stop)

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
    grid: xr.DataArray | xr.Dataset = DEFAULT_GRID["lens2"],
) -> xr.DataArray:
    """Load LENS2 data and align it to the LENS2 grid."""
    time_slice = slice(time_start, time_stop)

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
    grid: xr.DataArray | xr.Dataset = DEFAULT_GRID["goga2"],
    members: list[int] | None = None,
) -> xr.DataArray:
    """Load GOGA2 data and align it to the GOGA2 grid."""
    time_slice = slice(time_start, time_stop)

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
        default=DEFAULT_VARIABLE,
        help=(
            "Variable name.  Daily variable including frequency suffix (e.g. TREFMXAV_day_1). "
            f"Default: {DEFAULT_VARIABLE}"
        ),
    )
    parser.add_argument(
        "--gcomp",
        choices=["lnd", "atm"],
        default="lnd",
        help=(
            "General model component for history output. "
            "For use with GOGA2 and LENS2."
        ),
    )
    parser.add_argument(
        "--stream",
        default="h1",
        type=str,
        help=(
            "Stream for history output. "
            "For use with GOGA2 and LENS2; "
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
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=(
            "Length of window in days for calculating maximum running mean. "
            f"Default: {DEFAULT_WINDOW_DAYS}"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output NetCDF path. "
            f"Default: {DEFAULT_OUTPUT_PATH}"
        ),
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
        default='32GB',
        type=str,
        help=(
            "Amount of memory for Dask cluster. "
            "Default: '32GB'"
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

    if not args.output.is_dir():
        raise ValueError("`--output` must be a directory.")
    
    # Handle paths
    varname = "_".join(args.variable.split("_")[:-2])
    clim_path = args.output / "doy_climatology"
    base_path = args.output / f"{varname}_pool_{args.window_days}d_window"
    base_path.mkdir(parents=True, exist_ok=True)
    fname = f"{args.dataset.upper()}_{varname}_pool_{args.window_days}d_window_{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}"
    
    # Get current date and time
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    DEFAULT_LOG_PATH.mkdir(parents=True, exist_ok=True)
    dask_cluster_info = "no_dask"
    if args.dask_cluster:
        dask_cluster_info = f"dask_{args.dask_workers}ncores_{args.dask_memory}"
    log_path = Path(DEFAULT_LOG_PATH / f"{fname}.{dask_cluster_info}.{now}.log")

    # Setup logger
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    this_logger = logging.getLogger("caliper")
    this_logger.info(f"=== RUN START: {datetime.now().isoformat()} ===")

    # Start caliper timer
    c = Caliper(this_logger)

    # Log script settings
    this_logger.info(f"Output path: {base_path / fname}")
    this_logger.info(f"Dask - cluster: {int(args.dask_cluster)}, workers: {args.dask_workers}, memory: {args.dask_memory}")
    this_logger.info(f"Window length (days): {args.window_days}")
    
    # Parse time
    delta_time = to_datetime(args.time_stop) - to_datetime(args.time_start)
    this_logger.info(f"Start time: {to_datetime(args.time_start)}")
    this_logger.info(f"Stop time:  {to_datetime(args.time_stop)}")
    this_logger.info(f"Delta time: {format_timedelta_string(delta_time)}")

    # Create Dask cluster
    client_cluster = None
    if args.dask_cluster:
        this_logger.info("Creating Dask cluster...")
        client_cluster = xclim.create_dask_cluster(
            account='UWAS0155',
            nworkers=args.dask_workers,
            ncores=args.dask_workers,
            nmem=args.dask_memory,
            walltime=args.dask_walltime,
        )
        client_cluster[0].wait_for_workers(args.dask_workers)
        c.lap("Timing")

    try:
        # Load dataset
        this_logger.info(f"Loading {args.dataset.upper()}...")
        y = load_variable(
            dataset=args.dataset,
            variable=args.variable,
            gcomp=args.gcomp,
            stream=args.stream,
            time_start=args.time_start,
            time_stop=args.time_stop,
            members=args.members,
        )
        c.lap("Timing")

        # Load grid
        grid = DEFAULT_GRID[args.dataset]

        # Select latitude
        y = y.sel(lat=slice(-58, None))

        # Materialize da on workers
        if client_cluster is not None:
            da = client_cluster[0].persist(y.chunk(DEFAULT_ANALYSIS_CHUNKS))
            wait(da)  # block until all chunks are resident on workers
            c.lap("Timing")
        
        # Compute day-of-year climatology
        if (clim_path / f"{fname}_doy_climatology.nc").exists():
            this_logger.info("Loading day-of-year climatology...")
            y_clim = (
                xr.open_dataset(clim_path / f"{fname}_doy_climatology.nc")[varname]
                .reindex_like(grid, method="nearest", tolerance=1e-3)
                .sel(lat=slice(-58, None))
            )
            c.lap("Timing")
        else:
            this_logger.info("Computing day-of-year climatology...")
            y_clim = y.groupby("time.dayofyear").mean()
            y_clim.to_netcdf(clim_path / f"{fname}_doy_climatology.nc")
            c.lap("Timing")
        this_logger.info(f"y_clim {y_clim.dims} {y_clim.shape}")
        
        this_logger.info("Materialize day-of-year climatology...")
        y_clim = y_clim.compute()
        c.lap("Timing")

        # Compute anomaly
        this_logger.info("Computing day-of-year anomalies...")
        y_anom = y.groupby("time.dayofyear") - y_clim
        c.lap("Timing")

        # Compute climatological maximum n-days running mean
        if (base_path / f"{fname}_doy.nc").exists():
            this_logger.info(f"Loading climatological maximum {args.window_days}-day running mean...")
            window_doy = (
                xr.open_dataset(base_path / f"{fname}_doy.nc")["max_window_doy"]
                .reindex_like(grid, method="nearest", tolerance=1e-3)
                .sel(lat=slice(-58, None))
            )
            c.lap("Timing")
        else:
            this_logger.info(f"Computing climatological maximum {args.window_days}-day running mean...")
            window_doy = compute_max_window_doy(y_clim, args.window_days)
            window_doy.to_netcdf(base_path / f"{fname}_doy.nc")
            c.lap("Timing")
        this_logger.info(f"window_doy {window_doy.dims} {window_doy.shape}")

        # Materialize anomalies to reduce task graph size
        this_logger.info("Materializing anomalies before windowed selection...")
        # y_anom = y_anom.compute()
        y_anom_list = []
        for m in y_anom.member.values:
            y_anom_m = y_anom.sel(member=m).compute()
            y_anom_list.append(y_anom_m)
            this_logger.info(f"Materialized member {m}")
            c.lap("Timing")
        y_anom = xr.concat(y_anom_list, dim="member")
        c.lap("Timing")

        # Select corresponding days; should (almost) always use multi-year
        this_logger.info("Selecting corresponding days within window...")
        if delta_time > timedelta(days=365):
            y_window = select_window_multi_year(y_anom, window_doy)
            c.lap("Timing")
        else:
            raise ValueError("The time period must be at least one year long.")
        
        # Pool over (year, window_day) dimensions
        this_logger.info("Pooling (year, window_day) dimensions...")
        y_pooled = y_window.stack(pool=["year", "window_day"])
        c.lap("Timing")
        # y_pooled.to_netcdf(base_path / f"{fname}_pool_{args.window_days}d_window_sample.nc")

        # Compute statistics over the pooled values
        this_logger.info("Computing statistics...")
        q_frac = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        y_pooled_stats = descriptive_stats(y_pooled, "pool", q_frac)
        this_logger.info("Saving statistics...")
        y_pooled_stats.to_netcdf(base_path / f"{fname}_stats.nc")
        this_logger.info("Done saving statistics.")
        c.lap("Timing")

        # TODO: compute the effective degrees of freedom for a single time series -
        # the following computes the lag-1 autocorrelation used in significance tests
        # of the correlation between two times series
        # y_window_rho_lag1 = lag1_autocorr_pooled(y_window, "year", "window_day")

        # Compute Gaussian KDE
        this_logger.info("Computing Gaussian KDE...")
        y_pooled_kde = kde_pdf(y_pooled, dim="pool")
        y_pooled_kde.attrs["neff"] = "no adjustment for neff (=n)"
        this_logger.info("Saving Gaussian KDE...")
        y_pooled_kde.to_netcdf(base_path / f"{fname}_kde.nc")
        this_logger.info("Done saving Gaussian KDE.")
        c.lap("Timing")
    
    finally:        
        if client_cluster is not None:
            xclim.close_dask_cluster(client_cluster)
        this_logger.info(f"=== RUN END: {datetime.now().isoformat()} ===")


if __name__ == "__main__":
    main()
