"""
Calculate temperature distribution statistics of day-of-year anomalies.

Structure:
----------
Load daily temperature variable
For each gridcell, select n days with climatological maximum mean temperature
Compute per-member anomalies relative to the climatological day-of-year value
Pool annual n days across full time period
Compute distribution statistics for anomalies
    - mean, median, standard deviation, skewness, kurtosis
    - 5p, 10p , 25p, 50p, 75p, 90p, 95p,
Compute Gaussian kernel density estimate
    - same per-member eval points but vary across gridcells

Arguments:
----------
  --dataset {fhist,lens2,goga2}
        Which ensemble to load.
        'fhist' uses xclim.load_fhist;
        'lens2' uses xclim.load_cesm2le;
        'goga2' and 'goga2_regrid' uses xclim.load_goga2;
        Default: fhist
  --variable VARIABLE
        Variable name. Daily variable including frequency suffix (e.g. TREFMXAV_day_1). Default: TREFHT_day_1
  --gcomp {lnd,atm}
        General model component for history output. For use with GOGA2 and LENS2.
  --stream STREAM
        Stream for history output. For use with GOGA2 and LENS2; (e.g., h0 or h6).
  --time-start TIME_START
        Start of the analysis period, format YYYY-MM or YYYY-MM-DD. Default: 1985-01-01
  --time-stop TIME_STOP
        End of the analysis period, format YYYY-MM or YYYY-MM-DD. Default: 2014-12-31
  --window-days WINDOW_DAYS
        Length of window in days for calculating maximum running mean. Default: 30
  --output OUTPUT
        Output NetCDF path. Default: /glade/work/bbuchovecky/fhist_ppe_analysis/proc/dist
  --calc-clim
        Force calculation of the day-of-year climatology.
  --calc-doy
        Force calculation of the day-of-year labels for the maximum window mean.
  --tag TAG
        Tag to put at the end of the file name.
  --dask-cluster
        Spin up a Dask cluster via xclim.create_dask_cluster.
  --dask-workers DASK_WORKERS
        Number of Dask workers (equal to ncores). Default: 2
  --dask-memory DASK_MEMORY
        Amount of memory for Dask cluster. Default: '32GB'
  --dask-walltime DASK_WALLTIME
        Walltime for Dask cluster. Default: '01:00:00'
"""

from __future__ import annotations

import os
import time
import logging
import argparse
from pathlib import Path
from distributed import wait
import dask.config

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
    "goga2_2deg": xclim.load_fhist_ppe_grid(),
    "cmip6_fhist": xclim.load_fhist_ppe_grid(),
}

DEFAULT_ANALYSIS_CHUNKS = {
    "time": -1,
    "member": 1,
    "lat": 96,
    "lon": 144,
}
DEFAULT_ANALYSIS_CHUNKS_NOMEMBER = {
    "time": -1,
    "lat": 96,
    "lon": 144,
}
DEFAULT_ANALYSIS_CHUNKS_NOTIME = {
    "member": 1,
    "lat": 48,
    "lon": 96,
}
DEFAULT_OUTPUT_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/proc/dist")
DEFAULT_LOG_PATH = Path("/glade/u/home/bbuchovecky/projects/et_controls/temp_extremes/logs")
CACHE_PATHS = {
    "fhist": Path("/glade/derecho/scratch/bbuchovecky/derived/daily_distributions/fhist_anom_pooled_day"),
    "goga2": Path("/glade/derecho/scratch/bbuchovecky/derived/daily_distributions/goga2_anom_pooled_day"),
    "lens2": Path("/glade/derecho/scratch/bbuchovecky/derived/daily_distributions/lens2_anom_pooled_day"),
    "cmip6_fhist": Path("/glade/derecho/scratch/bbuchovecky/derived/daily_distributions/cmip6_fhist_anom_pooled_day"),
}

MEMBERS = {
    # FHIST PPE, n=27 (dropped outlier members 13 and 28 as done in xclimate.load_ppe)
    "fhist": [
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12,     14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 
    ],
    # GOGA2, n=10
    "goga2": [
        "01", "02", "03", "04", "05",
        "06", "07", "08", "09", "10",
    ],
    # LENS2, n=40 (members with daily temperature output)
    "lens2": [
        "1231.001", "1231.002", "1231.003", "1231.004", "1231.005",
        "1231.006", "1231.007", "1231.008", "1231.009", "1231.010",

        "1251.001", "1251.002", "1251.003", "1251.004", "1251.005",
        "1251.006", "1251.007", "1251.008", "1251.009", "1251.010",

        "1281.001", "1281.002", "1281.003", "1281.004", "1281.005",
        "1281.006", "1281.007", "1281.008", "1281.009", "1281.010",

        "1301.001", "1301.002", "1301.003", "1301.004", "1301.005",
        "1301.006", "1301.007", "1301.008", "1301.009", "1301.010",
    ],
    # CMIP6 FHIST
    "cmip6_fhist": ["001", "002", "003"],
}

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
            f"{label}: LAP = {self._format_hms(dt)}, TOTAL = {self._format_hms(dt0)}"
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


def format_timedelta_string(td: timedelta) -> str:
    """Generate formatted timedelta string."""
    years, days_remainder = divmod(td.days, 365)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{years:02d}y:{days_remainder:02d}d:{hours:02d}h:{minutes:02d}m:{seconds:02d}s"


def check_if_coords_equal(a: xr.DataArray, b: xr.DataArray):
    if a.dims != b.dims:
        raise ValueError(f"{a.dims} != {b.dims}")
    for dim in a.dims:
        if (dim in a.coords) and (dim in b.coords):
            if not (a[dim] == b[dim]).all():
                raise ValueError(f"`{dim}` does not match.")
        else:
            print(f"`{dim}` does not have a corresponding coordinate.")



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

    Note: intended to be called with `other` already NumPy-backed
    (computed) and typically already sliced to a single member. Calling
    this on a Dask-backed, multi-member array triggers a vectorized
    (fancy) index whose per-pixel index variation causes severe Dask
    task-graph inflation -- see project notes.
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
    x_q = x.quantile(q_frac, dim=dim)

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
        lo: xr.DataArray | float | None = None,
        hi: xr.DataArray | float | None = None,
        npoints: int = 200,
        pad_frac: float = 0.1,
        bw_method: str = "scott",
) -> xr.Dataset:
    """
    Gaussian KDE of x along `dim`, evaluated at per-gridcell points.

    `lo`/`hi` are broadcast against x's non-`dim` dims (e.g. lat, lon)
    and are constant along `dim`. Pass them as DataArrays computed
    once from the pooled range across all ensemble members
    (`x_all.min(dim=["member", dim])`, `.max(...)`) so that every
    member's KDE for a given gridcell is evaluated on the identical
    grid -- required for any downstream operation that treats
    `points` as an aligned axis (ensemble mean/spread of the density,
    differencing, etc.). Because `eval_points` is now built inside
    `_kde` per gridcell, it is returned as a (*non-dim-dims, 'points')
    array rather than a 1D array, and will differ across gridcells but
    not across calls sharing the same `lo`/`hi` (i.e. across members).

    Parameters
    ----------
    x : xr.DataArray
        Data with dimension `dim` containing the samples to fit the
        KDE to (e.g. pooled window_day x year values for one member).
    dim : str, optional
        Dimension containing samples.
    lo, hi : xr.DataArray or float, optional
        Lower/upper bound of the evaluation range, per gridcell.
        Should NOT vary across members if members are to be
        concatenated and compared. Defaults to this call's own
        x.min(dim)/x.max(dim) if omitted -- do not rely on this
        default when calling per-member, since it reintroduces the
        member-dependent-grid problem this signature is designed to
        avoid; compute lo/hi once from the pooled ensemble and pass
        explicitly.
    npoints : int, optional
        Number of evaluation points per gridcell. Defaults to 200.
    pad_frac : float, optional
        Fractional pad added to each side of [lo, hi]. Defaults to 0.1.
    bw_method : str or float, optional
        Passed to scipy.stats.gaussian_kde ('scott', 'silverman', or a
        scalar). Defaults to scipy's default ('scott').

    Returns
    -------
    xr.Dataset
        'kde': estimated density, dims = (*x.dims_minus_dim, 'points')
        'eval_points': evaluation locations, same dims as 'kde'
            (varies by gridcell, constant across members if lo/hi were
            supplied from a pooled ensemble range).
    """
    non_dim_dims = [d for d in x.dims if d != dim]

    if lo is None:
        lo = x.min(dim=dim)
    if hi is None:
        hi = x.max(dim=dim)
    # Broadcast to plain DataArrays over the non-dim dims so apply_ufunc
    # sees them as per-gridcell scalars (no core dims of their own).
    lo = xr.broadcast(x.isel({dim: 0}, drop=True), lo)[1] if not isinstance(lo, xr.DataArray) else lo
    hi = xr.broadcast(x.isel({dim: 0}, drop=True), hi)[1] if not isinstance(hi, xr.DataArray) else hi

    points = np.arange(npoints)

    def _kde(arr, lo_i, hi_i):
        pad = pad_frac * (hi_i - lo_i)
        pts = np.linspace(lo_i - pad, hi_i + pad, npoints)
        arr = arr[~np.isnan(arr)]
        if arr.size < 2 or not np.isfinite(lo_i) or not np.isfinite(hi_i):
            return np.full(npoints, np.nan), pts
        kde = gaussian_kde(arr, bw_method=bw_method)
        return kde(pts), pts

    kde_da, eval_points_da = xr.apply_ufunc(
        _kde,
        x, lo, hi,
        input_core_dims=[[dim], [], []],
        output_core_dims=[["points"], ["points"]],
        exclude_dims={dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
        dask_gufunc_kwargs={"output_sizes": {"points": npoints}},
    )

    kde_da = kde_da.rename("kde").assign_coords(points=points)
    eval_points_da = eval_points_da.rename("eval_points").assign_coords(points=points)

    ds = xr.Dataset(
        data_vars={"kde": kde_da, "eval_points": eval_points_da},
        coords={"points": points},
        attrs={
            "method": "scipy.stats.gaussian_kde",
            "bw_factor": bw_method,
            "note": "eval_points vary by gridcell; hold constant across "
                    "members via shared lo/hi to allow valid concatenation.",
        },
    )

    return ds


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
    stream: str,
    gcomp: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    member: int | None = None,
    grid: xr.DataArray | xr.Dataset = DEFAULT_GRID["fhist"],
) -> xr.DataArray:
    """Load FHIST data and align it to the PPE grid."""
    time_slice = slice(time_start, time_stop)

    vv = "_".join(variable.split("_")[:-2])
    freq = "_".join(variable.split("_")[-2:])

    da = xclim.load_ppe(vv, gcomp, freq, stream, members=member)[vv].sel(time=time_slice)
    print(f"Loaded member {member}.")

    da = da.reindex_like(grid, method="nearest", tolerance=1e-3).where(grid.LANDFRAC > 0)

    return da


def load_lens2_variable(
    variable: str,
    stream: str,
    gcomp: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    member: str = "",
    grid: xr.DataArray | xr.Dataset = DEFAULT_GRID["lens2"],
) -> xr.DataArray:
    """Load LENS2 data and align it to the LENS2 grid."""
    time_slice = slice(time_start, time_stop)

    vv = "_".join(variable.split("_")[:-2])
    freq = "_".join(variable.split("_")[-2:])

    da = xclim.load_cesm2le(vv, gcomp, freq, stream, member=member)[vv].sel(time=time_slice)
    print(f"Loaded member {member}.")

    da = da.reindex_like(grid, method="nearest", tolerance=1e-3).where(grid.LANDFRAC > 0)

    return _strip_bad_scalar_coords(da)


def load_goga2_variable(
    variable: str,
    stream: str,
    gcomp: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    member: str = "",
    grid: xr.DataArray | xr.Dataset = DEFAULT_GRID["goga2"],
    regridded: bool = False,
) -> xr.DataArray:
    """Load GOGA2 data and align it to the GOGA2 grid."""
    time_slice = slice(time_start, time_stop)

    vv = "_".join(variable.split("_")[:-2])
    freq = "_".join(variable.split("_")[-2:])

    da = xclim.load_goga2(vv, gcomp, freq, stream, member=member, regridded=regridded)[vv].sel(time=time_slice)
    print(f"Loaded member {member}.")

    da = da.reindex_like(grid, method="nearest", tolerance=1e-3).where(grid.LANDFRAC > 0)

    return da


def load_cmip6_fhist_variable(
    variable: str,
    stream: str,
    gcomp: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    member: str = "",
    grid: xr.DataArray | xr.Dataset = DEFAULT_GRID["fhist"],
) -> xr.DataArray:
    """Load CMIP6 FHIST data and align it to the FHIST grid."""
    time_slice = slice(time_start, time_stop)

    vv = "_".join(variable.split("_")[:-2])
    freq = "_".join(variable.split("_")[-2:])

    da = xclim.load_cmip6_fhist(vv, gcomp, freq, stream, member=member)[vv].sel(time=time_slice)
    print(f"Loaded member {member}.")

    da = da.reindex_like(grid, method="nearest", tolerance=1e-3).where(grid.LANDFRAC > 0)

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
    member: str | int | None = None,
    grid: xr.DataArray | None = None,
    regridded: bool = False,
) -> xr.DataArray:
    """
    Route to the appropriate dataset loader.

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
        if not isinstance(member, int):
            raise ValueError("`member` must be an int for FHIST.")
        return load_fhist_variable(
            variable=variable,
            gcomp=gcomp,
            stream=stream,
            time_start=time_start,
            time_stop=time_stop,
            member=member,
            grid=grid,
        )
    elif dataset == "lens2":
        if not isinstance(member, str):
            raise ValueError("`member` must be a single string for LENS2.")
        return load_lens2_variable(
            variable=variable,
            gcomp=gcomp,
            stream=stream,
            time_start=time_start,
            time_stop=time_stop,
            member=member,
            grid=grid,
        )
    elif dataset == "goga2":
        if not isinstance(member, str):
            raise ValueError("`member` must be a single string for GOGA2.")
        return load_goga2_variable(
            variable=variable,
            gcomp=gcomp,
            stream=stream,
            time_start=time_start,
            time_stop=time_stop,
            member=member,
            grid=grid,
            regridded=regridded,
        )
    elif dataset == "cmip6_fhist":
        if not isinstance(member, str):
            raise ValueError("`member` must be a single string for CMIP6 FHIST.")
        return load_cmip6_fhist_variable(
            variable=variable,
            gcomp=gcomp,
            stream=stream,
            time_start=time_start,
            time_stop=time_stop,
            member=member,
            grid=grid,
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.  Choose 'fhist', 'lens2', 'goga2', or 'cmip6_fhist'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate temperature distribution statistics."
        ),
    )
    parser.add_argument(
        "--anom",
        action="store_true",
        help="Compute distribution statistics of day-of-year anomalies."
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Compute distribution statistics of raw values."
    )
    parser.add_argument(
        "--dataset",
        choices=["fhist", "lens2", "goga2", "goga2_2deg", "cmip6_fhist"],
        default="fhist",
        help=(
            "Which ensemble to load.  "
            "'fhist' uses xclim.load_fhist; "
            "'lens2' uses xclim.load_cesm2le; "
            "'goga2' and 'goga2_2deg' uses xclim.load_goga2; "
            "'cmip6_fhist' uses xclim.load_cmip6_fhist"
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
            "For use with GOGA2, LENS2, and CMIP6 FHIST."
        ),
    )
    parser.add_argument(
        "--stream",
        default="h1",
        type=str,
        help=(
            "Stream for history output. "
            "For use with GOGA2, LENS2, and CMIP6 FHIST; "
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
        "--calc-clim",
        action="store_true",
        help="Force calculation of the day-of-year climatology."
    )
    parser.add_argument(
        "--calc-doy",
        action="store_true",
        help="Force calculation of the day-of-year labels of the maximum window mean."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag to put at the end of the file name.",
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
    
    # Handle dataset name
    dataset = args.dataset
    regrid_kwarg = {}
    regrid_tag = ""
    if "_2deg" in args.dataset:
        dataset = args.dataset.replace("_2deg", "")
        regrid_kwarg = {"regridded": True}
        regrid_tag = "_2DEG"
    print(regrid_kwarg)

    # Handle paths
    varname = "_".join(args.variable.split("_")[:-2])
    clim_path = args.output / "doy_climatology"
    base_path = args.output / f"{varname}_pool_{args.window_days}d_window"
    fname = f"{dataset.upper()}{regrid_tag}_{varname}_pool_{args.window_days}d_window_{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}{args.tag}"
    
    # Make directories if they do not altready exist
    clim_path.mkdir(parents=True, exist_ok=True)
    base_path.mkdir(parents=True, exist_ok=True)

    # Get current date and time
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    DEFAULT_LOG_PATH.mkdir(parents=True, exist_ok=True)
    dask_cluster_info = "no_dask"
    if args.dask_cluster:
        dask_cluster_info = f"dask_{args.dask_workers}cores_{args.dask_memory}"
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
    this_logger.info(f"{args.dataset.upper()} {args.variable.upper()}")
    this_logger.info(f"Do anomalies: {args.anom}")
    this_logger.info(f"Do raw values: {args.raw}")
    this_logger.info(f"Calculate climatology: {args.calc_clim}")
    this_logger.info(f"Calculate doy labels: {args.calc_doy}")
    this_logger.info(f"Output path: {base_path / fname}")
    this_logger.info(f"Dask - cluster: {int(args.dask_cluster)}, workers: {args.dask_workers}, memory: {args.dask_memory}")
    this_logger.info(f"Window length (days): {args.window_days}")
    this_logger.info(f"Chunks: {DEFAULT_ANALYSIS_CHUNKS}")

    # Parse time
    delta_time = to_datetime(args.time_stop) - to_datetime(args.time_start)
    this_logger.info(f"Start time: {to_datetime(args.time_start)}")
    this_logger.info(f"Stop time:  {to_datetime(args.time_stop)}")
    this_logger.info(f"Delta time: {format_timedelta_string(delta_time)}")

    if delta_time <= timedelta(days=365):
        raise ValueError("The time period must be at least one year long.")

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
        # Load grid - only use for reindexing variables
        grid = DEFAULT_GRID[args.dataset].compute()

        # Quantiles to compute
        q_frac = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        # Load derived quantities if already computed and saved to file
        # -- Climatology --
        clim_avail = (clim_path / f"{fname}_doy_climatology.nc").exists()
        if clim_avail and not args.calc_clim:
            this_logger.info("Day-of-year climatology exists, loading now...")
            this_logger.info(clim_path / f"{fname}_doy_climatology.nc")
            y_clim = (
                xr.open_dataset(clim_path / f"{fname}_doy_climatology.nc")[varname]
                .reindex_like(grid, method="nearest", tolerance=1e-3)
            )
            this_logger.info(f"y_clim {y_clim.dims} {y_clim.shape} {y_clim.nbytes / 1024 / 1024 / 1024: 0.3f}GB")
            c.lap("Timing")
        
        # -- Day-of-year labels --
        doy_avail = (clim_path / f"{fname}_doy_window.nc").exists()
        if doy_avail and not args.calc_clim:
            this_logger.info(f"Climatological maximum {args.window_days}-day running mean exists, loading now...")
            this_logger.info(clim_path / f"{fname}_doy_window.nc")
            window_doy = (
                xr.open_dataset(clim_path / f"{fname}_doy_window.nc")["max_window_doy"]
                .reindex_like(grid, method="nearest", tolerance=1e-3)
            )
            this_logger.info(f"window_doy {window_doy.dims} {window_doy.shape} {window_doy.nbytes / 1024 / 1024 / 1024: 0.3f}GB")
            c.lap("Timing")

        # Per-member loop for all computation
        member_list = MEMBERS[dataset]
        this_logger.info(f"Dataset: {dataset.upper()}")
        this_logger.info(f"Processing {len(member_list)} members individually: {list(member_list)}")

        # Lists to store DataArrays for each member
        clim_list = []        # member x (lat, lon, dayofyear)
        doy_list = []         # member x (lat, lon, dayofyear)

        anom_stats_list = []  # member x 7(member, lat, lon)
        anom_kde_list = []    # member x 2(lat, lon, points)
        anom_lo_list = []     # member x (lat, lon)
        anom_hi_list = []     # member x (lat, lon)
        anom_cache_paths = {}

        stats_list = []       # member x 7(member, lat, lon)
        kde_list = []         # member x 2(lat, lon, points)
        lo_list = []          # member x (lat, lon)
        hi_list = []          # member x (lat, lon)
        cache_paths = {}

        # ------------------------------------------------------------------
        # Pass 1: per-member stats + cache pooled anomaly + accumulate lo/hi
        # ------------------------------------------------------------------
        this_logger.info(f"### Pass 1: per-member stats + cache pooled anomaly + accumulate lo/hi ###")
        for m in member_list:
            this_logger.info(f"--- Member {m} START ---")

            # Load dataset (still lazy / Dask-backed at this point)
            this_logger.info(f"Member {m}: loading {args.variable.upper()}...")
            y_m = load_variable(
                dataset=dataset,
                variable=args.variable,
                gcomp=args.gcomp,
                stream=args.stream,
                time_start=args.time_start,
                time_stop=args.time_stop,
                member=m,
                grid=grid,
                **regrid_kwarg,  # only used for regridded GOGA2 output
            )
            this_logger.info(f"Member {m}: y_m {y_m.dims} {y_m.shape} {y_m.nbytes / 1024 / 1024 / 1024: 0.3f}GB")
            c.lap(f"Member {m}")

            if client_cluster is not None:
                if args.anom:
                    this_logger.info(f"Member {m}: persisting {args.variable.upper()}...")
                    y_m = client_cluster[0].persist(y_m.chunk(DEFAULT_ANALYSIS_CHUNKS_NOMEMBER))
                    wait(y_m)  # block until all chunks are resident on workers
                    c.lap(f"Member {m}")

            if args.raw:
                this_logger.info(f"Member {m}: computing {args.variable.upper()}...")
                y_m = y_m.compute()
                c.lap(f"Member {m}")

            # Compute day-of-year climatology
            if not clim_avail or args.calc_clim:
                this_logger.info(f"Member {m}: computing day-of-year climatology...")
                y_m_clim = y_m.groupby("time.dayofyear").mean()
                clim_list.append(y_m_clim)
                c.lap(f"Member {m}")
            else:
                y_m_clim = y_clim.sel(member=m)
            this_logger.info(f"Member {m}: y_m_clim {y_m_clim.dims} {y_m_clim.shape} {y_m_clim.nbytes / 1024 / 1024 / 1024: 0.3f}GB")
            y_m_clim = y_m_clim.compute()
            c.lap(f"Member {m}")

            # Compute climatological maximum n-days running mean
            if not doy_avail or args.calc_doy:
                this_logger.info(f"Member {m}: computing climatological maximum {args.window_days}-day running mean...")
                window_doy_m = compute_max_window_doy(y_m_clim, args.window_days)
                doy_list.append(window_doy_m)
            else:
                this_logger.info(f"Member {m}: selecting climatological maximum {args.window_days}-day running mean...")
                window_doy_m = window_doy.sel(member=m)
            this_logger.info(f"Member {m}: window_doy_m {window_doy_m.dims} {window_doy_m.shape} {window_doy_m.nbytes / 1024 / 1024 / 1024: 0.3f}GB")
            c.lap(f"Member {m}")

            if args.anom:
                this_logger.info(f"Member {m}: ANOMALIES")

                # Materialize this member's anomaly field only
                this_logger.info(f"Member {m}: computing the anomaly field...")
                y_anom_m = (y_m.groupby("time.dayofyear") - y_m_clim).compute()
                this_logger.info(f"Member {m}: y_anom_m {y_anom_m.dims} {y_anom_m.shape} {y_anom_m.nbytes / 1024 / 1024 / 1024: 0.3f}GB")
                c.lap(f"Member {m}")

                # Windowed selection for this member
                this_logger.info(f"Member {m}: selecting days within window...")
                y_anom_window_m = select_window_multi_year(y_anom_m, window_doy_m)
                c.lap(f"Member {m}")

                # Pool over (year, window_day)
                this_logger.info(f"Member {m}: pooling anomalies over (year, window_day)...")
                y_anom_pooled_m = y_anom_window_m.stack(pool=["year", "window_day"])
                c.lap(f"Member {m}")

                # Descriptive stats for this member
                this_logger.info(f"Member {m}: computing statistics...")
                anom_stats_m = descriptive_stats(y_anom_pooled_m, "pool", q_frac)
                anom_stats_list.append(anom_stats_m)
                c.lap(f"Member {m}")

                # Per-gridcell range for this member (cheap: (lat, lon), no pool/member dims)
                this_logger.info(f"Member {m}: computing min and max for KDE...")
                anom_lo_m = y_anom_pooled_m.min(dim="pool")
                anom_hi_m = y_anom_pooled_m.max(dim="pool")
                anom_lo_list.append(anom_lo_m)
                anom_hi_list.append(anom_hi_m)
                c.lap(f"Member {m}")

                # Cache the pooled anomaly to scratch so pass 2 can reload without
                # rerunning load/clim/window/anomaly
                this_logger.info(f"Member {m}: caching pooled anomalies to scratch...")
                cache_path = CACHE_PATHS[dataset] / f"{fname}_member{m}_anom_pooled.nc"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                y_anom_pooled_m.reset_index("pool").to_netcdf(cache_path)
                anom_cache_paths[m] = cache_path
                this_logger.info(f"Member {m}: {cache_path}")
                c.lap(f"Member {m}")
            
            if args.raw:
                this_logger.info(f"Member {m}: RAW VALUES")

                # Windowed selection for this member
                this_logger.info(f"Member {m}: selecting days within window...")
                y_window_m = select_window_multi_year(y_m, window_doy_m)
                c.lap(f"Member {m}")

                # Pool over (year, window_day)
                this_logger.info(f"Member {m}: pooling anomalies over (year, window_day)...")
                y_pooled_m = y_window_m.stack(pool=["year", "window_day"]).chunk({"pool": -1})
                c.lap(f"Member {m}")

                # Descriptive stats for this member
                this_logger.info(f"Member {m}: computing statistics...")
                stats_m = descriptive_stats(y_pooled_m, "pool", q_frac)
                stats_list.append(stats_m)
                c.lap(f"Member {m}")

                # Per-gridcell range for this member (cheap: (lat, lon), no pool/member dims)
                this_logger.info(f"Member {m}: computing min and max for KDE...")
                lo_m = y_pooled_m.min(dim="pool")
                hi_m = y_pooled_m.max(dim="pool")
                lo_list.append(lo_m)
                hi_list.append(hi_m)
                c.lap(f"Member {m}")

                # Cache the pooled values to scratch so pass 2 can reload without
                # rerunning load/clim/window/anomaly
                this_logger.info(f"Member {m}: caching pooled anomalies to scratch...")
                cache_path = CACHE_PATHS[dataset] / f"{fname}_member{m}_pooled.nc"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                y_pooled_m.reset_index("pool").to_netcdf(cache_path)
                cache_paths[m] = cache_path
                this_logger.info(f"Member {m}: {cache_path}")
                c.lap(f"Member {m}")

            this_logger.info(f"--- Member {m} DONE ---")
        
        # Ensemble-pooled per-gridcell range, computed once from the small (lat, lon) per-member arrays
        this_logger.info("Computing ensemble-pooled per-gridcell range for KDE eval points...")
        if args.anom:
            anom_lo = xr.concat(anom_lo_list, dim="member").min(dim="member").compute()
            anom_hi = xr.concat(anom_hi_list, dim="member").max(dim="member").compute()
            this_logger.info(f"anom_lo {anom_lo.dims} {anom_lo.shape} {anom_lo.nbytes / 1024 / 1024 / 1024: 0.3f}GB")
            this_logger.info(f"anom_hi {anom_hi.dims} {anom_hi.shape} {anom_hi.nbytes / 1024 / 1024 / 1024: 0.3f}GB")
            c.lap("Timing")
        if args.raw:
            lo = xr.concat(lo_list, dim="member").min(dim="member").compute()
            hi = xr.concat(hi_list, dim="member").max(dim="member").compute()
            this_logger.info(f"lo {lo.dims} {lo.shape} {lo.nbytes / 1024 / 1024 / 1024: 0.3f}GB")
            this_logger.info(f"hi {hi.dims} {hi.shape} {hi.nbytes / 1024 / 1024 / 1024: 0.3f}GB")
            c.lap("Timing")

        # ----------------------------------------------------------------------------
        # Pass 2: reload cached pooled anomaly per member, KDE with shared eval_points
        # ----------------------------------------------------------------------------
        this_logger.info(f"### Pass 2: reload cached pooled anomaly per member, KDE with shared eval_points ###")
        for m in member_list:
            this_logger.info(f"--- Member {m} START ---")

            if args.anom:
                this_logger.info(f"Member {m}: ANOMALIES")

                this_logger.info(f"Member {m}: loading cached {args.variable.upper()}...")
                y_anom_pooled_m = xr.open_dataarray(anom_cache_paths[m]).compute()
                this_logger.info(f"Member {m}: y_anom_pooled_m {y_anom_pooled_m.dims} {y_anom_pooled_m.shape}")
                y_anom_pooled_m = y_anom_pooled_m.reindex_like(grid, method="nearest", tolerance=1e-3)
                this_logger.info(f"Member {m}: y_anom_pooled_m {y_anom_pooled_m.dims} {y_anom_pooled_m.shape} {y_anom_pooled_m.nbytes / 1024 / 1024 / 1024: 0.3f}GB")

                anom_kde_m = kde_pdf(y_anom_pooled_m, dim="pool", lo=anom_lo, hi=anom_hi)
                anom_kde_m.attrs["neff"] = "no adjustment for neff (=n)"
                anom_kde_list.append(anom_kde_m)
                c.lap(f"Member {m}: KDE")
            
            if args.raw:
                this_logger.info(f"Member {m}: RAW VALUES")

                this_logger.info(f"Member {m}: loading cached {args.variable.upper()}...")
                y_pooled_m = xr.open_dataarray(cache_paths[m]).compute()
                this_logger.info(f"Member {m}: y_anom_pooled_m {y_pooled_m.dims} {y_pooled_m.shape}")
                y_pooled_m = y_pooled_m.reindex_like(grid, method="nearest", tolerance=1e-3)
                this_logger.info(f"Member {m}: y_anom_pooled_m {y_pooled_m.dims} {y_pooled_m.shape} {y_pooled_m.nbytes / 1024 / 1024 / 1024: 0.3f}GB")

                kde_m = kde_pdf(y_pooled_m, dim="pool", lo=lo, hi=hi).compute()
                kde_m.attrs["neff"] = "no adjustment for neff (=n)"
                kde_list.append(kde_m)
                c.lap(f"Member {m}: KDE")

            this_logger.info(f"--- Member {m} DONE ---")

        # Concatenate per-member results back along `member` and save once.
        if args.anom:
            this_logger.info("ANOMALIES")

            this_logger.info("Redundant per-member compute of statistics...")
            anom_stats_list = [s.compute() for s in anom_stats_list]

            this_logger.info("Redundant per-member compute of KDE...")
            anom_kde_list = [k.compute() for k in anom_kde_list]

            this_logger.info("Concatenating per-member results...")
            y_anom_pooled_stats = xr.concat(anom_stats_list, dim="member").assign_coords(member=member_list)
            y_anom_pooled_kde = xr.concat(anom_kde_list, dim="member").assign_coords(member=member_list)
            c.lap("Timing")

            this_logger.info("Saving statistics...")
            with dask.config.set(scheduler="synchronous"):
                y_anom_pooled_stats.to_netcdf(base_path / f"{fname}_anom_stats.nc")
            this_logger.info(base_path / f"{fname}_anom_stats.nc")
            c.lap("Done saving statistics")

            this_logger.info("Saving Gaussian KDE...")
            with dask.config.set(scheduler="synchronous"):
                y_anom_pooled_kde.to_netcdf(base_path / f"{fname}_anom_kde.nc")
            this_logger.info(base_path / f"{fname}_anom_kde.nc")
            c.lap("Done saving Gaussian KDE")
        
        if args.raw:
            this_logger.info("RAW VALUES")

            this_logger.info("Redundant per-member compute of statistics...")
            stats_list = [s.compute() for s in stats_list]

            this_logger.info("Redundant per-member compute of KDE...")
            kde_list = [k.compute() for k in kde_list]

            this_logger.info("Concatenating per-member results...")
            y_pooled_stats = xr.concat(stats_list, dim="member").assign_coords(member=member_list)
            y_pooled_kde = xr.concat(kde_list, dim="member").assign_coords(member=member_list)
            c.lap("Timing")

            this_logger.info("Saving statistics...")
            with dask.config.set(scheduler="synchronous"):
                y_pooled_stats.to_netcdf(base_path / f"{fname}_stats.nc")
            this_logger.info(base_path / f"{fname}_stats.nc")
            c.lap("Done saving statistics")

            this_logger.info("Saving Gaussian KDE...")
            with dask.config.set(scheduler="synchronous"):
                y_pooled_kde.to_netcdf(base_path / f"{fname}_kde.nc")
            this_logger.info(base_path / f"{fname}_kde.nc")
            c.lap("Done saving Gaussian KDE")

        # Concatenate per-member climatology and day-of-year labels along `member` and save once.
        if not clim_avail:
            this_logger.info("Concatenating per-member climatology...")
            y_clim = xr.concat(clim_list, dim="member").assign_coords(member=member_list)
            y_clim.to_netcdf(clim_path / f"{fname}_doy_climatology.nc")
            this_logger.info(clim_path / f"{fname}_doy_climatology.nc")
            c.lap("Done saving climatology")

        if not doy_avail:
            this_logger.info("Concatenating per-member day-of-year labels...")
            window_doy = xr.concat(doy_list, dim="member").assign_coords(member=member_list)
            window_doy.to_netcdf(clim_path / f"{fname}_doy_window.nc")
            this_logger.info(clim_path / f"{fname}_doy_window.nc")
            c.lap("Done saving day-of-year labels")

        # TODO: compute the effective degrees of freedom for a single time series -
        # the following computes the lag-1 autocorrelation used in significance tests
        # of the correlation between two times series. If reinstated, run this
        # inside the per-member loop as well (on y_window_m) for the same
        # memory-bounding reasons.
        # y_window_rho_lag1 = lag1_autocorr_pooled(y_window_m, "year", "window_day")

    finally:
        if client_cluster is not None:
            xclim.close_dask_cluster(client_cluster, remove_std_files=False)
        this_logger.info(f"=== RUN END: {datetime.now().isoformat()} ===")


if __name__ == "__main__":
    main()
