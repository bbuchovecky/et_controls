"""Compute JJA daily temperature distribution statistics for PPE members.

- load daily temperature data
- compute daily anomalies by day-of-year climatology
- subset to specified months (default to June/July/August)
- fit a Gaussian to the distribution for each member and grid cell
- compute skewness and kurtosis along the time dimension
- save the results to NetCDF

Additionally, this script computes the number of heat days (NHD)
exceeding climatological percentile thresholds.  Thresholds are derived
per ensemble member from a centerd DOY-window climatology following the
ETCCDI TX90p convention:

    T_p(member, d) = quantile_p { T(member, t) : DOY(t) ∈ [d-hw, d+hw] }

where hw = window // 2 and the window pools values across all years in the
analysis period.  NHD_p = ∑_{t ∈ JJA} 1[T(t) > T_p(DOY(t))].
"""

from __future__ import annotations

import argparse
from pathlib import Path
from distributed import wait

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew

import xclimate as xclim


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARAM_NAMES = ["amplitude", "mean", "sigma", "offset"]
DEFAULT_VARIABLE = "TREFMXAV_day_1"
DEFAULT_TIME_START = "1995-01"
DEFAULT_TIME_STOP = "2014-12"
DEFAULT_MONTHS: tuple[int, ...] = (6, 7, 8)
DEFAULT_BINS = 50
DEFAULT_PERCENTILES: tuple[int, ...] = (90, 95, 99)
DEFAULT_WINDOW_DAYS: int = 5


# ---------------------------------------------------------------------------
# Gaussian model
# ---------------------------------------------------------------------------

def gaussian(x, amplitude, mean, sigma, offset):
    """Shifted Gaussian PDF with a constant offset.

    Parameters
    ----------
    x : array-like
        Independent variable (bin centeres, K).
    amplitude : float
        Peak height above `offset`.
    mean : float
        Distribution centere (K).
    sigma : float
        Standard deviation (K).
    offset : float
        Vertical offset (density units).
    """
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset


# ---------------------------------------------------------------------------
# Moment statistics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Gaussian histogram fit
# ---------------------------------------------------------------------------

def fit_gaussian_histogram_along_dim(
    da: xr.DataArray,
    time_dim: str = "time",
    bins: int | np.ndarray = DEFAULT_BINS,
    density: bool = True,
    p0: tuple[float, float, float, float] | None = None,
    bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    maxfev: int = 10000,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Fit a Gaussian to the histogram of each (non-time) slice.

    Parameters
    ----------
    da : xr.DataArray
        Input array with `time_dim` as one of its dimensions.
    time_dim : str
        Name of the dimension over which the histogram is computed.
    bins : int or np.ndarray
        Number of bins, or explicit bin-edge array (K).
    density : bool
        Passed to np.histogram; if True the histogram integrates to 1.
    p0 : tuple or None
        Initial guess for (amplitude, mean [K], sigma [K], offset).
        Auto-estimated from data if None.
    bounds : tuple or None
        Lower and upper bounds for curve_fit; auto-set if None.
    maxfev : int
        Maximum function evaluations for curve_fit.

    Returns
    -------
    bin_centers : xr.DataArray  [K]
    histogram   : xr.DataArray  [density units]
    gaussian_fit: xr.DataArray  [density units]
    params      : xr.DataArray  [(amplitude, mean, sigma, offset) × grid]
    """

    if isinstance(bins, (int, np.integer)):
        pooled = np.asarray(da.values, dtype=float).ravel()
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size < 4:
            raise ValueError("Need at least four finite values to compute global bin edges")
        data_min = float(np.nanmin(pooled))
        data_max = float(np.nanmax(pooled))
        if not np.isfinite(data_min) or not np.isfinite(data_max):
            raise ValueError("Could not determine finite data bounds for histogram bins")
        if data_min == data_max:
            data_min -= 0.5
            data_max += 0.5
        edges = np.linspace(data_min, data_max, int(bins) + 1)
    else:
        edges = np.asarray(bins, dtype=float)

    nbin = edges.size - 1
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _fit_all(v1d):
        vals = np.asarray(v1d, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size < 4:
            return (
                np.array([np.nan, np.nan, np.nan, np.nan], dtype=float),
                np.full(nbin, np.nan, dtype=float),
                np.full(nbin, np.nan, dtype=float),
                np.full(nbin, np.nan, dtype=float),
            )

        hist = np.histogram(vals, bins=edges, density=density)[0]

        if p0 is None:
            offset0 = float(np.nanmin(hist))
            amplitude0 = float(np.nanmax(hist) - offset0)
            if not np.isfinite(amplitude0) or amplitude0 == 0:
                amplitude0 = 1.0
            mean0 = float(np.nanmean(vals))
            sigma0 = float(np.nanstd(vals))
            if not np.isfinite(sigma0) or sigma0 == 0:
                sigma0 = max(float(np.ptp(vals)) / 6.0, 1.0)
            p0_local = (amplitude0, mean0, sigma0, offset0)
        else:
            p0_local = p0

        if bounds is None:
            lower = (-np.inf, np.min(vals), 0.0, -np.inf)
            upper = (np.inf, np.max(vals), np.inf, np.inf)
            bounds_local = (lower, upper)
        else:
            bounds_local = bounds

        try:
            params, _ = curve_fit(
                gaussian,
                centers,
                hist,
                p0=p0_local,
                bounds=bounds_local,
                maxfev=maxfev,
            )
            fit = gaussian(centers, *params)
        except Exception:
            params = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
            fit = np.full(nbin, np.nan, dtype=float)

        return params.astype(float), hist.astype(float), fit.astype(float), centers.astype(float)

    params_da, hist_da, fit_da, centers_da = xr.apply_ufunc(
        _fit_all,
        da,
        input_core_dims=[[time_dim]],
        output_core_dims=[["param"], ["bin"], ["bin"], ["bin"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float, float],
        dask_gufunc_kwargs={"output_sizes": {"param": 4, "bin": nbin}},
    )

    params_da = params_da.assign_coords(param=("param", PARAM_NAMES))
    params_da.name = "gaussian_params"

    bin_centers_da = xr.DataArray(centers, dims=("bin",), name="bin_center")
    hist_da = hist_da.assign_coords(bin=bin_centers_da)
    hist_da.name = "histogram"
    fit_da = fit_da.assign_coords(bin=bin_centers_da)
    fit_da.name = "gaussian_fit"

    return bin_centers_da, hist_da, fit_da, params_da


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


def load_fhist_ppe_temperature(
    variable: str = DEFAULT_VARIABLE,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    members: list[int] | None = None,
) -> xr.DataArray:
    """Load FHIST temperature data and align it to the PPE grid."""

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


def load_lens2_temperature(
    variable: str = DEFAULT_VARIABLE,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    members: list[int] | None = None,
) -> xr.DataArray:
    """Load LENS2 temperature data and align it to the LENS2 grid."""

    time_slice = slice(time_start, time_stop)
    grid = xclim.load_cesm2le_grid()

    vv = "_".join(variable.split("_")[:-2])
    freq = "_".join(variable.split("_")[-2:])
    da = (
        xclim.load_cesm2le(vv, "lnd", freq, "h6")[vv]
        .sel(time=time_slice)
        .reindex_like(grid, method="nearest", tolerance=1e-3)
        .where(grid.LANDFRAC > 0)
    )

    if members is not None:
        da = da.sel(member=members)

    return _strip_bad_scalar_coords(da)


def load_goga2_temperature(
    variable: str = DEFAULT_VARIABLE,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    members: list[int] | None = None,
) -> xr.DataArray:
    """Load GOGA2 temperature data and align it to the GOGA2 grid."""

    time_slice = slice(time_start, time_stop)
    grid = xclim.load_goga2_grid()

    vv = "_".join(variable.split("_")[:-2])
    freq = "_".join(variable.split("_")[-2:])
    da = (
        xclim.load_goga2(vv, "lnd", freq, "h1")[vv]
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

def load_temperature_data(
    dataset: str,
    variable: str = DEFAULT_VARIABLE,
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
        return load_fhist_ppe_temperature(
            variable=variable,
            time_start=time_start,
            time_stop=time_stop,
            members=members,
        )
    elif dataset == "lens2":
        return load_lens2_temperature(
            variable=variable,
            time_start=time_start,
            time_stop=time_stop,
            members=members,
        )
    elif dataset == "goga2":
        return load_goga2_temperature(
            variable=variable,
            time_start=time_start,
            time_stop=time_stop,
            members=members,
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.  Choose 'fhist', 'lens2', or 'goga2'.")


# ---------------------------------------------------------------------------
# Month anomalies
# ---------------------------------------------------------------------------

def compute_anomalies(
        da: xr.DataArray,
        months: tuple[int, ...] | list[int] = DEFAULT_MONTHS,
    ) -> xr.DataArray:
    """Climatological anomaly by day-of-year, then month subset."""

    anomalies = da.groupby("time.dayofyear") - da.groupby("time.dayofyear").mean()
    return anomalies.sel(time=anomalies.time.dt.month.isin(months))


# ---------------------------------------------------------------------------
# Percentile-based heat day counts
# ---------------------------------------------------------------------------

# def compute_percentile_thresholds(
#     da: xr.DataArray,
#     window: int = DEFAULT_WINDOW_DAYS,
#     percentiles: tuple[int, ...] | list[int] = DEFAULT_PERCENTILES,
# ) -> xr.DataArray:
#     """Compute climatological per-DOY percentile thresholds with a centered window.

#     For each calendar day-of-year *d*, all values whose DOY falls within the
#     window [d - hw, …, d + hw] (hw = window // 2), pooled across all years
#     and within each member independently, are used to estimate the requested
#     percentile levels.  Year-boundary wrap-around is applied (DOY 1 wraps to
#     DOY 366 and vice versa).

#     The pooled sample size for a 20-year base period and a 5-day window is
#     ~100 values per member per grid cell, yielding stable quantile estimates
#     at the 90th and 95th percentile; the 99th percentile is noisier (~1 value
#     per year in the tail) and should be interpreted with appropriate caution.

#     Parameters
#     ----------
#     da : xr.DataArray
#         Raw (non-anomaly) temperature; dims ``(member, time, lat, lon)``.
#     window : int
#         Width of the centered DOY window in days.  Must be odd.  Default 5.
#     percentiles : sequence of int
#         Percentile levels in [0, 100].  Default ``(90, 95, 99)``.

#     Returns
#     -------
#     xr.DataArray
#         Dims ``(quantile, member, dayofyear, lat, lon)``.
#         Coordinate ``quantile`` carries values in [0, 1]
#         (e.g. 0.90, 0.95, 0.99) to match xarray's DataArray.quantile
#         convention.  Units are the same as ``da``.
#     """
#     if window % 2 == 0:
#         raise ValueError(f"window must be odd; got {window}")
#     half_w = window // 2
#     max_doy = 366  # accommodate leap-year DOY 366
#     q_vals = [p / 100.0 for p in percentiles]
#     all_doys = np.arange(1, max_doy + 1)

#     threshold_slices: list[xr.DataArray] = []
#     for doy in all_doys:
#         # Centered window with year-boundary wrap-around (1-indexed DOYs, mod 366)
#         win_doys = [
#             ((doy - 1 + offset) % max_doy) + 1
#             for offset in range(-half_w, half_w + 1)
#         ]
#         time_mask = da.time.dt.dayofyear.isin(win_doys)
#         subset = da.sel(time=time_mask)
#         # .quantile returns dims (quantile, member, lat, lon)
#         pct = subset.quantile(q=q_vals, dim="time")
#         threshold_slices.append(pct)

#     # After concat along dayofyear: (dayofyear, quantile, member, lat, lon)
#     doy_coord = xr.DataArray(
#         all_doys,
#         dims="dayofyear",
#         name="dayofyear",
#         attrs={"long_name": "calendar day of year"},
#     )
#     thresholds = xr.concat(threshold_slices, dim=doy_coord)
#     thresholds = thresholds.transpose("quantile", "member", "dayofyear", "lat", "lon")
#     thresholds.name = "percentile_threshold"
#     thresholds.attrs = {
#         "long_name": "Climatological percentile threshold",
#         "units": da.attrs.get("units", "K"),
#         "window_days": window,
#         "description": (
#             f"Per-DOY, per-member temperature percentile thresholds computed from "
#             f"a {window}-day centered DOY window pooled across all years in the "
#             f"analysis period.  Each member's threshold is independent."
#         ),
#     }
#     return thresholds


def compute_percentile_thresholds(
    da: xr.DataArray,
    window: int = DEFAULT_WINDOW_DAYS,
    percentiles: tuple[int, ...] | list[int] = DEFAULT_PERCENTILES,
) -> xr.DataArray:
    """Compute climatological per-DOY percentile thresholds with a centered window.

    Uses apply_ufunc to compute all 366 DOY thresholds in a single numpy pass
    per chunk, avoiding the task-graph explosion produced by concatenating 366
    separate lazy .quantile() results.

    Parameters
    ----------
    da : xr.DataArray
        Raw (non-anomaly) temperature; dims ``(member, time, lat, lon)``.
    window : int
        Width of the centered DOY window in days. Must be odd.
    percentiles : sequence of int
        Percentile levels in [0, 100].

    Returns
    -------
    xr.DataArray
        Dims ``(quantile, member, dayofyear, lat, lon)``.
        Coordinate ``quantile`` carries values in [0, 1].
    """
    if window % 2 == 0:
        raise ValueError(f"window must be odd; got {window}")

    half_w = window // 2
    max_doy = 366
    q_frac = [p / 100.0 for p in percentiles]
    q_pct = list(percentiles)           # as integers for np.nanpercentile
    nq = len(percentiles)

    # Eagerly compute DOY values once; small coordinate array, never Dask-backed.
    doy_values = da.time.dt.dayofyear.values  # shape (ntime,), dtype int

    def _windowed_quantiles(arr: np.ndarray) -> np.ndarray:
        """Compute windowed DOY quantiles for one chunk.

        Parameters
        ----------
        arr : ndarray, shape ``(..., ntime)``
            Temperature values with time as the last (core) axis.
            Leading axes correspond to non-core dims in the chunk
            (typically member=1, lat_chunk, lon_chunk).

        Returns
        -------
        ndarray, shape ``(..., nq, max_doy)``
        """
        *rest, ntime = arr.shape
        result = np.full(rest + [nq, max_doy], np.nan, dtype=np.float64)

        for d in range(1, max_doy + 1):
            win_doys = [((d - 1 + k) % max_doy) + 1 for k in range(-half_w, half_w + 1)]
            mask = np.isin(doy_values, win_doys)   # shape (ntime,)
            subset = arr[..., mask]                # shape (*rest, n_in_window)
            if subset.shape[-1] == 0:
                continue
            # np.nanpercentile returns shape (nq, *rest); move quantile axis last.
            pcts = np.nanpercentile(subset, q_pct, axis=-1)  # (nq, *rest)
            result[..., d - 1] = np.moveaxis(pcts, 0, -1)   # (*rest, nq) → [..., d-1]

        return result  # shape (*rest, nq, max_doy)

    # apply_ufunc dispatches one numpy call per (member, lat-chunk, lon-chunk) chunk.
    # Task graph size: n_member_chunks × n_lat_chunks × n_lon_chunks — not 366×.
    out = xr.apply_ufunc(
        _windowed_quantiles,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["quantile", "dayofyear"]],
        vectorize=False,           # function handles full chunk arrays directly
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"quantile": nq, "dayofyear": max_doy}},
    )

    out = out.assign_coords(
        quantile=("quantile", np.array(q_frac)),
        dayofyear=("dayofyear", np.arange(1, max_doy + 1, dtype=int)),
    )
    out = out.transpose("quantile", "member", "dayofyear", "lat", "lon")
    out.name = "percentile_threshold"
    out.attrs = {
        "long_name": "Climatological percentile threshold",
        "units": da.attrs.get("units", "K"),
        "window_days": window,
        "description": (
            f"Per-DOY, per-member temperature percentile thresholds from a "
            f"{window}-day centered DOY window pooled across all years. "
            f"Each member threshold is independent."
        ),
    }
    return out


def subset_hottest_month(
        da: xr.DataArray,
) -> xr.DataArray:
    """Compute the climatological hottest month.
    
    Parameters
    ----------
    da : xr.DataArray
        Raw (non-anomaly) temperature
    
    Returns
    -------
    xr.DataArray
        Climatological hottest month.
    """
    clim = da.groupby('time.month').mean()
    max_month = clim.where(~np.isnan(clim), other=0).argmax(dim='month') + 1
    max_month = max_month.where(~np.isnan(clim.isel(month=0)), other=-1)
    max_month.attrs = {
        "long_name": f"month with climatological max {da.name}, nans set to -1",
        "units": "month",
    }
    return max_month



def count_heat_days(
    da: xr.DataArray,
    thresholds: xr.DataArray,
    months: tuple[int, ...] | list[int] = DEFAULT_MONTHS,
) -> xr.DataArray:
    """Count days per member strictly exceeding each per-DOY percentile threshold.

    The exceedance criterion is strict (T > T_p), consistent with the ETCCDI
    TX90p definition.

    Parameters
    ----------
    da : xr.DataArray
        Raw (non-anomaly) temperature; dims ``(member, time, lat, lon)``.
    thresholds : xr.DataArray
        Output of :func:`compute_percentile_thresholds`;
        dims ``(quantile, member, dayofyear, lat, lon)``.

    Returns
    -------
    xr.DataArray
        NHD — annual number of heat days — with dims
        ``(member, quantile, year, lat, lon)`` and integer-valued counts.
        The ``quantile`` coordinate values are in [0, 1].
    """
    if months is not None:
        # Restrict to specified months on the raw temperature array
        subset = da.sel(time=da.time.dt.month.isin(months))

    # Map each timestep to a 0-based integer position in the dayofyear axis.
    # all_doys runs 1…366, so position = DOY - 1.  Using isel rather than sel
    # avoids potential vectorised-indexing issues with Dask-backed coordinates.
    doy_indices = subset.time.dt.dayofyear - 1  # DataArray with dim 'time', values in [0, 365]

    # Vectorised selection: replaces 'dayofyear' dim with 'time' dim from doy_indices.
    # Result dims: (quantile, member, time, lat, lon)
    thresh_subset = thresholds.isel(dayofyear=doy_indices)

    # Broadcast subset (member, time, lat, lon) against (quantile, member, time, lat, lon)
    exceed = (subset > thresh_subset).astype("int64")

    # Annual count
    nhd = exceed.groupby("time.year").sum(dim="time", dtype=np.int64)

    print("subset dtype:       ", subset.dtype)
    print("thresholds dtype:   ", thresholds.dtype)
    print("thresh_subset dtype:", thresh_subset.dtype)
    print("exceed dtype:       ", exceed.dtype)
    print("nhd dtype:          ", nhd.dtype)

    nhd = nhd.transpose("member", "quantile", "year", "lat", "lon")
    nhd.name = "nhd"
    nhd.attrs = {
        "long_name": f"Number of {months}-month heat days exceeding climatological percentile threshold",
        "units": "count",
        "description": (
            "Annual count of days where raw daily temperature strictly exceeds "
            "the per-DOY percentile threshold (T > T_p).  Thresholds are "
            "computed per ensemble member from a centered DOY-window climatology "
            "pooled across all years in the analysis period."
        ),
    }
    return nhd


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

# def build_output_dataset(
#     da: xr.DataArray,
#     bins: int | np.ndarray = DEFAULT_BINS,
#     percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
#     window: int = DEFAULT_WINDOW_DAYS,
#     months: tuple[int, ...] = DEFAULT_MONTHS,
# ) -> xr.Dataset:
#     """Compute all requested statistics and return them as a Dataset.

#     Parameters
#     ----------
#     da : xr.DataArray
#         Raw (non-anomaly) temperature; dims ``(member, time, lat, lon)``.
#     bins : int or np.ndarray
#         Histogram bin count or explicit edges; passed to
#         :func:`fit_gaussian_histogram_along_dim`.
#     percentiles : tuple of int
#         Percentile levels for NHD thresholds; passed to
#         :func:`compute_percentile_thresholds`.
#     window : int
#         DOY window width for threshold estimation; must be odd.
#     """
#     anom_chunks = {
#         "time": -1,
#         "member": 1,
#         "lat": 16,
#         "lon": 48,
#     }

#     # --- Anomaly-based statistics (skewness, kurtosis, Gaussian fit) --------
#     anom = compute_anomalies(da, months).chunk(anom_chunks)

#     skew_da = skewness_along_dim(anom, dim="time").rename("skewness")
#     kurt_da = kurtosis_along_dim(anom, dim="time").rename("kurtosis")

#     anom_min = float(anom.min().compute())
#     anom_max = float(anom.max().compute())
#     bin_edges = np.linspace(anom_min, anom_max, bins + 1)
#     bin_centers, hist_da, fit_da, params_da = fit_gaussian_histogram_along_dim(
#         anom,
#         time_dim="time",
#         bins=bin_edges,
#         density=True,
#     )

#     bin_edges_da = xr.DataArray(bin_edges, dims=("bin_edge",), name="bin_edges")

#     # --- Heat day counts (raw temperature; per-member DOY-window thresholds) -
#     # time=-1 keeps the full time axis unsplit, which is required for quantile.
#     da_chunked = da.chunk(anom_chunks)
#     thresholds = compute_percentile_thresholds(
#         da_chunked, window=window, percentiles=percentiles
#     )
#     nhd = count_heat_days(da_chunked, thresholds, months)

#     return xr.Dataset(
#         data_vars={
#             "skewness": skew_da,
#             "kurtosis": kurt_da,
#             "histogram": hist_da,
#             "gaussian_fit": fit_da,
#             "gaussian_params": params_da,
#             "nhd": nhd,
#         },
#         coords={
#             "bin_center": bin_centers,
#             "bin_edges": bin_edges_da,
#         },
#         attrs={
#             "description": f"{months} months daily temperature distribution statistics for FHIST PPE",
#             "temperature_method": f"Daily anomalies by day-of-year climatology, then {months} months subset",
#             "nhd_method": (
#                 f"Raw temperature; per-member {window}-day centerd DOY-window "
#                 f"percentile thresholds; strict exceedance (T > T_p); {months} months only."
#             ),
#         },
#     )


def default_analysis_chunks() -> dict[str, int]:
    """Chunking used for time-core statistics.

    time=-1 keeps the full time axis in each chunk, which is required for
    scipy moment functions, histogram fitting, and quantile operations.
    """
    return {
        "time": -1,
        "member": 1,
        "lat": 48,
        "lon": 96,
    }


def build_moment_dataset(
    da: xr.DataArray,
    months: tuple[int, ...] = DEFAULT_MONTHS,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """Compute anomaly-based skewness and kurtosis.

    Parameters
    ----------
    da : xr.DataArray
        Raw daily temperature with dims ``(member, time, lat, lon)``.
    months : tuple of int
        Months to retain after removing the day-of-year climatology.
    chunks : dict or None
        Dask chunks. If None, uses :func:`default_analysis_chunks`.

    Returns
    -------
    xr.Dataset
        Dataset containing ``skewness`` and ``kurtosis``.
    """
    if chunks is None:
        chunks = default_analysis_chunks()

    anom = compute_anomalies(da, months).chunk(chunks)

    skew_da = skewness_along_dim(anom, dim="time").rename("skewness")
    kurt_da = kurtosis_along_dim(anom, dim="time").rename("kurtosis")

    ds = xr.Dataset(
        data_vars={
            "skewness": skew_da,
            "kurtosis": kurt_da,
        },
        attrs={
            "description": f"{months} months daily temperature anomaly moment statistics",
            "temperature_method": (
                f"Daily anomalies by day-of-year climatology, then "
                f"{months} months subset"
            ),
        },
    )

    return ds


def build_gaussian_dataset(
    da: xr.DataArray,
    bins: int | np.ndarray = DEFAULT_BINS,
    months: tuple[int, ...] = DEFAULT_MONTHS,
    chunks: dict[str, int] | None = None,
    save_gaussian_fit: bool = False,
) -> xr.Dataset:
    """Compute anomaly histograms and Gaussian histogram-fit parameters.

    Parameters
    ----------
    da : xr.DataArray
        Raw daily temperature with dims ``(member, time, lat, lon)``.
    bins : int or np.ndarray
        Histogram bin count or explicit bin edges.
    months : tuple of int
        Months to retain after removing the day-of-year climatology.
    chunks : dict or None
        Dask chunks. If None, uses :func:`default_analysis_chunks`.
    save_gaussian_fit : bool
        If True, save the fitted Gaussian curve at each bin center.
        If False, save only the histogram and fitted parameters.

    Returns
    -------
    xr.Dataset
        Dataset containing ``histogram`` and ``gaussian_params``.
        Optionally contains ``gaussian_fit``.
    """
    if chunks is None:
        chunks = default_analysis_chunks()

    anom = compute_anomalies(da, months).chunk(chunks)

    if isinstance(bins, (int, np.integer)):
        # Compute global bin edges once. This avoids the dangerous path inside
        # fit_gaussian_histogram_along_dim that would call da.values.
        anom_min = float(anom.min().compute())
        anom_max = float(anom.max().compute())

        if anom_min == anom_max:
            anom_min -= 0.5
            anom_max += 0.5

        bin_edges = np.linspace(anom_min, anom_max, int(bins) + 1)
    else:
        bin_edges = np.asarray(bins, dtype=float)

    bin_centers, hist_da, fit_da, params_da = fit_gaussian_histogram_along_dim(
        anom,
        time_dim="time",
        bins=bin_edges,
        density=True,
    )

    data_vars = {
        "histogram": hist_da,
        "gaussian_params": params_da,
    }

    if save_gaussian_fit:
        data_vars["gaussian_fit"] = fit_da

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "bin_center": bin_centers,
            "bin_edges": xr.DataArray(
                bin_edges,
                dims=("bin_edge",),
                name="bin_edges",
            ),
        },
        attrs={
            "description": f"{months} months daily temperature anomaly Gaussian diagnostics",
            "temperature_method": (
                f"Daily anomalies by day-of-year climatology, then "
                f"{months} months subset"
            ),
            "histogram_density": int(True),
            "save_gaussian_fit": int(save_gaussian_fit),
        },
    )

    return ds


def build_threshold_dataset(
        da: xr.DataArray,
        percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
        window: int = DEFAULT_WINDOW_DAYS,
        months: tuple[int, ...] = DEFAULT_MONTHS,
        chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """Compute percentile thresholds for heat-day count.

    Parameters
    ----------
    da : xr.DataArray
        Raw daily temperature with dims ``(member, time, lat, lon)``.
    percentiles : tuple of int
        Percentile levels, e.g. ``(90, 95, 99)``.
    window : int
        Centered DOY-window width in days. Must be odd.
    months : tuple of int
        Months over which heat days are counted.
    chunks : dict or None
        Dask chunks. If None, uses :func:`default_analysis_chunks`.

    Returns
    -------
    xr.Dataset
        Dataset containing ``thresholds``.
    """
    if chunks is None:
        chunks = default_analysis_chunks()

    da_chunked = da.chunk(chunks)

    thresholds = compute_percentile_thresholds(
        da_chunked,
        window=window,
        percentiles=percentiles,
    )

    ds = xr.Dataset(
        data_vars={
            "thresholds": thresholds,
        },
        attrs={
            "description": f"{months} months heat-day thresholds",
            "threshold_method": (
                f"Per-member {window}-day centered DOY-window "
                f"percentile thresholds; {months} months only."
            ),
            "percentiles": percentiles,
            "window_days": window,
        },
    )

    return ds



def build_nhd_dataset(
    da: xr.DataArray,
    thresholds: xr.DataArray,
    window: int = DEFAULT_WINDOW_DAYS,
    months: tuple[int, ...] = DEFAULT_MONTHS,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """Compute percentile-threshold heat-day counts.

    Parameters
    ----------
    da : xr.DataArray
        Raw daily temperature with dims ``(member, time, lat, lon)``.
    thresholds: xr.DataArray
        Thresholds for heat-day count.
    months : tuple of int
        Months over which heat days are counted.
    chunks : dict or None
        Dask chunks. If None, uses :func:`default_analysis_chunks`.

    Returns
    -------
    xr.Dataset
        Dataset containing ``nhd``.
    """
    if chunks is None:
        chunks = default_analysis_chunks()

    da_chunked = da.chunk(chunks)

    nhd = count_heat_days(
        da_chunked,
        thresholds,
        months=months,
    )

    ds = xr.Dataset(
        data_vars={
            "nhd": nhd,
        },
        attrs={
            "description": f"{months} months heat-day counts",
            "nhd_method": (
                f"Raw temperature; per-member {window}-day centered DOY-window "
                f"percentile thresholds; strict exceedance (T > T_p); "
                f"{months} months only."
            ),
            "percentiles": thresholds["quantile"].values,
            "window_days": window,
        },
    )

    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute skewness, kurtosis, and Gaussian histogram fits for "
            "daily temperature from either FHIST PPE, LENS2, or GOGA2."
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
            f"Default: fhist"
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
        "--moments",
        action="store_true",
        help="Calculate skewness and kurtosis from daily anomaly moment statistics.",
    )
    parser.add_argument(
        "--gaussian-fit",
        action="store_true",
        help="Fit Gaussian to daily anomalies.",
    )
    parser.add_argument(
        "--nhd",
        action="store_true",
        help="Calculate the number of days above temperature percentiles.",
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
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help=(
            "Number of histogram bins. "
            f"Default: {DEFAULT_BINS}"
        ),
    )
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=list(DEFAULT_MONTHS),
        metavar="M",
        help=(
            "Integer months for NHD computation.  "
            "Multiple values may be supplied (space-separated).  "
            f"Default: {list(DEFAULT_MONTHS)}"
        ),
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=int,
        default=list(DEFAULT_PERCENTILES),
        metavar="P",
        help=(
            "Integer percentile levels [0-100] for NHD computation.  "
            "Multiple values may be supplied (space-separated).  "
            f"Default: {list(DEFAULT_PERCENTILES)}"
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        metavar="W",
        help=(
            "Width (odd integer, days) of the centerd DOY window used to "
            "estimate percentile thresholds.  "
            f"Default: {DEFAULT_WINDOW_DAYS}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("daily_temp_stats.nc"),
        help="Output NetCDF path",
    )
    parser.add_argument(
        "--dask-cluster",
        action="store_true",
        help="Spin up a Dask cluster via xclim.create_dask_cluster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.window % 2 == 0:
        raise ValueError(f"--window must be odd; got {args.window}")

    client_cluster = None
    if args.dask_cluster:
        client_cluster = xclim.create_dask_cluster(
            account='UWAS0155',
            nworkers=4,
            ncores=4,
            nmem='64GB',
            walltime='02:00:00',
            # queue='casper',
        )
        client_cluster[0].wait_for_workers(2)

    try:
        # stats = build_output_dataset(
        #     da,
        #     bins=args.bins,
        #     percentiles=tuple(args.percentiles),
        #     window=args.window,
        # )
        # stats.attrs["dataset"] = args.dataset

        # args.output.parent.mkdir(parents=True, exist_ok=True)
        # stats.to_netcdf(args.output)
        # print(f"Wrote {args.output}")

        import gc

        da = load_temperature_data(
            dataset=args.dataset,
            variable=args.variable,
            time_start=args.time_start,
            time_stop=args.time_stop,
            members=args.members,
        )

        chunks = default_analysis_chunks()

        # Materialize da on workers; replaces complex file-read graph
        # with direct references to in-memory worker data.
        if client_cluster is not None:
            da = client_cluster[0].persist(da.chunk(chunks))
            wait(da)  # block until all chunks are resident on workers

        output_base = args.output
        output_base.parent.mkdir(parents=True, exist_ok=True)

        moments_path = output_base.with_name(output_base.stem + "_moments.nc")
        gaussian_path = output_base.with_name(output_base.stem + "_gaussian.nc")
        thresholds_path = output_base.with_name(output_base.stem + "_nhd_thresholds.nc")
        nhd_path = output_base.with_name(output_base.stem + "_nhd.nc")

        if args.moments:
            # ------------------------------------------------------------------
            # 1. Moment statistics
            # ------------------------------------------------------------------
            print("Computing moments...")
            moments = build_moment_dataset(
                da,
                months=tuple(args.months),
                chunks=chunks,
            )
            moments.attrs["dataset"] = args.dataset
            moments.to_netcdf(moments_path)
            print(f"Wrote {moments_path}")

            del moments
            gc.collect()
            if client_cluster is not None:
                client_cluster[0].run(gc.collect)

        if args.gaussian_fit:
            # ------------------------------------------------------------------
            # 2. Gaussian histogram diagnostics
            # ------------------------------------------------------------------
            print("Computing Gaussian fit...")
            gaussian_ds = build_gaussian_dataset(
                da,
                bins=args.bins,
                months=tuple(args.months),
                chunks=chunks,
                save_gaussian_fit=False,
            )
            gaussian_ds.attrs["dataset"] = args.dataset
            gaussian_ds.to_netcdf(gaussian_path)
            print(f"Wrote {gaussian_path}")

            del gaussian_ds
            gc.collect()
            if client_cluster is not None:
                client_cluster[0].run(gc.collect)

        if args.nhd:
            # ------------------------------------------------------------------
            # 3. Heat-day counts
            # ------------------------------------------------------------------
            print("Computing NHD thresholds...")
            thresholds_ds = build_threshold_dataset(
                da,
                percentiles=tuple(args.percentiles),
                window=args.window,
                months=tuple(args.months),
                chunks=chunks,
            )
            thresholds_ds.attrs["dataset"] = args.dataset
            thresholds_ds.to_netcdf(thresholds_path)
            print(f"Wrote {thresholds_path}")

            print("Computing NHD...")
            nhd_ds = build_nhd_dataset(
                da,
                thresholds=thresholds_ds["thresholds"],
                window=args.window,
                months=tuple(args.months),
                chunks=chunks,
            )
            nhd_ds.attrs["dataset"] = args.dataset
            nhd_ds.to_netcdf(nhd_path)
            print(f"Wrote {nhd_path}")

            del thresholds_ds
            del nhd_ds
            gc.collect()
            if client_cluster is not None:
                client_cluster[0].run(gc.collect)
    
    finally:
        if client_cluster is not None:
            xclim.close_dask_cluster(client_cluster)
    
if __name__ == "__main__":
    main()
