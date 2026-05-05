#!/glade/work/bbuchovecky/miniforge3/envs/data-sci-py312/bin/python3.12
"""
ppe_2d_binning.py
=================
Bin any PPE variable into an arbitrary 2-D (y_var by x_var) space and
return per-bin means, retaining the ensemble member dimension.

Inputs
------
target  : xr.DataArray (time, lat, lon, member) - variable to bin
y_var   : xr.DataArray (time, lat, lon, member) - y-axis binning variable
x_var   : xr.DataArray (time, lat, lon, member) - x-axis binning variable

Output
------
bin_means : xr.DataArray (y_bin, x_bin, member)

Bin edge strategies (per axis, independently configurable)
----------------------------------------------------------
  "quantile" : edges placed at equally-spaced quantile levels of the pooled
               (across all ensemble members) finite data.  Guarantees equal
               sample counts per bin; robust to skewed distributions.
  "linear"   : edges linearly spaced over [vmin, vmax].  Preserves physical
               axis scaling; appropriate for variables with clear dimensional
               meaning (e.g. aridity index, temperature).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from dask.base import compute
from dask.delayed import delayed
import numpy as np
import xarray as xr


BinStrategy = Literal["quantile", "linear"]


# ---------------------------------------------------------------------------
# Bin edge constructors
# ---------------------------------------------------------------------------

def _build_edges(
    arr_flat: np.ndarray,
    n_bins: int,
    strategy: BinStrategy,
    value_range: Optional[Tuple[float, float]],
    collapse_duplicates: bool = False,
) -> np.ndarray:
    """
    Build bin edges from a flattened, potentially NaN-containing array.

    Parameters
    ----------
    arr_flat    : 1-D array from which edges are derived (NaN tolerated)
    n_bins      : number of bins
    strategy    : "quantile" | "linear"
    value_range : explicit (lo, hi); required for "linear" if the
                  caller wants reproducible edges independent of the data range.
                  If None, lo/hi are taken from the finite data min/max.
    collapse_duplicates : if True and strategy="quantile", remove duplicate
                  edges produced by tied values (for example many zeros).
                  This reduces the effective number of bins and removes
                  zero-width/always-empty bins.

    Returns
    -------
    edges : np.ndarray of shape (n_bins + 1,), strictly monotone non-decreasing.

    Notes
    -----
    Duplicate edges are intentionally preserved for the "quantile" strategy.
    A spike-distributed variable (e.g. LAI=0 over bare soil) will produce
    several zero-width bins at the spike value. np.searchsorted(side='right')
    in _bin_mean_single_member assigns all spike values to the *last*
    zero-width bin, yielding one high-count bin at the spike and zero-count
    bins before it. Nudging duplicates to enforce strict monotonicity would
    instead scatter spike values across many near-empty bins, which is
    incorrect.
    """
    finite = arr_flat[np.isfinite(arr_flat)]
    if finite.size == 0:
        raise ValueError("Binning variable contains no finite values.")

    if strategy == "quantile":
        levels = np.linspace(0.0, 1.0, n_bins + 1)
        edges  = np.quantile(finite, levels)
        if collapse_duplicates:
            edges = np.unique(edges)
            if edges.size < 2:
                raise ValueError(
                    "Quantile edges collapsed to a single value; cannot form bins. "
                    "Use fewer bins or disable collapse_duplicates."
                )

    elif strategy == "linear":
        lo, hi = value_range if value_range is not None else (finite.min(), finite.max())
        edges  = np.linspace(float(lo), float(hi), n_bins + 1)

    else:
        raise ValueError(f"Unknown bin strategy '{strategy}'. Choose quantile | linear | log.")

    return edges


# ---------------------------------------------------------------------------
# Core per-member binning (pure numpy, no Python loops over bins)
# ---------------------------------------------------------------------------

def _bin_stats_single_member(
    target_flat : np.ndarray,  # (N,)  values to average
    y_flat      : np.ndarray,  # (N,)  y-axis binning variable
    x_flat      : np.ndarray,  # (N,)  x-axis binning variable
    y_edges     : np.ndarray,  # (n_y + 1,)
    x_edges     : np.ndarray,  # (n_x + 1,)
) -> np.ndarray:
    """
    Compute 2-D bin means for one ensemble member via np.bincount.

    Algorithm
    ---------
    1. Map each sample to a (i, j) bin index pair via np.searchsorted(side='right').
    2. Collapse to a linear index l = i * n_x + j.
    3. Vectorised sum and count with np.bincount; divide for mean.

    searchsorted(side='right') is used rather than np.digitize so that spike
    values (e.g. LAI=0) coinciding with duplicate quantile edges are assigned
    to the *last* duplicate bin rather than the first. This groups the entire
    spike into one bin, matching the behaviour of get_bins() in the original
    codebase.

    Parameters
    ----------
    target_flat : variable whose mean is computed per bin
    y_flat      : y-axis binning values (same length as target_flat)
    x_flat      : x-axis binning values (same length as target_flat)
    y_edges     : monotone-increasing edges for y axis
    x_edges     : monotone-increasing edges for x axis

    Returns
    -------
    result : np.ndarray, shape (n_y, n_x); NaN where no samples fall in a bin
    """
    n_y = len(y_edges) - 1
    n_x = len(x_edges) - 1

    # searchsorted(side='right') returns the insertion point after all equal
    # edges; subtract 1 for 0-indexed bin, then clip to valid range.
    y_idx = np.clip(np.searchsorted(y_edges, y_flat, side="right") - 1, 0, n_y - 1)
    x_idx = np.clip(np.searchsorted(x_edges, x_flat, side="right") - 1, 0, n_x - 1)

    # Joint validity mask: exclude NaN in any of the three fields
    valid = np.isfinite(target_flat) & np.isfinite(y_flat) & np.isfinite(x_flat)

    v = target_flat[valid]
    lin_idx    = y_idx[valid] * n_x + x_idx[valid]
    total_bins = n_y * n_x

    bin_count = np.bincount(lin_idx, minlength=total_bins).astype(np.float64)
    bin_sum   = np.bincount(lin_idx, weights=v, minlength=total_bins)
    bin_sum2  = np.bincount(lin_idx, weights=v * v, minlength=total_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(bin_count > 0, bin_sum / bin_count, np.nan)
        ex2 = np.where(bin_count > 0, bin_sum2 / bin_count, np.nan)

    # Compute variance using: Var(X) = E[X^2] - E[X]^2
    var_pop = ex2 - mean * mean
    var_pop = np.maximum(var_pop, 0.0)

    # Unbiased sample variance (ddof=1)
    var_samp = np.where(bin_count > 1, var_pop * bin_count / (bin_count - 1.0), np.nan)
    
    bin_count = bin_count.reshape(n_y, n_x)
    mean = mean.reshape(n_y, n_x)
    var_pop = var_pop.reshape(n_y, n_x)
    var_samp = var_samp.reshape(n_y, n_x)

    return np.stack((mean, var_pop, var_samp, bin_count), axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_2d_bin_stats(
    target  : xr.DataArray,
    y_var   : xr.DataArray,
    x_var   : xr.DataArray,
    *,
    member_dim  : Optional[str] = None,
    n_y_bins : int = 15,
    n_x_bins : int = 15,
    y_strategy   : BinStrategy = "quantile",
    x_strategy   : BinStrategy = "quantile",
    y_range  : Optional[Tuple[float, float]] = None,
    x_range  : Optional[Tuple[float, float]] = None,
    collapse_duplicate_quantile_bins: bool = False,
    pool_edges_across_ensemble: bool = True,
    parallel : bool = True,
) -> xr.DataArray:
    """
    Bin `target` into 2-D (y_var x x_var) space and return bin stats.

    Parameters
    ----------
    target  : variable to bin
    y_var   : y-axis binning variable (same shape as target)
    x_var   : x-axis binning variable (same shape as target)
    member_dim  : name of member dimension
    n_y_bins : number of bins along y axis
    n_x_bins : number of bins along x axis
    y_strategy : edge strategy for y axis ("quantile" | "linear")
    x_strategy : edge strategy for x axis ("quantile" | "linear")
    y_range  : explicit (lo, hi) for y edges; None -> derived from data
    x_range  : explicit (lo, hi) for x edges; None -> derived from data
    collapse_duplicate_quantile_bins : if True, remove duplicate quantile
        edges caused by tied values. This condenses empty bins but may reduce
        the effective number of bins below n_y_bins/n_x_bins.
    pool_edges_across_ensemble : if True, edges are computed from the full
        ensemble pool, guaranteeing identical bin definitions across members.
        Should be True for cross-member PPE comparison; False is not yet
        implemented.
    parallel : dispatch ensemble members as dask.delayed tasks

    Returns
    -------
    xr.DataArray, dims=(y_bin, x_bin, member)
        Coordinates include bin-centre values and bin-edge attributes.
    """
    if not pool_edges_across_ensemble:
        raise NotImplementedError("Per-member edges not yet supported.")

    dim_order = tuple(d for d in target.dims if d != member_dim) + (member_dim,)
    target = target.transpose(*dim_order)
    y_var  = y_var.transpose(*dim_order)
    x_var  = x_var.transpose(*dim_order)

    # Materialise once; tolerated for PPE sizes (typically <10 GB per variable)
    def _np(da):
        return da.compute().values if hasattr(da.data, "compute") else da.values

    tgt_np = _np(target)
    y_np   = _np(y_var)
    x_np   = _np(x_var)

    # Build edges from pooled finite values
    y_edges = _build_edges(
        y_np.ravel(),
        n_y_bins,
        y_strategy,
        y_range,
        collapse_duplicates=collapse_duplicate_quantile_bins,
    )
    x_edges = _build_edges(
        x_np.ravel(),
        n_x_bins,
        x_strategy,
        x_range,
        collapse_duplicates=collapse_duplicate_quantile_bins,
    )

    n_y_eff = len(y_edges) - 1
    n_x_eff = len(x_edges) - 1

    members   = target.coords[member_dim].values
    n_members = len(members)

    def _process(m: int) -> np.ndarray:
        return _bin_stats_single_member(
            tgt_np[..., m].ravel(),
            y_np[...,   m].ravel(),
            x_np[...,   m].ravel(),
            y_edges, x_edges,
        )

    if parallel:
        results = compute(*[delayed(_process)(m) for m in range(n_members)])
    else:
        results = [_process(m) for m in range(n_members)]

    result_np = np.stack(results, axis=-1)  # (stats, n_y, n_x, n_members)

    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    # Human-readable quantile labels (used when strategy=="quantile")
    def _quantile_labels(n):
        lo = np.linspace(0, 100, n + 1)[:-1]
        hi = np.linspace(0, 100, n + 1)[1:]
        return [f"Q{a:.0f}-Q{b:.0f}" for a, b in zip(lo, hi)]

    y_name = y_var.name or "y_var"
    x_name = x_var.name or "x_var"

    coords = {
        "stats": np.array(["mean", "var_pop", "var_samp", "count"]),
        f"{y_name}_bin_center": ("y_bin", y_centers),
        f"{x_name}_bin_center": ("x_bin", x_centers),
        member_dim:                   members,
    }
    if y_strategy == "quantile":
        coords[f"{y_name}_bin_label"] = ("y_bin", _quantile_labels(n_y_eff))
    if x_strategy == "quantile":
        coords[f"{x_name}_bin_label"] = ("x_bin", _quantile_labels(n_x_eff))

    out = xr.DataArray(
        result_np,
        dims=("stats", "y_bin", "x_bin", member_dim),
        coords=coords,
        attrs={
            "long_name"            : f"2-D bin-mean {target.name or 'variable'}",
            "units"                : target.attrs.get("units", "unknown"),
            "y_variable"           : y_name,
            "x_variable"           : x_name,
            "y_strategy"           : y_strategy,
            "x_strategy"           : x_strategy,
            "y_bin_edges"          : y_edges.tolist(),
            "x_bin_edges"          : x_edges.tolist(),
            "n_y_bins"             : n_y_eff,
            "n_x_bins"             : n_x_eff,
            "n_y_bins_requested"   : n_y_bins,
            "n_x_bins_requested"   : n_x_bins,
            "collapse_duplicate_quantile_bins": collapse_duplicate_quantile_bins,
            "pool_edges"           : pool_edges_across_ensemble,
        },
    )
    return out


def compute_2d_bin_stats_single_member(
    target  : xr.DataArray,
    y_var   : xr.DataArray,
    x_var   : xr.DataArray,
    *,
    n_y_bins : int = 15,
    n_x_bins : int = 15,
    y_strategy   : BinStrategy = "quantile",
    x_strategy   : BinStrategy = "quantile",
    y_range  : Optional[Tuple[float, float]] = None,
    x_range  : Optional[Tuple[float, float]] = None,
    collapse_duplicate_quantile_bins: bool = False,
) -> xr.DataArray:
    """
    Bin `target` into 2-D (y_var x x_var) space and return bin stats.

    Parameters
    ----------
    target  : variable to bin
    y_var   : y-axis binning variable (same shape as target)
    x_var   : x-axis binning variable (same shape as target)
    n_y_bins : number of bins along y axis
    n_x_bins : number of bins along x axis
    y_strategy : edge strategy for y axis ("quantile" | "linear" | "log")
    x_strategy : edge strategy for x axis ("quantile" | "linear" | "log")
    y_range  : explicit (lo, hi) for y edges; None -> derived from data
    x_range  : explicit (lo, hi) for x edges; None -> derived from data
    collapse_duplicate_quantile_bins : if True, remove duplicate quantile
        edges caused by tied values. This condenses empty bins but may reduce
        the effective number of bins below n_y_bins/n_x_bins.

    Returns
    -------
    xr.DataArray, dims=(y_bin, x_bin, member_dim)
        Coordinates include bin-centre values and bin-edge attributes.
    """
    # dim_order = target.dims
    # target = target.transpose(*dim_order)
    # y_var  = y_var.transpose(*dim_order)
    # x_var  = x_var.transpose(*dim_order)

    # Materialise once; tolerated for PPE sizes (typically <10 GB per variable)
    def _np(da):
        return da.compute().values if hasattr(da.data, "compute") else da.values

    tgt_np = _np(target)
    y_np   = _np(y_var)
    x_np   = _np(x_var)

    # Build edges from pooled finite values
    y_edges = _build_edges(
        y_np.ravel(),
        n_y_bins,
        y_strategy,
        y_range,
        collapse_duplicates=collapse_duplicate_quantile_bins,
    )
    x_edges = _build_edges(
        x_np.ravel(),
        n_x_bins,
        x_strategy,
        x_range,
        collapse_duplicates=collapse_duplicate_quantile_bins,
    )

    n_y_eff = len(y_edges) - 1
    n_x_eff = len(x_edges) - 1

    result_np = _bin_stats_single_member(
        tgt_np.ravel(),
        y_np.ravel(),
        x_np.ravel(),
        y_edges, x_edges,
    )

    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    # Human-readable quantile labels (used when strategy=="quantile")
    def _quantile_labels(n):
        lo = np.linspace(0, 100, n + 1)[:-1]
        hi = np.linspace(0, 100, n + 1)[1:]
        return [f"Q{a:.0f}-Q{b:.0f}" for a, b in zip(lo, hi)]

    y_name = y_var.name or "y_var"
    x_name = x_var.name or "x_var"

    coords = {
        "stats": np.array(["mean", "var_pop", "var_samp", "count"]),
        f"{y_name}_bin_center": ("y_bin", y_centers),
        f"{x_name}_bin_center": ("x_bin", x_centers),
    }
    if y_strategy == "quantile":
        coords[f"{y_name}_bin_label"] = ("y_bin", _quantile_labels(n_y_eff))
    if x_strategy == "quantile":
        coords[f"{x_name}_bin_label"] = ("x_bin", _quantile_labels(n_x_eff))

    out = xr.DataArray(
        result_np,
        dims=("stats", "y_bin", "x_bin"),
        coords=coords,
        attrs={
            "long_name"            : f"2-D bin-mean {target.name or 'variable'}",
            "units"                : target.attrs.get("units", "unknown"),
            "y_variable"           : y_name,
            "x_variable"           : x_name,
            "y_strategy"           : y_strategy,
            "x_strategy"           : x_strategy,
            "y_bin_edges"          : y_edges.tolist(),
            "x_bin_edges"          : x_edges.tolist(),
            "n_y_bins"             : n_y_eff,
            "n_x_bins"             : n_x_eff,
            "n_y_bins_requested"   : n_y_bins,
            "n_x_bins_requested"   : n_x_bins,
            "collapse_duplicate_quantile_bins": collapse_duplicate_quantile_bins,
        },
    )
    return out
