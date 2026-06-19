"""
ppe_eof.py
==========
EOF decomposition for CESM2/CLM5 perturbed parameter ensembles (PPEs).

Two entry points
----------------
ppe_member_eof(da, ...)
    Member-space EOF of a PPE response field (e.g. a 2xCO2 minus control
    difference). The member dimension is treated as the sample dimension
    directly — no time averaging required. This is the primary function
    for characterising the spatial structure of parameter sensitivity.

ppe_temporal_eof(da, time_slice, ...)
    Temporal EOF computed per ensemble member over a specified period.
    Characterises the dominant modes of internal temporal variability
    for each member independently.

Both functions return an EOFResult named tuple with fields matching the
eofs library output plus the derived 2-sigma amplitude field.

Dependencies
------------
    numpy, xarray, eofs  (pip install eofs)

Typical usage (Jupyter)
-----------------------
    import numpy as np
    import xarray as xr
    from ppe_eof import apply_land_mask, validate_mask, ppe_member_eof, ppe_temporal_eof

    # --- Member EOF on a difference field (e.g. cppe_diff) ---
    da_masked = apply_land_mask(cppe_diff[v], land_frac)
    result = ppe_member_eof(da_masked, weights=ppe_atm_lw, n_eofs=4)

    result.eofs          # (n_eofs, lat, lon)  — unit-variance spatial patterns
    result.eofs_2sigma   # (n_eofs, lat, lon)  — 2σ amplitude in variable units
    result.pcs           # (n_eofs, member)    — amplitude-scaled scores
    result.varfrac       # (n_eofs,)           — variance fractions
    result.eig           # (n_eofs,)           — eigenvalues
    result.eig_err       # (n_eofs,)           — North et al. (1982) error bars
    result.reconstructed # (lat, lon)          — reconstruction from EOF 1

    # --- Temporal EOF per member ---
    results_t = ppe_temporal_eof(da, time_slice=slice("1980", "2010"), n_eofs=4)
    # Returns dict keyed by member value, each value an EOFResult.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

import numpy as np
import xarray as xr
from eofs.xarray import Eof


# --------------------------------------------------------------------------- #
# Return type
# --------------------------------------------------------------------------- #

EOFResult = namedtuple(
    "EOFResult",
    [
        "eofs",         # (n_eofs, lat, lon)  unit-variance spatial patterns
        "eofs_2sigma",  # (n_eofs, lat, lon)  2σ physical amplitude, variable units
        "varfrac",      # (n_eofs,)           fraction of variance explained
        "eig",          # (n_eofs,)           eigenvalues
        "eig_err",      # (n_eofs,)           North et al. (1982) eigenvalue errors
        "pcs",          # (n_eofs, n_samples) amplitude-scaled PC scores
        "solver",       # eofs.xarray.Eof     solver object for further queries
        "reconstructed",# (lat, lon)          field reconstructed from EOF 1
    ],
)


# --------------------------------------------------------------------------- #
# Land masking
# --------------------------------------------------------------------------- #

def apply_land_mask(
    da: xr.DataArray,
    land_frac: xr.DataArray,
    threshold: float = 0.5,
) -> xr.DataArray:
    """
    Mask sub-continental and ocean cells prior to EOF computation.

    Sets cells where land_frac < threshold to NaN. The eofs library
    automatically excludes NaN cells from SVD, provided the mask is
    spatially static (consistent across all samples); use validate_mask()
    to confirm this.

    Parameters
    ----------
    da : xr.DataArray
        Input array with at least dimensions (lat, lon).
    land_frac : xr.DataArray
        Land fraction field (lat, lon), values in [0, 1].
        Standard CESM variable: 'landfrac' from CLM history files.
    threshold : float
        Cells with land_frac < threshold are masked to NaN.
        Default 0.5 is conventional; lower values retain more coastal cells.

    Returns
    -------
    xr.DataArray
        Copy of da with ocean/marginal cells set to NaN.
    """
    mask = land_frac >= threshold
    return da.where(mask)


def validate_mask(da: xr.DataArray, sample_dim: str) -> None:
    """
    Assert that the NaN mask in da is spatially static across sample_dim.

    The eofs library handles static missing values correctly but produces
    incorrect results if NaN locations vary across the sample dimension.
    Raises AssertionError with a descriptive message on failure.

    Parameters
    ----------
    da : xr.DataArray
        Array to validate. Must contain sample_dim as a dimension.
    sample_dim : str
        The dimension across which NaN consistency is checked.
        Use 'member' for ppe_member_eof, 'time' for ppe_temporal_eof.
    """
    nan_any = da.isnull().any(dim=sample_dim)
    nan_all = da.isnull().all(dim=sample_dim)
    mixed   = nan_any & ~nan_all

    if mixed.any():
        n_bad = int(mixed.sum())
        raise AssertionError(
            f"NaN mask is not static across '{sample_dim}': {n_bad} spatial cells "
            f"are NaN for some but not all values of '{sample_dim}'. "
            "EOF computation will be unreliable. Check for fill values or "
            "inconsistent land masks across members."
        )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _build_lat_weights(da: xr.DataArray) -> xr.DataArray:
    """
    Construct sqrt(cos(lat)) area weights normalised to mean = 1.

    Multiplying the data matrix by sqrt(cos(lat)) before SVD is equivalent
    to computing a latitude-weighted spatial covariance. This prevents
    high-latitude grid cells — which are smaller but numerous on regular
    lat/lon grids — from dominating the leading EOF modes.

    Weights are normalised to mean = 1 so they do not rescale the overall
    variance of the field.

    Parameters
    ----------
    da : xr.DataArray
        Array containing a 'lat' coordinate in degrees.

    Returns
    -------
    xr.DataArray
        1-D weight array with dimension 'lat'.
    """
    lat_rad = np.deg2rad(da["lat"])
    w = np.cos(lat_rad).clip(min=0.0) ** 0.5
    return xr.DataArray(w / w.mean(), coords={"lat": da["lat"]})


def _compute_eofs_2sigma(
    eofs_: xr.DataArray,
    pcs_: xr.DataArray,
    weights: xr.DataArray | None,
) -> xr.DataArray:
    """
    Compute the 2-sigma physical amplitude of each EOF pattern.

    Definition
    ----------
    For each EOF mode k:

        eofs_2sigma_k(x) = 2 * σ_k * e_k(x) / w(x)

    where:
        σ_k   = standard deviation of PC_k across samples (scalar)
        e_k(x) = EOF pattern k at grid cell x  (eofscaling=0: unit variance)
        w(x)  = area weight at grid cell x (sqrt(cos(lat)), or 1 if no weights)

    Interpretation: the spatial field anomaly (in original variable units)
    associated with a 2-standard-deviation excursion along mode k.
    Division by w(x) undoes the weighting applied before SVD, returning
    values to the physical (unweighted) space of the input variable.

    Parameters
    ----------
    eofs_ : xr.DataArray
        EOF patterns, shape (n_eofs, lat, lon), eofscaling=0 (unit variance).
    pcs_ : xr.DataArray
        PC scores, shape (n_eofs, n_samples), pcscaling=0 (amplitude in PCs).
    weights : xr.DataArray or None
        Area weights (lat,), same weights passed to the Eof solver.
        If None, the division step is skipped (w=1 everywhere).

    Returns
    -------
    xr.DataArray
        2-sigma amplitude field, shape (n_eofs, lat, lon), same units as
        the input DataArray.
    """
    # σ_k: std of each PC across the sample dimension — shape (n_eofs,)
    std_pcs = pcs_.std(dim=pcs_.dims[1])   # dim 1 is the sample dim

    if weights is not None:
        # Reindex weights to match the spatial grid of eofs_
        w2d = weights.reindex_like(eofs_, method="nearest")  # (lat, lon)
        divisor = w2d.values[None, :, :]                     # (1, lat, lon)
    else:
        divisor = 1.0

    values_2sigma = (
        2.0
        * eofs_.values             # (n_eofs, lat, lon)
        * std_pcs.values[:, None, None]   # (n_eofs, 1, 1)
        / divisor
    )

    return xr.DataArray(
        values_2sigma,
        dims=eofs_.dims,
        coords=eofs_.coords,
        attrs=eofs_.attrs,
    )


def _build_eof_result(
    solver: Eof,
    n_eofs: int,
    weights: xr.DataArray | None,
    sample_dim_name: str,
    sample_coords: Any,
    n_reconstruct: int = 1,
) -> EOFResult:
    """
    Extract all EOFResult fields from a fitted Eof solver.

    Parameters
    ----------
    solver : eofs.xarray.Eof
        A fitted solver instance.
    n_eofs : int
        Number of leading EOFs to extract.
    weights : xr.DataArray or None
        Area weights used when fitting the solver. Passed to
        _compute_eofs_2sigma for the physical amplitude calculation.
    sample_dim_name : str
        Name for the sample dimension in the PC coordinate
        ('member' or 'time').
    sample_coords : array-like
        Coordinate values for the sample dimension.
    n_reconstruct : int
        Number of EOFs to use in reconstructedField. Default 1.

    Returns
    -------
    EOFResult
    """
    eof_idx = np.arange(n_eofs)

    eofs_ = solver.eofs(neofs=n_eofs, eofscaling=0)
    eofs_ = eofs_.assign_coords(eof=eof_idx)

    pcs_ = solver.pcs(npcs=n_eofs, pcscaling=0)
    pcs_ = (
        pcs_
        .rename({"pseudo_pcs": "eof"})
        .assign_coords(eof=eof_idx)
        .assign_coords({sample_dim_name: sample_coords})
        .transpose("eof", sample_dim_name)
    )

    varfrac = solver.varianceFraction(neigs=n_eofs).assign_coords(eof=eof_idx)
    eig     = solver.eigenvalues(neigs=n_eofs).assign_coords(eof=eof_idx)
    eig_err = solver.northTest(neigs=n_eofs, vfscaled=True).assign_coords(eof=eof_idx)

    eofs_2sigma = _compute_eofs_2sigma(eofs_, pcs_, weights)
    reconstructed = solver.reconstructedField(n_reconstruct)

    return EOFResult(
        eofs=eofs_,
        eofs_2sigma=eofs_2sigma,
        varfrac=varfrac,
        eig=eig,
        eig_err=eig_err,
        pcs=pcs_,
        solver=solver,
        reconstructed=reconstructed,
    )


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #

def ppe_member_eof(
    da: xr.DataArray,
    weights: xr.DataArray | None = None,
    n_eofs: int = 4,
    center: bool = False,
    lat_weights: bool = True,
    validate: bool = True,
    n_reconstruct: int = 1,
) -> EOFResult:
    """
    EOF decomposition of a PPE response field across the member dimension.

    The member dimension is treated directly as the sample dimension —
    each ensemble member's spatial field is one observation. This is the
    appropriate decomposition for a difference field (e.g. 2xCO2 minus
    control) where each member represents one realisation of the parameter
    sensitivity response.

    The leading EOFs describe the dominant spatial patterns of inter-member
    spread: EOF 1 is the spatial structure that accounts for the greatest
    fraction of variance across ensemble members.

    Parameters
    ----------
    da : xr.DataArray
        Input array with dimensions (member, lat, lon). For a variable
        with a time dimension, reduce to (member, lat, lon) before calling
        (e.g. via a time mean or by passing a difference field that has
        already been averaged).
    weights : xr.DataArray or None
        Explicit weight field (lat, lon) to pass to the Eof solver, e.g.
        a pre-computed area weight array such as ppe_atm_lw. If provided,
        lat_weights is ignored. If None and lat_weights=True, sqrt(cos(lat))
        weights are constructed automatically from the 'lat' coordinate.
    n_eofs : int
        Number of leading EOFs to retain. Must be <= number of members.
    center : bool
        If True, remove the ensemble-mean spatial pattern before SVD.
        Default False is correct for difference fields (already anomalies);
        set True for raw PPE output where ensemble-mean structure should
        be removed before decomposing inter-member spread.
    lat_weights : bool
        If True and weights=None, construct sqrt(cos(lat)) area weights
        automatically. Ignored if weights is provided explicitly.
    validate : bool
        If True, call validate_mask() to confirm NaN locations are
        consistent across all members before fitting the solver.
    n_reconstruct : int
        Number of leading EOFs to use in reconstructedField. Default 1.

    Returns
    -------
    EOFResult
        Named tuple with fields:
        eofs          (n_eofs, lat, lon)  unit-variance spatial patterns
        eofs_2sigma   (n_eofs, lat, lon)  2σ amplitude in variable units
        varfrac       (n_eofs,)           variance fractions
        eig           (n_eofs,)           eigenvalues
        eig_err       (n_eofs,)           North et al. (1982) error estimates
        pcs           (n_eofs, member)    PC scores (amplitude in PCs)
        solver        eofs.xarray.Eof     fitted solver for further queries
        reconstructed (lat, lon)          field reconstructed from n_reconstruct EOFs

    Raises
    ------
    ValueError
        If required dimensions are missing or n_eofs exceeds member count.
    AssertionError
        If validate=True and the NaN mask is not consistent across members.

    Notes
    -----
    eofscaling=0 / pcscaling=0 convention: EOF patterns have unit variance;
    PC scores carry the amplitude information. The 2-sigma field is computed
    in _compute_eofs_2sigma to recover the physical amplitude in the original
    variable's units.

    Sign convention: eofs orients each EOF so that the spatially-integrated
    absolute loading is positive. This is internally consistent but arbitrary.
    """
    required_dims = {"member", "lat", "lon"}
    missing = required_dims - set(da.dims)
    if missing:
        raise ValueError(f"Input DataArray missing required dimensions: {missing}")

    if n_eofs > da.sizes["member"]:
        raise ValueError(
            f"n_eofs ({n_eofs}) cannot exceed the number of members "
            f"({da.sizes['member']})."
        )

    if validate:
        validate_mask(da, sample_dim="member")

    # Resolve weights: explicit > auto lat weights > None
    if weights is not None:
        w = weights
    elif lat_weights:
        w = _build_lat_weights(da)
    else:
        w = None

    # Member must be the leading (sample) dimension for eofs
    da_ordered = da.transpose("member", "lat", "lon")

    # Rename member → time: eofs.xarray.Eof requires the sample dim
    # to be named 'time'. We assign a synthetic datetime coordinate so
    # the library does not reject the input on dtype grounds.
    n_members = da_ordered.sizes["member"]
    fake_times = np.arange(
        "1000-01-01",
        f"1000-01-{n_members + 1:02d}",
        dtype="datetime64[D]",
    )
    da_renamed = (
        da_ordered
        .rename({"member": "time"})
        .assign_coords(time=fake_times)
    )

    solver = Eof(da_renamed, weights=w, center=center)

    return _build_eof_result(
        solver=solver,
        n_eofs=n_eofs,
        weights=w,
        sample_dim_name="member",
        sample_coords=da.coords["member"].values,
        n_reconstruct=n_reconstruct,
    )


def ppe_temporal_eof(
    da: xr.DataArray,
    time_slice: slice,
    n_eofs: int = 4,
    center: bool = True,
    lat_weights: bool = True,
    validate: bool = True,
    n_reconstruct: int = 1,
) -> dict[Any, EOFResult]:
    """
    Temporal EOF decomposition computed independently per ensemble member.

    For each member, fits an Eof solver over the time dimension within
    time_slice. Returns a dict of EOFResult objects keyed by member value.

    This characterises the dominant modes of internal temporal variability
    for each member. Comparing EOFs across members reveals how parameter
    uncertainty affects the structure of temporal variability.

    Parameters
    ----------
    da : xr.DataArray
        Input array with dimensions (member, time, lat, lon).
        Land-only variables should be pre-masked with apply_land_mask().
    time_slice : slice
        Slice defining the analysis period, applied to the 'time' dimension.
        e.g. ``slice("1980-01-01", "2010-12-31")``.
    n_eofs : int
        Number of leading EOFs to retain per member.
        Must be <= number of time steps in time_slice.
    center : bool
        If True, remove the temporal mean before SVD (standard anomaly
        decomposition). Default True.
    lat_weights : bool
        If True, construct sqrt(cos(lat)) area weights automatically.
    validate : bool
        If True, validate NaN mask consistency across time for each member.
    n_reconstruct : int
        Number of leading EOFs to use in reconstructedField. Default 1.

    Returns
    -------
    dict[member_value, EOFResult]
        Dictionary keyed by the coordinate value of each member.
        Each value is an EOFResult with pcs shaped (n_eofs, time).

    Raises
    ------
    ValueError
        If required dimensions are missing or n_eofs exceeds time steps.
    AssertionError
        If validate=True and the NaN mask is not consistent across time
        for any member.
    """
    required_dims = {"member", "time", "lat", "lon"}
    missing = required_dims - set(da.dims)
    if missing:
        raise ValueError(f"Input DataArray missing required dimensions: {missing}")

    da_sub = da.sel(time=time_slice)
    n_time = da_sub.sizes["time"]

    if n_eofs > n_time:
        raise ValueError(
            f"n_eofs ({n_eofs}) cannot exceed the number of time steps in "
            f"the selected period ({n_time})."
        )

    weights_da = _build_lat_weights(da_sub) if lat_weights else None

    results = {}
    for m in da_sub["member"].values:
        da_m = da_sub.sel(member=m)   # (time, lat, lon)

        if validate:
            validate_mask(da_m, sample_dim="time")

        solver = Eof(da_m, weights=weights_da, center=center)

        results[m] = _build_eof_result(
            solver=solver,
            n_eofs=n_eofs,
            weights=weights_da,
            sample_dim_name="time",
            sample_coords=da_sub.coords["time"].values,
            n_reconstruct=n_reconstruct,
        )

    return results
