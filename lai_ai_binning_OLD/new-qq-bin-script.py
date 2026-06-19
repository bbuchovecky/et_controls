from __future__ import annotations

import numpy as np
import xarray as xr


def weighted_quantile(values, quantiles, weights):
    """
    Compute weighted quantiles of a 1D array.

    Parameters
    ----------
    values : (N,) array
        Data values.
    quantiles : array-like
        Quantile levels in [0, 1].
    weights : (N,) array
        Nonnegative weights.

    Returns
    -------
    q : ndarray
        Weighted quantile values.
    """
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]

    if values.size == 0:
        return np.full(quantiles.shape, np.nan, dtype=float)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return np.interp(quantiles, cdf, values)


def compute_bin_edges(values, nbins, mode="quantile", weights=None):
    """
    Compute 1D bin edges from ordinary or weighted quantiles.
    """
    q = np.linspace(0.0, 1.0, nbins + 1)
    values = np.asarray(values)

    if mode == "quantile":
        mask = np.isfinite(values)
        if not np.any(mask):
            return np.full(nbins + 1, np.nan)
        return np.quantile(values[mask], q)

    if mode == "equal_area":
        if weights is None:
            raise ValueError("weights must be provided when mode='equal_area'")
        return weighted_quantile(values, q, weights)

    raise ValueError("mode must be 'quantile' or 'equal_area'")


def _bin_2d_numpy(
    x,
    y,
    z,
    x_edges,
    y_edges,
    reducer="mean",
    weights=None,
    weighted=False,
):
    """
    Core 2D binning kernel on 1D numpy arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        valid &= np.isfinite(weights) & (weights > 0)

    x = x[valid]
    y = y[valid]
    z = z[valid]
    if weights is not None:
        weights = weights[valid]

    nbx = len(x_edges) - 1
    nby = len(y_edges) - 1

    x_bin = np.searchsorted(x_edges[1:-1], x, side="right")
    y_bin = np.searchsorted(y_edges[1:-1], y, side="right")
    x_bin = np.clip(x_bin, 0, nbx - 1)
    y_bin = np.clip(y_bin, 0, nby - 1)

    counts = np.zeros((nbx, nby), dtype=int)
    np.add.at(counts, (x_bin, y_bin), 1)

    out = np.full((nbx, nby), np.nan, dtype=float)

    if reducer == "count":
        out = counts.astype(float)

    elif reducer == "sum":
        sums = np.zeros((nbx, nby), dtype=float)
        if weighted:
            if weights is None:
                raise ValueError("weights must be provided when weighted=True")
            np.add.at(sums, (x_bin, y_bin), weights * z)
        else:
            np.add.at(sums, (x_bin, y_bin), z)
        out = sums

    elif reducer == "mean":
        num = np.zeros((nbx, nby), dtype=float)
        den = np.zeros((nbx, nby), dtype=float)

        if weighted:
            if weights is None:
                raise ValueError("weights must be provided when weighted=True")
            np.add.at(num, (x_bin, y_bin), weights * z)
            np.add.at(den, (x_bin, y_bin), weights)
        else:
            np.add.at(num, (x_bin, y_bin), z)
            np.add.at(den, (x_bin, y_bin), 1.0)

        out = np.divide(
            num,
            den,
            out=np.full_like(num, np.nan),
            where=den > 0,
        )

    else:
        raise ValueError("reducer must be one of: mean, sum, count")

    # ------------------------------------------------------------------
    # Per-bin unweighted sample variance (ddof=1).
    # Computed for every reducer so callers always have sample size
    # (counts) and variance available for significance testing.
    # ------------------------------------------------------------------
    bin_sum = np.zeros((nbx, nby), dtype=float)
    np.add.at(bin_sum, (x_bin, y_bin), z)
    bin_mean_for_var = np.where(
        counts > 0,
        bin_sum / np.maximum(counts, 1),
        np.nan,
    )
    ss = np.zeros((nbx, nby), dtype=float)
    np.add.at(ss, (x_bin, y_bin), (z - bin_mean_for_var[x_bin, y_bin]) ** 2)
    variance = np.divide(
        ss,
        counts - 1,
        out=np.full_like(ss, np.nan),
        where=counts > 1,
    )

    return out, counts, variance


def _compute_edges_for_subset(x, y, nbx, nby, x_mode, y_mode, area=None):
    """
    Compute x and y edges from one subset.
    """
    xv = np.asarray(x)
    yv = np.asarray(y)

    valid = np.isfinite(xv) & np.isfinite(yv)
    av = None
    if area is not None:
        av = np.asarray(area, dtype=float)
        valid &= np.isfinite(av) & (av > 0)

    xv = xv[valid]
    yv = yv[valid]
    if av is not None:
        av = av[valid]

    x_edges = compute_bin_edges(xv, nbx, mode=x_mode, weights=av)
    y_edges = compute_bin_edges(yv, nby, mode=y_mode, weights=av)
    return x_edges, y_edges


def bin_2d_grouped_xarray(
    x: xr.DataArray,
    y: xr.DataArray,
    z: xr.DataArray,
    group_dim: str = "month",
    nbx: int = 10,
    nby: int = 10,
    reducer: str = "mean",
    x_mode: str = "quantile",
    y_mode: str = "quantile",
    area: xr.DataArray | None = None,
    weighted: bool = False,
    edge_strategy: str = "fixed",
    fixed_x_edges: xr.DataArray | np.ndarray | None = None,
    fixed_y_edges: xr.DataArray | np.ndarray | None = None,
    parallel: bool = False,
) -> xr.Dataset:
    """
    Compute one 2D binned field per group (e.g. per month).

    Parameters
    ----------
    x, y, z : xr.DataArray
        Arrays with identical dimensions and coordinates.
    group_dim : str
        Name of the grouping dimension, e.g. "month".
    nbx, nby : int
        Number of x and y bins.
    reducer : {"mean", "median", "sum", "count"}
        Aggregation of z inside each 2D bin.
    x_mode, y_mode : {"quantile", "equal_area"}
        Method used to derive bin edges when edges are computed internally.
    area : xr.DataArray, optional
        Area/sample weights. Must be broadcastable to x/y/z.
        For example, x may be (month, lat, lon) while area is (lat, lon).
    weighted : bool
        If True and reducer is "mean" or "sum", aggregate z using `area`.
    edge_strategy : {"fixed", "monthly"}
        - "fixed": compute one set of x/y edges from the full dataset and reuse
          for every group, unless fixed_x_edges/fixed_y_edges are supplied.
        - "monthly": compute x/y edges separately within each group.
    fixed_x_edges, fixed_y_edges : array-like, optional
        Explicit bin edges to use. If provided, these override edge computation
        for that variable.
    parallel : bool, optional
        If True, execute per-group binning calls concurrently via
        ``dask.delayed`` / ``dask.compute``.  Requires dask to be installed.
        Dask-backed inputs are always materialised in one shot before the
        group loop regardless of this flag.  Default is False.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - zbinned(group_dim, x_bin, y_bin)
        - counts(group_dim, x_bin, y_bin)
        - x_edges(...)
        - y_edges(...)
        - x_center(...)
        - y_center(...)
    """
    if not isinstance(x, xr.DataArray) or not isinstance(y, xr.DataArray) or not isinstance(z, xr.DataArray):
        raise TypeError("x, y, and z must be xarray.DataArray objects")

    if x.dims != y.dims or x.dims != z.dims:
        raise ValueError("x, y, and z must have identical dimension order")

    if group_dim not in x.dims:
        raise ValueError(f"group_dim='{group_dim}' must be a dimension of x/y/z")

    if edge_strategy not in {"fixed", "monthly"}:
        raise ValueError("edge_strategy must be 'fixed' or 'monthly'")

    # Broadcast to common shape
    x, y, z = xr.broadcast(x, y, z)
    if area is not None:
        area, _ = xr.broadcast(area, z)

    # ------------------------------------------------------------------
    # Dask compatibility: detect Dask-backed inputs and materialise them
    # in one graph execution so that per-group .isel(...).values calls
    # do not each trigger a separate full recompute.
    # ------------------------------------------------------------------
    try:
        import dask.array as _dsa
        _any_dask = any(
            isinstance(a.data, _dsa.Array)
            for a in [x, y, z] + ([area] if area is not None else [])
        )
    except ImportError:
        _any_dask = False
        if parallel:
            raise ImportError(
                "parallel=True requires dask to be installed"
            ) from None

    if _any_dask:
        _to_compute = {"__x": x, "__y": y, "__z": z}
        if area is not None:
            _to_compute["__area"] = area
        _computed = xr.Dataset(_to_compute).compute()
        x = _computed["__x"]
        y = _computed["__y"]
        z = _computed["__z"]
        if area is not None:
            area = _computed["__area"]

    group_coord = x[group_dim]
    ng = group_coord.size

    other_dims = tuple(d for d in x.dims if d != group_dim)
    if len(other_dims) == 0:
        raise ValueError("Need at least one non-group dimension to bin over")

    z_name = z.name or "z"
    x_name = x.name or "x"
    y_name = y.name or "y"

    # ------------------------------------------------------------
    # Determine bin edges
    # ------------------------------------------------------------
    if fixed_x_edges is not None:
        fixed_x_edges = np.asarray(fixed_x_edges)
        if fixed_x_edges.ndim != 1 or fixed_x_edges.size != nbx + 1:
            raise ValueError("fixed_x_edges must be 1D with length nbx+1")

    if fixed_y_edges is not None:
        fixed_y_edges = np.asarray(fixed_y_edges)
        if fixed_y_edges.ndim != 1 or fixed_y_edges.size != nby + 1:
            raise ValueError("fixed_y_edges must be 1D with length nby+1")

    if edge_strategy == "fixed":
        # Use user-provided edges when supplied; otherwise compute from full series
        if fixed_x_edges is None or fixed_y_edges is None:
            xv = x.values
            yv = y.values
            av = area.values if area is not None else None
            x_edges_full, y_edges_full = _compute_edges_for_subset(
                xv, yv, nbx, nby, x_mode, y_mode, area=av
            )
            if fixed_x_edges is None:
                fixed_x_edges = x_edges_full
            if fixed_y_edges is None:
                fixed_y_edges = y_edges_full

        x_edges_out = xr.DataArray(
            fixed_x_edges,
            dims=("x_edge",),
            coords={"x_edge": np.arange(nbx + 1)},
            attrs={"source_variable": x_name, "bin_mode": x_mode, "edge_strategy": "fixed"},
        )
        y_edges_out = xr.DataArray(
            fixed_y_edges,
            dims=("y_edge",),
            coords={"y_edge": np.arange(nby + 1)},
            attrs={"source_variable": y_name, "bin_mode": y_mode, "edge_strategy": "fixed"},
        )

    else:  # monthly
        x_edges_month = np.full((ng, nbx + 1), np.nan, dtype=float)
        y_edges_month = np.full((ng, nby + 1), np.nan, dtype=float)

        for ig in range(ng):
            xg = x.isel({group_dim: ig}).values
            yg = y.isel({group_dim: ig}).values
            ag = area.isel({group_dim: ig}).values if area is not None else None

            if fixed_x_edges is not None and fixed_y_edges is not None:
                xeg, yeg = fixed_x_edges, fixed_y_edges
            else:
                xeg, yeg = _compute_edges_for_subset(
                    xg, yg, nbx, nby, x_mode, y_mode, area=ag
                )
                if fixed_x_edges is not None:
                    xeg = fixed_x_edges
                if fixed_y_edges is not None:
                    yeg = fixed_y_edges

            x_edges_month[ig, :] = xeg
            y_edges_month[ig, :] = yeg

        x_edges_out = xr.DataArray(
            x_edges_month,
            dims=(group_dim, "x_edge"),
            coords={group_dim: group_coord, "x_edge": np.arange(nbx + 1)},
            attrs={"source_variable": x_name, "bin_mode": x_mode, "edge_strategy": "monthly"},
        )
        y_edges_out = xr.DataArray(
            y_edges_month,
            dims=(group_dim, "y_edge"),
            coords={group_dim: group_coord, "y_edge": np.arange(nby + 1)},
            attrs={"source_variable": y_name, "bin_mode": y_mode, "edge_strategy": "monthly"},
        )

    # ------------------------------------------------------------
    # Bin group by group
    # ------------------------------------------------------------
    zbinned = np.full((ng, nbx, nby), np.nan, dtype=float)
    counts = np.zeros((ng, nbx, nby), dtype=int)
    variance = np.full((ng, nbx, nby), np.nan, dtype=float)

    # Pre-extract per-group numpy arrays once, then dispatch sequentially
    # or in parallel via dask.delayed.
    _group_args = []
    for ig in range(ng):
        xg = x.isel({group_dim: ig}).values.ravel()
        yg = y.isel({group_dim: ig}).values.ravel()
        zg = z.isel({group_dim: ig}).values.ravel()
        ag = area.isel({group_dim: ig}).values.ravel() if area is not None else None

        if edge_strategy == "fixed":
            xe, ye = fixed_x_edges, fixed_y_edges
        else:
            xe = x_edges_out.isel({group_dim: ig}).values
            ye = y_edges_out.isel({group_dim: ig}).values

        _group_args.append((xg, yg, zg, xe, ye, ag))

    if parallel:
        import dask
        _delayed = [
            dask.delayed(_bin_2d_numpy)(
                xg, yg, zg,
                x_edges=xe, y_edges=ye,
                reducer=reducer, weights=ag, weighted=weighted,
            )
            for xg, yg, zg, xe, ye, ag in _group_args
        ]
        _results = dask.compute(*_delayed)
    else:
        _results = [
            _bin_2d_numpy(
                xg, yg, zg,
                x_edges=xe, y_edges=ye,
                reducer=reducer, weights=ag, weighted=weighted,
            )
            for xg, yg, zg, xe, ye, ag in _group_args
        ]

    for ig, (zbin_g, count_g, var_g) in enumerate(_results):
        zbinned[ig, :, :] = zbin_g
        counts[ig, :, :] = count_g
        variance[ig, :, :] = var_g

    # ------------------------------------------------------------
    # Bin centers
    # ------------------------------------------------------------
    if edge_strategy == "fixed":
        x_center = 0.5 * (fixed_x_edges[:-1] + fixed_x_edges[1:])
        y_center = 0.5 * (fixed_y_edges[:-1] + fixed_y_edges[1:])

        x_center_da = xr.DataArray(
            x_center,
            dims=("x_bin",),
            coords={"x_bin": np.arange(nbx)},
        )
        y_center_da = xr.DataArray(
            y_center,
            dims=("y_bin",),
            coords={"y_bin": np.arange(nby)},
        )
    else:
        x_center = 0.5 * (x_edges_out.isel(x_edge=slice(0, -1)).values +
                          x_edges_out.isel(x_edge=slice(1, None)).values)
        y_center = 0.5 * (y_edges_out.isel(y_edge=slice(0, -1)).values +
                          y_edges_out.isel(y_edge=slice(1, None)).values)

        x_center_da = xr.DataArray(
            x_center,
            dims=(group_dim, "x_bin"),
            coords={group_dim: group_coord, "x_bin": np.arange(nbx)},
        )
        y_center_da = xr.DataArray(
            y_center,
            dims=(group_dim, "y_bin"),
            coords={group_dim: group_coord, "y_bin": np.arange(nby)},
        )

    # ------------------------------------------------------------
    # Build output dataset
    # ------------------------------------------------------------
    ds = xr.Dataset(
        data_vars={
            "zbinned": xr.DataArray(
                zbinned,
                dims=(group_dim, "x_bin", "y_bin"),
                coords={
                    group_dim: group_coord,
                    "x_bin": np.arange(nbx),
                    "y_bin": np.arange(nby),
                },
                attrs={
                    "long_name": f"{reducer} of {z_name} in 2D bins of {x_name} and {y_name}",
                    "reducer": reducer,
                    "weighted": weighted,
                    "x_mode": x_mode,
                    "y_mode": y_mode,
                    "edge_strategy": edge_strategy,
                },
            ),
            "counts": xr.DataArray(
                counts,
                dims=(group_dim, "x_bin", "y_bin"),
                coords={
                    group_dim: group_coord,
                    "x_bin": np.arange(nbx),
                    "y_bin": np.arange(nby),
                },
                attrs={"long_name": "number of samples in bin"},
            ),
            "variance": xr.DataArray(
                variance,
                dims=(group_dim, "x_bin", "y_bin"),
                coords={
                    group_dim: group_coord,
                    "x_bin": np.arange(nbx),
                    "y_bin": np.arange(nby),
                },
                attrs={
                    "long_name": f"unweighted sample variance of {z_name} in 2D bin (ddof=1)",
                },
            ),
            "x_edges": x_edges_out,
            "y_edges": y_edges_out,
            "x_center": x_center_da,
            "y_center": y_center_da,
        },
        attrs={
            "group_dim": group_dim,
            "x_variable": x_name,
            "y_variable": y_name,
            "z_variable": z_name,
        },
    )

    if "units" in z.attrs:
        ds["zbinned"].attrs["units"] = z.attrs["units"]
        ds["variance"].attrs["units"] = f"({z.attrs['units']})^2"
    if "units" in x.attrs:
        ds["x_edges"].attrs["units"] = x.attrs["units"]
        ds["x_center"].attrs["units"] = x.attrs["units"]
    if "units" in y.attrs:
        ds["y_edges"].attrs["units"] = y.attrs["units"]
        ds["y_center"].attrs["units"] = y.attrs["units"]

    return ds
