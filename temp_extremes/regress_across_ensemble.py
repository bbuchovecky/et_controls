"""
Is there a coherent relationship between the quantile shifts and ΔT_avg across the ensemble?

For each gridcell, regress the left shift and right shift of the anomaly
distribution against the raw change in average temperature across ensemble
members. The regression is performed along the member dimension.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import xarray as xr
from scipy.stats import t as t_dist
import xclimate as xclim

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import regionmask as regmask


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/proc/dist")
FIG_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/fig/temp_extremes")

PROJECTION = ccrs.PlateCarree(central_longitude=12)

MAP_SUBPLOT_KWARGS = {"figsize": (8, 4), "subplot_kw": {"projection": PROJECTION}}
MAP_SUBPLOTS_KWARGS = {
    "fhist": {"ncols": 7, "nrows": 4, "figsize": (21, 7), "subplot_kw": {"projection": PROJECTION}},
    "goga2": {"ncols": 5, "nrows": 2, "figsize": (18, 4), "subplot_kw": {"projection": PROJECTION}},
    "lens2": {"ncols": 7, "nrows": 6, "figsize": (21, 12), "subplot_kw": {"projection": PROJECTION}},
}

LAT_BNDS = slice(-58, 90)
MAP_LAT_BNDS = slice(-54, 90)

CMAPS = {
    "slope": "RdBu_r",
}

GRID = {
    "fhist": xclim.load_fhist_ppe_grid(),
    "lens2": xclim.load_cesm2le_grid(),
    "goga2": xclim.load_goga2_grid(),
}

SHIFT_LABELS = ["left_shift", "right_shift"]
SHIFT_PLOT_LABELS = [
    "Left tail: $(\\Delta\\hat{Q}_{50} - \\Delta\\hat{Q}_{5}) / \\Delta T_{avg}$",
    "Right tail: $(\\Delta\\hat{Q}_{95} - \\Delta\\hat{Q}_{50}) / \\Delta T_{avg}$",
]
STAT_LABELS = ["slope", "intercept", "r", "p_value", "slope_stderr"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_gridlines(ax, lat_bnds=LAT_BNDS):
    x_gls = np.arange(-180, 181, 30)
    y_gls = np.arange(-90, 91, 30)
    y_gls = y_gls[(y_gls >= lat_bnds.start) & (y_gls <= lat_bnds.stop)]
    ax.gridlines(
        draw_labels=False,
        xlocs=x_gls,
        ylocs=y_gls,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )


def share_colorbar_limits(da_list, sym):
    shared_vmin = xr.concat(da_list, dim="_panel").quantile(0.05, skipna=True)
    shared_vmax = xr.concat(da_list, dim="_panel").quantile(0.95, skipna=True)
    if sym:
        shared_vabs = max([abs(shared_vmin), abs(shared_vmax)])
        return -shared_vabs, shared_vabs
    return shared_vmin, shared_vmax


def add_significance_hatching(p_value, ax, alpha=0.05, transform=ccrs.PlateCarree(), hatch="///"):
    """Overlay hatching where the regression p-value is below the threshold."""
    significant = xr.where(p_value < alpha, 1, np.nan)
    if not np.any(np.isfinite(significant.values)):
        return

    ax.contourf(
        significant.lon,
        significant.lat,
        significant,
        levels=[0.5, 1.5],
        colors="none",
        hatches=[hatch],
        transform=transform,
    )


def load_stats_datasets(
    dataset: str,
    variable: str,
    window_days: int,
    time_start: str,
    time_stop: str,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Load anomaly and raw daily summary statistics."""
    proc_dir = OUTPUT_PATH / f"{variable}_pool_{window_days}d_window"
    fname = f"{dataset.upper()}_{variable}_pool_{window_days}d_window_{time_start.replace('-', '')}-{time_stop.replace('-', '')}"

    stats_anom = xr.open_dataset(proc_dir / f"{fname}_anom_stats.nc").sel(lat=LAT_BNDS).compute()
    stats_raw = xr.open_dataset(proc_dir / f"{fname}_stats.nc").sel(lat=LAT_BNDS).compute()

    return stats_anom, stats_raw


def ols_against_member_axis(
    y: xr.DataArray,
    x: xr.DataArray,
    dim: str = "member",
) -> xr.Dataset:
    """Vectorized OLS of y against x along a member axis."""
    valid = np.isfinite(y) & np.isfinite(x)
    y_valid = y.where(valid)
    x_valid = x.where(valid)

    n = valid.sum(dim=dim)
    n_safe = n.where(n > 0)

    x_sum = x_valid.sum(dim=dim)
    y_sum = y_valid.sum(dim=dim)
    xx_sum = (x_valid**2).sum(dim=dim)
    yy_sum = (y_valid**2).sum(dim=dim)
    xy_sum = (x_valid * y_valid).sum(dim=dim)

    x_mean = x_sum / n_safe
    y_mean = y_sum / n_safe

    sxx = xx_sum - (x_sum**2) / n_safe
    syy = yy_sum - (y_sum**2) / n_safe
    sxy = xy_sum - (x_sum * y_sum) / n_safe

    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    y_hat = slope * x_valid + intercept
    ss_res = ((y_valid - y_hat) ** 2).sum(dim=dim)

    dof = (n - 2).where(n > 2)
    slope_stderr = np.sqrt((ss_res / dof) / sxx)

    r = sxy / np.sqrt(sxx * syy)
    r = r.clip(min=-1, max=1)

    t_stat = slope / slope_stderr
    p_value = xr.apply_ufunc(
        lambda t, df: 2.0 * t_dist.sf(np.abs(t), df),
        t_stat,
        dof,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    valid_regression = (n > 2) & np.isfinite(sxx) & (sxx > 0) & np.isfinite(syy) & (syy > 0)

    return xr.Dataset(
        data_vars={
            "slope": slope.where(valid_regression),
            "intercept": intercept.where(valid_regression),
            "r": r.where(valid_regression),
            "p_value": p_value.where(valid_regression),
            "slope_stderr": slope_stderr.where(valid_regression),
        },
        attrs={
            "description": "Gridcell OLS regression of quantile shift against raw mean-temperature change",
            "x": "delta_mu = raw change in average temperature relative to the control member",
            "y_left": "left shift = Q(0.50) - Q(0.05)",
            "y_right": "right shift = Q(0.95) - Q(0.50)",
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regress quantile shifts against raw change in average temperature.",
    )
    parser.add_argument(
        "--dataset",
        choices=["fhist", "lens2", "goga2"],
        nargs="+",
        help="Which ensemble(s) to load.",
    )
    parser.add_argument(
        "--variable",
        nargs="+",
        help="Variable name(s). Daily variable excluding frequency suffix (e.g. TREFHT).",
    )
    parser.add_argument(
        "--time-start",
        help="Start of the analysis period, format YYYY-MM or YYYY-MM-DD.",
    )
    parser.add_argument(
        "--time-stop",
        help="End of the analysis period, format YYYY-MM or YYYY-MM-DD.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        nargs="+",
        help="Length(s) of window in days for calculating maximum running mean.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for hatching. Default: 0.05",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run, do not save any figures or NetCDF files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        print("=== DRY RUN ===")

    for dataset in args.dataset:
        for variable in args.variable:
            for window_days in args.window_days:

                print(f"\n--- {dataset.upper()} {variable.upper()} {window_days}d ---\n")

                name = f"{dataset.upper()}_{variable}_pool_{window_days}d_window_{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}"

                fig_subdir = FIG_PATH / name
                fig_subdir.mkdir(parents=True, exist_ok=True)
                proc_dir = OUTPUT_PATH / f"{variable}_pool_{window_days}d_window"
                proc_dir.mkdir(parents=True, exist_ok=True)
                print(f"Saving figures to: {fig_subdir}")
                print(f"Saving regression output to: {proc_dir}")

                stats_anom, stats_raw = load_stats_datasets(
                    dataset,
                    variable,
                    window_days,
                    args.time_start,
                    args.time_stop,
                )

                grid = (
                    GRID[dataset]
                    .reindex_like(stats_raw, method="nearest", tolerance=1e-3)
                    .sel(lat=LAT_BNDS)
                )

                if "LANDAREA" in grid:
                    mask = regmask.defined_regions.ar6.land.mask(grid.lon, grid.lat)
                    mask = xr.where((mask == 0) & (grid.LANDAREA > 0), False, grid.LANDAREA > 0)
                else:
                    mask = True

                delta_mu = (stats_raw.mu - stats_raw.mu.isel(member=0)).where(mask)
                left_shift = (stats_anom.quantiles.sel(quantile=0.50) - stats_anom.quantiles.sel(quantile=0.05)).where(mask)
                right_shift = (stats_anom.quantiles.sel(quantile=0.95) - stats_anom.quantiles.sel(quantile=0.50)).where(mask)

                print("Running OLS regression per gridcell...")
                left_ds = ols_against_member_axis(left_shift, delta_mu)
                right_ds = ols_against_member_axis(right_shift, delta_mu)

                regression_ds = xr.concat([left_ds, right_ds], dim="shift").assign_coords(shift=SHIFT_LABELS)
                regression_da = regression_ds.to_array(dim="stat").sel(stat=STAT_LABELS)
                regression_da.name = variable
                regression_da = regression_da.assign_attrs(
                    description="OLS regression statistics for quantile shift against delta_mu",
                    units="",
                    x="delta_mu = raw change in average temperature over relative to the control member",
                    y="left_shift = ΔQ(0.50) - ΔQ(0.05); right_shift = ΔQ(0.95) - ΔQ(0.50)",
                    regression_dim="member",
                    significance_alpha=args.alpha,
                    window_days=window_days,
                )

                out_file = proc_dir / f"{name}_ols_regression.nc"
                if not args.dry_run:
                    regression_da.to_netcdf(out_file)
                print(out_file)

                print("Plotting regression slope maps")
                slopes = regression_da.sel(stat="slope").sel(lat=MAP_LAT_BNDS)
                p_values = regression_da.sel(stat="p_value").sel(lat=MAP_LAT_BNDS)
                shift_values = slopes.coords["shift"].values
                vmin, vmax = share_colorbar_limits([slopes.sel(shift=s) for s in shift_values], sym=True)

                fg = slopes.plot.pcolormesh(
                    col="shift",
                    col_wrap=2,
                    subplot_kws={"projection": PROJECTION},
                    figsize=(10, 4),
                    transform=ccrs.PlateCarree(),
                    cmap=CMAPS["slope"],
                    vmin=vmin,
                    vmax=vmax,
                    add_labels=False,
                    cbar_kwargs={
                        "orientation": "horizontal",
                        "fraction": 0.05,
                        "pad": 0.05,
                        "label": f"OLS slope [\u00B0C/\u00B0C]; hatching is significant at $\\alpha=${args.alpha}",
                    },
                )

                for i, ax in enumerate(fg.axs.flat[: slopes.sizes["shift"]]):
                    shift_name = str(shift_values[i])
                    ax.set_title(SHIFT_PLOT_LABELS[i], fontsize=10)
                    ax.set_extent([-180, 180, LAT_BNDS.start, LAT_BNDS.stop], crs=ccrs.PlateCarree())
                    ax.coastlines(color="k", lw=0.7)
                    add_gridlines(ax)
                    add_significance_hatching(p_values.sel(shift=shift_name), ax, alpha=args.alpha)

                if not args.dry_run:
                    fg.fig.savefig(fig_subdir / "map_regression_slopes.png", dpi=200, bbox_inches="tight")
                print(fig_subdir / "map_regression_slopes.png")
                plt.close()


if __name__ == "__main__":
    main()

