"""
Plots:
------
- maps of window center
    - absolute
    - difference relative to control
    - ensemble range for each gridcell
- map of raw TAVG
- map of raw and anom T95p
- map of ratio (dT95 / dTAVG)
- for select gridcells:
    - line of anom KDE for each member: line color = TAVG
    - line of raw KDE for each member: line color = TAVG

Load doy_climatology.nc, doy_window.nc, stats.nc, and kde.nc
"""

from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import xarray as xr
import xclimate as xclim

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean.cm as cmo
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
    "cyclic": "twilight",
    "range": "magma_r",
    "temp": "inferno",
    "temp_diff": "RdBu_r",
}

COLORS = {
    "fhist": "tab:blue",
    "goga2": "indianred",
    "lens2": "tab:orange",
}

GRID = {
    "fhist": xclim.load_fhist_ppe_grid(),
    "lens2": xclim.load_cesm2le_grid(),
    "goga2": xclim.load_goga2_grid(),
}

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

UNITS = "\u00B0C"

GRIDCELLS = {
    "ngp": dict(lon=263, lat=50, method="nearest"),
    "eus": dict(lon=280, lat=39, method="nearest"),
    "amz": dict(lon=295, lat=0, method="nearest"),
    "nru": dict(lon=120, lat=63, method="nearest"),
    "ukr": dict(lon=42, lat=50, method="nearest"),
    "cng": dict(lon=22, lat=0, method="nearest"),
    "gui": dict(lon=140, lat=-5, method="nearest"),
    "aus": dict(lon=137, lat=-22, method="nearest"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cyclic_diff(a, b, period=365):
    return ((a - b) + period/2) % period - period / 2


def share_colorbar_limits(da_list, sym):
    shared_vmin = xr.concat(da_list, dim="_panel").quantile(0.05, skipna=True)
    shared_vmax = xr.concat(da_list, dim="_panel").quantile(0.95, skipna=True)
    if sym:
        shared_vabs = max([abs(shared_vmin), abs(shared_vmax)])
        return -shared_vabs, shared_vabs
    return shared_vmin, shared_vmax


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


def add_null_pool_hatching(null_pool_stat, da, ax, transform=ccrs.PlateCarree(), hatch="///"):
    """Overlay hatching where the data exceeds the null-pool bound."""
    significant = xr.where(abs(da) > null_pool_stat, 1, np.nan)
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
    

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------

def load_datasets(
    dataset: str,
    variable: str,
    window_days: int,
    kind: str,
    time_start: str,
    time_stop: str,
) -> tuple[xr.Dataset, ...]:
    """Load all datasets."""
    clim_dir = OUTPUT_PATH / "doy_climatology"
    proc_dir = OUTPUT_PATH / f"{variable}_pool_{window_days}d_window"
    fname = f"{dataset.upper()}_{variable}_pool_{window_days}d_window_{time_start.replace('-', '')}-{time_stop.replace('-', '')}"

    if kind == "anom":
        kind_str = "_anom"
    elif kind == "raw":
        kind_str = ""
    else:
        raise ValueError("`kind` must be either 'anom' or 'raw'.")

    try:
        clim = xr.open_dataset(clim_dir / f"{fname}_doy_climatology.nc").sel(lat=LAT_BNDS).compute()
    except OSError:
        clim = None
    
    try:
        window = xr.open_dataset(clim_dir / f"{fname}_doy_window.nc").sel(lat=LAT_BNDS).compute()
    except OSError:
        window = None

    try:
        stats = xr.open_dataset(proc_dir / f"{fname}{kind_str}_stats.nc").sel(lat=LAT_BNDS).compute()
    except OSError:
        stats = None

    try:
        kde = xr.open_dataset(proc_dir / f"{fname}{kind_str}_kde.nc").sel(lat=LAT_BNDS).compute()
    except OSError:
        kde = None

    return window, clim, stats, kde


def load_null_stats(
    dataset: str,
    variable: str,
    window_days: str,
    time_start: str,
    time_stop: str,
) -> xr.Dataset:
    """Load the statistics from the null pool."""
    proc_dir = OUTPUT_PATH / f"{variable}_pool_{window_days}d_window"
    fname = f"{dataset.upper()}_{variable}_pool_{window_days}d_window_{time_start.replace('-', '')}-{time_stop.replace('-', '')}"
    return xr.open_dataset(proc_dir / f"{fname}_null.nc").sel(lat=LAT_BNDS).compute()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plots of temperature distribution statistics.",
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
        help="Variable name(s). Daily variable excluding frequency suffix (e.g. TREFMXAV)."
    )
    parser.add_argument(
        "--time-start",
        help="Start of the analysis period, format YYYY-MM or YYYY-MM-DD."
    )
    parser.add_argument(
        "--time-stop",
        help="End of the analysis period, format YYYY-MM or YYYY-MM-DD."
    )
    parser.add_argument(
        "--window-days",
        type=int,
        nargs="+",
        help="Length(s) of window in days for calculating maximum running mean."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run, do not save any figures."
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

                subdir = FIG_PATH / name
                subdir.mkdir(parents=True, exist_ok=True)
                print(f"Saving figures to: {subdir}")

                window, clim, stats_anom, kde_anom = load_datasets(
                    dataset,
                    variable,
                    window_days,
                    "anom",
                    args.time_start,
                    args.time_stop,
                )

                _, _, stats_raw, kde_raw = load_datasets(
                    dataset,
                    variable,
                    window_days,
                    "raw",
                    args.time_start,
                    args.time_stop,
                )

                goga_null = load_null_stats(
                    "goga2_2deg",
                    variable,
                    window_days,
                    args.time_start,
                    args.time_stop,
                )
                if (goga_null.lat != clim.lat).all():
                    raise ValueError("latitude coordinate does not match between `goga_null` and `clim`")
                if (goga_null.lon != clim.lon).all():
                        raise ValueError("longitude coordinate does not match between `goga_null` and `clim`")

                # Get member names/labels
                if dataset == "fhist":
                    member_names = xclim.ppe.get_member_name(clim.member)
                else:
                    member_names = np.arange(len(clim.member))
                    member_names = [f"mem{int(m):02d}" for m in member_names]
                print(member_names)

                # Load grid, then mask oceans and Greenland
                grid = (
                    GRID[dataset]
                    .reindex_like(clim, method="nearest", tolerance=1e-3)
                    .sel(lat=LAT_BNDS)
                )
                
                if "LANDAREA" in grid:
                    mask = regmask.defined_regions.ar6.land.mask(grid.lon, grid.lat)
                    mask = xr.where((mask == 0) & (grid.LANDAREA > 0), False, grid.LANDAREA > 0)
                else:
                    mask = True


                # ------------------------------------------------------------------
                # Computations
                # ------------------------------------------------------------------

                # Compute the central day of the n-day window and range in the central day
                window_center = window.max_window_doy.sel(window_day=window_days // 2).where(mask)
                window_center_delta = cyclic_diff(window_center, window_center.isel(member=0))
                window_center_range = window_center_delta.max(dim="member")  # min(dim="member") is zero

                # Compute the change in mean
                delta_mu = stats_raw.mu - stats_raw.mu.isel(member=0)

                # Compute the change in quantiles between ensemble members
                qdelta = stats_anom.quantiles - stats_anom.quantiles.isel(member=0)

                # Compute the per-member quantile shift
                qshift_left = stats_anom.quantiles.sel(quantile=0.50) - stats_anom.quantiles.sel(quantile=0.05)
                qshift_right = stats_anom.quantiles.sel(quantile=0.95) - stats_anom.quantiles.sel(quantile=0.50)

                # Compute the quantile shift delta
                qdelta_shift_left = qdelta.sel(quantile=0.50) - qdelta.sel(quantile=0.05)
                qdelta_shift_right = qdelta.sel(quantile=0.95) - qdelta.sel(quantile=0.50)
                qdelta_symmetry = qdelta_shift_right - qdelta_shift_left

                # Compute the null pool statistics
                qdelta_median_max_goga = goga_null.delta_median.max(dim="pair")
                qdelta_left_shift_max_goga = goga_null.left_shift.max(dim="pair")
                qdelta_right_shift_max_goga = goga_null.right_shift.max(dim="pair")
                qdelta_symmetry_max_goga = goga_null.symmetry.max(dim="pair")

                # Compute the ratios of tail metrics (0.05 and 0.95 quantiles) to center metrics (mean)
                p_mu_ratio = qdelta / delta_mu

                # ------------------------------------------------------------------
                # Plot thumbnail maps of window center
                # ------------------------------------------------------------------
                print("Plotting window center")
                da = window_center.sel(lat=MAP_LAT_BNDS)
                fg = da.plot.pcolormesh(
                    col="member",
                    col_wrap=MAP_SUBPLOTS_KWARGS[dataset].get("ncols", 4),
                    subplot_kws=MAP_SUBPLOTS_KWARGS[dataset]["subplot_kw"],
                    figsize=MAP_SUBPLOTS_KWARGS[dataset]["figsize"],
                    transform=ccrs.PlateCarree(),
                    cmap=CMAPS["cyclic"],
                    add_labels=False,
                    cbar_kwargs={
                        "orientation": "horizontal",
                        "fraction": 0.05,
                        "pad": 0.025,
                        "label": f"Center of {window_days}-day window [day of year]",
                        "ticks": np.arange(30, 365, 30),
                    },
                )

                for i, ax in enumerate(fg.axs.flat[:len(da.member)]):
                    ax.set_title(member_names[i], fontsize=10)
                    ax.set_extent([-180, 180, LAT_BNDS.start, LAT_BNDS.stop], crs=ccrs.PlateCarree())
                    ax.coastlines(color="k", lw=0.7)

                # Remove extra subplots
                for i in range(1, len(fg.axs.flat) % len(window_center.member) + 1):
                    fg.axs.flat[-i].remove()

                if not args.dry_run:
                    fg.fig.savefig(subdir / "map_window_center_allmembers.png", dpi=200, bbox_inches="tight")
                print(subdir / "map_window_center_allmembers.png")
                plt.close()


                # ------------------------------------------------------------------
                # Plot thumbnail maps of change in window center
                # ------------------------------------------------------------------
                print("Plotting delta window center")
                da = window_center_delta.sel(lat=MAP_LAT_BNDS)
                fg = da.plot.pcolormesh(
                    col="member",
                    col_wrap=MAP_SUBPLOTS_KWARGS[dataset].get("ncols", 4),
                    subplot_kws=MAP_SUBPLOTS_KWARGS[dataset]["subplot_kw"],
                    figsize=MAP_SUBPLOTS_KWARGS[dataset]["figsize"],
                    transform=ccrs.PlateCarree(),
                    cmap="PiYG",
                    vmin=-15, vmax=15,
                    add_labels=False,
                    cbar_kwargs={
                        "orientation": "horizontal",
                        "fraction": 0.05,
                        "pad": 0.025,
                        "label": f"$\\Delta$(Center of {window_days}-day window) [day of year]",
                    },
                )

                for i, ax in enumerate(fg.axs.flat[:len(da.member)]):
                    ax.set_title(member_names[i], fontsize=10)
                    ax.set_extent([-180, 180, LAT_BNDS.start, LAT_BNDS.stop], crs=ccrs.PlateCarree())
                    ax.coastlines(color="k", lw=0.7)

                # Remove extra subplots
                for i in range(1, len(fg.axs.flat) % len(window_center.member) + 1):
                    fg.axs.flat[-i].remove()

                if not args.dry_run:
                    fg.fig.savefig(subdir / "map_delta_window_center_allmembers.png", dpi=200, bbox_inches="tight")
                print(subdir / "map_delta_window_center_allmembers.png")
                plt.close()


                # ------------------------------------------------------------------
                # Plot zonal mean window center
                # ------------------------------------------------------------------
                print("Plotting zonal mean window center")
                da = window_center.mean(dim="lon")
                fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
                for m in da.member:
                    da.sel(member=m).plot(ax=ax, color=COLORS[dataset], alpha=0.5)

                ax.set_xlim(-58, 90)
                ax.set_ylim(0, 365)
                ax.set_yticks(np.arange(30, 365, 30))
                ax.grid(linewidth=0.5, color="gray", alpha=0.6, linestyle="--")
                ax.set_title(f"{dataset.upper()}, Zonal mean of {window_days}-day window center")
                ax.set_ylabel(f"Center of {window_days}-day\nwindow [day of year]")
                ax.set_xlabel("Latitude [\u00B0N]")

                if not args.dry_run:
                    fig.savefig(subdir / "line_zonal_window_center.png", dpi=200, bbox_inches="tight")
                print(subdir / "line_zonal_window_center.png")
                plt.close()

                
                # ------------------------------------------------------------------
                # Plot window center range
                # ------------------------------------------------------------------
                print("Plotting range of window center across ensemble")
                fig, ax = plt.subplots(**MAP_SUBPLOT_KWARGS)
                im = window_center_range.sel(lat=MAP_LAT_BNDS).plot(ax=ax, cmap=CMAPS["range"], vmin=0, vmax=50, transform=ccrs.PlateCarree(), add_colorbar=False)
                cb = fig.colorbar(im, ax=ax, extend="max", orientation="horizontal", fraction=0.08, pad=0.05)
                cb.set_label("Range of $\\Delta$(window center) [day of year]")
                ax.set_extent([-180, 180, LAT_BNDS.start, LAT_BNDS.stop], crs=ccrs.PlateCarree())
                ax.coastlines(color="k", lw=0.7)
                ax.set_title(f"{dataset.upper()}, Range of $\\Delta$({window_days}-day window center) across ensemble")

                if not args.dry_run:
                    fig.savefig(subdir / "map_window_range.png", dpi=200, bbox_inches="tight")
                print(subdir / "map_window_range.png")
                plt.close()


                # ------------------------------------------------------------------
                # Plot full summary and quantile shift
                # ------------------------------------------------------------------
                print("Plotting full quantile shift")

                subsubdir = subdir / f"map_quantile_shift"
                subsubdir.mkdir(parents=True, exist_ok=True)

                for i, m in enumerate(qdelta.member):
   
                    fig, axes = plt.subplots(
                        3, 4, figsize=(20, 6.2), layout="constrained",
                        subplot_kw={"projection": PROJECTION},
                    )
                    axs = axes.ravel()

                    xr_subplot_kwargs = dict(robust=True, cbar_kwargs={"shrink": 1, "extend": "both"}, add_labels=False, transform=ccrs.PlateCarree())

                    panel20 = qshift_left.sel(member=m)
                    panel21 = qshift_right.sel(member=m)
                    panel_20_21_vmin, panel_20_21_vmax = share_colorbar_limits([panel20, panel21], sym=False)

                    panel02 = qdelta.sel(quantile=0.50, member=m)
                    panel12 = qdelta.sel(quantile=0.05, member=m)
                    panel13 = qdelta.sel(quantile=0.95, member=m)
                    panel_02_12_13_vmin, panel_02_12_13_vmax = share_colorbar_limits([panel02, panel12, panel13], sym=True)

                    panel22 = qdelta_shift_left.sel(member=m)
                    panel23 = qdelta_shift_right.sel(member=m)
                    panel_22_23_vmin, panel_22_23_vmax = share_colorbar_limits([panel22, panel23], sym=True)

                    # Left block
                    stats_anom.quantiles.sel(quantile=0.50, member=m).plot(ax=axes[0,0], **xr_subplot_kwargs)
                    delta_mu.sel(member=m).plot(ax=axes[0,1], cmap="RdBu_r", **xr_subplot_kwargs)

                    stats_anom.quantiles.sel(quantile=0.05, member=m).plot(ax=axes[1,0], **xr_subplot_kwargs)
                    stats_anom.quantiles.sel(quantile=0.95, member=m).plot(ax=axes[1,1], **xr_subplot_kwargs)

                    panel20.plot(ax=axes[2,0], vmin=panel_20_21_vmin, vmax=panel_20_21_vmax, **xr_subplot_kwargs)
                    panel21.plot(ax=axes[2,1], vmin=panel_20_21_vmin, vmax=panel_20_21_vmax, **xr_subplot_kwargs)

                    # Right block
                    panel02.plot(ax=axes[0,2], vmin=panel_02_12_13_vmin, vmax=panel_02_12_13_vmax, cmap="RdBu_r", **xr_subplot_kwargs)
                    qdelta_symmetry.sel(member=m).plot(ax=axes[0,3], cmap="RdBu_r", **xr_subplot_kwargs)

                    panel12.plot(ax=axes[1,2], vmin=panel_02_12_13_vmin, vmax=panel_02_12_13_vmax, cmap="RdBu_r", **xr_subplot_kwargs)
                    panel13.plot(ax=axes[1,3], vmin=panel_02_12_13_vmin, vmax=panel_02_12_13_vmax, cmap="RdBu_r", **xr_subplot_kwargs)

                    panel22.plot(ax=axes[2,2], vmin=panel_22_23_vmin, vmax=panel_22_23_vmax, cmap="RdBu_r", **xr_subplot_kwargs)
                    panel23.plot(ax=axes[2,3], vmin=panel_22_23_vmin, vmax=panel_22_23_vmax, cmap="RdBu_r", **xr_subplot_kwargs)

                    axes[0,0].set_title(f"$\\hat{{Q}}_{{{int(m):d}}}(0.50)$")
                    axes[0,1].set_title(f"$\\Delta${variable.upper()}$_{{avg}}$")
                    axes[0,2].set_title(f"$\\Delta\\hat{{Q}}(0.50) = \\hat{{Q}}_{{{int(m):d}}}(0.50) - \\hat{{Q}}_{{0}}(0.50)$")
                    axes[0,3].set_title(f"$S = R - L$")

                    axes[1,0].set_title(f"$\\hat{{Q}}_{{{int(m):d}}}(0.05)$")
                    axes[1,1].set_title(f"$\\hat{{Q}}_{{{int(m):d}}}(0.95)$")
                    axes[1,2].set_title(f"$\\Delta\\hat{{Q}}0.05) = \\hat{{Q}}_{{{int(m):d}}}(0.05) - \\hat{{Q}}_{{0}}(0.05)$")
                    axes[1,3].set_title(f"$\\Delta\\hat{{Q}}(0.95) = \\hat{{Q}}_{{{int(m):d}}}(0.95) - \\hat{{Q}}_{{0}}(0.95)$")

                    axes[2,0].set_title(f"$\\hat{{Q}}_{{{int(m):d}}}(0.50) - \\hat{{Q}}_{{{int(m):d}}}(0.05)$")
                    axes[2,1].set_title(f"$\\hat{{Q}}_{{{int(m):d}}}(0.95) - \\hat{{Q}}_{{{int(m):d}}}(0.50)$")
                    axes[2,2].set_title(f"$L = \\Delta\\hat{{Q}}(0.50) - \\Delta\\hat{{Q}}(0.05)$")
                    axes[2,3].set_title(f"$R = \\Delta\\hat{{Q}}(0.95) - \\Delta\\hat{{Q}}(0.50)$")

                    for ax in axs:
                        ax.coastlines(color="k", lw=0.8)
                        ax.set_extent((-180, 180, -58, 90), crs=PROJECTION)
                        add_gridlines(ax)

                    fig.suptitle(f"{dataset.upper()}, {window_days}-day window, {variable} [{UNITS}], {member_names[i]}", fontweight="bold")

                    if not args.dry_run:
                        fig.savefig(subsubdir / f"full_member{int(m.data):02d}.png", dpi=200, bbox_inches="tight")
                    print(subsubdir / f"full_member{int(m.data):02d}.png")
                    plt.close()


                # ------------------------------------------------------------------
                # Plot condensed summary and quantile shift
                # ------------------------------------------------------------------
                print("Plotting condensed quantile shift")

                subsubdir = subdir / f"map_quantile_shift"
                subsubdir.mkdir(parents=True, exist_ok=True)

                for i, m in enumerate(qdelta.member):

                    fig, axes = plt.subplots(
                        2, 4, figsize=(18, 4), layout="constrained",
                        subplot_kw={"projection": PROJECTION},
                    )
                    axs = axes.ravel()

                    xr_subplot_kwargs = dict(robust=True, cbar_kwargs={"shrink": 0.95, "extend": "both"}, add_labels=False, transform=ccrs.PlateCarree())

                    qdelta_panels = [
                        qdelta.sel(quantile=0.05, member=m),
                        qdelta.sel(quantile=0.50, member=m),
                        qdelta.sel(quantile=0.95, member=m),
                    ]
                    qdelta_vmin, qdelta_vmax = share_colorbar_limits(qdelta_panels, sym=True)

                    qdelta_shift_panels = [
                        qdelta_shift_left.sel(member=m),
                        qdelta_shift_right.sel(member=m),
                    ]
                    qdelta_shift_vmin, qdelta_shift_vmax = share_colorbar_limits(qdelta_shift_panels, sym=True)

                    qdelta.sel(quantile=0.50, member=m).plot(ax=axs[0], vmin=qdelta_vmin, vmax=qdelta_vmax, cmap="RdBu_r", **xr_subplot_kwargs)
                    qdelta.sel(quantile=0.05, member=m).plot(ax=axs[1], vmin=qdelta_vmin, vmax=qdelta_vmax, cmap="RdBu_r", **xr_subplot_kwargs)
                    qdelta.sel(quantile=0.95, member=m).plot(ax=axs[2], vmin=qdelta_vmin, vmax=qdelta_vmax, cmap="RdBu_r", **xr_subplot_kwargs)

                    delta_mu.sel(member=m).plot(ax=axs[4], **xr_subplot_kwargs)
                    qdelta_shift_left.sel(member=m).plot(ax=axs[5], vmin=qdelta_shift_vmin, vmax=qdelta_shift_vmax, cmap="RdBu_r", **xr_subplot_kwargs)
                    qdelta_shift_right.sel(member=m).plot(ax=axs[6], vmin=qdelta_shift_vmin, vmax=qdelta_shift_vmax, cmap="RdBu_r", **xr_subplot_kwargs)
                    qdelta_symmetry.sel(member=m).plot(ax=axs[7], **xr_subplot_kwargs)

                    add_null_pool_hatching(qdelta_left_shift_max_goga, qdelta_shift_left.sel(member=m), axs[5])
                    add_null_pool_hatching(qdelta_right_shift_max_goga, qdelta_shift_right.sel(member=m), axs[6])
                    add_null_pool_hatching(qdelta_symmetry_max_goga, qdelta_symmetry.sel(member=m), axs[7])

                    axs[0].set_title(f"$\\Delta\\hat{{Q}}(0.50) = \\hat{{Q}}_{{{int(m):d}}}(0.50) - \\hat{{Q}}_{{0}}(0.50)$")
                    axs[1].set_title(f"$\\Delta\\hat{{Q}}(0.05) = \\hat{{Q}}_{{{int(m):d}}}(0.05) - \\hat{{Q}}_{{0}}(0.05)$")
                    axs[2].set_title(f"$\\Delta\\hat{{Q}}(0.95) = \\hat{{Q}}_{{{int(m):d}}}(0.95) - \\hat{{Q}}_{{0}}(0.95)$")
                    axs[4].set_title(f"$\\Delta${variable.upper()}$_{{avg}}$")
                    axs[5].set_title(f"$L = \\Delta\\hat{{Q}}(0.50) - \\Delta\\hat{{Q}}(0.05)$")
                    axs[6].set_title(f"$R = \\Delta\\hat{{Q}}(0.95) - \\Delta\\hat{{Q}}(0.50)$")
                    axs[7].set_title(f"$S = R - L$")

                    for ax in axs:
                        ax.coastlines(color="k", lw=0.8)
                        ax.set_extent((-180, 180, -58, 90), crs=PROJECTION)
                        add_gridlines(ax)

                    axs[3].remove()

                    fig.suptitle(f"{dataset.upper()}, {window_days}-day window, {variable} [{UNITS}], {member_names[i]}", fontweight="bold")

                    if not args.dry_run:
                        fig.savefig(subsubdir / f"member{int(m.data):02d}.png", dpi=200, bbox_inches="tight")
                    print(subsubdir / f"member{int(m.data):02d}.png")
                    plt.close()

                

                # # -----------------------------------------------------------------------
                # # Plot indv maps of delta_T_YYp, delta_T_avg, and delta_T_YYp/delta_T_avg
                # # -----------------------------------------------------------------------
                # if dataset == "fhist":
                #     for q in QUANTILES:
                #         print(f"Plotting maps of individual member quantile: {q}")

                #         subsubdir = subdir / f"map_delta_anom_{int(q*100):02d}p_abs_avg"
                #         subsubdir.mkdir(parents=True, exist_ok=True)

                #         for i, m in enumerate(quantiles_anom_delta.member):
                #             print(f"Member {int(m.data):02d}")

                #             panel_0 = quantiles_anom_delta.where(mask).sel(member=m, quantile=q, lat=MAP_LAT_BNDS)
                #             panel_1 = mu_raw_delta.where(mask).sel(member=m, lat=MAP_LAT_BNDS)
                #             panel_2 = p_mu_ratio.where(mask).sel(member=m, quantile=q, lat=MAP_LAT_BNDS)

                #             shared_vmin = xr.concat([panel_0, panel_1], dim="_panel").quantile(0.025, skipna=True)
                #             shared_vmax = xr.concat([panel_0, panel_1], dim="_panel").quantile(0.975, skipna=True)
                #             shared_vabs = max([abs(shared_vmin), abs(shared_vmax)])

                #             fig, axes = plt.subplots(1, 3, figsize=(12, 3), layout="constrained", subplot_kw={"projection": ccrs.PlateCarree(central_longitude=12)})
                #             axs = axes.ravel()

                #             im0 = panel_0.plot.pcolormesh(ax=axs[0], transform=ccrs.PlateCarree(), vmin=-shared_vabs, vmax=shared_vabs, cmap="RdBu_r", add_colorbar=False, add_labels=False)
                #             im1 = panel_1.plot.pcolormesh(ax=axs[1], transform=ccrs.PlateCarree(), vmin=-shared_vabs, vmax=shared_vabs, cmap="RdBu_r", add_colorbar=False, add_labels=False)
                #             im2 = panel_2.where(abs(panel_2) < 3).plot.pcolormesh(ax=axs[2], transform=ccrs.PlateCarree(), vmin=-2, vmax=2, cmap="PiYG", add_colorbar=False, add_labels=False)
                #             # panel_2.plot.contour(ax=axs[2], transform=ccrs.PlateCarree(), levels=[1], colors="k", linewidths=0.4, linestyles="-", add_labels=False)
                #             # panel_2.plot.contour(ax=axs[2], transform=ccrs.PlateCarree(), levels=[-1], colors="k", linewidths=0.4, linestyles="--", add_labels=False)

                #             cb0 = fig.colorbar(im0, ax=axs[:2], orientation="horizontal", extend="both", fraction=0.07, pad=0.03)
                #             cb0.set_label(f"$\\Delta${variable} [{UNITS}]")

                #             cb2 = fig.colorbar(im2, ax=axs[2], orientation="horizontal", extend="both", fraction=0.07, pad=0.03)
                #             cb2.set_label(f"$\\Delta${variable}_anom_{int(q*100):02d}p/$\\Delta${variable}_abs_avg")
                #             cb2.ax.axvline(1, color="k", linewidth=0.8, linestyle="-")
                #             cb2.ax.axvline(-1, color="k", linewidth=0.8, linestyle="--")

                #             for ax in axs:
                #                 ax.coastlines(lw=0.7, color="k")

                #             axs[0].set_title(f"$\\Delta${variable}_anom_{int(q*100):02d}p")
                #             axs[1].set_title(f"$\\Delta${variable}_abs_avg")
                #             axs[2].set_title(f"$\\Delta${variable}_anom_{int(q*100):02d}p/$\\Delta${variable}_abs_avg")

                #             fig.suptitle(f"{dataset.upper()}, {member_names[i+1]} $-$ {member_names[0]}", y=0.94, x=0.5, ha="center", fontweight="bold")

                #             if not args.dry_run:
                #                 fig.savefig(subsubdir / f"member{int(m.data):02d}.png", dpi=200, bbox_inches="tight")
                #             print(subsubdir / f"member{int(m.data):02d}.png")
                #             plt.close()


                # # -----------------------------------------------------------------------
                # # Plot indv maps of delta_T_YYp, delta_T_50p, and delta_T_YYp/delta_T_50p
                # # -----------------------------------------------------------------------
                # if dataset == "fhist":
                #     for q in [0.05, 0.25, 0.75, 0.95]:
                #         print(f"Plotting maps of individual member quantile: {q}")

                #         subsubdir = subdir / f"map_delta_anom_{int(q*100):02d}p_abs_50p"
                #         subsubdir.mkdir(parents=True, exist_ok=True)

                #         for i, m in enumerate(p_50p_ratio.member):
                #             print(f"Member {int(m.data):02d}")

                #             panel_0 = quantiles_anom_delta.where(mask).sel(member=m, quantile=q, lat=MAP_LAT_BNDS)
                #             panel_1 = quantiles_raw_delta.where(mask).sel(member=m, quantile=0.50, lat=MAP_LAT_BNDS)
                #             panel_2 = p_50p_ratio.where(mask).sel(member=m, quantile=q, lat=MAP_LAT_BNDS)

                #             shared_vmin = xr.concat([panel_0, panel_1], dim="_panel").quantile(0.025, skipna=True)
                #             shared_vmax = xr.concat([panel_0, panel_1], dim="_panel").quantile(0.975, skipna=True)
                #             shared_vabs = max([abs(shared_vmin), abs(shared_vmax)])

                #             fig, axes = plt.subplots(1, 3, figsize=(12, 3), layout="constrained", subplot_kw={"projection": ccrs.PlateCarree(central_longitude=12)})
                #             axs = axes.ravel()

                #             im0 = panel_0.plot.pcolormesh(ax=axs[0], transform=ccrs.PlateCarree(), vmin=-shared_vabs, vmax=shared_vabs, cmap="RdBu_r", add_colorbar=False, add_labels=False)
                #             im1 = panel_1.plot.pcolormesh(ax=axs[1], transform=ccrs.PlateCarree(), vmin=-shared_vabs, vmax=shared_vabs, cmap="RdBu_r", add_colorbar=False, add_labels=False)
                #             im2 = panel_2.where(abs(panel_2) < 3).plot.pcolormesh(ax=axs[2], transform=ccrs.PlateCarree(), vmin=-2, vmax=2, cmap="PiYG", add_colorbar=False, add_labels=False)
                #             # panel_2.plot.contour(ax=axs[2], transform=ccrs.PlateCarree(), levels=[1], colors="k", linewidths=0.4, linestyles="-", add_labels=False)
                #             # panel_2.plot.contour(ax=axs[2], transform=ccrs.PlateCarree(), levels=[-1], colors="k", linewidths=0.4, linestyles="--", add_labels=False)

                #             cb0 = fig.colorbar(im0, ax=axs[:2], orientation="horizontal", extend="both", fraction=0.07, pad=0.03)
                #             cb0.set_label(f"$\\Delta${variable} [{UNITS}]")

                #             cb2 = fig.colorbar(im2, ax=axs[2], orientation="horizontal", extend="both", fraction=0.07, pad=0.03)
                #             cb2.set_label(f"$\\Delta${variable}_anom_{int(q*100):02d}p/$\\Delta${variable}_abs_50p")
                #             cb2.ax.axvline(1, color="k", linewidth=0.8, linestyle="-")
                #             cb2.ax.axvline(-1, color="k", linewidth=0.8, linestyle="--")

                #             for ax in axs:
                #                 ax.coastlines(lw=0.7, color="k")

                #             axs[0].set_title(f"$\\Delta${variable}_anom_{int(q*100):02d}p")
                #             axs[1].set_title(f"$\\Delta${variable}_abs_50p")
                #             axs[2].set_title(f"$\\Delta${variable}_anom_{int(q*100):02d}p/$\\Delta${variable}_abs_50p")

                #             fig.suptitle(f"{dataset.upper()}, {member_names[i+1]} $-$ {member_names[0]}", y=0.94, x=0.5, ha="center", fontweight="bold")

                #             if not args.dry_run:
                #                 fig.savefig(subsubdir / f"member{int(m.data):02d}.png", dpi=200, bbox_inches="tight")
                #             print(subsubdir / f"member{int(m.data):02d}.png")
                #             plt.close()


                # # ------------------------------------------------------------------
                # # Plot thumbnail maps of T_YYp
                # # ------------------------------------------------------------------
                # for q in QUANTILES:
                #     print(f"Plotting maps of all members quantile: {q}")

                #     da = stats_anom.quantiles.sel(quantile=q, lat=MAP_LAT_BNDS)
                #     fg = da.plot.pcolormesh(
                #         col="member",
                #         col_wrap=MAP_SUBPLOTS_KWARGS[dataset].get("ncols", 4),
                #         subplot_kws=MAP_SUBPLOTS_KWARGS[dataset]["subplot_kw"],
                #         figsize=MAP_SUBPLOTS_KWARGS[dataset]["figsize"],
                #         transform=ccrs.PlateCarree(),
                #         cmap=CMAPS["temp"],
                #         add_labels=False,
                #         cbar_kwargs={
                #             "orientation": "horizontal",
                #             "fraction": 0.05,
                #             "pad": 0.025,
                #             "label": f"{variable}_anom_{int(q*100):d}p [{UNITS}]",
                #         },
                #         robust=True,
                #     )

                #     for i, ax in enumerate(fg.axs.flat[:len(da.member)]):
                #         ax.set_title(member_names[i], fontsize=10)
                #         ax.set_extent([-180, 180, LAT_BNDS.start, LAT_BNDS.stop], crs=ccrs.PlateCarree())
                #         ax.coastlines(color="k", lw=0.7)

                #     # Remove extra subplots
                #     for i in range(1, len(fg.axs.flat) % len(stats_anom.member) + 1):
                #         fg.axs.flat[-i].remove()

                #     if not args.dry_run:
                #         fg.fig.savefig(subdir / f"map_anom_{int(q*100):02d}p_allmembers.png", dpi=200, bbox_inches="tight")
                #     print(subdir / f"map_anom_{int(q*100):02d}p_allmembers.png")
                #     plt.close()
                

                # ------------------------------------------------------------------
                # Plot thumbnail maps of delta_T_YYp
                # ------------------------------------------------------------------
                if dataset == "fhist":
                    for q in [0.05, 0.50, 0.95]:
                        print(f"Plotting maps of all members delta quantile: {q}")

                        da = qdelta.sel(quantile=q, lat=MAP_LAT_BNDS)
                        fg = da.plot.pcolormesh(
                            col="member",
                            col_wrap=MAP_SUBPLOTS_KWARGS[dataset].get("ncols", 4),
                            subplot_kws=MAP_SUBPLOTS_KWARGS[dataset]["subplot_kw"],
                            figsize=MAP_SUBPLOTS_KWARGS[dataset]["figsize"],
                            transform=ccrs.PlateCarree(),
                            cmap=CMAPS["temp_diff"],
                            add_labels=False,
                            cbar_kwargs={
                                "orientation": "horizontal",
                                "fraction": 0.05,
                                "pad": 0.025,
                                "label": f"$\\Delta${variable}_anom_{int(q*100):d}p [{UNITS}]",
                            },
                            robust=True,
                        )

                        for i, ax in enumerate(fg.axs.flat[:len(da.member)]):
                            ax.set_title(member_names[i], fontsize=10)
                            ax.set_extent([-180, 180, LAT_BNDS.start, LAT_BNDS.stop], crs=ccrs.PlateCarree())
                            ax.coastlines(color="k", lw=0.7)
                            add_gridlines(ax)

                        # Remove extra subplots
                        for i in range(1, len(fg.axs.flat) % len(qdelta.member) + 1):
                            fg.axs.flat[-i].remove()

                        if not args.dry_run:
                            fg.fig.savefig(subdir / f"map_delta_anom_{int(q*100):02d}p_allmembers.png", dpi=200, bbox_inches="tight")
                        print(subdir / f"map_delta_anom_{int(q*100):02d}p_allmembers.png")
                        plt.close()


                # # ------------------------------------------------------------------
                # # Plot thumbnail maps of T_avg
                # # ------------------------------------------------------------------
                # da = stats_raw.mu.sel(lat=MAP_LAT_BNDS) - 273.15
                # fg = da.plot.pcolormesh(
                #     col="member",
                #     col_wrap=MAP_SUBPLOTS_KWARGS[dataset].get("ncols", 4),
                #     subplot_kws=MAP_SUBPLOTS_KWARGS[dataset]["subplot_kw"],
                #     figsize=MAP_SUBPLOTS_KWARGS[dataset]["figsize"],
                #     transform=ccrs.PlateCarree(),
                #     cmap=CMAPS["temp"],
                #     add_labels=False,
                #     cbar_kwargs={
                #         "orientation": "horizontal",
                #         "fraction": 0.05,
                #         "pad": 0.025,
                #         "label": f"{variable}_abs_avg [{UNITS}]",
                #     },
                #     robust=True,
                # )

                # for i, ax in enumerate(fg.axs.flat[:len(da.member)]):
                #     ax.set_title(member_names[i], fontsize=10)
                #     ax.set_extent([-180, 180, LAT_BNDS.start, LAT_BNDS.stop], crs=ccrs.PlateCarree())
                #     ax.coastlines(color="k", lw=0.7)
                #     add_gridlines(ax)

                # # Remove extra subplots
                # for i in range(1, len(fg.axs.flat) % len(stats_raw.member) + 1):
                #     fg.axs.flat[-i].remove()

                # if not args.dry_run:
                #     fg.fig.savefig(subdir / f"map_abs_avg_allmembers.png", dpi=200, bbox_inches="tight")
                # print(subdir / f"map_abs_avg_allmembers.png")
                # plt.close()


                # ------------------------------------------------------------------
                # Plot thumbnail maps of delta T_avg
                # ------------------------------------------------------------------
                da = delta_mu.sel(lat=MAP_LAT_BNDS)
                fg = da.plot.pcolormesh(
                    col="member",
                    col_wrap=MAP_SUBPLOTS_KWARGS[dataset].get("ncols", 4),
                    subplot_kws=MAP_SUBPLOTS_KWARGS[dataset]["subplot_kw"],
                    figsize=MAP_SUBPLOTS_KWARGS[dataset]["figsize"],
                    transform=ccrs.PlateCarree(),
                    cmap=CMAPS["temp_diff"],
                    add_labels=False,
                    cbar_kwargs={
                        "orientation": "horizontal",
                        "fraction": 0.05,
                        "pad": 0.025,
                        "label": f"$\\Delta${variable}_avg [{UNITS}]",
                    },
                    robust=True,
                )

                for i, ax in enumerate(fg.axs.flat[:len(da.member)]):
                    ax.set_title(member_names[i], fontsize=10)
                    ax.set_extent([-180, 180, LAT_BNDS.start, LAT_BNDS.stop], crs=ccrs.PlateCarree())
                    ax.coastlines(color="k", lw=0.7)
                    add_gridlines(ax)

                # Remove extra subplots
                for i in range(1, len(fg.axs.flat) % len(delta_mu.member) + 1):
                    fg.axs.flat[-i].remove()

                if not args.dry_run:
                    fg.fig.savefig(subdir / "map_delta_abs_avg_allmembers.png", dpi=200, bbox_inches="tight")
                print(subdir / "map_delta_abs_avg_allmembers.png")
                plt.close()


                # # ------------------------------------------------------------------
                # # Plot thumbnail maps of delta T_YYp / delta T_avg
                # # ------------------------------------------------------------------
                # if dataset == "fhist":
                #     for q in QUANTILES:
                #         da = p_mu_ratio.sel(quantile=q, lat=MAP_LAT_BNDS)
                #         fg = da.plot.pcolormesh(
                #             col="member",
                #             col_wrap=MAP_SUBPLOTS_KWARGS[dataset].get("ncols", 4),
                #             subplot_kws=MAP_SUBPLOTS_KWARGS[dataset]["subplot_kw"],
                #             figsize=MAP_SUBPLOTS_KWARGS[dataset]["figsize"],
                #             transform=ccrs.PlateCarree(),
                #             add_labels=False,
                #             cbar_kwargs={
                #                 "orientation": "horizontal",
                #                 "fraction": 0.05,
                #                 "pad": 0.025,
                #                 "label": f"$\\Delta${variable}_anom_{int(q*100):02d}p/$\\Delta${variable}_abs_avg",
                #             },
                #             vmin=-2, vmax=2, cmap="PiYG",
                #         )

                #         for i, ax in enumerate(fg.axs.flat[:len(da.member)]):
                #             ax.set_title(member_names[i+1], fontsize=10)
                #             ax.set_extent([-180, 180, LAT_BNDS.start, LAT_BNDS.stop], crs=ccrs.PlateCarree())
                #             ax.coastlines(color="k", lw=0.7)

                #         # Remove extra subplots
                #         for i in range(1, len(fg.axs.flat) % len(p_mu_ratio.member) + 1):
                #             fg.axs.flat[-i].remove()

                #         if not args.dry_run:
                #             fg.fig.savefig(subdir / f"map_delta_anom_{int(q*100):02d}p_abs_avg_allmembers.png", dpi=200, bbox_inches="tight")
                #         print(subdir / f"map_delta_anom_{int(q*100):02d}p_abs_avg_allmembers.png")
                #         plt.close()
                

                # # ------------------------------------------------------------------
                # # Plot thumbnail maps of delta_T_YYp / delta_T_50p
                # # ------------------------------------------------------------------
                # if dataset == "fhist":
                #     for q in [0.05, 0.25, 0.75, 0.95]:
                #         da = p_50p_ratio.sel(quantile=q, lat=MAP_LAT_BNDS)
                #         fg = da.plot.pcolormesh(
                #             col="member",
                #             col_wrap=MAP_SUBPLOTS_KWARGS[dataset].get("ncols", 4),
                #             subplot_kws=MAP_SUBPLOTS_KWARGS[dataset]["subplot_kw"],
                #             figsize=MAP_SUBPLOTS_KWARGS[dataset]["figsize"],
                #             transform=ccrs.PlateCarree(),
                #             add_labels=False,
                #             cbar_kwargs={
                #                 "orientation": "horizontal",
                #                 "fraction": 0.05,
                #                 "pad": 0.025,
                #                 "label": f"$\\Delta${variable}_anom_{int(q*100):02d}p/$\\Delta${variable}_abs_50p",
                #             },
                #             vmin=-5, vmax=5,
                #             cmap="PiYG",
                #         )

                #         for i, ax in enumerate(fg.axs.flat[:len(da.member)]):
                #             ax.set_title(member_names[i+1], fontsize=10)
                #             ax.set_extent([-180, 180, LAT_BNDS.start, LAT_BNDS.stop], crs=ccrs.PlateCarree())
                #             ax.coastlines(color="k", lw=0.7)

                #         # Remove extra subplots
                #         for i in range(1, len(fg.axs.flat) % len(p_mu_ratio.member) + 1):
                #             fg.axs.flat[-i].remove()

                #         if not args.dry_run:
                #             fg.fig.savefig(subdir / f"map_delta_anom_{int(q*100):02d}p_abs_50p_allmembers.png", dpi=200, bbox_inches="tight")
                #         print(subdir / f"map_delta_anom_{int(q*100):02d}p_abs_50p_allmembers.png")
                #         plt.close()



if __name__ == "__main__":
    main()
