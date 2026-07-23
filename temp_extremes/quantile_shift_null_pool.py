"""
Compute the pairwise null pool using the GOGA2 ensemble.

Δ_jk(p) := Q_j(p) - Q_k(p)

Δ(0.50)
L = Δ(0.50) - Δ(0.05)
R = Δ(0.95) - Δ(0.50)
S = R - L

Load GOGA2 ensemble
Iterate through pairs of members, skipping same member pairings
Compute Δ(0.50), L, R, S
Save all to a single dataset as indv variables
    (pair, lat, lon) -> (90, 192, 288)
    include coordinate of string descriptions (e.g., "01-02", "05-01")
"""

from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs


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

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

UNITS = "\u00B0C"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def share_colorbar_limits(da_list, sym):
    shared_vmin = xr.concat(da_list, dim="_panel").quantile(0.05, skipna=True)
    shared_vmax = xr.concat(da_list, dim="_panel").quantile(0.95, skipna=True)
    if sym:
        shared_vabs = max([abs(shared_vmin), abs(shared_vmax)])
        return -shared_vabs, shared_vabs
    return shared_vmin, shared_vmax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the pairwise null pool from a large ensemble.",
    )
    parser.add_argument(
        "--dataset",
        choices=["lens2", "goga2", "goga2_2deg", "cmip6_fhist"],
        help="Which ensemble to load.",
    )
    parser.add_argument(
        "--variable",
        help="Variable name. Daily variable excluding frequency suffix (e.g. TREFMXAV)."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for window_days in args.window_days:

        print(f"\n--- {args.dataset.upper()} {args.variable.upper()} {window_days}d ---\n")

        proc_dir = OUTPUT_PATH / f"{args.variable}_pool_{window_days}d_window"
        name = f"{args.dataset.upper()}_{args.variable}_pool_{window_days}d_window_{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}"
        
        stats_anom = xr.open_dataset(proc_dir / f"{name}_anom_stats.nc").compute()

        descr = []
        qdelta_median = []
        qdelta_left_shift = []
        qdelta_right_shift = []
        qdelta_symmetry = []
    
        for j in stats_anom.member.values:
            for k in stats_anom.member.values:
                if j != k:
                    print(f"{int(j):02d}-{int(k):02d}")

                    qd = stats_anom.quantiles.sel(member=j) - stats_anom.quantiles.sel(member=k)
                    if "member" in qd.coords:
                        qd = qd.drop_vars("member")

                    qd_left_shift = qd.sel(quantile=0.50) - qd.sel(quantile=0.05)
                    qd_right_shift = qd.sel(quantile=0.95) - qd.sel(quantile=0.50)
                    qd_symmetry = qd_right_shift - qd_left_shift

                    descr.append(f"{int(j):02d}-{int(k):02d}")
                    qdelta_median.append(qd.sel(quantile=0.50))
                    qdelta_left_shift.append(qd_left_shift)
                    qdelta_right_shift.append(qd_right_shift)
                    qdelta_symmetry.append(qd_symmetry)
        
        descr_unique = []
        for i, j in enumerate(stats_anom.member.values):
            for k in stats_anom.member.values[i+1:]:
                descr_unique.append(f"{int(j):02d}-{int(k):02d}")
                
        extra_coords = dict(pair=np.arange(len(descr)), pair_name=("pair", descr))
        attrs = dict(
            delta="Δ_jk(p) := Q_j(p) - Q_k(p)",
            variable=args.variable,
            window_days=window_days,
            dataset=args.dataset,
        )
        
        null_stats = xr.Dataset(
            data_vars=dict(
                delta_median=xr.concat(qdelta_median, dim="pair").assign_coords(extra_coords).assign_attrs(description="Δ(0.50)", **attrs),
                left_shift=xr.concat(qdelta_left_shift, dim="pair").assign_coords(extra_coords).assign_attrs(description="Δ(0.50) - Δ(0.05)", **attrs),
                right_shift=xr.concat(qdelta_right_shift, dim="pair").assign_coords(extra_coords).assign_attrs(description="Δ(0.95) - Δ(0.50)", **attrs),
                symmetry=xr.concat(qdelta_symmetry, dim="pair").assign_coords(extra_coords).assign_attrs(description="R - L; L = Δ(0.50) - Δ(0.05), R = Δ(0.95) - Δ(0.50)", **attrs),
            ),
        )
        
        null_stats.to_netcdf(proc_dir / f"{name}_null.nc")
        print(proc_dir / f"{name}_null.nc")


        # ------------------------------------------------------------------
        # Compute the statistics from the null pool
        # ------------------------------------------------------------------
        delta_median_max_goga = null_stats.delta_median.max(dim="pair")
        left_shift_max_goga = null_stats.left_shift.max(dim="pair")
        right_shift_max_goga = null_stats.right_shift.max(dim="pair")
        symmetry_max_goga = null_stats.symmetry.max(dim="pair")

        delta_median_std_goga = null_stats.delta_median.std(dim="pair")
        qshift_left_shift_std_goga = null_stats.left_shift.std(dim="pair")
        qshift_right_shift_std_goga = null_stats.right_shift.std(dim="pair")
        qshift_symmetry_std_goga = null_stats.symmetry.std(dim="pair")


        # ------------------------------------------------------------------
        # Plot quantile shift statistics from the null pool
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(
            2, 4, figsize=(20, 4), layout="constrained",
            subplot_kw={"projection": PROJECTION},
        )
        axs = axes.ravel()

        xr_subplot_kwargs = dict(vmin=0, robust=True, cmap="inferno", cbar_kwargs={"shrink": 0.95, "extend": "max"}, add_labels=False, transform=ccrs.PlateCarree())

        shift_max_vmin, shift_max_vmax = share_colorbar_limits([left_shift_max_goga, right_shift_max_goga], sym=False)
        shift_std_vmin, shift_std_vmax = share_colorbar_limits([qshift_left_shift_std_goga, qshift_right_shift_std_goga], sym=False)

        delta_median_max_goga.plot(ax=axs[0], **xr_subplot_kwargs)
        left_shift_max_goga.plot(ax=axs[1], vmax=shift_max_vmax, **xr_subplot_kwargs)
        right_shift_max_goga.plot(ax=axs[2], vmax=shift_max_vmax, **xr_subplot_kwargs)
        symmetry_max_goga.plot(ax=axs[3], **xr_subplot_kwargs)

        delta_median_std_goga.plot(ax=axs[4], **xr_subplot_kwargs)
        qshift_left_shift_std_goga.plot(ax=axs[5], vmax=shift_std_vmax, **xr_subplot_kwargs)
        qshift_right_shift_std_goga.plot(ax=axs[6], vmax=shift_std_vmax, **xr_subplot_kwargs)
        qshift_symmetry_std_goga.plot(ax=axs[7], **xr_subplot_kwargs)

        axs[0].set_title("$\\Delta\\hat{Q}(0.50)$")
        axs[1].set_title("$L = \\Delta\\hat{Q}(0.50) - \\Delta\\hat{Q}(0.05)$")
        axs[2].set_title("$R = \\Delta\\hat{Q}(0.95) - \\Delta\\hat{Q}(0.50)$")
        axs[3].set_title("$S = R - L$")

        fig.text(
            -0.0025, 0.7,
            "maximum",
            rotation=90,
            ha="center", va="center",
            fontsize=12
        )
        fig.text(
            -0.0025, 0.23,
            "standard deviation",
            rotation=90,
            ha="center", va="center",
            fontsize=12
        )

        for ax in axs:
            ax.coastlines(color="k", lw=0.8)
            ax.set_extent((-180, 180, -58, 90), crs=PROJECTION)
            ax.gridlines(
                draw_labels=False,
                xlocs=np.arange(-180, 181, 30),
                ylocs=np.linspace(LAT_BNDS.start, LAT_BNDS.stop, 6),
                linewidth=0.5,
                color="gray",
                alpha=0.6,
                linestyle="--",
            )

        fig.suptitle(f"{args.dataset.replace("_", " ").upper()}, {window_days}-day window, {args.variable} [{UNITS}], pairwise null pool (n={len(null_stats.pair)})", fontweight="bold")

        subdir = FIG_PATH / name
        subdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(subdir / "map_null_pool_stats.png", dpi=200, bbox_inches="tight")
        print(subdir / "map_null_pool_stats.png")
        plt.close()


if __name__ == "__main__":
    main()
