#!/glade/work/bbuchovecky/miniforge3/envs/data-sci-py312/bin/python3.12
"""
Compute 2-D binned statistics for FHIST variables.

This script mirrors the structure of `pm_pet.py` and loads a target variable
plus two binning variables, optionally forms a member-wise difference against a
reference member, and then computes 2-D bin statistics with
`compute_2d_bin_stats`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from distributed import wait

import numpy as np
import scipy.stats as stats
import xarray as xr
import xclimate as xclim

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import regionmask as regmask

from matplotlib.patches import Patch
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIME_START = "1995-01"
DEFAULT_TIME_STOP = "2014-12"
DEFAULT_FHIST_GRID = xclim.load_fhist_ppe_grid()
DEFAULT_PET = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/proc/pet/FHIST.rad_pet.annual_mean.195001-201412.nc")
DEFAULT_OUTPUT_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/proc/qbin")
DEFAULT_FIG_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/fig/qbin")

# Masking
LAT_BNDS = slice(-58, 90)

# Plotting
PROJECTION = ccrs.PlateCarree()
ALPHA = 0.05
N_MIN = 10

DEFAULT_ANALYSIS_CHUNKS = {
    "time": -1,
    "member": 1,
    "lat": 96,
    "lon": 144,
}
DEFAULT_ANALYSIS_CHUNKS_NOTIME = {
    "member": 1,
    "lat": 96,
    "lon": 144,
}


sys.path.insert(0, str(Path(__file__).parent))
from ppe_2d_binning import compute_2d_bin_stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _strip_bad_scalar_coords(da: xr.DataArray) -> xr.DataArray:
    """Remove scalar coords that can confuse downstream writes."""
    bad_scalar_coords = {"landunit", "column", "pft"}
    clean_coords = {k: v for k, v in da.coords.items() if k not in bad_scalar_coords}
    return xr.DataArray(da.variable, coords=clean_coords, name=da.name, attrs=da.attrs)


def load_fhist_variable(
    variable: str,
    time_start: str,
    time_stop: str,
    members: list[int] | None,
    lat_bnds: slice[float],
    grid: xr.DataArray | xr.Dataset,
    mask: xr.DataArray | None,
) -> xr.DataArray:
    """Load FHIST data and align it to the PPE grid."""
    time_slice = slice(time_start, time_stop)

    vv = "_".join(variable.split("_")[:-2])
    da = (
        xclim.load_fhist(variable, keep_var_only=True)[vv]
        .sel(time=time_slice, lat=lat_bnds)
        .reindex_like(grid, method="nearest", tolerance=1e-3)
    )

    if mask is not None:
        da = da.where(mask)

    if members is not None:
        da = da.sel(member=members)

    return _strip_bad_scalar_coords(da)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def compute_annual_mean(da):
    days_in_month = da.time.dt.days_in_month
    weights = days_in_month.groupby('time.year') / days_in_month.groupby('time.year').sum()
    with xr.set_options(keep_attrs=True):
        return (da * weights).groupby('time.year').sum()


def compute_aridity_index(
    precip: xr.DataArray,
    pet: xr.DataArray,
    clip: bool = False,
) -> xr.DataArray:
    """Compute the annual mean aridity index."""
    if clip:
        pet = pet.clip(min=0)
    ai = precip / pet
    ai = ai.rename("AI")
    ai.attrs = {
        "long_name": "aridity index (P/PET)",
        "description": "computed from annual mean P and PET{' (PET floor at zero)' if clip else ''}",
    }
    return ai


def compute_inverse_aridity_index(
    precip: xr.DataArray,
    pet: xr.DataArray,
    clip: bool = False,
) -> xr.DataArray:
    """Compute the annual mean inverse aridity index."""
    if clip:
        pet = pet.clip(min=0)
    inv_ai = pet / precip
    inv_ai = inv_ai.rename("AI")
    inv_ai.attrs = {
        "long_name": "inverse aridity index (PET/P)",
        "description": f"computed from annual mean P and PET{' (PET floor at zero)' if clip else ''}",
    }
    return inv_ai


def apply_target_operation(
    target: xr.DataArray,
    *,
    op: str,
    member_dim: str = "member",
    ref_member: int = 0,
) -> xr.DataArray:
    """Apply the requested target operation before binning."""
    if op == "none":
        return target

    if op != "diff":
        raise ValueError("`op` must be either 'none' or 'diff'.")

    if member_dim not in target.dims:
        raise ValueError("`diff` requires a target variable with a member dimension.")

    member_values = np.asarray(target[member_dim].values)
    if ref_member not in member_values:
        raise ValueError(f"Reference member {ref_member} is not present in `{member_dim}`.")

    member_values = member_values[member_values != ref_member]
    if member_values.size == 0:
        raise ValueError("`diff` leaves no members to bin after removing the reference member.")

    reference = target.sel({member_dim: ref_member})
    return target.sel({member_dim: member_values}) - reference


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def quick_map(da: xr.DataArray, fout: Path, robust: bool = True, vmin=0):
    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained", subplot_kw={"projection": PROJECTION})
    da.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", vmin=vmin, robust=robust)
    ax.coastlines(color="k", lw=0.8)
    ax.set_extent((-180, 180, LAT_BNDS.start, LAT_BNDS.stop), crs=PROJECTION)

    x_gls = np.arange(-180, 181, 30)
    y_gls = np.arange(-90, 91, 30)
    y_gls = y_gls[(y_gls >= LAT_BNDS.start) & (y_gls <= LAT_BNDS.stop)]

    ax.gridlines(
        draw_labels=False,
        xlocs=x_gls,
        ylocs=y_gls,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )
    fig.savefig(fout, dpi=120, bbox_inches="tight")


def test_significance(
    bin_stats: xr.DataArray,
    alpha: float = 0.05,
    n_min: int = 10,
) -> xr.DataArray:
    """
    Test binned means against H0: μ = 0.
    For signficance, the sample size must also be greater than `n_min`
    """
    means = bin_stats.sel(stats='mean')
    stds = np.sqrt(bin_stats.sel(stats='var_samp'))
    ns = bin_stats.sel(stats='count')

    # Calculate t-statistic and critical value
    t_critical = stats.t.ppf(1 - alpha / 2, ns - 1)
    t_statistic = means / (stds / np.sqrt(ns))

    # Perform significance test
    return (np.abs(t_statistic) > t_critical) & (bin_stats.sel(stats='count') > n_min)


def plot_binned_mean(
    bm,
    jh,
    n_xy_bins,
    target_label,
    x_label,
    y_label,
    cmap,
    vmin,
    vmax,
    vabs=None,
    signif_dict=None,
    add_circles=True,
    ref_count=None,
    **kwargs,
):
    """
    Create a multi-panel binned mean plot for FHIST ensemble members.

    Generates a figure with subplots (6 columns x 5 rows) showing binned mean values
    for each ensemble member, with overlaid circles representing sample sizes. The color
    intensity indicates the magnitude of the binned mean, while circle size and linewidth
    represent the sample count in each bin.

    Parameters
    ----------
    bm : xarray.DataArray
        Binned mean values with dimensions including 'member', 'ibx', and 'iby'.
    jh : xarray.DataArray
        Sample count (joint histogram) data with dimensions including 'member', 'ibx', and 'iby'.
    n_xy_bins : int
        Number of bins along each axis. Used to set tick positions and labels.
    target_name : str
        Name of the variable being shaded. Used for labeling.
    x_var : str
        Name of the variable on the x-axis. Used for labeling.
    y_var : str
        Name of the variable on the y-axis. Used for labeling.
    cmap : str
        Colormap for the plot.
    vmin, vmax : float
        Define the data range that the colormap covers.
    vabs : float, optional
        If provided, centers the colormap range around zero and sets vmin=-vabs, vmax=vabs.
    signif : dict, optional
        Dictionary containing the keys 'alpha' and 'n_min'. If provided, adds annotation under the
        colorbar describing statistical significance.
        -> alpha : float
            Significance level (critical alpha value) for the statistical test.
        -> n_min : int
            Minimum sample size threshold for significance.
    add_circles : bool, optional
        Add circles with radii that correspond to the bin sample size.
    ref_count : int, optional
        Reference count for the scale circles.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the multi-panel plot with colorbar, legend, and
        reference circle scale.
    matplotlib.axes.Axes
        Ndarray of axes objects for each subplot.

    Notes
    -----
    - Circle size and linewidth are scaled using a power law (scaling_power=0.4) to
      amplify visibility of bins with smaller sample sizes.
    - A reference scale of circles is displayed showing sample sizes at 25%, 50%, and 100%
      of the maximum sample count in the data.
    - Empty subplots (beyond the number of ensemble members) are removed.
    """

    bm = bm.compute()
    jh = jh.compute()

    if vabs is not None:
        vmin = -vabs
        vmax = vabs

    scaling_power = 0.4  # use power scaling to draw out bins with lower sample size
    ref_count = ref_count if ref_count is not None else round(jh.max().values.item() / 1e2, 0) * 1e2
    radius_scalar = 0.4
    lw_scalar = 0.3
    line_alpha = 0.5

    ncols = 6
    nrows = 5

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        sharex=True,
        sharey=True,
        figsize=(14, 13),
        subplot_kw=dict(box_aspect=1),
        constrained_layout=True,
        **kwargs
    )
    ax = axs.flatten()

    pcm = None
    for i, m in enumerate(bm.member.values):
        pcm = ax[i].pcolormesh(
            bm.sel(member=m), shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )

        if add_circles:
            m_jh = jh.sel(member=m).T
            for ii in range(len(jh.x_bin)):
                for jj in range(len(jh.y_bin)):
                    count = m_jh.values[ii, jj]
                    if count > 0:
                        radius = radius_scalar * ((count / ref_count) ** scaling_power)
                        lw = lw_scalar + ((count / ref_count) ** scaling_power)
                        circle = Circle(
                            (ii + 0.5, jj + 0.5),
                            radius,
                            fill=False,
                            edgecolor="black",
                            linewidth=lw,
                            alpha=line_alpha,
                        )
                        ax[i].add_patch(circle)

        ax[i].set_title(
            xclim.ppe.get_member_name(m),
            color=xclim.ppe.get_member_cat_color(m),
            # color="k",
            fontsize=10,
            fontweight="bold",
            loc="center",
        )
        ax[i].set_xlim(0, len(bm.x_bin))
        ax[i].set_ylim(0, len(bm.y_bin))

        ax[i].set_yticks(np.arange(n_xy_bins) + 0.5)
        ax[i].set_xticks(np.arange(n_xy_bins) + 0.5)
        ax[i].tick_params(length=0)  # remove tick marks

        # Add 'low' and 'high' labels to the axes
        ticklabels = np.full((n_xy_bins), "", dtype=object)
        # ticklabels[0] = "low"
        # ticklabels[-1] = "high"

        ax[i].set_yticklabels(ticklabels, rotation=35, ha="right", va="center")
        ax[i].set_xticklabels(ticklabels, rotation=35, ha="center", va="top")

        if i % ncols == 0:
            ax[i].set_ylabel(f"{y_label}", fontsize=14)
        if i // ncols == nrows - 1:
            ax[i].set_xlabel(f"{x_label}", fontsize=14)

        # Color the outer edge of the subplot corresponding to the parameter functional category
        # for spine in ax[i].spines.values():
        #     spine.set_color(xclim.ppe.get_member_cat_color(m))
        #     spine.set_linewidth(4)

    # Add colorbar below all subplots
    if vmin == 0:
        extend = "max"
    else:
        extend = "both"
    cbar = fig.colorbar(
        pcm,
        ax=axs,
        orientation="horizontal",
        extend=extend,
        fraction=0.025,
        shrink=0.9,
        pad=0.025,
    )
    cbar.set_label(target_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    for i in range(len(bm.member), ncols * nrows):
        ax[i].remove()

    # Create custom legend for member categories
    # Get unique categories and their colors
    categories = {}
    for m in bm.member.values:
        cat = xclim.ppe.get_member_cat_name(m)
        color = xclim.ppe.get_member_cat_color(m)
        if cat not in categories:
            categories[cat] = color

    # Create legend handles
    legend_handles = [
        Patch(facecolor=color, label=cat) for cat, color in categories.items()
    ]

    # # Add legend using figure coordinates
    # fig.legend(
    #     handles=legend_handles,
    #     ncols=3,
    #     loc="lower right",
    #     bbox_to_anchor=(0.98, 0.15),
    #     bbox_transform=fig.transFigure,
    #     frameon=True,
    #     fontsize=12,
    #     title="Functional Categories",
    #     title_fontproperties={
    #         "weight": "bold",
    #         "size": 12,
    #     },
    # )

    # Draw the figure to apply and fix layout
    fig.canvas.draw()

    # Get colorbar position
    cbar_pos = cbar.ax.get_position()

    if signif_dict is not None:
        # Add text below legend for alpha and n_min parameters
        alpha = signif_dict.get('alpha', 'alpha')
        n_min = signif_dict.get('n_min', 'n_min')
        fig.text(
            (cbar_pos.x0 + cbar_pos.x1) / 2,
            cbar_pos.y0 - 0.05,
            f"$Significance$: $two$-$sided$ $t$-$test$ $\\alpha = {alpha}$, $n \\geq {n_min}$",
            ha="center",
            va="top",
            fontsize=10,
            transform=fig.transFigure,
        )

    if add_circles:
        # Create three reference circles of different sizes
        scale_counts = [ref_count * 0.25, ref_count * 0.5, ref_count]

        # Position for the scale circles
        scale_x = cbar_pos.x1 + 0.05
        scale_y_base = (cbar_pos.y0 + cbar_pos.y1) / 2

        # Calculate spacing
        max_radius_data = radius_scalar * ((scale_counts[-1] / ref_count) ** scaling_power)
        point_data = np.array([[0, 0], [max_radius_data, 0]])
        point_display = ax[0].transData.transform(point_data)
        point_figure = fig.transFigure.inverted().transform(point_display)
        max_radius_fig = np.linalg.norm(point_figure[1] - point_figure[0])

        spacing = max_radius_fig * 2.5

        for idx, scale_count in enumerate(scale_counts):
            # Calculate radius with same scaling
            data_radius = radius_scalar * ((scale_count / ref_count) ** scaling_power)
            scale_lw = lw_scalar + ((scale_count / ref_count) ** scaling_power)

            # Convert to figure coordinates
            point_data = np.array([[0, 0], [data_radius, 0]])
            point_display = ax[0].transData.transform(point_data)
            point_figure = fig.transFigure.inverted().transform(point_display)
            scale_radius = np.linalg.norm(point_figure[1] - point_figure[0])

            # Position circles vertically with spacing
            scale_y = scale_y_base + (idx - 1) * spacing

            scale_circle = Circle(
                (scale_x, scale_y),
                scale_radius,
                fill=False,
                edgecolor="black",
                linewidth=scale_lw,
                alpha=0.6,
                clip_on=False,
                transform=fig.transFigure,
            )
            fig.add_artist(scale_circle)

            fig.text(
                scale_x + max_radius_fig + 0.01,
                scale_y,
                f"{int(scale_count):,d}",
                ha="left",
                va="center",
                fontsize=9,
                transform=fig.transFigure,
            )

        # Add label above reference circles
        fig.text(
            scale_x - max_radius_fig,
            scale_y_base + idx * spacing + 0.005,
            "Bin sample size:",
            ha="left",
            va="center",
            fontsize=9,
            transform=fig.transFigure,
        )

    return fig, axs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 2-D binned statistics for FHIST variables using "
            "compute_2d_bin_stats."
        ),
    )
    parser.add_argument(
        "--x-bin",
        required=True,
        help="Name of the x-axis binning variable, e.g. AI_month_1.",
    )
    parser.add_argument(
        "--y-bin",
        required=True,
        help="Name of the y-axis binning variable, e.g. TLAI_month_1.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Name of the target variable to bin, e.g. EFLX_LH_TOT_month_1.",
    )
    parser.add_argument(
        "--ai-options",
        nargs="+",
        type=str,
        default=(),
        help="Contains: clip PET at zero ('clip') and inverse aridity index ('inv')"
    )
    parser.add_argument(
        "--op",
        choices=["none", "diff"],
        default="none",
        help=(
            "Target operation before binning. "
            "'diff' subtracts the reference member from all other members."
        ),
    )
    parser.add_argument(
        "--ref-member",
        default=0,
        type=int,
        help="Reference member to subtract when --op diff is used. Default: 0.",
    )
    parser.add_argument(
        "--n-y-bins",
        default=15,
        type=int,
        help="Number of bins along the y axis. Default: 15.",
    )
    parser.add_argument(
        "--n-x-bins",
        default=15,
        type=int,
        help="Number of bins along the x axis. Default: 15.",
    )
    parser.add_argument(
        "--y-strategy",
        choices=["quantile", "linear"],
        default="quantile",
        help="Bin edge strategy for the y axis. Default: quantile.",
    )
    parser.add_argument(
        "--x-strategy",
        choices=["quantile", "linear"],
        default="quantile",
        help="Bin edge strategy for the x axis. Default: quantile.",
    )
    parser.add_argument(
        "--y-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Explicit y-axis bin range for linear binning.",
    )
    parser.add_argument(
        "--x-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Explicit x-axis bin range for linear binning.",
    )
    parser.add_argument(
        "--collapse-duplicate-quantile-bins",
        action="store_true",
        help=(
            "Collapse duplicate quantile edges caused by tied values. "
            "This can reduce the effective number of bins. "
            "Default is False."
        ),
    )
    parser.add_argument(
        "--pool-edges-across-ensemble",
        "--pool-edge-across-ensemble",
        action="store_true",
        dest="pool_edges_across_ensemble",
        help=(
            "Compute bin edges from the pooled ensemble rather than per member. "
            "Default is false."
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
        help="Optional list of member indices to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output directory for the NetCDF file. "
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
        help="Number of Dask workers (equal to ncores). Default: 2.",
    )
    parser.add_argument(
        "--dask-memory",
        default="16GB",
        type=str,
        help="Amount of memory for the Dask cluster. Default: 16GB.",
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


def main() -> None:
    args = parse_args()
    args.ai_options = tuple(args.ai_options)

    if args.output.exists() and not args.output.is_dir():
        raise ValueError("`--output` must be a directory.")
    
    # ------------------------------------------------------------------
    # Handle output path
    # ------------------------------------------------------------------
    pool_tag = ""
    if args.pool_edges_across_ensemble:
        pool_tag = ".pooled"

    dupl_tag = ""
    if args.collapse_duplicate_quantile_bins:
        dupl_tag = ".no_dupl_bins"
    
    ai_tag = ""
    if len(args.ai_options) == 1:
        ai_tag = f".AI_{args.ai_options[0]}"
    elif len(args.ai_options) > 1:
        ai_tag = f".AI_{'_'.join(args.ai_options)}"

    fstem = (
        f"{args.target}.{args.op}.binned_by.{args.y_bin}_x_{args.x_bin}"
        f".{args.n_y_bins}_x_{args.n_x_bins}{pool_tag}{dupl_tag}{ai_tag}"
        f".{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}"
    )
    print(f"Output file stem: {fstem}")

    out_path = args.output / fstem
    out_path.mkdir(parents=True, exist_ok=True)

    fig_path = DEFAULT_FIG_PATH / fstem
    fig_path.mkdir(parents=True, exist_ok=True)


    client_cluster = None
    if args.dask_cluster:
        client_cluster = xclim.create_dask_cluster(
            account="UWAS0155",
            nworkers=args.dask_workers,
            ncores=args.dask_workers,
            nmem=args.dask_memory,
            walltime=args.dask_walltime,
        )
        client_cluster[0].wait_for_workers(args.dask_workers)

    try:
        print("=== FHIST 2-D BINNING ===")

        # Get grid and create Greenland mask
        grid = xclim.load_fhist_ppe_grid()
        if "LANDAREA" in grid:
            mask = regmask.defined_regions.ar6.land.mask(grid.lon, grid.lat)
            mask = xr.where((mask == 0) & (grid.LANDAREA > 0), False, grid.LANDAREA > 0)
        else:
            mask = None
        
        quick_map(mask, fig_path / f"{fstem}.mask.png", robust=False)

        # ------------------------------------------------------------------
        # Load variables
        # ------------------------------------------------------------------
        variables = [args.target, args.y_bin, args.x_bin]
        das: dict[str, xr.DataArray] = {}

        for var in variables:
            print(f"Loading {var}...")

            # Compute aridity index (AI)
            if (var == "AI_clim_1") or (var == "AI_year_1"):
                precip = load_fhist_variable(
                    variable="PRECT_calculated_month_1",
                    time_start=args.time_start,
                    time_stop=args.time_stop,
                    members=args.members,
                    lat_bnds=LAT_BNDS,
                    grid=grid,
                    mask=mask,
                )
                pet = xr.open_dataset(DEFAULT_PET)["PET"].where(mask)

                print(f"precip units: {precip.attrs.get('units', 'no units')}")
                print(f"PET units: {pet.attrs.get('units', 'no units')}")

                if precip.attrs.get('units', 'no units') != "m/s":
                    raise ValueError("precip must be in units of m/s for conversion to W/m2")
                
                precip = precip * 1000 * 2.45e6
                precip.attrs["units"] = "W/m2"

                if "clim" in var:
                    precip = compute_annual_mean(precip).mean(dim="year")
                    pet = pet.mean(dim="year")
                if "year" in var:
                    precip = compute_annual_mean(precip).mean(dim="year")
                
                quick_map(precip.isel(member=0), fig_path / f"{fstem}.precip.png")
                quick_map(pet.isel(member=0), fig_path / f"{fstem}.pet.png")

                to_clip = False
                if "clip" in args.ai_options:
                    to_clip = True
                
                if "inv" in args.ai_options:
                    da = compute_inverse_aridity_index(precip, pet, clip=to_clip)
                else:
                    da = compute_aridity_index(precip, pet, clip=to_clip)

                quick_map(da.isel(member=0), fig_path / f"{fstem}.{var}.png")
                
            else:
                if "year" in var:
                    da = load_fhist_variable(
                        variable=var.replace("year", "month"),
                        time_start=args.time_start,
                        time_stop=args.time_stop,
                        members=args.members,
                        lat_bnds=LAT_BNDS,
                        grid=grid,
                        mask=mask,
                    )
                    da = compute_annual_mean(da).where(mask)

                else:
                    da = load_fhist_variable(
                        variable=var,
                        time_start=args.time_start,
                        time_stop=args.time_stop,
                        members=args.members,
                        lat_bnds=LAT_BNDS,
                        grid=grid,
                        mask=mask,
                    )
    
            # Diagnostic plot
            print(f"{var}: {da.dims} {da.shape}")
            if "time" in da.dims:
                quick_map(da.isel(time=0, member=0), fig_path / f"{fstem}.{var}.png")
            elif "year" in da.dims:
                quick_map(da.isel(year=0, member=0), fig_path / f"{fstem}.{var}.png")
            else:
                quick_map(da.isel(member=0), fig_path / f"{fstem}.{var}.png")

            # Persist the variable
            if client_cluster is not None:
                if "time" in da.dims:
                    da = client_cluster[0].persist(da.chunk(DEFAULT_ANALYSIS_CHUNKS))
                else:
                    da = client_cluster[0].persist(da.chunk(DEFAULT_ANALYSIS_CHUNKS_NOTIME))
                wait(da)

            das[var] = da
        

        print("Input arrays")
        for var, da in das.items():
            print(f"{var}: {da.dims} {da.shape}")


        # ------------------------------------------------------------------
        # Apply the target operation
        # ------------------------------------------------------------------
        print(f"Applying target operation: {args.op}")
        target = apply_target_operation(
            das[args.target],
            op=args.op,
            ref_member=args.ref_member,
        )

        if args.op == "diff":
            member_dim = "member"
            member_values = target[member_dim].values
            das[args.y_bin] = das[args.y_bin].sel({member_dim: member_values})
            das[args.x_bin] = das[args.x_bin].sel({member_dim: member_values})
        
        print(f"target: {target.dims}, {target.shape}")
        print(f"y-bin: {das[args.y_bin].dims}, {das[args.y_bin].shape}")
        print(f"x-bin: {das[args.x_bin].dims}, {das[args.x_bin].shape}")


        # ------------------------------------------------------------------
        # Reshape binning variables if necessary
        # ------------------------------------------------------------------
        if "time" in target.dims:
            expand_tcoord = "time"
        elif "year" in target.dims:
            expand_tcoord = "year"
        else:
            raise ValueError(f"target {args.target} must contain `time` or `year` dimension; instead has dimensions {target.dims}")

        if (das[args.y_bin].shape != target.shape):
            old_shape = das[args.y_bin].shape
            das[args.y_bin] = das[args.y_bin].expand_dims({expand_tcoord: target[expand_tcoord]})
            print(f"Expanding time dimension of y-bin {args.y_bin}: {old_shape} -> {das[args.y_bin].shape}")
        
        if (das[args.x_bin].shape != target.shape):
            old_shape = das[args.x_bin].shape
            das[args.x_bin] = das[args.x_bin].expand_dims({expand_tcoord: target[expand_tcoord]})
            print(f"Expanding time dimension of x-bin {args.x_bin}: {old_shape} -> {das[args.x_bin].shape}")
        

        # ------------------------------------------------------------------
        # Compute arrays
        # ------------------------------------------------------------------
        target = target.compute()
        das[args.y_bin] = das[args.y_bin].compute()
        das[args.x_bin] = das[args.x_bin].compute()


        # ------------------------------------------------------------------
        # Compute bin stats
        # ------------------------------------------------------------------
        print("Computing 2-D bin statistics...")
        bin_stats = compute_2d_bin_stats(
            target=target,
            y_var=das[args.y_bin],
            x_var=das[args.x_bin],
            member_dim="member" if "member" in target.dims else None,
            n_y_bins=args.n_y_bins,
            n_x_bins=args.n_x_bins,
            y_strategy=args.y_strategy,
            x_strategy=args.x_strategy,
            y_range=tuple(args.y_range) if args.y_range is not None else None,
            x_range=tuple(args.x_range) if args.x_range is not None else None,
            collapse_duplicate_quantile_bins=args.collapse_duplicate_quantile_bins,
            pool_edges_across_ensemble=args.pool_edges_across_ensemble,
            parallel=True,
        )

        bin_stats.attrs["mask"] = f"masked out Greenland and Antarctica"

        if args.op == "diff":
            bin_stats.attrs["target_operation"] = f"difference; subtracted member {args.ref_member} from all other members"


        # ------------------------------------------------------------------
        # Write output
        # ------------------------------------------------------------------
        bin_stats.to_netcdf(out_path / f"{fstem}.nc")
        print(f"Wrote {out_path / fstem}.nc")


        # ------------------------------------------------------------------
        # Create summary plot
        # ------------------------------------------------------------------
        signif_mask = test_significance(bin_stats, alpha=ALPHA, n_min=N_MIN)

        if args.op == "diff":
            vabs = max(
                [
                    abs(bin_stats.sel(stats='mean').where(signif_mask).quantile(0.025)),
                    abs(bin_stats.sel(stats='mean').where(signif_mask).quantile(0.975)),
                ]
            )
            vmin = -vabs
            vmax = vabs
        else:
            vmin = 0
            vmax = bin_stats.sel(stats='mean').where(signif_mask).quantile(0.975)

        fig, axs = plot_binned_mean(
            bm=bin_stats.sel(stats='mean').where(signif_mask),
            jh=bin_stats.sel(stats='count'),
            n_xy_bins=15,
            target_label='$\\Delta$ET [W m$^{-2}$]',
            x_label='Aridity Index $\\rightarrow$',
            y_label='Leaf Area Index $\\rightarrow$',
            cmap="BrBG",
            vmin=vmin,
            vmax=vmax,
            signif_dict={'alpha': ALPHA, 'n_min': N_MIN},
            add_circles=True,
            ref_count=None,
            # ref_count=50_000,
            dpi=300,
        )
        fig.savefig(fig_path / f"{fstem}.png", dpi=200)

    finally:
        if client_cluster is not None:
            xclim.close_dask_cluster(client_cluster, remove_std_files=False)


if __name__ == "__main__":
    main()