"""
Create the clustered LAI-aridity plots.
"""


from __future__ import annotations

import math
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import scipy.stats as stats
import xclimate as xclim

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Circle


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
    members: list[int],
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
    Create a multi-panel binned mean plot for a subset of FHIST ensemble members.

    Generates a figure with subplots (6 columns x N rows, N sized to fit `members`)
    showing binned mean values for the selected ensemble members, with overlaid circles
    representing sample sizes. The color intensity indicates the magnitude of the binned
    mean, while circle size and linewidth represent the sample count in each bin.

    Parameters
    ----------
    bm : xarray.DataArray
        Binned mean values with dimensions including 'member', 'ibx', and 'iby'.
    jh : xarray.DataArray
        Sample count (joint histogram) data with dimensions including 'member', 'ibx', and 'iby'.
    n_xy_bins : int
        Number of bins along each axis. Used to set tick positions and labels.
    members : list[int]
        Subset of member IDs to plot, in the order they should appear. Must all be
        present in `bm.member` and `jh.member`.
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
    - Empty subplots (beyond the number of selected members) are removed.
    - `ncols` is fixed at 6; `nrows` is computed from `len(members)` so the grid is no
      larger than needed.
    """

    bm = bm.sel(member=members).compute()
    jh = jh.sel(member=members).compute()

    if vabs is not None:
        vmin = -vabs
        vmax = vabs

    scaling_power = 0.4  # use power scaling to draw out bins with lower sample size
    ref_count = ref_count if ref_count is not None else round(jh.max().values.item() / 1e2, 0) * 1e2
    radius_scalar = 0.4
    lw_scalar = 0.3
    line_alpha = 0.5

    ncols = 6
    nrows = math.ceil(len(members) / ncols)

    # Scale height with row count, anchored to original 5-row proportions
    fig_height = 13 * nrows / 5
    if nrows == 1:
        fig_height += 1
    if nrows == 2:
        fig_height += 0.25

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        sharex=True,
        sharey=True,
        figsize=(14, fig_height),  
        subplot_kw=dict(box_aspect=1),
        constrained_layout=True,
        **kwargs
    )
    ax = np.atleast_1d(axs).flatten()

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
            fontsize=10,
            fontweight="bold",
            loc="center",
        )
        ax[i].set_xlim(0, len(bm.x_bin))
        ax[i].set_ylim(0, len(bm.y_bin))

        ax[i].set_yticks(np.arange(n_xy_bins) + 0.5)
        ax[i].set_xticks(np.arange(n_xy_bins) + 0.5)
        ax[i].tick_params(length=0)  # remove tick marks

        ticklabels = np.full((n_xy_bins), "", dtype=object)

        ax[i].set_yticklabels(ticklabels, rotation=35, ha="right", va="center")
        ax[i].set_xticklabels(ticklabels, rotation=35, ha="center", va="top")

        if i % ncols == 0:
            ax[i].set_ylabel(f"{y_label}", fontsize=14)
        if i // ncols == nrows - 1:
            ax[i].set_xlabel(f"{x_label}", fontsize=14)

    # # Add colorbar below all subplots
    # if vmin == 0:
    #     extend = "max"
    # else:
    #     extend = "both"
    # cbar = fig.colorbar(
    #     pcm,
    #     ax=axs[-ncols:],
    #     orientation="horizontal",
    #     extend=extend,
    #     fraction=0.05,
    #     shrink=0.9,
    #     pad=0.025,
    # )
    # cbar.set_label(target_label, fontsize=14)
    # cbar.ax.tick_params(labelsize=12)

    for i in range(len(members), ncols * nrows):
        ax[i].remove()

    # # Create custom legend for member categories
    # categories = {}
    # for m in bm.member.values:
    #     cat = xclim.ppe.get_member_cat_name(m)
    #     color = xclim.ppe.get_member_cat_color(m)
    #     if cat not in categories:
    #         categories[cat] = color

    # legend_handles = [
    #     Patch(facecolor=color, label=cat) for cat, color in categories.items()
    # ]

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

    # fig.canvas.draw()

    # cbar_pos = cbar.ax.get_position()

    # if signif_dict is not None:
    #     alpha = signif_dict.get('alpha', 'alpha')
    #     n_min = signif_dict.get('n_min', 'n_min')
    #     fig.text(
    #         (cbar_pos.x0 + cbar_pos.x1) / 2,
    #         cbar_pos.y0 - 0.05,
    #         f"$Significance$: $two$-$sided$ $t$-$test$ $\\alpha = {alpha}$, $n \\geq {n_min}$",
    #         ha="center",
    #         va="top",
    #         fontsize=10,
    #         transform=fig.transFigure,
    #     )

    # if add_circles:
    #     scale_counts = [ref_count * 0.25, ref_count * 0.5, ref_count]

    #     scale_x = cbar_pos.x1 + 0.05
    #     scale_y_base = (cbar_pos.y0 + cbar_pos.y1) / 2

    #     max_radius_data = radius_scalar * ((scale_counts[-1] / ref_count) ** scaling_power)
    #     point_data = np.array([[0, 0], [max_radius_data, 0]])
    #     point_display = ax[0].transData.transform(point_data)
    #     point_figure = fig.transFigure.inverted().transform(point_display)
    #     max_radius_fig = np.linalg.norm(point_figure[1] - point_figure[0])

    #     spacing = max_radius_fig * 2.5

    #     for idx, scale_count in enumerate(scale_counts):
    #         data_radius = radius_scalar * ((scale_count / ref_count) ** scaling_power)
    #         scale_lw = lw_scalar + ((scale_count / ref_count) ** scaling_power)

    #         point_data = np.array([[0, 0], [data_radius, 0]])
    #         point_display = ax[0].transData.transform(point_data)
    #         point_figure = fig.transFigure.inverted().transform(point_display)
    #         scale_radius = np.linalg.norm(point_figure[1] - point_figure[0])

    #         scale_y = scale_y_base + (idx - 1) * spacing

    #         scale_circle = Circle(
    #             (scale_x, scale_y),
    #             scale_radius,
    #             fill=False,
    #             edgecolor="black",
    #             linewidth=scale_lw,
    #             alpha=0.6,
    #             clip_on=False,
    #             transform=fig.transFigure,
    #         )
    #         fig.add_artist(scale_circle)

    #         fig.text(
    #             scale_x + max_radius_fig + 0.01,
    #             scale_y,
    #             f"{int(scale_count):,d}",
    #             ha="left",
    #             va="center",
    #             fontsize=9,
    #             transform=fig.transFigure,
    #         )

    #     fig.text(
    #         scale_x - max_radius_fig,
    #         scale_y_base + idx * spacing + 0.005,
    #         "Bin sample size:",
    #         ha="left",
    #         va="center",
    #         fontsize=9,
    #         transform=fig.transFigure,
    #     )

    return fig, axs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot clustered binned-mean plots."
    )
    parser.add_argument(
        "name",
        type=str,
        help=(
            "File descriptor "
            "(e.g., EFLX_LH_TOT_month_1.diff.binned_by.TLAI_month_1_x_AI_clim_1.15_x_15.pooled.AI_inv.199501-201412)"
        ),
    )
    parser.add_argument(
        "--n-clusters",
        nargs="+",
        type=int,
        help="Number(s) of clusters",
    )
    parser.add_argument(
        "--abs",
        action="store_true",
        help="Use the absolute value of the images."
    )
    return parser.parse_args()


PROC_DIR = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/proc/qbin")
FIG_DIR = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/fig/qbin")

ALPHA = 0.05
N_MIN = 10


def main() -> None:
    args = parse_args()

    abs_tag = ""
    if args.abs:
        abs_tag = "abs"

    name = args.name

    proc_path = PROC_DIR / name 
    fig_dir = FIG_DIR / name / "clustered"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load binned mean
    bin_stats = xr.open_dataset(proc_path / f"{name}.nc")["EFLX_LH_TOT"]
    print(bin_stats.member.values)

    for nc in args.n_clusters:
        print(f"Number of clusters: {nc}")
        cluster = xr.open_dataset(proc_path / f"{name}.{nc}_{abs_tag}cluster_labels.nc")["EFLX_LH_TOT_cluster"]

        for i in range(nc):
            ic = cluster.where(cluster == i, drop=True)["member"].values
            print(f"cluster {i}: {ic}")

            signif_mask = test_significance(bin_stats, alpha=ALPHA, n_min=N_MIN)
            fig, axs = plot_binned_mean(
                bm=bin_stats.sel(stats='mean').where(signif_mask),
                jh=bin_stats.sel(stats='count'),
                n_xy_bins=15,
                members=ic,
                target_label='$\\Delta$ET [W m$^{-2}$]',
                x_label='Aridity Index $\\rightarrow$',
                y_label='Leaf Area Index $\\rightarrow$',
                cmap='BrBG',
                vmin=-8,
                vmax=8,
                signif_dict={'alpha': ALPHA, 'n_min': N_MIN},
                add_circles=True,
                ref_count=50_000,
            )

            cluster_fname = f"{nc}_{abs_tag}clusters_{i}.png"
            fig.savefig(fig_dir / cluster_fname, dpi=200)
            print(fig_dir / cluster_fname)


if __name__ == "__main__":
    main()


