"""Batch plotting of zonal means."""

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import utils_moved_to_xclimate


USE_DASK = False
START_TMEAN = "1995-01"
END_TMEAN = "2014-12"
TIME_SLICE = slice(START_TMEAN,END_TMEAN)
VARIABLES = [
    "RAIN_FROM_ATM",
    "FSH",
    "EFLX_LH_TOT",
    "FCTR",
    "FCEV",
    "FGEV",
    "TLAI",
    "TSA",
    "FIRE",
    "FLDS",
    "FSR",
    "FSDS",
]


def center_axis_at_zero(ax, axis='y'):
    """
    Center the specified axis around zero by making the limits symmetric.
    
    Parameters:
    ax : matplotlib axis object
        The axis to modify
    axis : str
        Which axis to center ('x' or 'y')
    """
    if axis == 'y':
        ymin, ymax = ax.get_ylim()
        max_abs = max(abs(ymin), abs(ymax))
        ax.set_ylim(-max_abs, max_abs)
    elif axis == 'x':
        xmin, xmax = ax.get_xlim()
        max_abs = max(abs(xmin), abs(xmax))
        ax.set_xlim(-max_abs, max_abs)


def facetgrid_line(
    da: xr.DataArray,
    dim: str,
    da2: xr.DataArray | None = None,
    da2_kwargs: dict | None = None,
    ncol: int = 6,
    figsize=(14, 8),
    show_outer_labels: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ylabel2: str | None = None,
    hspace: float = 0.25,
    **kwargs,
):
    fg = da.plot.line(col=dim, col_wrap=ncol, **kwargs)

    fig = fg.fig
    fig.set_size_inches(*figsize)

    # --- collect valid axes in row-major order ---
    axes = [ax for ax in fg.axs.ravel() if ax is not None]
    names = [name for name in fg.name_dicts.ravel() if name is not None]

    # --- center y-axis + custom titles ---
    for ax, nd in zip(axes, names):
        center_axis_at_zero(ax)
        ax.axhline(lw=0.8, c="k", zorder=0)

        val = nd[dim]
        try:
            val_item = val.item()
        except Exception:
            val_item = val
        ax.set_title(utils_moved_to_xclimate.get_member_name(val_item), fontsize=8)

        if da2 is not None:
            ax2 = ax.twinx()
            da2_sel = da2.sel({dim: val})
            
            _da2_kwargs = da2_kwargs if da2_kwargs is not None else {}
            da2_sel.plot.line(ax=ax2, **_da2_kwargs)
            ax2.set_title("")
            ax2.set_xlabel("")
            ax2.set_ylabel("")
            center_axis_at_zero(ax2)

    # --- axis label handling ---
    if not show_outer_labels:
        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")
    else:
        nrows, ncols = fg.axs.shape

        for r in range(nrows):
            for c in range(ncols):
                ax = fg.axs[r, c]
                if ax is None:
                    continue

                is_left = (c == 0)
                is_right = (c == ncols - 1) or (fg.axs[r, c+1] is None)
                is_bottom = (r == nrows - 1) or (np.all(fg.axs[r+1, :] == None))

                ax.set_xlabel("")
                ax.set_ylabel("")
                if da2 is not None:
                    ax.right_ax.set_ylabel("")

                if is_bottom and (xlabel is not None):
                    ax.set_xlabel(xlabel)
                if is_left and (ylabel is not None):
                    ax.set_ylabel(ylabel)
                if da2 is not None and is_right and (ylabel2 is not None):
                    ax.right_ax.set_ylabel(ylabel2)
                
    fig.subplots_adjust(hspace=hspace, bottom=0.1)

    return fg


if USE_DASK:
    client, cluster = utils_moved_to_xclimate.create_dask_cluster(
        account="UWAS0155",
        nworkers=5,
    )

# for v in VARIABLES:
#     print(v)
#     x = utils.load_var(v, "month_1", "lnd", stream="h0")
#     lx = x.sel(time=TIME_SLICE).mean(dim=["time", "lon"])
#     dlx = lx.sel(member=slice(1,27)) - lx.sel(member=0)

#     fg = facetgrid_line(dlx, dim="member", label=f"pert - default: {v} [{x.units}]")
#     fg.fig.suptitle(f"pert - default: {v} [{x.units}]")
#     plt.tight_layout()
#     plt.savefig(f"fig/zm_lnd_1995-2015_{v}.png", dpi=300, bbox_inches="tight")

for v1, v2 in zip(["EFLX_LH_TOT", "EFLX_LH_TOT"], ["TLAI", "TSA"]):
    x1 = utils_moved_to_xclimate.load_var(v1, "month_1", "lnd", stream="h0")
    lx1 = x1.sel(time=TIME_SLICE).mean(dim=["time", "lon"])
    dlx1 = lx1.sel(member=slice(1,27)) - lx1.sel(member=0)

    x2 = utils_moved_to_xclimate.load_var(v2, "month_1", "lnd", stream="h0")
    lx2 = x2.sel(time=TIME_SLICE).mean(dim=["time", "lon"])
    dlx2 = lx2.sel(member=slice(1,27)) - lx2.sel(member=0)

    fg = facetgrid_line(dlx1, "member", dlx2, label=f"pert - default: {v1} [{x1.units}]", ylabel2=f"pert - default: {v2} [{x2.units}]", da2_kwargs={"color": "tab:green"})
    fg.fig.suptitle(f"pert - default: x-axis {v1} [{x1.units}] , y-axis {v2} [{x2.units}]")
    plt.tight_layout()
    plt.savefig(f"fig/zm_lnd_1995-2015_{v1}_{v2}.png", dpi=300, bbox_inches="tight")

if USE_DASK:
    utils_moved_to_xclimate.close_dask_cluster(client, cluster)
