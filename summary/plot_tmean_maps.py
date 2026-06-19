"""Batch plotting of maps."""

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import utils_moved_to_xclimate

USE_DASK = True
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


def plot_facetgrid(
    da: xr.DataArray,
    dim: str,
    label: str,
    ncol: int = 6,
    figsize=(16, 8),
    projection=ccrs.Robinson(),
    show_outer_labels: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    hspace: float = 0.25,
    **kwargs,
):
    """
    Faceted map panels.

    Parameters
    ----------
    da : xr.DataArray
        Must be at least 2D over lon/lat and have coordinate `dim` for faceting.
    dim : str
        Dimension name to facet over (columns and wrapped rows).
    cmap : matplotlib colormap
        Colormap for pcolormesh.
    label : str
        Colorbar label.
    ncol : int
        Number of facet columns (via col_wrap).
    figsize : tuple
        Figure size in inches.
    projection : cartopy.crs or None
        Map projection for subplots.
    show_outer_labels : bool
        If True, only show x/y labels on the outer edge panels; else remove all.
    xlabel, ylabel : str or None
        Optional axis labels for the outer panels if show_outer_labels=True.
    hspace : float
        Vertical spacing between subplot rows (default: 0.25).

    Returns
    -------
    fg : xarray.plot.FacetGrid
    """
    if projection:
        transform = ccrs.PlateCarree()
        subplot_kws = {"projection": projection}
    else:
        transform = None
        subplot_kws = None

    # --- create facets without xarray colorbar ---
    fg = da.plot.pcolormesh(
        col=dim,
        col_wrap=ncol,
        transform=transform,
        subplot_kws=subplot_kws,
        add_colorbar=True,
        cbar_kwargs={
            "label": label,
            "location": "bottom",
            "orientation": "horizontal",
            "fraction": 0.03,
            "pad": 0.15,
            "extend": "both",
        },
        **kwargs,
    )

    fig = fg.fig
    fig.set_size_inches(*figsize)

    # --- collect valid axes in row-major order ---
    axes = [ax for ax in fg.axs.ravel() if ax is not None]
    names = [name for name in fg.name_dicts.ravel() if name is not None]

    # --- add coastlines + custom titles ---
    for ax, nd in zip(axes, names):
        if projection:
            ax.coastlines(color="k", lw=0.8)

        val = nd[dim]
        try:
            val_item = val.item()
        except Exception:
            val_item = val
        ax.set_title(utils_moved_to_xclimate.get_member_name(val_item), fontsize=8)

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
                is_bottom = (r == nrows - 1)

                ax.set_xlabel("")
                ax.set_ylabel("")

                if is_bottom and (xlabel is not None):
                    ax.set_xlabel(xlabel)
                if is_left and (ylabel is not None):
                    ax.set_ylabel(ylabel)

    # --- layout ---
    fig.subplots_adjust(hspace=hspace, bottom=0.1)
    # fig.subplots_adjust(hspace=hspace, right=0.92)
    # fig.tight_layout(rect=[0, 0, 0.95, 1])

    return fg


if USE_DASK:
    client, cluster = utils_moved_to_xclimate.create_dask_cluster(
        account="UWAS0155",
        nworkers=5,
    )

for v in VARIABLES:
    print(v)
    x = utils_moved_to_xclimate.load_var(v, "month_1", "lnd", stream="h0")
    tx = x.sel(time=TIME_SLICE).mean(dim="time")
    dtx = tx.sel(member=slice(1,None)) - tx.sel(member=0)
    fg = plot_facetgrid(dtx, dim="member", label=f"pert - default: {v} [{x.units}]", cmap="RdBu", robust=True, center=0)
    plt.savefig(f"map_tmean_1995-2015_{v}.png", dpi=300, bbox_inches="tight")


if USE_DASK:
    utils_moved_to_xclimate.close_dask_cluster(client, cluster)
