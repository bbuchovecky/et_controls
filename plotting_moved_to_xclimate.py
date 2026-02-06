"""Plotting utilities."""

from typing import Tuple, List, Optional, Hashable

import numpy as np
import xarray as xr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import xclimate as xclim

def whiten(color, alpha):
    """
    Linearly mix a color with white.

    Parameters
    ----------
    color : str or tuple
        Any matplotlib color spec.
    alpha : float
        Whitening fraction in [0, 1].

    Returns
    -------
    tuple
        RGB tuple in [0, 1].
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - alpha) * rgb + alpha)


def center_axis_at_zero(ax, axis="y"):
    """
    Center the specified axis around zero by making the limits symmetric.

    Parameters:
    ax : matplotlib axis object
        The axis to modify
    axis : str
        Which axis to center ('x', 'y', or 'both')
    """
    if axis == "y":
        ymin, ymax = ax.get_ylim()
        max_abs = max(abs(ymin), abs(ymax))
        ax.set_ylim(-max_abs, max_abs)

    elif axis == "x":
        xmin, xmax = ax.get_xlim()
        max_abs = max(abs(xmin), abs(xmax))
        ax.set_xlim(-max_abs, max_abs)

    elif axis == "both":
        ymin, ymax = ax.get_ylim()
        max_abs = max(abs(ymin), abs(ymax))
        ax.set_ylim(-max_abs, max_abs)

        xmin, xmax = ax.get_xlim()
        max_abs = max(abs(xmin), abs(xmax))
        ax.set_xlim(-max_abs, max_abs)

    else:
        raise ValueError("axis must be 'x', 'y', or 'both'")


def plot_fancy_timeseries(
    das: List[xr.DataArray],
    das_weights: List[xr.DataArray],
    das_labels: List[str],
    ylabel: str,
    xlabel: str = "Year",
    title: str = "",
    member_coord: str | List[str] = "member",
    colors: Optional[List] = None,
    ylim: Tuple = (None, None),
    xlim: Tuple = (1948, 2035),
    highlight_member: Optional[int | str | List] = None,
    time_mean_period: slice = slice("1995-01", "2014-12"),
    violin_settings: Optional[List] = None,
) -> Tuple:
    """
    Plot ensemble timeseries with violin plots showing distribution of time-mean values.
    This function creates a visualization of multiple ensemble timeseries with individual
    ensemble members shown as transparent lines, optional highlighted members, and violin
    plots showing the distribution of time-mean values for a specified period.
    Parameters
    ----------
    das : List[xr.DataArray]
        List of xarray DataArrays containing the timeseries data to plot. Each DataArray
        should have dimensions including time, lat, lon, and a member dimension.
    das_weights : List[xr.DataArray]
        List of weight arrays corresponding to each DataArray in `das`, used for weighted
        spatial averaging.
    das_labels : List[str]
        Labels for each dataset to be shown in the legend.
    ylabel : str
        Label for the y-axis.
    xlabel : str, optional
        Label for the x-axis. Default is "Year".
    title : str, optional
        Title for the plot. Default is an empty string.
    member_coord : str or List[str], optional
        Name(s) of the ensemble member coordinate dimension. Can be a single string applied
        to all datasets or a list of strings for each dataset. Default is "member".
    colors : Optional[List], optional
        List of colors to use for each dataset. Must have at least as many colors as datasets.
        If None, uses matplotlib's TABLEAU_COLORS. Default is None.
    ylim : Tuple, optional
        Y-axis limits as (min, max). Default is (None, None).
    xlim : Tuple, optional
        X-axis limits as (min, max). Default is (1898, 2035).
    highlight_member : Optional[int | str | List], optional
        Ensemble member(s) to highlight with a bold line. Can be a single value applied to
        all datasets or a list of values for each dataset. Default is None.
    time_mean_period : slice, optional
        Time period over which to compute the mean for the violin plots. Default is
        slice("1995-01", "2014-12").
    violin_settings : Optional[List], optional
        List of dictionaries containing violin plot settings for each dataset. Each dictionary
        can contain 'marker', 'facecolor', and 'edgecolor' keys. Default is None.
    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        A tuple containing the figure and axes objects.
    Notes
    -----
    - The function performs weighted spatial averaging over lat/lon dimensions and annual
      averaging over the time dimension.
    - A shaded region indicates the time period used for computing the violin plot distributions.
    - Violin plots are positioned at x=2026 by default and show the distribution of ensemble
      members' time-mean values.
    - The function creates a twin y-axis for visual balance.
    Examples
    --------
    >>> fig, ax = plot_fancy_timeseries(
    ...     das=[da1, da2],
    ...     das_weights=[weights1, weights2],
    ...     das_labels=['Model A', 'Model B'],
    ...     ylabel='Temperature [degree C]',
    ...     title='Ensemble Temperature Timeseries'
    ... )
    """
    
    if colors is None:
        colors = list(mcolors.TABLEAU_COLORS.keys())
    
    assert len(das) == len(das_weights) and len(das) == len(das_labels)
    assert len(colors) >= len(das)

    n = len(das)
    member_colors = [whiten(c, 0.1) for c in colors]

    if isinstance(member_coord, str):
        member_coord = [member_coord] * n
    else:
        assert isinstance(member_coord, list) and len(member_coord) == n

    if highlight_member is not None:
        if isinstance(highlight_member, list):
            assert len(highlight_member) == n
        elif isinstance(highlight_member, int | str):
            highlight_member = [highlight_member] * n
    else:
        highlight_member = [None] * n

    if violin_settings is None or len(violin_settings) != n:
        violin_settings = [None] * n

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.axvspan(
        int(time_mean_period.start[:4]),
        int(time_mean_period.stop[:4]),
        alpha=0.25,
        facecolor="silver",
        edgecolor=None,
    )

    for da, weight, label, member, highlight, color, mem_color, vs in zip(
        das,
        das_weights,
        das_labels,
        member_coord,
        highlight_member,
        colors,
        member_colors,
        violin_settings,
    ):

        vp_xpos = vs.get("x", 2020) if vs is not None else 2020
        vp_marker = vs.get("marker", "o") if vs is not None else "o"
        vp_facecolor = vs.get("facecolor", color) if vs is not None else color
        vp_edgecolor = vs.get("edgecolor", color) if vs is not None else color

        # Do computation
        da_ts = da.weighted(weight).mean(dim=["lat", "lon"]).groupby("time.year").mean()
        da_tm = (
            da.sel(time=time_mean_period)
            .weighted(weight)
            .mean(dim=["lat", "lon"])
            .groupby("time.year")
            .mean()
            .mean(dim="year")
        )

        # Plot all ensemble members
        for i, m in enumerate(da[member]):
            lab = f"{label} (n={len(da[member])})" if i == 0 else None
            da_ts.sel({member: m}).plot(
                ax=ax,
                color=mem_color,
                ls="-",
                alpha=0.5,
                lw=1,
                label=lab,
                _labels=False,
            )

        # Optionally highlight a single ensemble member
        if highlight is not None and isinstance(highlight, int | str):
            da_ts.sel({member: highlight}).plot(
                ax=ax,
                color=color,
                ls="-",
                alpha=1,
                lw=2,
                label=f"{label} {highlight}",
                _labels=False,
            )
            ax.scatter(
                vp_xpos + 3,
                da_tm.sel({member: highlight}),
                s=20,
                marker=vp_marker,
                facecolor=vp_facecolor,
                edgecolor=vp_edgecolor,
            )

        # Add the violin plot
        vp = ax.violinplot(
            da_tm,
            [vp_xpos],
            vert=True,
            widths=15,
            side="high",
            showmeans=False,
            showextrema=True,
            showmedians=True,
        )
        vp["bodies"][0].set(facecolor=vp_facecolor, alpha=0.3)
        vp["cbars"].set(linewidth=0)
        vp["cmedians"].set(linewidth=1, color=vp_edgecolor)
        vp["cmins"].set(linewidth=1, color=vp_edgecolor)
        vp["cmaxes"].set(linewidth=1, color=vp_edgecolor)

        segmed = vp["cmedians"].get_segments().copy()
        segmin = vp["cmins"].get_segments().copy()
        segmax = vp["cmaxes"].get_segments().copy()
        for smed, smin, smax in zip(segmed, segmin, segmax):
            for s in [smed, smin, smax]:
                s[0][0] = vp_xpos - 1
                s[1][0] = vp_xpos + 1
        vp["cmedians"].set_segments(segmed)
        vp["cmins"].set_segments(segmin)
        vp["cmaxes"].set_segments(segmax)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_ylabel(ylabel, labelpad=24)

    ax.yaxis.set_ticks_position("both")
    ax.tick_params(axis="y", labelright=True)
    ax.set_ylabel(ylabel)

    # ax.legend(loc="lower left", ncol=2, fontsize=8)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim[0], xlim[1])

    if ylim[0] is not None and ylim[1] is not None:
        ax.set_ylim(ylim[0], ylim[1])
        ax2.set_ylim(ylim[0], ylim[1])

    plt.tight_layout()

    return (fig, ax)


def plot_facetgrid_map(
    da: xr.DataArray,
    dim: str,
    label: str,
    x: Hashable = "lon",
    y: Hashable = "lat",
    ncol: int = 6,
    figsize: Tuple = (16, 8),
    projection = ccrs.Robinson(),
    show_outer_labels: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    hspace: float = 0.25,
    **kwargs,
):
    """
    Create faceted map panels for visualizing spatial data across multiple categories.
    
    This function generates a grid of map plots (facets) based on a specified dimension,
    with each panel showing the spatial distribution of a DataArray for one category.
    Coastlines are automatically added when using a projection, and custom titles are
    generated for each panel based on member names.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with at least 2D spatial dimensions (lon/lat) and the faceting
        dimension `dim`. Must contain valid data for pcolormesh plotting.
    dim : str
        Name of the dimension to facet over. Each unique value along this dimension will
        create a separate panel in the grid.
    label : str
        Label for the colorbar.
    x : Hashable, optional
        Name of the x-coordinate dimension (typically longitude). Default is "lon".
    y : Hashable, optional
        Name of the y-coordinate dimension (typically latitude). Default is "lat".
    ncol : int, optional
        Number of columns in the facet grid. Rows are added automatically as needed.
        Default is 6.
    figsize : Tuple, optional
        Figure size in inches as (width, height). Default is (16, 8).
    projection : cartopy.crs projection or None, optional
        Cartographic projection for the map subplots. If None, no projection is used.
        Default is ccrs.Robinson().
    show_outer_labels : bool, optional
        If True, show x/y axis labels only on the outer edge panels (left column and
        bottom row). If False, all axis labels are removed. Default is False.
    xlabel : str or None, optional
        Label for the x-axis on bottom row panels when show_outer_labels=True.
        Default is None.
    ylabel : str or None, optional
        Label for the y-axis on left column panels when show_outer_labels=True.
        Default is None.
    hspace : float, optional
        Vertical spacing between subplot rows. Default is 0.25.
    **kwargs
        Additional keyword arguments passed to xarray.plot.pcolormesh, such as
        'cmap', 'vmin', 'vmax', 'levels', etc.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, numpy.ndarray]
        A tuple containing:
        - fig : The matplotlib Figure object
        - axs : 2D numpy array of matplotlib Axes objects (may contain None for empty slots)
        
    Notes
    -----
    - The function uses xclim.get_member_name() to generate custom titles for each panel.
    - Coastlines are automatically added with black lines (linewidth=0.8) when a projection
      is specified.
    - The colorbar is positioned horizontally at the bottom of the figure.
    - Panel titles are displayed at fontsize 8 for compact presentation.
    
    Examples
    --------
    >>> fig, axs = plot_facetgrid_map(
    ...     da=temperature_data,
    ...     dim='member',
    ...     label='Temperature [°C]',
    ...     cmap='RdBu_r',
    ...     vmin=-2,
    ...     vmax=2
    ... )
    """

    if projection:
        transform = ccrs.PlateCarree()
        subplot_kws = {"projection": projection}
    else:
        transform = None
        subplot_kws = None

    # Create facets without xarray colorbar
    fg = da.plot.pcolormesh(
        x=x,
        y=y,
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

    # Collect valid axes in row-major order
    axes = [ax for ax in fg.axs.ravel() if ax is not None]
    names = [name for name in fg.name_dicts.ravel() if name is not None]

    # Add coastlines + custom titles
    for ax, nd in zip(axes, names):
        if projection:
            ax.coastlines(color="k", lw=0.8)

        val = nd[dim]
        try:
            val_item = val.item()
        except Exception:
            val_item = val
        ax.set_title(xclim.get_member_name(val_item), fontsize=8)

    # Axis label handling
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

                is_left = c == 0
                is_bottom = r == nrows - 1

                ax.set_xlabel("")
                ax.set_ylabel("")

                if is_bottom and (xlabel is not None):
                    ax.set_xlabel(xlabel)
                if is_left and (ylabel is not None):
                    ax.set_ylabel(ylabel)

    # Adjust layout
    fig.subplots_adjust(hspace=hspace, bottom=0.1)

    return (fg.fig, fg.axs)


def plot_facetgrid_line_old(
    da: xr.DataArray | List[xr.DataArray],
    dim: str,
    x: Hashable,
    da_kwargs: dict | List[dict] | None = None,
    da2: xr.DataArray | None = None,
    da2_kwargs: dict | None = None,
    center_y: bool = False,
    ncol: int = 6,
    figsize: Tuple = (14, 8),
    show_outer_labels: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ylabel2: str | None = None,
    labels: List[str] | None = None,
    hspace: float = 0.25,
    **kwargs,
):
    """
    Create faceted line plot panels for visualizing time series or other 1D data across multiple categories.
    
    This function generates a grid of line plots (facets) based on a specified dimension,
    with each panel showing the temporal or sequential evolution of a DataArray for one category.
    Optionally supports dual y-axes to plot a second DataArray on the right y-axis, and can
    center the y-axis around zero. Custom titles are generated for each panel based on member names.

    Parameters
    ----------
    da : xr.DataArray or List[xr.DataArray]
        Input data array (or list of data arrays) with at least 1D dimension (typically time) 
        and the faceting dimension `dim`. Must contain valid data for line plotting. This will 
        be plotted on the left y-axis. If a list is provided, each DataArray will be plotted 
        on the same y-axis within each panel with different colors.
    dim : str
        Name of the dimension to facet over. Each unique value along this dimension will
        create a separate panel in the grid.
    x : Hashable, optional
        Name of the x-coordinate dimension (typically time or another sequential variable).
        Default is "time".
    da_kwargs : dict, List[dict], or None, optional
        Dictionary (or list of dictionaries) of keyword arguments to pass to the plotting 
        function for `da`. Can include 'color', 'linewidth', 'linestyle', etc. If `da` is 
        a list and `da_kwargs` is a single dict, it will be applied to all DataArrays. If 
        `da_kwargs` is a list, it should have the same length as `da`. Default is None.
    da2 : xr.DataArray or None, optional
        Optional second data array to plot on the right y-axis. Must have the same
        dimensions as `da`. Default is None.
    da2_kwargs : dict or None, optional
        Dictionary of keyword arguments to pass to the plotting function for `da2`.
        Can include 'color', 'linewidth', 'linestyle', etc. Default is None.
    center_y : bool, optional
        If True, center the y-axis around zero by making the limits symmetric. Applies
        to both left and right y-axes when `da2` is provided. Default is False.
    ncol : int, optional
        Number of columns in the facet grid. Rows are added automatically as needed.
        Default is 6.
    figsize : Tuple, optional
        Figure size in inches as (width, height). Default is (14, 8).
    show_outer_labels : bool, optional
        If True, show x/y axis labels only on the outer edge panels (left column and
        bottom row for x/y, right column for y2). If False, all axis labels are removed.
        Default is False.
    xlabel : str or None, optional
        Label for the x-axis on bottom row panels when show_outer_labels=True.
        Default is None.
    ylabel : str or None, optional
        Label for the left y-axis on left column panels when show_outer_labels=True.
        Default is None.
    ylabel2 : str or None, optional
        Label for the right y-axis on right column panels when show_outer_labels=True
        and `da2` is provided. Default is None.
    labels : List[str] or None, optional
        List of labels for the DataArrays in `da` when `da` is a list. If provided,
        a legend will be created in the bottom right of the figure. Should have the
        same length as `da` list. Default is None.
    hspace : float, optional
        Vertical spacing between subplot rows. Default is 0.25.
    **kwargs
        Additional keyword arguments passed to xarray.plot.line for the primary DataArray,
        such as 'color', 'linewidth', 'linestyle', 'marker', etc.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, numpy.ndarray]
        A tuple containing:
        - fig : The matplotlib Figure object
        - axs : 2D numpy array of matplotlib Axes objects (may contain None for empty slots)
        
    Notes
    -----
    - The function uses xclim.get_member_name() to generate custom titles for each panel.
    - Panel titles are displayed at fontsize 8 for compact presentation.
    - All panels share the same y-axis scaling for easy comparison across facets.
    - When center_y=True, a horizontal line at zero is added for reference.
    - When using dual y-axes, both axes are centered independently if center_y=True.
    
    Examples
    --------
    >>> # Simple line plot
    >>> fig, axs = plot_facetgrid_line(
    ...     da=temperature_timeseries,
    ...     dim='member',
    ...     xlabel='Year',
    ...     ylabel='Temperature Anomaly [K]',
    ...     color='blue',
    ...     linewidth=1.5
    ... )
    
    >>> # With dual y-axes and centered
    >>> fig, axs = plot_facetgrid_line(
    ...     da=temperature_timeseries,
    ...     da2=et_timeseries,
    ...     dim='member',
    ...     center_y=True,
    ...     xlabel='Year',
    ...     ylabel='Temperature Anomaly [K]',
    ...     ylabel2='Evapotranspiration [W/m2]',
    ...     color='red',
    ...     da2_kwargs={'color': 'blue', 'linestyle': '--'}
    ... )
    """

    # Handle list of DataArrays
    is_list = isinstance(da, list)
    da_list = da if is_list else [da]
    
    # Handle da_kwargs
    if da_kwargs is None:
        da_kwargs_list = [{} for _ in range(len(da_list))]
    elif isinstance(da_kwargs, list):
        da_kwargs_list = da_kwargs
        if len(da_kwargs_list) != len(da_list):
            raise ValueError("da_kwargs list must have same length as da list")
    else:
        da_kwargs_list = [da_kwargs.copy() for _ in range(len(da_list))]
    
    # Validate labels if provided
    if labels is not None:
        if not isinstance(da, list):
            raise ValueError("labels can only be provided when da is a list")
        if len(labels) != len(da_list):
            raise ValueError("labels must have same length as da list")
    
    # Assign default colors if not specified
    default_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for i, kwargs_dict in enumerate(da_kwargs_list):
        if 'color' not in kwargs_dict:
            kwargs_dict['color'] = default_colors[i]
    
    # Create facets using xarray's built-in faceting with the first DataArray
    # Add label if labels are provided
    if labels is not None:
        da_kwargs_list[0] = {**da_kwargs_list[0], 'label': labels[0]}
    
    fg = da_list[0].plot.line(
        x=x,
        col=dim,
        col_wrap=ncol,
        **da_kwargs_list[0],
    )

    fig = fg.fig
    fig.set_size_inches(*figsize)

    # Collect valid axes in row-major order
    axes = [ax for ax in fg.axs.ravel() if ax is not None]
    names = [name for name in fg.name_dicts.ravel() if name is not None]

    # Store twin axes if da2 is provided
    twin_axes = {}
    
    # Add additional DataArrays and custom titles, optional centering, and optional second y-axis
    for ax, nd in zip(axes, names):
        val = nd[dim]
        try:
            val_item = val.item()
        except Exception:
            val_item = val
        
        # Plot additional DataArrays from the list (if any)
        if len(da_list) > 1:
            for i in range(1, len(da_list)):
                da_sel = da_list[i].sel({dim: val})
                # Add label only for the first panel (first axes)
                _kwargs = da_kwargs_list[i].copy()
                if labels is not None and ax == axes[0]:
                    _kwargs['label'] = labels[i]
                da_sel.plot.line(x=x, ax=ax, **_kwargs)
        
        # Center y-axis if requested
        if center_y:
            center_axis_at_zero(ax)
            ax.axhline(lw=0.8, c="k", zorder=0)
        
        ax.set_title(xclim.get_member_name(val_item), fontsize=8)
        
        # Add optional second y-axis
        if da2 is not None:
            ax2 = ax.twinx()
            twin_axes[ax] = ax2
            da2_sel = da2.sel({dim: val})
            
            _da2_kwargs = da2_kwargs if da2_kwargs is not None else {}
            
            # Determine colors before plotting
            # Use the first da color for axis coloring
            da_color = da_kwargs_list[0].get('color', 'C0')
            # If da2 color not specified, choose a different color than the first da
            if 'color' not in _da2_kwargs:
                # Find a color not used by any da in the list
                used_colors = [kwargs.get('color', 'C0') for kwargs in da_kwargs_list]
                for candidate in ['C1', 'C2', 'C3', 'C4', 'C5']:
                    if candidate not in used_colors:
                        da2_color = candidate
                        break
                else:
                    da2_color = 'C1'
                _da2_kwargs = {**_da2_kwargs, 'color': da2_color}
            else:
                da2_color = _da2_kwargs['color']
            
            da2_sel.plot.line(x=x, ax=ax2, **_da2_kwargs)
            ax2.set_title("")
            ax2.set_xlabel("")
            # Don't clear ylabel here - let label handling section decide
            
            if center_y:
                center_axis_at_zero(ax2)
            
            # Color left y-axis (da)
            ax.tick_params(axis='y', colors=da_color, which='both')
            ax.yaxis.label.set_color(da_color)
            
            # Color right y-axis (da2)
            ax2.tick_params(axis='y', colors=da2_color, which='both')
            ax2.yaxis.label.set_color(da2_color)

    # Axis label handling
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

                is_left = c == 0
                is_right = (c == ncols - 1) or (fg.axs[r, c + 1] is None)
                is_bottom = (r == nrows - 1) or (r < nrows - 1 and fg.axs[r + 1, c] is None)

                # Clear labels for non-outer axes
                if not is_bottom:
                    ax.set_xlabel("")
                if not is_left:
                    ax.set_ylabel("")
                if da2 is not None and ax in twin_axes and not is_right:
                    twin_axes[ax].set_ylabel("")
                    twin_axes[ax].set_yticklabels([])

                # Set custom labels if provided (override defaults)
                if is_bottom and (xlabel is not None):
                    ax.set_xlabel(xlabel)
                if is_left and (ylabel is not None):
                    ax.set_ylabel(ylabel)
                if da2 is not None and is_right and ax in twin_axes and (ylabel2 is not None):
                    twin_axes[ax].set_ylabel(ylabel2)

    # Add legend if labels were provided
    if labels is not None:
        # Get legend from first axes
        handles, legend_labels = axes[0].get_legend_handles_labels()
        # Create legend in bottom right of figure
        fig.legend(handles, legend_labels, loc='lower right', bbox_to_anchor=(0.95, 0.1), fontsize=12, ncols=3)
    
    # Adjust layout
    fig.subplots_adjust(hspace=hspace, bottom=0.1)

    return (fg.fig, fg.axs)


def _normalize_da_inputs(
    da: xr.DataArray | List[xr.DataArray],
    da_kwargs: dict | List[dict] | None,
    labels: List[str] | None,
) -> tuple[List[xr.DataArray], List[dict], List[str] | None]:
    """
    Normalize DataArray inputs to lists and validate labels.
    
    Parameters
    ----------
    da : xr.DataArray or List[xr.DataArray]
        Input data array(s)
    da_kwargs : dict, List[dict], or None
        Keyword arguments for plotting
    labels : List[str] or None
        Labels for legend
        
    Returns
    -------
    tuple
        (da_list, da_kwargs_list, labels)
    """
    # Convert to list if necessary
    da_list = da if isinstance(da, list) else [da]
    
    # Normalize kwargs
    if da_kwargs is None:
        da_kwargs_list = [{} for _ in range(len(da_list))]
    elif isinstance(da_kwargs, list):
        if len(da_kwargs) != len(da_list):
            raise ValueError("da_kwargs list must have same length as da list")
        da_kwargs_list = da_kwargs
    else:
        da_kwargs_list = [da_kwargs.copy() for _ in range(len(da_list))]
    
    # Validate labels
    if labels is not None:
        if not isinstance(da, list):
            raise ValueError("labels can only be provided when da is a list")
        if len(labels) != len(da_list):
            raise ValueError("labels must have same length as da list")
    
    return da_list, da_kwargs_list, labels


def _validate_and_assign_colors(
    da_kwargs_list: List[dict],
    da2_kwargs: dict | None,
) -> tuple[List[dict], dict, str, str]:
    """
    Assign default colors to DataArrays and ensure no conflicts.
    
    Parameters
    ----------
    da_kwargs_list : List[dict]
        List of kwargs dictionaries for primary DataArrays
    da2_kwargs : dict or None
        Kwargs for secondary DataArray
        
    Returns
    -------
    tuple
        (da_kwargs_list, da2_kwargs, da_color, da2_color)
    """
    default_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    # Assign colors to primary DataArrays
    for i, kwargs_dict in enumerate(da_kwargs_list):
        if 'color' not in kwargs_dict:
            kwargs_dict['color'] = default_colors[i]
    
    # Determine primary axis color (from first DataArray)
    da_color = da_kwargs_list[0].get('color', 'C0')
    
    # Assign color to secondary DataArray if needed
    _da2_kwargs = da2_kwargs.copy() if da2_kwargs is not None else {}
    if 'color' not in _da2_kwargs:
        used_colors = [kwargs.get('color', 'C0') for kwargs in da_kwargs_list]
        for candidate in default_colors[1:]:  # Start from C1
            if candidate not in used_colors:
                da2_color = candidate
                break
        else:
            da2_color = 'C1'
        _da2_kwargs['color'] = da2_color
    else:
        da2_color = _da2_kwargs['color']
    
    return da_kwargs_list, _da2_kwargs, da_color, da2_color


def _configure_panel_axis(
    ax,
    da_list: List[xr.DataArray],
    da_kwargs_list: List[dict],
    da2: xr.DataArray | None,
    da2_kwargs: dict,
    da_color: str,
    da2_color: str,
    dim: str,
    x: Hashable,
    val,
    center_y: bool,
    labels: List[str] | None,
    is_first_panel: bool,
) -> Optional[object]:
    """
    Configure a single panel with data, styling, and optional twin axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to configure
    da_list : List[xr.DataArray]
        List of DataArrays to plot
    da_kwargs_list : List[dict]
        Plotting kwargs for each DataArray
    da2 : xr.DataArray or None
        Optional second DataArray for twin axis
    da2_kwargs : dict
        Plotting kwargs for da2
    da_color : str
        Color for primary y-axis
    da2_color : str
        Color for secondary y-axis
    dim : str
        Dimension to select along
    x : Hashable
        X-axis coordinate name
    val
        Value along dim for this panel
    center_y : bool
        Whether to center y-axis at zero
    labels : List[str] or None
        Labels for legend
    is_first_panel : bool
        Whether this is the first panel (for legend)
        
    Returns
    -------
    ax2 or None
        Twin axis if created, else None
    """
    # Plot additional DataArrays (beyond the first)
    if len(da_list) > 1:
        for i in range(1, len(da_list)):
            da_sel = da_list[i].sel({dim: val})
            _kwargs = da_kwargs_list[i].copy()
            if labels is not None and is_first_panel:
                _kwargs['label'] = labels[i]
            da_sel.plot.line(x=x, ax=ax, **_kwargs)
    
    # Center y-axis if requested
    if center_y:
        center_axis_at_zero(ax)
        ax.axhline(lw=0.8, c="k", zorder=0)
    
    # Color primary y-axis
    ax.tick_params(axis='y', colors=da_color, which='both')
    ax.yaxis.label.set_color(da_color)
    
    # Add twin axis if da2 provided
    ax2 = None
    if da2 is not None:
        ax2 = ax.twinx()
        da2_sel = da2.sel({dim: val})
        da2_sel.plot.line(x=x, ax=ax2, **da2_kwargs)
        ax2.set_title("")
        ax2.set_xlabel("")
        
        if center_y:
            center_axis_at_zero(ax2)
        
        # Color secondary y-axis
        ax2.tick_params(axis='y', colors=da2_color, which='both')
        ax2.yaxis.label.set_color(da2_color)
    
    # Set panel title after all plotting to avoid it being overwritten
    try:
        val_item = val.item()
    except Exception:
        val_item = val
    ax.set_title(xclim.get_member_name(val_item), fontsize=8)
    
    return ax2


def _configure_outer_labels(
    fg,
    axes: List,
    twin_axes: dict,
    show_outer_labels: bool,
    xlabel: str | None,
    ylabel: str | None,
    ylabel2: str | None,
):
    """
    Configure axis labels to show only on outer panels.
    
    Parameters
    ----------
    fg : FacetGrid
        The xarray FacetGrid object
    axes : List
        List of valid axes
    twin_axes : dict
        Mapping from primary axis to twin axis
    show_outer_labels : bool
        Whether to show labels on outer panels
    xlabel : str or None
        X-axis label
    ylabel : str or None
        Primary y-axis label
    ylabel2 : str or None
        Secondary y-axis label
    """
    if not show_outer_labels:
        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")
        return
    
    nrows, ncols = fg.axs.shape
    
    for r in range(nrows):
        for c in range(ncols):
            ax = fg.axs[r, c]
            if ax is None:
                continue
            
            # Determine edge positions
            is_left = c == 0
            is_right = (c == ncols - 1) or (fg.axs[r, c + 1] is None)
            is_bottom = (r == nrows - 1) or (r < nrows - 1 and fg.axs[r + 1, c] is None)
            
            # Clear all labels first
            ax.set_xlabel("")
            ax.set_ylabel("")
            if ax in twin_axes:
                twin_axes[ax].set_ylabel("")
                if not is_right:
                    twin_axes[ax].set_yticklabels([])
            
            # Set labels for outer edges
            if is_bottom and xlabel is not None:
                ax.set_xlabel(xlabel)
            if is_left and ylabel is not None:
                ax.set_ylabel(ylabel)
            if is_right and ax in twin_axes and ylabel2 is not None:
                twin_axes[ax].set_ylabel(ylabel2)


def plot_facetgrid_line(
    da: xr.DataArray | List[xr.DataArray],
    dim: str,
    x: Hashable,
    da_kwargs: dict | List[dict] | None = None,
    da2: xr.DataArray | None = None,
    da2_kwargs: dict | None = None,
    center_y: bool = False,
    ncol: int = 6,
    figsize: Tuple = (14, 8),
    show_outer_labels: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ylabel2: str | None = None,
    labels: List[str] | None = None,
    hspace: float = 0.25,
    **kwargs,
):
    """
    Create faceted line plot panels for visualizing time series or other 1D data across multiple categories.
    
    This function generates a grid of line plots (facets) based on a specified dimension,
    with each panel showing the temporal or sequential evolution of a DataArray for one category.
    Optionally supports dual y-axes to plot a second DataArray on the right y-axis, and can
    center the y-axis around zero. Custom titles are generated for each panel based on member names.

    Parameters
    ----------
    da : xr.DataArray or List[xr.DataArray]
        Input data array (or list of data arrays) with at least 1D dimension (typically time) 
        and the faceting dimension `dim`. Must contain valid data for line plotting. This will 
        be plotted on the left y-axis. If a list is provided, each DataArray will be plotted 
        on the same y-axis within each panel with different colors.
    dim : str
        Name of the dimension to facet over. Each unique value along this dimension will
        create a separate panel in the grid.
    x : Hashable
        Name of the x-coordinate dimension (typically time or another sequential variable).
    da_kwargs : dict, List[dict], or None, optional
        Dictionary (or list of dictionaries) of keyword arguments to pass to the plotting 
        function for `da`. Can include 'color', 'linewidth', 'linestyle', etc. If `da` is 
        a list and `da_kwargs` is a single dict, it will be applied to all DataArrays. If 
        `da_kwargs` is a list, it should have the same length as `da`. Default is None.
    da2 : xr.DataArray or None, optional
        Optional second data array to plot on the right y-axis. Must have the same
        dimensions as `da`. Default is None.
    da2_kwargs : dict or None, optional
        Dictionary of keyword arguments to pass to the plotting function for `da2`.
        Can include 'color', 'linewidth', 'linestyle', etc. Default is None.
    center_y : bool, optional
        If True, center the y-axis around zero by making the limits symmetric. Applies
        to both left and right y-axes when `da2` is provided. Default is False.
    ncol : int, optional
        Number of columns in the facet grid. Rows are added automatically as needed.
        Default is 6.
    figsize : Tuple, optional
        Figure size in inches as (width, height). Default is (14, 8).
    show_outer_labels : bool, optional
        If True, show x/y axis labels only on the outer edge panels (left column and
        bottom row for x/y, right column for y2). If False, all axis labels are removed.
        Default is False.
    xlabel : str or None, optional
        Label for the x-axis on bottom row panels when show_outer_labels=True.
        Default is None.
    ylabel : str or None, optional
        Label for the left y-axis on left column panels when show_outer_labels=True.
        Default is None.
    ylabel2 : str or None, optional
        Label for the right y-axis on right column panels when show_outer_labels=True
        and `da2` is provided. Default is None.
    labels : List[str] or None, optional
        List of labels for the DataArrays in `da` when `da` is a list. If provided,
        a legend will be created in the bottom right of the figure. Should have the
        same length as `da` list. Default is None.
    hspace : float, optional
        Vertical spacing between subplot rows. Default is 0.25.
    **kwargs
        Additional keyword arguments passed to xarray.plot.line for the primary DataArray,
        such as 'color', 'linewidth', 'linestyle', 'marker', etc.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, numpy.ndarray]
        A tuple containing:
        - fig : The matplotlib Figure object
        - axs : 2D numpy array of matplotlib Axes objects (may contain None for empty slots)
        
    Notes
    -----
    - The function uses xclim.get_member_name() to generate custom titles for each panel.
    - Panel titles are displayed at fontsize 8 for compact presentation.
    - All panels share the same y-axis scaling for easy comparison across facets.
    - When center_y=True, a horizontal line at zero is added for reference.
    - When using dual y-axes, both axes are centered independently if center_y=True.
    - Y-axes are color-coded to match the line colors for clarity.
    
    Examples
    --------
    >>> # Simple line plot
    >>> fig, axs = plot_facetgrid_line(
    ...     da=temperature_timeseries,
    ...     dim='member',
    ...     x='time',
    ...     xlabel='Year',
    ...     ylabel='Temperature Anomaly [K]',
    ...     color='blue',
    ...     linewidth=1.5
    ... )
    
    >>> # Multiple DataArrays with legend
    >>> fig, axs = plot_facetgrid_line(
    ...     da=[temp_ts, precip_ts],
    ...     dim='member',
    ...     x='time',
    ...     labels=['Temperature', 'Precipitation'],
    ...     show_outer_labels=True
    ... )
    """
    # Phase 1: Normalize and validate inputs
    da_list, da_kwargs_list, labels = _normalize_da_inputs(da, da_kwargs, labels)
    da_kwargs_list, _da2_kwargs, da_color, da2_color = _validate_and_assign_colors(
        da_kwargs_list, da2_kwargs
    )
    
    # Phase 2: Add label to first DataArray if labels provided
    if labels is not None:
        da_kwargs_list[0] = {**da_kwargs_list[0], 'label': labels[0]}
    
    # Phase 3: Create initial facet grid with first DataArray
    fg = da_list[0].plot.line(
        x=x,
        col=dim,
        col_wrap=ncol,
        **da_kwargs_list[0],
    )
    
    fig = fg.fig
    fig.set_size_inches(*figsize)
    
    # Phase 4: Collect valid axes and coordinate values
    axes = [ax for ax in fg.axs.ravel() if ax is not None]
    names = [name for name in fg.name_dicts.ravel() if name is not None]
    twin_axes = {}
    
    # Phase 5: Configure each panel
    for i, (ax, nd) in enumerate(zip(axes, names)):
        val = nd[dim]
        is_first_panel = (i == 0)
        
        ax2 = _configure_panel_axis(
            ax=ax,
            da_list=da_list,
            da_kwargs_list=da_kwargs_list,
            da2=da2,
            da2_kwargs=_da2_kwargs,
            da_color=da_color,
            da2_color=da2_color,
            dim=dim,
            x=x,
            val=val,
            center_y=center_y,
            labels=labels,
            is_first_panel=is_first_panel,
        )
        
        if ax2 is not None:
            twin_axes[ax] = ax2
    
    # Phase 6: Configure axis labels
    _configure_outer_labels(
        fg=fg,
        axes=axes,
        twin_axes=twin_axes,
        show_outer_labels=show_outer_labels,
        xlabel=xlabel,
        ylabel=ylabel,
        ylabel2=ylabel2,
    )
    
    # Phase 7: Add legend if labels provided
    if labels is not None:
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels,
            loc='lower right',
            bbox_to_anchor=(0.95, 0.1),
            fontsize=12,
            ncols=3
        )
    
    # Phase 8: Final layout adjustment
    fig.subplots_adjust(hspace=hspace, bottom=0.1)
    
    return (fg.fig, fg.axs)


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
        ax.set_title(xclim.get_member_name(val_item), fontsize=8)

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

                is_left = c == 0
                is_right = (c == ncols - 1) or (fg.axs[r, c + 1] is None)
                is_bottom = (r == nrows - 1) or (np.all(fg.axs[r + 1, :] is None))

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

    if da2 is not None:
        fig.subplots_adjust(hspace=hspace, wspace=0.35, bottom=0.1)
    else:
        fig.subplots_adjust(hspace=hspace, bottom=0.1)

    return (fg.fig, fg.axs)
