#!/glade/work/bbuchovecky/miniforge3/envs/data-sci-py312/bin/python3.12
"""
Compute 2-D binned statistics from observational products.

Data:
    ILAMB datasets for ET and LAI
    ERA5 for aridity (net surface radiation and precipiation)

Compute quantile bins over 1980-2014 and use the same bins
for all ET datasets.

Compute the binned mean statistics for each combination of ET and LAI
dataset, using the same ERA5 data for aridity.

Plot the binned mean ET. This differs from the FHIST PPE plots
which aggregated the difference in ET relative to the default.

Aridity definition
-------------------
Aridity is computed as the Budyko dryness index, AI = Rn / (L*P), using
ERA5 net surface radiation Rn as a proxy for PET (PET ~ Rn/L). ILAMB and
ERA5 are assumed to already share a common lat/lon grid (per instruction);
`_check_same_grid` asserts this rather than silently reindexing, so a
grid mismatch fails loudly instead of producing silently-wrong bins.

How identical x-axis (aridity) bins are guaranteed across ET datasets
-----------------------------------------------------------------------
`ppe_2d_binning.compute_2d_bin_stats` (member_dim=None path) builds bin
edges from `np.quantile` on the *full, raveled* `x_var`/`y_var` arrays
passed in -- independent of `target`'s missing-data pattern. To exploit
this and guarantee bin-for-bin identical aridity edges across every ET
dataset, ET and LAI annual means are `reindex`-ed (NOT intersected) onto
the full ERA5 year coordinate (1980-2014), so `x_var` (the ERA5 aridity
index) is passed as the exact same array, unmodified, on every call.
Years a given ET/LAI product doesn't cover become NaN after reindexing
and are simply excluded from that dataset's bin means via the joint
`isfinite(target) & isfinite(y_var) & isfinite(x_var)` mask inside
`_bin_stats_single_member` -- they do not affect the aridity edges.
LAI (y-axis) bin edges are NOT forced identical across LAI products;
only the spec's stated requirement (identical ET-dataset-independent
aridity bins) is enforced.
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

from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIME_START = "1995-01"
DEFAULT_TIME_STOP = "2014-12"

# Latent heat of vaporization [J/kg], used to convert ET (kg/m2/s -> W/m2)
LATENT_HEAT_VAPORIZATION = 2.45e6

ET_DATASETS = [
    "FLUXCOM",
    "DOLCE",
    "CLASS",
    "WECANN",
    "GLEAMv3.3a",
    "MODIS",
    "MOD16A2",
]
LAI_DATASETS = [
    "AVHRR",
    "AVH15C1",
    "MODIS",
    "GIMMS_LAI4g",
]

DEFAULT_OUTPUT_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/proc/qbin_obs")
DEFAULT_FIG_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/fig/qbin_obs")

# Masking
LAT_BNDS = slice(-58, 90)

# Plotting
PROJECTION = ccrs.PlateCarree()
ALPHA = 0.05
N_MIN = 10

DEFAULT_ANALYSIS_CHUNKS = {
    "lat": -1,
    "lon": -1,
}


sys.path.insert(0, str(Path(__file__).parent.parent / "lai_ai_binning"))
from ppe_2d_binning import compute_2d_bin_stats
from obs_trends import load_ilamb_obs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _check_same_grid(da: xr.DataArray, ref: xr.DataArray | xr.Dataset, label: str, atol: float = 1e-3) -> None:
    """Assert `da` shares its lat/lon grid with `ref`; fail loudly rather than silently reindex."""
    if da.sizes.get("lat") != ref.sizes.get("lat") or da.sizes.get("lon") != ref.sizes.get("lon"):
        raise ValueError(
            f"{label}: grid shape lat={da.sizes.get('lat')}, lon={da.sizes.get('lon')} does not match "
            f"the reference ERA5 grid lat={ref.sizes.get('lat')}, lon={ref.sizes.get('lon')}. "
            "ILAMB and ERA5 were assumed to already share a grid."
        )
    if not np.allclose(da.lat.values, ref.lat.values, atol=atol) or not np.allclose(da.lon.values, ref.lon.values, atol=atol):
        raise ValueError(
            f"{label}: lat/lon coordinate values differ from the reference ERA5 grid by more than {atol}. "
            "ILAMB and ERA5 were assumed to already share a grid."
        )


def load_ilamb_variable(
    variable: str,
    dsname: str,
    time_start: str,
    time_stop: str,
    ref_grid: xr.DataArray | xr.Dataset,
) -> xr.DataArray:
    """Load an ILAMB obs product via `obs_trends.load_ilamb_obs`.

    Discards the returned cell-area fields (not needed for gridcell-level
    binning). Trims to `LAT_BNDS` and asserts the result shares its grid
    with `ref_grid` (the ERA5 grid) rather than silently reindexing.
    """
    da, _, _ = load_ilamb_obs(variable, dsname, time_start, time_stop)
    da = da.sel(lat=LAT_BNDS)
    _check_same_grid(da, ref_grid, f"{variable}_{dsname}")
    return da


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def compute_annual_mean(da):
    days_in_month = da.time.dt.days_in_month
    weights = days_in_month.groupby('time.year') / days_in_month.groupby('time.year').sum()
    with xr.set_options(keep_attrs=True):
        return (da * weights).groupby('time.year').sum()


def compute_dryness_index(
    precip_wm2: xr.DataArray,
    rn: xr.DataArray,
    clip: bool = False,
) -> xr.DataArray:
    """Compute the annual mean Budyko dryness index, AI = Rn / (L*P).

    Parameters
    ----------
    precip_wm2 : xr.DataArray
        Annual mean precipitation expressed as an energy flux, L*P [W/m2].
    rn : xr.DataArray
        Annual mean net surface radiation [W/m2], used as a proxy for
        PET (PET ~ Rn/L) per the Budyko framework.
    clip : bool, optional
        If True, floor `rn` at zero before dividing.
    """
    if clip:
        rn = rn.clip(min=0)
    ai = rn / precip_wm2
    ai = ai.rename("AI")
    ai.attrs = {
        "long_name": "Budyko dryness index (Rn / L*P)",
        "description": (
            "computed from ERA5 annual mean net surface radiation and "
            f"precipitation{' (Rn floor at zero)' if clip else ''}"
        ),
    }
    return ai


def load_era5_aridity(
    time_start: str,
    time_stop: str,
    clip: bool = False,
) -> tuple[xr.DataArray, xr.Dataset]:
    """Load ERA5 fields and compute the annual mean Budyko dryness index.

    Returns
    -------
    ai : xr.DataArray
        Annual mean dryness index (Rn / L*P), dims (year, lat, lon), trimmed
        to `LAT_BNDS`. This array is reused, unmodified, as `x_var` for
        every ET x LAI combination so that quantile bin edges are
        guaranteed identical across ET datasets (see module docstring).
    era_grid : xr.Dataset
        The ERA5 reference grid (trimmed to `LAT_BNDS`), used to check
        that ILAMB products share the same grid.
    """
    era_grid = xclim.load_era5_grid()

    era = {}
    era_ann = {}
    for v in ["msnswrf", "msnlwrf", "mtpr"]:
        era[v] = (
            xclim.load_era5(v, time_start, time_stop, kind="meanflux")[v.upper()]
            .where(era_grid.LANDFRAC > 0.50)
            .chunk({"lat": -1, "lon": -1})
        )
        era_ann[v] = compute_annual_mean(era[v])

    era_ann["rn"] = era_ann["msnswrf"] + era_ann["msnlwrf"]

    era_ann["mtpr"] = era_ann["mtpr"] * LATENT_HEAT_VAPORIZATION
    era_ann["mtpr"].attrs["units"] = "W/m2"

    ai = compute_dryness_index(era_ann["mtpr"], era_ann["rn"], clip=clip)
    ai = ai.sel(lat=LAT_BNDS)
    era_grid = era_grid.sel(lat=LAT_BNDS)
    return ai, era_grid


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
    Test binned means against H0: mu = 0.
    For significance, the sample size must also be greater than `n_min`.
    """
    means = bin_stats.sel(stats='mean')
    stds = np.sqrt(bin_stats.sel(stats='var_samp'))
    ns = bin_stats.sel(stats='count')

    t_critical = stats.t.ppf(1 - alpha / 2, ns - 1)
    t_statistic = means / (stds / np.sqrt(ns))

    return (np.abs(t_statistic) > t_critical) & (bin_stats.sel(stats='count') > n_min)


def plot_binned_mean_obs(
    bm: dict[tuple[str, str], xr.DataArray],
    jh: dict[tuple[str, str], xr.DataArray],
    et_datasets: list[str],
    lai_datasets: list[str],
    n_xy_bins: int,
    target_label: str,
    x_label: str,
    y_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    vabs: float | None = None,
    signif_dict: dict | None = None,
    add_circles: bool = True,
    ref_count: int | None = None,
    **kwargs,
):
    """
    Create a multi-panel binned mean ET plot: rows = ET datasets,
    columns = LAI datasets, all binned against the same ERA5 dryness index.

    Parameters
    ----------
    bm, jh : dict[(et_name, lai_name)] -> xr.DataArray
        Binned mean and sample-count arrays for each ET x LAI combination,
        each with dims ('y_bin', 'x_bin').
    et_datasets, lai_datasets : list[str]
        Row and column ordering.
    Remaining parameters mirror `calc_2d_binned_stats.plot_binned_mean`.
    """
    if vabs is not None:
        vmin = -vabs
        vmax = vabs

    scaling_power = 0.4
    all_jh_max = max(float(v.max().values) for v in jh.values())
    ref_count = ref_count if ref_count is not None else round(all_jh_max / 1e2, 0) * 1e2
    radius_scalar = 0.4
    lw_scalar = 0.3
    line_alpha = 0.5

    nrows = len(et_datasets)
    ncols = len(lai_datasets)

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        sharex=True,
        sharey=True,
        figsize=(2.6 * ncols, 2.6 * nrows),
        subplot_kw=dict(box_aspect=1),
        constrained_layout=True,
        **kwargs,
    )
    axs = np.atleast_2d(axs)

    pcm = None
    for i, et_name in enumerate(et_datasets):
        for j, lai_name in enumerate(lai_datasets):
            ax = axs[i, j]
            key = (et_name, lai_name)
            m_bm = bm[key].compute()
            m_jh = jh[key].compute()

            pcm = ax.pcolormesh(m_bm, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

            if add_circles:
                m_jh_t = m_jh.T
                for ii in range(len(m_jh.x_bin)):
                    for jj in range(len(m_jh.y_bin)):
                        count = m_jh_t.values[ii, jj]
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
                            ax.add_patch(circle)

            ax.set_xlim(0, len(m_bm.x_bin))
            ax.set_ylim(0, len(m_bm.y_bin))
            ax.set_yticks(np.arange(n_xy_bins) + 0.5)
            ax.set_xticks(np.arange(n_xy_bins) + 0.5)
            ax.tick_params(length=0)
            ax.set_yticklabels(np.full(n_xy_bins, "", dtype=object))
            ax.set_xticklabels(np.full(n_xy_bins, "", dtype=object))

            if j == 0:
                ax.set_ylabel(f"{y_label}", fontsize=11)
            if i == nrows - 1:
                ax.set_xlabel(f"{x_label}", fontsize=11)
            if i == 0:
                ax.set_title(lai_name, fontsize=11, fontweight="bold")
            if j == ncols - 1:
                ax.text(
                    1.05, 0.5, et_name, rotation=-90,
                    va="center", ha="left", fontsize=11, fontweight="bold",
                    transform=ax.transAxes,
                )

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
    cbar.set_label(target_label, fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.canvas.draw()
    cbar_pos = cbar.ax.get_position()

    if signif_dict is not None:
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

    return fig, axs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 2-D binned ET statistics from ILAMB ET/LAI products "
            "and an ERA5-derived Budyko dryness index, using compute_2d_bin_stats."
        ),
    )
    parser.add_argument(
        "--et-datasets",
        nargs="+",
        default=None,
        help=f"ILAMB ET products to use. Default: all of {ET_DATASETS}.",
    )
    parser.add_argument(
        "--lai-datasets",
        nargs="+",
        default=None,
        help=f"ILAMB LAI products to use. Default: all of {LAI_DATASETS}.",
    )
    parser.add_argument(
        "--ai-options",
        nargs="+",
        type=str,
        default=(),
        help="Contains: clip net radiation at zero ('clip').",
    )
    parser.add_argument(
        "--n-y-bins",
        default=15,
        type=int,
        help="Number of bins along the y (LAI) axis. Default: 15.",
    )
    parser.add_argument(
        "--n-x-bins",
        default=15,
        type=int,
        help="Number of bins along the x (dryness index) axis. Default: 15.",
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
        help="Walltime for Dask cluster. Default: '01:00:00'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.ai_options = tuple(args.ai_options)

    et_datasets = args.et_datasets if args.et_datasets is not None else ET_DATASETS
    lai_datasets = args.lai_datasets if args.lai_datasets is not None else LAI_DATASETS
    for et_name in et_datasets:
        assert et_name in ET_DATASETS, f"Unknown ET dataset: {et_name}"
    for lai_name in lai_datasets:
        assert lai_name in LAI_DATASETS, f"Unknown LAI dataset: {lai_name}"

    if args.output.exists() and not args.output.is_dir():
        raise ValueError("`--output` must be a directory.")

    dupl_tag = ""
    if args.collapse_duplicate_quantile_bins:
        dupl_tag = ".no_dupl_bins"

    ai_tag = ""
    if len(args.ai_options) == 1:
        ai_tag = f".AI_{args.ai_options[0]}"
    elif len(args.ai_options) > 1:
        ai_tag = f".AI_{'_'.join(args.ai_options)}"

    fstem = (
        f"ET.obs.binned_by.LAI_x_AI_ERA5"
        f".{args.n_y_bins}_x_{args.n_x_bins}{dupl_tag}{ai_tag}"
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
        print("=== OBS 2-D BINNING ===")

        # ------------------------------------------------------------------
        # ERA5 dryness index -- computed once over the full analysis period.
        # This exact array is reused, unmodified, as x_var for every ET x
        # LAI combination below, which guarantees identical quantile bin
        # edges across ET datasets (see module docstring).
        # ------------------------------------------------------------------
        print("Loading ERA5 and computing dryness index...")
        to_clip = "clip" in args.ai_options
        ai, era_grid = load_era5_aridity(args.time_start, args.time_stop, clip=to_clip)
        ai = ai.transpose("year", "lat", "lon")

        if client_cluster is not None:
            ai = client_cluster[0].persist(ai.chunk(DEFAULT_ANALYSIS_CHUNKS))
            wait(ai)

        quick_map(ai.mean(dim="year"), fig_path / f"{fstem}.AI_ERA5.png")
        common_years = ai.year.values

        # ------------------------------------------------------------------
        # Load and pre-convert ET products (kg/m2/s -> W/m2), then reindex
        # (not intersect) onto the full ERA5 year coordinate so shapes
        # match `ai` for the elementwise binning below. Years the product
        # doesn't cover become NaN and are excluded per-gridcell/year by
        # the joint validity mask inside compute_2d_bin_stats -- they do
        # not shrink or alter the AI quantile edges.
        # ------------------------------------------------------------------
        et_annual: dict[str, xr.DataArray] = {}
        for et_name in et_datasets:
            print(f"Loading ET: {et_name}...")
            da = load_ilamb_variable("et", et_name, args.time_start, args.time_stop, era_grid)

            if da.attrs.get("units", "no units") not in ("kg/m2/s", "kg m-2 s-1"):
                raise ValueError(
                    f"{et_name}: expected ET units of kg/m2/s for conversion to W/m2, "
                    f"got '{da.attrs.get('units', 'no units')}'."
                )
            da = da * LATENT_HEAT_VAPORIZATION
            da.attrs["units"] = "W/m2"

            annual = compute_annual_mean(da).reindex(year=common_years)
            annual = annual.transpose("year", "lat", "lon")
            if client_cluster is not None:
                annual = client_cluster[0].persist(annual.chunk(DEFAULT_ANALYSIS_CHUNKS))
                wait(annual)

            quick_map(annual.mean(dim="year", skipna=True), fig_path / f"{fstem}.ET_{et_name}.png")
            et_annual[et_name] = annual

        # ------------------------------------------------------------------
        # Load LAI products, same reindexing treatment
        # ------------------------------------------------------------------
        lai_annual: dict[str, xr.DataArray] = {}
        for lai_name in lai_datasets:
            print(f"Loading LAI: {lai_name}...")
            da = load_ilamb_variable("lai", lai_name, args.time_start, args.time_stop, era_grid)
            annual = compute_annual_mean(da).reindex(year=common_years)
            annual = annual.transpose("year", "lat", "lon")
            if client_cluster is not None:
                annual = client_cluster[0].persist(annual.chunk(DEFAULT_ANALYSIS_CHUNKS))
                wait(annual)

            quick_map(annual.mean(dim="year", skipna=True), fig_path / f"{fstem}.LAI_{lai_name}.png")
            lai_annual[lai_name] = annual

        # ------------------------------------------------------------------
        # Compute bin stats for every ET x LAI combination
        # ------------------------------------------------------------------
        bin_stats_all: dict[tuple[str, str], xr.DataArray] = {}
        for et_name in et_datasets:
            target = et_annual[et_name].compute()

            for lai_name in lai_datasets:
                print(f"Binning ET={et_name} x LAI={lai_name}...")
                y_var = lai_annual[lai_name].compute()
                x_var = ai.compute()

                bin_stats = compute_2d_bin_stats(
                    target=target,
                    y_var=y_var,
                    x_var=x_var,
                    member_dim=None,
                    n_y_bins=args.n_y_bins,
                    n_x_bins=args.n_x_bins,
                    y_strategy="quantile",
                    x_strategy="quantile",
                    y_range=None,
                    x_range=None,
                    collapse_duplicate_quantile_bins=args.collapse_duplicate_quantile_bins,
                    pool_edges_across_ensemble=False,
                    parallel=True,
                )
                bin_stats.attrs["et_dataset"] = et_name
                bin_stats.attrs["lai_dataset"] = lai_name
                bin_stats.attrs["aridity_source"] = "ERA5 Budyko dryness index (Rn / L*P)"

                out_file = out_path / f"{fstem}.ET_{et_name}.LAI_{lai_name}.nc"
                bin_stats.to_netcdf(out_file)
                print(f"  Wrote {out_file}")

                bin_stats_all[(et_name, lai_name)] = bin_stats

        # ------------------------------------------------------------------
        # Sanity check: confirm the AI (x-axis) bin edges are identical
        # across every ET dataset, as required.
        # ------------------------------------------------------------------
        x_edge_sets = {k: tuple(v.attrs["x_bin_edges"]) for k, v in bin_stats_all.items()}
        unique_edge_sets = set(x_edge_sets.values())
        if len(unique_edge_sets) > 1:
            print(
                "WARNING: x_bin_edges (aridity) are NOT identical across all "
                f"ET x LAI combinations ({len(unique_edge_sets)} distinct edge sets found). "
                "This should not happen given the reindexing scheme above -- investigate."
            )
        else:
            print("Confirmed: identical aridity (x_bin) edges across all ET x LAI combinations.")

        # ------------------------------------------------------------------
        # Summary plot: binned mean ET (rows = ET dataset, cols = LAI dataset)
        # ------------------------------------------------------------------
        bm = {}
        jh = {}
        for key, bs in bin_stats_all.items():
            signif_mask = test_significance(bs, alpha=ALPHA, n_min=N_MIN)
            bm[key] = bs.sel(stats='mean').where(signif_mask)
            jh[key] = bs.sel(stats='count')

        fig, axs = plot_binned_mean_obs(
            bm=bm,
            jh=jh,
            et_datasets=et_datasets,
            lai_datasets=lai_datasets,
            n_xy_bins=args.n_x_bins,
            target_label='ET [W m$^{-2}$]',
            x_label='Dryness Index (Rn/LP) $\\rightarrow$',
            y_label='Leaf Area Index $\\rightarrow$',
            cmap='viridis',
            vmin=0,
            vmax=120,
            signif_dict={'alpha': ALPHA, 'n_min': N_MIN},
            add_circles=True,
            dpi=300,
        )
        fig.savefig(fig_path / f"{fstem}.png", dpi=200)
        print(f"Wrote {fig_path / f'{fstem}.png'}")

    finally:
        if client_cluster is not None:
            xclim.close_dask_cluster(client_cluster)


if __name__ == "__main__":
    main()
