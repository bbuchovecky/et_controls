"""
Plot FPPE, CMIP6, and observational products.

Plots:
------
- global map
- zonal mean
- timeseries
    - global mean
    - regional mean
- trends
    - gridcell
    - regional

Outline:
--------
Load FPPE - EFLX_LH_TOT, TLAI, FLDS, FSDS, PRECT
Load CMIP6 - evspsbl, lai, rlds, rsds, pr
Load ILAMB - et
Load ERA5 - mer, msnswrf, msnlwrf, mtpr

Compute time mean over 198001-201412

Plot maps of time mean
- ET, LAI

Compute zonal mean in native grid
Interpolate zonal mean to 1deg grid
Plot zonal mean
- individual ensemble (FPPE, CMIP6, ILAMB)
- each ensemble with shaded range
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import regionmask as regmask
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cartopy.crs as ccrs

import xesmf as xe
from ILAMB import ilamblib

from load_cmip_esgf import CMIPESGFLoader, align_dicts
import xclimate as xclim

sys.path.insert(0, str(Path(__file__).parent.parent / "lai_ai_binning"))
from ppe_2d_binning import compute_2d_bin_stats, _build_edges, _bin_stats_single_member


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DASK = True
DASK_WORKERS = 8
DASK_MEMORY = "32GB"
DASK_WALLTIME = "01:00:00"


PROC_ROOTS = {
    "fppe": Path("/glade/work/bbuchovecky/et_unc/proc/fppe"),   # FHIST PPE
    "ippe": Path("/glade/work/bbuchovecky/et_unc/proc/ippe"),   # IHIST PPE
    "goga": Path("/glade/work/bbuchovecky/et_unc/proc/goga2"),  # CESM2 GOGA2
    "lens": Path("/glade/work/bbuchovecky/et_unc/proc/lens2"),  # CESM2 LENS2
    "cmip": Path("/glade/work/bbuchovecky/et_unc/proc/cmip6"),  # CMIP6
    "obs":  Path("/glade/work/bbuchovecky/et_unc/proc/obs"),    # observations
}
SUBDIRS = [
    "trends",
    "time_mean",
    "zonal_mean",
    "binned",
]
BIN_EDGES_ROOT = Path("/glade/work/bbuchovecky/et_unc/proc/qbin_edges")


# Save figures as {'fppe.CESM2.*.png', 'ippe.CESM2.*.png', 'goga2.CESM2.*.png',
# 'lens2.CESM2.*.png', 'cmip6.<model>.*.png', 'obs.<product>.*.png'}
FIG_ROOT = Path("/glade/work/bbuchovecky/et_unc/fig")
PROJECTION = ccrs.PlateCarree()
LAT_BNDS = slice(-58, 90)
DPI = 120
CESM_TITLES = {
    "fppe": "FHIST PPE",
    "ippe": "IHIST PPE",
    "goga": "GOGA2",
    "lens": "LENS2",
}


# Binned mean ET plots
N_XBINS = 15
N_YBINS = 15


LF_THRESH = 0.5  # gridcell land fraction threshold
LAT_INTERP_GRID = np.arange(-90, 91, 1)  # 1-deg latitude array to interpolate zonal mean

LATENT_HEAT_VAPORIZATION = 2.45e6  # J/kg
LIQ_WATER_DENSITY = 1e3            # kg/m3

CESM_VARIABLES = ["EFLX_LH_TOT_month_1", "TLAI_month_1", "FLDS_month_1", "FSDS_month_1", "PRECT_month_1"]
CMIP_VARIABLES = ["evspsbl", "lai", "rlds", "rsds", "rlus", "rsus", "pr"]
ILAMB_VARIABLES = ["et", "lai"]
ERA_VARIBLES = ["mer", "msnswrf", "msnlwrf", "mtpr"]

# CESM_VARIABLES = ["EFLX_LH_TOT_month_1"]
# CMIP_VARIABLES = ["evspsbl"]
# ILAMB_VARIABLES = ["et"]
# ERA_VARIBLES = ["mer"]

FPPE_SETTINGS = {
    "TIME_SLICE": slice("1980-01", "2014-12"),
}

GOGA_SETTINGS = {
    "GCOMP": "lnd",
    "STREAM": "h0",
    "TIME_SLICE": slice("1980-01", "2014-12"),
}

LENS_SETTINGS = {
    "GCOMP": "lnd",
    "STREAM": "h0",
    "TIME_SLICE": slice("1980-01", "2014-12"),
}

CMIP_SETTINGS = {
    "EXPERIMENT_ID":      "historical",
    "MEMBER_ID":          "top",
    "SOURCE_IDS":         None,
    "OMIT_SOURCE_IDS":    ["EC-Earth3-AerChem", "EC-Earth3-ESM-1", "EC-Earth3-HR", "EC-Earth3-Veg", "EC-Earth3-Veg-LR"],
    "CATALOG":            Path("/glade/campaign/univ/uwas0155/catalogs/cmip6.csv"),
    "CESM_CATALOG":       Path("/glade/campaign/univ/uwas0155/catalogs/cmip6_cesm_mon.csv"),
    "CESM_GRID_CATALOG":  Path("/glade/campaign/univ/uwas0155/catalogs/cmip6_cesm_fx.csv"),
    "TIME_SLICE":         slice("1995-01", "2014-12"),
    "GRID_VARIABLES":     ["areacella", "sftlf"],
}

ILAMB_SETTINGS = {
    "ROOT": Path("/glade/campaign/univ/uwas0155/obs/ilamb"),
    "PRODUCTS": {
        "et": [
            "CLASS",
            "DOLCE",
            "FLUXCOM",
            "GLEAMv3.3a",
            "MODIS",
            "MOD16A2",
            # "WECANN",
        ],
        "lai": [
            "AVHRR",
            "AVH15C1",
            "GIMMS_LAI4g", 
            "MODIS",
        ]
    },
    "TIME_SLICE": slice(None, None),
}

ERA_SETTINGS = {
    "ROOT": Path("/glade/campaign/univ/uwas0155/obs/era5/regridded"),
    "KIND": "meanflux",
    "TIME_SLICE": slice("1980-01", "2014-12"),
    "RESOLUTION": "regridded_0.5_deg",
}



# ------------------------------------------------------------------
# General helpers
# ------------------------------------------------------------------

def parse_time_string(time_str: str) -> tuple[int, ...]:
    """
    Parse time string with format 'year-month-day', 'year-month', or 'year'
    to a tuple of integers with format (year, month, day). If month or day are
    not included, sets them to 1 by default.
    """

    time_str_split = time_str.split("-")
    if len(time_str_split) == 2:
        time_str_split += ["01"]
    elif len(time_str_split) == 1:
        time_str_split += ["01", "01"]
    elif (len(time_str_split) > 3) or (len(time_str_split) == 0):
        raise ValueError("Time string must contain year, month, or day.")
    return tuple(int(s) for s in time_str_split)


def get_one_mid(da: xr.DataArray) -> str:
    if ("member" in da.dims):
        if ("member_id" in da.coords):
            return str(da.member_id[0].item())
    return "onemember"


def _safe_squeeze(da: xr.DataArray, dim: str, drop: bool = True):
    if dim in da.dims:
        return da.squeeze(dim=dim, drop=drop)
    return da


def to_yyyymm(time) -> str:
    time_raw = time.values
    if isinstance(time_raw, np.datetime64):
        return pd.Timestamp(time_raw).strftime("%Y%m")
    elif isinstance(time_raw, np.ndarray):
        return time.item().strftime("%Y%m")
    raise TypeError(f"Unsupported type {type(time)!r} for time")


def check_same_grid(da: xr.DataArray, ref: xr.DataArray | xr.Dataset, label: str, atol: float = 1e-3) -> None:
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


def compute_cell_area(ds):
    land = regmask.defined_regions.natural_earth_v5_1_2.land_50

    if "lat_bounds" in ds and "lon_bounds" in ds:
        method = "bounds"
        lat_bounds = ds["lat_bounds"].values
        lon_bounds = ds["lon_bounds"].values
    else:
        method = "coords"
        lat_bounds = None
        lon_bounds = None

    area = ilamblib.CellAreas(
        lat=ds["lat"].values,
        lon=ds["lon"].values,
        lat_bnds=lat_bounds,
        lon_bnds=lon_bounds,
    )

    area = xr.DataArray(area, dims=["lat", "lon"], coords={"lat": ds["lat"], "lon": ds["lon"]})
    area.attrs["units"] = "m2"
    area.attrs["long_name"] = "grid cell area"
    area.attrs["method"] = method

    mask = xr.where(np.isnan(land.mask(lon_or_obj=area.lon, lat=area.lat)), 0, 1)
    la = area * mask
    la.attrs["units"] = "m2"
    la.attrs["long_name"] = "land grid cell area"
    la.attrs["method"] = method

    return area, la


def mask_greenland(landfrac: xr.DataArray) -> xr.DataArray:
    mask = regmask.defined_regions.ar6.land.mask(landfrac.lon, landfrac.lat)
    return xr.where((mask == 0) & (landfrac > LF_THRESH), False, landfrac > LF_THRESH)


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

def add_gridlines(ax, lat_bnds):
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


def quick_map(da: xr.DataArray, fout: Path, title: str = "", robust: bool = True, vmin: float = 0, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained", subplot_kw={"projection": PROJECTION})
    da.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", vmin=vmin, robust=robust, **kwargs)
    ax.set_title(title)
    ax.coastlines(color="k", lw=0.8)
    ax.set_extent((-180, 180, LAT_BNDS.start, LAT_BNDS.stop), crs=PROJECTION)
    add_gridlines(ax, LAT_BNDS)
    fig.savefig(fout, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_edges(edges, out):
    fig, axs = plt.subplots(2, 1, layout="constrained")
    edges.plot(ax=axs[0], marker="o")
    edges[1:-1].plot(ax=axs[1], marker="o")
    for ax in axs:
        ax.set_xlim(0, len(edges))
    fig.savefig(out, dpi=120, bbox_inches="tight")


# ------------------------------------------------------------------
# Summary plots
# ------------------------------------------------------------------

# def plot_cesm_summary(
#     dsname: str,
#     da_ann: xr.DataArray,
#     da_gm: xr.DataArray,
#     var: str,
#     units: str
# ) -> None:
#     if "member" not in da_gm.dims:
#         raise ValueError(f"'member' not in da_gm for {dsname} {var}")
#     if "member" not in da_ann.dims:
#         raise ValueError(f"'member' not in da_ann for {dsname} {var}")
#     if "year" not in da_ann.dims:
#         raise ValueError(f"'year' not in da_ann for {dsname} {var}")

#     # Get time bounds
#     start_time = f"{da_ann.year[0]}01"
#     stop_time = f"{da_ann.year[-1]}12"

#     # Plot map of time mean
#     for m in da_ann.member.values:
#         title = f"{CESM_TITLES[dsname]} {int(m):02d}, {start_time}-{stop_time}"
#         out = FIG_ROOT / dsname / f"{dsname}.CESM2.{var}.map.{start_time}-{stop_time}.{int(m):02d}.png"
#         out.parent.mkdir(exist_ok=True)
#         quick_map(da_ann.sel(member=m).mean(dim="year"), out, title=title, cbar_kwargs={"label": f"{var} [{units}]"})
#         print(out)

#     # Plot timeseries of global mean
#     fig, ax = plt.subplots(figsize=(6,4), layout="constrained")
#     for m in da_gm.member.values:
#         da_gm.sel(member=m).plot(ax=ax, color="tab:blue", alpha=0.5, _labels=False)
#     ax.set_xlim(1979, 2015)
#     ax.set_xlabel("Year")
#     ax.set_ylabel(f"{var} [{units}]")
#     ax.set_title(f"{CESM_TITLES[dsname]} (n={len(da_gm.member)}), Global Mean {var} [{units}]")
#     out = FIG_ROOT / dsname / f"{dsname}.CESM2.{var}.gmts.{start_time}-{stop_time}.png"
#     out.parent.mkdir(exist_ok=True)
#     fig.savefig(out, bbox_inches="tight", dpi=DPI)
#     print(out)


# def plot_cmip_summary(
#     sid: str,
#     dict_ann: xr.DataArray,
#     dict_gm: xr.DataArray,
#     var: str,
#     units: str
# ) -> None:
#     # Plot map of time mean
#     for sid, vardict in dict_ann.items():
#         if var not in vardict:
#             print(f"skipping {sid}")
#             continue
            
#         da = vardict[var]

#         # Check for members
#         if ("member" in da.dims) and (len(da.member) > 1):
#             raise ValueError(f"{sid} {var} more than one member")

#         # Get time bounds
#         start_time = f"{da.year[0]}01"
#         stop_time = f"{da.year[-1]}12"
        
#         out = FIG_ROOT / "cmip6" / f"cmip6.{sid}.{var}.map.{start_time}-{stop_time}.png"
#         quick_map(da.mean(dim="year"), out, title=f"{sid}, {start_time}-{stop_time}")
#         print(out)

#     # Plot timeseries of global mean
#     fig, axes = plt.subplots(nrows=7, ncols=6, sharex=True, sharey=True, figsize=(20,16), layout="constrained")
#     axs = axes.ravel()
#     for i, (sid, vardict) in enumerate(dict_gm.items()):
#         if var not in vardict:
#             print(f"skipping {sid}")
#             continue

#         da = vardict[var]

#         vardict[var].plot(ax=axs[i], color="tab:green", alpha=0.5, _labels=False)
#         axs[i].set_xlim(1979, 2015)
#         axs[i].set_title(sid)
#     for j in range(6):
#         axes[-1, j].set_xlabel("Year")
#     for i in range(7):
#         axes[i, 0].set_ylabel("ET [W/m2]")
#     out = FIG_ROOT / "cmip6" / f"cmip6.{var}.gmts.198001-201412.png"
#     fig.savefig(out, bbox_inches="tight", dpi=DPI)


# def plot_obs_summary():
#     return None


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

def compute_annual_mean(da):
    days_in_month = da.time.dt.days_in_month
    weights = days_in_month.groupby('time.year') / days_in_month.groupby('time.year').sum()
    with xr.set_options(keep_attrs=True):
        return (da * weights).groupby('time.year').sum()


def compute_aridity_index(
    precip_wm2: xr.DataArray,
    rn: xr.DataArray,
    clip: bool = False,
) -> xr.DataArray:
    """
    Compute the annual mean aridity index, AI = Rn / (L*P).

    Parameters
    ----------
    precip_wm2 : xr.DataArray
        Annual mean precipitation expressed as an energy flux, L*P [W/m2].
    rn : xr.DataArray
        Annual mean net surface radiation [W/m2], used as a proxy for
        PET (PET ~ Rn) per the Budyko framework.
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


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_fppe_variable(
    variable: str, 
    lat_bnds: slice,
    grid: xr.DataArray,
    mask: xr.DataArray | bool = True,
) -> xr.DataArray:
    if not isinstance(mask, bool) and isinstance(mask, xr.DataArray):
        mask = mask.reindex_like(grid, method="nearest", tolerance=1e-3)
    print(f"FPPE: Loading {variable}")
    v = "_".join(variable.split("_")[:-2])
    return (
        xclim.load_fhist(variable, keep_var_only=True)[v]
        .sel(time=FPPE_SETTINGS["TIME_SLICE"], lat=lat_bnds)
        .reindex_like(grid, method="nearest", tolerance=1e-3)
        .where(mask)
    )


def load_goga2_variable(
    variable: str, 
    lat_bnds: slice,
    grid: xr.DataArray,
    mask: xr.DataArray | bool = True,
) -> xr.DataArray:
    if not isinstance(mask, bool) and isinstance(mask, xr.DataArray):
        mask = mask.reindex_like(grid, method="nearest", tolerance=1e-3)
    print(f"GOGA2: Loading {variable}")
    v = "_".join(variable.split("_")[:-2])
    frq = "_".join(variable.split("_")[-2:])
    return (
        xclim.load_goga2(v, GOGA_SETTINGS["GCOMP"], frq, GOGA_SETTINGS["STREAM"])[v]
        .sel(time=GOGA_SETTINGS["TIME_SLICE"], lat=lat_bnds)
        .reindex_like(grid, method="nearest", tolerance=1e-3)
        .where(mask)
    )


def load_lens2_variable(
    variable: str, 
    lat_bnds: slice,
    grid: xr.DataArray,
    mask: xr.DataArray | bool = True,
    bb: str = "cmip6",
) -> xr.DataArray:
    if not isinstance(mask, bool) and isinstance(mask, xr.DataArray):
        mask = mask.reindex_like(grid, method="nearest", tolerance=1e-3)
    print(f"LENS2: Loading {variable}")
    v = "_".join(variable.split("_")[:-2])
    frq = "_".join(variable.split("_")[-2:])
    return (
        xclim.load_cesm2le(v, LENS_SETTINGS["GCOMP"], frq, LENS_SETTINGS["STREAM"], bb=bb)[v]
        .sel(time=LENS_SETTINGS["TIME_SLICE"], lat=lat_bnds)
        .reindex_like(grid, method="nearest", tolerance=1e-3)
        .where(mask)
    )


def load_ilamb_variable(
    product: str,
    variable: str,  
    lat_bnds: slice,
    mask: xr.DataArray | bool = True,
) -> xr.DataArray:
    path = ILAMB_SETTINGS["ROOT"] / variable / f"{variable}_{product}.nc"
    print(f"ILAMB: Loading {variable}")
    print(path)
    return (
        xr.open_dataset(path)[variable]
        .sel(time=ILAMB_SETTINGS["TIME_SLICE"], lat=lat_bnds)
        .where(mask)
    )


def load_era5_variable(
    variable: str, 
    lat_bnds: slice,
    mask: xr.DataArray | bool = True,
) -> xr.DataArray:
    time_bnds = f"{ERA_SETTINGS['TIME_SLICE'].start.replace('-', '')}-{ERA_SETTINGS['TIME_SLICE'].stop.replace('-', '')}"
    path = ERA_SETTINGS["ROOT"] / f"e5.{ERA_SETTINGS['KIND']}.{variable}.{time_bnds}.{ERA_SETTINGS['RESOLUTION']}.nc"
    print(f"ERA5: Loading {variable}")
    print(path)
    ds = xr.open_dataset(path)
    if variable in ds.data_vars:
        v = variable
    elif variable.lower() in ds.data_vars:
            v = variable.lower()
    elif variable.upper() in ds.data_vars:
        v = variable.upper()
    else:
        raise ValueError(f"[{variable}, {variable.lower()}, {variable.upper()}] not in ERA5 dataset")
    return (
        ds[v]
        .sel(time=ERA_SETTINGS["TIME_SLICE"], lat=lat_bnds)
        .where(mask)
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:

    client_cluster = None
    if DASK:
        client_cluster = xclim.create_dask_cluster(
            account="UWAS0155",
            nworkers=DASK_WORKERS,
            ncores=DASK_WORKERS,
            nmem=DASK_MEMORY,
            walltime=DASK_WALLTIME,
        )
        client_cluster[0].wait_for_workers(DASK_WORKERS)

    try:
        # # ------------------------------------------------------------------
        # # Load FPPE
        # # ------------------------------------------------------------------
        # fppe_time_str = (
        #     f"{FPPE_SETTINGS['TIME_SLICE'].start.replace('-', '')}-"
        #     f"{FPPE_SETTINGS['TIME_SLICE'].stop.replace('-', '')}"
        # )

        # print("Load FHIST PPE")
        # fppe_grid = xclim.load_fhist_ppe_grid()
        # fppe = {}
        # for v in CESM_VARIABLES:
        #     fppe[v] = load_fppe_variable(v, slice(-90, 90), fppe_grid, mask=fppe_grid.LANDFRAC>LF_THRESH)

        #     if v == "PRECT_calculated_month_1":
        #         print("Converting units from m/s -> W/m2")
        #         fppe[v] = fppe[v] * LATENT_HEAT_VAPORIZATION * LIQ_WATER_DENSITY
        #         fppe[v].attrs["units"] = "W/m2"


        # # ------------------------------------------------------------------
        # # Load GOGA2
        # # ------------------------------------------------------------------
        # goga_time_str = (
        #     f"{GOGA_SETTINGS['TIME_SLICE'].start.replace('-', '')}-"
        #     f"{GOGA_SETTINGS['TIME_SLICE'].stop.replace('-', '')}"
        # )
    
        # print("Load GOGA2")
        # goga_grid = xclim.load_goga2_grid()
        # goga = {}
        # for v in CESM_VARIABLES:
        #     goga[v] = load_goga2_variable(v, slice(-90, 90), goga_grid, mask=goga_grid.LANDFRAC>LF_THRESH)

        #     if v == "PRECT_month_1":
        #         print("Converting units from m/s -> W/m2")
        #         goga[v] = goga[v] * LATENT_HEAT_VAPORIZATION * LIQ_WATER_DENSITY
        #         goga[v].attrs["units"] = "W/m2"


        # # ------------------------------------------------------------------
        # # Load LENS2
        # # ------------------------------------------------------------------
        # lens_time_str = (
        #     f"{LENS_SETTINGS['TIME_SLICE'].start.replace('-', '')}-"
        #     f"{LENS_SETTINGS['TIME_SLICE'].stop.replace('-', '')}"
        # )

        # print("Load LENS2")
        # lens_grid = xclim.load_cesm2le_grid()
        # # lens_smbb = {}
        # lens_cmip = {}
        # # for bb, dct in zip(["smbb", "cmip6"], [lens_smbb, lens_cmip]):
        # for bb, dct in zip(["cmip6"], [lens_cmip]):
        #     for v in CESM_VARIABLES:
        #         dct[v] = load_lens2_variable(v, slice(-90, 90), lens_grid, mask=lens_grid.LANDFRAC>LF_THRESH, bb=bb)

        #         if v == "PRECT_month_1":
        #             print("Converting units from m/s -> W/m2")
        #             dct[v] = dct[v] * LATENT_HEAT_VAPORIZATION * LIQ_WATER_DENSITY
        #             dct[v].attrs["units"] = "W/m2"


        # ------------------------------------------------------------------
        # Load CMIP
        # ------------------------------------------------------------------
        cmip_time_str = (
            f"{CMIP_SETTINGS['TIME_SLICE'].start.replace('-', '')}-"
            f"{CMIP_SETTINGS['TIME_SLICE'].stop.replace('-', '')}"
        )

        loader_cmip = CMIPESGFLoader(CMIP_SETTINGS["CATALOG"])
        loader_cesm = CMIPESGFLoader(CMIP_SETTINGS["CESM_CATALOG"])
        loader_cesm_grid = CMIPESGFLoader(CMIP_SETTINGS["CESM_GRID_CATALOG"])

        # Load CMIP grid variables
        print("Load CMIP grid variables")
        cmip_grid = loader_cmip.load_data(
            variables=CMIP_SETTINGS["GRID_VARIABLES"],
            experiment_id=CMIP_SETTINGS["EXPERIMENT_ID"],
            source_id=CMIP_SETTINGS["SOURCE_IDS"],
            omit_source_id=CMIP_SETTINGS["OMIT_SOURCE_IDS"],
            member_id=CMIP_SETTINGS["MEMBER_ID"],
            time_slice=CMIP_SETTINGS["TIME_SLICE"],
            parallel=True,
            compute=False,
        )
        cesm_grid = loader_cesm_grid.load_data(
            variables=CMIP_SETTINGS["GRID_VARIABLES"],
            experiment_id=CMIP_SETTINGS["EXPERIMENT_ID"],
            source_id=None,
            omit_source_id=CMIP_SETTINGS["OMIT_SOURCE_IDS"],
            member_id=CMIP_SETTINGS["MEMBER_ID"],
            time_slice=CMIP_SETTINGS["TIME_SLICE"],
            parallel=True,
            compute=False,
        )
        cmip_grid = cmip_grid | cesm_grid

        # Load CMIP variables
        print("Load CMIP variables")
        cmip = loader_cmip.load_data(
            variables=CMIP_VARIABLES,
            experiment_id=CMIP_SETTINGS["EXPERIMENT_ID"],
            source_id=CMIP_SETTINGS["SOURCE_IDS"],
            omit_source_id=CMIP_SETTINGS["OMIT_SOURCE_IDS"],
            member_id=CMIP_SETTINGS["MEMBER_ID"],
            time_slice=CMIP_SETTINGS["TIME_SLICE"],
            parallel=True,
            compute=False,
        )
        cesm = loader_cesm.load_data(
            variables=CMIP_VARIABLES,
            experiment_id=CMIP_SETTINGS["EXPERIMENT_ID"],
            source_id=None,
            source_candidates=["CESM2", "CESM2-WACCM", "CESM2-FV2", "CESM2-WACCM-FV2"],
            omit_source_id=CMIP_SETTINGS["OMIT_SOURCE_IDS"],
            member_id=CMIP_SETTINGS["MEMBER_ID"],
            time_slice=CMIP_SETTINGS["TIME_SLICE"],
            parallel=True,
            compute=False,
        )
        cmip = cmip | cesm

        print("Align data dictionaries")
        cmip_aligned, cmip_grid_aligned = align_dicts(cmip, cmip_grid)

        print("Compute land area")
        for sid, vardict in cmip_grid_aligned.items():
            if len(vardict.keys()) > 0:
                cmip_grid_aligned[sid]["sftlf"] = cmip_grid_aligned[sid]["sftlf"].sel(lat=LAT_BNDS)
                cmip_grid_aligned[sid]["areacella"] = cmip_grid_aligned[sid]["areacella"].sel(lat=LAT_BNDS)

                cmip_grid_aligned[sid]["lf"] = cmip_grid_aligned[sid]["sftlf"].reindex_like(cmip_grid_aligned[sid]["areacella"], method="nearest", tolerance=1e-3)
                cmip_grid_aligned[sid]["la"] = cmip_grid_aligned[sid]["areacella"] * cmip_grid_aligned[sid]["lf"]
                cmip_grid_aligned[sid]["mask"] = mask_greenland(cmip_grid_aligned[sid]["lf"])
                del cmip_grid_aligned[sid]["sftlf"]

        print("Reindex analysis variables to atmospheric grid")
        for sid, vardict in cmip_aligned.items():
            print(f"=== {sid} ===")
            for v, da in vardict.items():
                vardict[v] = (
                    da.sel(lat=LAT_BNDS)
                    .reindex_like(cmip_grid_aligned[sid]["mask"], method="nearest", tolerance=1e-3)
                    .where(cmip_grid_aligned[sid]["mask"])
                )
                print(f"{v:12}: {vardict[v].dims} {vardict[v].shape}")

                if v in ("pr", "evspsbl"):
                    print("Converting units from kg/m2/s -> W/m2")
                    vardict[v] = vardict[v] * LATENT_HEAT_VAPORIZATION
                    vardict[v].attrs["units"] = "W/m2"

        print("ORIGINAL")
        print(f"n_models: {len(cmip.keys())}")
        print(f"n_models: {len(cmip_grid.keys())}")
        print("\nALIGNED")
        print(f"n_models: {len(cmip_aligned.keys())}")
        print(f"n_models: {len(cmip_grid_aligned.keys())}")
        print("\nDROPPED")
        print(f"{set(cmip.keys()) - set(cmip_aligned.keys())}")
        print(f"{set(cmip_grid.keys()) - set(cmip_grid_aligned.keys())}")
        print(f"{set(cmip_aligned.keys()) - set(cmip_grid_aligned.keys())}")
        print(f"{set(cmip_grid_aligned.keys()) - set(cmip_aligned.keys())}")


        # # ------------------------------------------------------------------
        # # Load ILAMB
        # # ------------------------------------------------------------------

        # # Load variables
        # ilamb = {}
        # ilamb_la = {}
        # for v in ILAMB_VARIABLES:
        #     for p in ILAMB_SETTINGS["PRODUCTS"][v]:
        #         ilamb[p] = load_ilamb_variable(p, v, slice(-90, 90))

        #         if v == "et":
        #             print("Converting units from kg/m2/s -> W/m2")
        #             ilamb[p] = ilamb[p] * LATENT_HEAT_VAPORIZATION
        #             ilamb[p].attrs["units"] = "W/m2"

        #         _, ilamb_la[p] = compute_cell_area(ilamb[p])

        # # Create land and missing data mask
        # ilamb_mask = {}
        # for v in ILAMB_VARIABLES:
        #     ilamb_mask[v] = xr.ones_like(ilamb[ILAMB_SETTINGS["PRODUCTS"][v][0]].isel(time=0))
        #     for p in ILAMB_SETTINGS["PRODUCTS"][v]:
        #         ilamb_mask[v] = xr.where(np.isnan(ilamb[p].isel(time=0)), 0, ilamb_mask[v])

        # # Mask variables
        # for v in ILAMB_VARIABLES:
        #     for p in ILAMB_SETTINGS["PRODUCTS"][v]:
        #         ilamb[p] = ilamb[p].where(ilamb_mask[v], other=0).where(ilamb_la[p]>0)
        #         print(f"{p} nans: {np.sum(np.isnan(ilamb[p].isel(time=0))).values}")
    

        # # ------------------------------------------------------------------
        # # Load ERA5
        # # ------------------------------------------------------------------

        # # !! This requires ERA5 regridded to ILAMB 0.5-degree grid !!
        # v_grid = ILAMB_VARIABLES[0]
        # p_grid = ILAMB_SETTINGS["PRODUCTS"][v_grid][0]
        # obs_area, obs_la = compute_cell_area(ilamb[p_grid])
        # obs_lf = obs_la / obs_area

        # era = {}
        # for v in ERA_VARIBLES:
        #     era[v] = load_era5_variable(v, slice(-90, 90), mask=obs_lf>LF_THRESH)

        #     if v in ("mtpr", "mer"):
        #         print("Converting units from kg/m2/s -> W/m2")
        #         era[v] = era[v] * LATENT_HEAT_VAPORIZATION
        #         era[v].attrs["units"] = "W/m2"

        #     if v == "mer":
        #         era[v] = -1 * era[v]


        # ------------------------------------------------------------------
        # Subset time and compute the annual mean --> use this onwards
        # ------------------------------------------------------------------
        print("Computing annual mean and persisting")

        # fppe_ann = {}
        # for v, da in fppe.items():
        #     fppe_ann[v] = compute_annual_mean(da.sel(time=FPPE_SETTINGS["TIME_SLICE"])).persist()

        # goga_ann = {}
        # for v, da in goga.items():
        #     goga_ann[v] = compute_annual_mean(da.sel(time=GOGA_SETTINGS["TIME_SLICE"])).persist()

        # lens_smbb_ann = {}
        # for v, da in lens_smbb.items():
        #     lens_smbb_ann[v] = compute_annual_mean(da.sel(time=LENS_SETTINGS["TIME_SLICE"])).persist()

        # lens_cmip_ann = {}
        # for v, da in lens_cmip.items():
        #     lens_cmip_ann[v] = compute_annual_mean(da.sel(time=LENS_SETTINGS["TIME_SLICE"])).persist()

        cmip_ann = {}
        for sid, vardict in cmip_aligned.items():
            cmip_ann[sid] = {}
            for v, da in vardict.items():
                cmip_ann[sid][v] = compute_annual_mean(da.sel(time=CMIP_SETTINGS["TIME_SLICE"])).persist()
        
        # ilamb_ann = {}
        # for p, da in ilamb.items():
        #     ilamb_ann[p] = compute_annual_mean(da.sel(time=ILAMB_SETTINGS["TIME_SLICE"])).persist()

        # era_ann = {}
        # for v, da in era.items():
        #     era_ann[v] = compute_annual_mean(da.sel(time=ERA_SETTINGS["TIME_SLICE"])).persist()


        # # ------------------------------------------------------------------
        # # Compute the global mean
        # # ------------------------------------------------------------------
        # fppe_gm = {}
        # for v, da in fppe_ann.items():
        #     fppe_gm[v] = da.weighted(fppe_grid.LANDAREA).mean(dim=["lat", "lon"])

        # goga_gm = {}
        # for v, da in goga_ann.items():
        #  goga_gm[v] = da.weighted(goga_grid.LANDAREA).mean(dim=["lat", "lon"])

        # # lens_smbb_gm = {}
        # # for v, da in lens_smbb_ann.items():
        # #     lens_smbb_gm[v] = da.weighted(lens_grid.LANDAREA).mean(dim=["lat", "lon"])

        # lens_cmip_gm = {}
        # for v, da in lens_cmip_ann.items():
        #     lens_cmip_gm[v] = da.weighted(lens_grid.LANDAREA).mean(dim=["lat", "lon"])

        # cmip_gm = {}
        # for sid, vardict in cmip_ann.items():
        #     cmip_gm[sid] = {}
        #     for v, da in vardict.items():
        #         cmip_gm[sid][v] = da.weighted(cmip_grid_aligned[sid]["la"]).mean(dim=["lat", "lon"])
        
        # ilamb_gm = {}
        # for p, da in ilamb_ann.items():
        #     ilamb_gm[p] = da.weighted(ilamb_la[p]).mean(dim=["lat", "lon"])

        # era_gm = {}
        # for v, da in era_ann.items():
        #     era_gm[v] = da.weighted(obs_la).mean(dim=["lat", "lon"])


        # # ------------------------------------------------------------------
        # # Compute the zonal mean --> interpolate to 1-degree latitude grid
        # # ------------------------------------------------------------------
        # interp_method = "linear"

        # fppe_zm = {}
        # for v, da in fppe_ann.items():
        #     fppe_zm[v] = da.mean(dim=["year", "lon"]).interp(lat=LAT_INTERP_GRID, method=interp_method)

        # goga_zm = {}
        # for v, da in goga_ann.items():
        #  goga_zm[v] = da.mean(dim=["year", "lon"]).interp(lat=LAT_INTERP_GRID, method=interp_method)

        # # lens_smbb_zm = {}
        # # for v, da in lens_smbb_ann.items():
        # #     lens_smbb_zm[v] = da.mean(dim=["year", "lon"]).interp(lat=LAT_INTERP_GRID, method=interp_method)

        # lens_cmip_zm = {}
        # for v, da in lens_cmip_ann.items():
        #     lens_cmip_zm[v] = da.mean(dim=["year", "lon"]).interp(lat=LAT_INTERP_GRID, method=interp_method)

        # cmip_zm = {}
        # for sid, vardict in cmip_ann.items():
        #     cmip_zm[sid] = {}
        #     for v, da in vardict.items():
        #         cmip_zm[sid][v] = da.mean(dim=["year", "lon"]).interp(lat=LAT_INTERP_GRID, method=interp_method)
        
        # ilamb_zm = {}
        # for p, da in ilamb_ann.items():
        #     ilamb_zm[p] = da.mean(dim=["year", "lon"]).interp(lat=LAT_INTERP_GRID, method=interp_method)

        # era_zm = {}
        # for v, da in era_ann.items():
        #     era_zm[v] = da.mean(dim=["year", "lon"]).interp(lat=LAT_INTERP_GRID, method=interp_method)


        # --------------------------------------------------------------------
        # Compute mean ET binned by LAI (monthly) and aridity (climatological)
        # --------------------------------------------------------------------

        print("Create consistent LAI and aridity bins across CMIP6 models")
        lai_flat_list = []
        ai_flat_list = []
        sid_qbin_list = []
        for sid in cmip_ann.keys():
            print(f"=== {sid} ===")

            vs = set(cmip_ann[sid].keys())
            if len(set(CMIP_VARIABLES) - vs) > 0:
                print(f"skipping {sid}")
                continue

            sid_qbin_list.append(sid)

            cmip_ann[sid]["rn"] = cmip_ann[sid]["rlds"] + cmip_ann[sid]["rsds"] - cmip_ann[sid]["rlus"] - cmip_ann[sid]["rsus"]

            rn = cmip_ann[sid]["rn"].mean(dim="year").where(cmip_grid_aligned[sid]["mask"])
            pr = cmip_ann[sid]["pr"].mean(dim="year").where(cmip_grid_aligned[sid]["mask"])
            ai = compute_aridity_index(pr, rn)

            ai = _safe_squeeze(ai, dim="member", drop=True)
            lai = _safe_squeeze(cmip_aligned[sid]["lai"], dim="member", drop=True)

            ai_flat = ai.values.ravel()
            lai_flat = lai.values.ravel()

            print(f"ai: {ai.dims} {ai.shape}")
            print(f"lai: {lai.dims} {lai.shape}")

            print(f"ai_flat: {len(ai_flat)} [{np.nanmin(ai_flat):0.3e}, {np.nanmax(ai_flat):0.3e}]")
            print(f"lai_flat: {len(lai_flat)} [{np.nanmin(lai_flat):0.3e}, {np.nanmax(lai_flat):0.3e}]")

            ai_flat_list.append(ai_flat)
            lai_flat_list.append(lai_flat)

        ai_flat_all = np.concatenate(ai_flat_list)
        lai_flat_all = np.concatenate(lai_flat_list)

        print("=== concatenated ===")
        print(f"ai_flat_all: {len(ai_flat_all)} [{np.nanmin(ai_flat_all):0.3e}, {np.nanmax(ai_flat_all):0.3e}]")
        print(f"lai_flat_all: {len(lai_flat_all)} [{np.nanmin(lai_flat_all):0.3e}, {np.nanmax(lai_flat_all):0.3e}]")

        x_edges = _build_edges(
            arr_flat=ai_flat_all,
            n_bins=N_XBINS,
            strategy="quantile",
            value_range=None,
            collapse_duplicates=False
        )
        y_edges = _build_edges(
            arr_flat=lai_flat_all,
            n_bins=N_YBINS,
            strategy="quantile",
            value_range=None,
            collapse_duplicates=False
        )

        del ai_flat_list, ai_flat_all
        del lai_flat_list, lai_flat_all

        n_y_eff = len(y_edges) - 1
        n_x_eff = len(x_edges) - 1

        print(f"x_edges ({n_x_eff}): {x_edges}")
        print(f"y_edges ({n_y_eff}): {y_edges}")

        x_edges_da = xr.DataArray(
            data=x_edges,
            dims=["edge"],
            coords=dict(edge=np.arange(n_x_eff+1)),
            name="ai",
            attrs={
                "long_name"       : "aridity index bin edges",
                "units"           : "W/m2",
                "variable"        : "aridity index (Rn/P)",
                "strategy"        : "quantile",
                "n_bins"          : n_x_eff,
                "n_bins_requested": N_XBINS,
                "collapse_duplicate_quantile_bins": 0,
                "pool_edges"      : 1,
                "source_id"       : sid_qbin_list,
                "time_period"     : cmip_time_str,
            },
        )
        fstem = f"cmip6.ai_clim.{N_XBINS}_quantiles_pooled.{cmip_time_str}"
        x_edges_da.to_netcdf(BIN_EDGES_ROOT / f"{fstem}.nc")
        print(BIN_EDGES_ROOT / f"{fstem}.nc")
        plot_edges(x_edges_da, BIN_EDGES_ROOT / f"{fstem}.png")
        print(BIN_EDGES_ROOT / f"{fstem}.png")
        

        y_edges_da = xr.DataArray(
            data=y_edges,
            dims=["edge"],
            coords=dict(edge=np.arange(n_y_eff+1)),
            name="lai",
            attrs={
                "long_name"       : "leaf area index bin edges",
                "units"           : "m2/m2",
                "variable"        : "leaf area index",
                "strategy"        : "quantile",
                "n_bins"          : n_y_eff,
                "n_bins_requested": N_YBINS,
                "collapse_duplicate_quantile_bins": 0,
                "pool_edges"      : 1,
                "source_id"       : sid_qbin_list,
                "time_period"     : cmip_time_str,
            },
        )
        fstem = f"cmip6.lai_mon.{N_YBINS}_quantiles_pooled.{cmip_time_str}"
        y_edges_da.to_netcdf(BIN_EDGES_ROOT / f"{fstem}.nc")
        print(BIN_EDGES_ROOT / f"{fstem}.nc")
        plot_edges(y_edges_da, BIN_EDGES_ROOT / f"{fstem}.png")
        print(BIN_EDGES_ROOT / f"{fstem}.png")

    
        print("Compute mean ET using LAI and aridity bins")
        for sid in cmip_ann.keys():
            print(f"=== {sid} ===")

            vs = set(cmip_ann[sid].keys())
            if len(set(CMIP_VARIABLES) - vs) > 0:
                print(f"skipping {sid}")
                continue

            # Calculate the climatological fields
            lai_clim = cmip_ann[sid]["lai"].mean(dim="year").where(cmip_grid_aligned[sid]["mask"])
            pr_clim = cmip_ann[sid]["pr"].mean(dim="year").where(cmip_grid_aligned[sid]["mask"])
            rn_clim = cmip_ann[sid]["rn"].mean(dim="year").where(cmip_grid_aligned[sid]["mask"])
            ai_clim = compute_aridity_index(pr_clim, rn_clim).compute()

            lai_clim = _safe_squeeze(lai_clim, dim="member", drop=True)
            rn_clim = _safe_squeeze(rn_clim, dim="member", drop=True)
            pr_clim = _safe_squeeze(pr_clim, dim="member", drop=True)
            ai_clim = _safe_squeeze(ai_clim, dim="member", drop=True)

            mid = get_one_mid(cmip[sid]["evspsbl"])
            for v, da in zip(["lai", "rn", "pr", "ai"], [lai_clim, rn_clim, pr_clim, ai_clim]):
                out = FIG_ROOT / "cmip6" / v / f"cmip6.{sid}.{v}.map.{cmip_time_str}.{mid}.png"
                out.parent.mkdir(exist_ok=True, parents=True)
                quick_map(da, out, title=f"{sid}, {cmip_time_str}")
                print(out)

            target = _safe_squeeze(cmip_aligned[sid]["evspsbl"], dim="member", drop=True)
            y_var = _safe_squeeze(cmip_aligned[sid]["lai"], dim="member", drop=True)
            x_var = ai_clim

            if "time" in target.dims:
                expand_tcoord = "time"
            elif "year" in target.dims:
                expand_tcoord = "year"
            else:
                raise ValueError(f"target must contain `time` or `year` dimension; instead has dimensions {target.dims}")

            if (y_var.shape != target.shape):
                print(f"y_var: {y_var.dims} {y_var.shape}  target: {target.dims} {target.shape}")
                if expand_tcoord not in y_var.coords:
                    old_shape = y_var.shape
                    y_var = y_var.expand_dims({expand_tcoord: target[expand_tcoord]})
                    print(f"Expanding time dimension of y-bin: {old_shape} -> {y_var.shape}")
                else:
                    print(f"{expand_tcoord} in y_var.coords {list(y_var.coords)}, skipping {sid}")
                    continue
            
            if (x_var.shape != target.shape):
                print(f"x_var: {x_var.shape}  target: {target.shape}")
                if expand_tcoord not in x_var.coords:
                    old_shape = x_var.shape
                    x_var = x_var.expand_dims({expand_tcoord: target[expand_tcoord]})
                    print(f"Expanding time dimension of x-bin: {old_shape} -> {x_var.shape}")
                else:
                    print(f"{expand_tcoord} in x_var.coords {list(x_var.coords)}, skipping {sid}")
                    continue

            print(f"target: {target.dims} {target.shape} lat=[{target.lat.min().item()}, {target.lat.max().item()}]")
            print(f"y_var: {y_var.dims} {y_var.shape} lat=[{y_var.lat.min().item()}, {y_var.lat.max().item()}]")
            print(f"x_var: {x_var.dims} {x_var.shape} lat=[{x_var.lat.min().item()}, {x_var.lat.max().item()}]")

            print("Binning")
            def _np(da):
                return da.compute().values if hasattr(da.data, "compute") else da.values

            tgt_np = _np(target)
            y_np   = _np(y_var)
            x_np   = _np(x_var)

            print(f"target_np: {tgt_np.shape}")
            print(f"y_var_np: {y_np.shape}")
            print(f"x_var_np: {x_np.shape}")

            result_np = _bin_stats_single_member(
                tgt_np.ravel(),
                y_np.ravel(),
                x_np.ravel(),
                y_edges, x_edges,
            )

            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        
            # Human-readable quantile labels (used when strategy=="quantile")
            def _quantile_labels(n):
                lo = np.linspace(0, 100, n + 1)[:-1]
                hi = np.linspace(0, 100, n + 1)[1:]
                return [f"Q{a:.0f}-Q{b:.0f}" for a, b in zip(lo, hi)]
        
            y_name = y_var.name or "y_var"
            x_name = x_var.name or "x_var"
        
            coords = {
                "stats": np.array(["mean", "var_pop", "var_samp", "count"]),
                f"{y_name}_bin_center": ("y_bin", y_centers),
                f"{x_name}_bin_center": ("x_bin", x_centers),
            }
            coords[f"{y_name}_bin_label"] = ("y_bin", _quantile_labels(n_y_eff))
            coords[f"{x_name}_bin_label"] = ("x_bin", _quantile_labels(n_x_eff))
        
            bin_stats = xr.DataArray(
                result_np,
                name="evspsbl",
                dims=("stats", "y_bin", "x_bin"),
                coords=coords,
                attrs={
                    "long_name"            : f"2-D bin-mean {target.name or 'variable'}",
                    "units"                : target.attrs.get("units", "unknown"),
                    "y_variable"           : y_name,
                    "x_variable"           : x_name,
                    "y_strategy"           : "quantile",
                    "x_strategy"           : "quantile",
                    "y_bin_edges"          : y_edges.tolist(),
                    "x_bin_edges"          : x_edges.tolist(),
                    "n_y_bins"             : n_y_eff,
                    "n_x_bins"             : n_x_eff,
                    "n_y_bins_requested"   : N_YBINS,
                    "n_x_bins_requested"   : N_XBINS,
                    "collapse_duplicate_quantile_bins": 0,
                    "pool_edges"           : 1,
                    "source_id"            : sid,
                    "time_period"          : cmip_time_str,
                },
            )


            # bin_stats = compute_2d_bin_stats(
            #     target=target,
            #     y_var=y_var,
            #     x_var=x_var,
            #     member_dim=None,
            #     n_y_bins=N_YBINS,
            #     n_x_bins=N_XBINS,
            #     y_strategy="quantile",
            #     x_strategy="quantile",
            #     y_range=None,
            #     x_range=None,
            #     collapse_duplicate_quantile_bins=False,
            #     pool_edges_across_ensemble=False,
            #     parallel=True,
            # )
            # bin_stats.name = "evspsbl"
            # bin_stats.attrs["collapse_duplicate_quantile_bins"] = 0
            # bin_stats.attrs['pool_edges'] = 0
            # bin_stats.attrs["source_id"] = sid
            # bin_stats.attrs["time_period"] = cmip_time_str

            mid = get_one_mid(cmip[sid]["evspsbl"])
            out = PROC_ROOTS["cmip"] / "qbin" / f"cmip6.{sid}.evspsbl.qbin_pooled.{cmip_time_str}.{mid}.nc"
            out.parent.mkdir(exist_ok=True, parents=True)
            bin_stats.to_netcdf(out)
            print(out)


        # # ------------------------------------------------------------------
        # # Plot zonal mean ET (zm)
        # # ------------------------------------------------------------------
        # print("Plot zonal mean ET")

        # fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained", sharex=True, sharey=True)
        # axs = axes.ravel()

        # # Subplot 1: CESM2 FHIST PPE
        # fppe_da = fppe_zm["EFLX_LH_TOT_month_1"]
        # for m in fppe_da.member:
        #     fppe_da.sel(member=m).plot(ax=axs[0], color="tab:blue", alpha=0.5, _labels=False)
        # axs[0].set_title(f"FHIST PPE (n={len(fppe_da.member)})")

        # # Subplot 2: CESM2 LEs
        # goga_da = goga_zm["EFLX_LH_TOT_month_1"]
        # for m in goga_da.member:
        #     goga_da.sel(member=m).plot(ax=axs[1], color="indianred", alpha=0.5, _labels=False)
        
        # # lens_smbb_da = lens_smbb_zm["EFLX_LH_TOT_month_1"]
        # # for m in lens_smbb_da.member:
        # #     lens_smbb_da.sel(member=m).plot(ax=axs[1], color="tab:orange", alpha=0.5, _labels=False)
        
        # lens_cmip_da = lens_cmip_zm["EFLX_LH_TOT_month_1"]
        # for m in lens_cmip_da.member:
        #     lens_cmip_da.sel(member=m).plot(ax=axs[1], color="tab:orange", alpha=0.5, _labels=False)

        # # axs[1].legend(ncols=2)
        # # axs[1].set_title(f"GOGA2 (n={len(goga_da.member)}) + LENS2 (n={len(lens_smbb_da.member) + len(lens_cmip_da.member)})")
        # axs[1].set_title(f"GOGA2 (n={len(goga_da.member)}) + LENS2 (n={len(lens_cmip_da.member)})")

        # # Subplot 3: CMIP6
        # for sid, vardict in cmip_zm.items():
        #     if "evspsbl" not in vardict:
        #         print(f"skipping {sid}")
        #         continue
        #     vardict["evspsbl"].plot(ax=axs[2], color="tab:green", alpha=0.5, _labels=False)
        # axs[2].set_title(f"CMIP6 (n={len(cmip_zm.keys())})")

        # # Subplot 4: Observations + Reanalysis
        # for p in ILAMB_SETTINGS["PRODUCTS"]["et"]:
        #     ilamb_zm[p].plot(ax=axs[3], alpha=0.75, label=p, _labels=False)
        # era_zm["mer"].plot(ax=axs[3], alpha=0.75, label="ERA5", _labels=False)
        # axs[3].legend(ncols=2)
        # axs[3].set_title("OBS + ERA5")

        # axs[0].set_ylabel("ET [W/m2]")
        # axs[2].set_ylabel("ET [W/m2]")

        # axs[2].set_xlabel("Latitude [N]")
        # axs[3].set_xlabel("Latitude [N]")

        # for ax in axs:
        #     ax.set_xlim(-60, 85)
        #     ax.set_xticks([-60, -45, -30, -15, 0, 15, 30, 45, 60, 75])
        #     ax.grid(lw=0.75, color="gray", ls=":")

        # fig.savefig(FIG_ROOT / "all.et.zm.198001-201412.png", bbox_inches="tight", dpi=DPI)
        # plt.close()



        # # ------------------------------------------------------------------
        # # Plot map of time mean ET (map)
        # # ------------------------------------------------------------------
        # print("Plot map of time mean ET")

        # # FHIST PPE
        # fppe_da = fppe_ann["EFLX_LH_TOT_month_1"]
        # for m in fppe_da.member.values:
        #     out = FIG_ROOT / "fppe" / "EFLX_LH_TOT" / f"fppe.CESM2.EFLX_LH_TOT.map.198001-201412.{int(m):02d}.png"
        #     out.parent.mkdir(exist_ok=True, parents=True)
        #     quick_map(fppe_da.sel(member=m).mean(dim="year"), out, title=f"FHIST PPE {int(m):02d}, 198001-201412")
        #     print(out)

        # # GOGA2
        # goga_da = goga_ann["EFLX_LH_TOT_month_1"]
        # for m in goga_da.member:
        #     out = FIG_ROOT / "goga2" / "EFLX_LH_TOT" / f"goga2.CESM2.EFLX_LH_TOT.map.198001-201412.{int(m):02d}.png"
        #     out.parent.mkdir(exist_ok=True, parents=True)
        #     quick_map(goga_da.sel(member=m).mean(dim="year"), out, title=f"GOGA2 {int(m):02d}, 198001-201412")
        #     print(out)

        # # LENS2
        # lens_cmip_da = lens_cmip_ann["EFLX_LH_TOT_month_1"]
        # for m in lens_cmip_da.member:
        #     out = FIG_ROOT / "lens2" / "EFLX_LH_TOT" / f"lens2.CESM2.EFLX_LH_TOT.map.198001-201412.{int(m):02d}.png"
        #     out.parent.mkdir(exist_ok=True, parents=True)
        #     quick_map(lens_cmip_da.sel(member=m).mean(dim="year"), out, title=f"LENS2 {int(m):02d}, 198001-201412")
        #     print(out)

        # # CMIP6
        # for sid, vardict in cmip_ann.items():
        #     if "evspsbl" not in vardict:
        #         print(f"skipping {sid}")
        #         continue
        #     da = vardict["evspsbl"].mean(dim="year")
        #     mid = "onemember"
        #     if ("member" in da.dims):
        #         if ("member_id" in da.coords):
        #             mid = vardict["evspsbl"].member_id[0].item()
        #         da = da.squeeze(dim="member", drop=True)
        #     print(f"{sid}: {da.dims} {da.shape}")
        #     out = FIG_ROOT / "cmip6" / "evspsbl" / f"cmip6.{sid}.evspsbl.map.198001-201412.{mid}.png"
        #     out.parent.mkdir(exist_ok=True, parents=True)
        #     quick_map(da, out, title=f"{sid}, 198001-201412")
        #     print(out)

        # # Observations + Reanalysis
        # for p in ILAMB_SETTINGS["PRODUCTS"]["et"]:
        #     start_year = ilamb_ann[p].year[0].item()
        #     stop_year = ilamb_ann[p].year[-1].item()
        #     out = FIG_ROOT / "obs" / "et" / f"obs.{p}.et.map.{start_year}01-{stop_year}12.png"
        #     out.parent.mkdir(exist_ok=True, parents=True)
        #     quick_map(ilamb_ann[p].mean(dim="year"), out, title=f"{p}, {start_year}01-{stop_year}12")
        #     print(out)
        
        # out = FIG_ROOT / "obs" / "et" / f"obs.era5.mer.map.198001-201412.png"
        # out.parent.mkdir(exist_ok=True, parents=True)
        # quick_map(era_ann["mer"].mean(dim="year"), out, title="ERA5, 198001-201412")
        # print(out)



        # # ------------------------------------------------------------------
        # # Plot timeseries of global mean ET (gmts)
        # # ------------------------------------------------------------------

        # # FHIST PPE
        # fig, ax = plt.subplots(figsize=(6,4), layout="constrained")
        # fppe_da = fppe_gm["EFLX_LH_TOT_month_1"]
        # for m in fppe_da.member.values:
        #     fppe_da.sel(member=m).plot(ax=ax, color="tab:blue", alpha=0.5, _labels=False)
        # ax.set_xlim(1979, 2015)
        # ax.set_xlabel("Year")
        # ax.set_ylabel("ET [W/m2]")
        # ax.set_title(f"CESM2 FHIST PPE (n={len(fppe_da.member)}), Global Mean ET [W/m2]")
        # out = FIG_ROOT / "fppe" / "EFLX_LH_TOT" / f"fppe.CESM2.EFLX_LH_TOT.gmts.198001-201412.png"
        # out.parent.mkdir(exist_ok=True, parents=True)
        # fig.savefig(out, bbox_inches="tight", dpi=DPI)
        # print(out)
        # plt.close()

        # # GOGA2
        # fig, ax = plt.subplots(figsize=(6,4), layout="constrained")
        # goga_da = goga_gm["EFLX_LH_TOT_month_1"]
        # for m in goga_da.member.values:
        #     goga_da.sel(member=m).plot(ax=ax, color="indianred", alpha=0.5, _labels=False)
        # ax.set_xlim(1979, 2015)
        # ax.set_xlabel("Year")
        # ax.set_ylabel("ET [W/m2]")
        # ax.set_title(f"CESM2 GOGA2 (n={len(goga_da.member)}), Global Mean ET [W/m2]")
        # out = FIG_ROOT / "goga2" / "EFLX_LH_TOT" / f"goga2.CESM2.EFLX_LH_TOT.gmts.198001-201412.png"
        # out.parent.mkdir(exist_ok=True, parents=True)
        # fig.savefig(out, bbox_inches="tight", dpi=DPI)
        # print(out)
        # plt.close()

        # # LENS2
        # fig, ax = plt.subplots(figsize=(6,4), layout="constrained")
        # lens_cmip_da = lens_cmip_gm["EFLX_LH_TOT_month_1"]
        # for m in lens_cmip_da.member.values:
        #     lens_cmip_da.sel(member=m).plot(ax=ax, color="tab:blue", alpha=0.5, _labels=False)
        # ax.set_xlim(1979, 2015)
        # ax.set_xlabel("Year")
        # ax.set_ylabel("ET [W/m2]")
        # ax.set_title(f"CESM2 LENS2 (n={len(lens_cmip_da.member)}), Global Mean ET [W/m2]")
        # out = FIG_ROOT / "lens2" / "EFLX_LH_TOT" / f"lens2.CESM2.EFLX_LH_TOT.gmts.198001-201412.png"
        # out.parent.mkdir(exist_ok=True, parents=True)
        # fig.savefig(out, bbox_inches="tight", dpi=DPI)
        # print(out)
        # plt.close()

        # # CMIP6
        # fig, axes = plt.subplots(nrows=7, ncols=6, sharex=True, sharey=True, figsize=(20,16), layout="constrained")
        # axs = axes.ravel()
        # for i, (sid, vardict) in enumerate(cmip_gm.items()):
        #     if "evspsbl" not in vardict:
        #         print(f"skipping {sid}")
        #         continue
        #     vardict["evspsbl"].plot(ax=axs[i], color="tab:green", alpha=0.5, _labels=False)
        #     axs[i].set_xlim(1979, 2015)
        #     axs[i].set_title(sid)
        # for j in range(6):
        #     axes[-1, j].set_xlabel("Year")
        # for i in range(7):
        #     axes[i, 0].set_ylabel("ET [W/m2]")
        # out = FIG_ROOT / "cmip6" / "evspsbl" / f"cmip6.evspsbl.gmts.198001-201412.png"
        # out.parent.mkdir(exist_ok=True, parents=True)
        # fig.savefig(out, bbox_inches="tight", dpi=DPI)
        # print(out)
        # plt.close()

        # # Observations + Reanalysis
        # fig, ax = plt.subplots(figsize=(6,4), layout="constrained")
        # for p in ILAMB_SETTINGS["PRODUCTS"]["et"]:
        #     print(f"{p}: {ilamb_gm[p].dims} {ilamb_gm[p].shape}")
        #     ilamb_gm[p].plot(ax=ax, alpha=0.75, label=p, _labels=False)
        # era_gm["mer"].plot(ax=ax, alpha=0.75, label="ERA5", _labels=False)
        # ax.set_xlim(1979, 2015)
        # ax.legend(ncols=2, fontsize=8)
        # ax.set_xlabel("Year")
        # ax.set_ylabel("ET [W/m2]")
        # ax.set_title(f"Obs + ERA5, Global Mean ET [W/m2]")
        # out = FIG_ROOT / "obs" / "et"/ "obs.et.gmts.198001-201412.png"
        # out.parent.mkdir(exist_ok=True, parents=True)
        # fig.savefig(out, bbox_inches="tight", dpi=DPI)
        # print(out)
        # plt.close()



    finally:
        if client_cluster is not None:
            xclim.close_dask_cluster(client_cluster, remove_std_files=True)


if __name__ == "__main__":
    main()

