"""
Computed mean ET binned by LAI and aridity.
"""

from __future__ import annotations

from collections.abc import Iterable
import sys
from pathlib import Path
from typing import Sequence, Iterable

import numpy as np
import pandas as pd
import xarray as xr
import regionmask as regmask
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cartopy.crs as ccrs

import xesmf as xe
from ILAMB import ilamblib

from load_cmip_esgf import CMIPESGFLoader
import xclimate as xclim

sys.path.insert(0, str(Path(__file__).parent.parent / "lai_ai_binning"))
from ppe_2d_binning import _build_edges, _bin_stats_single_member



# ------------------------------------------------------------------
# Data to load
# ------------------------------------------------------------------
DO_FPPE = False
DO_GOGA = False
DO_LENS = False
DO_CMIP = True
DO_ILAMB = False
DO_ERA = False


# ------------------------------------------------------------------
# Dask Cluster
# ------------------------------------------------------------------

DASK = True
DASK_WORKERS = 8
DASK_MEMORY = "32GB"
DASK_WALLTIME = "01:00:00"


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROC_ROOTS = {
    "fppe": Path("/glade/work/bbuchovecky/et_unc/proc/fppe"),   # FHIST PPE
    "ippe": Path("/glade/work/bbuchovecky/et_unc/proc/ippe"),   # IHIST PPE
    "goga": Path("/glade/work/bbuchovecky/et_unc/proc/goga2"),  # CESM2 GOGA2
    "lens": Path("/glade/work/bbuchovecky/et_unc/proc/lens2"),  # CESM2 LENS2
    "cmip": Path("/glade/work/bbuchovecky/et_unc/proc/cmip6"),  # CMIP6
    "obs":  Path("/glade/work/bbuchovecky/et_unc/proc/obs"),    # observations (ILAMB)
}
SUBDIRS = [
    "trends",
    "time_mean",
    "zonal_mean",
    "binned",
]
BIN_EDGES_ROOT = Path("/glade/work/bbuchovecky/et_unc/proc/qbin_edges")


# ------------------------------------------------------------------
# Settings for plotting
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Settings for analysis
# ------------------------------------------------------------------

# Binned mean ET plots
N_XBINS = 15
N_YBINS = 15

# Masking and grid things
LF_THRESH = 0.5  # gridcell land fraction threshold
LAT_INTERP_GRID = np.arange(-90, 91, 1)  # 1-deg latitude array to interpolate zonal mean

# Scientific constants
LATENT_HEAT_VAPORIZATION = 2.45e6  # J/kg
LIQ_WATER_DENSITY = 1e3            # kg/m3

# Variables to load
CESM_VARIABLES = ["EFLX_LH_TOT_month_1", "TLAI_month_1", "FLDS_month_1", "FSDS_month_1", "PRECT_month_1"]
CMIP_VARIABLES = ["evspsbl", "lai", "rlds", "rsds", "rlus", "rsus", "pr"]
ILAMB_VARIABLES = ["et", "lai"]
ERA_VARIBLES = ["mer", "msnswrf", "msnlwrf", "mtpr"]


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
    "CATALOG":            Path("/glade/campaign/univ/uwas0155/catalogs/cmip6_all.csv"),
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
# Plotting
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

def data_dict_nybtes(data_dict):
    total_ngb = 0
    sid_ngb = {}
    for sid, vardict in data_dict.items():
        sid_ngb[sid] = 0
        for var, da in vardict.items():
            sid_ngb[sid] += da.nbytes / 1024 / 1024 / 1024
        total_ngb += sid_ngb[sid]
    
    print(f"total: {total_ngb:0.3f} GB")
    for sid, ngb in sid_ngb.items():
        print(f"{sid:20}: {ngb:0.3f} GB")
    

def check_coords(da: xr.DataArray, coords: Iterable[str]) -> bool:
    """Check that da has coords."""
    for coord in coords:
        if coord not in da.coords:
            return False
        if da[coord].ndim == 0:
            continue  # scalar coord counts as present
        if len(da[coord]) == 0:
            return False
    return True


def equal_coords(
    a: xr.DataArray,
    b: xr.DataArray,
    coords: Iterable[str],
    atol: float = 1e-3
) -> bool:
    """Check that a and b have the same coords."""
    if not check_coords(a, coords) or not check_coords(b, coords):
        return False
    for crd in coords:
        if a[crd].shape != b[crd].shape:
            return False
        if crd != "time":  # should actually check if dtype is numeric
            if not np.allclose(a[crd], b[crd], atol=atol):
                return False
    return True


def convert_units(v, da):
    """Convert to water flux units to energy equivalent W/m2."""

    # CESM2: precip from m/s -> W/m2
    if v in ("PRECT_calculated_month_1", "PRECT_month_1", "PRECT_calculated", "PRECT"):
        print("Converting units from m/s -> W/m2")
        da = da * LATENT_HEAT_VAPORIZATION * LIQ_WATER_DENSITY
        da.attrs["units"] = "W/m2"

    # CMIP6, ERA5, ILAMB: precip and et from kg/m2/s -> W/m2
    if v in ("pr", "evspsbl", "mtpr", "mer", "et"):
        print("Converting units from kg/m2/s -> W/m2")
        da = da * LATENT_HEAT_VAPORIZATION
        da.attrs["units"] = "W/m2"

    # ERA5: et sign convention
    if v == "mer":
        da = -1 * da

    return da


def filter_all_variables_available(
    data_dict: dict,
    variables: Sequence[str],
    coords: Iterable[str] | None = None,
) -> dict:
    sid_avail = []
    variables_set = set(variables)

    print("=== Not available ===")
    for sid, vardict in data_dict.items():
        intersection = variables_set - set(vardict.keys())
        if len(intersection) > 0:
            print(f"{sid:16}: missing {intersection}")
            continue

        good_coords = True
        test_v = variables[0]
        test_da = vardict[test_v]
        for v in variables[1:]:
            da = vardict[v]
            if coords is not None:
                if not equal_coords(da, test_da, coords):
                    good_coords = False
                    print(f"{sid:16}: {v} mismatched coords {coords} {test_da.shape} {da.shape}")
                    continue
        if good_coords:
            sid_avail.append(sid)
        
    print(f"\n=== Available: {len(sid_avail)} === ")
    print(sid_avail)

    data_dict_filt = {}
    for sid in sid_avail:
        data_dict_filt[sid] = data_dict[sid]

    return data_dict_filt


def load_fppe_variable(
    variable: str, 
    lat_bnds: slice,
    grid: xr.DataArray,
    mask: xr.DataArray | bool = True,
) -> xr.DataArray:
    if not isinstance(mask, bool) and isinstance(mask, xr.DataArray):
        assert equal_coords(mask, grid, ("lat", "lon"))
        mask = mask.reindex_like(grid, method="nearest", tolerance=1e-3)
    print(f"FPPE: Loading {variable}")
    v = "_".join(variable.split("_")[:-2])
    da = (
        xclim.load_fhist(variable, keep_var_only=True)[v]
        .sel(time=FPPE_SETTINGS["TIME_SLICE"], lat=lat_bnds)
    )
    if not equal_coords(da, grid, ("lat", "lon")):
        raise IndexError("da and grid do not have the same 'lat', 'lon' coordinates")
    if isinstance(mask, xr.DataArray) and not equal_coords(da, mask, ("lat", "lon")):
        raise IndexError("da and grid do not have the same 'lat', 'lon' coordinates")
    da = da.reindex_like(grid, method="nearest", tolerance=1e-3).where(mask)
    return convert_units(variable, da)


def load_goga2_variable(
    variable: str, 
    lat_bnds: slice,
    grid: xr.DataArray,
    mask: xr.DataArray | bool = True,
) -> xr.DataArray:
    if not isinstance(mask, bool) and isinstance(mask, xr.DataArray):
        assert equal_coords(mask, grid, ("lat", "lon"))
        mask = mask.reindex_like(grid, method="nearest", tolerance=1e-3)
    print(f"GOGA2: Loading {variable}")
    v = "_".join(variable.split("_")[:-2])
    frq = "_".join(variable.split("_")[-2:])
    da = (
        xclim.load_goga2(v, GOGA_SETTINGS["GCOMP"], frq, GOGA_SETTINGS["STREAM"])[v]
        .sel(time=GOGA_SETTINGS["TIME_SLICE"], lat=lat_bnds)
    )
    if not equal_coords(da, grid, ("lat", "lon")):
        raise IndexError("da and grid do not have the same 'lat', 'lon' coordinates")
    if isinstance(mask, xr.DataArray) and not equal_coords(da, mask, ("lat", "lon")):
        raise IndexError("da and grid do not have the same 'lat', 'lon' coordinates")
    da = da.reindex_like(grid, method="nearest", tolerance=1e-3).where(mask)
    return convert_units(variable, da)


def load_lens2_variable(
    variable: str, 
    lat_bnds: slice,
    grid: xr.DataArray,
    mask: xr.DataArray | bool = True,
    bb: str = "cmip6",
) -> xr.DataArray:
    if not isinstance(mask, bool) and isinstance(mask, xr.DataArray):
        assert equal_coords(mask, grid, ("lat", "lon"))
        mask = mask.reindex_like(grid, method="nearest", tolerance=1e-3)
    print(f"LENS2: Loading {variable}")
    v = "_".join(variable.split("_")[:-2])
    frq = "_".join(variable.split("_")[-2:])
    da = (
        xclim.load_cesm2le(v, LENS_SETTINGS["GCOMP"], frq, LENS_SETTINGS["STREAM"], bb=bb)[v]
        .sel(time=LENS_SETTINGS["TIME_SLICE"], lat=lat_bnds)
    )
    if not equal_coords(da, grid, ("lat", "lon")):
        raise IndexError("da and grid do not have the same 'lat', 'lon' coordinates")
    if isinstance(mask, xr.DataArray) and not equal_coords(da, mask, ("lat", "lon")):
        raise IndexError("da and grid do not have the same 'lat', 'lon' coordinates")
    da = da.reindex_like(grid, method="nearest", tolerance=1e-3).where(mask)
    return convert_units(variable, da)


def load_ilamb_variable(
    product: str,
    variable: str,  
    lat_bnds: slice,
    mask: xr.DataArray | bool = True,
) -> xr.DataArray:
    path = ILAMB_SETTINGS["ROOT"] / variable / f"{variable}_{product}.nc"
    print(f"ILAMB: Loading {variable}")
    print(path)
    da = (
        xr.open_dataset(path)[variable]
        .sel(time=ILAMB_SETTINGS["TIME_SLICE"], lat=lat_bnds)
        .where(mask)
    )
    return convert_units(variable, da)


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
    da = (
        ds[v]
        .sel(time=ERA_SETTINGS["TIME_SLICE"], lat=lat_bnds)
        .where(mask)
    )
    return convert_units(variable, da)


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
        # ------------------------------------------------------------------
        # Load FPPE
        # ------------------------------------------------------------------
        fppe_time_str = (
            f"{FPPE_SETTINGS['TIME_SLICE'].start.replace('-', '')}-"
            f"{FPPE_SETTINGS['TIME_SLICE'].stop.replace('-', '')}"
        )
        print("Load FHIST PPE")
        fppe_grid = xclim.load_fhist_ppe_grid()
        fppe = {}
        for v in CESM_VARIABLES:
            fppe[v] = load_fppe_variable(v, slice(-90, 90), fppe_grid, mask=fppe_grid.LANDFRAC>LF_THRESH)


        # ------------------------------------------------------------------
        # Load GOGA2
        # ------------------------------------------------------------------
        goga_time_str = (
            f"{GOGA_SETTINGS['TIME_SLICE'].start.replace('-', '')}-"
            f"{GOGA_SETTINGS['TIME_SLICE'].stop.replace('-', '')}"
        )
        print("Load GOGA2")
        goga_grid = xclim.load_goga2_grid()
        goga = {}
        for v in CESM_VARIABLES:
            goga[v] = load_goga2_variable(v, slice(-90, 90), goga_grid, mask=goga_grid.LANDFRAC>LF_THRESH)


        # ------------------------------------------------------------------
        # Load LENS2
        # ------------------------------------------------------------------
        lens_time_str = (
            f"{LENS_SETTINGS['TIME_SLICE'].start.replace('-', '')}-"
            f"{LENS_SETTINGS['TIME_SLICE'].stop.replace('-', '')}"
        )
        print("Load LENS2")
        lens_grid = xclim.load_cesm2le_grid()
        lens_cmip = {}
        for bb, dct in zip(["cmip6"], [lens_cmip]):
            for v in CESM_VARIABLES:
                dct[v] = load_lens2_variable(v, slice(-90, 90), lens_grid, mask=lens_grid.LANDFRAC>LF_THRESH, bb=bb)


        # ------------------------------------------------------------------
        # Load CMIP6
        # ------------------------------------------------------------------
        cmip_time_str = (
            f"{CMIP_SETTINGS['TIME_SLICE'].start.replace('-', '')}-"
            f"{CMIP_SETTINGS['TIME_SLICE'].stop.replace('-', '')}"
        )

        loader = CMIPESGFLoader(CMIP_SETTINGS["CATALOG"])
        cmip = loader.load_data(
            variables=CMIP_VARIABLES + CMIP_SETTINGS["GRID_VARIABLES"],
            experiment_id=CMIP_SETTINGS["EXPERIMENT_ID"],
            source_id=CMIP_SETTINGS["SOURCE_IDS"],
            omit_source_id=CMIP_SETTINGS["OMIT_SOURCE_IDS"],
            member_id=CMIP_SETTINGS["MEMBER_ID"],
            time_slice=CMIP_SETTINGS["TIME_SLICE"],
            parallel=True,
            compute=False,
        )
        cmip = filter_all_variables_available(cmip, CMIP_VARIABLES, coords=("lat", "lon", "time"))
        cmip = filter_all_variables_available(cmip, CMIP_SETTINGS["GRID_VARIABLES"], coords=None)

        print("Compute land area")
        for sid, vardict in cmip.items():
            if len(vardict.keys()) > 0:
                cmip[sid]["sftlf"] = cmip[sid]["sftlf"].sel(lat=LAT_BNDS)
                cmip[sid]["areacella"] = cmip[sid]["areacella"].sel(lat=LAT_BNDS)

                cmip[sid]["lf"] = cmip[sid]["sftlf"].reindex_like(cmip[sid]["areacella"], method="nearest", tolerance=1e-3)
                cmip[sid]["la"] = cmip[sid]["areacella"] * cmip[sid]["lf"]
                cmip[sid]["mask"] = mask_greenland(cmip[sid]["lf"], LF_THRESH)
                del cmip[sid]["sftlf"]

        print("Reindex analysis variables to atmospheric grid")
        for sid, vardict in cmip.items():
            print(f"=== {sid} ===")
            for v, da in vardict.items():
                vardict[v] = (
                    da.sel(lat=LAT_BNDS)
                    .reindex_like(cmip[sid]["mask"], method="nearest", tolerance=1e-3)
                    .where(cmip[sid]["mask"])
                )
                vardict[v] = convert_units(v, vardict[v])
                print(f"{v:12}: {vardict[v].dims} {vardict[v].shape}")



        # ------------------------------------------------------------------
        # Load ILAMB
        # ------------------------------------------------------------------
        # Load variables
        ilamb = {}
        ilamb_la = {}
        for v in ILAMB_VARIABLES:
            for p in ILAMB_SETTINGS["PRODUCTS"][v]:
                ilamb[p] = load_ilamb_variable(p, v, slice(-90, 90))
                ilamb[p] = _convert_units(v, ilamb[p])
                _, ilamb_la[p] = compute_cell_area(ilamb[p])

        # Create land and missing data mask
        ilamb_mask = {}
        for v in ILAMB_VARIABLES:
            ilamb_mask[v] = xr.ones_like(ilamb[ILAMB_SETTINGS["PRODUCTS"][v][0]].isel(time=0))
            for p in ILAMB_SETTINGS["PRODUCTS"][v]:
                ilamb_mask[v] = xr.where(np.isnan(ilamb[p].isel(time=0)), 0, ilamb_mask[v])

        # Mask variables
        for v in ILAMB_VARIABLES:
            for p in ILAMB_SETTINGS["PRODUCTS"][v]:
                ilamb[p] = ilamb[p].where(ilamb_mask[v], other=0).where(ilamb_la[p]>0)
                print(f"{p} nans: {np.sum(np.isnan(ilamb[p].isel(time=0))).values}")
    

        # ------------------------------------------------------------------
        # Load ERA5
        # ------------------------------------------------------------------
        # !! This requires ERA5 regridded to ILAMB 0.5-degree grid !!
        v_grid = ILAMB_VARIABLES[0]
        p_grid = ILAMB_SETTINGS["PRODUCTS"][v_grid][0]
        obs_area, obs_la = compute_cell_area(ilamb[p_grid])
        obs_lf = obs_la / obs_area

        era = {}
        for v in ERA_VARIBLES:
            era[v] = load_era5_variable(v, slice(-90, 90), mask=obs_lf>LF_THRESH)
            era[v] = _convert_units(v, era[v])


        # ------------------------------------------------------------------
        # Subset time and compute the annual mean --> use this onwards
        # ------------------------------------------------------------------
        print("Computing annual mean and persisting")

        fppe_ann = {}
        for v, da in fppe.items():
            fppe_ann[v] = compute_annual_mean(da.sel(time=FPPE_SETTINGS["TIME_SLICE"])).persist()

        goga_ann = {}
        for v, da in goga.items():
            goga_ann[v] = compute_annual_mean(da.sel(time=GOGA_SETTINGS["TIME_SLICE"])).persist()

        lens_cmip_ann = {}
        for v, da in lens_cmip.items():
            lens_cmip_ann[v] = compute_annual_mean(da.sel(time=LENS_SETTINGS["TIME_SLICE"])).persist()

        cmip_ann = {}
        for sid, vardict in cmip.items():
            cmip_ann[sid] = {}
            for v, da in vardict.items():
                cmip_ann[sid][v] = compute_annual_mean(da.sel(time=CMIP_SETTINGS["TIME_SLICE"])).persist()

        ilamb_ann = {}
        for p, da in ilamb.items():
            ilamb_ann[p] = compute_annual_mean(da.sel(time=ILAMB_SETTINGS["TIME_SLICE"])).persist()

        era_ann = {}
        for v, da in era.items():
            era_ann[v] = compute_annual_mean(da.sel(time=ERA_SETTINGS["TIME_SLICE"])).persist()


        # --------------------------------------------------------------------
        # Compute mean ET binned by LAI and aridity
        # --------------------------------------------------------------------
        
        print("Test different LAI thresholds")

        cmip_lai_max = {}
        for sid, vardict in cmip.items():
            cmip_lai_max[sid] = vardict["lai"].groupby("time.month").mean().max(dim="month").compute()




    finally:
        if client_cluster is not None:
            xclim.close_dask_cluster(client_cluster, remove_std_files=True)


if __name__ == "__main__":
    main()
