"""
Create summary figures for the FHIST PPE.

This script generates comprehensive visualization outputs comparing FHIST Perturbed Parameter
Ensemble (PPE) simulations with CESM2 Large Ensemble (LE) where available. It creates:

1. Panel map plots: Time-averaged spatial patterns for each ensemble member
2. Panel zonal plots: Zonal mean profiles for each ensemble member
3. Panel latband plots: Time series over latitude bands for each ensemble member
4. Combined global plots: Spaghetti plots with violin distributions
5. Combined zonal plots: Ensemble spread of zonal means
6. Combined latband plots: Ensemble spread over latitude bands

Outputs are saved to /glade/work/bbuchovecky/et_controls/fig/ with subdirectories:
- member_panel/map_timemean/
- member_panel/zonal_timemean/
- member_panel/latband_timeseries/
- member_combined/global_timeseries/
- member_combined/zonal_timemean/
- member_combined/latband_timeseries/
"""

from dataclasses import dataclass
from pathlib import Path
import os
from datetime import datetime

import xarray as xr
import matplotlib.pyplot as plt

import xclimate as xclim


@dataclass
class Variable:
    """Variable metadata for CESM output processing.
    
    Attributes
    ----------
    name : str
        CESM variable name (e.g., 'FSH', 'EFLX_LH_TOT')
    long_name : str
        Descriptive name for plotting labels
    gcomp : str
        Component where variable resides ('lnd' or 'atm')
    units : str
        Variable units for axis labels
    derived : str | None
        Formula for derived variables (e.g., 'sqrt(uIVT**2 + vIVT**2)'), None for direct variables
    """
    name: str
    long_name: str
    gcomp: str
    units: str
    derived: str | None


# Configuration
USE_DASK = True  # Whether to use Dask for parallel processing

# Time period for climatological averages
START_TMEAN = "1995-01"
END_TMEAN = "2014-12"
TIME_SLICE = slice(START_TMEAN, END_TMEAN)
YEAR_SLICE = slice(int(START_TMEAN[4:]), int(END_TMEAN[4:]))

# Latitude bands for regional analysis
LATBANDS = {
    "55S-30S": slice(-55, -30),
    "30S-15S": slice(-30, -15),
    "15S-15N": slice(-15, 15),
    "15N-30N": slice(15, 30),
    "30N-65N": slice(30, 65),
}

VARIABLES = {
    # "ALBEDO": Variable("ALBEDO", "surface albedo", "lnd", "1", "(FLDS + FSDS) / (FIRE + FSR)"),
    # "IVT": Variable("IVT", "integrated vapor transport", "atm", "kg/m/s", "sqrt(uIVT**2 + vIVT**2)")
    # "TOTVEGC": Variable("TOTVEGC", "total vegetation carbon, excluding cpool", "lnd", "gC/m2", None), # pool so sum instead of average
    "EFLX_LH_TOT": Variable("EFLX_LH_TOT", "total latent heat flux", "lnd", "W/m2", None),
    # "FSH": Variable("FSH", "sensible heat flux", "lnd", "W/m2", None),
    # "FCEV": Variable("FCEV", "canopy evaporation", "lnd", "W/m2", None),
    # "FCTR": Variable("FCTR", "canopy transpiration", "lnd", "W/m2", None),
    # "FGEV": Variable("FGEV", "ground evaporation", "lnd", "W/m2", None),
    # "TLAI": Variable("TLAI", "total leaf area index", "lnd", "m2/m2", None),
    # "GPP": Variable("GPP", "gross primary productivity", "lnd", "gC/m2/s", None),
    # "QOVER": Variable("QOVER", "surface runoff", "lnd", "mm/s", None),
    # "QRUNOFF": Variable("QRUNOFF", "total runoff", "lnd", "mm/s", None),
    # "VPD_CAN": Variable("VPD_CAN", "canopy vapor pressure deficit", "lnd", "Pa", None),
    # "WIND": Variable("WIND", "wind speed", "lnd", "m/s", None),
    # "CLDLOW": Variable("CLDLOW", "low cloud fraction", "atm", "1", None),
    # "CLDTOT": Variable("CLDTOT", "total cloud fraction", "atm", "1", None),
    # "PRECT": Variable("PRECT", "total precipitation", "atm", "m/s", None),
    # "RAIN_FROM_ATM": Variable("RAIN_FROM_ATM", "rain from atmosphere", "lnd", "mm/s", None),
    # "FLNT": Variable("FLNT", "net longwave flux at top of model", "atm", "W/m2", None),
    # "FSNT": Variable("FSNT", "net shortwave flux at top of model", "atm", "W/m2", None),
    # "uIVT": Variable("uIVT", "zonal integrated vapor transport", "atm", "kg/m/s", None),
    # "vIVT": Variable("vIVT", "meridional integrated vapor transport", "atm", "kg/m/s", None),
}

PATH_LE = Path("/glade/campaign/collections/gdex/data/d651056/CESM2-LE")

OUTDIR = Path("/glade/work/bbuchovecky/et_controls")
OUTDIR_PANEL_MAP = OUTDIR / Path("fig/member_panel/map_timemean")
OUTDIR_PANEL_ZONAL = OUTDIR / Path("fig/member_panel/zonal_timemean")
OUTDIR_PANEL_LATBAND = OUTDIR / Path("fig/member_panel/latband_timeseries")
OUTDIR_COMBINED_GLOBAL = OUTDIR / Path("fig/member_combined/global_timeseries")
OUTDIR_COMBINED_ZONAL = OUTDIR / Path("fig/member_combined/zonal_timemean")
OUTDIR_COMBINED_LATBAND = OUTDIR / Path("fig/member_combined/latband_timeseries")
OUTDIRS = [
    OUTDIR_PANEL_MAP,
    OUTDIR_PANEL_ZONAL,
    OUTDIR_PANEL_LATBAND,
    OUTDIR_COMBINED_GLOBAL,
    OUTDIR_COMBINED_ZONAL,
    OUTDIR_COMBINED_LATBAND,
]
# Create output directories (allow re-running script)
for out in OUTDIRS:
    os.makedirs(out, exist_ok=True)

# Metadata tag for figure attribution
FNAME = Path(__file__).name
NOW = datetime.now().strftime("%Y-%m-%d")
TAG = f"{FNAME} {NOW}"

# Initialize Dask cluster if enabled
CLIENT_CLUSTER = None
if USE_DASK:
    CLIENT_CLUSTER = xclim.create_dask_cluster(
        account="UWAS0155",
        nworkers=5,
        nmem="8GB",
        walltime="02:00:00",
    )

# Load grid information and area weights for spatial averaging
print("Loading grids and gridcell area fields...")
A_FHIST = xr.open_dataset(
    "/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/" \
    "f.e21.FHIST_BGC.f19_f19_mg17.CMIP6-AMIP-2deg.001/atm/proc/tseries/month_1/" \
    "f.e21.FHIST_BGC.f19_f19_mg17.CMIP6-AMIP-2deg.001.cam.h0.AREA.200001-201412.nc",
    decode_timedelta=False)["AREA"].isel(time=0).fillna(0) / 1e6
A_FHIST.attrs["units"] = "km^2"
LND_GRID_FHIST = xclim.load_coupled_fhist_ppe("EFLX_LH_TOT", "lnd", "month_1")[["area", "landfrac"]].isel(member=0).fillna(0)
LND_GRID_FHIST = LND_GRID_FHIST.reindex_like(A_FHIST, method="nearest", tolerance=1e-3)
LA_FHIST = LND_GRID_FHIST.area * LND_GRID_FHIST.landfrac

A_LE = xclim.load_cesm2le("AREA", "atm", "month_1", "h0", keep_var_only=True)["AREA"].isel(member=0, time=0).fillna(0) / 1e6
A_LE.attrs["units"] = "km^2"
LND_GRID_LE = xclim.load_cesm2le("EFLX_LH_TOT", "lnd", "month_1", "h0")[["area", "landfrac"]].isel(member=0, time=0).fillna(0)
LND_GRID_LE = LND_GRID_LE.reindex_like(A_LE, method="nearest", tolerance=1e-3)
LA_LE = LND_GRID_LE.area * LND_GRID_LE.landfrac
print("Done loading grids and gridcell area fields.")


# Process each variable
for _, variable in VARIABLES.items():
    name = variable.name
    gcomp = variable.gcomp
    units = variable.units

    print(name)

    # Load FHIST PPE data
    print(f"\tLoading {name} from FHIST PPE...", end=" ")
    fhist = xclim.load_coupled_fhist_ppe(name, gcomp, "month_1", "h0", keep_var_only=True, chunk={"time": -1})[name].reindex_like(LA_FHIST, method="nearest", tolerance=1e-3).load()
    # Compute perturbation relative to default member (member 0)
    dfhist = fhist.sel(member=slice(1, None)) - fhist.sel(member=0)
    print("Done.")

    das = [fhist]
    das_weights = [LA_FHIST]
    das_labels = ["FHIST PPE"]

    # Load CESM2 LE if available for this component
    le = None  # Initialize to avoid unbound variable warnings
    le_exists = False
    le_tag = ""
    if (PATH_LE / f"{gcomp}/proc/tseries/month_1").exists():
        print(f"\tLoading {name} from CESM2 LE...", end=" ")
        le = xclim.load_cesm2le(name, gcomp, "month_1", "h0", keep_var_only=True, chunk={"time": -1}).sel(time=slice("1950-01", None))[name].reindex_like(LA_LE, method="nearest", tolerance=1e-3).load()
        le_exists = True
        le_tag = "-CESM2LE"
        das.append(le)
        das_weights.append(LA_LE)
        das_labels.append("CESM2 LE")
        print("Done.")

    # Mask ocean grid cells (only plot values over land)
    if gcomp == "atm":
        fhist = fhist.where(LA_FHIST > 0)
        dfhist = dfhist.where(LA_FHIST > 0)
        if le_exists:
            le = le.where(LA_LE > 0)
    
    # 1. Panel map plots of time-averaged perturbations
    print("\tPanel map plots")
    fig, axs = xclim.plot.plot_facetgrid_map(
        da=dfhist.sel(time=TIME_SLICE).mean(dim="time"),
        dim="member",
        label=f"$\\delta${name} [{units}]",
        robust=True,
    )
    fig.suptitle(f"Averaged over {START_TMEAN[:4]}-{END_TMEAN[:4]}, Perturbed $-$ Default, {name}", y=1.025, va="center", ha="center")
    fig.text(y=1.025, x=0.975, s=TAG, fontsize=6, ha="right", va="center")
    plt.savefig(OUTDIR_PANEL_MAP / f"d.global.lnd.{START_TMEAN[:4]}-{END_TMEAN[:4]}.FHIST.{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Panel line plots of zonal mean perturbations
    print("\tPanel line plots - zonal mean")
    fig, axs = xclim.plot.plot_facetgrid_line(
        da=dfhist.sel(time=TIME_SLICE).mean(dim=["time", "lon"]),
        dim="member",
        x="lat",
        xlabel="Latitude [degrees N]",
        ylabel=f"$\\delta${name} [{units}]",
        show_outer_labels=True,
    )
    fig.suptitle(f"Zonal Mean, Averaged over {START_TMEAN[:4]}-{END_TMEAN[:4]}, Perturbed $-$ Default, {name}", y=1.025, va="center", ha="center")
    fig.text(y=1.025, x=0.975, s=TAG, fontsize=6, ha="right", va="center")
    plt.savefig(OUTDIR_PANEL_ZONAL / f"d.zonal.lnd.{START_TMEAN[:4]}-{END_TMEAN[:4]}.FHIST.{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Panel line plots of latitude band time series (perturbations)
    print("\tPanel line plots - latitude band mean:", end=" ")
    for latname, latbnd in LATBANDS.items():
        print(latname, end=" ")
        fig, axs = xclim.plot.plot_facetgrid_line(
            da=dfhist.sel(lat=latbnd).weighted(LA_FHIST).mean(dim=["lon", "lat"]).groupby("time.year").mean(),
            dim="member",
            x="year",
            xlabel="Year",
            ylabel=f"$\\delta${name} [{units}]",
            show_outer_labels=True,
        )
        fig.suptitle(f"Annual Mean, Averaged over {latname}, Perturbed $-$ Default, {name}", y=1.025, va="center", ha="center")
        fig.text(y=1.025, x=0.975, s=TAG, fontsize=6, ha="right", va="center")
        os.makedirs(OUTDIR_PANEL_LATBAND / latname, exist_ok=True)
        plt.savefig(OUTDIR_PANEL_LATBAND / f"{latname}/d.{latname}.lnd.FHIST.{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print()

    # 4. Combined spaghetti plot of global mean time series with violin distribution
    print("\tCombined line plots - global mean")
    # Process data: compute global weighted means and annual averages   
    das_global = []
    das_global_violin = []
    for da, weight in zip(das, das_weights):
        global_mean = da.weighted(weight).mean(dim=["lon", "lat"]).groupby("time.year").mean()
        das_global.append(global_mean)
        global_violin = global_mean.sel(year=YEAR_SLICE).mean(dim="year")
        das_global_violin.append(global_violin)
    
    fig, axs = xclim.plot.plot_ensemble_line(
        das=das_global,
        das_violin=das_global_violin,
        das_labels=das_labels,
        plot_dim="year",
        ylabel=f"{name} [{units}]",
        xlabel="Year",
        violin_xrange=(YEAR_SLICE.start, YEAR_SLICE.stop),
        violin_settings={"x": 2017},
    )
    fig.suptitle(f"Global Mean, Annual Mean, {name}", y=1.025, va="center", ha="center")
    fig.text(y=1.025, x=0.975, s=TAG, fontsize=6, ha="right", va="center")
    plt.savefig(OUTDIR_COMBINED_GLOBAL / f"a.global.lnd.FHIST{le_tag}.{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 5. Combined plot of zonal mean ensemble spread
    print("\tCombined line plots - zonal mean")
    # Process data: compute zonal and time means
    das_zonal = []
    for da in das:
        zonal_mean = da.sel(time=TIME_SLICE).mean(dim=["time", "lon"])
        das_zonal.append(zonal_mean)
    
    fig, axs = xclim.plot.plot_ensemble_line(
        das=das_zonal,
        das_labels=das_labels,
        plot_dim="lat",
        ylabel=f"{name} [{units}]",
        xlabel="Latitude [degrees N]",
    )
    fig.suptitle(f"Zonal Mean, Averaged over {START_TMEAN[:4]}-{END_TMEAN[:4]}, {name}", y=1.025, va="center", ha="center")
    fig.text(y=1.025, x=0.975, s=TAG, fontsize=6, ha="right", va="center")
    plt.savefig(OUTDIR_COMBINED_ZONAL / f"a.zonal.lnd.{START_TMEAN[:4]}-{END_TMEAN[:4]}.FHIST{le_tag}.{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 6. Combined plots of latitude band time series with ensemble spread
    print("\tCombined line plots - latitude band mean:", end=" ")
    for latname, latbnd in LATBANDS.items():
        print(latname, end=" ")
    
        if le_exists:
            weights_latband = [LA_FHIST.sel(lat=latbnd), LA_LE.sel(lat=latbnd)]
        else:
            weights_latband = [LA_FHIST.sel(lat=latbnd)]
        
        # Process data: compute weighted mean over latitude band and annual averages
        das_latband = []
        das_latband_violin = []
        for da, weight in zip(das, weights_latband):
            latband_mean = da.sel(lat=latbnd).weighted(weight).mean(dim=["lon", "lat"]).groupby("time.year").mean()
            das_latband.append(latband_mean)
            latband_violin = latband_mean.sel(year=YEAR_SLICE).mean(dim="year")
            das_latband_violin.append(latband_violin)

        fig, axs = xclim.plot.plot_ensemble_line(
            das=das_latband,
            das_violin=das_latband_violin,
            das_labels=das_labels,
            plot_dim="year",
            ylabel=f"{name} [{units}]",
            xlabel="Year",
            violin_xrange=(YEAR_SLICE.start, YEAR_SLICE.stop),
            violin_settings={"x": 2017},
        )
        fig.suptitle(f"Averaged over {latname} and {START_TMEAN[:4]}-{END_TMEAN[:4]}, {name}", y=1.025, va="center", ha="center")
        fig.text(y=1.025, x=0.975, s=TAG, fontsize=6, ha="right", va="center")
        os.makedirs(OUTDIR_COMBINED_LATBAND / latname, exist_ok=True)
        plt.savefig(OUTDIR_COMBINED_LATBAND / f"{latname}/a.{latname}.lnd.FHIST{le_tag}.{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print()

    # Free memory for next iteration
    print("Cleaning workspace...")
    del fhist, dfhist, das, das_weights, das_labels, das_global, das_zonal
    if le_exists and le is not None:
        del le

# Shutdown Dask cluster if it was initialized
if USE_DASK and CLIENT_CLUSTER is not None:
    xclim.close_dask_cluster(CLIENT_CLUSTER)
