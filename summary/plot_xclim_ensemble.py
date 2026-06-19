"""Create plots using xclimate utilities."""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import xarray as xr
import matplotlib.pyplot as plt

import xclimate as xclim


@dataclass
class Variable:
    """Variable dataclass."""
    name: str
    long_name: str
    gcomp: str
    units: str
    derived: str | None

# Metadata tag for figure attribution
FNAME = Path(__file__).name
NOW = datetime.now().strftime("%Y-%m-%d")
TAG = f"{FNAME} {NOW}"

# Time period for climatological averages
START_TMEAN = "1995-01"
END_TMEAN = "2014-12"
TIME_SLICE = slice(START_TMEAN,END_TMEAN)
YEAR_SLICE = slice(int(START_TMEAN[:4]), int(END_TMEAN[:4]))

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
    # "TOTVEGC": Variable("TOTVEGC", "total vegetation carbon, excluding cpool", "lnd", "gC/m2", None), # pool - sum instead of average

    # "EFLX_LH_TOT": Variable("EFLX_LH_TOT", "total latent heat flux", "lnd", "W/m2", None),
    "FSH": Variable("FSH", "sensible heat flux", "lnd", "W/m2", None),
    "FCEV": Variable("FCEV", "canopy evaporation", "lnd", "W/m2", None),
    "FCTR": Variable("FCTR", "canopy transpiration", "lnd", "W/m2", None),
    "FGEV": Variable("FGEV", "ground evaporation", "lnd", "W/m2", None),
    "TLAI": Variable("TLAI", "total leaf area index", "lnd", "m2/m2", None),
    "GPP": Variable("GPP", "gross primary productivity", "lnd", "gC/m2/s", None),
    "QOVER": Variable("QOVER", "surface runoff", "lnd", "mm/s", None),
    "QRUNOFF": Variable("QRUNOFF", "total runoff", "lnd", "mm/s", None),
    "VPD_CAN": Variable("VPD_CAN", "canopy vapor pressure deficit", "lnd", "Pa", None),
    "WIND": Variable("WIND", "wind speed", "lnd", "m/s", None),
    "RAIN_FROM_ATM": Variable("RAIN_FROM_ATM", "rain from atmosphere", "lnd", "mm/s", None),
    "CLDLOW": Variable("CLDLOW", "low cloud fraction", "atm", "1", None),
    "CLDTOT": Variable("CLDTOT", "total cloud fraction", "atm", "1", None),
    
    # "PRECT": Variable("PRECT", "total precipitation", "atm", "m/s", None),
    # "FLNT": Variable("FLNT", "net longwave flux at top of model", "atm", "W/m2", None),
    # "FSNT": Variable("FSNT", "net shortwave flux at top of model", "atm", "W/m2", None),
    # "uIVT": Variable("uIVT", "zonal integrated vapor transport", "atm", "kg/m/s", None),
    # "vIVT": Variable("vIVT", "meridional integrated vapor transport", "atm", "kg/m/s", None),
}

PATH_LE = Path("/glade/campaign/collections/gdex/data/d651056/CESM2-LE")

OUTDIR = Path("/glade/work/bbuchovecky/et_controls")
OUTDIR_PANEL_MAP = OUTDIR / Path("fig/member_panel/map_timemean")
OUTDIR_PANEL_ZONAL = OUTDIR / Path("fig/member_panel/zonal_timemean")
OUTDIR_COMBINED_GLOBAL = OUTDIR / Path("fig/member_combined/global_timeseries")
OUTDIR_COMBINED_LATBAND = OUTDIR / Path("fig/member_combined/latband_timeseries")

CLIENT_CLUSTER = xclim.create_dask_cluster(
        account="UWAS0155",
        nworkers=4,
        nmem="4GB",
        walltime="01:00:00",
    )

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


AREA = [A_FHIST, A_LE]
LA = [LA_FHIST, LA_LE]


# Process each variable
for _, variable in VARIABLES.items():
    name = variable.name
    gcomp = variable.gcomp
    units = variable.units

    print(name)

    fhist = xclim.load_coupled_fhist_ppe(name, gcomp, "month_1", "h0", keep_var_only=True, chunk=True)[name]
    fhist = fhist.reindex_like(A_FHIST, method="nearest", tolerance=1e-3)
    dfhist = fhist.sel(member=slice(1, None)) - fhist.sel(member=0)
    das = [fhist]
    das_labels = ["FHIST PPE"]
    highlight_member = [0]
    la = [LA_FHIST]

    le = None
    le_tag = ""
    if (PATH_LE / f"{gcomp}/proc/tseries/month_1").exists():
        le_tag = "-CESM2LE"
        le = xclim.load_cesm2le(name, gcomp, "month_1", "h0", keep_var_only=True, chunk={"time": -1})[name]
        le = le.sel(time=slice("1950-01", None)).reindex_like(A_LE, method="nearest", tolerance=1e-3)
        das.append(le)
        das_labels.append("CESM2 LE")
        highlight_member += [None]
        la.append(LA_LE)

    das_global = []
    das_violin = []
    for i, (da, w) in enumerate(zip(das, la)):
        das_global.append(da.weighted(w).mean(dim=["lat", "lon"]).groupby("time.year").mean())
        das_violin.append(das_global[i].sel(year=YEAR_SLICE).mean(dim="year"))

        print(f"da {i}: {das[i].dims} {das[i].shape}")
        print(f"violin {i}: {das_violin[i].dims}, {das_violin[i].shape}")

    # 1. Absolute, global mean, timeseries
    fig, ax = xclim.plot.plot_ensemble_line(
        das=das_global,
        das_violin=das_violin,
        das_labels=das_labels,
        ylabel=f"{name} [{units}]",
        plot_dim="year",
        xlabel="Year",
        highlight_member=highlight_member,
        violin_xrange=(YEAR_SLICE.start, YEAR_SLICE.stop),
        violin_settings={"x": YEAR_SLICE.stop + 2},
        add_legend=True
    )
    fig.suptitle(f"Global Land Mean, Annual Mean Timeseries, {name} [{units}]", x=0.065, y=0.96, ha="left")
    fig.text(y=1.025, x=0.975, s=TAG, fontsize=6, ha="right", va="center")
    plt.tight_layout()
    plt.savefig(OUTDIR_COMBINED_GLOBAL / f"a.global.lnd.timeseries.FHIST{le_tag}.{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Perturbation, panel zonal mean, time mean
    fig, axs = xclim.plot.plot_facetgrid_line(
        da=dfhist.sel(time=TIME_SLICE).mean(dim=["time", "lon"]),
        dim="member",
        x="lat",
        xlabel="Latitude [degrees N]",
        ylabel=f"$\\Delta${name} [{units}]",
        show_outer_labels=True,
    )
    fig.suptitle(f"Perturbed $-$ Default, Zonal Land Mean, Time Mean {YEAR_SLICE.start}-{YEAR_SLICE.stop}, {name} [{units}]", x=0.065, y=0.96, ha="left")
    fig.text(y=1.025, x=0.975, s=TAG, fontsize=6, ha="right", va="center")
    plt.tight_layout()
    plt.savefig(OUTDIR_PANEL_ZONAL / f"d.zonal.lnd.{YEAR_SLICE.start}-{YEAR_SLICE.stop}.FHIST.{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Perturbation, panel map, time mean
    fig, axs = xclim.plot.plot_facetgrid_map(
        da=dfhist.sel(time=TIME_SLICE).mean(dim="time"),
        dim="member",
        label=f"$\\Delta${name} [{units}]",
        robust=True,
    )
    fig.suptitle(f"Perturbed $-$ Default, Time Mean {YEAR_SLICE.start}-{YEAR_SLICE.stop}, {name} [{units}]", y=1.025, va="center", ha="center")
    fig.text(y=1.025, x=0.975, s=TAG, fontsize=6, ha="right", va="center")
    plt.savefig(OUTDIR_PANEL_MAP / f"d.global.lnd.{YEAR_SLICE.start}-{YEAR_SLICE.stop}.FHIST.{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # # 4. Absolute, laitude band mean, timeseries
    # for latname, latbnd in LATBANDS.items():
    #     if le is not None:
    #         weights_latbnd = [LA_FHIST.sel(lat=latbnd), LA_LE.sel(lat=latbnd)]
    #     else:
    #         weights_latbnd = [LA_FHIST.sel(lat=latbnd)]
        
    #     das_latbnd = []
    #     das_latbnd_violin = []
    #     for i, (da, w) in enumerate(zip(das, weights_latbnd)):
    #         das_latbnd.append(da.sel(lat=latbnd).weighted(w).mean(dim=["lat", "lon"]).groupby("time.year").mean())
    #         da_latbnd_violin = das_latbnd[i].sel(year=slice(YEAR_SLICE.start, YEAR_SLICE.stop)).mean(dim="year")
            
    #         # Debug: Check for NaN values before dropna
    #         print(f"  Latband {latname} violin {i} before dropna: shape={da_latbnd_violin.shape}, has_nans={da_latbnd_violin.isnull().any().values}, n_nans={da_latbnd_violin.isnull().sum().values}")
            
    #         # Drop NaN values to prevent violin plot errors
    #         da_latbnd_violin = da_latbnd_violin.dropna(dim="member")
    #         das_latbnd_violin.append(da_latbnd_violin)
            
    #         print(f"  Latband {latname} violin {i} after dropna: shape={das_latbnd_violin[i].shape}, min={das_latbnd_violin[i].min().values:.3f}, max={das_latbnd_violin[i].max().values:.3f}")
        
    #     fig, ax = xclim.plot.plot_ensemble_line(
    #         das=das_latbnd,
    #         das_violin=das_latbnd_violin,
    #         das_labels=das_labels,
    #         ylabel=f"{name} [{units}]",
    #         plot_dim="year",
    #         xlabel="Year",
    #         highlight_member=[0, None],
    #         violin_xrange=(YEAR_SLICE.start, YEAR_SLICE.stop),
    #         violin_settings={"x": YEAR_SLICE.stop + 2},
    #         add_legend=True
    #     )
    #     fig.suptitle(f"{latname} Land Mean, Annual Mean Timeseries, {name} [{units}]", x=0.065, y=0.96, ha="left")
    #     fig.text(y=1.025, x=0.975, s=TAG, fontsize=6, ha="right", va="center")
    #     plt.tight_layout()
    #     plt.savefig(OUTDIR_COMBINED_LATBAND / f"a.{latname}.lnd.timeseries.FHIST{le_tag}.{name}.png", dpi=300, bbox_inches="tight")
    #     plt.close(fig)


if CLIENT_CLUSTER is not None:
    xclim.close_dask_cluster(CLIENT_CLUSTER)
