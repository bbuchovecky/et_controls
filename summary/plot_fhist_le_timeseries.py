"""Create fancy timeseries plots of the FHIST PPE and CESM2 LE."""

from pathlib import Path
import xarray as xr
import pandas as pd
import matplotlib.colors as mcolors

import xclimate as xclim
import plotting_moved_to_xclimate as myplt


VARIABLES = {
    # "FSH": "W m$^{-2}$",
    # "FCTR": "W m$^{-2}$",
    # "FCEV": "W m$^{-2}$",
    # "FGEV": "W m$^{-2}$",
    "TLAI": "m$^2$ m$^{-2}$",
    "TSA": "degree C",
}

GRID_FHIST = xclim.load_coupled_fhist_ppe("EFLX_LH_TOT", "lnd", "month_1")[["area", "landfrac"]].isel(member=0).fillna(0)
LA_FHIST = GRID_FHIST.area * GRID_FHIST.landfrac

GRID_LE = xclim.load_cesm2le("EFLX_LH_TOT", "lnd", "month_1", "h0")[["area", "landfrac"]].isel(member=0, time=0).fillna(0)
LA_LE = GRID_LE.area * GRID_LE.landfrac

for v, units in VARIABLES.items():
    print(v)

    fhist = xclim.load_coupled_fhist_ppe(v, "lnd", "month_1", stream="h0", keep_var_only=True)[v]
    le = xclim.load_cesm2le(v, "lnd", "month_1", "h0", keep_var_only=True).sel(time=slice("1950-01", None))[v]

    if v == "TSA":
        fhist = fhist - 273.15
        le = le - 273.15
    
    fig, ax = myplt.plot_fancy_timeseries(
        das=[fhist, le],
        das_weights=[LA_FHIST, LA_LE],
        das_labels=["FHIST PPE", "CESM2 LE"],
        ylabel=f"{v} [{units}]",
        xlabel="Year",
        title=f"Land Global Mean {v}: CESM2 FHIST PPE and CESM2 LE",
        member_coord="member",
        colors=["tab:blue", "tab:orange"],
        highlight_member=[0, None],
    )

    fig.savefig(f"fig/global.timeseries.FHIST-CESM2LE.{v}.png", dpi=300)



v = "EFLX_LH_TOT"
ilamb_v = "hfls"

## GET OUTPUT

# Ensembles
fhist = xclim.load_coupled_fhist_ppe(v, "lnd", "month_1", stream="h0", keep_var_only=True)[v]
le = xclim.load_cesm2le(v, "lnd", "month_1", "h0", keep_var_only=True).sel(time=slice("1950-01", None))[v]

# ILAMB
indir = Path("/glade/work/bbuchovecky/CPL_PPE_CO2/select_parameters/ILAMB_data/processed")
ilamb_zm = xr.open_dataset(indir / f"zonal_mean/{ilamb_v.upper()}_ZONAL_MEAN_ILAMB_2003-2009.nc")["et_itzavg_overlap"]
ilamb_gm = {}
for f in indir.glob("global_mean/*.nc"):
    key = f.stem.split("_")[3]
    ilamb_gm[key] = xr.open_dataset(f)[ilamb_v.lower()+"_"+key]

# CLM6 PPE
# Load parameter settings key
clm6_key = pd.read_csv("/glade/campaign/cgd/tss/projects/PPE/ctsm6_oaat/ctsm6_oaat_key.csv")

# Load postprocessed temporally averaged dataset 
in_file = Path("/glade/campaign/cgd/tss/projects/PPE/ctsm6_oaat/postp/ctsm6_oaat_postp_1985-2023.nc")
clm6_tm = xr.open_dataset(in_file).rename({"ens": "member"})

# Append key to ds
clm6_tm = clm6_tm.assign_coords(
    key=("member", clm6_key["key"].values),
    param=("member", clm6_key["param"].values),
    minmax=("member", clm6_key["minmax"].values)
)
clm6_tm = clm6_tm.set_index(member=["param", "minmax", "key"], append=True)
clm6_tm = clm6_tm[[vv for vv in clm6_tm.data_vars if str(vv).startswith(v)]]


# Load postprocessed timeseries dataset
in_file = Path("/glade/campaign/cgd/tss/projects/PPE/ctsm6_oaat/postp/ctsm6_oaat_postp_timeseries_1901-2023.nc")
clm6_ts = xr.open_dataset(in_file).rename({"ens": "member"})

# Append key to ds
clm6_ts = clm6_ts.assign_coords(
    key=("member", clm6_key["key"].values),
    param=("member", clm6_key["param"].values),
    minmax=("member", clm6_key["minmax"].values)
)
clm6_ts = clm6_ts.set_index(member=["param", "minmax", "key"], append=True)
clm6_ts = clm6_ts[[vv for vv in clm6_ts.data_vars if str(vv).startswith(v)]]

## PLOTTING

# Plot the ensembles
fig, ax = myplt.plot_fancy_timeseries(
    das=[fhist, le],
    das_weights=[LA_FHIST, LA_LE],
    das_labels=["FHIST PPE", "CESM2 LE"],
    ylabel=f"{v} [W m$^{{-2}}$]",
    xlabel="Year",
    title=f"Land Global Mean {v}: CESM2 FHIST PPE, CESM2 LE, CLM6 IHIST PPE, and ILAMB",
    member_coord="member",
    colors=["tab:blue", "tab:orange"],
    highlight_member=[0, None],
)

## Manual plotting from here on
vp_xpos = 2026

# Plot the CLM6 PPE
for i, e in enumerate(clm6_ts.member):
    label = None
    if i == 0:
        label = "CLM6 PPE"
    clm6_ts[f"{v}_global_mean"].sel(member=e).plot(ax=ax, color="silver", alpha=0.3, lw=1, label=label, _labels=False, zorder=0)

# Highlight the CLM6 default
clm6_ts[f"{v}_global_mean"].sel(param="default", minmax="max").plot(ax=ax, color="k", lw=1, ls="-", label="CLM6 default", _labels=False, zorder=1)
ax.scatter(
    vp_xpos+3,
    clm6_ts[f"{v}_global_mean"].sel(param="default", minmax="max").mean(dim="year"),
    s=15,
    marker="o",
    color="k",
)

# Add a violin plot for the time average of the CLM6 PPE ensemble
vp = ax.violinplot(
    clm6_ts[f"{v}_global_mean"].mean(dim="year"),
    [vp_xpos],
    vert=True,
    widths=15,
    side="high",
    showmeans=False,
    showextrema=True,
    showmedians=True,
)
vp["bodies"][0].set(facecolor="gray")
vp["cbars"].set(linewidth=0)
vp["cmedians"].set(linewidth=1, color="dimgray")
vp["cmins"].set(linewidth=1, color="dimgray")
vp["cmaxes"].set(linewidth=1, color="dimgray")

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


# Plot the ILAMB datasets
cs_ilamb = list(mcolors.TABLEAU_COLORS.keys())[2:]ˇ
for i, (key, da) in enumerate(ilamb_gm.items()):
    ax.plot(da.year, da, c=cs_ilamb[i], ls="-", lw=2.5, label=key, zorder=100)
    ax.scatter(vp_xpos+3, da.mean(dim="year"), marker="o", s=15, edgecolor=cs_ilamb[i], facecolor="none")

ax.legend(loc="lower left", ncol=2, fontsize=8)
fig.savefig(f"fig/global.timeseries.FHIST-CESM2LE-CLM6PPE-ILAMB.{v}.png", dpi=300)
