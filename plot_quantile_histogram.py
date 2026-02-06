#!/glade/work/bbuchovecky/miniforge3/envs/data-sci/bin/python3.14

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import xclimate as xclim


# Important parameters:

# -- science --
NONGLC_PCT_THRESHOLD = 80
TIME_SLICE = slice("1995-01", "2014-12")
NB = 15
Z_VAR = "ET"
XB_VAR = "WDFRQ"
YB_VAR = "TLAI"

# -- plotting --
VABS = 300  # abs(colormap limits)
REF_COUNT = 300  # reference bin count for scaling circles
SCALE_COUNT = 200  # count for scale circle next to colorbar

NCOLS = 6
NROWS = 5
FIGSIZE = (14, 13)

# -- dask --
CLIENT_CLUSTER = xclim.create_dask_cluster(
    account="UWAS0155",
    nworkers=5,
    nmem="4GB",
    walltime="01:00:00",
)

###

A_FHIST = xr.open_dataset(
    "/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/" \
    "f.e21.FHIST_BGC.f19_f19_mg17.CMIP6-AMIP-2deg.001/atm/proc/tseries/month_1/" \
    "f.e21.FHIST_BGC.f19_f19_mg17.CMIP6-AMIP-2deg.001.cam.h0.AREA.200001-201412.nc",
    decode_timedelta=False)["AREA"].isel(time=0).fillna(0) / 1e6
A_FHIST.attrs["units"] = "km^2"

LND_GRID_FHIST = xclim.load_coupled_fhist_ppe("EFLX_LH_TOT", "lnd", "month_1")[["area", "landfrac"]].isel(member=0).fillna(0)
LND_GRID_FHIST = LND_GRID_FHIST.reindex_like(A_FHIST, method="nearest", tolerance=1e-3)
LA_FHIST = LND_GRID_FHIST.area * LND_GRID_FHIST.landfrac



PCT_LANDUNIT = xclim.load_coupled_fhist_ppe("PCT_LANDUNIT", "lnd", "month_1").reindex_like(A_FHIST, method="nearest", tolerance=1e-3)
LANDUNIT_TYPE = np.empty(9, dtype=object)
for key, item in PCT_LANDUNIT.attrs.items():
    if "ltype" in key:
        LANDUNIT_TYPE[item-1] = key
PCT_LANDUNIT = PCT_LANDUNIT.assign_coords(ltype=np.arange(1,10), landunit=("ltype", LANDUNIT_TYPE))["PCT_LANDUNIT"]
NONGLC_MASK = PCT_LANDUNIT.isel(member=0, time=0, ltype=3) < NONGLC_PCT_THRESHOLD


# Load variables
FHIST = {}
FHIST["PRECT_day_1"] = xclim.load_coupled_fhist_ppe("PRECT", "atm", "day_1", "h2", keep_var_only=True)["PRECT"].sel(time=TIME_SLICE).persist()
FHIST["PRECC_month_1"] = xclim.load_coupled_fhist_ppe("PRECC", "atm", "month_1", "h0", keep_var_only=True)["PRECC"].sel(time=TIME_SLICE).persist()
FHIST["PRECL_month_1"] = xclim.load_coupled_fhist_ppe("PRECL", "atm", "month_1", "h0", keep_var_only=True)["PRECL"].sel(time=TIME_SLICE).persist()
FHIST["TLAI_month_1"] = xclim.load_coupled_fhist_ppe("TLAI", "lnd", "month_1", "h0", keep_var_only=True)["TLAI"].sel(time=TIME_SLICE).reindex_like(A_FHIST, method="nearest", tolerance=1e-3).persist()
FHIST["ET_month_1"] = xclim.load_coupled_fhist_ppe("EFLX_LH_TOT", "lnd", "month_1", "h0", keep_var_only=True)["EFLX_LH_TOT"].sel(time=TIME_SLICE).reindex_like(A_FHIST, method="nearest", tolerance=1e-3).persist()

# Monthly precipitation rate (PRECC + PRECL)
FHIST["PRECT_month_1"] = FHIST["PRECC_month_1"] + FHIST["PRECL_month_1"]

# Annual mean TLAI
FHIST["TLAI_year_1"] = FHIST["TLAI_month_1"].sel(time=TIME_SLICE).groupby("time.year").map(lambda x: x.weighted(x.time.dt.days_in_month).mean("time"))

# Annual mean ET
FHIST["ET_year_1"] = FHIST["ET_month_1"].sel(time=TIME_SLICE).groupby("time.year").map(lambda x: x.weighted(x.time.dt.days_in_month).mean("time"))

# Wet-day frequency as defined in Feldman et al. (2024) - the annual number of days with above 1 mm/day of precipitation
DAILY_PRECT_THRESH =  1 / (1000 * 24 * 60 * 60)  # [m/s] = 1 [mm/day]
FHIST["WDFRQ_year_1"] = (FHIST["PRECT_day_1"].sel(time=TIME_SLICE).where(LA_FHIST>0) > DAILY_PRECT_THRESH).groupby("time.year").sum()

# Annual difference between highest and lowest precipitation months
FHIST["PRDIFF_year_1"] = FHIST["PRECT_month_1"].groupby("time.year").map(lambda x: x.max(dim="time") - x.min(dim="time"))


Z = (FHIST[f"{Z_VAR}_year_1"].sel(member=slice(1, None)) - FHIST[f"{Z_VAR}_year_1"].sel(member=0)).where(NONGLC_MASK).mean(dim="year")
xb = FHIST[f"{XB_VAR}_year_1"].sel(member=slice(1, None)).where(NONGLC_MASK).mean(dim="year")
yb = FHIST[f"{YB_VAR}_year_1"].sel(member=slice(1, None)).where(NONGLC_MASK).mean(dim="year")

N_MEMBER = len(FHIST["ET_year_1"].member)
qbinned_mean = xclim.get_quantile_binned_mean(Z=Z, xb=xb, yb=yb, xnb=NB, ynb=NB, agg_dims=["gridcell"])
bm_data = qbinned_mean.binned_mean
jh_data = qbinned_mean.joint_hist

fig, axs = plt.subplots(
    ncols=NCOLS, nrows=NROWS,
    sharex=True, sharey=True,
    figsize=FIGSIZE,
    subplot_kw={"box_aspect": 1},
    constrained_layout=True
)
ax = axs.flatten()

pcm = None
for i, m in enumerate(bm_data.member.values):
    pcm = ax[i].pcolormesh(jh_data.sel(member=m), shading="auto", cmap="BrBG", vmin=-VABS, vmax=VABS)

    m_jh_data = jh_data.sel(member=m)
    for ii in range(len(jh_data.x_bin)):
        for jj in range(len(jh_data.y_bin)):
            count = m_jh_data.values[ii, jj]
            if count > 0:
                radius = 0.4 * (count / REF_COUNT)
                lw = 0.5 + (count / REF_COUNT)
                circle = Circle(
                    (ii + 0.5, jj + 0.5),
                    radius,
                    fill=False,
                    edgecolor="black",
                    linewidth=lw,
                    alpha=0.6,
                )
                ax[i].add_patch(circle)

    ax[i].set_title(xclim.ppe.get_member_name(m), fontsize=8)
    ax[i].set_xlim(0, len(bm_data.x_bin))
    ax[i].set_ylim(0, len(bm_data.y_bin))

    ax[i].set_yticks(np.arange(NB) + 0.5)
    ax[i].set_xticks(np.arange(NB) + 0.5)

    ticklabels = np.full((NB), "", dtype=object)
    ticklabels[0] = "1"
    ticklabels[-1] = str(NB)

    ax[i].set_yticklabels(ticklabels)
    ax[i].set_xticklabels(ticklabels)

    if i % NCOLS == 0:
        ax[i].set_ylabel(f"{YB_VAR} Quantiles", fontsize=12)
    if i // NCOLS == NROWS - 1:
        ax[i].set_xlabel(f"{XB_VAR} Quantiles", fontsize=12)

# Add colorbar below all subplots
cbar = fig.colorbar(
    pcm,
    ax=axs,
    orientation="horizontal",
    extend="both",
    fraction=0.025,
    shrink=0.9,
    pad=0.025,
)
cbar.set_label(f"$\\Delta${Z_VAR} [{Z.attrs.get("units")}]", fontsize=12)

for i in range(len(bm_data.member), NCOLS * NROWS):
    ax[i].remove()


fig.canvas.draw()


# Add xlabels to second-to-last row where no subplot exists below
for i in range(N_MEMBER):
    if i // NCOLS == NROWS - 2 and i + NCOLS >= N_MEMBER:
        pos = ax[i].get_position()  # Get position after layout is finalized
        fig.text(
            (pos.x0 + pos.x1) / 2,  # centered horizontally
            pos.y0 - 0.025,  # slightly below the subplot
            f"$\\Delta${Z_VAR} [{Z.attrs.get("units")}]",
            ha="center",
            va="top",
            fontsize=12,
        )

# Add scale circle to the right of colorbar
SCALE_LW = (SCALE_COUNT / REF_COUNT) + 0.5
DATA_RADIUS = 0.4 * (SCALE_COUNT / REF_COUNT)  # radius in data coordinates

# Get colorbar position
cbar_pos = cbar.ax.get_position()

# Position to the right of colorbar in figure coordinates
scale_x = cbar_pos.x1 + 0.05  # 0.05 figure units to the right
scale_y = (cbar_pos.y0 + cbar_pos.y1) / 2  # vertically centered

# Transform from data coordinates to display (pixel) coordinates, then to figure coordinates
point_data = np.array([[0, 0], [DATA_RADIUS, 0]])  # origin and point at radius distance
point_display = ax[0].transData.transform(point_data)
point_figure = fig.transFigure.inverted().transform(point_display)
scale_radius = np.linalg.norm(point_figure[1] - point_figure[0])  # distance in figure units

scale_circle = Circle(
    (scale_x, scale_y),
    scale_radius,
    fill=False,
    edgecolor="black",
    linewidth=SCALE_LW,
    alpha=0.6,
    clip_on=False,
    transform=fig.transFigure,
)
fig.add_artist(scale_circle)

fig.text(
    scale_x + scale_radius + 0.01,
    scale_y,
    f"{SCALE_COUNT} points",
    ha="left",
    va="center",
    fontsize=10,
    transform=fig.transFigure,
)

plt.savefig(f"/glade/work/bbuchovecky/fig/et_controls/d.quantile.lnd.1995-2014.FHIST.{XB_VAR}xbin-{YB_VAR}ybin-histogram.png", dpi=300, bbox_inches="tight")

###

# Dask:
if xclim.is_dask_available():
    xclim.close_dask_cluster(CLIENT_CLUSTER)

