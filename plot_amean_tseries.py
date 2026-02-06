"""Batch plotting of timeseries."""

import numpy as np
import matplotlib.pyplot as plt
import utils_moved_to_xclimate
from constants import COLORMAP, LS

np.random.seed(42)

USE_DASK = True
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

if USE_DASK:
    client, cluster = utils_moved_to_xclimate.create_dask_cluster(
        account="UWAS0155",
        nworkers=5,
    )

ga = utils_moved_to_xclimate.load_var("FSH", "month_1", "lnd", extract="area").isel(member=0).fillna(0)
lf = utils_moved_to_xclimate.load_var("FSH", "month_1", "lnd", extract="landfrac").isel(member=0).fillna(0)
la = lf * ga

for v in VARIABLES:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    da = utils_moved_to_xclimate.load_var(v, "month_1", "lnd", stream="h0")
    for m in da.member:
        info = utils_moved_to_xclimate.get_member_info(m.item())
        p = info[1]
        mm = info[2]
        ts = (
            da.sel(member=m)
            .weighted(la)
            .mean(dim=["lat", "lon"])
            .groupby("time.year")
            .mean()
        )
        ts.plot(ax=ax, color=COLORMAP[p], alpha=0.8, lw=1, ls=LS[mm], _labels=False)
        ax.text(2015+np.random.uniform(-1, 1), ts.sel(year=2014), m.item(), fontsize=4, color=COLORMAP[p], ha="left", va="center")

    da.sel(member=0).weighted(la).mean(dim=["lat", "lon"]).groupby(
        "time.year"
    ).mean().plot(ax=ax, color="k", alpha=1, lw=2, _labels=False)
    ax.set_title(f"Global Land Annual Mean {v} [{da.units}]")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{v} [{da.units}]")
    fig.tight_layout()
    fig.savefig(f"./fig/g_lnd_yr_ts_{v}.png", dpi=300)

if USE_DASK:
    utils_moved_to_xclimate.close_dask_cluster(client, cluster)
