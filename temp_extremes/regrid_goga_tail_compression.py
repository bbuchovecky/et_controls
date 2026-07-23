"""
Compute the weighted average of the per-gridcell native 1-degree statistics
for the 2-degree grid.
"""

from pathlib import Path
import numpy as np
import xesmf as xe
import xarray as xr
import xcdat as xc
import scipy.sparse as sp

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean.cm as cmo


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/proc/dist")
FIG_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/fig/temp_extremes")

PROJECTION = ccrs.PlateCarree(central_longitude=12)


def main():
    # ---------------------------------------------------------------------------
    # Regrid the statistics using a weighted average
    # ---------------------------------------------------------------------------
    rootdir = Path("/glade/campaign/collections/rda/data/d651010/global/CESM2.1_GOGA_ERSSTv5/atm/proc/tseries/month_1/TREFHT")
    ds_1deg = xc.open_dataset(rootdir / "f.e21.FHIST_BGC.f09_f09.historical.ersstv5.goga.ens01.cam.h0.TREFHT.188001-201412.nc")

    rootdir = Path("/glade/campaign/univ/uwas0155/ppe/historical/coupled_simulations/f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE.000/atm/proc/tseries/month_1")
    ds_2deg = xc.open_dataset(rootdir / "f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE.000.cam.h0.TREFHT.195001-201412.nc")

    src_lat = ds_1deg["lat"]
    src_lon = ds_1deg["lon"]

    tgt_lat = ds_2deg["lat"]
    tgt_lon = ds_2deg["lon"] 

    source_grid_shape = (ds_1deg.sizes["lat"], ds_1deg.sizes["lon"])
    target_grid_shape = (ds_2deg.sizes["lat"], ds_2deg.sizes["lon"])

    n_src = ds_1deg.sizes["lat"] * ds_1deg.sizes["lon"]
    n_tgt = ds_2deg.sizes["lat"] * ds_2deg.sizes["lon"]

    print(f"source: n = {n_src}, shape = {source_grid_shape}, lat = [{src_lat.min().data}, {src_lat.max().data}], lon = [{src_lon.min().data}, {src_lon.max().data}]")
    print(f"target: n = {n_tgt}, shape = {target_grid_shape}, lat = [{tgt_lat.min().data}, {tgt_lat.max().data}], lon = [{tgt_lon.min().data}, {tgt_lon.max().data}]")


    regridder = xe.Regridder(
        ds_1deg, ds_2deg,
        method="conservative",
        periodic=True,
        filename="weights_1deg_to_2deg_conservative.nc",
        reuse_weights=False,
    )


    w = xr.open_dataset("weights_1deg_to_2deg_conservative.nc")
    row = w["row"].values - 1   # xESMF weight files are 1-indexed
    col = w["col"].values - 1
    S   = w["S"].values

    # Sanity check against index maxima
    print(row.max(), col.max())
    assert row.max() < n_tgt, f"row.max()={row.max()} exceeds n_tgt={n_tgt} — grid mismatch"
    assert col.max() < n_src, f"col.max()={col.max()} exceeds n_src={n_src} — grid mismatch"

    # W[j, i] = weight a_i of source cell i contributing to target cell j
    W = sp.csr_matrix((S, (row, col)), shape=(n_tgt, n_src))

    # Debug poles
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums_2d = row_sums.reshape((96, 144))
    print("South pole row sum stats:", row_sums_2d[0, :].min(), row_sums_2d[0, :].max())
    print("North pole row sum stats:", row_sums_2d[-1, :].min(), row_sums_2d[-1, :].max())
    print("n affected at lat row 0:", np.isclose(row_sums_2d[0, :], row_sums_2d[0, :].min(), atol=1e-4).sum())
    print("n affected at lat row -1:", np.isclose(row_sums_2d[-1, :], row_sums_2d[-1, :].min(), atol=1e-4).sum())

    plt.plot(np.arange(len(row_sums)), np.sort(row_sums))
    plt.savefig(FIG_PATH / "regrid_goga_row_sums_check.png", dpi=120)
    plt.close()
    

    # ---------------------------------------------------------------------------
    # Compute the weighted averages
    # TODO: iterate through members, compute weighted-average regrid, then build DataArray
    # TODO: compute the tail change between
    #       (a) native output -> pooled daily stats -> weighted-average regrid
    #       (b) native output -> conservative regrid -> pooled daily stats
    # ---------------------------------------------------------------------------
    rootdir = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/proc/dist/TREFHT_pool_90d_window")
    anom_stats_1deg = xr.open_dataset(rootdir / "GOGA2_TREFHT_pool_90d_window_19850101-20141231_anom_stats.nc")
    anom_stats_2deg = xr.open_dataset(rootdir / "GOGA2_2DEG_TREFHT_pool_90d_window_19850101-20141231_anom_stats.nc")

    qdelta_1deg = anom_stats_1deg.quantiles.isel(member=1) - anom_stats_1deg.quantiles.isel(member=0)
    qdelta_left_shift_1deg = qdelta_1deg.sel(quantile=0.50) - qdelta_1deg.sel(quantile=0.05)
    qdelta_right_shift_1deg = qdelta_1deg.sel(quantile=0.95) - qdelta_1deg.sel(quantile=0.50)

    qdelta_2deg = anom_stats_2deg.quantiles.isel(member=1) - anom_stats_2deg.quantiles.isel(member=0)
    qdelta_left_shift_2deg = qdelta_2deg.sel(quantile=0.50) - qdelta_2deg.sel(quantile=0.05)
    qdelta_right_shift_2deg = qdelta_2deg.sel(quantile=0.95) - qdelta_2deg.sel(quantile=0.50)

    qdelta_right_shift_1deg_flat = qdelta_right_shift_1deg.values.flatten()
    qdelta_right_shift_wavg_2deg_flat = W @ qdelta_right_shift_1deg_flat
    qdelta_right_shift_wavg_2deg = qdelta_right_shift_wavg_2deg_flat.reshape(target_grid_shape)

    qdelta_right_shift_wavg_2deg = xr.DataArray(
        data=qdelta_right_shift_wavg_2deg,
        coords=dict(
            lat=ds_2deg.lat,
            lon=ds_2deg.lon
        )
    )


if __name__ == "__main__":
    main()
