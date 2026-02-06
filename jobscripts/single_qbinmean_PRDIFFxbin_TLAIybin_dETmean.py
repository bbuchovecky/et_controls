#!/glade/work/bbuchovecky/miniforge3/envs/data-sci/bin/python3.14

import os
import sys
import time
import argparse
from pathlib import Path
from dask.distributed import Client, LocalCluster
import xclimate as xclim


def main(
    member: int,
    ncpus: int,
    nmem: float,
    nbin: int,
    timeagg: bool,
    ):

    memory_limit = f"{nmem}GB"
    tmpdir = os.environ.get("TMPDIR", "/tmp")

    print("Python version:", sys.version)
    print(f"ncpus: {ncpus}")
    print(f"memory_per_worker: {memory_limit}")

    # Avoid oversubscription if libraries spawn processes internally
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Start local cluster
    cluster = LocalCluster(
        n_workers=ncpus,
        threads_per_worker=1,
        processes=True,                # use processes, not threads
        memory_limit=memory_limit,     # per-worker memory limit
        local_directory=tmpdir,        # spill + temp files
        dashboard_address=None,        # no dashboard in batch
    )
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)
    print(f"Workers: {ncpus}, Memory per worker: {memory_limit}")


    ###########################
    #### START COMPUTATION ####
    ###########################


    nonglc_pct_threshold = 80  # maximum percent of glaciated land allowable for a NON-glaciated gridcell
    time_slice = slice("1995-01", "2014-12")
    z_var = "ET"
    xb_var = "PRDIFF"
    yb_var = "TLAI"

    if timeagg:
        quantile_dims = ["lat", "lon", "year"]  # aggregate and compute quantiles over (lat, lon, year) dimensions
        time_tag = "agg"
    else:
        quantile_dims = ["lat", "lon"]
        time_tag = "mean"


    # Load grid areas
    grid = xclim.load_fhist_ppe_grid()
    nonglc_mask = grid.PCT_GLC < nonglc_pct_threshold


    # Load variables
    fhist = {}
    fhist["PRECC_month_1"] = xclim.load_coupled_fhist_ppe("PRECC", "atm", "month_1", "h0", keep_var_only=True)["PRECC"].sel(time=time_slice)
    fhist["PRECL_month_1"] = xclim.load_coupled_fhist_ppe("PRECL", "atm", "month_1", "h0", keep_var_only=True)["PRECL"].sel(time=time_slice)
    fhist["TLAI_month_1"] = xclim.load_coupled_fhist_ppe("TLAI", "lnd", "month_1", "h0", keep_var_only=True)["TLAI"].sel(time=time_slice).reindex_like(grid, method="nearest", tolerance=1e-3)
    fhist["ET_month_1"] = xclim.load_coupled_fhist_ppe("EFLX_LH_TOT", "lnd", "month_1", "h0", keep_var_only=True)["EFLX_LH_TOT"].sel(time=time_slice).reindex_like(grid, method="nearest", tolerance=1e-3)


    # Annual mean TLAI
    fhist["TLAI_year_1"] = fhist["TLAI_month_1"].groupby("time.year").map(lambda x: x.weighted(x.time.dt.days_in_month).mean("time"))

    # Annual mean ET
    fhist["ET_year_1"] = fhist["ET_month_1"].groupby("time.year").map(lambda x: x.weighted(x.time.dt.days_in_month).mean("time"))

    # Monthly precipitation rate (PRECC + PRECL)
    fhist["PRECT_month_1"] = fhist["PRECC_month_1"] + fhist["PRECL_month_1"]

    # Annual precipitation difference between wettest and driest months
    fhist["PRDIFF_year_1"] = fhist["PRECT_month_1"].groupby("time.year").map(lambda x: x.max(dim="time") - x.min(dim="time"))
    fhist["PRDIFF_year_1"].attrs["long_name"] = "annual precipitation difference between wettest and driest months"

    # Wet-day frequency as defined in Feldman et al. (2024) - the annual number of days with above 1 mm/day of precipitation
    # DAILY_PRECT_THRESH =  1 / (1000 * 24 * 60 * 60)  # [m/s] = 1 [mm/day]
    # fhist["WDFRQ_year_1"] = (fhist["PRECT_day_1"].where(LA_fhist>0) > DAILY_PRECT_THRESH).groupby("time.year").sum()


    # Output directory
    case = f"TIME{time_tag}_{xb_var}xbin_{yb_var}ybin_d{z_var}mean"
    rootdir = Path("/glade/work/bbuchovecky/et_controls/proc") / case
    os.makedirs(rootdir, exist_ok=True)

    # Metadata
    str_mem = str(fhist[list(fhist.keys())[0]].isel(member=member).member.item())
    year_start = time_slice.start[:4]
    year_end = time_slice.stop[:4]
    
    print("\n=== Starting quantile binning ===")
    print(f"Z: {z_var}, XB: {xb_var}, YB: {yb_var}")
    print(f"Data shape: {fhist[f'{z_var}_year_1'].shape}")
    print(f"Quantile dims: {quantile_dims}")
    print(f"Bins: {nbin}x{nbin}")
    print(f"Time period: {year_start}-{year_end}, {time_tag.upper()}")


    # Compute for the single member and store in distributed memory (persist)
    t0 = time.time()
    if timeagg:
        fhist[f"{z_var}_year_1"] = fhist[f"{z_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).persist()
        fhist[f"{xb_var}_year_1"] = fhist[f"{xb_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).persist()
        fhist[f"{yb_var}_year_1"] = fhist[f"{yb_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).persist()
        rechunk = {'year': -1, 'lat': -1, 'lon': -1}
    else:
        fhist[f"{z_var}_year_1"] = fhist[f"{z_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).mean(dim="year").persist()
        fhist[f"{xb_var}_year_1"] = fhist[f"{xb_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).mean(dim="year").persist()
        fhist[f"{yb_var}_year_1"] = fhist[f"{yb_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).mean(dim="year").persist()
        rechunk = {'lat': -1, 'lon': -1}
    print(f"Persist: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    z = fhist[f"{z_var}_year_1"].isel(member=1) - fhist[f"{z_var}_year_1"].isel(member=0)
    xb = fhist[f"{xb_var}_year_1"].isel(member=1)
    yb = fhist[f"{yb_var}_year_1"].isel(member=1)
    print(f"Data prep: {time.time()-t0:.1f}s")
    print(f"Data shapes: Z - {z.shape}, XB - {xb.shape}, YB - {yb.shape}")

    t0 = time.time()
    z = z.chunk(rechunk)
    xb = xb.chunk(rechunk)
    yb = yb.chunk(rechunk)
    print(f"Rechunking: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    result = xclim.get_quantile_binned_mean(
        Z=z, xb=xb, yb=yb, xnb=nbin, ynb=nbin, quantile_dims=quantile_dims, agg_dims=["gridcell"])
    print(f"Quantile binning: {time.time()-t0:.1f}s")    

    t0 = time.time()
    result.to_netcdf(rootdir / f"qbin{nbin}.{year_start}-{year_end}.{case}.{str_mem.zfill(3)}.nc")
    print(rootdir / f"qbin{nbin}.{year_start}-{year_end}.{case}.{str_mem.zfill(3)}.nc")
    print(f"Save results to NetCDF: {time.time()-t0:.1f}s")


    #########################
    #### END COMPUTATION ####
    #########################


    client.close()
    cluster.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--member", type=int, required=True, help="member index")
    parser.add_argument("--ncpus", type=int, required=True, help="number of cpus")
    parser.add_argument("--nmem", type=float, required=True, help="memory in GB per cpu")
    parser.add_argument("--nbin", type=int, required=True, help="number of quantile bins (x and y)")
    parser.add_argument("--timeagg", type=int, required=True, choices=[0, 1], help="1 for time aggregated quantiles, 0 for time mean")
    args = parser.parse_args()

    main(member=args.member, ncpus=args.ncpus, nmem=args.nmem, nbin=args.nbin, timeagg=bool(args.timeagg))
