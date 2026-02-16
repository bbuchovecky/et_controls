#!/glade/work/bbuchovecky/miniforge3/envs/data-sci-py312/bin/python3.12

import os
import sys
import time
import argparse
from pathlib import Path
from dask.distributed import Client, LocalCluster
import xarray as xr
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

    snow_pct_threshold = 80 / 100  # maximum allowable percent of snow cover on all months of the average year
    nonglc_pct_threshold = 80      # maximum allowable percent of glaciated land for a NON-glaciated gridcell
    time_slice = slice("1995-01", "2014-12")

    z_var = "ET"
    x_var = "EVAPIDX"  # ET / P
    y_var = "AI"  # Rn / (Lv * P)

    variable_names = [
        "FSNO",
        "EFLX_LH_TOT", "TLAI",
        "FLDS", "FIRE", "FIRA",
        "FSDS", "FSR", "FSA",
        "PRECC", "PRECL",
    ]

    if timeagg:
        quantile_dims = ["lat", "lon", "year"]  # aggregate and compute quantiles over (lat, lon, year) dimensions
        time_tag = "agg"
    else:
        quantile_dims = ["lat", "lon"]
        time_tag = "mean"


    # Load grid areas
    grid = xclim.load_fhist_ppe_grid()


    # Load variables
    fhist = {}
    variables = []
    for v in variable_names:
        print(v)
        key = f"{v}_month_1"
        fhist[key] = xclim.load_fhist(key, keep_var_only=True)[v].sel(time=time_slice).reindex_like(grid, method="nearest", tolerance=1e-3)
        variables.append(key)
    
    # Rename ET
    fhist["ET_month_1"] = fhist["EFLX_LH_TOT_month_1"]
    fhist.pop("EFLX_LH_TOT_month_1")

    # Monthly net radiation at the surface, + down
    #   Rn = (net SW) + (net LW) = (down SW - up SW) - (down LW - up LW)
    #   Rn = FSDS - FSR + FLDS - FIRE
    fhist["RN_month_1"] = fhist["FSDS_month_1"] - fhist["FSR_month_1"] + fhist["FLDS_month_1"] - fhist["FIRE_month_1"]
    fhist["RN_month_1"] = fhist["RN_month_1"].rename("RN")
    fhist["RN_month_1"].attrs = {
        "long_name": "net radiation at surface [+ down]",
        "description": "FSDS - FSR + FLDS - FIRE",
        "units": "W/m2",
    }

    # Climatological net radiation at the surface
    fhist["RN_clim"] = fhist["RN_month_1"].weighted(fhist["RN_month_1"].time.dt.days_in_month).mean("time")
    fhist["RN_clim"].attrs["time_mean"] = f"{time_slice.start} to {time_slice.stop}"

    # Monthly precipitation rate (PRECC + PRECL)
    fhist["PRECT_month_1"] = (fhist["PRECC_month_1"] + fhist["PRECL_month_1"]) * 1000 * 2.5e6
    fhist["PRECT_month_1"] = xr.where(fhist["PRECT_month_1"] < 1e-3, 1e-3, fhist["PRECT_month_1"])
    fhist["PRECT_month_1"].attrs["units"] = "W/m2"

    # Climatological precipitation rate
    fhist["PRECT_clim"] = fhist["PRECT_month_1"].weighted(fhist["PRECT_month_1"].time.dt.days_in_month).mean("time")
    fhist["PRECT_clim"].attrs["time_mean"] = f"{time_slice.start} to {time_slice.stop}"

    # Climatological Aridity Index (AI = RN / PRECT)
    fhist["AI_clim"] = fhist["RN_clim"] / fhist["PRECT_clim"]
    fhist["AI_clim"] = fhist["AI_clim"].rename("AI")
    fhist["AI_clim"].attrs = {
        "long_name": "aridity index",
        "description": "Rn / (Lv * P)",
        "time_mean": f"{time_slice.start} to {time_slice.stop}"
    }

    # Monthly evaporative index (ET / P)
    fhist["EI_month_1"] = fhist["ET_month_1"] / fhist["PRECT_month_1"]
    fhist["EI_month_1"].attrs["long_name"] = "evaporative index (ET / P)"
    fhist["EI_month_1"].attrs["units"] = "1"


    # Create masks
    fhist["FSNO_clim"] = fhist["FSNO_month_1"].weighted(fhist["FSNO_month_1"].time.dt.days_in_month).mean("time")
    snow_mask = xr.where(fhist["FSNO_clim"] > snow_pct_threshold, 0, 1)
    nonglc_mask = grid.PCT_GLC < nonglc_pct_threshold
    

    # Output directory
    case = f"TIME{time_tag}.{x_var}xbin_{y_var}ybin_d{z_var}mean"
    rootdir = Path("/glade/work/bbuchovecky/et_controls/proc") / case
    os.makedirs(rootdir, exist_ok=True)

    # Metadata
    str_mem = str(fhist[list(fhist.keys())[0]].isel(member=member).member.item())
    year_start = time_slice.start[:4]
    year_end = time_slice.stop[:4]
    
    print("\n=== Starting quantile binning ===")
    print(f"Z: {z_var}, XB: {x_var}, YB: {y_var}")
    print(f"Data shape: {fhist[f'{z_var}_year_1'].shape}")
    print(f"Quantile dims: {quantile_dims}")
    print(f"Bins: {nbin}x{nbin}")
    print(f"Time period: {year_start}-{year_end}, {time_tag.upper()}")


    # Compute for the single member and store in distributed memory (persist)
    t0 = time.time()
    if timeagg:
        fhist[f"{z_var}_year_1"] = fhist[f"{z_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).persist()
        fhist[f"{x_var}_year_1"] = fhist[f"{x_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).persist()
        fhist[f"{y_var}_year_1"] = fhist[f"{y_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).persist()
        rechunk = {'year': -1, 'lat': -1, 'lon': -1}
    else:
        fhist[f"{z_var}_year_1"] = fhist[f"{z_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).mean(dim="year").persist()
        fhist[f"{x_var}_year_1"] = fhist[f"{x_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).mean(dim="year").persist()
        fhist[f"{y_var}_year_1"] = fhist[f"{y_var}_year_1"].isel(member=[0, member]).where(nonglc_mask).mean(dim="year").persist()
        rechunk = {'lat': -1, 'lon': -1}
    print(f"Persist: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    z = fhist[f"{z_var}_year_1"].isel(member=1) - fhist[f"{z_var}_year_1"].isel(member=0)
    xb = fhist[f"{x_var}_year_1"].isel(member=1)
    yb = fhist[f"{y_var}_year_1"].isel(member=1)
    print(f"Data prep: {time.time()-t0:.1f}s")
    print(f"Data shapes: Z - {z.shape}, XB - {xb.shape}, YB - {yb.shape}")

    t0 = time.time()
    z = z.chunk(rechunk)
    xb = xb.chunk(rechunk)
    yb = yb.chunk(rechunk)
    print(f"Rechunking: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    # TODO: (1) compute quantiles, (2) compute bins, (3) compute bin stats
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
