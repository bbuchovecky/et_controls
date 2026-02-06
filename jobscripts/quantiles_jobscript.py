#!/glade/work/bbuchovecky/miniforge3/envs/data-sci/bin/python3.14
#PBS -N quantiles
#PBS -A UWAS0155
#PBS -l select=1:ncpus=4:mem=32GB
#PBS -l walltime=01:00:00
#PBS -q develop
#PBS -j oe
#PBS -o logs/

import os
import sys
from pathlib import Path
from dask.distributed import Client, LocalCluster, wait

import numpy as np
import xarray as xr
import xclimate as xclim


def main():

    ncpus = 4  # must match allocated ncpus
    nmem = 8  # must match allocated mem / ncpus (memory PER cpu)

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
        processes=True,                # use processes, not threads (this is nuanced...)
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


    nonglc_pct_threshold = 80
    grid = xclim.load_fhist_ppe_grid()
    nonglc_mask = grid.PCT_GLC < nonglc_pct_threshold

    time_slice = slice("1995-01", "2014-12")
    year_start = time_slice.start[:4]
    year_end = time_slice.stop[:4]

    variables = [
        # "PRECC_month_1",
        # "PRECL_month_1",
        "TSA_month_1",
        "TLAI_month_1",
        "EFLX_LH_TOT_month_1",
        # "PRECT_day_1",
        # "TSA_day_1",
    ]
    time_slice = slice("1995-01", "2014-12")
    n_qbin = np.array([15, 25, 50, 75, 100])

    rootdir = Path("/glade/work/bbuchovecky/et_controls/proc/quantiles")
    time_tag = "agg"


    print("Loading variables:")
    fhist = {}
    for v in variables:
        print(f"  {v}")
        name = "_".join(v.split("_")[:-2])
        fhist[v] = xclim.load_fhist(v, keep_var_only=True)[name].sel(time=time_slice).reindex_like(grid, method="nearest", tolerance=1e-3)

    # # Monthly PRECT (PRECC + PRECL)
    # print("  PRECT_month_1")
    # fhist["PRECT_month_1"] = fhist["PRECC_month_1"] + fhist["PRECL_month_1"]
    # fhist["PRECT_month_1"].attrs["long_name"] = "total monthly precipitation rate (PRECC + PRECL)"
    # fhist.pop("PRECC_month_1")
    # fhist.pop("PRECL_month_1")

    # Annual mean TSA
    fhist["TSA_year_1"] = fhist["TSA_month_1"].groupby("time.year").map(lambda x: x.weighted(x.time.dt.days_in_month).mean("time")) - 273.15
    fhist["TSA_year_1"].attrs["units"] = "degrees C"
    fhist.pop("TSA_month_1")

    # Annual mean TLAI
    print("  TLAI_year_1")
    fhist["TLAI_year_1"] = fhist["TLAI_month_1"].groupby("time.year").map(lambda x: x.weighted(x.time.dt.days_in_month).mean("time"))
    fhist.pop("TLAI_month_1")

    # Annual mean EFLX_LH_TOT
    print("  EFLX_LH_TOT_year_1")
    fhist["EFLX_LH_TOT_year_1"] = fhist["EFLX_LH_TOT_month_1"].groupby("time.year").map(lambda x: x.weighted(x.time.dt.days_in_month).mean("time"))
    fhist.pop("EFLX_LH_TOT_month_1")

    # # Annual precipitation difference between wettest and driest months
    # fhist["PRDIFF_year_1"] = fhist["PRECT_month_1"].groupby("time.year").map(lambda x: x.max(dim="time") - x.min(dim="time"))
    # fhist["PRDIFF_year_1"].attrs["long_name"] = "annual precipitation rate (PRECT) difference between wettest and driest months"

    # # Total annual precipitation (PRECC + PRECL) in mm: m/s * 1000mm/m * 86400s/day * days/month
    # print("  TOTANNPRECT_year_1")
    # fhist["TOTANNPRECT_year_1"] = fhist["PRECT_month_1"].groupby("time.year").map(lambda x: (x * 1000 * 86400 * x.time.dt.days_in_month).sum(dim="time"))
    # fhist["TOTANNPRECT_year_1"].attrs["long_name"] = "total annual precipitation (PRECC + PRECL)"
    # fhist["TOTANNPRECT_year_1"].attrs["units"] = "mm"

    # # Wet-day frequency as defined in Feldman et al. (2024) - the annual number of days with above 1 mm/day of precipitation
    # DAILY_PRECT_THRESH =  1 / (1000 * 24 * 60 * 60)  # [m/s] = 1 [mm/day]
    # fhist["WDFRQ_year_1"] = (fhist["PRECT_day_1"].where(nonglc_mask) > DAILY_PRECT_THRESH).groupby("time.year").sum()
    # fhist["WDFRQ_year_1"].attrs["long_name"] = "number of days with total precipitation rate (PRECT) > 1 mm/day"
    # fhist["WDFRQ_year_1"].attrs["notes"] = "defined in Feldman et al. Nature (2024)"
    # fhist["WDFRQ_year_1"].attrs["units"] = "days"


    print(list(fhist.keys()), flush=True)


    print("Computing quantiles:", flush=True)
    for v, da in fhist.items():
        print(f"\n  {v}:", end="", flush=True)

        if "year" in v:
            tdim = "year"
            chunks = {"member": 1, "lat": -1, "lon": -1, tdim: -1}
        elif "day" in v:
            tdim = "time"
            chunks = {"member": 1, "lat": -1, "lon": -1, tdim: -1}
        else: # month
            tdim = "time"
            chunks = {"member": 1, "lat": -1, "lon": -1, tdim: 365}
        
        # Rechunk to optimize for stacking and quantile computation
        # Chunk along member dimension only, consolidate spatial/temporal dims
        print(" rechunking...", end="", flush=True)
        da = da.chunk(chunks)
        
        # Persist this variable in distributed memory
        print(" persisting...", end="", flush=True)
        da_persisted = da.persist()
        wait([da_persisted])  # Wait for persist to complete
        
        print(" computing:", end="", flush=True)
        qs_list = []
        for nb in n_qbin:
            print(f" {nb}", end="", flush=True)
            x = da_persisted.where(nonglc_mask)
            x_s = x.stack(gridcell=["lat", "lon", tdim])
            qs_list.append(xclim.get_quantiles(x_s, nb, ["gridcell"]).rename(f"qbin_{nb}", quantile=f"qb{nb}").assign_attrs(description=f"{nb} quantiles of {v}"))
        
        print(" saving...", end="", flush=True)
        fname = f"quantiles.{year_start}-{year_end}.TIME{time_tag}.{v}.nc"
        xr.merge(qs_list).to_netcdf(rootdir / fname)
        print(f" done: {fname}", flush=True)
        
        # Free distributed memory
        client.cancel(da_persisted)
        del da_persisted, qs_list


    #########################
    #### END COMPUTATION ####
    #########################

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
