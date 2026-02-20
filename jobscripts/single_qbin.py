#!/glade/work/bbuchovecky/miniforge3/envs/data-sci-py312/bin/python3.12

import os
import sys
import time
import argparse
from pathlib import Path
from dask.distributed import Client, LocalCluster, wait
import numpy as np
import xarray as xr
import xclimate as xclim


def main(
    member: int,
    ncpus: int,
    nmem: float,
    nbin: int,
):
    
    ############################
    #### SETUP DASK CLUSTER ####
    ############################

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

    # Number of months for the growing season
    growsn_nmon = 3

    # Time period to select
    time_slice = slice("1995-01", "2014-12")
    # time_slice = slice("1950-01", "2014-12")
    year_start = time_slice.start[:4]
    year_end = time_slice.stop[:4]
    
    # Root directory to save output
    rootdir = Path("/glade/work/bbuchovecky/et_controls/proc/qbin")

    # Define mask thresholds
    snow_pct_threshold = 80    # maximum allowable percent of snow cover on all months of the average year
    nonglc_pct_threshold = 80  # maximum allowable percent of glaciated land for a NON-glaciated gridcell

    # Load grid data
    grid = xclim.load_fhist_ppe_grid()

    # Load snow cover
    fsno = xclim.load_fhist("FSNO_month_1", keep_var_only=True)["FSNO"].sel(time=time_slice).reindex_like(grid, method="nearest", tolerance=1e-3)
    fsno_clim_min = fsno.groupby("time.month").mean().min(dim="month")

    # Formatted member string
    str_mem = str(fsno.isel(member=member).member.item())

    # Create masks
    snow_mask = fsno_clim_min.isel(member=member) <= (snow_pct_threshold / 100)
    nonglc_mask = grid.PCT_GLC <= nonglc_pct_threshold
    full_mask = snow_mask & nonglc_mask

    variables = [
        # "PRECC_month_1", "PRECL_month_1",
        # "TSA_month_1",
        # "TLAI_month_1",
        # "EFLX_LH_TOT_month_1",
        "FCTR_month_1",
        "FGEV_month_1",
        "FCEV_month_1",
        # "SOILWATER_10CM_month_1",
        # "TOTSOILLIQ_month_1",
        # "FSDS_month_1", "FSR_month_1", "FLDS_month_1", "FIRE_month_1",  # for net radiation
        # "PRECT_day_1", "TSA_day_1", "SOILWATER_10CM_day_1",  # daily variables

    ]

    print("Loading variables:")
    fhist = {}
    for v in variables:
        print(f"  {v}", flush=True)
        name = "_".join(v.split("_")[:-2])
        fhist[v] = xclim.load_fhist(v, keep_var_only=True)[name].sel(time=time_slice).reindex_like(grid, method="nearest", tolerance=1e-3)
        # fhist[v] = fhist[v].where(full_mask).isel(member=member)
        fhist[v] = fhist[v].where(full_mask).isel(member=[0, member])
        fhist[v].attrs["masks"] = f"gridcell percent glaciated land <= {nonglc_pct_threshold}\ngridcell percent snow cover on all months of average year <= {snow_pct_threshold}"


    # # Monthly net radiation at the surface, + down
    # #   Rn = (net SW) + (net LW) = (down SW - up SW) - (down LW - up LW)
    # #   Rn = FSDS - FSR + FLDS - FIRE
    # print("  RN_month_1")
    # fhist["RN_month_1"] = fhist["FSDS_month_1"] - fhist["FSR_month_1"] + fhist["FLDS_month_1"] - fhist["FIRE_month_1"]
    # fhist["RN_month_1"] = fhist["RN_month_1"].rename("RN")
    # fhist["RN_month_1"].attrs = {
    #     "long_name": "net radiation at surface [+ down]",
    #     "description": "FSDS - FSR + FLDS - FIRE",
    #     "units": "W/m2",
    # }

    # # Monthly PRECT (PRECC + PRECL)
    # print("  PRECT_month_1")
    # fhist["PRECT_month_1"] = fhist["PRECC_month_1"] + fhist["PRECL_month_1"]
    # fhist["PRECT_month_1"].attrs["long_name"] = "total monthly precipitation rate (PRECC + PRECL)"
    # fhist.pop("PRECC_month_1")
    # fhist.pop("PRECL_month_1")

    # # Monthly TSA
    # print("  TSA_month_1")
    # fhist["TSA_month_1"] = fhist["TSA_month_1"] - 273.15
    # fhist["TSA_month_1"].attrs["units"] = "degrees C"

    # # Monthly precipitation rate in energy units
    # print("  EPRECT_month_1")
    # fhist["EPRECT_month_1"] = fhist["PRECT_month_1"] * 1000 * 2.5e6
    # fhist["EPRECT_month_1"] = xr.where(fhist["EPRECT_month_1"] < 1e-3, 1e-3, fhist["EPRECT_month_1"])
    # fhist["EPRECT_month_1"].attrs = {
    #     "long_name": f"{fhist['PRECT_month_1'].attrs['long_name']} in energy units",
    #     "units": "W/m2"
    # }



    
    # print("\nComputing annual mean:")
    # for v in list(fhist.keys()):
    #     if "month" in v:
    #         name = "_".join(v.split("_")[:-2])
    #         print(f"{name}_year_1")
    #         fhist[f"{name}_year_1"] = fhist[f"{name}_month_1"].groupby("time.year").map(lambda x: x.weighted(x.time.dt.days_in_month).mean("time")).chunk({"year": -1})


    print("\nComputing climatological mean:")
    variables_clim = [
        "FCTR",
        "FGEV",
        "FCEV",
    ]
    for name in variables_clim:
        print(f"{name}_clim_1")
        if f"{name}_month_1" in fhist.keys():
            fhist[f"{name}_clim_1"] = fhist[f"{name}_month_1"].weighted(fhist[f"{name}_month_1"].time.dt.days_in_month).mean("time")
            fhist[f"{name}_clim_1"].attrs["time_mean"] = f"{time_slice.start} to {time_slice.stop}"
        elif f"{name}_year_1" in fhist.keys():
            fhist[f"{name}_clim_1"] = fhist[f"{name}_year_1"].mean("year")
            fhist[f"{name}_clim_1"].attrs["time_mean"] = f"{time_slice.start} to {time_slice.stop}"
        else:
            print(f"  {name} not loaded")


    # print("\nComputing growing season mean:")
    # variables_growsn = []
    # for name in variables_growsn:
    #     if f"{name}_month_1" in fhist.keys():
    #         fhist[f"GROWSN_{name}_year_1"] = xclim.science.growing_season_mean(fhist[f"{name}_month_1"], fhist["TLAI_month_1"], nmon=growsn_nmon)
    #     else:
    #         print(f"  {name} not loaded")
    

    # def detrend(da, dim):
    #     regress = xclim.regression.ols_field(da[dim],  da, sample_dim=dim)
    #     trend = regress.intercept + da[dim] * regress.slope
    #     return da - trend

    # # Detrended annual mean SOILWATER_10CM
    # fhist["SOILWATER_10CM_DT_year_1"] = detrend(fhist["SOILWATER_10CM_year_1"], "year").persist()

    # # Detrended annual mean TOTSOILLIQ
    # fhist["TOTSOILLIQ_DT_year_1"] = detrend(fhist["TOTSOILLIQ_year_1"], "year").persist()

    # # Detrended annual mean FCTR
    # fhist["FCTR_DT_year_1"] = detrend(fhist["FCTR_year_1"], "year").persist()

    # # cor(SOILWATER_10CM_DT_year_1, FCTR_DT_year_1)
    # fhist["cor_SMSFC_FCTR_clim_1"] = xclim.regression.ols_field(fhist["SOILWATER_10CM_DT_year_1"], fhist["FCTR_DT_year_1"], "year").pearsonr

    # # cor(TOTSOILLIQ_DT_year_1, FCTR_DT_year_1)
    # fhist["cor_SMTOT_FCTR_clim_1"] = xclim.regression.ols_field(fhist["TOTSOILLIQ_DT_year_1"], fhist["FCTR_DT_year_1"], "year").pearsonr


    # # Annual precipitation difference between wettest and driest months
    # fhist["PRDIFF_year_1"] = fhist["PRECT_month_1"].groupby("time.year").map(lambda x: x.max(dim="time") - x.min(dim="time"))
    # fhist["PRDIFF_year_1"].attrs["long_name"] = "annual precipitation rate (PRECT) difference between wettest and driest months"

    # # Total annual precipitation (PRECC + PRECL) in mm: m/s * 1000mm/m * 86400s/day * days/month
    # print("  TOTANNPRECT_year_1")
    # fhist["TOTANNPRECT_year_1"] = fhist["PRECT_month_1"].groupby("time.year").map(lambda x: (x * 1000 * 86400 * x.time.dt.days_in_month).sum(dim="time", min_count=1))
    # fhist["TOTANNPRECT_year_1"].attrs = {
    #     "long_name": "total annual precipitation (PRECC + PRECL)",
    #     "units": "mm",
    # }

    # # Wet-day frequency as defined in Feldman et al. (2024) - the annual number of days with above 1 mm/day of precipitation
    # print("  WDFRQ_year_1")
    # daily_prect_threshold =  1 / (1000 * 24 * 60 * 60)  # [m/s] = 1 [mm/day]
    # fhist["WDFRQ_year_1"] = (fhist["PRECT_day_1"].where(nonglc_mask) > daily_prect_threshold).groupby("time.year").sum()
    # fhist["WDFRQ_year_1"].attrs = {
    #     "long_name": "number of days with total precipitation rate (PRECT) > 1 mm/day",
    #     "description": "defined in Feldman et al. Nature (2024)",
    #     "units": "days",
    # }

    # # Growing season evaporative index (ET/P)
    # print("  GROWSN_EI_year_1")
    # fhist["GROWSN_EI_year_1"] = fhist["GROWSN_EFLX_LH_TOT_year_1"] / fhist["GROWSN_EPRECT_year_1"]
    # fhist["GROWSN_EI_year_1"] = fhist["GROWSN_EI_year_1"].rename("GROWSN_EI")
    # fhist["GROWSN_EI_year_1"].attrs = {
    #     "long_name": "annual growing season mean evaporative index ET/P",
    #     "description": (
    #         "ET / P, filtered monthly P = max(P, 0.001) [W/m2]\n" \
    #         f"growing season is defined as the {growsn_nmon} adjacent months with the greatest climatological LAI, computed with xclimate.science.growing_season_month()"
    #     ),
    # }

    # # Climatological evaporative index (ET/P)
    # print("  EI_clim_1")
    # fhist["EI_clim_1"] = fhist["EFLX_LH_TOT_clim_1"] / fhist["EPRECT_clim_1"]
    # fhist["EI_clim_1"] = fhist["EI_clim_1"].rename("EI")
    # fhist["EI_clim_1"].attrs = {
    #     "long_name": "climatological evaporative index ET/P",
    #     "description": "ET / P, filtered monthly P = max(P, 0.001) [W/m2]",
    # }

    # # Climatological aridity index (AI = PET / P = RN / EPRECT)
    # print("  AI_clim_1")
    # fhist["AI_clim_1"] = fhist["RN_clim_1"] / fhist["EPRECT_clim_1"]
    # fhist["AI_clim_1"] = fhist["AI_clim_1"].rename("AI")
    # fhist["AI_clim_1"].attrs = {
    #     "long_name": "aridity index PET/P, higher is more arid",
    #     "description": "Rn / (Lv * P), filtered P = max(P, 0.001) [W/m2]",
    #     "time_mean": f"{time_slice.start} to {time_slice.stop}"
    # }

    # # Growing season aridity index (AI = PET / P = RN / EPRECT)
    # print("  GROWSN_AI_year_1")
    # fhist["GROWSN_AI_year_1"] = fhist["GROWSN_RN_year_1"] / fhist["GROWSN_EPRECT_year_1"]
    # fhist["GROWSN_AI_year_1"] = fhist["GROWSN_AI_year_1"].rename("GROWSN_AI")
    # fhist["GROWSN_AI_year_1"].attrs = {
    #     "long_name": "annual growing season mean aridity index PET/P, higher is more arid",
    #     "description": (
    #         "Rn / (Lv * P), filtered monthly P = max(P, 0.001) [W/m2]\n" \
    #         f"growing season is defined as the {growsn_nmon} adjacent months with the greatest climatological LAI, computed with xclimate.science.growing_season_month()"
    #     ),
    #     "time_mean": f"{time_slice.start} to {time_slice.stop}"
    # }


    # print("\nComputing delta:")
    # variables_delta = [
    #     "EFLX_LH_TOT_month_1",
    #     "FCTR_month_1",
    #     "FGEV_month_1",
    #     "FCEV_month_1",
    #     "TLAI_month_1",
    # ]
    # for v in variables_delta:
    #     print(f"DELTA_{v}")
    #     name = "_".join(v.split("_")[:-2])

    #     prt = fhist[v].isel(member=1)
    #     prt = prt.weighted(prt.time.dt.days_in_month).mean(dim="time")

    #     ctl = fhist[v].isel(member=0)
    #     ctl = ctl.weighted(ctl.time.dt.days_in_month).mean(dim="time")

    #     fhist[f"DELTA_{name}_clim_1"] = prt - ctl


    # Remove variables
    drop_variables = [
        "TLAI_month_1",
        "EFLX_LH_TOT_month_1",
        "FCTR_month_1",
        "FGEV_month_1",
        "FCEV_month_1",

        "EPRECT_month_1",
        "GROWSN_EPRECT_year_1",
        "EPRECT_clim_1",
        "FSDS_month_1",
        "FSR_month_1",
        "FLDS_month_1",
        "FIRE_month_1",

        "FCTR_month_1",
        "SOILWATER_10CM_month_1",
        "TOTSOILLIQ_month_1",
        "SOILWATER_10CM_DT_year_1",
        "TOTSOILLIQ_DT_year_1",
        "FCTR_DT_year_1",
    ]
    for v in drop_variables:
        try:
            fhist.pop(v)
        except KeyError:
            continue


    print("\n=== Compute quantiles and bins ===", flush=True)
    print(list(fhist.keys()), flush=True)
    print(f"Bins: {nbin}")
    print(f"Time period: {year_start}-{year_end}, AGG")

    for v, da in fhist.items():
        print(f"\n  {v} {da.dims} {da.shape}:", flush=True)

        if "year" in v:
            tdim = "year"
            chunks = {"lat": -1, "lon": -1, tdim: -1}
            stack_dims = ["lat", "lon", tdim]
        elif "day" in v:
            tdim = "time"
            chunks = {"lat": -1, "lon": -1, tdim: 365}
            stack_dims = ["lat", "lon", tdim]
        elif "month" in v:
            tdim = "time"
            chunks = {"lat": -1, "lon": -1, tdim: -1}
            stack_dims = ["lat", "lon", tdim]
        else:  # climatology
            chunks = {"lat": -1, "lon": -1}
            stack_dims = ["lat", "lon"]

        # Rechunk to optimize for stacking and quantile computation
        # Chunk along member dimension only, consolidate spatial/temporal dims
        print(" rechunking...", end="", flush=True)
        da = da.chunk(chunks)

        # Persist this variable in distributed memory
        print(" persisting...", end="", flush=True)
        t0 = time.time()
        da_persisted = da.persist()
        wait([da_persisted])  # Wait for persist to complete
        print(f"done in {time.time()-t0:.1f}s", end="", flush=True)

        # Compute quantiles and bins
        print(" computing...", end="", flush=True)
        t0 = time.time()
        x_s = da_persisted.stack(gridcell=stack_dims)
        qs = xclim.get_quantiles(x_s, nbin, ["gridcell"])
        bn = xclim.get_bins(x_s, qs, dim="quantile")
        print(f"done in {time.time()-t0:.1f}s", end="", flush=True)

        # Add metadata to quantiles
        qs = qs.rename("x_edge", quantile="qx")
        qs.attrs = {
            "long_name": f"edges for x bins: {x_s.name}",
            "units": x_s.attrs.get('units', ''),
            "x_long_name": x_s.attrs.get('long_name', ''),
            "x_description": x_s.attrs.get('description', ''),
            "x_time_mean": x_s.attrs.get('time_mean', ''),
        }
        qs["qx"].attrs = {
            "long_name": "quantile edges for x bins",
            "units": "quantile",
        }

        # Add metadata to bins
        bn = bn.unstack().rename("x_bin")
        bn.attrs = {
            "long_name": f"gridcell x bin index: {x_s.name}",
            "units": "index",
            "x_long_name": x_s.attrs.get('long_name', ''),
            "x_description": x_s.attrs.get('description', ''),
            "x_time_mean": x_s.attrs.get('time_mean', ''),
        }
        
        # Combine into a single Dataset
        qs_bn = xr.merge([qs, bn])
        qs_bn = qs_bn.assign_coords(
            {
                'iex': np.arange(nbin + 1),
                'ix': np.arange(nbin),
            }
        )
        qs_bn["iex"].attrs = {
            "long_name": "x edge index",
            "units": "index"
        }
        qs_bn["ix"].attrs = {
            "long_name": "x bin index",
            "units": "index",
        }

        # Drop unwanted variables
        for var_to_drop in ["ltype", "landunit"]:
            if var_to_drop in qs_bn.variables:
                qs_bn = qs_bn.drop_vars(var_to_drop)

        # Select directory for output
        vrootdir = rootdir / f"mask_glc{nonglc_pct_threshold}_snow{snow_pct_threshold}" / v
        os.makedirs(vrootdir, exist_ok=True)

        # Save to NetCDF file
        print(" saving...", end="", flush=True)
        t0 = time.time()
        fname = f"qbin{nbin}.{year_start}-{year_end}.TIMEagg.{v}.{str_mem.zfill(3)}.nc"
        qs_bn.to_netcdf(vrootdir / fname)
        print(f"done in {time.time()-t0:.1f}s to {fname}", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--member",
        type=int,
        required=True,
        help="member index",
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        required=True,
        help="number of cpus",
    )
    parser.add_argument(
        "--nmem",
        type=float,
        required=True,
        help="memory in GB per cpu",
    )
    parser.add_argument(
        "--nbin",
        type=int,
        required=True,
        help="number of quantile bins (x and y)",
    )
    args = parser.parse_args()

    main(
        member=args.member,
        ncpus=args.ncpus,
        nmem=args.nmem,
        nbin=args.nbin,
    )
