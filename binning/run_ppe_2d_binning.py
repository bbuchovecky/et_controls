#!/glade/work/bbuchovecky/miniforge3/envs/data-sci-py312/bin/python3.12
#PBS -N ppe_2d_binning
#PBS -A UWAS0155
#PBS -l select=1:ncpus=4:mem=56gb
#PBS -l walltime=02:00:00
#PBS -q develop
#PBS -j oe
#PBS -o ppe_2d_binning.log
"""
run_ppe_2d_binning.py
=====================
PBS job script: bin a CESM2 PPE variable into arbitrary 2-D space and
write per-bin means to NetCDF.

Execution
---------
    qsub run_ppe_2d_binning.py          # submit to PBS
    python run_ppe_2d_binning.py        # run interactively (uses same config)

Resource assumptions (adjust #PBS directives to match your allocation)
----------------------------------------------------------------------
  select=1           : 1 node
  ncpus=36           : matches typical NCAR Derecho node core count;
                       also the Dask n_workers value below
  mem=256gb          : enough for ~10 ensemble members x 3 variables
                       at monthly, 1-degree resolution
  walltime=02:00:00  : conservative estimate; reduce for small ensembles
"""

import os
import sys
import time
import logging
from pathlib import Path

import xarray as xr
from dask.distributed import Client, LocalCluster

# Ensure the binning module is importable when run from any working directory.
# If ppe_2d_binning.py lives elsewhere, update this path.
sys.path.insert(0, str(Path(__file__).parent))
from ppe_2d_binning import compute_2d_bin_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# ▶  USER CONFIGURATION – edit this block only
# ===========================================================================

# --- Variable names inside the NetCDF files --------------------------------
TARGET_VARNAME = "EFLX_LH_TOT"  # variable to bin (e.g. evapotranspiration)
Y_VARNAME      = "LAI"          # y-axis binning variable (e.g. leaf area index)
X_VARNAME      = "AI"           # x-axis binning variable (e.g. aridity index PET/P)

# --- Binning configuration -------------------------------------------------
N_Y_BINS    = 15
N_X_BINS    = 15
Y_STRATEGY  = "quantile"   # "quantile" | "linear" | "log"
X_STRATEGY  = "quantile"   # "quantile" | "linear" | "log"
Y_RANGE     = None         # e.g. (0.0, 8.0); None → derived from data
X_RANGE     = None         # Budyko aridity index range; None → from data

# --- Dask LocalCluster configuration ---------------------------------------
# n_workers is set equal to the number of CPUs requested from PBS so each
# worker maps to one physical core.  memory_limit caps per-worker RAM to
# prevent a single slow member from exhausting the node.
N_WORKERS = int(os.environ.get("NCPUS", 4))
THREADS_PER_WORKER = 1         # 1 thread/worker avoids GIL contention
MEMORY_LIMIT       = "16GB"    # per worker; total = N_WORKERS x MEMORY_LIMIT

# --- xarray chunking for open_mfdataset ------------------------------------
# Chunk along time so each variable fits comfortably in worker memory.
# Adjust if your files are structured differently (e.g. one file per member).
CHUNKS = {"time": 12}         # 12 months per chunk (monthly data)

# --- Output ----------------------------------------------------------------
OUTPUT_NC  = f"ppe_bin_means_{TARGET_VARNAME}_{Y_VARNAME}_x_{X_VARNAME}.nc"

# ===========================================================================


def load_dataset(pattern: str, varname: str, chunks: dict) -> xr.DataArray:
    """
    Open a multi-file dataset and return the target DataArray.

    Assumes files can be concatenated along 'ensemble_member' if that
    dimension is not already present.  Adjust concat_dim to match your
    file naming convention.
    """
    ds = xr.open_mfdataset(
        pattern,
        combine="nested",
        concat_dim="ensemble_member",
        chunks=chunks,
        parallel=True,     # use Dask to open files concurrently
    )
    da = ds[varname]
    da.name = varname
    return da


def main() -> None:
    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # 1.  Start Dask LocalCluster
    # -----------------------------------------------------------------------
    log.info("Starting Dask LocalCluster: %d workers, %s/worker",
             N_WORKERS, MEMORY_LIMIT)

    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        memory_limit=MEMORY_LIMIT,
        silence_logs=logging.WARNING,
    )
    client = Client(cluster)
    log.info("Dask dashboard: %s", client.dashboard_link)

    try:
        # -------------------------------------------------------------------
        # 2.  Load data
        # -------------------------------------------------------------------
        log.info("Loading datasets …")
        target = load_dataset(TARGET_PATTERN, TARGET_VARNAME, CHUNKS)
        y_var  = load_dataset(Y_VAR_PATTERN,  Y_VARNAME,      CHUNKS)
        x_var  = load_dataset(X_VAR_PATTERN,  X_VARNAME,      CHUNKS)

        log.info("target : %s  %s", target.dims, target.shape)
        log.info("y_var  : %s  %s", y_var.dims,  y_var.shape)
        log.info("x_var  : %s  %s", x_var.dims,  x_var.shape)

        # -------------------------------------------------------------------
        # 3.  Compute 2-D bin means
        # -------------------------------------------------------------------
        log.info(
            "Binning %s into %dx%d (%sx%s) bins...",
            TARGET_VARNAME, N_Y_BINS, N_X_BINS, Y_STRATEGY, X_STRATEGY,
        )

        bin_means = compute_2d_bin_stats(
            target     = target,
            y_var      = y_var,
            x_var      = x_var,
            n_y_bins   = N_Y_BINS,
            n_x_bins   = N_X_BINS,
            y_strategy = Y_STRATEGY,
            x_strategy = X_STRATEGY,
            y_range    = Y_RANGE,
            x_range    = X_RANGE,
            pool_edges_across_ensemble = True,
            parallel   = True,   # dask.delayed over ensemble_member axis
        )

        log.info("bin_means shape: %s", bin_means.shape)

        # -------------------------------------------------------------------
        # 4.  Write NetCDF
        # -------------------------------------------------------------------
        log.info("Writing %s …", OUTPUT_NC)
        # # Bin edges are stored in attrs as Python lists; ensure they
        # # survive round-trip serialisation by converting to numpy arrays.
        # encoding = {bin_means.name or "bin_means": {"zlib": True, "complevel": 4}}
        bin_means.to_netcdf(OUTPUT_NC)
        log.info("NetCDF written.")

    finally:
        client.close()
        cluster.close()

    elapsed = time.perf_counter() - t_start
    log.info("Done.  Total wall time: %.1f s", elapsed)


if __name__ == "__main__":
    main()
