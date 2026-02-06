#!/glade/work/bbuchovecky/miniforge3/envs/data-sci/bin/python3.14
#PBS -N TEMPLATE
#PBS -A ACCOUNT
#PBS -l select=1:ncpus=4:mem=16GB
#PBS -l walltime=00:20:00
#PBS -q develop
#PBS -j oe

"""
Template Python script that can be submitted as a PBS job.
"""

import os
import sys
from dask.distributed import Client, LocalCluster


def main():

    ncpus = 4 # must match allocated ncpus
    nmem = 4  # must match allocated mem / ncpus (memory PER cpu)

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

    # Do computation here

    #########################
    #### END COMPUTATION ####
    #########################

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
