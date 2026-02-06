"""
Utilities for FHIST PPE analysis.

Eventually will move these over to ppe-intake and xclimate packages.
"""

from collections.abc import Iterable
from typing import List, Tuple, Optional
from pathlib import Path
import yaml
import os
import platform
from glob import glob
import time
import numpy as np
import xarray as xr

from dask_jobqueue import PBSCluster
from dask.distributed import Client, get_client

CESM2_COMPONENT_MAP = {
    "atm": "cam",
    "lnd": "clm2",
    "cice": "cice",
    "mosart": "rof",
    "glc": "cism",
    "cpl": "cpl",
}


def is_dask_available() -> bool:
    """Check if a Dask cluster is running and accessible"""
    try:
        client = get_client()
        return True
    except ValueError:
        return False


def create_dask_cluster(
    account: str,
    nworkers: int,
    ncores: int = 1,
    nmem: str = "5GB",
    walltime: str = "01:00:00",
    print_dash: Optional[bool] = True,
    **kwargs,
):
    """
    Create and scale a dask cluster on either casper or derecho.
    https://ncar.github.io/dask-tutorial/notebooks/05-dask-hpc.html

    Parameters
    ----------
        account : str
            Account to charge core hours for dask workers.
        nworkers : int
            Number of workers to scale up.
        ncores : int
            Requested number of cores.
        nmem : str
            Requested amount of memory, in the form 'XGB'.
        walltime : str
            Requested walltime, in the form '00:00:00'.
        print_dash : Optional[bool]
            Whether to print instrcutions to access dask dashboard, defaults to True.
        **kwargs
            Arguments to pass to PBSCluster.

    Returns
    -------
    client, cluster
        Dask objects corresponding to the client and cluster.
    """
    node = platform.node()
    if "crlogin" in node:
        queue = "casper"
        interface = "ext"
    elif "derecho" in node:
        queue = "develop"
        interface = "hsn0"
    else:
        print(node)
        raise KeyError(
            'must be on "casper" or "derecho", other machines not implemented'
        )

    # Print requested resources
    print(f"account:  {account}")
    print(f"nworkers: {nworkers}")
    print(f"ncores:   {ncores}")
    print(f"nmemory:  {nmem}")
    print(f"walltime: {walltime}")

    # Create the cluster and scale to size
    cluster = PBSCluster(
        cores=ncores,
        memory=nmem,
        queue=queue,
        interface=interface,
        resource_spec=f"select=1:ncpus={str(ncores)}:mem={nmem}",
        account=account,
        walltime=walltime,
        **kwargs,
    )
    client = Client(cluster)
    cluster.scale(nworkers)
    time.sleep(5)

    print(cluster.workers)

    # Create a SSH tunnel to access the dask dashboard locally
    if print_dash:
        user = os.environ.get("USER")
        port = cluster.dashboard_link.split(":")[2].split("/")[0]
        address = cluster.dashboard_link.split(":")[1][2:]
        print("\nTo view the dasl dashboard")
        print("Run the following command in your local terminal:")
        print(
            f"> ssh -N -L {port}:{address}:{port} {user}@{node}.hpc.ucar.edu"
        )  # local command line argument
        print("Open the following link in your local browser:")
        print(f"> http://localhost:{port}/status")  # link to local dask dashboard

    return client, cluster


def close_dask_cluster(
    client,
    cluster,
    remove_std_files: Optional[bool] = True,
) -> None:
    """Close dask cluster and clean up the workspace."""
    client.close()
    cluster.close()
    if remove_std_files:
        for f in glob("dask-worker.*"):
            os.remove(f)



def load_member_id_map() -> dict:
    """Load a member map dictionary from the YAML file."""
    member_id_map_path = Path("members.yml")
    with open(member_id_map_path, "r") as f:
        member_id_map = yaml.safe_load(f)
    return member_id_map


def invert_member_id_map(d):
    inverted = {}
    for param, minmax_dict in d.items():
        for minmax, mem_id in minmax_dict.items():
            inverted[int(mem_id)] = (mem_id, param, minmax)
    return inverted


def get_member_info(
    member_id: int | float | str | List[int | float | str] | np.ndarray | xr.DataArray,
    no_id: Optional[bool] = False,
) -> Tuple | List[Tuple]:
    """Get the tuple (member_id, parameter_name, minmax)."""
    member_id_map = load_member_id_map()
    inverted = invert_member_id_map(member_id_map)

    # Convert all inputs to list
    if isinstance(member_id, (int, float, str)):
        member_id = [member_id]
    elif isinstance(member_id, xr.DataArray):
        member_id = member_id.values.flatten()
    elif isinstance(member_id, np.ndarray):
        member_id = member_id.flatten()
    
    # Ensure list elements are appropriate type
    member_id = [int(m) if isinstance(m, (float, np.floating)) else m for m in member_id]

    info = []
    for mem_id in member_id:
        result = inverted.get(int(mem_id) if isinstance(mem_id, str) else mem_id)
        if result is not None:
            if no_id:
                info.append((result[1], result[2]))
            else:
                info.append(result)

    if len(info) == 1:
        return info[0]
    return info


def get_member_name(
    member_id: int | float | str | List[int | float | str] | np.ndarray | xr.DataArray,
    no_id: Optional[bool] = False,
    delimiter: str = "."
) -> str | List[str]:
    """Get a formatted member name string."""
    info = get_member_info(member_id, no_id)
    
    if isinstance(info, List):
        return [delimiter.join(str(x) for x in i) for i in info]
    return delimiter.join(str(x) for x in info)


def shift_time(ds: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """Shifts time coordinate from [startyear-02, endyear-01] to [startyear-01, (endyear-1)-12]"""
    assert "time" in ds.dims
    if (ds.time[0].dt.month.item() == 2) and (ds.time[-1].dt.month.item() == 1):
        new_time = xr.date_range(
            start=str(ds.time[0].dt.year.item()) + "-01",
            end=str(ds.time[-1].dt.year.item() - 1) + "-12",
            freq="MS",
            calendar="noleap",
            use_cftime=True,
        )
        return ds.assign_coords(time=new_time)
    return ds


def load_var(varname, freq, gcomp, stream="*", extract=None):
    rootpath = Path("/glade/campaign/univ/uwas0155/ppe/historical/coupled_simulations")
    basename = "f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE"
    scomp = CESM2_COMPONENT_MAP[gcomp]

    if not extract:
        extract = varname

    das = []
    for m in range(29):
        ms = str(m).zfill(3)
        mcase = f"{basename}.{ms}"
        da = xr.open_mfdataset(
            rootpath.glob(
                f"{mcase}/{gcomp}/proc/tseries/{freq}/{mcase}.{scomp}.{stream}.{varname}.*.nc"
            )
        )[extract]
        if "time" in da.dims:
            da = shift_time(da)
        das.append(da)

    da = xr.concat(das, dim="member").assign_coords(member=np.arange(29))
    return da


