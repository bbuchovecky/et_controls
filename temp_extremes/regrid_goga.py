"""
Regrid GOGA2 to 2-degree resolution using conservative (area-weighted) remapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import xarray as xr
import xclimate as xclim


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_PATH = Path("/glade/derecho/scratch/bbuchovecky/derived/regrid_goga")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regrid a GOGA2 variable to the FHIST PPE 2-degree grid and save output.",
    )
    parser.add_argument(
        "--variable",
        required=True,
        help="Variable name including frequency suffix, e.g. TREFHT_day_1"
    )
    parser.add_argument(
        "--stream",
        required=True,
        help="GOGA2 stream (e.g. h0, h1)"
    )
    parser.add_argument(
        "--gcomp",
        required=True,
        choices=["lnd", "atm"],
        help="General model component for GOGA2: lnd or atm"
    )
    parser.add_argument(
        "--time-start",
        default="1950-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--time-stop",
        default="2014-12-31",
        help="Stop date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except write output file"
    )
    parser.add_argument(
        "--dask-cluster",
        action="store_true",
        help="Spin up a Dask cluster via xclim.create_dask_cluster.",
    )
    parser.add_argument(
        "--dask-workers",
        default=2,
        type=int,
        help=(
            "Number of Dask workers (equal to ncores). "
            "Default: 2"
        )
    )
    parser.add_argument(
        "--dask-memory",
        default='32GB',
        type=str,
        help=(
            "Amount of memory for Dask cluster. "
            "Default: '32GB'"
        )
    )
    parser.add_argument(
        "--dask-walltime",
        default="01:00:00",
        type=str,
        help=(
            "Walltime for Dask cluster. "
            "Default: '01:00:00'"
        )
    )
    return parser.parse_args()


def regrid_to_target(da: xr.DataArray, target_grid: xr.DataArray) -> xr.DataArray:
    """
    Attempt conservative (area-weighted) regridding using available tools.

    Use xESMF conservative regridding. Raises RuntimeError if xesmf is not
    installed or the regridding fails.
    """
    try:
        import xesmf as xe
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "xESMF is required for regridding. Install xesmf (pip install xesmf)."
        ) from exc

    # Build minimal source/target grid structures for xESMF. xESMF accepts
    # xarray.Dataset or dict-like inputs with 1D lon/lat coordinates.
    try:
        # xESMF expects xarray Dataset/DataArray inputs with lon/lat variables
        # Build minimal xarray.Datasets retaining the original dims (1D or 2D).
        lon_dims_src = da["lon"].dims
        lat_dims_src = da["lat"].dims
        src = xr.Dataset(
            {
                "lon": (lon_dims_src, da["lon"].values),
                "lat": (lat_dims_src, da["lat"].values),
            }
        )
        print(f"Source: lon={len(src.lon)}, lat={len(src.lat)}")

        lon_dims_tgt = target_grid["lon"].dims
        lat_dims_tgt = target_grid["lat"].dims
        tgt = xr.Dataset(
            {
                "lon": (lon_dims_tgt, target_grid["lon"].values),
                "lat": (lat_dims_tgt, target_grid["lat"].values),
            }
        )
        print(f"Target: lon={len(tgt.lon)}, lat={len(tgt.lat)}")

        regridder = xe.Regridder(src, tgt, "conservative", reuse_weights=False)
        # xESMF returns an xarray.DataArray when called with a DataArray
        return regridder(da)
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError("xESMF regridding failed") from exc


def main() -> None:
    args = parse_args()
    variable = args.variable
    stream = args.stream
    gcomp = args.gcomp

    # Derive name and frequency parts to call xclimate loader
    vv = "_".join(variable.split("_")[:-2])
    freq = "_".join(variable.split("_")[-2:])
    
    # Create Dask cluster
    client_cluster = None
    if args.dask_cluster:
        print("Creating Dask cluster...")
        client_cluster = xclim.create_dask_cluster(
            account='UWAS0155',
            nworkers=args.dask_workers,
            ncores=args.dask_workers,
            nmem=args.dask_memory,
            walltime=args.dask_walltime,
        )
        client_cluster[0].wait_for_workers(args.dask_workers)
    
    try:
        print(f"Loading GOGA2 variable {variable} (var={vv}, freq={freq}, stream={stream})")
        try:
            ds = xclim.load_goga2(vv, gcomp, freq, stream)
        except Exception as exc:
            print(f"Failed to load GOGA2 data: {exc}", file=sys.stderr)
            raise

        if vv not in ds:
            print(f"Loaded dataset does not contain expected variable '{vv}'", file=sys.stderr)
            raise KeyError(vv)

        da = (
            ds[vv].sel(time=slice(args.time_start, args.time_stop))
            .chunk({"member": 2, "time": 365, "lat": -1, "lon": -1})
        )
        print("Original:")
        print(da)

        # Load target FHIST PPE grid
        print("Loading target FHIST PPE grid")
        target_grid = xclim.load_fhist_ppe_grid()

        print("Regridding...")
        da_regridded = regrid_to_target(da, target_grid)
        da_regridded.attrs = ds[vv].attrs
        print("Regridded:")
        print(da_regridded)

        out_dir = args.output
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"GOGA2_{variable}_{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}_regridded.nc"
        out_path = out_dir / fname

        print(f"Saving output to {out_path}")
        if not args.dry_run:
            # ensure dataset wrapper for attributes
            xr.Dataset({vv: da_regridded}).to_netcdf(out_path)
        else:
            print("Dry run; not writing output")
        
    finally:
        if client_cluster is not None:
            xclim.close_dask_cluster(client_cluster, remove_std_files=False)


if __name__ == "__main__":
    main()



