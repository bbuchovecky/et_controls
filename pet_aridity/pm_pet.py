"""
Copmute Penman-Monteith PET.
"""

from __future__ import annotations

import warnings
import argparse
from pathlib import Path
from distributed import wait

import numpy as np
import xarray as xr

import xclimate as xclim


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PM_PET_VARIABLES = [
    "EFLX_LH_TOT_month_1",
    "FSH_month_1",
    "TSA_month_1",
    "RH2M_month_1",
    "WIND_month_1",
    "PS_month_1",
]
RAD_PET_VARIABLES = [
    "FSNS_month_1",
    "FLNS_month_1",
]
PET_VARIABLES = {
    "pm": PM_PET_VARIABLES,
    "rad": RAD_PET_VARIABLES,
}
PRECIP_VARIABLES = [
    "PRECC_month_1",
    "PRECL_month_1",
]

DEFAULT_FHIST_GRID = xclim.load_fhist_ppe_grid()
DEFAULT_LENS2_GRID = xclim.load_cesm2le_grid()
DEFAULT_GOGA2_GRID = xclim.load_goga2_grid()

DEFAULT_TIME_START = "1950-01"
DEFAULT_TIME_STOP = "2014-12"
DEFAULT_ANALYSIS_CHUNKS = {
    "time": -1,
    "member": 1,
    "lat": 48,
    "lon": 96,
}
DEFAULT_OUTPUT_PATH = Path("/glade/work/bbuchovecky/fhist_ppe_analysis/proc/pet")


# ---------------------------------------------------------------------------
# Annual mean
# ---------------------------------------------------------------------------

def compute_annual_mean(da):
    days_in_month = da.time.dt.days_in_month
    weights = days_in_month.groupby('time.year') / days_in_month.groupby('time.year').sum()
    with xr.set_options(keep_attrs=True):
        return (da * weights).groupby('time.year').sum()


# ---------------------------------------------------------------------------
# P-M PET calculation
# ---------------------------------------------------------------------------


def compute_esat_empirical(temp_celsius):
    """Saturation vapor pressure from empirical form in Schefff & Frierson (2014)."""
    return 610.8 * np.exp((17.27 * temp_celsius) / (temp_celsius + 273.15))


def compute_pm_pet(
    sensible_heat_flux: xr.DataArray | np.ndarray,
    latent_heat_flux: xr.DataArray | np.ndarray,
    air_temperature: xr.DataArray | np.ndarray,
    relative_humidity: xr.DataArray | np.ndarray,
    wind_speed: xr.DataArray | np.ndarray,
    surface_pressure: xr.DataArray | np.ndarray,
) -> xr.DataArray:
    """Calculate Penman-Monteith PET from Schefff & Frierson (2014)

    sensible_heat_flux : xr.DataArray or np.ndarray
        Sensible heat flux [W m-2]
    latent_heat_flux : xr.DataArray or np.ndarray
        Latent heat flux [W m-2]
    air_temperature : xr.DataArray or np.ndarray
        Near-surface air temperature [K]
    relative_humidity : xr.DataArray or np.ndarray
        Near-surface relative humidity [1]
    wind_speed : xr.DataArray or np.ndarray
        Near-surface wind speed [m s-1]
    surface_pressure : xr.DataArray or np.ndarray
        Surface pressure [Pa]

    Scheff, J., & Frierson, D. M. W. (2014). Scaling Potential
    Evapotranspirationwith Greenhouse Warming. Journal of Climate,
    27(4), 1539-1558. https://doi.org/10.1175/JCLI-D-13-00233.1
    """

    # Constants
    L_v = 2.45e6     # [J kg-1]
    r_s = 40         # [s m-1], set to match Cook et al. (2014)
    epsilon = 0.622  # [1]
    c_p = 1005       # [J kg-1 K-1]
    R_v = 461        # [J kg-1 K-1]

    # Drag coefficient constants
    k = 0.41           # [1]
    h = 0.5            # [m], assumed vegetation height for alfalfa
    z_w = 10           # [m]
    z_h = 2            # [m]
    z_om = 0.123 * h   # [m]
    z_oh = 0.0123 * h  # [m]
    d = 0              # [m]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        C_H = (k * k) / (np.log((z_w - d) / z_om) * np.log((z_h - d) / z_oh))

        rho_a = 1.01 * surface_pressure / R_v / air_temperature
        gamma = surface_pressure * c_p / epsilon / L_v
        e_sat = compute_esat_empirical(air_temperature - 273.15)
        delta = L_v * e_sat / R_v / air_temperature / air_temperature

        numerator = (
            (delta * (sensible_heat_flux + latent_heat_flux))
            + (rho_a * c_p * e_sat * (1 - relative_humidity) * C_H * wind_speed)
        )
        denominator = delta + gamma * (1 + r_s * C_H * wind_speed)
        pet = numerator / denominator

    if isinstance(pet, xr.DataArray):
        pet = pet.rename("PET")
        pet.attrs = {
            "long_name": "Penman-Monteith potential evapotranspiration",
            "units": "W/m2",
        }

    return pet


def compute_rad_pet(
    net_shortwave: xr.DataArray,
    net_longwave: xr.DataArray,
) -> xr.DataArray:
    """Compute energy-only PET from Milly & Dunne (2016)."""
    pet = 0.8 * (net_longwave + net_shortwave)
    pet = pet.rename("PET")
    pet.attrs = {
        "long_name": "energy-only potential evapotranspiration (0.8*Rn)",
        "units": "W/m2",
    }
    return pet


def compute_aridity_index(
    precip: xr.DataArray,
    pet: xr.DataArray,
) -> xr.DataArray:
    """Compute the annual mean aridity index."""
    pet_clipped = pet.clip(min=0)
    ai = precip / pet_clipped
    ai = ai.rename("AI")
    ai.attrs = {
        "long_name": "aridity index (P/PET)",
        "description": "computed from annual mean P and PET (PET floor at zero)",
    }
    return ai


def classify_aridity_index(ai: xr.DataArray) -> xr.DataArray:
    """
    Classify gridcells as {hyperarid, arid, semiarid, dry subhumid}.

    Aridity classifications:
    ------------------------
    hyperarid:            AI < 0.05
    arid:         0.05 <= AI < 0.20
    semiarid:     0.20 <= AI < 0.50
    dry subhumid: 0.50 <= AI < 0.65
    """
    ARIDITY_CLASS_BOUNDS = [
        [-np.inf, 0.05],
        [0.05, 0.2],
        [0.2, 0.5],
        [0.5, 0.65],
    ]
    ARIDITY_CLASSES = ["hyperarid", "arid", "semiarid", "dry_subhumid"]

    ai_class_masks = []
    for lo, hi in ARIDITY_CLASS_BOUNDS:
        mask = (ai >= lo) & (ai < hi)
        ai_class_masks.append(mask)
    
    ai_class_masks = xr.concat(ai_class_masks, dim="class")
    ai_class_masks = ai_class_masks.assign_coords(
        {
            "class": np.arange(4),
            "class_name": ("class", ARIDITY_CLASSES)
        }
    )
    ai_class_masks.attrs = {
        "long_name": "aridity classifications",
        "description": (
            "hyperarid:  AI < 0.05\n"
            "arid:  0.05 <= AI < 0.20\n"
            "semiarid:  0.20 <= AI < 0.50\n"
            "dry semiarid  0.50 <= AI < 0.65"
        )
    }
    return ai_class_masks


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


_BAD_SCALAR_COORDS = {"landunit", "column", "pft"}


def _strip_bad_scalar_coords(da: xr.DataArray) -> xr.DataArray:
    """Reconstruct a DataArray without scalar coords that alias a dimension name.
    
    Methods like drop_vars() internally call _to_temp_dataset(), which raises
    ValueError if a scalar coordinate name matches a dimension name. This avoids
    that path entirely by constructing a new DataArray from the raw Variable.
    """
    clean_coords = {k: v for k, v in da.coords.items() if k not in _BAD_SCALAR_COORDS}
    return xr.DataArray(da.variable, coords=clean_coords, name=da.name, attrs=da.attrs)


def load_fhist_variable(
    variable: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    members: list[int] | None = None,
    grid: xr.DataArray | xr.Dataset = DEFAULT_FHIST_GRID,
) -> xr.DataArray:
    """Load FHIST data and align it to the PPE grid."""
    time_slice = slice(time_start, time_stop)

    vv = "_".join(variable.split("_")[:-2])
    da = (
        xclim.load_fhist(variable, keep_var_only=True)[vv]
        .sel(time=time_slice)
        .reindex_like(grid, method="nearest", tolerance=1e-3)
        .where(grid.LANDFRAC > 0)
    )

    if members is not None:
        da = da.sel(member=members)

    return da


def load_lens2_variable(
    variable: str,
    stream: str,
    gcomp: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    members: list[int] | None = None,
    grid: xr.DataArray | xr.Dataset = DEFAULT_LENS2_GRID,
) -> xr.DataArray:
    """Load LENS2 data and align it to the LENS2 grid."""
    time_slice = slice(time_start, time_stop)

    vv = "_".join(variable.split("_")[:-2])
    freq = "_".join(variable.split("_")[-2:])
    da = (
        xclim.load_cesm2le(vv, gcomp, freq, stream)[vv]
        .sel(time=time_slice)
        .reindex_like(grid, method="nearest", tolerance=1e-3)
        .where(grid.LANDFRAC > 0)
    )

    if members is not None:
        da = da.sel(member=members)

    return _strip_bad_scalar_coords(da)


def load_goga2_variable(
    variable: str,
    stream: str,
    gcomp: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    members: list[int] | None = None,
    grid: xr.DataArray | xr.Dataset = DEFAULT_GOGA2_GRID,
) -> xr.DataArray:
    """Load GOGA2 data and align it to the GOGA2 grid."""
    time_slice = slice(time_start, time_stop)

    vv = "_".join(variable.split("_")[:-2])
    freq = "_".join(variable.split("_")[-2:])
    da = (
        xclim.load_goga2(vv, gcomp, freq, stream)[vv]
        .sel(time=time_slice)
        .reindex_like(grid, method="nearest", tolerance=1e-3)
        .where(grid.LANDFRAC > 0)
    )

    if members is not None:
        da = da.sel(member=members)

    return da


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_variable(
    dataset: str,
    variable: str,
    gcomp: str,
    stream: str,
    time_start: str = DEFAULT_TIME_START,
    time_stop: str = DEFAULT_TIME_STOP,
    members: list[int] | None = None,
) -> xr.DataArray:
    """Route to the appropriate dataset loader.

    Parameters
    ----------
    dataset : {"fhist", "lens2", "goga2"}
        Which ensemble to load.
    variable : str
        Variable name; interpretation is dataset-specific (see individual
        loader docstrings).
    time_start, time_stop : str
        ISO time bounds.
    members : list of int or None
        Member indices to load; None loads all.

    Returns
    -------
    xr.DataArray
        Dimensions: (member, time, lat, lon).
    """
    if dataset == "fhist":
        return load_fhist_variable(
            variable=variable,
            time_start=time_start,
            time_stop=time_stop,
            members=members,
        )
    elif dataset == "lens2":
        return load_lens2_variable(
            variable=variable,
            gcomp=gcomp,
            stream=stream,
            time_start=time_start,
            time_stop=time_stop,
            members=members,
        )
    elif dataset == "goga2":
        return load_goga2_variable(
            variable=variable,
            gcomp=gcomp,
            stream=stream,
            time_start=time_start,
            time_stop=time_stop,
            members=members,
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.  Choose 'fhist', 'lens2', or 'goga2'.")


def load_area_weights(dataset: str) -> xr.DataArray:
    """
    Load grid-cell land area weights for the specified dataset.

    Parameters
    ----------
    dataset : {"fhist", "lens2", "goga2"}

    Returns
    -------
    xr.DataArray, dims (lat, lon)
        Land area weights in whatever units the source provides;
        normalization is handled downstream in compute_regional_ols.
    """
    if dataset == "fhist":
        return xclim.load_fhist_ppe_grid().LANDAREA
    elif dataset == "lens2":
        return xclim.load_cesm2le_grid().LANDAREA
    elif dataset == "goga2":
        return xclim.load_goga2_grid().LANDAREA
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'fhist', 'lens2', or 'goga2'.")
    

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute annual mean Penman-Monteith PET from either FHIST PPE, LENS2, or GOGA2."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["fhist", "lens2", "goga2"],
        default="fhist",
        help=(
            "Which ensemble to load. "
            "'fhist' uses xclim.load_fhist; "
            "'lens2' uses xclim.load_cesm2le; "
            "'goga2' uses xclim.load_goga2. "
            "Default: fhist"
        ),
    )
    parser.add_argument(
        "--form",
        choices=["pm", "rad"],
        default="pm",
        help=(
            "Which formulation of PET to use. "
            "'pm' uses Penman-Monteith from Scheff & Frierson (2014); "
            "'rad' uses energy-only from Milly & Dunne (2014). "
            "Default: pm."
        ),
    )
    parser.add_argument(
        "--save-precip",
        action="store_true",
        help="Save annual mean total precipitation."
    )
    parser.add_argument(
        "--stream",
        default="h0",
        type=str,
        help=(
            "Stream for history output."
            "For use with GOGA2 and LENS2;"
            "(e.g., h0 or h6)."
        ),
    )
    parser.add_argument(
        "--time-start",
        default=DEFAULT_TIME_START,
        help=(
            "Start of the analysis period, format YYYY-MM or YYYY-MM-DD. "
            f"Default: {DEFAULT_TIME_START}"
        ),
    )
    parser.add_argument(
        "--time-stop",
        default=DEFAULT_TIME_STOP,
        help=(
            "End of the analysis period, format YYYY-MM or YYYY-MM-DD. "
            f"Default: {DEFAULT_TIME_STOP}"
        ),
    )
    parser.add_argument(
        "--members",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of member indices to process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output NetCDF path",
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
        default='16GB',
        type=str,
        help=(
            "Amount of memory for Dask cluster. "
            "Default: '16GB'"
        )
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.output.is_dir():
        raise ValueError("`--output` must be a directory.")

    client_cluster = None
    if args.dask_cluster:
        client_cluster = xclim.create_dask_cluster(
            account='UWAS0155',
            nworkers=args.dask_workers,
            ncores=args.dask_workers,
            nmem=args.dask_memory,
            walltime='02:00:00',
        )
        client_cluster[0].wait_for_workers(args.dask_workers)
    
    try:
        print(args.dataset.upper())

        # ------------------------------------------------------------------
        # Load variables and compute annual mean
        # ------------------------------------------------------------------
        variables = PET_VARIABLES[args.form]

        monthly_da = {}
        for var in variables + PRECIP_VARIABLES:
            print(f"Loading {var}...")

            if var in ("PS_month_1", "PRECC_month_1", "PRECL_month_1", "FSNS_month_1", "FLNS_month_1"):
                gcomp = "atm"
            else:
                gcomp = "lnd"

            da = load_variable(
                dataset=args.dataset,
                variable=var,
                gcomp=gcomp,
                stream=args.stream,
                time_start=args.time_start,
                time_stop=args.time_stop,
                members=args.members,
            )

            if var == "RH2M_month_1":
                da = da / 100
            
            if client_cluster is not None:
                da = client_cluster[0].persist(da.chunk(DEFAULT_ANALYSIS_CHUNKS))
                wait(da)
            
            monthly_da[var] = da
        
        # Compute total precipitation
        monthly_da["PRECT_month_1"] = monthly_da["PRECL_month_1"] + monthly_da["PRECC_month_1"]

        # ------------------------------------------------------------------
        # Compute PET
        # ------------------------------------------------------------------
        print("Computing PET...")
        if args.form == "pm":
            monthly_pet = compute_pm_pet(
                monthly_da["FSH_month_1"],
                monthly_da["EFLX_LH_TOT_month_1"],
                monthly_da["TSA_month_1"],
                monthly_da["RH2M_month_1"],
                monthly_da["WIND_month_1"],
                monthly_da["PS_month_1"],
            )
        elif args.form == "rad":
            monthly_pet = compute_rad_pet(
                monthly_da["FSNS_month_1"],
                monthly_da["FLNS_month_1"],
            )
        else:
            raise ValueError("`form` must be either 'pm' or 'rad'")
    
        # ------------------------------------------------------------------
        # Compute aridity index
        # ------------------------------------------------------------------
        print("Computing aridity index...")
        annual_pet = compute_annual_mean(monthly_pet)
        annual_precip = compute_annual_mean(monthly_da["PRECT_month_1"]) * 1000 * 2.45e6
        annual_precip.attrs = dict(
            long_name="total precipitation (PRECC+PRECL)",
            units="W/m2",
        )
        annual_ai = compute_aridity_index(annual_precip, annual_pet)
        annual_ai.attrs["pet_form"] = annual_pet.attrs["long_name"]
    
        # ------------------------------------------------------------------
        # Classify gridcells by aridity index
        # ------------------------------------------------------------------
        annual_ai_class = classify_aridity_index(annual_ai)

        # ------------------------------------------------------------------
        # Build output dataset
        # ------------------------------------------------------------------
        pet_ds = xr.Dataset(
            data_vars=dict(PET=annual_pet),
            attrs=dict(dataset=args.dataset),
        )

        precip_ds = xr.Dataset(
            data_vars=dict(PRECT=annual_precip),
            attrs=dict(dataset=args.dataset),
        )

        ai_ds = xr.Dataset(
            data_vars=dict(AI=annual_ai),
            attrs=dict(dataset=args.dataset),
        )

        ai_class_ds = xr.Dataset(
            data_vars=dict(AI_CLASS=annual_ai_class),
            attrs=dict(dataset=args.dataset),
        )

        # ------------------------------------------------------------------
        # Handle paths
        # ------------------------------------------------------------------            
        args.output.parent.mkdir(parents=True, exist_ok=True)

        fname_pet = (
            f"{args.dataset.upper()}.{args.form}_pet.annual_mean"
            f".{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}.nc"
        )
        fname_precip = (
            f"{args.dataset.upper()}.prect.annual_mean"
            f".{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}.nc"
        )
        fname_ai = (
            f"{args.dataset.upper()}.{args.form}_pet_ai.annual_mean"
            f".{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}.nc"
        )
        fname_ai_class = (
            f"{args.dataset.upper()}.{args.form}_pet_ai_class.annual_mean"
            f".{args.time_start.replace('-', '')}-{args.time_stop.replace('-', '')}.nc"
        )

        # ------------------------------------------------------------------
        # Write to NetCDF
        # ------------------------------------------------------------------
        pet_ds.to_netcdf(args.output / fname_pet)
        print(f"Wrote {args.output / fname_pet}")

        if args.save_precip:
            precip_ds.to_netcdf(args.output / fname_precip)
            print(f"Wrote {args.output / fname_precip}")

        ai_ds.to_netcdf(args.output / fname_ai)
        print(f"Wrote {args.output / fname_ai}")

        ai_class_ds.to_netcdf(args.output / fname_ai_class)
        print(f"Wrote {args.output / fname_ai_class}")


    finally:
        if client_cluster is not None:
            xclim.close_dask_cluster(client_cluster)


if __name__ == "__main__":
    main()
