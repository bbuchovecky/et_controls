"""
Script to load variables using xclimate.load_fhist and save time mean to netcdf.

This script demonstrates loading FHIST variables and computing/saving time means
over a selected time period to a netcdf file.
"""

from pathlib import Path
import xarray as xr
import xclimate as xclim


# Configuration
OUTPUT_DIR = Path("/glade/derecho/scratch/bbuchovecky/derived")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Variables to load: (varname, component, stream)
VARIABLES = [
    "PRECC_month_1",
    "PRECL_month_1",
    "EFLX_LH_TOT_month_1",
    "TLAI_month_1",
    "ELAI_month_1",
    "TSA_month_1",
    "SOILLIQ_month_1",
    "TOTSOILLIQ_month_1",
    "SOILWATER_10CM_month_1",
    "FSH_month_1",
    "PS_month_1", 
    "TSA_month_1", 
    "RH2M_month_1", 
    "U10_month_1", 
    "FSNO_month_1",
]

# Time period for time mean calculation
START_PERIOD = "1995-01"
END_PERIOD = "2014-12"
TIME_SLICE = slice(START_PERIOD, END_PERIOD)


def main():
    """Load variables and save time mean to netcdf."""
    print("Loading FHIST variables and computing time means...")
    
    # Load each variable and compute time mean
    for v in VARIABLES:
        print(f"  Loading {v}...")
        varname = "_".join(v.split("_")[:-2])
        
        # Load the variable using xclimate
        ds = xclim.load_fhist(v, keep_var_only=True)
        
        # Select time period and compute mean
        print(f"    Computing time mean over {START_PERIOD} to {END_PERIOD}...")
        ds[varname] = ds[varname].sel(time=TIME_SLICE).mean(dim="time")
        
        # Output netcdf filename
        output_file = OUTPUT_DIR / f"fhist.{v}.tmean.{START_PERIOD.replace('-', '')}-{END_PERIOD.replace('-', '')}.nc"

        # Save to netcdf
        print(f"\nSaving to: {output_file}")
        ds.to_netcdf(output_file, mode="w", engine="netcdf4")
    
    print("Done!")


if __name__ == "__main__":
    main()
