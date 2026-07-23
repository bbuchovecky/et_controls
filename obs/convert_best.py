"""
Parse a Berkeley Earth land-surface TAVG summary file, convert monthly
anomalies to absolute monthly-mean temperatures, and return the result
as an xarray.DataArray on a monthly time axis.

Berkeley Earth convention: reported anomalies are relative to the
Jan 1951-Dec 1980 climatological monthly mean. That climatology
(12 values, one per calendar month) is given in the file header and is
extracted automatically via regex rather than hardcoded, since the
baseline period/mean can differ across Berkeley Earth product versions.

    T_abs(t) = T_anom(t) + T_clim(month_of_t)

Only the "Monthly Anomaly" and "Monthly Unc." columns (columns 3-4 of
the data block) are used here; the trailing annual/5-yr/10-yr/20-yr
smoothed columns are ignored since they are non-independent running
means of the same monthly series and would be redundant to carry as
a "monthly" DataArray.
"""

import re
import numpy as np
import pandas as pd
import xarray as xr

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _parse_monthly_climatology(header_text: str) -> np.ndarray:
    """
    Extract the 12 Jan-Dec absolute monthly temperatures (°C) from the
    header block, e.g.:

        % Estimated Jan 1951-Dec 1980 monthly absolute temperature:
        %      Jan   Feb   Mar ...
        %      2.56  3.19  5.29 ...
        % +/-  0.07  0.05  0.05 ...

    Returns
    -------
    clim : np.ndarray, shape (12,)
        Absolute monthly climatology, ordered Jan->Dec, matching the
        baseline stated in the file (nominally 1951-1980).
    """
    lines = header_text.splitlines()
    for i, line in enumerate(lines):
        if "monthly absolute temperature" in line.lower():
            # Next line: month abbreviations; line after that: the values.
            value_line = lines[i + 2]
            # Strip leading '%' and whitespace, split on whitespace.
            vals = value_line.lstrip("%").split()
            clim = np.array([float(v) for v in vals])
            if clim.size != 12:
                raise ValueError(
                    f"Expected 12 monthly climatology values, got {clim.size}."
                )
            return clim
    raise ValueError("Monthly absolute temperature climatology not found in header.")


def load_berkeley_monthly_absolute(filepath: str) -> xr.DataArray:
    """
    Read a Berkeley Earth *-TAVG-Trend.txt (or similarly formatted)
    summary file and return absolute monthly-mean land-surface
    temperature as an xarray.DataArray indexed by time.

    Parameters
    ----------
    filepath : str
        Path to the raw Berkeley Earth text file.

    Returns
    -------
    da : xr.DataArray, dims=("time",)
        Absolute monthly TAVG (°C). Carries a co-located 1-sigma-scaled
        uncertainty (Berkeley's reported 95% CI / 1.96, converted to
        an approximate standard error) as the coordinate
        'monthly_unc_95ci' for reference; the raw 95% CI as reported
        is also attached as 'monthly_unc_95ci'.
    """
    with open(filepath, "r") as f:
        raw_text = f.read()

    monthly_clim = _parse_monthly_climatology(raw_text)  # (12,), Jan..Dec

    # Data rows: "YYYY MM  anomaly  unc  [annual anomaly unc] ... "
    # Match only lines that start (after whitespace) with a 4-digit year
    # and a 1-2 digit month, followed by numeric/NaN fields.
    row_pattern = re.compile(
        r"^\s*(\d{4})\s+(\d{1,2})\s+"
        r"(NaN|-?\d+\.\d+)\s+(NaN|-?\d+\.\d+)"  # monthly anomaly, unc
    )

    years, months, anoms, uncs = [], [], [], []
    for line in raw_text.splitlines():
        m = row_pattern.match(line)
        if m is None:
            continue
        yr, mo, anom_str, unc_str = m.groups()
        years.append(int(yr))
        months.append(int(mo))
        anoms.append(np.nan if anom_str == "NaN" else float(anom_str))
        uncs.append(np.nan if unc_str == "NaN" else float(unc_str))

    if not years:
        raise ValueError("No monthly data rows were parsed from the file.")

    df = pd.DataFrame(
        {"year": years, "month": months, "anomaly": anoms, "unc_95ci": uncs}
    )

    # Anomaly -> absolute: add the calendar-month climatological mean.
    df["climatology"] = monthly_clim[df["month"].values - 1]
    df["t_abs"] = df["anomaly"] + df["climatology"]

    # Build a monthly datetime axis (day=1 by convention for monthly data).
    time = pd.to_datetime(
        {"year": df["year"], "month": df["month"], "day": 1}
    )

    da = xr.DataArray(
        data=df["t_abs"].values,
        dims=["time"],
        coords={
            "time": time.values,
            "monthly_unc_95ci": ("time", df["unc_95ci"].values),
        },
        name="tavg_absolute",
        attrs={
            "long_name": "Berkeley Earth land-surface absolute monthly mean temperature",
            "units": "degC",
            "baseline_period": "1951-01-01 to 1980-12-31",
            "method": "anomaly + calendar-month climatology from file header",
            "monthly_climatology_JanDec": monthly_clim.tolist(),
            "uncertainty_note": (
                "monthly_unc_95ci is Berkeley Earth's reported 95% CI "
                "for the monthly anomaly; it is carried unchanged and "
                "does NOT include additional uncertainty in the "
                "1951-1980 climatological baseline itself."
            ),
        },
    )
    da = da.sortby("time")
    return da


if __name__ == "__main__":
    import sys

    fp = sys.argv[1] if len(sys.argv) > 1 else "berkeley_land_tavg.txt"
    da = load_berkeley_monthly_absolute(fp)
    da.to_netcdf("/glade/work/bbuchovecky/data/best/best_global_land_surface_tavg.nc")
