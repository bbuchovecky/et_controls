"""
Utilities for data dictionaries with the structure:
- source_id
    - variable
        - xr.DataArray
"""

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import xarray as xr


def data_dict_nybtes(data_dict):
    total_ngb = 0
    sid_ngb = {}
    for sid, vardict in data_dict.items():
        sid_ngb[sid] = 0
        for var, da in vardict.items():
            sid_ngb[sid] += da.nbytes / 1024 / 1024 / 1024
        total_ngb += sid_ngb[sid]
    
    print(f"total: {total_ngb:0.3f} GB")
    for sid, ngb in sid_ngb.items():
        print(f"{sid:20}: {ngb:0.3f} GB")
    

def check_coords(da: xr.DataArray, coords: Iterable[str]) -> bool:
    """Check that da has coords."""
    for coord in coords:
        if coord not in da.coords:
            return False
        if da[coord].ndim == 0:
            continue  # scalar coord counts as present
        if len(da[coord]) == 0:
            return False
    return True


def equal_coords(a: xr.DataArray, b: xr.DataArray, coords: Iterable[str], atol: float = 1e-3) -> bool:
    """Check that a and b have the same coords."""
    if not check_coords(a, coords) or not check_coords(b, coords):
        return False
    for crd in coords:
        if a[crd].shape != b[crd].shape:
            return False
        if not np.allclose(a[crd], b[crd], atol=atol):
            return False
    return True


def filter_all_variables_available(data_dict: dict, variables: Sequence[str]) -> dict:
    sid_avail = []
    variables_set = set(variables)

    print("=== Not available ===")
    for sid, vardict in data_dict.items():
        if sorted(vardict.keys()) != sorted(variables):
            missing = variables_set - set(vardict.keys())
            print(f"{sid:16}: {missing}")
            continue

        good_coords = True
        test_v = list(vardict.keys())[0]
        test_da = vardict[test_v]
        for v, da in vardict.items():
            if not equal_coords(da, test_da, ("lat", "lon")):
                good_coords = False
                print(f"{sid:16}: {v} {test_da.shape} {da.shape}")
        if good_coords:
            sid_avail.append(sid)
        
    print(f"\n=== Available: {len(sid_avail)} === ")
    print(sid_avail)

    data_dict_filt = {}
    for sid in sid_avail:
        data_dict_filt[sid] = data_dict[sid]

    return data_dict_filt
