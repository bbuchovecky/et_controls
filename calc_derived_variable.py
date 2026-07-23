"""
Compute a derived variable and save to 
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew

import xclimate as xclim

