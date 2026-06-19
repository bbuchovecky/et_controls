"""
Detect flash drought events using the following criteria:
---------------------------------------------------------
1. initiate FD: SM pentad percentile decreases from >40th to <20th
2. average SM decline rate >=5% in percentile for each pentad
3. duration >=3 pentads 
4. terminate FD: SM pentad percentile rise above 20th


References:
-----------
Yuan, X., Wang, L., Wu, P., Ji, P., Sheffield, J., & Zhang, M. (2019).
Anthropogenic shift towards higher risk of flash drought over China. Nature
Communications, 10(1), 4661. https://doi.org/10.1038/s41467-019-12692-7

Yuan, X., Wang, Y., Ji, P., Wu, P., Sheffield, J., & Otkin, J. A. (2023).
A global transition to flash droughts under climate change. Science,
380(6641), 187-191. https://doi.org/10.1126/science.abn6301

Description of algorithm:
-------------------------
INPUT:
    SM_daily(time, lat, lon)  = daily soil moisture
    upper_threshold           = 40th percentile
    lower_threshold           = 20th percentile
    min_decline_rate          = 5 percentile points per pentad
    min_duration              = 3 pentads

OUTPUT:
    FD_mask(time, lat, lon)      = true during accepted flash drought events
    onset_mask(time, lat, lon)   = true during rapid onset phase
    FD_event_id(time, lat, lon)  = integer event labels

ALGORITHM:
    candidate_start = last pentad with P > 40 before rapid decline
    onset_end       = first pentad with P < 20
    decline_rate    = [P(candidate_start) - P(onset_end)] / [onset_end - candidate_start]

    Accept onset if:
        decline_rate ≥ 5 percentile points per pentad

    Terminate drought when:
        P rises back to ≥ 20

    Accept full event if:
        event duration ≥ 3 pentads
"""

import numpy as np
import xarray as xr


def detect_flash_drought_1d(
    p,
    upper_thresh=40.0,
    lower_thresh=20.0,
    min_decline_rate=5.0,
    min_duration=3,
):
    """
    Detect flash drought events from a 1D pentad percentile time series.

    Parameters
    ----------
    p : 1D array
        Soil moisture percentile time series, values 0-100.
    upper_thresh : float
        Initial wetness threshold. Event onset must begin above this percentile.
    lower_thresh : float
        Drought threshold. Onset is complete once p falls below this value.
    min_decline_rate : float
        Minimum average decline rate in percentile points per pentad.
    min_duration : int
        Minimum event duration in pentads.

    Returns
    -------
    event_mask : 1D boolean array
        True during accepted flash drought events.
    onset_mask : 1D boolean array
        True during accepted onset periods.
    event_id : 1D integer array
        Event number at each pentad. Zero means no event.
    """
    n = len(p)

    event_mask = np.zeros(n, dtype=bool)
    onset_mask = np.zeros(n, dtype=bool)
    event_id = np.zeros(n, dtype=np.int32)

    state = "normal"
    candidate_start = None
    event_start = None
    event_counter = 0

    for t in range(n):
        pt = p[t]

        if np.isnan(pt):
            state = "normal"
            candidate_start = None
            event_start = None
            continue

        if state == "normal":
            # Potential onset begins only from above the 40th percentile.
            if pt > upper_thresh:
                candidate_start = t
                state = "candidate"

        elif state == "candidate":
            p0 = p[candidate_start]

            # If soil moisture gets wetter again, reset candidate start.
            # This keeps the onset monotonic-ish without requiring strict monotonicity.
            if pt > p0:
                candidate_start = t
                p0 = pt

            # If it remains above 40, update the candidate to the latest high point.
            # This prevents artificially long onset periods.
            if pt > upper_thresh:
                candidate_start = t
                p0 = pt

            # Candidate successfully reaches drought threshold.
            if pt < lower_thresh:
                dt = t - candidate_start

                if dt > 0:
                    decline_rate = (p0 - pt) / dt
                else:
                    decline_rate = np.inf

                if decline_rate >= min_decline_rate:
                    event_start = candidate_start
                    onset_end = t
                    state = "active"
                else:
                    # Decline was too slow; this is not a flash drought onset.
                    state = "normal"
                    candidate_start = None
                    event_start = None

            # If candidate drops below 40 but not fast enough yet, keep watching.
            # Optional: reject if max possible rate becomes impossible.

        elif state == "active":
            # Drought terminates once soil moisture rises to or above 20th percentile.
            if pt >= lower_thresh:
                event_end = t - 1

                duration = event_end - event_start + 1

                if duration >= min_duration:
                    event_counter += 1
                    event_mask[event_start:event_end + 1] = True
                    onset_mask[event_start:onset_end + 1] = True
                    event_id[event_start:event_end + 1] = event_counter

                state = "normal"
                candidate_start = None
                event_start = None

                # The recovery point itself may be above 40 and seed a new candidate.
                if pt > upper_thresh:
                    candidate_start = t
                    state = "candidate"

    # Handle event still active at the end of the time series.
    if state == "active":
        event_end = n - 1
        duration = event_end - event_start + 1

        if duration >= min_duration:
            event_counter += 1
            event_mask[event_start:event_end + 1] = True
            onset_mask[event_start:onset_end + 1] = True
            event_id[event_start:event_end + 1] = event_counter

    return event_mask, onset_mask, event_id


def _detect_wrapper(p):
    event_mask, onset_mask, event_id = detect_flash_drought_1d(p)
    return event_mask, onset_mask, event_id


# Load daily soil-moisture



event_mask, onset_mask, event_id = xr.apply_ufunc(
    _detect_wrapper,
    P,
    input_core_dims=[["time"]],
    output_core_dims=[["time"], ["time"], ["time"]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[bool, bool, np.int32],
)
