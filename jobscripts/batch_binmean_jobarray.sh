#!/bin/bash

SCRIPT="single_ebinmean_TASxbin_TOTANNPRECTybin_dTLAImean.py"
TIMEAGG=(0 1)
NBINS=(15 25 50 75 100)

if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: Required file '$SCRIPT' not found." >&2
    exit 1
fi

for TA in "${TIMEAGG[@]}"; do
    for NB in "${NBINS[@]}"; do
        echo "NBIN=$NB"
        echo "TIMEAGG=$TA"
        qsub -v SCRIPT=$SCRIPT,NBIN=$NB,TIMEAGG=$TA ./single_binmean_jobarray.pbs
    done
done
