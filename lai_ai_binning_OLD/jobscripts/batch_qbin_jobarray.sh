#!/bin/bash

SCRIPT="single_qbin.py"
NBINS=(15 25 50 75 100)

if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: Required file '$SCRIPT' not found." >&2
    exit 1
fi

for NB in "${NBINS[@]}"; do
    echo "NBIN=$NB"
    qsub -v SCRIPT=$SCRIPT,NBIN=$NB ./single_qbin_jobarray.pbs
done
