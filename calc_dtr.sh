#!/bin/bash

module load nco

ROOTDIR="/glade/campaign/univ/uwas0155/ppe/historical"
SUBDIR="coupled_simulations"
CASE="f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE"
MEMBERS=(000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028)

GCOMP="lnd"
SCOMP="clm2"

TIMES=(19500101-19591231 19600101-19691231 19700101-19791231 19800101-19891231 19900101-19991231 20000101-20091231 20100101-20141231)
FREQ="day_1"
STREAM="h2"

# TIMES=(195001-201412)
# FREQ="month_1"
# STREAM="h0"

for m in "${MEMBERS[@]}"; do

    echo "MEMBER = ${m}"

    for t in "${TIMES[@]}"; do

        echo "  TIME = ${t}"

        tmax="${ROOTDIR}/${SUBDIR}/${CASE}.${m}/${GCOMP}/proc/tseries/${FREQ}/${CASE}.${m}.${SCOMP}.${STREAM}.TREFMXAV.${t}.nc"
        tmin="${ROOTDIR}/${SUBDIR}/${CASE}.${m}/${GCOMP}/proc/tseries/${FREQ}/${CASE}.${m}.${SCOMP}.${STREAM}.TREFMNAV.${t}.nc"
        dtr="${ROOTDIR}/${SUBDIR}/${CASE}.${m}/${GCOMP}/proc/tseries/${FREQ}/${CASE}.${m}.${SCOMP}.${STREAM}.TREFAV_DTR_calculated.${t}.nc"

        tmax_tmp="${ROOTDIR}/${SUBDIR}/${CASE}.${m}/${GCOMP}/proc/tseries/${FREQ}/tmax.nc"
        tmin_tmp="${ROOTDIR}/${SUBDIR}/${CASE}.${m}/${GCOMP}/proc/tseries/${FREQ}/tmin.nc"

        ncrename -O -v TREFMXAV,DTR $tmax $tmax_tmp
        ncrename -O -v TREFMNAV,DTR $tmin $tmin_tmp

        ncbo -O --op_typ=sbt $tmax_tmp $tmin_tmp $dtr
        ncatted -O -a long_name,DTR,o,c,"diurnal temperature range (TREFMXAV - TREFMNAV)" $dtr $dtr

        rm $tmax_tmp
        rm $tmin_tmp

        echo "  ${dtr}"

    done

done
