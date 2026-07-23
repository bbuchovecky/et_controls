#!/bin/bash

module load nco

ROOTDIR="/glade/campaign/univ/uwas0155/ppe/historical"
SUBDIR="coupled_simulations"
CASE="f.e21.FHIST_BGC.f19_f19_mg17.historical.coupPPE"
MEMBERS=(000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028)

GCOMP="atm"
SCOMP="cam"

# TIMES=(19500101-19591231 19600101-19691231 19700101-19791231 19800101-19891231 19900101-19991231 20000101-20091231 20100101-20141231)
# FREQ="day_1"
# STREAM="h2"

TIMES=(195001-201412)
FREQ="month_1"
STREAM="h0"

for m in "${MEMBERS[@]}"; do

    echo "MEMBER = ${m}"

    for t in "${TIMES[@]}"; do

        echo "  TIME = ${t}"

        precc="${ROOTDIR}/${SUBDIR}/${CASE}.${m}/${GCOMP}/proc/tseries/${FREQ}/${CASE}.${m}.${SCOMP}.${STREAM}.PRECC.${t}.nc"
        precl="${ROOTDIR}/${SUBDIR}/${CASE}.${m}/${GCOMP}/proc/tseries/${FREQ}/${CASE}.${m}.${SCOMP}.${STREAM}.PRECL.${t}.nc"
        prect="${ROOTDIR}/${SUBDIR}/${CASE}.${m}/${GCOMP}/proc/tseries/${FREQ}/${CASE}.${m}.${SCOMP}.${STREAM}.PRECT_calculated.${t}.nc"

        precc_tmp="${ROOTDIR}/${SUBDIR}/${CASE}.${m}/${GCOMP}/proc/tseries/${FREQ}/precc.nc"
        precl_tmp="${ROOTDIR}/${SUBDIR}/${CASE}.${m}/${GCOMP}/proc/tseries/${FREQ}/precl.nc"

        ncrename -O -v PRECC,PRECT_calculated $precc $precc_tmp
        ncrename -O -v PRECL,PRECT_calculated $precl $precl_tmp

        ncbo -O --op_typ=add $precc_tmp $precl_tmp $prect
        ncatted -O -a long_name,PRECT_calculated,o,c,"total precipitation rate (PRECC + PRECL)" $prect $prect

        rm $precc_tmp
        rm $precl_tmp

        echo "  ${prect}"

    done

done
