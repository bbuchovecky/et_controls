#!/glade/work/bbuchovecky/miniforge3/envs/data-sci-py312/bin/python3.12
#PBS -N make_maps
#PBS -A UWAS0155
#PBS -l select=1:ncpus=2:mem=16GB
#PBS -l walltime=04:00:00
#PBS -q casper
#PBS -j oe
#PBS -o jobscripts/logs/

import os
import sys
from pathlib import Path
from datetime import datetime
from dask.distributed import Client, LocalCluster

import numpy as np
import xarray as xr
from scipy import stats
import xskillscore as xs

import xclimate as xclim

# Metadata tag for figure attribution
FNAME = Path(__file__).name
NOW = datetime.now().strftime("%Y-%m-%d")
TAG = f"{FNAME} {NOW}"

OUTDIR = Path("/glade/work/bbuchovecky/et_controls")
OUTDIR_PANEL_MAP = OUTDIR / Path("fig/member_panel/map_timemean")

ALPHA_FDR = 0.05
START_TMEAN = '1995-01'
END_TMEAN = '2014-12'
TIME_SLICE = slice(START_TMEAN,END_TMEAN)
VARIABLES = {
    'RAIN_FROM_ATM_month_1': 'water',
    'QOVER_month_1': 'water',
    'QRUNOFF_month_1': 'water',

    'EFLX_LH_TOT_month_1': 'water',
    'FCTR_month_1': 'water',
    'FCEV_month_1': 'water',
    'FGEV_month_1': 'water',

    'TLAI_month_1': 'veg',
    'GPP_month_1': 'veg',

    'TSA_month_1': 'temp',
    'FSH_month_1': 'rad',
    'FIRE_month_1': 'rad',
    'FLDS_month_1': 'rad',
    'FSR_month_1': 'rad',
    'FSDS_month_1': 'rad',
}


def main():
    ncpus = 2  # must match allocated ncpus
    nmem = 8  # must match allocated mem / ncpus (memory PER cpu)

    memory_limit = f"{nmem}GB"
    tmpdir = os.environ.get("TMPDIR", "/tmp")

    print("Python version:", sys.version)
    print(f"ncpus: {ncpus}")
    print(f"memory_per_worker: {memory_limit}")

    # Avoid oversubscription if libraries spawn processes internally
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Start local cluster
    cluster = LocalCluster(
        n_workers=ncpus,
        threads_per_worker=1,
        processes=True,             # use processes, not threads (this is nuanced...)
        memory_limit=memory_limit,  # per-worker memory limit
        local_directory=tmpdir,     # spill + temp files
        dashboard_address=None,     # no dashboard in batch
    )
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)
    print(f"Workers: {ncpus}, Memory per worker: {memory_limit}")

    ###########################
    #### START COMPUTATION ####
    ###########################

    cmaps = {
        'water': xclim.plot.cmap_smooth_center_color('BrBG'),
        'temp': xclim.plot.cmap_smooth_center_color('RdBu_r'),
        'veg': xclim.plot.cmap_smooth_center_color('PRGn'),
        'rad': xclim.plot.cmap_smooth_center_color('PuOr'),
    }


    grid = xclim.load_fhist_ppe_grid()

    for v, cat in VARIABLES.items():
        print(v)
        name = '_'.join(v.split('_')[:-2])

        x = xclim.load_fhist(v, keep_var_only=True)[name].sel(time=TIME_SLICE).reindex_like(grid, method='nearest', tolerance=1e-3).chunk({'time': -1})

        x_ens = x.sel(member=slice(1,None))
        x_def = x.sel(member=0)

        delta = x_ens - x_def
        dof_eff = xs.effective_sample_size(x_ens, x_def)
        t_stat = delta.mean(dim='time') / (delta.std(dim='time') / np.sqrt(dof_eff))
        p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), dof_eff))
        p_value = xr.DataArray(p_value, coords=t_stat.coords, dims=t_stat.dims)
        fdr_p_value = xclim.multiple_testing.calculate_pval_fdr(p_value, alpha_fdr=ALPHA_FDR)

        fg = xclim.plot.plot_facetgrid_map(delta.where(p_value < fdr_p_value).mean(dim='time'), dim='member', label=f'$\\Delta$ {name} [{x.units}]', cmap=cmaps[cat], robust=True, center=0)
        fg.fig.suptitle(f"Perturbed $-$ Default, Time Mean {TIME_SLICE.start}-{TIME_SLICE.stop}, {name} [{x.units}]", y=1.025, va="center", ha="center")
        fg.fig.text(y=1.025, x=0.975, s=TAG, fontsize=6, ha="right", va="center")
        fg.fig.savefig(OUTDIR_PANEL_MAP / f"d.global.lnd.{TIME_SLICE.start}-{TIME_SLICE.stop}.fdr{str(ALPHA_FDR).replace('.', '')}.FHIST.{name}.png", dpi=300, bbox_inches='tight')

    #########################
    #### END COMPUTATION ####
    #########################

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
