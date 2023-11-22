# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jul 20, 2023

8:40:05 AM

Keywords:

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.stats import norm, rankdata
import matplotlib.pyplot as plt

np.seterr(all='raise')
DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms\krig_sims')
    os.chdir(main_dir)

    # NOTE: A flag is manually set in the steps.py file around line 259 that allows
    # for estimation variance of OK only. The input file should have been created
    # with that flag turned on.
    path_to_krig_nc = Path(r'test_krig_sims_2010_05\kriging.nc')

    path_to_pt_ppt_pkl = Path(r'daily_neckar_ppt_50km_buff_Y1961_2022.pkl')

    path_to_crds_csv = Path(r'daily_ppt_epsg32632.csv')

    time_stamp = '2010-06-19'
    nc_idx = 0

    var_min_val = 0
    var_max_val = 17.0

    truncate_zero_flag = True
    # truncate_zero_flag = False
    #==========================================================================

    ppt_ser = pd.read_pickle(path_to_pt_ppt_pkl).loc[time_stamp]
    ppt_ser.dropna(inplace=True)

    crds_df = pd.read_csv(path_to_crds_csv, sep=';', index_col=0)

    cmn_cols = ppt_ser.index.intersection(crds_df.index)

    assert cmn_cols.size

    ppt_ser = ppt_ser.loc[cmn_cols].copy()
    crds_df = crds_df.loc[cmn_cols, ['X', 'Y']].copy()

    with nc.Dataset(path_to_krig_nc, 'r') as nc_hdl:
        x_crds = nc_hdl['X'][...].data
        y_crds = nc_hdl['Y'][...].data

        data = nc_hdl['OK'][nc_idx].data
        est_vars = nc_hdl['EST_VARS_OK'][nc_idx].data

    if x_crds.ndim == 1:
        x_crds_plt_msh, y_crds_plt_msh = np.meshgrid(x_crds, y_crds)

    else:
        x_crds_plt_msh, y_crds_plt_msh = (x_crds, y_crds)

    probs = rankdata(ppt_ser.values, method='max') / (ppt_ser.shape[0] + 1.0)

    sim = np.full_like(data, np.nan)

    var_in = ppt_ser.values.var()

    for i in range(x_crds.shape[0]):
        for j in range(y_crds.shape[0]):

            mean = data[j, i]

            if np.isnan(mean):
                continue

            var = est_vars[j, i]

            if (var <= 0) or (var > (2 * var_in)) or np.isnan(var):
                data[j, i] = np.nan
                continue
                # raise ValueError((i, j, var))

            x, y = x_crds[i], y_crds[j]

            if True:

                neb_idx = np.argmin(
                    (((crds_df['X'].values - x) ** 2) +
                     ((crds_df['Y'].values - y) ** 2)) ** 0.5)

                neb_prob = probs[neb_idx]

            else:
                neb_idxs = np.argsort(
                    (((crds_df['X'].values - x) ** 2) +
                     ((crds_df['Y'].values - y) ** 2)) ** 0.5)

                neb_prob = np.max(probs[neb_idxs[:5]])

            try:
                est_val = norm.ppf(neb_prob, loc=mean, scale=var ** 0.5)

            except:
                err = None

            if truncate_zero_flag and (est_val < 0):
                est_val = 0.0

            sim[j, i] = est_val

    #==========================================================================

    print('mean:', np.nanmean(data), np.nanmean(sim), ppt_ser.values.mean())
    print('var:', np.nanvar(data), np.nanvar(sim), ppt_ser.values.var())
    print('min:', np.nanmin(data), np.nanmin(sim), ppt_ser.values.min())
    print('max:', np.nanmax(data), np.nanmax(sim), ppt_ser.values.max())

    cmap = 'jet'

    axs = plt.subplots(
        1,
        3,
        width_ratios=(1, 1, 0.1),
        figsize=(10, 5))[1]

    interp_fld_1 = data

    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    #
    #     grd_min_1 = np.nanmin(interp_fld_1)
    #     grd_max_1 = np.nanmax(interp_fld_1)
    #======================================================================

    pclr = axs[0].pcolormesh(
        x_crds_plt_msh,
        y_crds_plt_msh,
        interp_fld_1,
        vmin=var_min_val,
        vmax=var_max_val,
        shading='auto',
        cmap=cmap)

    # cb = plt.colorbar(pclr)
    #
    # cb.set_label(cbar_label)

    axs[-1].set_axis_off()

    plt.colorbar(
        pclr,
        ax=axs[-1],
        label='MEAN',
        orientation='vertical',
        fraction=0.8)

    axs[0].scatter(
        crds_df['X'].values, crds_df['Y'].values, c='k', alpha=0.8, s=1.0)

    axs[0].set_xlabel('Easting')
    axs[0].set_ylabel('Northing')
    #======================================================================

    plt.setp(axs[0].get_xmajorticklabels(), rotation=70)
    axs[0].set_aspect('equal', 'datalim')
    #======================================================================

    interp_fld_2 = sim

    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    #
    #     grd_min_2 = np.nanmin(interp_fld_2)
    #     grd_max_2 = np.nanmax(interp_fld_2)
    #======================================================================

    axs[1].pcolormesh(
        x_crds_plt_msh,
        y_crds_plt_msh,
        interp_fld_2,
        vmin=var_min_val,
        vmax=var_max_val,
        shading='auto',
        cmap=cmap)

    axs[1].scatter(
        crds_df['X'].values, crds_df['Y'].values, c='k', alpha=0.8, s=1.0)
    #======================================================================

    # cb = fig.colorbar(pclr)
    #
    # cb.set_label(cbar_label)

    axs[1].set_xlabel('Easting')
    # axs[1].set_ylabel('Northing')
    # axs[1].yaxis.set_tick_params(labelright=True, labelleft=False)

    plt.setp(axs[1].get_xmajorticklabels(), rotation=70)
    axs[1].set_aspect('equal', 'datalim')
    #======================================================================

    plt.show()

    plt.cla()

    plt.close()

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
