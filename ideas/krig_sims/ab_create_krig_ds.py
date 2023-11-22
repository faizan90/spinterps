# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jul 20, 2023

5:40:30 PM

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

np.seterr(all='raise')
DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms\krig_sims\test_krig_sims_1961_2020_01')
    os.chdir(main_dir)

    path_to_krig_nc = Path(r'kriging.nc')

    path_to_pt_ppt_pkl = Path(r'../daily_neckar_ppt_50km_buff_Y1961_2022.pkl')

    path_to_crds_csv = Path(r'../daily_ppt_epsg32632.csv')

    truncate_zero_flag = True
    # truncate_zero_flag = False
    #==========================================================================

    ppt_df = pd.read_pickle(path_to_pt_ppt_pkl)

    crds_df_all = pd.read_csv(
        path_to_crds_csv, sep=';', index_col=0).loc[:, ['X', 'Y']]

    cmn_cols = ppt_df.columns.intersection(crds_df_all.index)

    assert cmn_cols.size

    crds_df_all = crds_df_all.loc[cmn_cols, ['X', 'Y']].copy()

    with nc.Dataset(path_to_krig_nc, 'r') as nc_hdl:
        x_crds = nc_hdl['X'][...].data
        y_crds = nc_hdl['Y'][...].data

        data = nc_hdl['OK'][:].data
        est_vars = nc_hdl['EST_VARS_OK'][:].data

        nc_times = pd.DatetimeIndex(nc.num2date(
            nc_hdl['time'][:].data,
            units=nc_hdl['time'].units,
            calendar=nc_hdl['time'].calendar,
            only_use_cftime_datetimes=False))

    sim = np.full_like(data, np.nan)

    for t in range(data.shape[0]):

        nc_time = nc_times[t]

        print(nc_time)

        ppt_ser = ppt_df.loc[nc_time].copy()
        ppt_ser.dropna(inplace=True)

        cmn_cols = ppt_ser.index.intersection(crds_df_all.index)

        ppt_ser = ppt_ser.loc[cmn_cols].copy()
        crds_df = crds_df_all.loc[cmn_cols].copy()

        probs = rankdata(ppt_ser.values, method='max') / (ppt_ser.shape[0] + 1.0)

        var_in = ppt_ser.values.var()

        for i in range(x_crds.shape[0]):
            for j in range(y_crds.shape[0]):

                mean = data[t, j, i]

                if np.isnan(mean):
                    continue

                var = est_vars[t, j, i]

                if (var <= 0) or (var > (2 * var_in)) or np.isnan(var):
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

                est_val = norm.ppf(neb_prob, loc=mean, scale=var ** 0.5)

                if truncate_zero_flag and (est_val < 0):
                    est_val = 0.0

                sim[t, j, i] = est_val

    with nc.Dataset(path_to_krig_nc, 'r+') as nc_hdl:
        sim_nc = nc_hdl.createVariable(
            'OK_SIM',
            'd',
            dimensions=('dimt', 'dimy', 'dimx'))

        sim_nc[:] = sim

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
