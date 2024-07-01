# -*- coding: utf-8 -*-

'''
@author: Faizan-Uni-Stuttgart

Jul 10, 2023

10:40:37 AM

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
import matplotlib.pyplot as plt
from scipy.stats import rankdata

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    # tss_file = Path(r'temperature_avg.csv')
    # crd_file = Path(r'temp_daily_coords.csv')
    # time_step = '1989-06-30'

    tss_file = Path(
        r'U:\dwd_meteo\daily\dfs__merged_subset\daily_bayern_50km_buff_ppt_Y1961_2022.pkl')

    crd_file = Path(
        r'U:\dwd_meteo\daily\crds\daily_bayern_50km_buff\daily_ppt_epsg32632.csv')

    min_vld_vals = 365 * 10

    sep = ';'
    #==========================================================================

    tss_df = pd.read_pickle(tss_file)

    for col in tss_df.columns:

        if tss_df[col].count() > min_vld_vals:
            continue

        tss_df[col][:] = np.nan

    tss_df.dropna(how='all', axis=1, inplace=True)

    tss_sr = tss_df.mean(axis=0)

    tss_sr.dropna(inplace=True)

    crd_df = pd.read_csv(crd_file, index_col=0, sep=sep)

    cmn_stns = tss_sr.index.intersection(crd_df.index)

    assert cmn_stns.size

    print('Total common stations:', cmn_stns.size)

    tss_sr = tss_sr.loc[cmn_stns].copy()
    crd_df = crd_df.loc[cmn_stns, ['X', 'Y', 'Z_SRTM']].copy()
    # crd_df = crd_df.loc[cmn_stns, ['Z']].copy()
    # crd_df = crd_df.loc[cmn_stns, ['X', 'Y', ]].copy()

    assert tss_sr.shape[0] == crd_df.shape[0]

    clab = 'Z_SRTM'

    vals = tss_sr.values.copy()
    crds = crd_df.loc[:, clab].values.copy()

    assert np.isfinite(vals).all()
    assert np.isfinite(crds).all()
    #==========================================================================

    plt.scatter(
        crds,
        vals,
        # rankdata(crds),
        # rankdata(vals),
        alpha=0.5,
        label=clab,
        edgecolor='none')

    plt.xlabel(clab)
    plt.ylabel('vals')

    plt.legend()
    #==========================================================================

    plt.grid()
    plt.show()
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
