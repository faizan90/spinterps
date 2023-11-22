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

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms\stat_vg')
    os.chdir(main_dir)

    # TODO: implement axis rotation along with scaling.

    # tss_file = Path(r'temperature_avg.csv')
    # crd_file = Path(r'temp_daily_coords.csv')
    # time_step = '1989-06-30'

    tss_file = Path(r'precipitation.csv')
    crd_file = Path(r'ppt_daily_coords.csv')
    time_step = '1998-09-10'

    drop_zero_val_flag = True
    # drop_zero_evg_flag = False

    sep = ';'
    #==========================================================================

    tss_sr = pd.read_csv(tss_file, index_col=0, sep=sep).loc[time_step]

    if drop_zero_val_flag:
        tss_sr.replace(0.0, np.nan)

    tss_sr.dropna(inplace=True)

    crd_df = pd.read_csv(crd_file, index_col=0, sep=sep)

    crd_df = crd_df.loc[crd_df.index.drop_duplicates(keep='last')]

    crd_df = crd_df.reset_index().drop_duplicates(
        subset='index', keep='last').set_index('index')

    cmn_stns = tss_sr.index.intersection(crd_df.index)

    assert cmn_stns.size

    print('Total common stations:', cmn_stns.size)

    tss_sr = tss_sr.loc[cmn_stns].copy()
    crd_df = crd_df.loc[cmn_stns, ['X', 'Y', 'Z']].copy()
    # crd_df = crd_df.loc[cmn_stns, ['Z']].copy()
    # crd_df = crd_df.loc[cmn_stns, ['X', 'Y', ]].copy()

    assert tss_sr.shape[0] == crd_df.shape[0]

    clab = 'Z'

    vals = tss_sr.values.copy()
    crds = crd_df.loc[:, clab].values.copy()

    assert np.isfinite(vals).all()
    assert np.isfinite(crds).all()
    #==========================================================================

    plt.scatter(
        rankdata(crds),
        rankdata(vals),
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
