# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

20.05.2025

11:28:52

Description:

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

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\DEBY\observed_point_time_series\dwd_gkd_atr_merged_crctd\daily')

    os.chdir(main_dir)

    itp_tss_pth = Path(r'ppt_1D_gkd_dwd_atr_tss.pkl')

    tme_bgn = '2007-07-01'
    tme_end = '2023-12-30'

    gap_max_sps = 7

    otp_tss_pth = Path(
        f'{itp_tss_pth.stem}__homog_2007_2023{itp_tss_pth.suffix}')
    #==========================================================================

    itp_tss_dfe = pd.read_pickle(itp_tss_pth).loc[tme_bgn:tme_end].copy()

    itp_tss_dfe.dropna(axis=0, how='all', inplace=True)

    itp_tss_cts = itp_tss_dfe.count(axis=1)

    print('Input shape:', itp_tss_dfe.shape)

    itp_tss_fts = np.isfinite(itp_tss_dfe.values)

    tke_ixs = []
    for j in range(itp_tss_fts.shape[1]):

        tss_whr = np.where(itp_tss_fts[:, j])[0]

        if not itp_tss_fts[+0, j]: tss_whr = np.concatenate(
            [[0], tss_whr])

        if not itp_tss_fts[-1, j]: tss_whr = np.concatenate(
            [tss_whr, [itp_tss_fts.shape[0] - 1]])

        tss_whr_dfs = tss_whr[1:] - tss_whr[:-1]

        if tss_whr_dfs.max() > gap_max_sps: continue

        tke_ixs.append(j)

    otp_tss_dfe = itp_tss_dfe.iloc[:, tke_ixs]

    otp_tss_cts = otp_tss_dfe.count(axis=1)

    print('Output shape:', otp_tss_dfe.shape)

    otp_tss_dfe.to_pickle(otp_tss_pth)
    #==========================================================================

    fgr = plt.figure(figsize=(10, 7))

    plt.plot(itp_tss_cts.index, itp_tss_cts.values, label='ITP')
    plt.plot(otp_tss_cts.index, otp_tss_cts.values, label='OTP')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Time')
    plt.ylabel('Active stations')

    plt.ylim(0, None)

    plt.savefig(otp_tss_pth.with_suffix('.png'), bbox_inches='tight')

    plt.close(fgr)
    return


if __name__ == '__main__':
    print('#### Started PID: %d on %s ####\n' % (os.getpid(), time.asctime()))
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
