# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

09.07.2026

10:27:47

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

from spinterps import PointToPointDisAgg

DEBUG_FLAG = True


def main():

    '''
    Important: Time series should be shifted in a way where the value
               corresponds to accumulation till a given time step and
               not for all hours or minutes in that time step.
    '''

    main_dir = Path(r'P:\Synchronize\IWS\Testings\spinterps\dsg')
    os.chdir(main_dir)

    if False:
        ref_tss_pth = Path(r'RS_tss.csv')
        ref_cds_pth = Path(r'RS_cds.csv')
        ref_tfq = pd.Timedelta(1, unit='day')
        ref_tda = pd.Timedelta(24, unit='h')
        ref_tft = '%Y-%m-%d'

        dst_tss_pth = Path(r'RS_05_tss.csv')
        dst_cds_pth = Path(r'RS_05_cds.csv')
        dst_tfq = pd.Timedelta(5, unit='min')
        dst_tda = pd.Timedelta(7, unit='h')
        dst_tft = '%Y-%m-%d %H:%M:%S'

        # dst_tss_pth = Path(r'R1_tss.csv')
        # dst_cds_pth = Path(r'R1_cds.csv')
        # dst_tfq = pd.Timedelta(1, unit='h')
        # dst_tda = pd.Timedelta(6, unit='h')
        # dst_tft = '%Y-%m-%d %H:%M:%S'

        ott_tss_nme = r'RR_1h_to_5min__tss.csv'
        ott_cds_nme = r'RR_1h_to_5min__cds.csv'

    elif False:
        ref_tss_pth = Path(r'R1_tss.csv')
        ref_cds_pth = Path(r'R1_cds.csv')
        ref_tfq = pd.Timedelta(1, unit='h')
        ref_tda = pd.Timedelta(0, unit='h')
        ref_tft = '%Y-%m-%d %H:%M:%S'

        dst_tss_pth = Path(r'RS_05_tss.csv')
        dst_cds_pth = Path(r'RS_05_cds.csv')
        dst_tfq = pd.Timedelta(5, unit='min')
        dst_tda = pd.Timedelta(0, unit='h')
        dst_tft = '%Y-%m-%d %H:%M:%S'

        ott_tss_nme = r'RR_1h_to_5min__tss.csv'
        ott_cds_nme = r'RR_1h_to_5min__cds.csv'

    elif True:
        ref_tss_pth = Path(r'RS_tss.csv')
        ref_cds_pth = Path(r'RS_cds.csv')
        ref_tfq = pd.Timedelta(1, unit='day')
        ref_tda = pd.Timedelta(24, unit='h')
        ref_tft = '%Y-%m-%d'

        dst_tss_pth = Path(r'RS_15_tss.csv')
        dst_cds_pth = Path(r'RS_15_cds.csv')
        dst_tfq = pd.Timedelta(15, unit='min')
        dst_tda = pd.Timedelta(7, unit='h')
        dst_tft = '%Y-%m-%d %H:%M:%S'

        # dst_tss_pth = Path(r'R1_tss.csv')
        # dst_cds_pth = Path(r'R1_cds.csv')
        # dst_tfq = pd.Timedelta(1, unit='h')
        # dst_tda = pd.Timedelta(6, unit='h')
        # dst_tft = '%Y-%m-%d %H:%M:%S'

        ott_tss_nme = r'RR_1d_to_15min__tss.csv'
        ott_cds_nme = r'RR_1d_to_15min__cds.csv'

    ott_dir = Path(Path(__file__).stem)  # / ''
    #==========================================================================

    ott_dir.mkdir(exist_ok=True, parents=True)

    ref_dfe_tss = pd.read_csv(ref_tss_pth, sep=';', index_col=0)
    ref_dfe_tss.index = pd.to_datetime(ref_dfe_tss.index, format=ref_tft)
    ref_dfe_tss = ref_dfe_tss.astype(np.float64)
    ref_dfe_tss.index += ref_tda

    if False:
        print(ref_dfe_tss.mean(axis=1))

    ref_dfe_cds = pd.read_csv(
        ref_cds_pth, sep=';', index_col=0).loc[:, ['X', 'Y']]

    ref_dfe_cds = ref_dfe_cds.astype(np.float64)

    dst_dfe_tss = pd.read_csv(dst_tss_pth, sep=';', index_col=0)
    dst_dfe_tss.index = pd.to_datetime(
        dst_dfe_tss.index, format=dst_tft)

    dst_dfe_tss.index -= dst_tda

    dst_dfe_tss = dst_dfe_tss.astype(np.float64)

    if False:
        dst_dfe_tss_ = dst_dfe_tss.resample(
            ref_tfq, closed='right', label='right').sum()

        print(dst_dfe_tss_.mean(axis=1))

    dst_dfe_cds = pd.read_csv(
        dst_cds_pth, sep=';', index_col=0).loc[:, ['X', 'Y']]

    dst_dfe_cds = dst_dfe_cds.astype(np.float64)
    #==========================================================================

    dsg_ojt = PointToPointDisAgg()
    dsg_ojt.set_ref(ref_dfe_tss, ref_dfe_cds, ref_tfq)
    dsg_ojt.set_dst(dst_dfe_tss, dst_dfe_cds, dst_tfq)

    dsg_ojt.prepare()
    dsg_ojt.disaggregate()

    ott_dfe_tss, ott_dfe_cds = dsg_ojt.get_ott()

    ott_dfe_tss.index += dst_tda

    print(ott_dfe_tss.shape, ott_dfe_cds.shape)

    ott_dfe_tss.to_csv(ott_dir / ott_tss_nme, sep=';')
    ott_dfe_cds.to_csv(ott_dir / ott_cds_nme, sep=';')
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
