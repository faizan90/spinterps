# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

02.09.2025

10:12:00

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

from spinterps import sth_rts

DEBUG_FLAG = True


def main():

    '''
    Stitch in space.

    Works for any number of bands.

    Good for rasters with floating dtypes due to NaN.
    Even if inputs are integers, use floating dtypes for outputs!

    Only band-count and cell-size matter for stiching.

    Written with BIGTIFF in mind.

    The inputs should be the ones that are already snipped to proper extents.
    '''

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    # Order matters due to ncs and pts.
    # ipt_rts_phs = [
    #     Path(r'O:\DEBY_Neckar\ka_sbst_forr_poly\ba_snip_dstd__neckar_1km\fac.tif'),
    #     Path(r'P:\DEBY_Danube\ka_sbst_forr_poly\ba_snip_dstd__berg_1km\fac.tif'),
    #     ]

    ipt_rts_phs = [
        Path(r'P:\DEBY_Danube\cc_static_grid_subset\a_corine_b_rescld_clss\U2000_CLC1990_V2020_20u1__L1.tif'),
        Path(r'P:\DEBY_Neckar\cc_static_grid_subset\a_corine_b_rescld_clss\U2000_CLC1990_V2020_20u1__L1.tif'),
        ]

    otp_rtr_nme = r'test_stich2.tif'
    otp_dpe = np.float32

    otp_dir = Path(r'O:\\')
    #==========================================================================

    otp_dir.mkdir(exist_ok=True)

    print('Stitching...')
    sth_rts(ipt_rts_phs, otp_dir / otp_rtr_nme, otp_dpe)
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
