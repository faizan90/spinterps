# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

02.09.2025

12:48:36

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

from spinterps import sth_ncs

DEBUG_FLAG = True


def main():

    '''
    Stitch in space.

    Only time and cell sizes are checked.

    Constant cell size is expected for all ncs.

    The inputs should be the ones that are already snipped to proper extents.
    '''

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    ipt_ncs_phs_lbs = [
        ('x', 'y', 'time', 'petn', Path(r'O:\DEBY_Neckar\ka_sbst_forr_poly\ba_snip_dstd__neckar_1km\petn__daily_1970_2022.nc')),
        ('x', 'y', 'time', 'petn', Path(r'P:\DEBY_Danube\ka_sbst_forr_poly\ba_snip_dstd__berg_1km\petn__daily_1970_2022.nc')),
        ]

    otp_ncf_nme = r'test_stich.nc'
    otp_dpe = np.float32

    cpn_lvl = 1

    otp_dir = Path(r'O:\\')
    #==========================================================================

    otp_dir.mkdir(exist_ok=True)

    print('Stitching...')
    sth_ncs(ipt_ncs_phs_lbs, otp_dir / otp_ncf_nme, otp_dpe, cpn_lvl)
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
