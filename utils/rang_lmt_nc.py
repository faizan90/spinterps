# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jul 2, 2024

10:44:17 AM

Description:

Keywords:

'''

import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from shutil import copy2

import numpy as np
import netCDF4 as nc

DEBUG_FLAG = False


def main():

    main_dir = Path(r'D:\fradnc')
    os.chdir(main_dir)

    inp_pth = Path(r'regen_radolan_ppt_2006_2022.nc')
    otp_pth = Path(rf'{inp_pth.stem}_ltd{inp_pth.suffix}')  # Ovrwrt, if same.

    var_nam = 'RW'

    # Beyond limits are set to NaN.
    var_llm = 0.000
    var_ulm = 500.0
    #==========================================================================

    tst_flg = False

    assert var_llm < var_ulm, (var_llm, var_ulm)

    if inp_pth != otp_pth: copy2(inp_pth, otp_pth)

    llm_ctr = 0
    ulm_ctr = 0
    with nc.Dataset(otp_pth, 'r+') as ncf_hdl:

        var_dts = ncf_hdl[var_nam]
        var_vls = var_dts[:].data  # Post testing!

        print('NaN Min Before:', np.nanmin(var_vls))
        print('NaN Max Before:', np.nanmax(var_vls))

        var_vls_sts = var_vls.shape[0]

        # Lower limits.
        var_llm_ixs = var_vls < var_llm

        var_llm_ixs_any = var_llm_ixs.any(axis=(1, 2))

        if tst_flg:
            assert var_llm_ixs_any.shape[0] == var_vls_sts, (
                var_llm_ixs_any.shape[0], var_vls.shape[0])

        if var_llm_ixs_any.any():
            for i in range(var_vls_sts):

                if not var_llm_ixs_any[i]: continue

                llm_ctr += var_llm_ixs[i].sum()

                var_stp = var_vls[i,:,:]

                var_stp[var_llm_ixs[i]] = np.nan

                var_dts[i] = var_stp

                if tst_flg:
                    assert np.isnan(var_dts[i][var_llm_ixs[i]]).all()

            ncf_hdl.sync()

        var_llm_ixs = None

        print('Total llms:', llm_ctr)

        # Upper limits.
        var_ulm_ixs = var_vls > var_ulm

        var_ulm_ixs_any = var_ulm_ixs.any(axis=(1, 2))

        if tst_flg:
            assert var_ulm_ixs_any.shape[0] == var_vls_sts, (
                var_ulm_ixs_any.shape[0], var_vls.shape[0])

        if var_ulm_ixs_any.any():
            for i in range(var_vls_sts):

                if not var_ulm_ixs_any[i]: continue

                ulm_ctr += var_ulm_ixs[i].sum()

                var_stp = var_vls[i,:,:]

                var_stp[var_ulm_ixs[i]] = np.nan

                var_dts[i] = var_stp

                if tst_flg:
                    assert np.isnan(var_dts[i][var_ulm_ixs[i]]).all()

            ncf_hdl.sync()

        print('Total ulms:', ulm_ctr)

        var_ulm_ixs = var_vls = None

        var_vls = var_dts[:].data  # Post testing!

        print('NaN Min After:', np.nanmin(var_vls))
        print('NaN Max After:', np.nanmax(var_vls))
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
