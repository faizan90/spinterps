# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jul 2, 2024

11:23:21 AM

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
import pandas as pd
import netCDF4 as nc

DEBUG_FLAG = True


def main():

    main_dir = Path(r'D:\fradnc')
    os.chdir(main_dir)

    ref_tss_pth = Path(
        r'U:\TUM\projects\altoetting\tss\final_hydmod_data\ppt_1H_gkd_dwd_tss.pkl')

    ref_crd_pth = Path(
        r'U:\TUM\projects\altoetting\tss\final_hydmod_data\ppt_1H_gkd_dwd_crds.csv')

    ifl_pth = Path(r'regen_radolan_ppt_2006_2022_ltd.nc')
    ifl_vlb = 'RW'
    ifl_xlb = 'x_utm32n'
    ifl_ylb = 'y_utm32n'
    ifl_tlb = 'time'

    ifl_bfr = 20e3

    otp_pth = Path(rf'{ifl_pth.stem}_ifl{ifl_pth.suffix}')  # Ovrwrt if same.
    #==========================================================================

    tst_flg = False

    if ifl_pth != otp_pth: copy2(ifl_pth, otp_pth)

    with nc.Dataset(otp_pth, 'r+') as ncf_hdl:

        print('Reading netCDF File...')
        var_dts = ncf_hdl[ifl_vlb]
        var_vls = var_dts[:].data

        print('NaN Min Before:', np.nanmin(var_vls))
        print('NaN Max Before:', np.nanmax(var_vls))
        #======================================================================

        var_vls_sts = var_vls.shape[0]

        var_nan_ixs = np.isnan(var_vls)

        var_nan_ixs_any = var_nan_ixs.any(axis=(1, 2))

        if tst_flg:
            assert var_nan_ixs_any.shape[0] == var_vls_sts, (
                var_nan_ixs_any.shape[0], var_vls.shape[0])

        print('Total NaNs before fixing:', var_nan_ixs.sum())

        if not var_nan_ixs_any.any():

            print('All values are valid!')
            return
        #======================================================================

        xcs = ncf_hdl[ifl_xlb][:].data
        ycs = ncf_hdl[ifl_ylb][:].data

        assert xcs.ndim == 2, xcs.ndim

        # Centroid manual.
        # assert var_vls.shape[1:] == (xcs.shape[0] + 1, xcs.shape[1] + 1), (
        #     var_vls.shape[1:], (xcs.shape[0] + 1, xcs.shape[1] + 1))
        #
        # xcs = xcs[:-1,:-1] + (0.5 * (xcs[+1:, +1:] - xcs[:-1,:-1]))
        # ycs = ycs[:-1,:-1] - (0.5 * (ycs[:-1,:-1] - ycs[+1:, +1:]))

        assert var_vls.shape[1:] == (xcs.shape[0], xcs.shape[1]), (
            var_vls.shape[1:], (xcs.shape[0], xcs.shape[1]))

        if tst_flg:
            assert ncf_hdl[ifl_xlb][:].data.min() < xcs.min()
            assert ncf_hdl[ifl_ylb][:].data.min() < ycs.min()

            assert ncf_hdl[ifl_xlb][:].data.max() > xcs.max()
            assert ncf_hdl[ifl_ylb][:].data.max() > ycs.max()

        tcs_dts = ncf_hdl[ifl_tlb]

        tcs_vls = tcs_dts[:].data

        tcs_tim = pd.DatetimeIndex(nc.num2date(
            tcs_vls,
            tcs_dts.units,
            tcs_dts.calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True))
        #======================================================================

        print('Reading TSS File...')
        ref_tss_dfe = pd.read_pickle(ref_tss_pth)

        ref_tcs_itt = ref_tss_dfe.index.intersection(tcs_tim)

        assert ref_tcs_itt.size

        ref_tss_dfe = ref_tss_dfe.loc[ref_tcs_itt,:].copy()

        ref_tss_dfe.columns = ref_tss_dfe.columns.astype(str)

        ref_crd_dfe = pd.read_csv(
            ref_crd_pth, sep=';', index_col=0).loc[:, ['X', 'Y']]

        ref_crd_dfe.index = ref_crd_dfe.index.astype(str)

        crd_ixs = (
            (ref_crd_dfe['X'] > (xcs.min() - ifl_bfr)).values &
            (ref_crd_dfe['X'] < (xcs.max() + ifl_bfr)).values &
            (ref_crd_dfe['Y'] > (ycs.min() - ifl_bfr)).values &
            (ref_crd_dfe['Y'] < (ycs.max() + ifl_bfr)).values)

        assert crd_ixs.any()

        print('Neighbor Count:', crd_ixs.sum())

        ref_crd_dfe = ref_crd_dfe.loc[crd_ixs]

        cmn_idx = ref_tss_dfe.columns.intersection(ref_crd_dfe.index)

        ref_tss_dfe = ref_tss_dfe.loc[:, cmn_idx].copy()
        ref_crd_dfe = ref_crd_dfe.loc[cmn_idx,:].copy()

        tss_ixs = (ref_tss_dfe.count(axis=0) > 0).values

        assert tss_ixs.any()

        print('TSS Count:', tss_ixs.sum())

        ref_tss_dfe = ref_tss_dfe.loc[:, tss_ixs].copy()
        ref_crd_dfe = ref_crd_dfe.loc[tss_ixs,:].copy()

        print('REF TSS Shape:', ref_tss_dfe.shape)
        print('REF CRD Shape:', ref_crd_dfe.shape)
        #======================================================================

        for i in range(var_vls.shape[1]):
            print('PNT:', i)
            for j in range(var_vls.shape[2]):

                nan_tss = var_nan_ixs[:, i, j]

                if not nan_tss.any(): continue

                nan_sum = nan_tss.sum()

                pnt_tss = var_vls[:, i, j]

                ref_crd_dst = (
                    ((ref_crd_dfe['X'].values - xcs[i, j]) ** 2) +
                    ((ref_crd_dfe['Y'].values + xcs[i, j]) ** 2)) ** 0.5

                ref_crd_dst_ixs = np.argsort(ref_crd_dst)

                for k in ref_crd_dst_ixs:
                    if nan_sum == 0: break

                    fnt_ixs = tcs_tim[nan_tss].intersection(
                        ref_tss_dfe.iloc[:, k].dropna().index)

                    if not fnt_ixs.size: continue

                    nan_sum -= fnt_ixs.size

                    fnt_ixs_int = tcs_tim.get_indexer(fnt_ixs)

                    nan_tss[fnt_ixs_int] = False

                    pnt_tss[fnt_ixs_int] = (
                        ref_tss_dfe.loc[fnt_ixs, ref_tss_dfe.columns[k]])

                    # print(k, nan_sum)

                var_vls[:, i, j] = pnt_tss

        var_nan_ixs_fix = np.isnan(var_vls)
        print('Total NaNs after fixing:', var_nan_ixs_fix.sum())

        print('NaN Min After:', np.nanmin(var_vls))
        print('NaN Max After:', np.nanmax(var_vls))
        #======================================================================

        for i in range(var_vls_sts):

            if not var_nan_ixs_any[i]: continue

            var_dts[i] = var_vls[i,:,:]

            if tst_flg:
                assert np.isnan(var_dts[i][var_nan_ixs[i]]).all()

        ncf_hdl.sync()

        var_nan_ixs = None

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
