# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

05.06.2025

17:07:38

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
from scipy.stats import rankdata
# from scipy.optimize import Bounds, differential_evolution

from fdiffevo import devn_host_vrn2 as devn_host

import pyximport
pyximport.install()

from vg_nd_cy import cmpt_wgtd_dist

DEBUG_FLAG = False

np.seterr(all='raise')


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms\vg_nd')
    os.chdir(main_dir)

    # Order of coordinate columns is: Z;X;Y!

    # itp_tss_pth = Path(r'regen\ec_no_trend_series\TMK_tss.csv')
    # itp_cds_pth = Path(r'regen\ec_no_trend_series\TMK_cds.csv')
    # cds_cln_lbs = 'Stationshoehe;X_32632;Y_32632'.split(';')

    itp_tss_pth = Path(r'regen\ec_no_trend_series\RSK_tss.csv')
    itp_cds_pth = Path(r'regen\ec_no_trend_series\RSK_cds.csv')
    cds_cln_lbs = 'Stationshoehe;X_32632;Y_32632'.split(';')

    # itp_tss_pth = Path(r'ppt_1D_gkd_dwd_tss.pkl')
    # itp_cds_pth = Path(r'ppt_1D_gkd_dwd_crds.csv')
    # cds_cln_lbs = 'Z;X;Y'.split(';')

    min_cnt_thd = 365 * 5 // 30

    mth = None

    dst_max_thd = 100 * 1e3
    #==========================================================================

    # Read time series and coordinates as they are.
    if itp_tss_pth.suffix == '.csv':
        itp_tss_dfe = pd.read_csv(itp_tss_pth, sep=';', index_col=0)

        itp_tss_dfe.index = pd.to_datetime(
            itp_tss_dfe.index, format='%Y-%m-%d')

    elif itp_tss_pth.suffix == '.pkl':
        itp_tss_dfe = pd.read_pickle(itp_tss_pth)

    else:
        raise NotImplementedError(itp_tss_pth.suffix)

    if mth is not None:
        itp_tss_dfe = itp_tss_dfe.loc[itp_tss_dfe.index.month == mth]

    itp_cds_dfe = pd.read_csv(itp_cds_pth, sep=';', index_col=0)

    itp_cds_dfe = itp_cds_dfe.loc[:, cds_cln_lbs].copy()

    # if itp_tss_dfe.shape[1] > 100:
    #     rnd_ixs = np.random.randint(
    #         0, itp_tss_dfe.shape[1], itp_tss_dfe.shape[1])
    #
    #     rnd_ixs = np.unique(rnd_ixs[:50])
    #
    #     itp_tss_dfe = itp_tss_dfe.iloc[:, rnd_ixs]
    #     itp_cds_dfe = itp_cds_dfe.iloc[rnd_ixs,:]

    print('Shape of input time series:', itp_tss_dfe.shape)
    print('Shape of input coordinates:', itp_cds_dfe.shape)

    evs = cmpt_evgs(itp_tss_dfe.values, min_cnt_thd)

    fnt_ixs = np.isfinite(evs)

    assert fnt_ixs.any()

    pts_cnt = itp_cds_dfe.shape[0]
    dts_cnt = (pts_cnt * (pts_cnt - 1)) // 2
    dts_wtd = np.full(dts_cnt, np.nan, dtype=np.float32)

    pts = np.full(
        (itp_cds_dfe.shape[0], 2), np.nan, dtype=np.float32)

    dts_all = cmpt_wgtd_dist(
        itp_cds_dfe.values[:, [1, 2]].astype(np.float32, order='c'),
        np.full(2, 1.0, dtype=np.float32),
        dts_wtd,
        pts)

    dts_thd_ixs = dts_all <= dst_max_thd

    fnt_ixs &= dts_thd_ixs

    assert fnt_ixs.any()

    evs = evs[fnt_ixs]
    #==========================================================================

    # bds = np.full((itp_cds_dfe.shape[1], 2), np.nan)
    # # bds[0, 0] = 10.
    # # bds[0, 1] = 1e4
    #
    # bds[0, 0] = 0
    # bds[0, 1] = 0
    #
    # # Fix bounds to same value for any one so that the answer is unique.
    # bds[1, 0] = 1.00
    # bds[1, 1] = 1.00
    #
    # bds[2, 0] = 1.00
    # bds[2, 1] = 1.00
    #
    # # bds = Bounds(bds[:, 0], bds[:, 1])
    #
    opt_ags = OPTAGS()
    opt_ags.cds = itp_cds_dfe.values.astype(np.float32, order='c')
    opt_ags.evs = evs
    opt_ags.fnt_ixs = fnt_ixs
    #
    # opt_itr_cnt = [0]
    #
    # icvs_args = (100, 1, 0.02, 1e-4, 0.5, 2.0)
    # size_popn = 400
    # incs = np.array((0.1, 1.0, 0.01))
    # bds_updt_iters = 100
    # bds_updt_alpha = 0.1
    # size_mprg = 24
    #
    # # print('Optimizing...')
    # tmr_bgn = timeit.default_timer()
    #
    # opt_res = devn_host(
    #     True,
    #     (0.01, 0.5),  # buds_mutn
    #     bds,  # buds_prms
    #     (0.7, 1.0),  # buds_rcbn
    #     None,  # crts_args
    #     icvs_args,  # icvs_args
    #     obj_ftn,  # objt_fntn
    #     size_mprg,  # size_mprg
    #     size_popn,  # size_popn
    #     0.001,  # acpt_objt_rato
    #     100000,  # acpt_objt_minm
    #     1,  # algm_cnfg_type
    #     opt_ags,  # objt_fntn_args
    #     None,  # mprg_pool_extl
    #     None,  # mprg_mngr_extl
    #     incs,  # prms_inct_sizs
    #     0.01,  # stop_acpt_rate
    #     5,  # stop_acpt_wndw
    #     50,  # stop_itrs_maxm
    #     20,  # stop_itrs_rjtn
    #     0.02,  # stop_tlrc_prms
    #     1e-8,  # stop_tlrc_objt
    #     bds_updt_iters,  # updt_buds_itrs
    #     bds_updt_alpha,  # bds_updt_alpha
    #     False,  # diag_flag_snap_shot
    #     False,  # diag_flag_optn_args
    #     )
    #
    # # ress = differential_evolution(
    # #     obj_ftn,
    # #     bounds=bds,
    # #     args=((itp_cds_dfe.values, evs, fnt_ixs, opt_itr_cnt),),
    # #     # maxiter=500,
    # #     popsize=500,
    # #     polish=False)
    #
    # tmr_end = timeit.default_timer()
    #
    # # dts_wts = ress.x
    # dts_wts = opt_res.prms_modl_best.astype(np.float32)
    #
    # print(dts_wts)
    #
    # # print(f'{ress.fun:0.3E} using {opt_itr_cnt[0]} iterations')
    # print(f'{opt_res.ofve_best:0.3E} using {opt_itr_cnt[0]} iterations')
    #
    # print('Took', tmr_end - tmr_bgn, 'secs')
    #==========================================================================

    dts_wts = np.array([0, 1, 1], dtype=np.float32)

    pts_cnt = opt_ags.cds.shape[0]
    dts_cnt = (pts_cnt * (pts_cnt - 1)) // 2
    dts_wtd = np.full(dts_cnt, np.nan, dtype=np.float32)
    pts = np.full_like(opt_ags.cds, np.nan, dtype=np.float32)

    cmpt_wgtd_dist(opt_ags.cds, dts_wts, dts_wtd, pts)

    scn = np.corrcoef(rankdata(evs), dts_wtd[fnt_ixs])[0, 1]
    print('SCN:', scn)

    if True:
        fgr = plt.figure()

        plt.scatter(dts_wtd[fnt_ixs], evs, alpha=0.65, edgecolor='none')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Weighted distance [-]')
        plt.ylabel('Variogram [-]')

        plt.title(f'Month: {mth}, SCN: {scn:0.3f}, MTD: {dst_max_thd}')

        plt.xlim(0, None)
        plt.ylim(0, None)

        plt.show()

        plt.close(fgr)
    return


def obj_ftn(pms, opt_ags):

    pms = pms.astype(np.float32)

    cds = opt_ags.cds
    evs = opt_ags.evs
    fnt_ixs = opt_ags.fnt_ixs

    if not hasattr(opt_ags, 'pts'):
        pts_cnt = cds.shape[0]

        dts_cnt = (pts_cnt * (pts_cnt - 1)) // 2

        opt_ags.dts_wtd = np.full(dts_cnt, np.nan, dtype=np.float32)
        opt_ags.pts = np.full_like(cds, np.nan, dtype=np.float32)

    cmpt_wgtd_dist(cds, pms, opt_ags.dts_wtd, opt_ags.pts)

    scn = np.corrcoef(rankdata(evs), rankdata(opt_ags.dts_wtd[fnt_ixs]))[0, 1]

    return 1. - abs(scn)

# def obj_ftn(pms, ags):
#
#     cds, evs, fnt_ixs, itr_cnt = ags
#
#     dts_wtd = cmpt_wgtd_dist(cds, pms)[fnt_ixs]
#
#     # QQ-TFM and then pcn?
#     scn = np.corrcoef(rankdata(evs), rankdata(dts_wtd))[0, 1]
#
#     itr_cnt[0] += 1
#
#     # if (itr_cnt[0] % 100) == 0:
#     #     print(itr_cnt[0], round(scn, 3), dts_wtd_thd_ixs.sum())
#
#     return 1. - abs(scn)

# def cmpt_wgtd_dist(pts, wts):
#
#     pts_cnt = pts.shape[0]
#
#     dts_cnt = (pts_cnt * (pts_cnt - 1)) // 2
#     dts = np.full(dts_cnt, np.nan, dtype=float)
#
#     pnt_ctr = 0
#     for i in range(pts_cnt):
#         for j in range(i + 1, pts_cnt, 1):
#
#             dst = ((((pts[i] - pts[j]) * wts) ** 2).sum()) ** 0.5
#
#             dts[pnt_ctr] = dst
#
#             pnt_ctr += 1
#
#     assert pnt_ctr == dts_cnt
#
#     if False: assert np.all(np.isfinite(dts))
#
#     return dts


def cmpt_evgs(pts, min_cnt_thd):

    pnt_cnt = pts.shape[1]

    vgs_cnt = (pnt_cnt * (pnt_cnt - 1)) // 2
    vgs = np.full(vgs_cnt, np.nan, dtype=float)

    fnt_ixs = np.isfinite(pts)

    pnt_ctr = 0
    for i in range(pnt_cnt):
        for j in range(i + 1, pnt_cnt, 1):

            fnt_ixs_lft_rht = fnt_ixs[:, i] & fnt_ixs[:, j]

            if fnt_ixs_lft_rht.sum() < min_cnt_thd:
                pnt_ctr += 1
                continue

            # From Sampson and Guttorp 1992.
            nrm_lft = pts[:, i][fnt_ixs_lft_rht]
            nrm_rht = pts[:, j][fnt_ixs_lft_rht]

            nrm_lft -= nrm_lft.mean()
            nrm_rht -= nrm_rht.mean()

            sii = (nrm_lft ** 2).mean()
            sjj = (nrm_rht ** 2).mean()

            sij = (nrm_lft * nrm_rht).mean()

            svg = sii + sjj - (2. * sij)

            vgs[pnt_ctr] = svg

            pnt_ctr += 1

    assert pnt_ctr == vgs_cnt
    if False: assert np.all(np.isfinite(vgs))

    return vgs


class OPTAGS:

    def __init__(self):
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
