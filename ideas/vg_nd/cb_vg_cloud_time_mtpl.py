# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

08.06.2025

14:16:28

Description:

Keywords:

'''

import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

import pyximport
pyximport.install()

from vg_nd_cy import cmpt_wgtd_dist

DEBUG_FLAG = False

np.seterr(all='raise')


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms\vg_nd')
    os.chdir(main_dir)

    # Order of coordinate columns is: Z;X;Y!

    # itp_tss_pth = Path(r'regnitz\ec_no_trend_series\TMK_tss.csv')

    itp_tss_pth = Path(r'regnitz\ec_no_trend_series\RSK_tss.csv')

    mpg_pol_sze = 1

    otp_dir = Path('otps_time_ppt')
    #==========================================================================

    otp_dir.mkdir(exist_ok=True)

    # Read time series and coordinates as they are.
    if itp_tss_pth.suffix == '.csv':
        itp_tss_dfe = pd.read_csv(itp_tss_pth, sep=';', index_col=0)

        itp_tss_dfe.index = pd.to_datetime(
            itp_tss_dfe.index, format='%Y-%m-%d')

    elif itp_tss_pth.suffix == '.pkl':
        itp_tss_dfe = pd.read_pickle(itp_tss_pth)

    else:
        raise NotImplementedError(itp_tss_pth.suffix)

    itp_tss_srs = itp_tss_dfe.iloc[:30,:].mean(axis=1)

    itp_cds_srs = pd.DataFrame(
        index=np.arange(itp_tss_srs.shape[0]),
        data=np.arange(itp_tss_srs.shape[0]))

    print('Shape of input time series:', itp_tss_srs.shape)
    print('Shape of input coordinates:', itp_cds_srs.shape)

    if mpg_pol_sze == 1:
        ags = (
            itp_tss_srs.values[None,:],
            itp_cds_srs.values,
            otp_dir)

        fit_vgm_sgl_stp(ags)

    else:
        mpg_pol = Pool(mpg_pol_sze)

        mpg_gnr = (
            (itp_tss_srs.values[None,:], itp_cds_srs.values, otp_dir))

        list(mpg_pol.imap_unordered(fit_vgm_sgl_stp, mpg_gnr, chunksize=1))

        mpg_pol.close()
        mpg_pol.join()

    return


def fit_vgm_sgl_stp(ags):

    vls, cds, otp_dir = ags

    evs = cmpt_evgs(vls, 1)

    evs_unq = np.unique(evs)

    if evs_unq.size < 2: return

    fnt_ixs = np.isfinite(evs)

    if not fnt_ixs.any(): return
    assert fnt_ixs.any()

    evs = evs[fnt_ixs]
    #==========================================================================

    cds = cds.astype(np.float32, order='c')

    pts_cnt = cds.shape[0]
    dts_cnt = (pts_cnt * (pts_cnt - 1)) // 2
    dts_wtd = np.full(dts_cnt, np.nan, dtype=np.float32)
    pts = np.full_like(cds, np.nan, dtype=np.float32)

    cmpt_wgtd_dist(cds, np.array([1.0], dtype=np.float32), dts_wtd, pts)

    scn = np.corrcoef(rankdata(evs), dts_wtd[fnt_ixs])[0, 1]
    # print('SCN:', scn)

    if True:
        fgr = plt.figure()

        plt.scatter(dts_wtd[fnt_ixs], evs, alpha=0.65, edgecolor='none')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Weighted distance [-]')
        plt.ylabel('Variogram [-]')

        plt.title(f'SCN: {scn:0.3f}')

        plt.xlim(0, None)
        plt.ylim(0, None)

        # plt.show()

        plt.savefig(otp_dir / f'tme_vg_cloud.png', dpi=100, bbox_inches='tight')

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


def cmpt_evgs(pts, min_cnt_thd):

    pnt_cnt = pts.shape[1]

    vgs_cnt = (pnt_cnt * (pnt_cnt - 1)) // 2
    vgs = np.full(vgs_cnt, np.nan, dtype=np.float32)

    fnt_ixs = np.isfinite(pts)

    pnt_ctr = 0
    for i in range(pnt_cnt):
        for j in range(i + 1, pnt_cnt, 1):

            fnt_ixs_lft_rht = fnt_ixs[:, i] & fnt_ixs[:, j]

            if fnt_ixs_lft_rht.sum() < min_cnt_thd:
                pnt_ctr += 1
                continue

            svg = 0.5 * (pts[:, i] - pts[:, j]) ** 2

            assert svg.size == 1, svg.size

            if svg[0] != 0: vgs[pnt_ctr] = svg[0]

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
