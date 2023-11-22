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
from scipy.optimize import differential_evolution, Bounds

from spinterps import get_theo_vg_vals, get_nd_dists

DEBUG_FLAG = True


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms\stat_vg')
    os.chdir(main_dir)

    # tss_file = Path(r'temperature_avg.csv')
    # crd_file = Path(r'temp_daily_coords.csv')
    # time_step = '1989-03-30'

    tss_file = Path(r'precipitation.csv')
    crd_file = Path(r'ppt_daily_coords.csv')
    time_step = '1998-09-10'

    sep = ';'
    #==========================================================================

    tss_sr = pd.read_csv(tss_file, index_col=0, sep=sep).loc[time_step]

    tss_sr.dropna(inplace=True)

    crd_df = pd.read_csv(crd_file, index_col=0, sep=sep)

    crd_df = crd_df.loc[crd_df.index.drop_duplicates(keep='last')]

    crd_df = crd_df.reset_index().drop_duplicates(
        subset='index', keep='last').set_index('index')

    cmn_stns = tss_sr.index.intersection(crd_df.index)

    assert cmn_stns.size

    print('Total common stations:', cmn_stns.size)

    tss_sr = tss_sr.loc[cmn_stns].copy()
    # crd_df = crd_df.loc[cmn_stns, ['X', 'Y', 'Z']].copy()
    crd_df = crd_df.loc[cmn_stns, ['X', 'Y', ]].copy()

    crd_df -= crd_df.min()

    assert tss_sr.shape[0] == crd_df.shape[0]

    assert np.isfinite(tss_sr.values).all()

    values = tss_sr.values.copy()
    crds = crd_df.values.copy()

    svgs_0 = cmpt_svgs(values)
    dists_0 = cmpt_dist(crds)

    dists_srt = np.sort(dists_0)
    svgs_0_srt = np.sort(svgs_0)

    srt_dist_lim = 1.5e5

    srt_dist_thresh_idxs = dists_srt <= srt_dist_lim

    n_sel_vals = srt_dist_thresh_idxs.sum()

    dists_srt = dists_srt[srt_dist_thresh_idxs]
    svgs_0_srt = svgs_0_srt[srt_dist_thresh_idxs]

    plt.scatter(
        dists_srt,
        svgs_0_srt,
        alpha=0.4,
        s=30,
        label='ref',
        edgecolors='none')

    if True:
        bds = np.full((3, 2), np.nan)

        # Sill.
        bds[0, 0] = 0.1
        bds[0, 1] = 1e1

        # Range.
        bds[1, 0] = 1.0
        bds[1, 1] = 7.5

        # Scaler.
        bds[2, 0] = 1e-8
        bds[2, 1] = 1e-4

        print(bds)

        bds = Bounds(bds[:, 0], bds[:, 1])

        print('Optimizing...')
        t_beg = timeit.default_timer()

        ress = differential_evolution(
            obj_ftn_pow,
            bounds=bds,
            args=((dists_srt, svgs_0_srt,),),
            # maxiter=1,
            popsize=100,
            polish=False)

        print(ress.x)

        print('Done.')

        print(f'{ress.fun:0.3E}')

        print('Took', timeit.default_timer() - t_beg, 'secs')

        vg_sill = ress.x[0]
        vg_rnge = ress.x[1]
        sclr = ress.x[2]

        vg_str = f'{vg_sill:0.4f} Pow({vg_rnge:0.4f})'

        print(vg_str)

        plt.plot(
            dists_srt,
            get_theo_vg_vals(vg_str, sclr * dists_srt),
            alpha=0.5,
            label=vg_str,
            c='k')

    # plt.ylim(-0.1, 0.5)
    #==========================================================================

    if True:

        bds = np.full((crds.shape[1] * 4, 2), np.nan)

        for i in range(0, crds.shape[1] * 4, 4):

            # Offset 1.
            bds[i + 0, 0] = -1e4
            bds[i + 0, 1] = +1e4

            # Scaler 1.
            bds[i + 1, 0] = 0.0
            bds[i + 1, 1] = 10.0

            # Exponent 1.
            bds[i + 2, 0] = 1.0
            bds[i + 2, 1] = 4.1

            # Scaler 2.
            bds[i + 3, 0] = 0.0
            bds[i + 3, 1] = 1.0

        print(bds)

        bds = Bounds(bds[:, 0], bds[:, 1])

        print('Optimizing...')
        t_beg = timeit.default_timer()

        ress = differential_evolution(
            obj_ftn,
            bounds=bds,
            args=((crds, svgs_0, vg_str, n_sel_vals),),
            # maxiter=1,
            # popsize=1,
            polish=False)

        print(ress.x)

        print('Done.')

        print(f'{ress.fun:0.3E}')

        print('Took', timeit.default_timer() - t_beg, 'secs')

        crds_1 = get_hi_dim_crds(ress.x, crds)

        dists_1 = cmpt_dist(crds_1)

        plt.scatter(
            dists_1,
            svgs_0,
            alpha=0.4,
            s=10,
            label='sim',
            edgecolors='none')

        # plt.scatter(
        #     dists_1,
        #     get_theo_vg_vals(vg_str, dists_1),
        #     alpha=0.5,
        #     s=10,
        #     label='fit',
        #     edgecolors='none')
    #==========================================================================

    plt.xlim(0, srt_dist_lim)
    plt.ylim(0, svgs_0[np.argmin((dists_1 - srt_dist_lim) ** 2)] * 1.05)

    plt.legend()

    plt.grid()

    plt.gca().set_axisbelow(True)

    plt.savefig('db_dist_srt_vg.png', bbox_inches='tight')

    plt.close()
    return


def get_hi_dim_crds(prms, crds_in):

    crds_ot = np.full(
        (crds_in.shape[0], crds_in.shape[1] * 2), np.nan)

    for i in range(crds_in.shape[1]):

        crds_ot[:,:crds_in.shape[1]] = crds_in

        crds_ot[:, crds_in.shape[1] + i] = (
            ((prms[(i * 4) + 1] * (crds_in[:, i] - prms[(i * 4) + 0])) ** int(prms[(i * 4) + 2])) *
            prms[(i * 4) + 3])

    assert np.all(np.isfinite(crds_ot))

    return crds_ot


def gau_vg(h_arr, sill, rnge):

    # arg = (range, sill)
    a = -3 * ((h_arr ** 2 / rnge ** 2))
    gau_vg = (sill * (1 - np.exp(a)))
    return gau_vg


def pow_vg(h_arr, sill, rnge):

    # arg = (range, sill)
    pow_vg = (sill * (h_arr ** rnge))
    return pow_vg


def sph_vg(h_arr, sill, rnge):

    # arg = (range, sill)
    a = (1.5 * h_arr) / rnge
    b = h_arr ** 3 / (2 * rnge ** 3)
    sph_vg = (sill * (a - b))
    sph_vg[h_arr > rnge] = sill
    return sph_vg


def obj_ftn_pow(prms, args):

    dists_0_srt, svgs_0_srt = args

    sill, rnge, sclr = prms

    svgs_1 = pow_vg(dists_0_srt * sclr, sill, rnge)

    obj_val = ((svgs_0_srt - svgs_1) ** 2).sum()

    # print(f'{obj_val:0.3E}')

    return obj_val


def obj_ftn(prms, args):

    crds_0, svgs_0, vg_str, n_sel_vals = args

    crds_1 = get_hi_dim_crds(prms, crds_0)
    #==========================================================================

    dists_1 = cmpt_dist(crds_1)

    if False:
        srt_idxs = np.argsort(dists_1)

        svgs_1 = svgs_0[srt_idxs]

        obj_val = ((svgs_1[1:] - svgs_1[:-1]) < 0).sum()

    else:

        srt_idxs = np.argsort(dists_1)[:n_sel_vals]

        svgs_1 = svgs_0[srt_idxs]

        theo_vg_vals = get_theo_vg_vals(vg_str, dists_1[srt_idxs])
        obj_val = ((theo_vg_vals - svgs_1) ** 2).sum()

    # print(f'{obj_val:0.3E}, {dists_1.max():0.3E}')

    return obj_val


def cmpt_dist(pts):

    return get_nd_dists(pts)
    #
    # # rows are points, cols are coords
    # assert pts.ndim == 2
    #
    # n_pts = pts.shape[0]
    #
    # n_dists = (n_pts * (n_pts - 1)) // 2
    # dists = np.full(n_dists, np.nan, dtype=float)
    #
    # pt_ctr = 0
    # for i in range(n_pts):
    #     for j in range(n_pts):
    #         if i <= j:
    #             continue
    #
    #         dist = (((pts[i] - pts[j]) ** 2).sum()) ** 0.5
    #
    #         dists[pt_ctr] = dist
    #
    #         pt_ctr += 1
    #
    # assert pt_ctr == n_dists
    #
    # dists_cy = get_nd_dists(pts)
    #
    # assert np.isclose(dists, dists_cy).all()
    # return dists


def cmpt_svgs(pts):

    assert pts.ndim == 1

    n_pts = pts.shape[0]

    n_svgs = (n_pts * (n_pts - 1)) // 2
    svgs = np.full(n_svgs, np.nan, dtype=float)

    pt_ctr = 0
    for i in range(n_pts):
        for j in range(n_pts):
            if i <= j:
                continue

            svg = 0.5 * ((pts[i] - pts[j]) ** 2)

            svgs[pt_ctr] = svg

            pt_ctr += 1

    assert pt_ctr == n_svgs

    assert np.all(np.isfinite(svgs))

    return svgs


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
