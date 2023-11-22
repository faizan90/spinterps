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

from spinterps import get_theo_vg_vals

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms\stat_vg')
    os.chdir(main_dir)

    tss_file = Path(r'temperature_avg.csv')
    crd_file = Path(r'temp_daily_coords.csv')
    time_step = '1989-03-30'

    # tss_file = Path(r'precipitation.csv')
    # crd_file = Path(r'ppt_daily_coords.csv')
    # time_step = '1998-09-10'

    drop_zero_val_flag = False
    # drop_zero_evg_flag = True

    vg_type = 'Sph'
    vg_rnge = 1e7

    sep = ';'
    #==========================================================================

    tss_sr = pd.read_csv(tss_file, index_col=0, sep=sep).loc[time_step]

    if drop_zero_val_flag:
        tss_sr.replace(0.0, np.nan)

    tss_sr.dropna(inplace=True)

    crd_df = pd.read_csv(crd_file, index_col=0, sep=sep)

    crd_df = crd_df.loc[crd_df.index.drop_duplicates(keep='last')]

    crd_df = crd_df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')

    cmn_stns = tss_sr.index.intersection(crd_df.index)

    assert cmn_stns.size

    print('Total common stations:', cmn_stns.size)

    tss_sr = tss_sr.loc[cmn_stns].copy()
    crd_df = crd_df.loc[cmn_stns, ['X', 'Y', 'Z']].copy()

    # crd_df = crd_df.loc[cmn_stns, ['Z']].copy()
    # crd_df = crd_df.loc[cmn_stns, ['X', 'Y', ]].copy()

    crd_df -= crd_df.min()

    assert tss_sr.shape[0] == crd_df.shape[0]

    assert np.isfinite(tss_sr.values).all()

    values = tss_sr.values.copy()
    crds = crd_df.values.copy()

    svgs_0 = cmpt_svgs(values)
    dists_0 = cmpt_dist(crds)

    vg_sill = svgs_0.max()

    vg_str = f'{vg_sill} {vg_type}({vg_rnge})'
    #==========================================================================

    bds = np.full((18, 2), np.nan)

    # Scaler 1.
    bds[0:3, 0] = 0.0
    bds[0:3, 1] = 1.0
    # bds[2, 0:2] = 1.0

    # Exponent 1.
    bds[3:6, 0] = 0.0
    bds[3:6, 1] = 3.1

    # Scaler 2.
    bds[6:9, 0] = 0.0
    bds[6:9, 1] = 2.0

    # Exponent 2.
    bds[9:12, 0] = 0.0
    bds[9:12, 1] = 3.1

    # Offset 1.
    bds[12:15, 0] = -1e4
    bds[12:15, 1] = +1e4

    # Offset 2.
    bds[15:18, 0] = -1e4
    bds[15:18, 1] = +1e4

    print(bds)

    bds = Bounds(bds[:, 0], bds[:, 1])

    print('Optimizing...')
    t_beg = timeit.default_timer()

    ress = differential_evolution(
        obj_ftn,
        bounds=bds,
        args=((crds, svgs_0, vg_str),),
        # maxiter=500,
        # popsize=1,
        polish=False)

    print(ress.x)

    print('Done.')

    print(f'{ress.fun:0.3E}')

    print('Took', timeit.default_timer() - t_beg, 'secs')
    #==========================================================================

    crds_1 = (
        ress.x[0:3] * ((crds - ress.x[12:15]) ** ress.x[3:6].astype(int)))

    crds_2 = (
        ress.x[6:9] * ((crds - ress.x[15:18]) ** ress.x[9:12].astype(int)))

    crds_1 = np.concatenate((crds, crds_1, crds_2), axis=1)

    dists_1 = cmpt_dist(crds_1)

    srt_idxs = np.argsort(dists_1)

    svgs_1 = svgs_0[srt_idxs]

    obj_val = ((svgs_1[1:] - svgs_1[:-1]) < 0).sum()

    print(obj_val, dists_1.max())

    plt.scatter(
        dists_0,
        svgs_0,
        alpha=0.4,
        s=10,
        label='ref',
        edgecolors='none')

    plt.scatter(
        dists_1,
        svgs_0,
        alpha=0.4,
        s=10,
        label='sim',
        edgecolors='none')

    plt.scatter(
        dists_1,
        get_theo_vg_vals(vg_str, dists_1),
        alpha=0.5,
        s=10,
        label='fit',
        edgecolors='none')

    plt.legend()

    plt.grid()

    plt.savefig('c_nonlin_vg.png', bbox_inches='tight')

    plt.close()
    return


def sph_vg(h_arr, sill, rnge):

    # arg = (range, sill)
    a = (1.5 * h_arr) / rnge
    b = h_arr ** 3 / (2 * rnge ** 3)
    sph_vg = (sill * (a - b))
    sph_vg[h_arr > rnge] = sill
    return sph_vg


def obj_ftn(prms, args):

    crds_0, svgs_0, vg_str = args

    assert prms.size == 18

    crds_1 = (prms[0:3] * ((crds_0 - prms[12:15]) ** prms[3:6].astype(int)))
    crds_2 = (prms[6:9] * ((crds_0 - prms[15:18]) ** prms[9:12].astype(int)))

    crds_1 = np.concatenate((crds_0, crds_1, crds_2), axis=1)
    #==========================================================================

    dists_1 = cmpt_dist(crds_1)

    if False:
        srt_idxs = np.argsort(dists_1)

        svgs_1 = svgs_0[srt_idxs]

        obj_val = ((svgs_1[1:] - svgs_1[:-1]) < 0).sum()

    else:
        theo_vg_vals = get_theo_vg_vals(vg_str, dists_1)
        obj_val = ((theo_vg_vals - svgs_0) ** 2).sum()

    print(round(obj_val), round(dists_1.max()))

    return obj_val


def cmpt_dist(pts):

    # rows are points, cols are coords
    assert pts.ndim == 2

    n_pts = pts.shape[0]

    n_dists = (n_pts * (n_pts - 1)) // 2
    dists = np.full(n_dists, np.nan, dtype=float)

    pt_ctr = 0
    for i in range(n_pts):
        for j in range(n_pts):
            if i <= j:
                continue

            dist = (((pts[i] - pts[j]) ** 2).sum()) ** 0.5

            dists[pt_ctr] = dist

            pt_ctr += 1

    assert pt_ctr == n_dists

    return dists


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
