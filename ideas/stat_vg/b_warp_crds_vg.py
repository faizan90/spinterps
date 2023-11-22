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

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms\stat_vg')
    os.chdir(main_dir)

    # tss_file = Path(r'temperature_avg.csv')
    # crd_file = Path(r'temp_daily_coords.csv')
    # time_step = '1989-06-30'

    tss_file = Path(r'precipitation.csv')
    crd_file = Path(r'ppt_daily_coords.csv')
    time_step = '1998-09-10'

    dist_thresh = 50e3

    drop_zero_val_flag = True
    # drop_zero_evg_flag = True

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

    assert tss_sr.shape[0] == crd_df.shape[0]

    assert np.isfinite(tss_sr.values).all()

    values = tss_sr.values.copy()
    crds = crd_df.values.copy()

    svgs_0 = cmpt_svgs(values)
    dists_0 = cmpt_dist(crds)
    #==========================================================================

    bds = np.empty((crd_df.shape[1] + (crd_df.shape[1] * 3), 2))
    bds[0, 0] = 0.00
    bds[0, 1] = 0.01

    bds[1, 0] = 0.00
    bds[1, 1] = 0.01

    bds[2, 0] = 0.00
    bds[2, 1] = 1.00

    bds[3:, 0] = -np.pi
    bds[3:, 1] = +np.pi

    bds = Bounds(bds[:, 0], bds[:, 1])

    print('Optimizing...')
    t_beg = timeit.default_timer()

    ress = differential_evolution(
        obj_ftn,
        bounds=bds,
        args=((crds, svgs_0, dists_0, dist_thresh),),
        maxiter=500,
        # popsize=1,
        polish=False)

    print(ress.x)

    print('Done.')

    print(f'{ress.fun:0.3E}')

    print('Took', timeit.default_timer() - t_beg, 'secs')
    #==========================================================================

    plt.scatter(
        dists_0[dists_0 < dist_thresh],
        svgs_0[dists_0 < dist_thresh],
        alpha=0.4)

    crds_12 = ress.x[:crds.shape[1]] * crds

    uvec_r = get_rotated_unit_vec2(ress.x[crds_12.shape[1]:])

    crds_1 = crds_12.copy()
    for i in range(crds_1.shape[1]):
        crds_1[:, i] = np.dot(crds_12, uvec_r[i,:])

    dists_1 = cmpt_dist(crds_1)

    svgs_1 = svgs_0[dists_0 < dist_thresh]

    dists_1 = dists_1[dists_0 < dist_thresh]

    plt.scatter(dists_1, svgs_1, alpha=0.4)

    plt.grid()
    plt.show()
    return


def obj_ftn(prms, args):

    crds_0, svgs_0, dists_0, dist_thresh = args

    assert prms.size == (crds_0.shape[1] + (crds_0.shape[1] * 3))

    crds_12 = prms[:crds_0.shape[1]] * crds_0
    #==========================================================================

    uvec_r = get_rotated_unit_vec2(prms[crds_0.shape[1]:])

    crds_1 = crds_12.copy()
    for i in range(crds_1.shape[1]):
        crds_1[:, i] = np.dot(crds_12, uvec_r[i,:])
    #==========================================================================

    dists_1 = cmpt_dist(crds_1)

    dists_1 = dists_1[dists_0 < dist_thresh]
    svgs_1 = svgs_0[dists_0 < dist_thresh]

    srt_idxs = np.argsort(dists_1)

    svgs_1 = svgs_1[srt_idxs]

    obj_val = ((svgs_1[1:] - svgs_1[:-1]) < 0).sum()

    return obj_val


def get_rotated_unit_vec(angles):

    cos_r1, cos_r2, cos_r3 = np.cos(angles)
    sin_r1, sin_r2, sin_r3 = np.sin(angles)

    xax_x = (cos_r1 * cos_r2)
    xax_y = (cos_r1 * sin_r2 * sin_r3) - (sin_r1 * cos_r3)
    xax_z = (cos_r1 * sin_r2 * cos_r3) + (sin_r1 * sin_r3)

    yax_x = (sin_r1 * cos_r2)
    yax_y = (sin_r1 * sin_r2 * sin_r3) + (cos_r1 * cos_r3)
    yax_z = (sin_r1 * sin_r2 * cos_r3) - (cos_r1 * sin_r3)

    zax_x = -sin_r2
    zax_y = (cos_r2 * sin_r3)
    zax_z = (cos_r2 * cos_r3)

    uvec_r = np.array([
        [xax_x, xax_y, xax_z],
        [yax_x, yax_y, yax_z],
        [zax_x, zax_y, zax_z],
        # [0, 0, 1],
        ],
        dtype=float)

    return uvec_r


def get_rotated_unit_vec2(angles):

    cos_r1, cos_r2, cos_r3, cos_s1, cos_s2, cos_s3, cos_t1, cos_t2, cos_t3 = np.cos(angles)
    sin_r1, sin_r2, sin_r3, sin_s1, sin_s2, sin_s3, sin_t1, sin_t2, sin_t3 = np.sin(angles)

    _ = cos_t1, sin_t1

    xax_x = (cos_r1 * cos_r2)
    xax_y = (cos_r1 * sin_r2 * sin_r3) - (sin_r1 * cos_r3)
    xax_z = (cos_r1 * sin_r2 * cos_r3) + (sin_r1 * sin_r3)

    yax_x = (sin_s1 * cos_s2)
    yax_y = (sin_s1 * sin_s2 * sin_s3) + (cos_s1 * cos_s3)
    yax_z = (sin_s1 * sin_s2 * cos_s3) - (cos_s1 * sin_s3)

    zax_x = -sin_t2
    zax_y = (cos_t2 * sin_t3)
    zax_z = (cos_t2 * cos_t3)

    uvec_r = np.array([
        [xax_x, xax_y, xax_z],
        [yax_x, yax_y, yax_z],
        [zax_x, zax_y, zax_z],
        ],
        dtype=float)

    return uvec_r


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

    assert np.all(np.isfinite(dists))

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
