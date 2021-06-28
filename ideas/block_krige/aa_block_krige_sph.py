'''
@author: Faizan-Uni-Stuttgart

Jun 28, 2021

9:19:46 AM

'''
import os
import sys
import time
import timeit
from math import exp
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def exp_vg(dist):

    vrange = 50000.
    vsill = 1.0

    return (vsill * (1 - exp(-3 * dist / vrange)))


def sph_vg(dist):

#     return exp_vg(dist)

    vrange = 50000.
    vsill = 1.0

    if dist < vrange:
        a = (1.5 * dist) / vrange
        b = dist ** 3 / (2 * vrange ** 3)
        sph_vg = (vsill * (a - b))

    else:
        sph_vg = vsill

    return sph_vg


def cmpt_block_vg(crds_concate, ref_x, ref_y, cell_size, n_disc):

    if n_disc == 1:
        cell_diffs = np.array([0])

    else:
        cell_diffs = np.linspace(-cell_size * 0.5, cell_size * 0.5, n_disc)

    print(cell_diffs)

    print('\n\n')

    crds = []

    n_pts = crds_concate.shape[1]

    n_cell_vg_vals = n_disc ** 2

    pt_vg_vals = np.empty(n_pts, dtype=float)
    blk_vg_vals = np.empty(n_pts, dtype=float)

    for i in range(n_pts):
        x_crd_cen = crds_concate[0, i]
        y_crd_cen = crds_concate[1, i]

#         dist = (((ref_x - x_crd_cen) ** 2) + ((ref_y - y_crd_cen) ** 2)) ** 0.5
#
#         pt_vg_vals[i] = sph_vg(dist)

        mean_dist = 0
        blk_vg_val = 0
        for j in range(n_disc):
            x_crd = x_crd_cen + cell_diffs[j]
            for k in range(n_disc):
                y_crd = y_crd_cen + cell_diffs[k]

#                 print(x_crd_cen, y_crd_cen, x_crd, y_crd)

                crds.append((x_crd, y_crd))

                dist = (((ref_x - x_crd) ** 2) + ((ref_y - y_crd) ** 2)) ** 0.5

                blk_vg_val += sph_vg(dist)

                mean_dist += dist

        mean_dist /= n_cell_vg_vals

        pt_vg_vals[i] = sph_vg(mean_dist)

        blk_vg_vals[i] = blk_vg_val / n_cell_vg_vals

    pt_vg_vals = np.round(pt_vg_vals, 5)
    blk_vg_vals = np.round(blk_vg_vals, 5)

    return np.array(crds), pt_vg_vals, blk_vg_vals


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    crds_file = Path(
        r'P:\Synchronize\IWS\Testings\variograms\ppt_monthly_1971_2010_crds.csv')

    crds_df = pd.read_csv(crds_file, sep=';', index_col=0)

#     x_crds = np.array([100., 200., 300.])
#     y_crds = np.array([400., 500., 600.])

#     mesh_x, mesh_y = np.meshgrid(x_crds, y_crds)

#     print(mesh_x)
#     print(mesh_y)

#     crds_concate = np.concatenate(
#         (mesh_x.ravel().reshape(1, -1), mesh_y.ravel().reshape(1, -1)),
#         axis=0)

    x_crds = crds_df['X'].values
    y_crds = crds_df['Y'].values

    crds_concate = np.concatenate(
        (x_crds.reshape(1, -1), y_crds.reshape(1, -1)),
        axis=0)

    cell_size = 10000.0

    n_disc = 20

#     print(crds_concate)

    crds, pt_vg_vals, blk_vg_vals = cmpt_block_vg(
        crds_concate, x_crds.mean(), y_crds.mean(), cell_size, n_disc)

    for i in range(pt_vg_vals.size):
        print(pt_vg_vals[i], blk_vg_vals[i])

    mean_err = (
        ((pt_vg_vals - blk_vg_vals) ** 2).sum() ** 0.5) / blk_vg_vals.size

#     mean_err = (
#         ((pt_vg_vals - blk_vg_vals)).sum()) / blk_vg_vals.size

    print(mean_err)

#     plt.scatter(crds[:, 0], crds[:, 1], s=40, alpha=0.7)
#     plt.scatter(crds_concate[0,:], crds_concate[1,:], s=15)
#
#     plt.show()

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
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
