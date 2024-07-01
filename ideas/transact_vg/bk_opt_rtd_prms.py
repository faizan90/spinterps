# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jan 13, 2024

11:24:21 AM

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
from scipy.stats import boxcox
import matplotlib.pyplot as plt

from fdiffevo import fde_host

from za_cmn_ftns import (
    get_vg_cloud2, rotate_poly, tst_pts_cntn, get_smthd_ar, OptArgs)

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\Testings\geostat\transact_vg')
    os.chdir(main_dir)

    path_crds_df = Path(
        r'U:\dwd_meteo\daily\crds\daily_bayern_50km_buff\daily_ppt_epsg32632.csv')

    path_tss_df = Path(
        r'U:\dwd_meteo\daily\dfs__merged_subset\daily_bayern_50km_buff_ppt_Y1961_2022.pkl')

    strip_width = 10 * 1e3

    min_vld_vals = int(365 * 5 / 12)

    dist_lim = 20 * 1e3

    n_nrst_nebs = 5

    n_cpus = 16

    pop_size = 20

    max_iters = 1

    ot_dir = Path(rf'opt_vgs_ppt_by_dwd4')
    #==========================================================================

    ot_dir.mkdir(exist_ok=True)

    crds_df = pd.read_csv(path_crds_df, sep=';', index_col=0)[['X', 'Y', 'Z_SRTM']]
    crds_df.index = crds_df.index.astype(str)

    tss_df = pd.read_pickle(path_tss_df)

    # tss_df = tss_df.loc['1990-01-01':'1990-01-31',:].sum(axis=0)
    # tss_df = pd.DataFrame(tss_df).T

    tss_df = tss_df.loc[tss_df.index.month == 1]

    ignr_cols = []
    for col in tss_df.columns:

        if tss_df[col].count() >= min_vld_vals:
            continue

        ignr_cols.append(col)

    tss_df.drop(columns=ignr_cols, inplace=True)

    cmn_cols = crds_df.index.intersection(tss_df.columns)
    assert cmn_cols.size

    crds_df = crds_df.loc[cmn_cols]
    tss_df = tss_df.loc[:, cmn_cols]

    # tss_df = tss_df.rank(axis=0)
    # crds_df['Z_SRTM'][:] = crds_df['Z_SRTM'].rank()

    x_mn = crds_df['X'].min()
    x_mx = crds_df['X'].max()

    y_mn = crds_df['Y'].min()
    y_mx = crds_df['Y'].max()

    crds_df['X'] -= (x_mn + x_mx) * 0.5
    crds_df['Y'] -= (y_mn + y_mx) * 0.5

    x_mn = crds_df['X'].min()
    x_mx = crds_df['X'].max()

    y_mn = crds_df['Y'].min()
    y_mx = crds_df['Y'].max()

    opt_ctr = [0]

    esvg_args = OptArgs()

    esvg_args.x_mn = x_mn
    esvg_args.x_mx = x_mx
    esvg_args.y_mn = y_mn
    esvg_args.y_mx = y_mx
    esvg_args.strip_width = strip_width
    esvg_args.crds_df = crds_df
    esvg_args.tss_df = tss_df
    esvg_args.min_vld_vals = min_vld_vals
    esvg_args.dist_lim = dist_lim
    esvg_args.opt_ctr = opt_ctr
    esvg_args.n_nrst_nebs = n_nrst_nebs

    # esvg_args = (
    #     x_mn,
    #     x_mx,
    #     y_mn,
    #     y_mx,
    #     strip_width,
    #     crds_df,
    #     tss_df,
    #     min_vld_vals,
    #     dist_lim,
    #     opt_ctr,
    #     n_nrst_nebs)

    # fde_host signature.
    # best_prms, best_obj_val, state_snapshots, ret_ftn_args = fde_host(
    #     obj_ftn,
    #     obj_ftn_args,
    #     prm_bds,
    #     n_prm_vecs,
    #     n_cpus,
    #     mu_sc_fac_bds,
    #     cr_cnst_bds,
    #     max_iters,
    #     max_cont_iters,
    #     prm_pcnt_tol,
    #     new_obj_acpt_ratio,
    #     obj_ftn_tol,
    #     min_thresh_obj_val,
    #     save_snapshots_flag,
    #     ret_opt_args_flag,
    #     alg_cfg_type,
    #     verbose,
    #     bds_updt_iters)

    opt_res = fde_host(
        get_esvg_obj_val,
        esvg_args,
        np.array([[0., 90], [+0.5, +0.5], [1, 1], [1, 1], [1, 1]], dtype=float),
        pop_size,
        n_cpus,
        (0.01, 0.5),
        (0.7, 1.0),
        max_iters,
        20,
        0.02,
        0.001,
        1e-3,
        100000,
        False,
        False,
        1,
        True,
        10)

    best_prms = opt_res.prms_best
    best_obj_val = opt_res.obj_best

    print(best_prms, best_obj_val)

    # import pdb; pdb.set_trace()

    esvg_cloud = get_esvg_cloud(best_prms, esvg_args)

    # opt_res = differential_evolution(
    #     get_esvg_obj_val,
    #     bounds=np.array([[0.0, +90.], [-0.2, +1.2]]),
    #     args=(esvg_args,),
    #     polish=False,
    #     maxiter=max_iters,
    #     updating='deferred',
    #     workers=n_cpus,
    #     popsize=pop_size)
    #
    # print(opt_res.x, opt_res.fun)
    # esvg_cloud = get_esvg_cloud(opt_res.x, esvg_args)

    plt.figure()
    for grp in np.unique(esvg_cloud[:, 2]).astype(np.int64):

        print('grp:', grp)

        grp_idxs = esvg_cloud[:, 2] == grp

        plt.scatter(
            esvg_cloud[grp_idxs, 0],
            esvg_cloud[grp_idxs, 1],
            alpha=0.5,
            edgecolors='none',
            c=f'C{grp}')

        plt.gca().set_yscale('symlog')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Distance')
        plt.ylabel('SVG')

        plt.savefig(ot_dir / f'vg_cloud_{grp:02d}.png', bbox_inches='tight')

        plt.clf()

    plt.scatter(
        esvg_cloud[:, 0],
        esvg_cloud[:, 1],
        alpha=0.1,
        edgecolors='none',
        c='k')

    plt.gca().set_yscale('symlog')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Distance')
    plt.ylabel('SVG')

    plt.savefig(ot_dir / f'vg_cloud_all.png', bbox_inches='tight')

    plt.clf()

    plt.close()
    return


def get_esvg_obj_val(prms, args):

    esvg_cloud = get_esvg_cloud(prms, args)

    assert esvg_cloud.size

    opt_ctr = args.opt_ctr

    idxs_pve_all = esvg_cloud[:, 0] >= 0
    idxs_nve_all = ~idxs_pve_all

    ws = 10
    smth_meth = 'mean'
    #==========================================================================

    obj_val = 0.0
    for grp in np.unique(esvg_cloud[:, 2]).astype(np.int64):

        grp_idxs = esvg_cloud[:, 2] == grp
        #======================================================================

        idxs_pve = grp_idxs & idxs_pve_all
        if idxs_pve.sum() > ws:
            esvg_cloud_pve = esvg_cloud[idxs_pve,:]
            esvg_cloud_pve = esvg_cloud_pve[np.argsort(esvg_cloud_pve[:, 0]),:]

            dists_pve = get_smthd_ar(esvg_cloud_pve[:, 0], ws, smth_meth)
            esvgs_pve = get_smthd_ar(esvg_cloud_pve[:, 1], ws, smth_meth)

            wts_pve = (1.0 / (dists_pve[:-1] / 1e3)) ** 0.5

            obj_val_pve = ((
                (esvgs_pve[:-1] - esvgs_pve[1:]) > 0) *
                wts_pve).sum(dtype=np.float64) / idxs_pve.sum()

        else:
            obj_val_pve = 0.0
        #======================================================================

        idxs_nve = grp_idxs & idxs_nve_all
        if idxs_nve.sum() > ws:
            esvg_cloud_nve = esvg_cloud[idxs_nve,:]
            esvg_cloud_nve[:, 0] = np.abs(esvg_cloud_nve[:, 0])
            esvg_cloud_nve = esvg_cloud_nve[np.argsort(esvg_cloud_nve[:, 0]),:]

            dists_nve = get_smthd_ar(esvg_cloud_nve[:, 0], ws, smth_meth)
            esvgs_nve = get_smthd_ar(esvg_cloud_nve[:, 1], ws, smth_meth)

            wts_nve = (1.0 / (dists_nve[:-1] / 1e3)) ** 0.5

            obj_val_nve = ((
                (esvgs_nve[:-1] - esvgs_nve[1:]) > 0) *
                wts_nve).sum(dtype=np.float64) / idxs_nve.sum()

        else:
            obj_val_nve = 0.0
        #======================================================================

        obj_val += obj_val_pve + obj_val_nve

    print(f'{opt_ctr[0]}, obj_val: {obj_val:0.4f}, {esvg_cloud.shape[0]}\n')
    return obj_val


def get_esvg_cloud(prms, args):

    ang, pvt_sclr, x_sclr, y_sclr, z_sclr = prms

    # (x_mn,
    #  x_mx,
    #  y_mn,
    #  y_mx,
    #  strip_width,
    #  crds_df,
    #  tss_df,
    #  min_vld_vals,
    #  dist_lim,
    #  opt_ctr,
    #  n_nrst_nebs) = args

    x_mn = args.x_mn * x_sclr
    x_mx = args.x_mx * x_sclr
    y_mn = args.y_mn * y_sclr
    y_mx = args.y_mx * y_sclr
    strip_width = args.strip_width
    crds_df = args.crds_df.copy()
    tss_df = args.tss_df
    min_vld_vals = args.min_vld_vals
    dist_lim = args.dist_lim
    opt_ctr = args.opt_ctr
    n_nrst_nebs = args.n_nrst_nebs

    crds_df['X'] *= x_sclr
    crds_df['Y'] *= y_sclr
    crds_df['Z_SRTM'] *= z_sclr

    strip_width *= ((x_sclr ** 2) + (y_sclr ** 2)) ** 0.5

    strip_width /= 2 ** 0.5

    opt_ctr[0] += 1

    t_beg = timeit.default_timer()

    assert -90 <= ang <= +90

    strip_width_sclr = 1.0

    if abs(ang) > 45:
        ax_rot = 0
        ang -= np.sign(ang) * 90

    else:
        ax_rot = 1

    ang = (ang) * np.pi / 180

    if ax_rot == 1:

        y0 = (y_mn + y_mx) * 0.50
        y1 = (y_mx - y_mn) * 1.00
        y2 = (y_mx - y_mn) * 1.00

        poly_ref_x = np.array([
            [+strip_width, +0.],
            [+strip_width, +y1],
            [-strip_width, +y1],
            [-strip_width, +0.],
            [-strip_width, -y2],
            [+strip_width, -y2],
            [+strip_width, +0.],
            ])

        poly_rot_x = rotate_poly(poly_ref_x, ang)

        # No overlapping between polygons.
        x0s = np.arange(
            (poly_rot_x[:, 0] + x_mn).min() + poly_rot_x[0, 0],
            (poly_rot_x[:, 0] + x_mx).max() - poly_rot_x[0, 0],
            poly_rot_x[0, 0] * 2.0 * strip_width_sclr * (1 / np.cos(ang) ** 2))

        n_strips = x0s.shape[0]

    elif ax_rot == 0:

        x0 = (x_mn + x_mx) * 0.50
        x1 = (x_mx - x_mn) * 1.00
        x2 = (x_mx - x_mn) * 1.00

        poly_ref_y = np.array([
            [+0., +strip_width],
            [+x1, +strip_width],
            [+x1, -strip_width],
            [+0., -strip_width],
            [-x2, -strip_width],
            [-x2, +strip_width],
            [+0., +strip_width],
            ])

        poly_rot_y = rotate_poly(poly_ref_y, ang)

        # No overlapping between polygons.
        y0s = np.arange(
            (poly_rot_y[:, 1] + y_mn).min() + poly_rot_y[0, 1],
            (poly_rot_y[:, 1] + y_mx).max() - poly_rot_y[0, 1],
            poly_rot_y[0, 1] * 2.0 * strip_width_sclr * (1 / np.cos(ang) ** 2))[::-1]

        n_strips = y0s.shape[0]

    else:
        raise NotImplementedError(ax_rot)

    strip_pts_grps = {}

    all_stns_flags = np.zeros(crds_df.shape[0], dtype=bool)
    for j in range(n_strips):

        # if j != int(n_strips // 2): continue

        if ax_rot == 1:
            poly_r = poly_rot_x.copy()

            poly_r[:, 0] += x0s[j]
            poly_r[:, 1] += y0

        elif ax_rot == 0:
            poly_r = poly_rot_y.copy()

            poly_r[:, 0] += x0
            poly_r[:, 1] += y0s[j]

        else:
            raise NotImplementedError(ax_rot)

        cntn_flags = tst_pts_cntn(poly_r, crds_df.values)

        ctr = 0
        pts_poly_in = []
        pts_poly_ot = []
        for i in range(crds_df.shape[0]):

            if cntn_flags[i]:
                pts_poly_in.append(crds_df.index[i])

                if strip_width_sclr >= 1:
                    assert all_stns_flags[i] == 0

                all_stns_flags[i] = 1

                ctr += 1

            else:
                pts_poly_ot.append(crds_df.index[i])

        strip_pts_grps[j] = pts_poly_in

    print('Missed pts:', (~all_stns_flags).sum())
    #==========================================================================

    esvg_cloud = []
    for j in strip_pts_grps:

        if False:
            if ax_rot == 1:
                poly_r = poly_rot_x.copy()

                poly_r[:, 0] += x0s[j]
                poly_r[:, 1] += y0

            elif ax_rot == 0:
                poly_r = poly_rot_y.copy()

                poly_r[:, 0] += x0
                poly_r[:, 1] += y0s[j]

            else:
                raise NotImplementedError(ax_rot)

            mean_lb_x = poly_r[[1, 2], 0].mean()
            mean_lb_y = poly_r[[1, 2], 1].mean()

            mean_le_x = poly_r[[4, 5], 0].mean()
            mean_le_y = poly_r[[4, 5], 1].mean()

            lb_x = pvt_sclr * (mean_lb_x + mean_le_x)
            lb_y = pvt_sclr * (mean_lb_y + mean_le_y)

            poly_r[:, 0] += lb_x
            poly_r[:, 1] += lb_y

            if ax_rot == 1:
                poly_r = poly_rot_x.copy()

                poly_r[:, 0] += lb_x
                poly_r[:, 1] += lb_y

            elif ax_rot == 0:
                poly_r = poly_rot_y.copy()

                poly_r[:, 0] += lb_x
                poly_r[:, 1] += lb_y

            else:
                raise NotImplementedError(ax_rot)

        tss_df_j = tss_df.loc[:, strip_pts_grps[j]].copy()

        if False:
            for col in tss_df_j.columns:

                fnt_idxs = np.isfinite(tss_df_j[col].values)

                if fnt_idxs.sum() < min_vld_vals:
                    tss_df_j.loc[:, col] = np.nan
                    continue

                pt_min_val = tss_df_j.loc[:, col].min()

                if pt_min_val == 0:
                    pt_min_val = 0.1

                elif pt_min_val < 0:

                    pt_min_val = abs(pt_min_val) + 0.1

                else:
                    pt_min_val = 0

                tss_df_j.loc[fnt_idxs, col] = boxcox(
                    tss_df_j[col].values[fnt_idxs] + pt_min_val)[0]

        ignr_cols = []
        for ref_col in strip_pts_grps[j]:

            ref_x = crds_df.loc[ref_col, 'X']
            ref_y = crds_df.loc[ref_col, 'Y']

            if True:
                if ax_rot == 1:
                    poly_r = poly_rot_x.copy()

                elif ax_rot == 0:
                    poly_r = poly_rot_y.copy()

                else:
                    raise NotImplementedError(ax_rot)

                poly_r[:, 0] += ref_x
                poly_r[:, 1] += ref_y

            esvg_cloud_j_ref = get_vg_cloud2(
                tss_df_j,
                ref_col,
                ref_x,
                ref_y,
                crds_df,
                min_vld_vals,
                poly_r[0, 0],
                poly_r[0, 1],
                poly_r[3, 0],
                poly_r[3, 1],
                dist_lim,
                ignr_cols,
                n_nrst_nebs,
                j)

            ignr_cols.append(ref_col)

            if not esvg_cloud_j_ref.size: continue

            esvg_cloud.append(esvg_cloud_j_ref)

    t_end = timeit.default_timer()

    print('Took:', round(t_end - t_beg, 6))

    if len(esvg_cloud):
        esvg_cloud = np.concatenate(esvg_cloud, axis=0)

    else:
        esvg_cloud = np.atleast_2d(esvg_cloud)

    return esvg_cloud


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
