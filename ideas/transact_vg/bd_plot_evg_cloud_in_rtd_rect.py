# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jan 12, 2024

1:40:50 PM

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
from osgeo import ogr
import matplotlib.pyplot as plt
from scipy.stats import boxcox

from za_cmn_ftns import get_vg_cloud2, rotate_poly, tst_pts_cntn

# np.seterr(all='raise')

DEBUG_FLAG = True


def main():

    main_dir = Path(r'P:\Synchronize\Testings\geostat\transact_vg')
    os.chdir(main_dir)

    path_crds_df = Path(
        r'U:\dwd_meteo\daily\crds\daily_bayern_50km_buff\daily_ppt_epsg32632.csv')

    path_tss_df = Path(
        r'U:\dwd_meteo\daily\dfs__merged_subset\daily_bayern_50km_buff_ppt_Y1961_2022.pkl')

    ang = 29
    strip_width = 15 * 1e3

    min_vld_vals = 365 * 5

    dist_lim = 50 * 1e3

    pvt_sclr = 0.59

    n_nrst_nebs = 5

    ot_dir = Path(rf'vgs_A{ang}_SW{int(strip_width)}_ppt_by_dwd')
    #==========================================================================

    ot_dir.mkdir(exist_ok=True)

    assert -90 <= ang <= +90

    strip_width_sclr = 1.0

    if abs(ang) > 45:
        ax_rot = 0
        ang -= np.sign(ang) * 90

    else:
        ax_rot = 1

    ang = (ang) * np.pi / 180

    crds_df = pd.read_csv(path_crds_df, sep=';', index_col=0)[['X', 'Y']]
    crds_df.index = crds_df.index.astype(str)

    tss_df = pd.read_pickle(path_tss_df)

    cmn_cols = crds_df.index.intersection(tss_df.columns)
    assert cmn_cols.size

    crds_df = crds_df.loc[cmn_cols]
    tss_df = tss_df.loc[:, cmn_cols]

    x_mn = crds_df['X'].min()
    x_mx = crds_df['X'].max()

    y_mn = crds_df['Y'].min()
    y_mx = crds_df['Y'].max()

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

        t_beg = timeit.default_timer()

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

        ctr = 0
        pts_poly_in = []
        pts_poly_ot = []

        if True:
            cntn_flags = tst_pts_cntn(poly_r, crds_df.values)

            for i in range(crds_df.shape[0]):

                if cntn_flags[i]:
                    pts_poly_in.append(crds_df.index[i])

                    if strip_width_sclr >= 1:
                        assert all_stns_flags[i] == 0

                    all_stns_flags[i] = 1

                    ctr += 1

                else:
                    pts_poly_ot.append(crds_df.index[i])

        else:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for i in range(poly_r.shape[0]):
                ring.AddPoint(poly_r[i, 0], poly_r[i, 1])

            poly_rv = ogr.Geometry(ogr.wkbPolygon)
            poly_rv.AddGeometry(ring)

            for i in range(crds_df.shape[0]):

                pt = ogr.CreateGeometryFromWkt(
                    f'POINT ({crds_df.iloc[i, 0]} {crds_df.iloc[i, 1]})')

                if poly_rv.Contains(pt):
                    pts_poly_in.append(crds_df.index[i])

                    if strip_width_sclr >= 1:
                        assert all_stns_flags[i] == 0

                    all_stns_flags[i] = 1

                    ctr += 1

                else:
                    pts_poly_ot.append(crds_df.index[i])

        t_end = timeit.default_timer()

        print('Took:', round(t_end - t_beg, 6), ctr, j)

        plt.plot(poly_r[:, 0], poly_r[:, 1], c=f'C{j}', alpha=0.2)

        plt.scatter(
            crds_df.loc[pts_poly_in, 'X'].values,
            crds_df.loc[pts_poly_in, 'Y'].values,
            c=f'C{j}',
            alpha=0.5,
            edgecolors='none')

        strip_pts_grps[j] = pts_poly_in

    print('Missed pts:', (~all_stns_flags).sum())

    plt.gca().set_aspect('equal')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.savefig(ot_dir / f'grps.png', bbox_inches='tight')
    # plt.show()

    plt.close()
    #==========================================================================

    # if ax_rot == 1:
    #     poly_r = poly_rot_x.copy()
    #
    # elif ax_rot == 0:
    #     poly_r = poly_rot_y.copy()
    #
    # else:
    #     raise NotImplementedError(ax_rot)

    plt.figure()
    for j in strip_pts_grps:

        t_beg = timeit.default_timer()

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

        # poly_r[:, 0] += lb_x
        # poly_r[:, 1] += lb_y

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

        plt_j_flag = False
        for ref_col in strip_pts_grps[j]:

            ref_x = crds_df.loc[ref_col, 'X']
            ref_y = crds_df.loc[ref_col, 'Y']

            vg_cloud_j_ref = get_vg_cloud2(
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

            if not vg_cloud_j_ref.size: continue

            plt.scatter(
                vg_cloud_j_ref[:, 0],
                vg_cloud_j_ref[:, 1],
                alpha=0.5,
                edgecolors='none',
                c=f'C{j}')

            plt_j_flag = True

        if plt_j_flag:
            plt.grid()
            plt.gca().set_axisbelow(True)

            plt.xlabel('Distance')
            plt.ylabel('SVG')

            plt.savefig(ot_dir / f'vg_cloud_{j:02d}.png', bbox_inches='tight')

        plt.clf()

        t_end = timeit.default_timer()

        print('Took:', round(t_end - t_beg, 6), j)

    plt.close()
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
