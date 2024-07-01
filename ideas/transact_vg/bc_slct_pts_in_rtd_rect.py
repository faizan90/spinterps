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

# np.seterr(all='raise')

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    path_crds_df = Path(
        r'U:\dwd_meteo\daily\crds\daily_de\daily_fm_epsg32632.csv')

    ang = -80
    strip_width = 50 * 1e3
    #==========================================================================

    assert -90 <= ang <= +90

    strip_width_sclr = 1.0

    if abs(ang) > 45:
        ax_rot = 0
        ang -= np.sign(ang) * 90

    else:
        ax_rot = 1

    ang = (ang) * np.pi / 180

    crds_df = pd.read_csv(path_crds_df, sep=';')[['X', 'Y']]

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

        if True:
            cntn_flags = tst_pts_cntn(poly_r, crds_df.values)

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

        else:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for i in range(poly_r.shape[0]):
                ring.AddPoint(poly_r[i, 0], poly_r[i, 1])

            poly_rv = ogr.Geometry(ogr.wkbPolygon)
            poly_rv.AddGeometry(ring)

            pts_poly_in = []
            pts_poly_ot = []
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

        print('Took:', round(t_end - t_beg, 6), ctr)

        plt.plot(poly_r[:, 0], poly_r[:, 1], c=f'C{j}', alpha=0.2)

        plt.scatter(
            crds_df.loc[pts_poly_in, 'X'].values,
            crds_df.loc[pts_poly_in, 'Y'].values,
            c=f'C{j}',
            alpha=0.5,
            edgecolors='none')

    print('Missed pts:', (~all_stns_flags).sum())

    plt.gca().set_aspect('equal')

    plt.grid()
    plt.show()
    return


def tst_pts_cntn(poly, pts):

    x_min = poly[:, 0].min()
    y_min = poly[:, 1].min()

    x_max = poly[:, 0].max()
    y_max = poly[:, 1].max()

    n_pts = pts.shape[0]
    n_poly = poly.shape[0]

    cntn_flags = np.zeros(n_pts, dtype=bool)

    for k in range(n_pts):

        pt_x = pts[k, 0]
        pt_y = pts[k, 1]

        if pt_x < x_min:
            continue

        if pt_x > x_max:
            continue

        if pt_y < y_min:
            continue

        if pt_y > y_max:
            continue

    # Taken from https://web.archive.org/web/20110314030147/http:
    # //paulbourke.net/geometry/insidepoly/
    # The following code is by Randolph Franklin, it returns 1 for interior
    # points and 0 for exterior points.
    #
    # int pnpoly(int npol, float *xp, float *yp, float x, float y)
    # {
    #   int i, j, c = 0;
    #   for (i = 0, j = npol-1; i < npol; j = i++) {
    #     if ((((yp[i] <= y) && (y < yp[j])) ||
    #          ((yp[j] <= y) && (y < yp[i]))) &&
    #         (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]))
    #       c = !c;
    #   }
    #   return c;
    # }

        cntn_flag = False

        i = 0
        j = n_poly - 1

        while i < n_poly:

            c30 = (poly[j, 1] - poly[i, 1]) != 0

            if c30:
                c31 = pt_x < (
                    ((poly[j, 0] - poly[i, 0]) * (pt_y - poly[i, 1]) /
                     (poly[j, 1] - poly[i, 1])) + poly[i, 0])

                if c31:
                    c11 = poly[i, 1] <= pt_y
                    c12 = pt_y < poly[j, 1]

                    c21 = poly[j, 1] <= pt_y
                    c22 = pt_y < poly[i, 1]

                    if ((c11 and c12) or (c21 and c22)) and c31:
                        cntn_flag = not cntn_flag

            i += 1
            j = i - 1

        cntn_flags[k] = cntn_flag

    return cntn_flags


def rotate_poly(poly_ref, ang):

    xr0, yr0 = rotate_pt_2d(poly_ref[0, 0], poly_ref[0, 1], ang)
    xr1, yr1 = rotate_pt_2d(poly_ref[1, 0], poly_ref[1, 1], ang)
    xr2, yr2 = rotate_pt_2d(poly_ref[2, 0], poly_ref[2, 1], ang)
    xr3, yr3 = rotate_pt_2d(poly_ref[3, 0], poly_ref[3, 1], ang)
    xr4, yr4 = rotate_pt_2d(poly_ref[4, 0], poly_ref[4, 1], ang)
    xr5, yr5 = rotate_pt_2d(poly_ref[5, 0], poly_ref[5, 1], ang)

    poly_r = np.array([
        [xr0, yr0],
        [xr1, yr1],
        [xr2, yr2],
        [xr3, yr3],
        [xr4, yr4],
        [xr5, yr5],
        [xr0, yr0],
        ])

    return poly_r


def rotate_pt_2d(dx, dy, ang):

    xr = (dx * np.cos(ang)) - (dy * np.sin(ang))
    yr = (dx * np.sin(ang)) + (dy * np.cos(ang))

    return xr, yr


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
