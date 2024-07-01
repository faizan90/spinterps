# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jan 13, 2024

11:25:02 AM

Keywords:
'''

import numpy as np


class OptArgs:

    def __init__(self):
        return


def get_smthd_ar(in_arr, win_size, smooth_ftn_type):

    n_vals = in_arr.shape[0]

    smooth_ftn = getattr(np, smooth_ftn_type)

    smoothed_arr = np.zeros(n_vals - win_size + 1)
    for i in range(smoothed_arr.size):
        smoothed_arr[i] = smooth_ftn(in_arr[i:i + win_size])

    return smoothed_arr


def get_vg_cloud2(
        tss_df,
        ref_col,
        ref_x,
        ref_y,
        crds_df,
        min_vld_vals,
        lb_x,
        lb_y,
        le_x,
        le_y,
        dist_lim,
        ignr_cols,
        n_nrst_nebs,
        grp):

    dists = []
    for col in tss_df.columns:

        dist = (
            ((ref_x - crds_df.loc[col, 'X']) ** 2) +
            ((ref_y - crds_df.loc[col, 'Y']) ** 2)) ** 0.5

        dists.append(dist)

    dists = np.array(dists)

    ref_z = crds_df.loc[ref_col, 'Z_SRTM']

    ref_fnt_idxs = np.isfinite(tss_df.loc[:, ref_col].values)

    n_ref_fnt = ref_fnt_idxs.sum()

    vg_cloud = []
    acpt_nebs = 0
    for col in tss_df.columns[np.argsort(dists)]:

        if col == ref_col: continue

        if col in ignr_cols: continue

        dist = (
            ((ref_x - crds_df.loc[col, 'X']) ** 2) +
            ((ref_y - crds_df.loc[col, 'Y']) ** 2)) ** 0.5

        if dist == 0: continue

        if dist > dist_lim: continue

        if (acpt_nebs == 0) and (n_ref_fnt < min_vld_vals): break

        fnt_idxs = np.isfinite(tss_df.loc[:, col].values) & ref_fnt_idxs

        if fnt_idxs.sum() < min_vld_vals: continue

        dist_sgn = get_side(
            lb_x,
            lb_y,
            le_x,
            le_y,
            crds_df.loc[col, 'X'],
            crds_df.loc[col, 'Y'])

        if dist_sgn == -1:
            dist *= -1

        vg_tss = (tss_df[ref_col].values - tss_df[col].values)[fnt_idxs]

        if False:
            vg_val = np.mean(vg_tss) * (ref_z - crds_df.loc[col, 'Z_SRTM'])

        else:
            vg_val = np.mean(vg_tss ** 2)

        vg_val *= 0.5

        if vg_val == 0: continue

        vg_cloud.append([dist, vg_val, grp])

        acpt_nebs += 1

        if acpt_nebs == n_nrst_nebs:
            break

    vg_cloud = np.array(vg_cloud)

    return vg_cloud


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


def get_side(lb_x, lb_y, le_x, le_y, pt_x, pt_y):

    # From: https://stackoverflow.com/questions/1560492/
    # how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
    #
    # a and b are starting and ending points of the line.
    # c is the point of interest.
    #
    # public bool isLeft(Point a, Point b, Point c) {
    #   return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x) > 0;
    # }

    cp = (le_x - lb_x) * (pt_y - lb_y) - (le_y - lb_y) * (pt_x - lb_x)

    if cp > 0:
        sgn = +1

    elif cp < 0:
        sgn = -1

    elif cp == 0:
        sgn = +0

    else:
        raise ValueError(cp)

    return sgn


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
