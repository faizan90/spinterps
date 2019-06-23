# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

ctypedef double DT_D
ctypedef long DT_L
ctypedef unsigned long DT_UL

cdef DT_D INF = np.inf


cdef extern from 'math.h' nogil:
    cdef:
        DT_D atan(DT_D x)
        DT_D M_PI


cpdef void fill_dists_one_pt(
        const DT_D x, 
        const DT_D y, 
        const DT_D[::1] xs, 
        const DT_D[::1] ys,
              DT_D[::1] dists):

    cdef:
        Py_ssize_t i

    for i in range(xs.size):
        dists[i] = (((x - xs[i])**2) + ((y - ys[i])**2))**0.5

    return


cpdef DT_D fill_wts_and_sum(
        const DT_D[::1] dists, DT_D[::1] wts, const DT_D idw_exp):

    cdef:
        Py_ssize_t i
        DT_D wts_sum = 0.0

    for i in range(dists.size):
        wts[i] = 1.0 / ((dists[i] ** idw_exp))
        wts_sum += wts[i]

    return wts_sum


cpdef DT_D get_mults_sum(
        const DT_D[::1] wts, const DT_D[::1] data):

    cdef:
        Py_ssize_t i
        DT_D mults_sum = 0.0

    for i in range(wts.size):
        mults_sum += wts[i] * data[i]

    return mults_sum


cpdef void sel_equidist_refs(
        const DT_D dst_x,
        const DT_D dst_y,
        const DT_D[::1] ref_xs,
        const DT_D[::1] ref_ys,
        const DT_UL n_pies,
        const DT_D min_dist_thresh,
        const long long not_neb_flag,
              DT_D[::1] dists,
              DT_D[::1] tem_ref_sel_dists,
              long long[::1] ref_sel_pie_idxs,
              DT_UL[::1] ref_pie_idxs,
              DT_UL[::1] ref_pie_cts):

    cdef:
        DT_UL i, j, min_dist_idx, refs_in_pie_flag
        DT_UL n_refs = ref_xs.size, ref_pie_idx
        DT_D min_dist = INF, two_pi = 2 * M_PI
        DT_D x_dist, y_dist, ref_dst_ang

        long long[::1] srtd_tem_ref_sel_dist_idxs

    for i in range(n_refs):
        ref_sel_pie_idxs[i] = not_neb_flag

    fill_dists_one_pt(dst_x, dst_y, ref_xs, ref_ys, dists)
    for i in range(n_refs):
        if (dists[i] <= min_dist_thresh) and (dists[i] < min_dist):
            min_dist_idx = i
            min_dist = dists[i]

    if min_dist < INF:
        ref_sel_pie_idxs[min_dist_idx] = 0

    else:
        for j in range(n_pies):
            ref_pie_cts[j] = 0

        for j in range(n_refs):
            x_dist = ref_xs[j] - dst_x
            y_dist = ref_ys[j] - dst_y

            if not x_dist:
                ref_dst_ang = 0.0

            else:
                ref_dst_ang = atan(y_dist / x_dist)

                if (x_dist < 0) and (y_dist > 0):
                    ref_dst_ang = M_PI + ref_dst_ang
    
                elif (x_dist < 0) and (y_dist < 0):
                    ref_dst_ang = M_PI + ref_dst_ang
    
                elif (x_dist > 0) and (y_dist < 0):
                    ref_dst_ang = two_pi + ref_dst_ang

            ref_pie_idx = int(ref_dst_ang * n_pies / two_pi)
            ref_pie_idxs[j] = ref_pie_idx
            ref_pie_cts[ref_pie_idx] += 1

        for j in range(n_pies):
            if not ref_pie_cts[j]:
                continue

            for i in range(n_refs):
                if ref_pie_idxs[i] == j:
                    tem_ref_sel_dists[i] = dists[i]

                else:
                    tem_ref_sel_dists[i] = INF

            srtd_tem_ref_sel_dist_idxs = np.argsort(tem_ref_sel_dists)

            for i in range(n_refs):
                if tem_ref_sel_dists[srtd_tem_ref_sel_dist_idxs[i]] == INF:
                    break

                ref_sel_pie_idxs[srtd_tem_ref_sel_dist_idxs[i]] = i
    return

