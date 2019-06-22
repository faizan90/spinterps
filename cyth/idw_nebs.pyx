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
        DT_D fabs(DT_D x)
        DT_D M_PI
        DT_D pow(DT_D x, DT_D y)


cdef extern from 'misc.h' nogil:
    cdef:
        DT_D get_dist(
            const DT_D x1, 
            const DT_D y1, 
            const DT_D x2, 
            const DT_D y2)

        DT_D get_sum(const DT_D *x, const DT_L len_x)

        void fill_idw_wts_arr(
            const DT_D *x_arr,
                  DT_D *wts_arr,
            const DT_D idw_exp,
            const DT_L len_x)

        void fill_mult_arr(
            const DT_D *x,
            const DT_D *y,
                  DT_D *mult_arr,
            const DT_L len_x)


cpdef DT_D get_idw(
        const DT_D idw_x,
        const DT_D idw_y,
        const DT_D[:] xs,
        const DT_D[:] ys,
        const DT_D[:] zs,
        const DT_D idw_exp,
              DT_D[:] idw_wts,
              DT_D[:] dists,
              DT_D[:] mult_arr):

    """
    Get IDW value at a point given distances of other points 
    from it with values and the exponent of the IDW.
    """

    cdef:
        Py_ssize_t i

        DT_L len_x = xs.shape[0]

        DT_D idw_val, the_sum, the_wts

    for i in range(len_x):
        dists[i] = get_dist(idw_x, idw_y, xs[i], ys[i])

    fill_idw_wts_arr(&dists[0], &idw_wts[0], idw_exp, len_x)

    fill_mult_arr(&idw_wts[0], &zs[0], &mult_arr[0], len_x)

    the_sum = get_sum(&mult_arr[0], len_x)

    the_wts = get_sum(&idw_wts[0], len_x)

    idw_val = the_sum / the_wts

    return idw_val


cpdef np.ndarray get_idw_arr(
        DT_D[:] idw_x_arr,
        DT_D[:] idw_y_arr,
        DT_D[:] xs,
        DT_D[:] ys,
        DT_D[:] zs,
        DT_D idw_exp):

    cdef:
        Py_ssize_t i

        DT_L len_x = xs.shape[0], len_idw_z = idw_x_arr.shape[0]

        DT_D[:] idw_wts = np.zeros(shape=len_x, dtype=np.float64)
        DT_D[:] dists = np.zeros(shape=len_x, dtype=np.float64)
        DT_D[:] mult_arr = np.zeros(shape=len_x, dtype=np.float64)
        DT_D[:] idw_z_arr = np.zeros(shape=len_idw_z, dtype=np.float64)

    for i in range(len_idw_z):
        idw_z_arr[i] = get_idw(
            idw_x_arr[i],
            idw_y_arr[i],
            xs,
            ys,
            zs,
            idw_exp,
            idw_wts,
            dists,
            mult_arr)

    return np.asarray(idw_z_arr)


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

    for i in range(n_refs):
        dists[i] = get_dist(
            dst_x,
            dst_y,
            ref_xs[i],
            ref_ys[i])

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

# cpdef void slct_nebrs_cy_old(
#         const DT_D x,
#         const DT_D y,
#         const DT_D[:] nebrs_x_crds_arr,
#         const DT_D[:] nebrs_y_crds_arr,
#         const DT_UL n_quads,
#         const DT_UL n_per_quad,
#         const DT_D min_dist_thresh,
#         const DT_UL n_nebs,
#               DT_UL[:] prcssed_nebrs_arr,
#               DT_UL[:] slctd_nebrs_arr,
#               long long[:] nebs_idxs_arr,
#               DT_D[:] dists_arr,
#               DT_D[:] slctd_nebrs_dists_arr,
#               DT_UL[:] idxs_fin_arr):
# 
#     cdef:
#         DT_UL i, j, min_dist_idx, nebs_avail_cond, n_prcssed_nebrs = 0
#         DT_D min_dist, curr_min_ang, curr_max_ang
#         DT_D quad_ang_incr, x_dist, y_dist, nebr_ang
# 
#     min_dist = INF
#     quad_ang_incr = 1. / n_quads
# 
#     for i in range(n_nebs):
#         dists_arr[i] = 0
#         prcssed_nebrs_arr[i] = 0
#         slctd_nebrs_arr[i] = 0
#         slctd_nebrs_dists_arr[i] = 0
#         nebs_idxs_arr[i] = 0
#         idxs_fin_arr[i] = 0
# 
#     for i in range(n_nebs):
#         dists_arr[i] = get_dist(
#             x, 
#             y, 
#             nebrs_x_crds_arr[i],
#             nebrs_y_crds_arr[i])
# 
#         if (dists_arr[i] <= min_dist_thresh) and (dists_arr[i] < min_dist):
#             min_dist_idx = i
#             min_dist = dists_arr[i]
# 
#     if min_dist < INF:
#         idxs_fin_arr[min_dist_idx] = 1
#         return
# 
#     for i in range(n_quads):
#         if n_prcssed_nebrs == n_nebs:
#             break
# 
#         curr_min_ang = quad_ang_incr * i * 2 * M_PI
#         curr_max_ang = quad_ang_incr * (i + 1) * 2 * M_PI
# 
#         for j in range(n_nebs):
#             slctd_nebrs_arr[j] = 0
#             slctd_nebrs_dists_arr[j] = INF
# 
#         nebs_avail_cond = 0
#         for j in range(n_nebs):
#             if prcssed_nebrs_arr[j]:
#                 continue
# 
#             x_dist = nebrs_x_crds_arr[j] - x
#             y_dist = nebrs_y_crds_arr[j] - y
# 
#             if not x_dist:
#                 nebr_ang = 0.0
# 
#             else:
#                 nebr_ang = atan(y_dist / x_dist)
# 
#             if (x_dist < 0) and (y_dist > 0):
#                 nebr_ang = M_PI + nebr_ang
# 
#             elif (x_dist < 0) and (y_dist < 0):
#                 nebr_ang = M_PI + nebr_ang
# 
#             elif (x_dist > 0) and (y_dist < 0):
#                 nebr_ang = (2.0 * M_PI) + nebr_ang
# 
#             if (nebr_ang >= curr_min_ang) and (nebr_ang < curr_max_ang):
#                 slctd_nebrs_arr[j] = 1
#                 prcssed_nebrs_arr[j] = 1
#                 slctd_nebrs_dists_arr[j] = dists_arr[j]
# 
#                 n_prcssed_nebrs += 1
# 
#         for j in range(n_nebs):
#             if slctd_nebrs_arr[j]:
#                 nebs_avail_cond = 1
#                 break
# 
#         if nebs_avail_cond:
#             nebs_idxs_arr = np.argsort(slctd_nebrs_dists_arr)
# 
#             for j in range(n_per_quad):
# 
#                 idxs_fin_arr[nebs_idxs_arr[j]] = 1
#     return
