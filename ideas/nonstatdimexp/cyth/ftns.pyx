# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
import numpy as np
cimport numpy as np

ctypedef double DT_D
ctypedef unsigned long uint32_t

cdef DT_D INF = np.inf

cdef extern from "math.h" nogil:
    cdef:
        DT_D sin(DT_D x)
        DT_D exp(DT_D x)
        DT_D M_PI
        DT_D pow(DT_D x, DT_D y)


cdef inline DT_D rng_vg(DT_D h, DT_D r, DT_D s) nogil:
    return h


cdef inline DT_D nug_vg(DT_D h, DT_D r, DT_D s) nogil:
    return s


cdef inline DT_D sph_vg(DT_D h, DT_D r, DT_D s) nogil:
    cdef:
        DT_D a, b

    if h >= r:
        return s
    else:
        a = (1.5 * h) / r
        b = (h*h*h) / (2 * (r*r*r))
        return (s * (a - b))


cdef inline DT_D exp_vg(DT_D h, DT_D r, DT_D s) nogil:
    return (s * (1 - exp(-3 * h / r)))


cdef inline DT_D lin_vg(DT_D h, DT_D r, DT_D s) nogil:
    if h > r:
        return s
    else:
        return s * (h / r)


cdef inline DT_D gau_vg(DT_D h, DT_D r, DT_D s) nogil:
    return (s * (1 - exp(-3 * (((h*h) / (r*r))))))


cdef inline DT_D pow_vg(DT_D h, DT_D r, DT_D s) nogil:
    return (s * (h**r))


cdef inline DT_D hol_vg(DT_D h, DT_D r, DT_D s) nogil:
    cdef DT_D a
    if h == 0:
        return 0
    else:
        a = (M_PI * h) / r
        return (s * (1 - (sin(a)/a)))


ctypedef DT_D (*f_type)(DT_D h, DT_D r, DT_D s) nogil
cdef map[string, f_type] all_vg_ftns
all_vg_ftns[b'Rng'] = rng_vg
all_vg_ftns[b'Nug'] = nug_vg
all_vg_ftns[b'Sph'] = sph_vg
all_vg_ftns[b'Exp'] = exp_vg
all_vg_ftns[b'Lin'] = lin_vg
all_vg_ftns[b'Gau'] = gau_vg
all_vg_ftns[b'Pow'] = pow_vg
all_vg_ftns[b'Hol'] = hol_vg


cdef void fill_theo_vg_vals(
        string vg_str, 
        const DT_D[:] h_arr, 
        DT_D r, 
        DT_D s, 
        DT_D[:] vg_arr) nogil except +:

    cdef:
        Py_ssize_t i, n
        
        f_type vg_ftn = all_vg_ftns[vg_str]

    n = h_arr.shape[0]

#     assert n, n
#     assert s >= 0
#     assert r >= 0

    for i in range(n):
        vg_arr[i] += vg_ftn(h_arr[i], r, s) 

    return

cdef void get_l2_norm(
        const DT_D[:, ::1] crds_1, 
        const DT_D[:, ::1] crds_2, 
              DT_D[::1] dists) nogil except +:

    cdef:
        Py_ssize_t i, j, n_vals = crds_1.shape[0], n_dims = crds_1.shape[1] 
        DT_D dist

    for i in range(n_vals):
        dist = 0
        for j in range(n_dims):
            dist += pow(crds_1[i, j] - crds_2[i, j], 2)

        dist **= 0.5
        dists[i] = dist

    return


cpdef double get_evgs_tvgs_diff_01(
        const DT_D[::1] new_crds,
              DT_D[::1, :] old_crds,
              DT_D[:, ::1] crds,
        const DT_D[::1] evgs,
              DT_D[::1] tvgs,
              DT_D[::1] dists,
        const uint32_t n_extra_dims,
        const uint32_t n_lags,
              fit_vg_str,
              DT_D[::1] calls_ctr) nogil except +:

    cdef:
        Py_ssize_t i, j, k, l, ctr, beg_extr_dim_idx = crds.shape[1] - n_extra_dims
        Py_ssize_t n_dims = crds.shape[1]
        uint32_t n_vals = new_crds.shape[0]
        DT_D obj_val, dist, r, s

        vector[DT_D] sills, ranges
        vector[string] vgs, vg_models

        f_type vg_ftn

    for j in range(n_extra_dims):
        for i in range(n_vals):
            old_crds[i, j] = crds[i, beg_extr_dim_idx + j]
            crds[i, beg_extr_dim_idx + j] = new_crds[(j * n_vals) + i]

    with gil:
        vg_models = bytes(fit_vg_str, 'utf-8').split(b'+')
        for submodel in vg_models:
            submodel = submodel.strip()
            sill, vg = submodel.split(b' ')
            vg, rng = vg.split(b'(')
            rng = rng.split(b')')[0]
       
            vgs.push_back(vg)
            sills.push_back(float(sill))
            ranges.push_back(max(1e-5, float(rng)))

    for i in range(tvgs.shape[0]):
        tvgs[i] = 0

    ctr = 0
    for i in range(1, n_lags + 1):
        get_l2_norm(crds[i:, :], crds[:n_vals - i, :], dists)
        # Inline implementation of the l2_norm.
#         for k in range(n_vals - i):
#             dist = 0.0
#             for l in range(n_dims):
#                 dist += pow(crds[i + k, l] - crds[k, l], 2)
# 
#             dist = pow(dist, 0.5)
#             dists[k] = dist

        for j in range(vgs.size()):
            # Inline implementation of fill_theo_vg_vals.
#             vg_ftn = all_vg_ftns[vgs[j]]
#             r = ranges[j]
#             s = sills[j]
#             for k in range(n_vals - i):
#                 tvgs[ctr + k] += vg_ftn(dists[k], r, s)

            fill_theo_vg_vals(
                vgs[j], 
                dists[:n_vals - i], 
                ranges[j], 
                sills[j], 
                tvgs[ctr:])

        ctr += n_vals - i

    obj_val = 0
    for i in range(evgs.shape[0]):
        obj_val += (evgs[i] - tvgs[i]) ** 2

    for j in range(n_extra_dims):
        for i in range(n_vals):
            crds[i, beg_extr_dim_idx + j] = old_crds[i, j]

    calls_ctr[0] += 1

    with gil:
        assert ctr == evgs.shape[0], (ctr, evgs.shape[0])
        print(f'{obj_val:0.5E}', calls_ctr[0])
    return obj_val
