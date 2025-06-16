# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

# To avoid the annoying type not found warning from PyDev.
from numpy cimport (
    uint8_t, int8_t, int64_t, npy_float32 as float32, npy_float64 as float64)

# To avoid the annoying type not found warning from PyDev.
cdef extern from 'Python.h':
    ctypedef Py_ssize_t Py_ssize_t

cdef extern from 'math.h' nogil:
    cdef float32 powf(float32 x, float32 e)


def cmpt_wgtd_dist(pts_org, wts, dts, pts):
    
    pts_cnt = pts_org.shape[0]
    dts_cnt = (pts_cnt * (pts_cnt - 1)) // 2

    pnt_ctr = _cmpt_wgtd_dist(wts, dts, pts, pts_org)

    assert pnt_ctr == dts_cnt, (pnt_ctr, dts_cnt)

    return dts

cdef Py_ssize_t _cmpt_wgtd_dist(
        const float32[::1] wts,
              float32[::1] dts,
              float32[:, ::1] pts,
        const float32[:, ::1] pts_org,) nogil except +:


    cdef:
        Py_ssize_t i, j, k

        Py_ssize_t dms = pts_org.shape[1]
        Py_ssize_t pts_cnt = pts_org.shape[0]
        Py_ssize_t pnt_ctr = 0

        float32 dst

    for j in range(dms):
        for i in range(pts_org.shape[0]):
            pts[i, j] = pts_org[i, j] * wts[j]

    for i in range(pts_cnt):
        for j in range(i + 1, pts_cnt, 1):

            dst = <float32> 0.0
            for k in range(dms):
                dst += powf(pts[i, k] - pts[j, k], 2)
            dst = powf(dst, 0.5)

            dts[pnt_ctr] = dst

            pnt_ctr += 1

    return pnt_ctr