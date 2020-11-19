# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# cython: language_level=3

from __future__ import division
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map


ctypedef double DT_D
ctypedef long DT_L
ctypedef unsigned long DT_UL

cdef DT_D INF = np.inf

cdef extern from "math.h" nogil:
    cdef:
        DT_D sin(DT_D x)
        DT_D exp(DT_D x)
        DT_D atan(DT_D x)
        DT_D M_PI
        DT_D pow(DT_D x, DT_D y)


cdef inline DT_D get_dist(DT_D x1, DT_D y1, DT_D x2, DT_D y2) nogil:
    """Get distance between points
    """
    cdef DT_D dist
    dist = pow((((x1 - x2)*(x1 - x2)) + ((y1 - y2)*(y1 - y2))), 0.5)
    return dist


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


ctypedef DT_D (*f_type)(DT_D h, DT_D r, DT_D s)
cdef map[string, f_type] all_vg_ftns
all_vg_ftns[b'Rng'] = rng_vg
all_vg_ftns[b'Nug'] = nug_vg
all_vg_ftns[b'Sph'] = sph_vg
all_vg_ftns[b'Exp'] = exp_vg
all_vg_ftns[b'Lin'] = lin_vg
all_vg_ftns[b'Gau'] = gau_vg
all_vg_ftns[b'Pow'] = pow_vg
all_vg_ftns[b'Hol'] = hol_vg


cpdef fill_theo_vg_vals(
        vg_str, 
        const DT_D[:] h_arr, 
        DT_D r, 
        DT_D s, 
        DT_D[:] vg_arr) except +:

    cdef:
        Py_ssize_t i, n
        
        f_type vg_ftn = all_vg_ftns[bytes(vg_str, 'utf-8').strip()]

    n = h_arr.shape[0]

    assert n
    assert h_arr.shape[0] == vg_arr.shape[0]
    assert s >= 0
    assert r >= 0

    for i in range(n):
        vg_arr[i] += vg_ftn(h_arr[i], r, s) 

    return


cpdef void fill_dists_2d_mat(
        const DT_D[::1] x1s,
        const DT_D[::1] y1s,
        const DT_D[::1] x2s,
        const DT_D[::1] y2s,
              DT_D[:, ::1] dists) except +:

    cdef:
        Py_ssize_t i, j
        DT_D x, y

    # dont use the ifs
    # ifs only work if distance with self
    for i in range(x1s.shape[0]):
        x = x1s[i]
        y = y1s[i]

        for j in range(x2s.shape[0]):
            dists[i, j] = (((x - x2s[j])**2) + ((y - y2s[j])**2))**0.5

    return


cpdef void fill_vg_var_arr(
        const DT_D[:, ::1] dists,
              DT_D[:, ::1] in_vars,
        int covar_flag,
        int diag_mat_flag,
        vg_models_str) except +:

    cdef:
        Py_ssize_t g, h
        int row_ct = dists.shape[0], col_ct = dists.shape[1]
        int cov_sign, cov_mult
        string vg_model, vg_s, range_s, sill_s
        vector[string] vg_models
        f_type vg_ftn

    if covar_flag:
        cov_sign = -1
        cov_mult = +1

    else:
        cov_sign = +1
        cov_mult = +0

    for h in range(row_ct):
        for g in range(col_ct):
            in_vars[h, g] = 0.0

    vg_models = bytes(vg_models_str, 'utf-8').split(b'+')
    for vg_model in vg_models:
        vg_model = vg_model.strip()

        sill_s, vg_s = vg_model.split(b' ')
        vg_s, range_s = vg_s.split(b'(')
        range_s = range_s.split(b')')[0]

        vg_ftn = all_vg_ftns[vg_s]
        range_f = max(1e-5, float(range_s))
        sill_f = float(sill_s)

        if diag_mat_flag:
            for h in range(row_ct):
                for g in range(col_ct):
                    if g < h:
                        continue

                    in_vars[h, g] += (
                        (cov_mult * sill_f) +
                        (cov_sign * vg_ftn(dists[h, g], range_f, sill_f)))

        else:
            for h in range(row_ct):
                for g in range(col_ct):
                    in_vars[h, g] += (
                        (cov_mult * sill_f) +
                        (cov_sign * vg_ftn(dists[h, g], range_f, sill_f)))

    if diag_mat_flag:
        for h in range(row_ct):
            for g in range(col_ct):
                if g >= h:
                    continue

                in_vars[h, g] = in_vars[g, h]
    return


cdef class OrdinaryKriging:
    '''Do ordinary kriging
    '''
    cdef:
        readonly np.ndarray xi, yi, zi, xk, yk, zk, est_vars, rhss, mus
        readonly np.ndarray in_dists, in_vars, out_vars, lambdas, in_vars_inv
        readonly string model, submodel, vg, rng, sill
        np.ndarray lambdas_k
        readonly unsigned long in_count, out_count
        readonly vector[DT_D] sills, ranges
        readonly vector[string] vgs, vg_models
        unsigned long f, g, h, i, j, k, l, m, n
        DT_D x_interp, y_interp, out_dist
        DT_D range_f, sill_f
        vector[f_type] vg_ftns
        f_type vg_ftn

    def __init__(self, xi, yi, zi, xk, yk, model='1.0 Sph(2)'):

        self.xi = xi
        self.yi = yi
        self.zi = zi
        self.xk = xk
        self.yk = yk
        self.model = bytes(model, 'utf-8')

        self.in_count = self.xi.shape[0]
        self.out_count = self.xk.shape[0]
        self.zk = np.zeros(shape=(self.out_count), dtype=np.float64)
        self.in_dists = np.zeros(shape=(self.in_count, self.in_count),
                                 dtype=np.float64)
        self.mus = np.zeros(shape=(self.out_count,), dtype=np.float64)
        self.est_vars = np.zeros(shape=(self.out_count,), dtype=np.float64)
        self.lambdas = np.zeros(shape=(self.out_count, self.in_count),
                                dtype=np.float64)
        self.in_vars = np.zeros(shape=(self.in_count + 1, self.in_count + 1),
                                dtype=np.float64)
        self.in_vars_inv = self.in_vars.copy()
        self.rhss = np.zeros(shape=(self.out_count, self.in_count + 1),
                             dtype=np.float64)
        return

    def krige(self):

        vg_models = self.model.split(b'+')
        for submodel in vg_models:
            submodel = submodel.strip()
            sill, vg = submodel.split(b' ')
            vg, rng = vg.split(b'(')
            rng = rng.split(b')')[0]
            self.vgs.push_back(vg)
            self.sills.push_back(float(sill))
            self.ranges.push_back(max(1e-5, float(rng)))
            self.vg_ftns.push_back(all_vg_ftns[vg])

        for i in range(self.in_count):
            x1 = self.xi[i]
            y1 = self.yi[i]
            for j in range(self.in_count):
                if j > i:
                    self.in_dists[i, j] = get_dist(x1,
                                                   y1,
                                                   self.xi[j],
                                                   self.yi[j])
                elif i > j:
                    self.in_dists[i, j] = self.in_dists[j, i]

        for f in range(self.vg_ftns.size()):
            vg_ftn = self.vg_ftns[f]
            range_f = self.ranges[f]
            sill_f = self.sills[f]
            for h in range(self.in_count):
                for g in range(self.in_count):
                    if g > h:
                        self.in_vars[h, g] += vg_ftn(self.in_dists[h, g],
                                                     range_f,
                                                     sill_f)
                    elif h > g:
                        self.in_vars[h, g] = self.in_vars[g, h]

        self.in_vars[self.in_count, :self.in_count] = np.ones(self.in_count)
        self.in_vars[:, self.in_count] = np.ones(self.in_count + 1)
        self.in_vars[self.in_count, self.in_count] = 0

        self.in_vars_inv = np.linalg.inv(self.in_vars)

        for k in range(self.out_count):
            x_interp = self.xk[k]
            y_interp = self.yk[k]
            out_vars = np.zeros(shape=(self.in_count + 1))

            for l in range(self.in_count):
                out_dist = get_dist(self.xi[l], self.yi[l], x_interp, y_interp)
                if out_dist == 0.0:
                    continue
                for n in range(self.vg_ftns.size()):
                    out_vars[l] += self.vg_ftns[n](out_dist,
                                                   self.ranges[n],
                                                   self.sills[n])

            out_vars[self.in_count] = 1
            lambdas_k = np.matmul(self.in_vars_inv, out_vars)
#             lambdas_k = np.linalg.solve(self.in_vars, out_vars)
#            lambdas_k = np.linalg.lstsq(self.in_vars, out_vars)
            self.mus[k] = lambdas_k[self.in_count]
            lambdas_k = lambdas_k[:self.in_count]
            self.zk[k] = np.sum((lambdas_k * self.zi))
            self.est_vars[k] = max(0.0,
                np.sum(lambdas_k * out_vars[:self.in_count]) + self.mus[k])
            self.lambdas[k, :] = lambdas_k
            self.rhss[k] = out_vars
        return


cdef class SimpleKriging:
    '''Do simple kriging
    '''
    cdef:
        readonly np.ndarray xi, yi, zi, xk, yk, zk, est_covars, rhss
        readonly np.ndarray in_dists, in_covars, out_covars, lambdas, in_covars_inv
        readonly string model, submodel, vg, rng, sill
        np.ndarray lambdas_k, out_covars_k
        readonly unsigned long in_count, out_count
        readonly vector[DT_D] sills, ranges
        readonly vector[string] vgs, vg_models # cannot access it outside
        unsigned long f, g, h, i, j, k, l, m, n
        DT_D x1, y1, x_interp, y_interp, out_dist
        vector[f_type] vg_ftns
        f_type vg_ftn
        DT_D covar, range_f, sill_f
 
    def __init__(self, xi, yi, zi, xk, yk, model='1.0 Sph(2)'):
 
        self.xi = xi
        self.yi = yi
        self.zi = zi
        self.xk = xk
        self.yk = yk
        self.model = bytes(model, 'utf-8')
 
        self.in_count = self.xi.shape[0]
        self.out_count = self.xk.shape[0]
        self.zk = np.zeros(shape=(self.out_count,))
        self.in_dists = np.zeros(shape=(self.in_count, self.in_count))
        self.est_covars = np.zeros(shape=(self.out_count,))
        self.lambdas = np.zeros(shape=(self.out_count, self.in_count))
        self.covar = 0.0
        self.in_covars = np.full(shape=(self.in_count, self.in_count),
                                 fill_value=self.covar)
        self.in_covars_inv = self.in_covars.copy()
        self.out_covars = np.zeros(shape=(self.out_count, self.in_count))
        self.rhss = np.zeros(shape=(self.out_count, self.in_count))
        return
 
    def krige(self):
        vg_models = self.model.split(b'+')
        for submodel in vg_models:
            submodel = submodel.strip()
            sill, vg = submodel.split(b' ')
            vg, rng = vg.split(b'(')
            rng = rng.split(b')')[0]

            self.covar += float(sill)

            self.vgs.push_back(vg)
            self.sills.push_back(float(sill))
            self.ranges.push_back(max(1e-5, float(rng)))
            self.vg_ftns.push_back(all_vg_ftns[vg])
 
        for i in range(self.in_count):
            x1 = self.xi[i]
            y1 = self.yi[i]
            for j in range(self.in_count):
                if j > i:
                    self.in_dists[i, j] = get_dist(x1,
                                                   y1,
                                                   self.xi[j],
                                                   self.yi[j])
                elif i > j:
                    self.in_dists[i, j] = self.in_dists[j, i]
 
        for f in range(self.vg_ftns.size()):
            vg_ftn = self.vg_ftns[f]
            range_f = self.ranges[f]
            sill_f = self.sills[f]
            for h in range(self.in_count):
                for g in range(self.in_count):
                    if g > h:
                        self.in_covars[h, g] -= vg_ftn(self.in_dists[h, g],
                                                       range_f,
                                                       sill_f)
                    elif h > g:
                        self.in_covars[h, g] = self.in_covars[g, h]

        self.in_covars_inv = np.linalg.inv(self.in_covars)

        for k in range(self.out_count):
            x_interp = self.xk[k]
            y_interp = self.yk[k]
            out_covars_k = np.full(shape=(self.in_count,),
                                   fill_value=self.covar)
 
            for l in range(self.in_count):
                out_dist = get_dist(self.xi[l], self.yi[l], x_interp, y_interp)
                if out_dist == 0.0:
                    continue
                for n in range(self.vg_ftns.size()):
                    out_covars_k[l] -= self.vg_ftns[n](out_dist,
                                                       self.ranges[n],
                                                       self.sills[n])
 
#             lambdas_k = np.linalg.solve(self.in_covars, out_covars_k)
            lambdas_k = np.matmul(self.in_covars_inv, out_covars_k)
            self.zk[k] = np.sum((lambdas_k * self.zi))
            self.est_covars[k] = max(0.0,
                self.covar - np.sum(lambdas_k * out_covars_k))
 
            self.lambdas[k, :] = lambdas_k
            self.out_covars[k, :] = out_covars_k
            self.rhss[k] = out_covars_k
        return
 
 
cdef class ExternalDriftKriging:
    '''Do external drift kriging
    '''
    cdef:
        readonly np.ndarray xi, yi, zi, xk, yk, zk
        readonly np.ndarray si, sk, mus_1, mus_2, rhss
        readonly np.ndarray in_dists, in_vars, out_vars, lambdas, in_vars_inv
        readonly string model, submodel, vg, rng, sill
        np.ndarray lambdas_k
        readonly unsigned long in_count, out_count
        readonly vector[DT_D] sills, ranges
        readonly vector[string] vgs, vg_models
        unsigned long f, g, h, i, j, k, l, m, n
        DT_D x1, y1, x_interp, y_interp, out_dist
        DT_D range_f, sill_f
        vector[f_type] vg_ftns
        f_type vg_ftn
 
 
    def __init__(self, xi, yi, zi, si, xk, yk, sk, model='1.0 Sph(2)'):
 
        self.xi = xi
        self.yi = yi
        self.zi = zi
        self.si = si
        self.xk = xk
        self.yk = yk
        self.sk = sk
        self.model = bytes(model, 'utf-8')
 
        self.in_count = self.xi.shape[0]
        self.out_count = self.xk.shape[0]
        self.zk = np.zeros(shape=(self.out_count,))
        self.in_dists = np.zeros(shape=(self.in_count, self.in_count))
        self.mus_1 = np.zeros(shape=(self.out_count,))
        self.mus_2 = np.zeros(shape=(self.out_count,))
        self.lambdas = np.zeros(shape=(self.out_count, self.in_count))
        self.in_vars = np.zeros(shape=(self.in_count + 2, self.in_count + 2))
        self.in_vars_inv = self.in_vars.copy()
        self.rhss = np.zeros(shape=(self.out_count, self.in_count + 2))
        return
 
    def krige(self):
        vg_models = self.model.split(b'+')
        for submodel in vg_models:
            submodel = submodel.strip()
            submodel = submodel.strip()
            sill, vg = submodel.split(b' ')
            vg, rng = vg.split(b'(')
            rng = rng.split(b')')[0]
            self.vgs.push_back(vg)
            self.sills.push_back(float(sill))
            self.ranges.push_back(max(1e-5, float(rng)))
            self.vg_ftns.push_back(all_vg_ftns[vg])
 
        for i in range(self.in_count):
            x1 = self.xi[i]
            y1 = self.yi[i]
            for j in range(self.in_count):
                if j > i:
                    self.in_dists[i, j] = get_dist(x1,
                                                   y1,
                                                   self.xi[j],
                                                   self.yi[j])
                elif i > j:
                    self.in_dists[i, j] = self.in_dists[j, i]
 
        for f in range(self.vg_ftns.size()):
            vg_ftn = self.vg_ftns[f]
            range_f = self.ranges[f]
            sill_f = self.sills[f]
            for h in range(self.in_count):
                for g in range(self.in_count):
                    if g > h:
                        self.in_vars[h, g] += vg_ftn(self.in_dists[h, g],
                                                     range_f,
                                                     sill_f)
                    elif h > g:
                        self.in_vars[h, g] = self.in_vars[g, h]
 
        self.in_vars[self.in_count, :self.in_count] = np.ones(self.in_count)
        self.in_vars[:self.in_count, self.in_count] = np.ones(self.in_count)
        self.in_vars[self.in_count + 1, :self.in_count] = self.si
        self.in_vars[:self.in_count, self.in_count + 1] = self.si
 
        self.in_vars_inv = np.linalg.inv(self.in_vars)

        for k in range(self.out_count):
            x_interp = self.xk[k]
            y_interp = self.yk[k]
            out_vars = np.zeros(shape=(self.in_count + 2))
 
            for l in range(self.in_count):
                out_dist = get_dist(self.xi[l], self.yi[l], x_interp, y_interp)
                if out_dist == 0.0:
                    continue
                for n in range(self.vg_ftns.size()):
                    out_vars[l] += self.vg_ftns[n](out_dist,
                                                   self.ranges[n],
                                                   self.sills[n])
 
            out_vars[self.in_count] = 1
            out_vars[self.in_count + 1] = self.sk[k]
#             lambdas_k = np.linalg.solve(self.in_vars, out_vars)
            lambdas_k = np.matmul(self.in_vars_inv, out_vars)
            self.mus_1[k] = lambdas_k[self.in_count]
            self.mus_2[k] = lambdas_k[self.in_count + 1]
            lambdas_k = lambdas_k[:self.in_count]
            self.zk[k] = np.sum((lambdas_k * self.zi))
            self.lambdas[k, :] = lambdas_k
            self.rhss[k] = out_vars
        return
 
 
cdef class ExternalDriftKriging_MD:
    '''Do external drift kriging with multiple drifts
    '''
    cdef:
        readonly np.ndarray xi, yi, zi, xk, yk, zk
        readonly np.ndarray si, sk, mus_arr, rhss
        readonly np.ndarray in_dists, in_vars, out_vars, lambdas, in_vars_inv
        readonly string model, submodel, vg, rng, sill
        np.ndarray lambdas_k
        readonly unsigned long in_count, out_count
        readonly vector[DT_D] sills, ranges
        readonly vector[string] vgs, vg_models
        unsigned long f, g, h, i, j, k, l, m, n, n_drifts
        DT_D x1, y1, x_interp, y_interp, out_dist
        DT_D range_f, sill_f
        vector[f_type] vg_ftns
        f_type vg_ftn
 
 
    def __init__(self, xi, yi, zi, si, xk, yk, sk, model='1.0 Sph(2)'):
 
        assert len(xi.shape) == 1
        assert len(si.shape) == 2
 
        assert xi.shape[0] == yi.shape[0] == zi.shape[0] == si.shape[1], \
            'Observation points and drift shapes are unequal!'
        assert xk.shape[0] == yk.shape[0] == sk.shape[1], \
            'Resulting points and drifts shapes are unequal!'
        assert si.shape[0] == sk.shape[0], \
            'Observation and reulting drifts have unequal shapes!'
 
        self.xi = xi
        self.yi = yi
        self.zi = zi
        self.si = si
        self.xk = xk
        self.yk = yk
        self.sk = sk
        self.model = bytes(model, 'utf-8')
        self.n_drifts = sk.shape[0]
 
        self.in_count = self.xi.shape[0]
        self.out_count = self.xk.shape[0]
        self.zk = np.zeros(shape=(self.out_count,))
        self.in_dists = np.zeros(shape=(self.in_count, self.in_count))
#        self.mus_1 = np.zeros(shape=(self.out_count,))
#        self.mus_2 = np.zeros(shape=(self.out_count,))
        self.mus_arr = np.zeros(shape=(self.n_drifts + 1, self.out_count))
        self.lambdas = np.zeros(shape=(self.out_count, self.in_count))
        self.in_vars = np.zeros(shape=(self.in_count + self.n_drifts + 1,
                                       self.in_count + self.n_drifts + 1))
        self.in_vars_inv = self.in_vars.copy()
        self.rhss = np.zeros(shape=(self.out_count,
                                    self.in_count + self.n_drifts + 1))
        return
 
    def krige(self):
        vg_models = self.model.split(b'+')
        for submodel in vg_models:
            submodel = submodel.strip()
            submodel = submodel.strip()
            sill, vg = submodel.split(b' ')
            vg, rng = vg.split(b'(')
            rng = rng.split(b')')[0]
            self.vgs.push_back(vg)
            self.sills.push_back(float(sill))
            self.ranges.push_back(max(1e-5, float(rng)))
            self.vg_ftns.push_back(all_vg_ftns[vg])
 
        for i in range(self.in_count):
            x1 = self.xi[i]
            y1 = self.yi[i]
            for j in range(self.in_count):
                if j > i:
                    self.in_dists[i, j] = get_dist(x1,
                                                   y1,
                                                   self.xi[j],
                                                   self.yi[j])
                elif i > j:
                    self.in_dists[i, j] = self.in_dists[j, i]
 
        for f in range(self.vg_ftns.size()):
            vg_ftn = self.vg_ftns[f]
            range_f = self.ranges[f]
            sill_f = self.sills[f]
            for h in range(self.in_count):
                for g in range(self.in_count):
                    if g > h:
                        self.in_vars[h, g] += vg_ftn(self.in_dists[h, g],
                                                     range_f,
                                                     sill_f)
                    elif h > g:
                        self.in_vars[h, g] = self.in_vars[g, h]
 
        self.in_vars[self.in_count, :self.in_count] = np.ones(self.in_count)
        self.in_vars[:self.in_count, self.in_count] = np.ones(self.in_count)
 
        for m in range(self.n_drifts):
            self.in_vars[self.in_count + 1 + m, :self.in_count] = self.si[m]
            self.in_vars[:self.in_count, self.in_count + 1 + m] = self.si[m]
 
        self.in_vars_inv = np.linalg.inv(self.in_vars)

        for k in range(self.out_count):
            x_interp = self.xk[k]
            y_interp = self.yk[k]
            out_vars = np.zeros(shape=(self.in_count + self.n_drifts + 1))
 
            for l in range(self.in_count):
                out_dist = get_dist(self.xi[l], self.yi[l], x_interp, y_interp)
                if out_dist == 0.0:
                    continue
                for n in range(self.vg_ftns.size()):
                    out_vars[l] += self.vg_ftns[n](out_dist,
                                                   self.ranges[n],
                                                   self.sills[n])
 
            out_vars[self.in_count] = 1
            for m in range(self.n_drifts):
                out_vars[self.in_count + 1 + m] = self.sk[m, k]
#             lambdas_k = np.linalg.solve(self.in_vars, out_vars)
            lambdas_k = np.matmul(self.in_vars_inv, out_vars) 
            for m in range(self.n_drifts + 1):
                self.mus_arr[m, k] = lambdas_k[self.in_count + m]
 
#            self.mus_1[k] = lambdas_k[self.in_count]
#            self.mus_2[k] = lambdas_k[self.in_count + 1]
            lambdas_k = lambdas_k[:self.in_count]
            self.zk[k] = np.sum((lambdas_k * self.zi))
            self.lambdas[k, :] = lambdas_k
            self.rhss[k] = out_vars
        return
 
 
cdef class OrdinaryIndicatorKriging(OrdinaryKriging):
    '''Do indicator kriging based on ordinary kriging
    '''
    cdef:
        readonly DT_D lim
        unsigned long o
        readonly np.ndarray ik, ixi
 
    def __init__(self, xi, yi, zi, xk, yk, lim=1, model='1.0 Sph(2)'):
        OrdinaryKriging.__init__(self, xi, yi, zi, xk, yk, model=model)
        self.lim = lim
        self.ixi = np.where(self.zi <= self.lim, 1., 0.)
        self.ik = np.zeros(shape=(self.out_count,))
        return
 
    def ikrige(self):
        self.krige()
        for o in range(self.out_count):
            self.ik[o] = max(0.0, np.sum((self.lambdas[o, :] * self.ixi)))
            self.est_vars[o] = max(0.0, self.ik[o] * (1. - self.ik[o]))
        return

 
cdef class SimpleIndicatorKriging(SimpleKriging):
    '''Do indicator kriging based on simple kriging
    '''
    cdef:
        readonly DT_D lim
        unsigned long o
        readonly np.ndarray ik, ixi
 
    def __init__(self, xi, yi, zi, xk, yk, lim=1, model='1.0 Sph(2)'):
        SimpleKriging.__init__(self, xi, yi, zi, xk, yk, model=model)
        self.lim = lim
        self.ixi = np.where(self.zi <= self.lim, 1., 0.)
        self.ik = np.zeros(shape=(self.out_count,))
        return
 
    def ikrige(self):
        self.krige()
        for o in range(self.out_count):
            self.ik[o] = max(0.0, np.sum((self.lambdas[o, :] * self.ixi)))
            self.est_covars[o] = max(0.0, self.ik[o] * (1. - self.ik[o]))
        return


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
