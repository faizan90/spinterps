import warnings
from math import pi, sqrt
from copy import deepcopy
from itertools import permutations

import numpy as np
from pandas import DataFrame
from scipy.optimize import differential_evolution

__all__ = ['Variogram']


class Variogram:
    """Calculate variograms based on different criteria.

    This class also changes coordinates of points to a grid.
    """

    def __init__(self,
                 x,
                 y,
                 z,
                 nk=30,
                 mdr=0.5,
                 typ='var',
                 perm_r_list=None,
                 fil_nug_vg=None,
                 ngp=5,
                 fit_thresh=0.05,
                 ld=None,
                 uh=None,
                 h_itrs=100,
                 opt_meth='L-BFGS-B',
                 opt_iters=1000,
                 grid='no',
                 cell_size='auto',
                 xmin='auto',
                 ymin='auto',
                 xmax='auto',
                 ymax='auto',
                 fit_vgs=['all'],
                 n_best=1,
                 evg_name='classic',
                 use_wts=True):

        # the horizontal coordinates of the points
        x = np.array(x, dtype=np.float64)

        # the vertical coordinates of the points
        y = np.array(y, dtype=np.float64)

        # the z values of all points
        z = np.array(z, dtype=np.float64)

        assert x.shape == y.shape == z.shape, 'Unequal input shapes!'

        # =====================================================================
        # Parameter 'nk':
        # minimum number of pairs in a single bin if the
        # function call_cnst_vg(self) with ld=None is used. Each bin will have
        # 'nk' pairs if the function call_var_vg(self) is used, except for the
        # last bin. That number depends on the total number of
        # pairs and the value of 'nk'. One can select a value that
        # gives equal number of values in all the bins.
        # =====================================================================

        self.nk = nk

        # =====================================================================
        # Parameter 'mdr':
        # the ratio between the distance (after which no pairs
        # are considered for the analysis) and the maximum
        # pair distance. e.g. if mdr=0.5 then pairs that have a
        # a distance greater than 0.5 * maximum pair distance are
        # not considered for the analysis.
        # =====================================================================

        self.mdr = mdr

        # =====================================================================
        # Paramter 'typ':
        # The type of bin width to use. 'var' means 'nk' number of pairs in all
        # bins except for the last bin (depending on the number of pairs
        # remaining at the end). 'cnst' means constant bin width which gives
        # different number of pairs in each bin. The bin width can be
        # calculated automatically such that each bin has atleast 'nk' number
        # of pairs (if 'ld' is None) or it can be given any real value through
        # the parameter 'ld'.
        # =====================================================================

        self.typ = typ

        # =====================================================================
        # Parameter 'perm_r_list':
        # The number of permutations to use while fitting variograms e.g. if
        # it is [1] then only one variogram type is fitted to the data. The
        # type of variograms used depends on the 'fit_vgs'. if 'Nug' is also
        # given in the list and the list is [1] then Nugget + other variograms
        # are fitted. The Nugget is not considered a variogram like others.
        # if it is [1, 2] then a permutation of two types of variograms is
        # fitted plus the Nugget (if specified in 'fit_vgs').
        # =====================================================================

        self.perm_r_list = perm_r_list
        if self.perm_r_list is None:
            self.perm_r_list = [1]

        # =====================================================================
        # Parameter 'fil_nug_vg':
        # The variogram to use for the distance between 0 and the minimum lag
        # distance in the output variogram. This is done to avoid the
        # discontinuity at the origin. It helps in random mixing. The range of
        # this variogram is the minimum lag distance in the output variogram.
        # =====================================================================

        self.fil_nug_vg = fil_nug_vg

        self.ngp = ngp  # no. of grid points to use in Brute force algorithm

        # =====================================================================
        # Parameter 'fit_thresh':
        # one of the criterion for solution selection after optimization.
        # a solution is selected if:
        #   new objective function < (old solution * (1 - fit_thresh))
        # =====================================================================

        self.fit_thresh = fit_thresh

        # =====================================================================
        # Parameter 'ld':
        # a constant lag distance. Used in the vg_cnst_raw function.
        # if ld is None then a lag distance is calculated
        # (iteratively) such that each bin has a minimum or greater
        # than 'nk' number of pairs in it.
        # =====================================================================

        self.ld = ld

        # =====================================================================
        # Parameter 'uh':
        # maximum cutoff pair distance. can also be specified as a
        # number as opposed to the ratio 'n_ld'. Pairs having
        # distance greater than uh' are not considered. if both
        # 'mdr'and 'uh' are specifed then the minimum is selected
        # Specify in input unts only
        # =====================================================================

        self.uh = uh

        # =====================================================================
        # Parameter: h_itrs
        # maximum number of iterations to calculate
        # lag distance in case the vg_cnst_raw is used
        # =====================================================================

        self.h_itrs = int(h_itrs)

        # =====================================================================
        # Parameter 'opt_meth:
        # name of the optimization function to be used
        # in minimize function e.g. Nelder-Mead, BFGS,
        # TNS, Powell
        # =====================================================================

        self.opt_meth = opt_meth

        # =====================================================================
        # Parameter 'opt_iters':
        # maximum number of iteration that the
        # minimize function can perform. It is used
        # in some methods only.
        # =====================================================================

        self.opt_iters = opt_iters

        self.cell_size = cell_size  # the cell size if grid = 'yes'
        self.xmin = xmin  # lower limit of x
        self.ymin = ymin  # lower limit of y
        self.xmax = xmax  # upper limit of x
        self.ymax = ymax  # upper limit of y

        # =====================================================================
        # Parameter 'n_best':
        # return the n_best number of variograms (based on the objective
        # function value)
        # =====================================================================

        self.n_best = n_best

        # =====================================================================
        # method to create empirical variogram ('classic' or 'robust')
        # =====================================================================

        self.evg_name = evg_name

        # =====================================================================
        # Parameter 'use_wts':
        # Use the weighting technique based on giving more weights to bins with
        # more pairs in it
        # =====================================================================

        self.use_wts = use_wts

        if 'all' in fit_vgs:
            # add all the available variograms
            self.fit_vgs = ['Nug', 'Sph', 'Exp', 'Lin', 'Gau', 'Pow', 'Hol']

        else:
            # all this to get an ordered list
            self.fit_vgs = []
            if 'Nug' in fit_vgs:
                self.fit_vgs.append('Nug')
            if 'Sph' in fit_vgs:
                self.fit_vgs.append('Sph')
            if 'Exp' in fit_vgs:
                self.fit_vgs.append('Exp')
            if 'Lin' in fit_vgs:
                self.fit_vgs.append('Lin')
            if 'Gau' in fit_vgs:
                self.fit_vgs.append('Gau')
            if 'Pow' in fit_vgs:
                self.fit_vgs.append('Pow')
            if 'Hol' in fit_vgs:
                self.fit_vgs.append('Hol')

        if len(self.fit_vgs) == 0:
            raise ValueError(r"No relevant variograms in fit_vgs list.")

        pairs_list = []  # a temporary list for the 'z' value pairs

        # =====================================================================
        # Parameter 'bin_min_pairs_list':
        # a list containing minimum number of pairs and their respective lag
        # distances. this is used when 'ld' is None and the vg_const_h function
        # is called.
        # =====================================================================

        self._bin_min_pairs_list = []

        self._itrs = 0  # a counter variable used to estimate lag distance
        self._emsg = ''  # an empty string to store error messages in

        # get limits of coordinate values
        if self.xmin == 'auto':
            self.xmin = np.min(x)
        elif (type(self.xmin) == int) or (type(self.xmin) == float):
            pass
        else:
            print("""
            \r\a The parameter 'xmin' is neither 'auto' nor its data type
            \ris 'int' or 'float'. Using the 'auto' option anyways...\n
            """)
            self.xmin = np.min(x)

        if self.ymin == 'auto':
            self.ymin = np.min(y)
        elif (type(self.ymin) == int) or (type(self.ymin) == float):
            pass
        else:
            print("""
            \r\a The parameter 'ymin' is neither 'auto' nor its data type
            \ris 'int' or 'float'. Using the 'auto' option anyways...\n
            """)
            self.ymin = np.min(y)

        if self.xmax == 'auto':
            self.xmax = np.max(x)
        elif (type(self.xmax) == int) or (type(self.xmax) == float):
            pass
        else:
            print("""
            \r\a The parameter 'xmax' is neither 'auto' nor its data type
            \ris 'int' or 'float'. Using the 'auto' option anyways...\n
            """)
            self.xmax = np.max(x)

        if self.ymax == 'auto':
            self.ymax = np.max(y)
        elif (type(self.ymax) == int) or (type(self.ymax) == float):
            pass
        else:
            print("""
            \r\a The parameter 'ymax' is neither 'auto' nor its data type
            \ris 'int' or 'float'. Using the 'auto' option anyways...\n
            """)
            self.ymax = np.max(y)

        if self.xmin > self.xmax:
            raise ValueError("""\r\a The parameter 'xmin' is greater than 'xmax'.
            \rCannot continue...
            """)

        if self.ymin > self.ymax:
            raise ValueError("""\r\a The parameter 'ymin' is greater than 'ymax'.
            \rCannot continue...
            """)

        # drop values that are out of xmin, ymin, xmax and ymax
        self._coords_df = DataFrame(data={'x': x, 'y': y, 'z': z})

        x_vals = self._coords_df['x'].values
        y_vals = self._coords_df['y'].values

        x_vals[x_vals < self.xmin] = np.nan
        x_vals[x_vals > self.xmax] = np.nan
        y_vals[y_vals < self.ymin] = np.nan
        y_vals[y_vals > self.ymax] = np.nan
        self._coords_df.dropna(inplace=True)

        # creating a matrix that gives distances of all the points
        # with respect to each other
        x = self._coords_df['x'].values[:]
        y = self._coords_df['y'].values[:]
        z = self._coords_df['z'].values[:]

        self.tot_pts = x.shape[0]

        # get a distance matrix
        xx1, xx2 = np.meshgrid(x, x)
        yy1, yy2 = np.meshgrid(y, y)
        dists = np.sqrt((xx2 - xx1) ** 2 + (yy2 - yy1) ** 2)

        # convert regular coordinates to pixels based on given info
        if (grid == 'yes') or (grid != 'no'):
            cell_size_adj = 0.99  # a constant to multiply with cell_size
            if (grid != 'yes'):
                print("""
                \r\a The parameter 'grid' is neither 'yes' nor 'no'.
                \rUsing 'yes' anyways...\n\r
                """)

            if self.cell_size == 'auto':
                self.cell_size = cell_size_adj * sqrt(pow(np.unique(dists)[1],
                                                        2) / 2)
            elif ((type(self.cell_size) == int) or
                    (type(self.cell_size) == float)):
                pass
            else:
                print("""
                \r\a The parameter 'cell_size' is neither 'auto' nor its
                \rdata type is 'int' or 'float'. Using the 'auto' option
                \ranyways...\n
                """)
                self.cell_size = cell_size_adj * sqrt(pow(np.unique(dists)[1],
                                                        2) / 2)

            if self. ld is not None:
                self.ld = float(self.ld) / self.cell_size
            # reassign x and y as column and row numbers
            x = np.array(((x - self.xmin) / self.cell_size),
                         dtype=np.int64)

            y = np.array(((y - self.ymin) / self.cell_size),
                         dtype=np.int64)

            # calculate distances in the form of pixels
            xx1, xx2 = np.meshgrid(x, x)
            yy1, yy2 = np.meshgrid(y, y)
            dists = np.sqrt((xx2 - xx1) ** 2 + (yy2 - yy1) ** 2)

        elif grid == 'no':
            pass

        # making pairs of all the values along with the respective
        # distance/pixels between them. The output array format is
        # [value1, value2, distance] the points having zero distance
        # are dropped.
        for i in range(z.shape[0]):
            for j in range(z.shape[0]):
                if (i > j):
                    pairs_list.append((z[i], z[j], dists[i, j]))
                else:
                    break

        self.pair_df = DataFrame(pairs_list, columns=['z1', 'z2', 'dist'])

        # check to see if required and calculated number of pairs are equal
        self._tot_pairs = ((x.shape[0] * (x.shape[0] - 1)) / 2)

        self._tot_pairs_a = self.pair_df.shape[0]

        if self._tot_pairs != self._tot_pairs_a:
            self._emsg = """
            \r\a Calculated and required number of pairs
            \rare not the same. Calculated number of pairs
            \rare: %d and required number of pairs are: %d
            """
            raise ValueError(self._emsg % (self._tot_pairs_a, self._tot_pairs))

        self.pair_df.sort_values(by=['dist'], inplace=True)

        self.n_best_objs = []
        self.vg_str_list = []
        self.vg_fit = []
        self.variance = np.var(z)
        return

    def evg_classic(self, xs, ys):
        """Calculate the empirical variogram value for given z values."""
        p2x = (2 * xs.shape[0])
        if p2x == 0:
            return np.nan
#             raise ZeroDivisionError
        else:
#             sqs = (np.subtract(xs, ys)) ** 2
#             sum_vg = np.sum(sqs)
#             evg = (sum_vg / p2x)

            evg = np.median((np.subtract(xs, ys)) ** 2)

            return evg

    def evg_robust(self, xs, ys):
        """Calculate the robust variogram value for given z values."""
        Nh = xs.shape[0]
        if Nh == 0:
            return 0
        else:
            evg = ((np.sum((np.abs(xs - ys)) ** 0.5, axis=0)) / Nh) ** 4
            evg = np.divide(evg, (0.457 + (0.494 / Nh))) / 2
            return evg

    def _evg(self, xs, ys, evg_name='classic'):
        if evg_name == 'classic':
            return self.evg_classic(xs, ys)
        elif evg_name == 'robust':
            return self.evg_robust(xs, ys)
        else:
            raise NameError('Incorrect empirical variogram estimator name: %s!'
                            % evg_name)

    def _vg_cloud(self):
        """Calculate the cloud variogram.

        The ouput format of the array is: [distance between the points,
        variogram value].
        """
        p = ((self._pairs_arr[:, 0] - self._pairs_arr[:, 1]) ** 2) / 2

        return np.append(self._pairs_arr[:, 2].reshape(self._tot_pairs_a, 1),
                         p.reshape(self._tot_pairs_a, 1), axis=1)

    def _vg_cnst_raw(self):
        """Calculate variogram values based on a constant lag-distance.

        If lag distance is not specified then it is calculated using the
        _lag_dist function. based on lag distance the bins are made. These
        bins have the distance values in the format: [lower limit of bin, mean
        distance in the bin, upper limit of the bin]. The returned array
        contains data in the format: [mean distance in the bin, variogram
        value for the given pairs in the bin, number of pairs in the bin,
        lower limit of the bin, upper limit of the bin]. Bin limits are
        adjusted based on data. The mean of the bin distances can be used as
        the 'h' or just the average value of h_min and h_max.
        """
        bins_h = []

        if self.ld is None:
            self.ld = self._lag_dist()

        h_min = self.lh
        h_max = h_min + self.ld

        while (h_min < self.max_dist):
            if h_max > self.max_dist:
                h_max = self.max_dist * 1.000000001  # to have the last pair
                if round(h_max, 6) == round(h_min, 6):
                    break

            h_avg = (h_min + h_max) / 2

            bins_h.append((h_min, h_avg, h_max))

            h_min = h_max
            h_max = h_min + self.ld

        vg_h_list = []

        for k in bins_h:
            l = self._pairs_arr.copy()
            l = l[(l[:, 2] >= k[0]) & (l[:, 2] < k[2])]
            avg_h = k[1]  # np.mean(l[:, 2])
            vg_h_list.append((avg_h, self._evg(l[:, 0], l[:, 1], self.evg_name),
                              l.shape[0], k[0], k[2]))

        self._vg_arr_raw = np.array(vg_h_list)

        self.vg_vg_arr = self._vg_arr_raw[:, 1]
        self.vg_h_arr = self._vg_arr_raw[:, 0]
        self.pairs_no = self._vg_arr_raw[:, 2]  # number of pairs in each bin

        return

    def _vg_var_raw(self):
        """Calculate the variogram based on equal pairs in each bin.

        The last bin may have pairs less than specified because of the value of
        'nk' not dividing the 'tot_pairs_a' exactly. The ouput array is of the
        form: [mean distance in the bin, variogram value, number of pairs in
        the bin lowest lag distance in the bin, highest lag distance in
        the bin]
        """
        vg_h_list = []

        for m in range(0, self._tot_pairs_a, self.nk):
            n = self._pairs_arr[m:m + self.nk]
            if n.shape[0] == 0:
                print('n is zero')

            n_mid = np.mean(n[:, 2])  # (n[-1, 2] + n[0, 2]) / 2.
            vg_h_list.append((n_mid, self._evg(n[:, 0], n[:, 1], self.evg_name),
                              n.shape[0], n[0, 2], n[-1, 2]))

        self._vg_arr_raw = np.array(vg_h_list)

        self.vg_vg_arr = self._vg_arr_raw[:, 1]
        self.vg_h_arr = self._vg_arr_raw[:, 0]
        self.pairs_no = self._vg_arr_raw[:, 2]  # number of pairs in each bin
        return None

    def _lag_dist(self):
        """Calculate the lag distance for a given bin width.

        This function uses vg_const_h function by trying different bin widths.
        it iterates over different sizes and stops when the bin spacing is less
        than the minimum distance in all of the pairs. Minimum number of
        pairs along with the calculated bin width is appended to a list if
        minimum number of pairs are above 'nk'. The final bin width is the one
        that has the minimum bin spacing of all and greater than 'nk'.
        """
        self.ld = 0.95 * (self.max_dist - self.lh)

        while (self._itrs < self.h_itrs and self.ld > self.lh):
            self._vg_cnst_raw()
            bin_min_pairs = np.min(self._vg_arr_raw[:, 2])
            if bin_min_pairs >= self.nk:
                self._bin_min_pairs_list.append((self.ld, bin_min_pairs))

            self.ld = 0.95 * self.ld
            self._itrs += 1

        if len(self._bin_min_pairs_list) == 0:
            self._emsg = ("""
            \r\a Could not calculate the lag distance for which
            \rthe minimum number of pairs is: %d. Please specify it manually.
            """) % self.nk
            raise RuntimeError(self._emsg)
        else:
            self._bin_min_pairs_list = np.array(self._bin_min_pairs_list)
            return np.min(self._bin_min_pairs_list[:, 0])

    def nug_vg(self, h_arr, arg):
        # arg = (range, sill)
        nug_vg = np.full(h_arr.shape, arg[1])
        return nug_vg

    def sph_vg(self, h_arr, arg):
        # arg = (range, sill)
        a = (1.5 * h_arr) / arg[0]
        b = h_arr ** 3 / (2 * arg[0] ** 3)
        sph_vg = (arg[1] * (a - b))
        sph_vg[h_arr > arg[0]] = arg[1]
        return sph_vg

    def exp_vg(self, h_arr, arg):
        # arg = (range, sill)
        a = -3 * (h_arr / arg[0])
        exp_vg = (arg[1] * (1 - np.exp(a)))
        return exp_vg

    def lin_vg(self, h_arr, arg):
        # arg = (range, sill)
        lin_vg = arg[1] * (h_arr / arg[0])
        lin_vg[h_arr > arg[0]] = arg[1]
        return lin_vg

    def gau_vg(self, h_arr, arg):
        # arg = (range, sill)
        a = -3 * ((h_arr ** 2 / arg[0] ** 2))
        gau_vg = (arg[1] * (1 - np.exp(a)))
        return gau_vg

    def pow_vg(self, h_arr, arg):
        # arg = (range, sill)
        pow_vg = (arg[1] * (h_arr ** arg[0]))
        return pow_vg

    def hol_vg(self, h_arr, arg):
        # arg = (range, sill)
        hol_vg = np.zeros(h_arr.shape[0])  # do somethig about the zero
        idxs = np.where(h_arr > 0)
        a = (pi * h_arr[idxs]) / arg[0]
        hol_vg[idxs] = (arg[1] * (1 - (np.sin(a) / a)))
        return hol_vg

    def _max_dist_est(self):
        """To get the max_dist parameter
        """

        act_max_dist = self.pair_df['dist'].values.max(axis=0)
        self.lh = np.min(self.pair_df.values[:, 2])
        self._pairs_arr = np.array(self.pair_df.values, dtype=np.float64)

        if ((type(self.mdr) == float) or (type(self.mdr) == int)):
            # calculate maximum cutoff distance based on 'mdr'
            self.max_dist = (act_max_dist * self.mdr)

        elif self.mdr == 'auto':
            self.max_dist = act_max_dist
            evg_intvl = act_max_dist / 20.0
            ld_orig = self.ld
            self.ld = evg_intvl
            self._vg_cnst_raw()
            self.ld = ld_orig
            self.max_dist = None
            sil_tol = 0.025
            vg_diffs = np.round(np.ediff1d(self.vg_vg_arr[:int(0.8 * self.vg_vg_arr.shape[0])]) / self.variance, 2)
            min_diff_arg = np.argmin(vg_diffs)
            if vg_diffs[min_diff_arg] > sil_tol:
                warnings.warn('Could not calculate the max_dist parameter automatically. Using mdr = 0.5', RuntimeWarning, stacklevel=3)
            else:
                for dist_step, val in enumerate(vg_diffs[min_diff_arg + 1:]):
                    if ((abs(val) < sil_tol) and (val > 0)):
                        self.max_dist = self.vg_h_arr[min_diff_arg + 1 + dist_step]
                    else:
                        continue
        else:
            ValueError('Value of the parameter mdr is not specified. Use either auto, an int or a float value.')

        if self.max_dist is None:
            self.max_dist = (act_max_dist * 0.5)
        # calculate cutoff distance based on 'uh'
        if (self.uh is not None):
            # convert uh to distance in terms of cell size
            if ((type(self.cell_size) == int) or
                    (type(self.cell_size) == float)):
                self.uh = (self.uh / self.cell_size)

            if (self.uh < self.max_dist):
                self.max_dist = self.uh

        idx_clip = np.argmin(np.abs(self.pair_df.values[:, 2] - self.max_dist), axis=0)

        clip_ul = self.pair_df.values[:, 2][idx_clip]

        if (clip_ul - self.max_dist) > 0:
            pass
        else:
            clip_ul = self.max_dist

        if idx_clip != (self.pair_df.shape[0] - 1):
            self.pair_df['dist'] = np.clip(self.pair_df.values[:, 2], 0, clip_ul)
            self.pair_df['dist'].replace(clip_ul, np.nan, inplace=True)
            self.pair_df.dropna(inplace=True)

        self._pairs_arr = np.array(self.pair_df.values, dtype=np.float64)

        self._tot_pairs_a = self._pairs_arr.shape[0]  # reassign total pairs

    def _vg_calib(self, arg, *mix_vg_names):
        # arg = (range, sill)
        vg_mix = np.zeros(self.vg_h_arr.shape)  # to hold the variogram values

        for i, name in enumerate(mix_vg_names):
            sub_arg = arg[((i * 2)):((i * 2) + 2)]  # get vg params
            sub_vg = name[1](self.vg_h_arr, sub_arg)
            vg_mix += sub_vg

        if self.use_wts:
            wts = self.pairs_no / vg_mix ** 2
            sq_diff = np.square(self.vg_vg_arr - vg_mix)
            obj = np.sum(wts * sq_diff)
            return obj
        else:
            sq_diff = np.square(self.vg_vg_arr - vg_mix)
            obj = np.sum(sq_diff)
            return obj

    def _vg_get_vg(self):
        """To fit a mixed variogram to a given set of data.

        Brute force is used to get rough estimates of variogram parameters.
        Then these are used as initial guesses for the minimize function to get
        better values.

        Parameters:
            perm_r: A list, refers to the 'r' in the Permutations(nPr).
        """
        self.mix_vg_list = deepcopy(self.fit_vgs)  # names of vgs to use

        tot_theo_vgs = len(self.mix_vg_list)
        for idx, perm_r in enumerate(self.perm_r_list):
            if perm_r > tot_theo_vgs:
                self.perm_r_list[idx] = tot_theo_vgs
                warnings.warn(
                    'Value %d in the perm_r_list adjusted.' % (perm_r),
                    RuntimeWarning,
                    stacklevel=3)

        self.perm_r_list = np.array(np.unique(self.perm_r_list), dtype=int)

        all_mix_vg_ftns = {
                           'Nug': self.nug_vg,
                           'Sph': self.sph_vg,
                           'Exp': self.exp_vg,
                           'Lin': self.lin_vg,
                           'Gau': self.gau_vg,
                           'Pow': self.pow_vg,
                           'Hol': self.hol_vg
                           }
        if self.fil_nug_vg is not None:
            self.mix_vg_list.append(self.fil_nug_vg)

        min_obj = np.inf  # using AIC
        self.best_vg_names = []
        self.best_vg_params = []
        lb_thresh = 0.000000000001  # lower bound (used instead of zero)
        for n in self.perm_r_list:
            perm = permutations(self.mix_vg_list, int(n))

            skip_perm_list = []

            for o in perm:
                if o in skip_perm_list:
                    # if a given permutation exists then don't run further
                    continue

                cond_1_sel = (n == 1)
                cond_2_sel = ((n > 1) and (o[0] == self.fil_nug_vg))

                if (cond_1_sel or cond_2_sel):

                    mix_vg_names = []  # to hold the variogram names and ftns
                    bounds = []

                    for p, vg_name in enumerate(o):
                        mix_vg_names.append((vg_name, all_mix_vg_ftns[vg_name]))

                        if vg_name == 'Pow':
                            sub_bounds = [(lb_thresh, 2),
                                          (lb_thresh, 2 * self.variance)]

                        elif (cond_2_sel and (p == 0)):
                            sub_bounds = [(self.vg_h_arr[0], self.vg_h_arr[0]),
                                          (self.vg_vg_arr[0], self.vg_vg_arr[0])]

                        else:
#                             sub_bounds = [(self.vg_h_arr[0], self.max_dist),
#                                           (lb_thresh, 2 * self.variance)]

                            sub_bounds = [(self.vg_h_arr[0], 1e9),
                                          (lb_thresh, 2 * self.variance)]

                        [bounds.append(tuple(l)) for l in sub_bounds]

                    opt = differential_evolution(
                        self._vg_calib,
                        tuple(bounds),
                        tuple(mix_vg_names),
                        maxiter=self.opt_iters,
                        popsize=len(bounds) * 50,
                        polish=False)

                    assert opt.success, 'Optimization did not succeed!'

                else:
                    continue

                # Conditions for an optimization result to be selected:
                # 1: Obj ftn value less than the previous * fit_thresh
                # 2: Range of the variograms is in ascending order

                # minimize type optimization:
                rngs = opt.x[0::2].copy()
                sills = opt.x[1::2].copy()

                #  using Akaike Information Criterion (AIC) to select a model
                curr_AIC = ((self.tot_pts * np.log(opt.fun)) + (2 * opt.x.shape[0]))

                cond_1_fun = curr_AIC < min_obj * (1. - self.fit_thresh)
                cond_2_fun = np.all(np.where(np.ediff1d(rngs) < 0,
                                         False, True))

                if not cond_2_fun:
                    # flipping ranges and sills into correct order
                    sort_idxs = np.argsort(rngs)
                    rngs = rngs[sort_idxs]
                    sills = sills[sort_idxs]

#                    if nug_cond:
#                        sort_idxs_adj = np.delete(sort_idxs-1, 0, 0)
#                        adj_perm = np.array(o)[[sort_idxs_adj]]
#                    else:
                    adj_perm = np.array(o)[sort_idxs]

                    skip_perm_list.append(tuple(adj_perm))

                    mix_vg_names = np.array(mix_vg_names)[sort_idxs]

                cond_2_fun = np.all(np.where(np.ediff1d(rngs) < 0,
                                         False, True))

                prms = np.zeros((2 * rngs.shape[0]), dtype=np.float64)
                prms[0::2] = rngs
                prms[1::2] = sills

                if (cond_1_fun and cond_2_fun):
                    min_obj = curr_AIC
                    self.best_vg_names.append(mix_vg_names)
                    self.best_vg_params.append(prms)
                    self.n_best_objs.append(min_obj)

                if len(self.best_vg_names) > self.n_best:
                    del self.best_vg_names[0]
                    del self.best_vg_params[0]
                    del self.n_best_objs[0]

        h_arr = np.insert(self.vg_h_arr, 0, 0)

        self.vg_variance_list = []

        for o, n in enumerate(self.best_vg_names):
            vg_temp_str = ''  # final nested variogram string
            vg_mix = np.full(h_arr.shape, 0.0)
            vg_var = 0.0

            for m in range(len(n)):
                prms = self.best_vg_params[o][(m * 2): (m * 2 + 2)]
                # rng, sill = prms
                sill = prms[0]

#                 cond_sill = (round(sill, 6) == 0.0)
#                 if cond_sill:
#                     continue

                vg_var += sill

                vg_mix += n[m][1](h_arr, prms)
                vg_temp_str += (' + %0.5f %s(%0.5f)' % (prms[1], n[m][0],
                                prms[0]))

            self.vg_variance_list.append(vg_var)
            self.vg_str_list.append(vg_temp_str[3:])
            self.vg_fit.append(np.append(h_arr[np.newaxis].T,
                                         vg_mix[np.newaxis].T, axis=1))
        return

    def fit(self):
        """Perform calculations to get a mixed variogram fitted to the data

        Call this function only for getting a mixed variogram, no need to call
        other functions.
        """
        self._max_dist_est()

        if self.typ == 'cnst':
            self._vg_cnst_raw()
        elif self.typ == 'var':
            self._vg_var_raw()
        else:
            raise ValueError("""The value of the parameter \'typ\' can be either
            \'cnst\' or \'var\'.""")

        self._vg_get_vg()
        if len(self.vg_fit) == 0:
            raise RuntimeError("""Couldn\'t fit any variograms.
                                  Try again with different parameters.""")
        return
