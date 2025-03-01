'''
@author: Faizan3800X-Uni

Jul 5, 2021

9:43:19 AM
'''
import timeit
from math import factorial

import numpy as np
import matplotlib.pyplot as plt; plt.ioff()
from pathos.multiprocessing import ProcessPool

from ..misc import print_sl, print_el
from .settings import VGCSettings as CS


class CEVG(CS):

    '''
    Fit empirical variograms to clouds of points.

    The way to do this is specified in the
    set_empirical_variogram_clustering_parameters of the parent class.
    The main idea is that instead of taking on vector of points,
    multiple can be taken to have a more robust cloud of points. An
    empirical variogram is fitted to these then. FOr example, consider
    temperature that is observed at multiple locations for the whole
    year daily. Instead of fitting a variogram to each day, all the days
    of each month can be taken and a variogram can be fitted to them
    as temperature is also a function of time and the points, are not
    moving of course. Then a single variogram represents the whole month,
    instead of 30 or so variograms. Clustering in this way reduces the
    effect of outliers on the emprical variogram. The main advantage is
    seen when the kriging matrix is inverted the number of variogram times
    rather then the number of steps.

    The description of outputs is given in the method cluster_evgs.

    The following sequence of method calls is required to get the empirical
    variograms:
    - set_data
    - set_empirical_variogram_clustering_parameters
    - verify
    - cluster_evgs

    Last updated: 2021-Sep-29
    '''

    def __init__(self, verbose=True):

        CS.__init__(self, verbose)

        self._cevg_pref = None
        self._cevg_args = None
        self._cevg_txts_dir = None
        self._cevg_figs_dir = None

        self._cevgs_dict = None

        self._cevg_prepare_flag = False
        self._cevg_verify_flag = False
        self._cevg_fitted_flag = False
        return

    def _get_evg(self, data_df):

        # For a given cluster of vectors.

        n_stns = data_df.shape[1]

        nnan_idxs_df = data_df.notna()

        n_combs = int(
            factorial(n_stns) / (factorial(2) * factorial(n_stns - 2)))

        dists = np.full(n_combs, np.nan)
        vg_vals = np.full(n_combs, np.nan)

        at_lst_one = False

        comb_ctr = -1
        for i, stn_i in enumerate(self._crds_df.index):
            stn_i_vals = data_df[stn_i].values.copy()
            stn_i_nnan_idxs = nnan_idxs_df[stn_i].values

            crds_i = self._crds_df.loc[stn_i,:].values

            for j, stn_j in enumerate(self._crds_df.index):

                if i <= j:
                    continue

                comb_ctr += 1

                crds_j = self._crds_df.loc[stn_j,:].values

                dist = ((crds_i - crds_j) ** 2).sum() ** 0.5

                if dist > self._sett_clus_cevg_max_dist_thresh:
                    continue

                stn_j_nnan_idxs = nnan_idxs_df[stn_j].values

                cmn_nnan_idxs = stn_i_nnan_idxs & stn_j_nnan_idxs
                cmn_n_idxs = cmn_nnan_idxs.sum()

                if cmn_n_idxs < self._sett_clus_cevg_min_vlds:
                    continue

                stn_j_vals = data_df[stn_j].values.copy()

                if False:
                    comb_vg_vals = 0.5 * (
                        (stn_i_vals[cmn_nnan_idxs] -
                         stn_j_vals[cmn_nnan_idxs]) ** 2)

                    if self._sett_clus_cevg_ignr_zeros_flag:
                        zero_vg_idxs = comb_vg_vals > 0

                        n_zero_vals = zero_vg_idxs.sum()

                        if not n_zero_vals: continue

                        comb_vg_vals = comb_vg_vals[zero_vg_idxs]

                    comb_vg_vals_stat = getattr(
                        np, self._sett_clus_cevg_evg_stat)(comb_vg_vals)

                else:
                    # From Sampson and Guttorp 1992.
                    stn_i_nrmd = stn_i_vals[cmn_nnan_idxs]
                    stn_j_nrmd = stn_j_vals[cmn_nnan_idxs]

                    if self._sett_clus_cevg_ignr_zeros_flag:
                        stn_i_zg_ixs = stn_i_nrmd > 0
                        stn_j_zg_ixs = stn_j_nrmd > 0

                        stn_ij_zg_ixs = stn_i_zg_ixs & stn_j_zg_ixs

                        if not stn_ij_zg_ixs.any(): continue

                        stn_i_nrmd = stn_i_nrmd[stn_ij_zg_ixs]
                        stn_j_nrmd = stn_j_nrmd[stn_ij_zg_ixs]

                    stn_i_nrmd -= stn_i_nrmd.mean()
                    stn_j_nrmd -= stn_j_nrmd.mean()

                    sii = (stn_i_nrmd ** 2).mean()
                    sjj = (stn_j_nrmd ** 2).mean()

                    sij = (stn_i_nrmd * stn_j_nrmd).mean()

                    comb_vg_vals_stat = sii + sjj - (2 * sij)

                vg_vals[comb_ctr] = comb_vg_vals_stat
                dists[comb_ctr] = dist

                at_lst_one = True

        assert at_lst_one, 'No pairs formed!'

        dists_nnan_idxs = np.isfinite(dists)

        assert dists_nnan_idxs.sum()

        dists = dists[dists_nnan_idxs]
        vg_vals = vg_vals[dists_nnan_idxs]

        assert np.all(np.isfinite(dists)) and np.all(np.isfinite(vg_vals))

        dists_sort_idxs = np.argsort(dists)

        dists_sorted = dists[dists_sort_idxs]
        vg_vals_sorted = vg_vals[dists_sort_idxs]

        return dists_sorted, vg_vals_sorted

    def _get_smoothed_arr(self, in_arr, win_size, smooth_ftn_type):

        n_vals = in_arr.shape[0]

        smooth_ftn = getattr(np, smooth_ftn_type)

        smoothed_arr = np.zeros(n_vals - win_size + 1)
        for i in range(smoothed_arr.size):
            smoothed_arr[i] = smooth_ftn(in_arr[i:i + win_size])

        return smoothed_arr

    def _get_postve_dfnt_evg(self, evg_vals):

        n_vals = evg_vals.size

        pd_vg_vals = np.full(n_vals, np.nan)
        pd_vg_vals[0] = evg_vals[0]

        for i in range(1, n_vals):
            pval = pd_vg_vals[i - 1]
            cval = evg_vals[i]

            if cval < pval:
                pd_vg_vals[i] = pval

            else:
                pd_vg_vals[i] = cval

        assert np.all(np.isfinite(pd_vg_vals))
        assert np.all((pd_vg_vals[1:] - pd_vg_vals [:-1]) >= 0)

        return pd_vg_vals

    def _get_simplified_evg(self, dists, vg_vals):

        assert vg_vals.size > 1, vg_vals.size

        vg_vals_diffs = vg_vals[1:] - vg_vals[:-1]

        take_idxs_raw = ~(np.isclose(vg_vals_diffs, 0))
        take_idxs = take_idxs_raw.copy()

        assert take_idxs.size

        for i in range(1, take_idxs_raw.size):
            if take_idxs_raw[i]:
                take_idxs[i - 1] = True

        take_idxs[-1] = True

        sdists = np.full(take_idxs.sum() + 1, np.nan)
        svg_vals = sdists.copy()

        sdists[0] = dists[0]
        svg_vals[0] = vg_vals[0]

        sdists[1:] = dists[1:][take_idxs]
        svg_vals[1:] = vg_vals[1:][take_idxs]

        assert np.all(np.isfinite(sdists)) & np.all(np.isfinite(svg_vals))

        return sdists, svg_vals

    def _plot_evg(
            self,
            distances,
            evg_vals,
            dists_smoothed,
            evg_vals_smoothed,
            idntfr):

        plt.figure(figsize=(13, 7))

        plt.scatter(distances, evg_vals, alpha=0.3, c='blue')
        plt.plot(dists_smoothed, evg_vals_smoothed, alpha=0.75, c='red')

        plt.yscale('log')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Distance')
        plt.ylabel('Semi-variogram')

        evg_vals_idxs = evg_vals > 1e-15

        if evg_vals_idxs.sum():
            min_evg_val = evg_vals[evg_vals_idxs].min()
            max_evg_val = evg_vals[evg_vals_idxs].max()

            min_evg_val_str = '%e' % min_evg_val
            max_evg_val_str = '%e' % max_evg_val

            ylim_lower = float(min_evg_val_str.split('e')[-1]) - 0
            ylim_upper = float(max_evg_val_str.split('e')[-1]) + 1

            plt.ylim(10 ** ylim_lower, 10 ** ylim_upper)

        ttl = (
            f'ignore_zero_vg_flag: {self._sett_clus_cevg_ignr_zeros_flag}, '
            f'+ve_dfnt_flag: {self._sett_clus_cevg_pstv_dfnt_flag}, '
            f'simplify_flag: {self._sett_clus_cevg_simplify_flag }\n'
            f'smoothing: {self._sett_clus_cevg_smoothing}, '
            f'n_thresh_vals: {self._sett_clus_cevg_min_vlds}, '
            f'clustering_type: {self._sett_clus_cevg_type}')

        if self._sett_clus_cevg_type == 'months':
            ttl += f', use_months: {idntfr} '

        elif self._sett_clus_cevg_type == 'years':
            ttl += f', use_years: {idntfr} '

        elif self._sett_clus_cevg_type == 'manual':
            ttl += f', use_manuals: {idntfr} '

        elif self._sett_clus_cevg_type == 'none':
            pass

        else:
            raise ValueError(
                f'Unknown clus_type: {self._sett_clus_cevg_type}!')

        plt.title(ttl, loc='left')

        out_fig_path = self._cevg_figs_dir / f'{self._cevg_pref}{idntfr}.png'

        plt.savefig(str(out_fig_path), dpi=200, bbox_inches='tight')
        plt.close()
        return

    def _cluster_evgs_single(self, args):

        data_df, idntfr = args

        assert self._cevg_verify_flag, f'Call verify first!'
        #======================================================================

        distances, evg_vals = self._get_evg(data_df)

        np.save(
            str(self._cevg_txts_dir /
                f'{self._cevg_pref}{idntfr}_evg_distances'),
            distances)

        np.save(
            str(self._cevg_txts_dir /
                f'{self._cevg_pref}{idntfr}_evg_values'),
            evg_vals)
        #======================================================================

        if self._sett_clus_cevg_smoothing > 1:
            n_smooth_vals = min(
                self._sett_clus_cevg_smoothing, distances.size)

        elif self._sett_clus_cevg_smoothing == 0:
            n_smooth_vals = 1

        else:
            n_smooth_vals = int(
                self._sett_clus_cevg_smoothing * distances.size)

        assert n_smooth_vals > 0

        dists_smoothed = self._get_smoothed_arr(
            distances, n_smooth_vals, 'mean')

        evg_vals_smoothed = self._get_smoothed_arr(
            evg_vals, n_smooth_vals, self._sett_clus_cevg_evg_stat)

        np.save(
            str(self._cevg_txts_dir /
                f'{self._cevg_pref}{idntfr}_smoothed_evg_distances'),
            dists_smoothed)

        np.save(
            str(self._cevg_txts_dir /
                f'{self._cevg_pref}{idntfr}_smoothed_evg_values'),
            evg_vals_smoothed)
        #======================================================================

        if self._sett_clus_cevg_pstv_dfnt_flag:
            evg_vals_smoothed = self._get_postve_dfnt_evg(evg_vals_smoothed)

            np.save(
                str(self._cevg_txts_dir /
                    f'{self._cevg_pref}{idntfr}_pstv_dfnt_evg_distances'),
                dists_smoothed)

            np.save(
                str(self._cevg_txts_dir /
                    f'{self._cevg_pref}{idntfr}_pstv_dfnt_evg_values'),
                evg_vals_smoothed)
        #======================================================================

        if self._sett_clus_cevg_simplify_flag:
            dists_smoothed, evg_vals_smoothed = self._get_simplified_evg(
                dists_smoothed, evg_vals_smoothed)

            np.save(
                str(self._cevg_txts_dir /
                    f'{self._cevg_pref}{idntfr}_simplify_evg_distances'),
                dists_smoothed)

            np.save(
                str(self._cevg_txts_dir /
                    f'{self._cevg_pref}{idntfr}_simplify_evg_values'),
                evg_vals_smoothed)
        #======================================================================

        # Plotting should be here, before norming.
        if self._sett_clus_figs_flag:
            self._plot_evg(
                distances,
                evg_vals,
                dists_smoothed,
                evg_vals_smoothed,
                idntfr)
        #======================================================================

        if self._sett_clus_cevg_norm_flag:
            norming_value = getattr(
                np, self._sett_clus_cevg_evg_stat)(evg_vals_smoothed)

            evg_vals_smoothed /= norming_value

            np.save(
                str(self._cevg_txts_dir /
                    f'{self._cevg_pref}{idntfr}_normed_evg_distances'),
                dists_smoothed)

            np.save(
                str(self._cevg_txts_dir /
                    f'{self._cevg_pref}{idntfr}_normed_evg_values'),
                evg_vals_smoothed)

            np.save(
                str(self._cevg_txts_dir /
                    f'{self._cevg_pref}{idntfr}_normed_norming_value'),
                np.array([norming_value]))
        #======================================================================

        if self._sett_clus_cevg_type in ('months', 'years', 'manual'):
            out_tup = (dists_smoothed, evg_vals_smoothed, idntfr)

        elif self._sett_clus_cevg_type == 'none':
            out_tup = (dists_smoothed, evg_vals_smoothed, idntfr)

        else:
            raise ValueError(
                f'Unknown clus_type: {self._sett_clus_cevg_type}!')

        if self._vb:
            print(
                f'Done finding empirical variogram for '
                f'{self._cevg_pref}{idntfr}')

        return out_tup

    def _plot_all_cevgs(self, ress):

        plt.figure(figsize=(13, 7))

        min_evg_val = +np.inf
        max_evg_val = -np.inf
        for res in ress:
            plt.plot(
                res[0],
                res[1],
                alpha=0.3,
                color='k')

            evg_vals = res[1]
            evg_vals_idxs = evg_vals > 1e-15

            if evg_vals_idxs.sum():
                cmin_evg_val = evg_vals[evg_vals_idxs].min()
                cmax_evg_val = evg_vals[evg_vals_idxs].max()

                if cmin_evg_val < min_evg_val:
                    min_evg_val = cmin_evg_val

                if cmax_evg_val > max_evg_val:
                    max_evg_val = cmax_evg_val

        plt.yscale('log')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Distance')
        plt.ylabel('Semi-variogram')

        if np.all(np.isfinite([min_evg_val, max_evg_val])):
            min_evg_val_str = '%e' % min_evg_val
            max_evg_val_str = '%e' % max_evg_val

            ylim_lower = float(min_evg_val_str.split('e')[-1]) - 0
            ylim_upper = float(max_evg_val_str.split('e')[-1]) + 1

            plt.ylim(10 ** ylim_lower, 10 ** ylim_upper)

        plt.savefig(
            str(self._cevg_figs_dir / f'{self._cevg_pref}_evgs_cmpr_all.png'),
            dpi=200,
            bbox_inches='tight')

        plt.close()
        return

    def cluster_evgs(self):

        '''
        Fit empirical variograms, by (may be) clustering various vectors
        based on the information passed to the method
        set_empirical_variogram_clustering_parameters.

        Numpy (.npy) data for various settings is saved in the directory
        cevgs_txt and cevgs_figs (if the figures flag was set).
        Following are the various outputs and their formats based on the
        flags set. X is the letter representing the type of clustering
        (clus_type in the set_empirical_variogram_clustering_parameters).
        It is M for months, Y for years, A for manual and N for none.
        Y represents the cluster e.g. Y = 3 means the month of March,
        if months are chosen as the clustering type or Y = 2000 means the
        year 2000 if clustering type is years or A = 10 means the
        that all vectors occuring whenvever clus_ts was equal to 10 or
        it is the index of the vector when clustering type is none.
        Each preceeding output below is applied to the corresponding
        outputs before it (except the first one of course).

        X_Y_evg_distances.npy : The distances at which empirical variogram
            values are computed.
        X_Y_evg_values.npy: The empirical variogram values. Each value is
            the mean/median/min/max of all the empirical variogram values
            (the terms 0.5 * (a - b)**2) for each pair. For each distance,
            there is one lumped empirical variogram value. Suppose,
            the mean of all the empirical variogram values for a pair of
            time series for the months of June in a time series.
        X_Y_smoothed_evg_distances.npy: The moving window mean distances.
            The size of the moving window depends on the value of the
            smoothing parameter.
        X_Y_smoothed_evg_values.npy: The moving window of the mean of the
            empirical varigram values. The size of the moving window
            depends on the value of the smoothing parameter.
        X_Y_pstv_dfnt_evg_distances : Distances of the monotonically
            increasing empirical variogram values if the
            pstv_dfnt_flag is set.
        X_Y_pstv_dfnt_evg_values: Monotonically increasing empirical
            variogram values corresponding to the previous distances
            if the pstv_dfnt_flag is set.
        X_Y_normed_evg_distances: Distances for normed empirical variogram
            values.
        X_Y_normed_evg_values: Normed empirical variogram values. These
            are computed by dividing the mean/median/min/max of the
            empirical variogram by the empirical variogram. This is done
            to bring all variogram to a similar scale. Such a norming
            makes sense for clustering variograms. Kriging weights are
            not affected by scaling the variogram by a constant.
            But it does affect the Estimation Variance. There, the normed
            variogram has to be rescaled by multiplying by whatever was
            used to norm it. Here, it is important to note that the proper
            scaling required to unnorm the later fitted theoretical
            variogram may not be the same scaling that was applied to norm
            the variogram. So, if estimation variance is to be computed
            then set the norm_flag to False.
        X_Y_normed_norming_value: The floating-point value used to norm the
            empirical variogram in the previous output.
        X_Y.png: The figure showing the variogram cloud and the empirical
            variogram. The Y-axis is logarithmic (IMHO, it looks better).
            Some information such as the state of various flags and the
            clustering type is shown in the title.
        '''

        if self._vb:
            print_sl()

            print('Computing clustered empirical variograms...')

            evg_beg_time = timeit.default_timer()

        assert self._cevg_verify_flag, 'Call verify first!'

        if not self._cevg_prepare_flag:
            CEVG._CEVG__prepare(self)
        #======================================================================

        data_df = self._data_df

        # So that it is not pickled.
        self._data_df = None

        mp_args_gen = (
            (data_df.iloc[args[0],:], args[1])
            for args in self._cevg_args)

        n_cpus = min(self._sett_clus_misc_n_cpus, len(self._cevg_args))

        if n_cpus == 1:
            ress = []
            for args in mp_args_gen:
                ress.append(self._cluster_evgs_single(args))

        else:

            mp_pool = ProcessPool(n_cpus)
            mp_pool.restart(True)

            ress = list(mp_pool.uimap(self._cluster_evgs_single, mp_args_gen))

            mp_pool.close()
            mp_pool.join()
            mp_pool = None

        final_distances_path = (
            self._cevg_txts_dir / f'{self._cevg_pref}_final_distances.csv')

        final_evg_vals_path = (
            self._cevg_txts_dir / f'{self._cevg_pref}_final_evg_values.csv')

        with open(final_distances_path, 'w') as dists_hdl:

            for res in ress:
                dists_hdl.write(f'{self._cevg_pref}{res[2]};')

                dists_hdl.write(f';'.join([f'{dist}' for dist in res[0]]))

                dists_hdl.write('\n')

        with open(final_evg_vals_path, 'w') as vg_vals_hdl:

            for res in ress:
                vg_vals_hdl.write(f'{self._cevg_pref}{res[2]};')

                vg_vals_hdl.write(
                    f';'.join([f'{vg_val}' for vg_val in res[1]]))

                vg_vals_hdl.write('\n')

        if self._sett_clus_figs_flag:
            self._plot_all_cevgs(ress)

        # Get it back for the rest of the program.
        self._data_df = data_df

        self._cevgs_dict = {}
        for res in ress:
            self._cevgs_dict[f'{self._cevg_pref}{res[2]}'] = (res[0], res[1])

        if self._vb:
            evg_end_time = timeit.default_timer()

            print(
                f'Done with computing clustered empirical variograms in '
                f'{evg_end_time-evg_beg_time:0.1f} seconds.')

            print_el()

        self._cevg_fitted_flag = True
        return

    def prepare(self):

        '''
        Prepare some intermediate data for the empirical variogram
        computation. Must be called before calling cluster_evg. It is
        called automatically, in case you forget.
        '''

        if self._sett_clus_cevg_type == 'months':
            cevg_pref = 'M'

            all_idntfrs = np.unique(self._data_df.index.month)

            time_idxs = []
            for idx in all_idntfrs:
                time_idxs.append(np.where(self._data_df.index.month == idx)[0])

                assert time_idxs[-1].size

            args = [(time_idxs[i], all_idntfrs[i])
                    for i in range(all_idntfrs.size)]

        elif self._sett_clus_cevg_type == 'years':
            cevg_pref = 'Y'

            all_idntfrs = np.unique(self._data_df.index.year)

            time_idxs = []
            for idx in all_idntfrs:
                time_idxs.append(np.where(self._data_df.index.year == idx)[0])

                assert time_idxs[-1].size

            args = [(time_idxs[i], all_idntfrs[i])
                    for i in range(all_idntfrs.size)]

        elif self._sett_clus_cevg_type == 'manual':
            cevg_pref = 'A'

            # Reindex the manual time series to match that of the data_df.
            diff_idxs = self._data_df.index.difference(
                self._sett_clus_cevg_ts.index)

            if self._vb and diff_idxs.size:
                print(
                    f'{diff_idxs.size} steps are different in data and '
                    f'manual ts!')

            self._sett_clus_cevg_ts = self._sett_clus_cevg_ts.reindex(
                self._data_df.index, fill_value=self._sett_clus_cevg_ts_nan)

            assert self._sett_clus_cevg_ts.shape[0] == self._data_df.shape[0]

            all_idntfrs = np.unique(self._sett_clus_cevg_ts.values)

            all_idntfrs = all_idntfrs[
                all_idntfrs != self._sett_clus_cevg_ts_nan]

            time_idxs = []
            for idx in all_idntfrs:
                time_idxs.append(
                    np.where(self._sett_clus_cevg_ts.values == idx)[0])

                assert time_idxs[-1].size

            args = [(time_idxs[i], all_idntfrs[i])
                    for i in range(all_idntfrs.size)]

        elif self._sett_clus_cevg_type == 'none':
            cevg_pref = 'N'
            args = [(np.arange(self._data_df.shape[0]), 0)]

        else:
            raise ValueError(
                f'Unknown clus_type: {self._sett_clus_cevg_type}!')

        # The length needs to be one, this is what some methods assume.
        assert len(cevg_pref) == 1, (f'This is not supposed to happen!')

        assert len(args), (f'This is not supposed to happen!')

        self._cevg_pref = cevg_pref
        self._cevg_args = args
        #======================================================================

        self._cevg_txts_dir = (
                self._sett_clus_misc_outs_dir / f'cevgs_txts')

        self._cevg_txts_dir.mkdir(exist_ok=True)

        if self._sett_clus_figs_flag:
            self._cevg_figs_dir = (
                self._sett_clus_misc_outs_dir / f'cevgs_figs')

            self._cevg_figs_dir.mkdir(exist_ok=True)

        self._cevg_prepare_flag = True
        return

    def verify(self):

        '''
        Verify all inputs set till this point. Must be called explicitly
        before calling cluster_evg.
        '''

        CS._VGCSettings__verify(self)

        assert self._data_df.shape[1] >= 3, (
            f'Input data must have at least three columns!')

        assert self._sett_clus_verify_flag, (
            f'Verify method of setting not called!')

        self._cevg_verify_flag = True
        return

    __verify = verify
    __prepare = prepare

