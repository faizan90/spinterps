'''
@author: Faizan3800X-Uni

Jul 6, 2021

9:16:42 AM
'''
import timeit

import numpy as np
from scipy.stats import rankdata
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt;plt.ioff()

from .tvg_fit import TVGFit
from ..cyth import OrdinaryKriging, copy_2d_arr_at_idxs
from ..misc import (
    print_sl, print_el, get_theo_vg_vals, disagg_vg_str, get_dist_mat)


class TVGSClus(TVGFit):

    '''
    Cluster the fitted theoretical variograms to fewer ones, if possible.

    The clustering works by iteratively fitting a theoretical variogram
    to the mean of all the theoretical variograms that do not belong to
    a cluster.

    The fitting criteria is the Kolmogorov-Smirnov test at a given
    confidence level. It is applied to the weights of the considered
    kriging weights of the considered theoretical variograms and the single
    one that represents them. If the distribution of weights pass the
    test, then it belongs to the new variogram. The rest are tested and
    the algorithm runs till there are no theoretical variograms remainging
    to cluster.

    Some more details about the fitting procedure and the format of the
    outputs is described in cluster_theoretical_variograms method.

    To cluster the theoretical variograms into fewer ones the following
    sequence of method calls is required:
    - set_data
    - set_empirical_variogram_clustering_parameters
    - set_theoretical_variogram_parameters
    - set_theoretical_variogram_clustering_parameters
    - verify
    - cluster_evgs
    - fit_theoretical_variograms
    - cluster_theoretical_variograms

    Last updated: 2021-Sep-30
    '''

    def __init__(self, verbose=True):

        TVGFit.__init__(self, verbose)

        self._tvgs_clus_figs_dir = None
        self._tvgs_clus_txts_dir = None

        self._tvgs_clus_prepare_flag = False
        self._tvgs_clus_verify_flag = False
        self._tvgs_clus_clustered_flag = False
        return

    def _get_mean_tvg(self, tvg_strs, distances):

        n_tvgs = len(tvg_strs)

        assert  n_tvgs

        if n_tvgs == 1:
            mean_tvg_str = tvg_strs[0]

        else:
            tvg_stat_vals = np.zeros((n_tvgs, distances.size))

            for j, vg_str in enumerate(tvg_strs):
                tvg_stat_vals[j,:] = get_theo_vg_vals(vg_str, distances)

            tvg_vals = tvg_stat_vals.mean(axis=0)

            assert distances.size == tvg_vals.size

            mean_tvg_str = self._fit_tvg_single(
                (None, (distances, tvg_vals)))[1]

        return mean_tvg_str

    def _get_srtd_krg_wts(self, tvg_str, rnd_pts):

        # First coordinates are xk, yk.

        ord_krg_cls = OrdinaryKriging(
            xi=rnd_pts[1:, 0],
            yi=rnd_pts[1:, 1],
            zi=np.zeros(rnd_pts.shape[0] - 1),
            xk=np.array([rnd_pts[0, 0]]),
            yk=np.array([rnd_pts[0, 1]]),
            model=tvg_str)

        ord_krg_cls.krige()

        wts = ord_krg_cls.lambdas.ravel()

        wts.sort()

        if self._sett_clus_ctvg_min_kg_abs_wt:
            abs_wts = np.abs(wts)

            abs_wts_sum = abs_wts.sum()

            rel_wts = abs_wts / abs_wts_sum

            zero_idxs = rel_wts <= self._sett_clus_ctvg_min_kg_abs_wt

            # If all are zero.
            if zero_idxs.sum() == zero_idxs.size:
                zero_idxs = np.zeros_like(zero_idxs, dtype=bool)

            non_zero_wts = wts[~zero_idxs]

            sclr = 1 / non_zero_wts.sum()

            assert zero_idxs.sum() < zero_idxs.size

            wts[zero_idxs] = 0.0
            wts[~zero_idxs] *= sclr

        assert (wts[1:] - wts[:-1]).min() >= 0.0, (wts[1:] - wts[:-1]).min()

        assert np.isclose(wts.sum(), 1.0)

        return wts

    def _plot_tvgs_clus(self):

        distances = np.linspace(
            0,
            self._get_tvgs_max_range(),
            self._sett_clus_ctvg_n_distances)

        plt.figure(figsize=(10, 7))

        leg_flag = True
        plotted_new_tvgs = []
        n_unq_old_tvgs = len(set(self._tvgs_fit_dict.values()))
        n_unq_new_tvgs = len(set(self._tvgs_clus_dict.values()))

        for label, old_tvg_str in self._tvgs_fit_dict.items():

            if leg_flag:
                label_a = f'old (n={n_unq_old_tvgs})'
                label_b = f'new (n={n_unq_new_tvgs})'
                leg_flag = False

            else:
                label_a = label_b = None

            plt.plot(
                distances,
                get_theo_vg_vals(old_tvg_str, distances),
                label=label_a,
                alpha=0.5,
                c='red',
                lw=2,
                zorder=10)

            new_tvg_str = self._tvgs_clus_dict[label]

            if new_tvg_str in plotted_new_tvgs:
                continue

            plt.plot(
                distances,
                get_theo_vg_vals(new_tvg_str, distances),
                label=label_b,
                alpha=0.5,
                c='blue',
                lw=1.5,
                zorder=20)

            plotted_new_tvgs.append(new_tvg_str)

        plt.legend()

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Distance')
        plt.ylabel('Semi-variogram')

        out_fig_path = (
            self._tvgs_clus_figs_dir /
            f'{self._cevg_pref}_tvgs_clus_cmpr_all.png')

        plt.savefig(out_fig_path, bbox_inches='tight', dpi=200)
        plt.close()
        return

    def _get_2d_arr_subset(
            self, arr, row_idxs, col_idxs):

        assert arr.ndim == 2

        assert row_idxs.ndim == 1
        assert col_idxs.ndim == 1

        assert arr.size
        assert row_idxs.size
        assert col_idxs.size

        assert row_idxs.min() >= 0
        assert row_idxs.max() < arr.shape[0]

        assert col_idxs.min() >= 0
        assert col_idxs.max() < arr.shape[1]

        subset_arr = np.full(
            (row_idxs.size, col_idxs.size), np.nan, dtype=np.float64)

        copy_2d_arr_at_idxs(arr, row_idxs, col_idxs, subset_arr)

        return subset_arr

    def _get_stn_cfgs(self, data_df):

        '''
        Select all stations that have valid values at a given step,
        select one at random from these. This will be the reference station.
        Select self._sett_clus_ctvg_max_nebs stations that a closest to it.
        Output a list of stations with the first station being the reference
        one.
        '''

        dists_mat = get_dist_mat(
            self._crds_df['X'].values, self._crds_df['Y'].values)

        stn_cfgs = []
        stn_hashes = []
        for i in range(data_df.shape[0]):
            step_stns = data_df.iloc[i,:].dropna().index.values

            if step_stns.size < 3:
                continue

            rand_stn = np.random.choice(step_stns)

            rand_stn_idx = self._crds_df.index.get_loc(rand_stn)

            step_stns_idxs = self._crds_df.index.get_indexer_for(step_stns)

            stn_dists = dists_mat[rand_stn_idx, step_stns_idxs]

            srted_stn_dist_idxs = np.argsort(stn_dists)

            assert stn_dists[srted_stn_dist_idxs[0]] == 0
            assert np.all(stn_dists[srted_stn_dist_idxs][1:] > 0)

            # One added because first does not count.
            n_step_stns = min(
                step_stns.size, self._sett_clus_ctvg_max_nebs + 1)

            fin_srted_stn_dist_idxs = srted_stn_dist_idxs[:n_step_stns]

            step_stns_rand = step_stns[fin_srted_stn_dist_idxs]

            stns_hash = hash(step_stns_rand.tobytes())

            if stns_hash in stn_hashes:
                continue

            stn_hashes.append(stns_hash)
            stn_cfgs.append(step_stns_rand)

        # First station is the reference one, the others are its neighbors.
        return stn_cfgs

    def _get_tvgs_max_range(self):

        max_range = -np.inf
        for tvg_str in self._tvgs_fit_dict.values():

            rng = max(disagg_vg_str(tvg_str)[2])

            if rng >= max_range:
                max_range = rng

        max_range = min(self._sett_clus_cevg_max_dist_thresh, max_range)

        return max_range

    def _get_clustered_tvgs(self):

        stn_cfgs = self._get_stn_cfgs(self._data_df)

        n_sims = len(stn_cfgs)

        if self._vb:
            print('Total scenarios to test per attempt:', n_sims)

        sim_print_idx = max(1, int(0.1 * n_sims))

        vg_clusters = {}

        remn_labels = list(self._tvgs_fit_dict.keys())
        remn_labels_pre = list(self._tvgs_fit_dict.keys())

        # In case when the clustering type is none.
        if len(remn_labels) == 1:
            vg_clusters[remn_labels[0]] = self._tvgs_fit_dict[remn_labels[0]]

            remn_labels = []

        ctr = 0
        print('')
        while len(remn_labels):
            if self._vb:
                print('Clustering attempt:', ctr + 1)

            tvg_strs = [
                self._tvgs_fit_dict[remn_label] for remn_label in remn_labels]

            n_tvg_strs = len(tvg_strs)

            tvg_test_fail_ct = {remn_label:0 for remn_label in remn_labels}

            stat_tvg_str = self._get_mean_tvg(
                tvg_strs, np.linspace(
                    0,
                    self._get_tvgs_max_range(),
                    self._sett_clus_ctvg_n_distances))

            if self._vb:
                print('Theoretical variogram for this attempt:', stat_tvg_str)

            if self._sett_clus_figs_flag:
                plt.figure()

                leg_flag = True

            for sim_idx in range(n_sims):
                if self._vb and sim_idx and (not (sim_idx % sim_print_idx)):
                    print('Scenario:', sim_idx)

                rnd_pts = self._crds_df.loc[stn_cfgs[sim_idx]].values
                n_rnd_pts = rnd_pts.shape[0] - 1  # The first station.

                krg_wts = np.full((n_tvg_strs, n_rnd_pts), np.nan)
                krg_probs = np.full((n_tvg_strs, n_rnd_pts), np.nan)

                for i, tvg_str in enumerate(tvg_strs):
                    krg_wts[i,:] = self._get_srtd_krg_wts(tvg_str, rnd_pts)

                    krg_probs[i,:] = rankdata(
                        krg_wts[i,:], method='average') / (n_rnd_pts + 1.0)

                assert np.all(np.isfinite(krg_wts))
                assert np.all(np.isfinite(krg_probs))

                stat_krg_wts = self._get_srtd_krg_wts(stat_tvg_str, rnd_pts)

                # KS test. N and M are same.
                d_nm = 1 / (stat_krg_wts.size ** 0.5)

                d_nm *= (-np.log(self._sett_clus_ctvg_ks_alpha * 0.5)) ** 0.5

                stat_probs = rankdata(
                    stat_krg_wts, method='average') / (n_rnd_pts + 1.0)

                stat_interp_ftn = interp1d(
                    np.unique(stat_krg_wts),
                    np.unique(stat_probs),
                    bounds_error=False,
                    assume_sorted=True,
                    fill_value=(0, 1))

                # Upper and lower bounds.
                ks_u_bds = stat_probs - d_nm
                ks_l_bds = stat_probs + d_nm

                for i, label in enumerate(remn_labels):
                    interp_probs = stat_interp_ftn(krg_wts[i,:])

                    max_d_nm = np.abs(interp_probs - krg_probs[i,:]).max()

                    if max_d_nm > d_nm:
                        tvg_test_fail_ct[label] += 1

                        if self._sett_clus_figs_flag:
                            if leg_flag:
                                lab_a = 'orig'
                                lab_b = 'fitt'

                            else:
                                lab_a = None
                                lab_b = None

                            plt.plot(
                                krg_wts[i,:],
                                krg_probs[i,:],
                                color='b',
                                lw=1,
                                alpha=0.5,
                                label=lab_a)

                            plt.plot(
                                krg_wts[i,:],
                                interp_probs,
                                color='k',
                                lw=1,
                                alpha=0.5,
                                label=lab_b)

                            leg_flag = False

                # Stop if all labels fail to pass the test atleast once.
                if all([value for value in tvg_test_fail_ct.values()]):
                    if self._vb:
                        print(
                            f'Failed at scenario {sim_idx + 1} out of '
                            f'{n_sims}!')

                    break

            if self._vb:
                print('Test fail counts:', tvg_test_fail_ct)

            if self._sett_clus_figs_flag:
                plt.plot(
                    stat_krg_wts,
                    ks_u_bds,
                    color='r',
                    lw=1.5,
                    alpha=0.75,
                    label='ks_bds')

                plt.plot(
                    stat_krg_wts,
                    ks_l_bds,
                    color='r',
                    lw=1.5,
                    alpha=0.75)

                plt.xlim(-1.1, +1.1)
                plt.ylim(-0.1, +1.1)

                plt.grid()
                plt.gca().set_axisbelow(True)

                plt.legend()

                plt.xlabel(f'Kriging weight')
                plt.ylabel('Probability')

                out_fig_path = (
                    self._tvgs_clus_figs_dir /
                    f'{self._cevg_pref}_ctvg_ks_lims_{ctr}.png')

                plt.savefig(str(out_fig_path), bbox_inches='tight', dpi=200)

                plt.close()

            remn_labels = []
            cluster_labels = []
            for label, test_fail_ct in tvg_test_fail_ct.items():
                if test_fail_ct:
                    remn_labels.append(label)

                else:
                    cluster_labels.append(label)

            if remn_labels == remn_labels_pre:
                for remn_label in remn_labels:
                    vg_clusters[remn_label] = self._tvgs_fit_dict[remn_label]

                remn_labels = []

            else:
                remn_labels_pre = [label for label in remn_labels]

            if len(cluster_labels):
                for label in cluster_labels:
                    vg_clusters[label] = stat_tvg_str

            if len(remn_labels) == 1:
                vg_clusters[remn_labels[0]] = (
                    self._tvgs_fit_dict[remn_labels[0]])

                remn_labels = []

            ctr += 1

            if self._vb:
                print('')

        return vg_clusters

    def cluster_theoretical_variograms(self):

        '''
        Cluster the previously fitted theoretical variograms in to fewer
        variograms (yes, clustering the clusters). The idea is that it is
        possible that even though different theoretical variograms are
        fitted to distinct theoretical variograms, the kriging weights
        that they result it may be very similar. How similar, depends on
        how different the variograms were to begin with but also the
        configuration of points that are used for interpolating at another
        point. Unlike inverse distance weighting, the configuration of
        points matters for Kriging. Google the screening effect of Kriging.

        KS test is used to test the similarity of the Kriging weights. The
        closer the weights of a theoretical variogram to that of a cluster
        variogram, the more its chances to get accepted. There is a couple
        of problems that must be addressed to do this correctly.

        The first problem:
        Empirical distributions of the weights are used for the KS test.
        The non-excceedence probabilites are derived from the ranks of
        values (rank divided by the numer of values plus one). The problem
        with ranks is that they change for difference in the sixteenth
        decimal place, something which, probably, does not matter. Weights
        close to zero are rounded to zero. This absolute weight threshold
        is min_abs_kriging_weight.

        The other problem:
        The configuration of points to use. This means the number of
        points to sample and which ones to sample. The number is set by
        the user by the max_nrst_nebs parameter. Then points are randomly
        chosen from all. This is done by going through each time step,
        taking all the points that have valid values and taking a random
        sample from those. The minimum threshold is that the number of
        points be greater than 2 and max_nrst_nebs at maximum.
        A random point among these is then taken as the origin. This is
        point for which the Kriging weights are computed.

        The following ouputs are produced. These are similar to the ones
        produced by the fit_theoretical_Vvriograms method. Figures are
        saved to the directory tvgs_clus_figs while the rest to
        tvg_clus_txts. X is the acronym for clus_type (see other fitting
        methods for what it means).

        X_final_tvgs_clus.csv: The final variograms that are supposed to
            cluster the previous theoretical variograms. They may be the
            same as previous ones because of clustering being not possible.
            The clustered variograms replace the theoretical variograms.
            The labels for the clusters remain the same but one variogram
            may be used for multiple clusters from the outputs
            fit_theoretical_variograms. The file format is similar to
            X_final_tvgs.csv.
        X_final_tvgs_clus_ts.csv: The series of variograms. Here the
            format is similar to that of X_final_tvgs_ts.csv. If the clustering
            of theoretical variograms is successful, multiple theoretical
            variograms are
        '''

        if self._vb:
            print_sl()

            print('Clustering theoretical variograms...')

            ctvg_beg_time = timeit.default_timer()

        assert self._tvgs_fit_fitted_flag
        assert self._tvgs_clus_verify_flag

        if not self._tvgs_clus_prepare_flag:
            TVGSClus._TVGSClus__prepare(self)
        #======================================================================

        self._tvgs_clus_dict = self._get_clustered_tvgs()

        out_txt_path = (
            self._tvgs_clus_txts_dir /
            f'{self._cevg_pref}_final_tvgs_clus.csv')

        with open(out_txt_path, 'w') as vgs_hdl:
            vgs_hdl.write(f'label;ctvg\n')
            for label, new_tvg_str in self._tvgs_clus_dict.items():
                vgs_hdl.write(f';'.join([label, new_tvg_str]))
                vgs_hdl.write('\n')

                if self._vb:
                    old_tvg_str = self._tvgs_fit_dict[label]
                    print(f'For {label}: {old_tvg_str} -> {new_tvg_str}')
        #======================================================================

        out_txt_path = (
            self._tvgs_clus_txts_dir /
            f'{self._cevg_pref}_final_tvgs_clus_ts.csv')

        tvgs_ts = self._get_assemble_ts(self._tvgs_clus_dict)

        tvgs_ts.to_csv(out_txt_path, sep=';')
        #======================================================================

        if self._sett_clus_figs_flag:
            self._plot_tvgs_clus()

        if self._vb:
            ctvg_end_time = timeit.default_timer()

            print(
                f'Done clustering theoretical variograms in '
                f'{ctvg_end_time - ctvg_beg_time:0.1f} seconds.')

            print_el()

        self._tvgs_clus_clustered_flag = True
        return

    def prepare(self):

        self._tvgs_clus_txts_dir = (
                self._sett_clus_misc_outs_dir / f'tvgs_clus_txts')

        self._tvgs_clus_txts_dir.mkdir(exist_ok=True)

        if self._sett_clus_figs_flag:
            self._tvgs_clus_figs_dir = (
                self._sett_clus_misc_outs_dir / f'tvgs_clus_figs')

            self._tvgs_clus_figs_dir.mkdir(exist_ok=True)

        self._tvgs_clus_prepare_flag = True
        return

    def verify(self):

        TVGFit._TVGFit__verify(self)

        assert self._tvgs_fit_verify_flag

        assert self._sett_clus_ctvg_set_flag

        self._tvgs_clus_verify_flag = True
        return

    __verify = verify
    __prepare = prepare
