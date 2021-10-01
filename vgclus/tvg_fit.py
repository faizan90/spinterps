'''
@author: Faizan3800X-Uni

Jul 5, 2021

3:34:30 PM
'''
import timeit
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt;plt.ioff()
from pathos.multiprocessing import ProcessPool
from scipy.optimize import differential_evolution

from .evg_clus import CEVG
from ..misc import print_sl, print_el, get_theo_vg_vals, all_mix_vg_ftns


class TVGFit(CEVG):

    '''
    Fit theoretical variograms to the empirical ones.

    The way to fit the theoretical variograms is described in the
    fit_theoretical_variograms method. The outputs of that method are
    also described there.

    To get the theoretical variograms the following sequence of
    method calls is required:
    - set_data
    - set_empirical_variogram_clustering_parameters
    - set_theoretical_variogram_parameters
    - verify
    - cluster_evgs
    - fit_theoretical_variograms

    Last updated: 2021-Sep-30
    '''

    def __init__(self, verbose=True):

        CEVG.__init__(self, verbose)

        self._tvgs_fit_vrnc_mult = 2

        self._tvgs_fit_figs_dir = None
        self._tvgs_fit_txts_dir = None

        self._tvgs_fit_dict = None

        self._tvgs_fit_prepare_flag = False
        self._tvgs_fit_verify_flag = False
        self._tvgs_fit_fitted_flag = False
        return

    def _get_assemble_ts(self, tvgs_dict):

        out_tvgs_ts_df = pd.Series(index=self._data_df.index, dtype=str)

        if self._sett_clus_cevg_type == 'months':
            for month_lab in tvgs_dict.keys():
                month_idxs = out_tvgs_ts_df.index.month == int(month_lab[1:])

                assert month_idxs.sum()

                out_tvgs_ts_df.loc[month_idxs] = tvgs_dict[month_lab]

        elif self._sett_clus_cevg_type == 'years':
            for year_lab in tvgs_dict.keys():
                year_idxs = out_tvgs_ts_df.index.year == int(year_lab[1:])

                assert year_idxs.sum()

                out_tvgs_ts_df.loc[year_idxs] = tvgs_dict[year_lab]

        elif self._sett_clus_cevg_type == 'manual':
            for manual_lab in tvgs_dict.keys():
                manual_idxs = (
                    self._sett_clus_cevg_ts.values == int(manual_lab[1:]))

                assert manual_idxs.sum()

                out_tvgs_ts_df.loc[manual_idxs] = tvgs_dict[manual_lab]

        elif self._sett_clus_cevg_type == 'none':
            for tvgs_str in tvgs_dict.values():
                out_tvgs_ts_df.iloc[:] = tvgs_str
                break

        else:
            raise ValueError(
                f'Unknown clus_type: {self._sett_clus_cevg_type}!')

        return out_tvgs_ts_df

    def _plot_tvg(
            self,
            label,
            vg_str,
            distances,
            evg_values,
            tdists,
            tvg_vals):

        plt.figure(figsize=(10, 7))

        plt.plot(
            distances,
            evg_values,
            label='empirical',
            lw=3,
            alpha=0.4,
            color='red')

        plt.plot(
            tdists,
            tvg_vals,
            label='theoretical',
            lw=1,
            alpha=0.6,
            color='blue')

        plt.legend()
        plt.xlabel('Distance')
        plt.ylabel('Semi-variogram')
        plt.title(f'{label}\n{vg_str}')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.savefig(
            str(self._tvgs_fit_figs_dir / f'{label}_tvg_evg.png'),
            bbox_inches='tight',
            dpi=200)

        plt.close()
        return

    def _tvg_obj_ftn(self, prms, *args):

        mix_vg_names, dists, vg_vals = args

        vg_mix = np.zeros_like(dists)  # to hold the variogram values

        for i, name in enumerate(mix_vg_names):
            sub_arg = prms[((i * 2)):((i * 2) + 2)]  # get vg params

            sub_vg = name[1](dists, sub_arg)

            vg_mix += sub_vg

        sq_diffs = (vg_vals - vg_mix) ** 2

        if self._sett_clus_tvg_apply_wts_flag and (dists[0] == 0):
            sq_diffs[1:] /= dists[1:] ** 0.5

        elif self._sett_clus_tvg_apply_wts_flag and (dists[0] > 0):
            sq_diffs /= dists ** 0.5

        else:
            assert not self._sett_clus_tvg_apply_wts_flag

        obj = sq_diffs.sum()

        assert np.isfinite(obj)

        return obj

    def _fit_tvg_single(self, args):

        label, (distances, evg_values) = args

        fit_idxs = distances <= self._sett_clus_tvg_max_range

        distances_fit = distances[fit_idxs]
        evg_vals_fit = evg_values[fit_idxs]

        perm_r_list = np.array(self._sett_clus_tvg_perm_sizes, dtype=int)

        min_obj = np.inf
        best_vg_name = ''
        best_vg_param = ''
        lb_thresh = 1e-8  # lower bound (used instead of zero)

        sub_bds_a = [
            (lb_thresh, 2),
            (lb_thresh, self._tvgs_fit_vrnc_mult * evg_values.max())]

        sub_bds_b = [
            (distances.min(), self._sett_clus_tvg_max_range),
            (lb_thresh, self._tvgs_fit_vrnc_mult * evg_values.max())]

        for perm_r in perm_r_list:
            vg_perms = permutations(self._sett_clus_tvg_tvgs, int(perm_r))

            skip_perm_list = []

            for vg_strs in vg_perms:
                if vg_strs in skip_perm_list:
                    # if a given permutation exists then don't run further
                    continue

                mix_vg_names = []  # to hold the variogram names and ftns
                bounds = []

                for i, vg_name in enumerate(vg_strs):
                    mix_vg_names.append((vg_name, all_mix_vg_ftns[vg_name]))

                    if vg_name == 'Pow':
                        sub_bounds = sub_bds_a

                    else:
                        sub_bounds = sub_bds_b

                    [bounds.append(tuple(l)) for l in sub_bounds]

                opt = differential_evolution(
                    self._tvg_obj_ftn,
                    tuple(bounds),
                    args=(mix_vg_names, distances_fit, evg_vals_fit),
                    maxiter=self._sett_clus_tvg_max_iters,
                    popsize=len(bounds) * 20,
                    polish=True)

                assert opt.success, 'Optimization did not succeed!'

                # Conditions for an optimization result to be selected:
                # 1: Obj ftn value less than the previous * fit_thresh
                # 2: Range of the variograms is in ascending order

                # minimize type optimization:
                rngs = opt.x[0::2].copy()
                sills = opt.x[1::2].copy()

                #  using Akaike Information Criterion (AIC) to select a model
                curr_AIC = (
                    (evg_vals_fit.size * np.log(opt.fun)) +
                    (2 * opt.x.shape[0]))

                cond_1_fun = curr_AIC < min_obj * (1. - 1e-2)

                cond_2_fun = np.all(
                    np.where(np.ediff1d(rngs) < 0, False, True))

                if not cond_2_fun:
                    # flipping ranges and sills into correct order
                    sort_idxs = np.argsort(rngs)
                    rngs = rngs[sort_idxs]
                    sills = sills[sort_idxs]

                    adj_perm = np.array(vg_strs)[sort_idxs]

                    skip_perm_list.append(tuple(adj_perm))

                    mix_vg_names = np.array(mix_vg_names)[sort_idxs]

                cond_2_fun = np.all(
                    np.where(np.ediff1d(rngs) < 0, False, True))

                prms = np.zeros((2 * rngs.shape[0]), dtype=np.float64)
                prms[0::2] = rngs
                prms[1::2] = sills

                if (cond_1_fun and cond_2_fun):
                    min_obj = curr_AIC
                    best_vg_name = mix_vg_names
                    best_vg_param = prms

        tvg_str = ''  # final nested variogram string

        for i in range(len(best_vg_name)):
            prms = best_vg_param[(i * 2): (i * 2 + 2)]

            tvg_str += (
                ' + %0.5f %s(%0.1f)' % (prms[1], best_vg_name[i][0], prms[0]))

        if tvg_str:
            tvg_str = tvg_str[3:]

        if self._vb and (not self._tvgs_fit_fitted_flag):
            print(f'Fitted {tvg_str} to {label}')

        assert tvg_str, 'No vg fitted!'

        tdists = np.concatenate(([0.0], distances))

        tvg_vals = get_theo_vg_vals(tvg_str, tdists)

        if self._sett_clus_figs_flag and (not self._tvgs_fit_fitted_flag):
            self._plot_tvg(
                label,
                tvg_str,
                distances,
                evg_values,
                tdists,
                tvg_vals)

        return (label, tvg_str, tdists, tvg_vals)

    def _plot_all_tvgs(self, ress):

        max_legend_vals = 15

        plt.figure(figsize=(10, 7))
        for (dist_lab, _, tdistances, tvg_values) in ress:

            plt.plot(
                tdistances,
                tvg_values,
                label=dist_lab,
                lw=1,
                alpha=0.6)

        if len(ress) <= max_legend_vals:
            plt.legend()

        plt.xlabel('Distance')
        plt.ylabel('Semi-variogram')

        plt.grid()
        plt.gca().set_axisbelow(True)

        out_fig_path = str(
            self._tvgs_fit_figs_dir / f'{self._cevg_pref}_tvgs_cmpr_all.png')

        plt.savefig(out_fig_path, bbox_inches='tight', dpi=200)

        plt.close()
        return

    def fit_theoretical_variograms(self):

        '''
        Fit theoretical variograms to empiricals based on the criteria
        set in the set_theoretical_variogram_parameters.

        The objective function here is the sum of the squared differences
        and the AIC. The AIC serves to limit overfitting when multiple
        variogram combinations are used instead of just one.

        The following outputs are created in the tvgs_fit_txts if they
        are arrays (.npy) or pd.Series (.csv) and tvgs_fit_figs if they
        are figures. The X represent clus_type acronym, same as that
        in cluster_evgs method's outputs.

        The variograms are of the format "A B(C)". Where, A is the sill,
        B is the label of variogram and C is the range.

        X_final_tvgs.csv: The theoretical variograms fitted to each cluster
            or step if no clustering was chose. The index (first column)
            is a combination of X and the cluster label. For example, it
            is M04 for April if clus_type was months; Y2001 for the year
            2000 if clus_type was years; A7 if clus_type was manual and the
            label in the time clus_ts was 7; for clus_type none, it should
            be N50 for the 50th vector (N is for none).
        X_final_tvgs_ts.csv: A series of theoretical variograms. The index is
            is the same as that of the data that was set in set_data.
        Y_tvg_evg.png: A fitted unique theoretical variogram Y along with its
            empirical variogram. Y in this case represents the cluster which
            may be like M05, Y1999, A7 depending on the settings.
        '''

        if self._vb:
            print_sl()

            print(
                'Fitting theoretical variograms to the clustered empirical '
                'variograms...')

            tvg_fit_beg_time = timeit.default_timer()

        assert self._cevg_fitted_flag
        assert self._tvgs_fit_verify_flag

        if not self._tvgs_fit_prepare_flag:
            TVGFit._TVGFit__prepare(self)
        #======================================================================

        data_df = self._data_df

        self._data_df = None

        mp_args_gen = (args for args in self._cevgs_dict.items())

        n_cpus = min(self._sett_clus_misc_n_cpus, len(self._cevgs_dict))

        if n_cpus == 1:
            ress = []
            for args in mp_args_gen:
                ress.append(self._fit_tvg_single(args))

        else:
            mp_pool = ProcessPool(n_cpus)
            mp_pool.restart(True)

            ress = list(mp_pool.uimap(self._fit_tvg_single, mp_args_gen))

            mp_pool.close()
            mp_pool.join()
            mp_pool = None

        self._tvgs_fit_dict = {}
        for res in ress:
            self._tvgs_fit_dict[f'{res[0]}'] = res[1]

        self._data_df = data_df
        #======================================================================

        out_txt_path = (
            self._tvgs_fit_txts_dir / f'{self._cevg_pref}_final_tvgs.csv')

        with open(out_txt_path, 'w') as vgs_hdl:
            vgs_hdl.write(f'label;tvg\n')

            for dist_lab, tvg_str in self._tvgs_fit_dict.items():
                vgs_hdl.write(f';'.join([dist_lab, tvg_str]))
                vgs_hdl.write('\n')
        #======================================================================

        out_txt_path = (
            self._tvgs_fit_txts_dir / f'{self._cevg_pref}_final_tvgs_ts.csv')

        tvgs_ts = self._get_assemble_ts(self._tvgs_fit_dict)

        tvgs_ts.to_csv(out_txt_path, sep=';')
        #======================================================================

        if self._sett_clus_figs_flag:
            self._plot_all_tvgs(ress)

        if self._vb:
            tvg_fit_end_time = timeit.default_timer()

            print(
                f'Done fitting theoretical variograms to the clustered '
                f'empirical variograms in '
                f'{tvg_fit_end_time - tvg_fit_beg_time:0.1f} seconds.')

            print_el()

        self._tvgs_fit_fitted_flag = True
        return

    def prepare(self):

        '''
        Prepare some intermediate variables before fitting the
        theoretical variograms to the empiricals. Must be called
        before calling fit_theoretical_variograms. It is called
        automatically if you forget.
        '''

        self._tvgs_fit_txts_dir = (
                self._sett_clus_misc_outs_dir / f'tvgs_fit_txts')

        self._tvgs_fit_txts_dir.mkdir(exist_ok=True)

        if self._sett_clus_figs_flag:
            self._tvgs_fit_figs_dir = (
                self._sett_clus_misc_outs_dir / f'tvgs_fit_figs')

            self._tvgs_fit_figs_dir.mkdir(exist_ok=True)

        self._tvgs_fit_prepare_flag = True
        return

    def verify(self):

        '''
        Verify if all the required inputs to get the theoretical
        variograms are set. Must be explicitly called before calling
        fit_theoretical_variograms.
        '''

        CEVG._CEVG__verify(self)

        assert self._cevg_verify_flag

        assert self._sett_clus_tvg_set_flag

        self._tvgs_fit_verify_flag = True
        return

    __verify = verify
    __prepare = prepare
