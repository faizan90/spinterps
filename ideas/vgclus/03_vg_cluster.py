'''
@author: Faizan-Uni-Stuttgart

Dec 11, 2020

10:08:12 AM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.interpolate import interp1d

from comb_ftns import get_vg
from spinterps import OrdinaryKriging
from depth_funcs import gen_usph_vecs_norm_dist
from spinterps.misc import disagg_vg_str, get_theo_vg_vals

plt.ioff()

DEBUG_FLAG = True


def get_mean_vg(vg_strs_ser, dists):

    assert vg_strs_ser.size

    if vg_strs_ser.size == 1:
        mean_vg_str = vg_strs_ser.iloc[0]

    else:
        vgs = []
        vg_perm_rs = []
        vg_stat_vals = np.zeros((vg_strs_ser.size, dists.size))
        for j, vg_str in enumerate(vg_strs_ser):
            vg_stat_vals[j,:] = get_theo_vg_vals(vg_str, dists)

            for i, vg in enumerate(disagg_vg_str(vg_str)[1], start=1):
                if i not in vg_perm_rs:
                    vg_perm_rs.append(i)

                if vg in vgs:
                    continue

                vgs.append(vg)

        vg_vals = vg_stat_vals.mean(axis=0)

        # median might be a problem if vgs don't have the same rise rate.
#         vg_vals = np.median(vg_stat_vals, axis=0)

        assert dists.size == vg_vals.size

        get_vg_args = (
            dists,
            vg_vals,
            'mean_vg',
            vgs,
            vg_perm_rs,
            1000,
            False,
            False,
            None,
            None,
            False,
            dists[-1] + 1)

        mean_vg_str = get_vg(get_vg_args)[1]

    return mean_vg_str


def get_sorted_krg_wts(vg_str, rnd_pts, abs_thresh_wt):

    ord_krg_cls = OrdinaryKriging(
        xi=rnd_pts[:, 0],
        yi=rnd_pts[:, 1],
        zi=np.zeros(rnd_pts.shape[0]),
        xk=np.array([0.0]),
        yk=np.array([0.0]),
        model=vg_str)

    ord_krg_cls.krige()

    wts = ord_krg_cls.lambdas.ravel()

    wts.sort()

    if abs_thresh_wt:
        abs_wts = np.abs(wts)

        abs_wts_sum = abs_wts.sum()

        rel_wts = abs_wts / abs_wts_sum

        zero_idxs = rel_wts <= abs_thresh_wt

        non_zero_wts = wts[~zero_idxs]

        sclr = 1 / non_zero_wts.sum()

        assert zero_idxs.sum() < zero_idxs.size

        wts[zero_idxs] = 0.0
        wts[~zero_idxs] *= sclr

    assert (wts[1:] - wts[:-1]).min() >= 0.0

    assert np.isclose(wts.sum(), 1.0)

    return wts


def get_rnd_pts(n_pts, max_dist, n_dims=2):

    uvecs = gen_usph_vecs_norm_dist(n_pts, n_dims, 1)

    pts = (np.linspace(1 / n_pts, 1 - (1 / n_pts), n_pts) * max_dist
        ).reshape(-1, 1) * uvecs

#     mags_pts = (pts ** 2).sum(axis=1) ** 0.5

#     plt.scatter(pts[:, 0], pts[:, 1], alpha=0.005)
#
#     plt.gca().set_aspect('equal')
#     plt.show()

    return pts


def get_clustered_vgs(args):

    (vg_strs_ser_main,
     max_rng,
     n_fit_dists,
     n_sims,
     n_rnd_pts,
     abs_thresh_wt,
     ks_alpha,
     n_sel_thresh,
     max_nbr_dist) = args

    remn_labels = vg_strs_ser_main.index.tolist()

    vg_clusters = []

    ctr = 0
    while True:
        print('ctr:', ctr)

        vg_strs_ser = vg_strs_ser_main.loc[remn_labels].copy()

        vg_test_pass_ser = pd.Series(
            index=vg_strs_ser.index,
            data=np.zeros(vg_strs_ser.shape[0], dtype=int),
            dtype=float)

        print('vg_strs_ser:', vg_strs_ser)

        print('Fitting stat_vg...')
        stat_vg_str = get_mean_vg(
            vg_strs_ser, np.linspace(0, max_rng, n_fit_dists))

        print('Done fitting stat_vg:', stat_vg_str)

        plt.figure()
        for sim_idx in range(n_sims):
            print('Simulation:', sim_idx)

            rnd_pts = get_rnd_pts(n_rnd_pts, max_nbr_dist)

            krg_wts = np.full((vg_strs_ser.shape[0], n_rnd_pts), np.nan)
            krg_probs = np.full((vg_strs_ser.shape[0], n_rnd_pts), np.nan)
            for i, vg_str in enumerate(vg_strs_ser):
                krg_wts[i,:] = get_sorted_krg_wts(
                    vg_str, rnd_pts, abs_thresh_wt)

                krg_probs[i,:] = rankdata(
                    krg_wts[i,:], method='average') / (n_rnd_pts + 1.0)

            assert np.all(np.isfinite(krg_wts))
            assert np.all(np.isfinite(krg_probs))

    #         stat_krg_wts = np.mean(krg_wts, axis=0)
            stat_krg_wts = get_sorted_krg_wts(
                stat_vg_str, rnd_pts, abs_thresh_wt)

            # KS test. N and M are same.
            d_nm = 1 / (stat_krg_wts.size ** 0.5)

            d_nm *= (-np.log(ks_alpha * 0.5)) ** 0.5

            stat_probs = rankdata(
                stat_krg_wts, method='average') / (n_rnd_pts + 1.0)

            stat_interp_ftn = interp1d(
                np.unique(stat_krg_wts),
                np.unique(stat_probs),
                bounds_error=False,
                assume_sorted=True,
                fill_value=(0, 1))

            # upper and lower bounds.
            ks_u_bds = stat_probs - d_nm
            ks_l_bds = stat_probs + d_nm

            for i, label in enumerate(vg_strs_ser.index):
                interp_probs = stat_interp_ftn(krg_wts[i,:])

                max_d_nm = np.abs(interp_probs - krg_probs[i,:]).max()

                if max_d_nm > d_nm:
                    vg_test_pass_ser.loc[label] += 1

                    print(label)

    #                 for j in range(ks_l_bds.size):
    #                     acpt_flag = ks_u_bds[j] < interp_probs[j] < ks_l_bds[j]
    #
    #                     print(f'{krg_wts[i, j]:10.7f} | {ks_l_bds[j]:10.7f} | {interp_probs[j]:10.7f} | {ks_u_bds[j]:10.7f} | {acpt_flag} | {max_d_nm:10.7f}')

                    plt.plot(
                        krg_wts[i,:],
                        krg_probs[i,:],
                        color='b',
                        lw=1,
                        alpha=0.5)

                    plt.plot(
                        krg_wts[i,:],
                        interp_probs,
                        color='k',
                        lw=1,
                        alpha=0.5)

        print(vg_test_pass_ser)

        plt.plot(
            stat_krg_wts,
            ks_u_bds,
            color='r',
            lw=1.5,
            alpha=0.75)

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

        plt.xlabel(f'Kriging weight')
        plt.ylabel('Probability')

#         plt.show(block=False)
        plt.show()
        plt.close()

        remn_labels = vg_test_pass_ser.loc[vg_test_pass_ser > n_sel_thresh].index.tolist()
        cluster_labels = vg_test_pass_ser.loc[vg_test_pass_ser <= n_sel_thresh].index.tolist()

        vg_clusters.append([stat_vg_str, cluster_labels])

#         plt.close()
        if len(remn_labels) == 1:
            vg_clusters.append(
                (vg_strs_ser_main.loc[remn_labels[0]], remn_labels))

            remn_labels = []

        if not remn_labels:
            break

        ctr += 1
        print('\n\n')

    return vg_clusters


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\variograms\comb_vg\temp_1961_2015_with_zeros\vgs_CP')

    os.chdir(main_dir)

    # Something needed with an actual range.
    allowed_vgs = ['Sph', 'Exp']  # , 'Gau']

    in_vg_strs_file = Path('vgs.csv')

    sep = ';'

    # max_rng can be None or a float.
    # When None, then maximum range from all vgs is taken.
    max_rng = 250e3
    n_fit_dists = 50
    max_nbr_dist = 50e3

    n_rnd_pts = int(1e2)
    n_sims = int(1e2)

    ks_alpha = 0.99
    n_sel_thresh = 1000

    abs_thresh_wt = (1e-2)  # * n_rnd_pts

    out_fig_name = 'clustered_vgs.png'
    fig_size = (10, 7)
    out_vgs_sers_name = 'clustered_vgs.csv'

#     krg_wts_exp = 0.1

    vg_strs_ser_main = pd.read_csv(
        in_vg_strs_file, sep=sep, index_col=0, squeeze=True)

    if max_rng is None:
        max_rng = -np.inf
        for vg_str in vg_strs_ser_main:

            _, vgs, rngs = disagg_vg_str(vg_str)

            assert all([vg in allowed_vgs for vg in vgs])

            rng = max(rngs)

            if rng >= max_rng:
                max_rng = rng

    elif isinstance(max_rng, (int, float)):
        max_rng = float(max_rng)

    else:
        raise ValueError('Invalid max_rng:', max_rng)

    print('max_rng:', max_rng)

    cluster_args = (
        vg_strs_ser_main,
        max_rng,
        n_fit_dists,
        n_sims,
        n_rnd_pts,
        abs_thresh_wt,
        ks_alpha,
        n_sel_thresh,
        max_nbr_dist)

    vg_clusters = get_clustered_vgs(cluster_args)

    print('Done fitting.')
    print('Refitting...')
    theo_dists = np.linspace(0, max_rng, n_fit_dists)
    refit_vgs = []

    out_clustered_ser = pd.Series(
        index=vg_strs_ser_main.index, dtype=object)

    for vg_cluster in vg_clusters:
        print(vg_cluster)

        refit_vg_str = get_mean_vg(
            vg_strs_ser_main.loc[vg_cluster[1]], theo_dists)

        refit_vgs.append(refit_vg_str)

        print(vg_cluster[0], refit_vg_str)

        for vg_label in vg_cluster[1]:
            out_clustered_ser.loc[vg_label] = refit_vg_str

    out_clustered_ser.to_csv(out_vgs_sers_name, sep=sep)

    plt.figure(figsize=fig_size)
    leg_flag = True
    for vg_str in vg_strs_ser_main:

        if leg_flag:
            label = f'old(n={vg_strs_ser_main.size})'
            leg_flag = False

        else:
            label = None

        plt.plot(
            theo_dists,
            get_theo_vg_vals(vg_str, theo_dists),
            label=label,
            alpha=0.5,
            c='red')

    leg_flag = True
    for vg_str in refit_vgs:

        if leg_flag:
            label = f'new(n={len(refit_vgs)})'
            leg_flag = False

        else:
            label = None

        plt.plot(
            theo_dists,
            get_theo_vg_vals(vg_str, theo_dists),
            label=label,
            alpha=0.5,
            c='blue')

    plt.legend()

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Distance')
    plt.ylabel('Semi-variogram')

#     plt.show()

    plt.savefig(out_fig_name, bbox_inches='tight')
    plt.close()

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
