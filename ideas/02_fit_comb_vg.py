'''
@author: Faizan-Uni-Stuttgart

Dec 10, 2020

3:33:35 PM

'''
import os
import time
import timeit
from math import pi
from pathlib import Path
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from spinterps.misc import get_theo_vg_vals

plt.ioff()

DEBUG_FLAG = False


def nug_vg(h_arr, arg):

    # arg = (range, sill)
    nug_vg = np.full(h_arr.shape, arg[1])
    return nug_vg


def sph_vg(h_arr, arg):

    # arg = (range, sill)
    a = (1.5 * h_arr) / arg[0]
    b = h_arr ** 3 / (2 * arg[0] ** 3)
    sph_vg = (arg[1] * (a - b))
    sph_vg[h_arr > arg[0]] = arg[1]
    return sph_vg


def exp_vg(h_arr, arg):

    # arg = (range, sill)
    a = -3 * (h_arr / arg[0])
    exp_vg = (arg[1] * (1 - np.exp(a)))
    return exp_vg


def lin_vg(h_arr, arg):

    # arg = (range, sill)
    lin_vg = arg[1] * (h_arr / arg[0])
    lin_vg[h_arr > arg[0]] = arg[1]
    return lin_vg


def gau_vg(h_arr, arg):

    # arg = (range, sill)
    a = -3 * ((h_arr ** 2 / arg[0] ** 2))
    gau_vg = (arg[1] * (1 - np.exp(a)))
    return gau_vg


def pow_vg(h_arr, arg):

    # arg = (range, sill)
    pow_vg = (arg[1] * (h_arr ** arg[0]))
    return pow_vg


def hol_vg(h_arr, arg):

    # arg = (range, sill)
    hol_vg = np.zeros(h_arr.shape[0])  # do somethig about the zero
    idxs = np.where(h_arr > 0)
    a = (pi * h_arr[idxs]) / arg[0]
    hol_vg[idxs] = (arg[1] * (1 - (np.sin(a) / a)))
    return hol_vg


def vg_calib(prms, *args):

    # prms = (range, sill)

    mix_vg_names, dists, vg_vals, wt_by_dist_flag = args

    vg_mix = np.zeros_like(dists)  # to hold the variogram values

    for i, name in enumerate(mix_vg_names):
        sub_arg = prms[((i * 2)):((i * 2) + 2)]  # get vg params

        sub_vg = name[1](dists, sub_arg)

        vg_mix += sub_vg

    sq_diffs = (vg_vals - vg_mix) ** 2

    if wt_by_dist_flag:
        sq_diffs /= dists ** 0.5

    obj = sq_diffs.sum()

    assert np.isfinite(obj)

    return obj


def get_vg_fit_inputs(args):

    (in_dists_file,
     in_vg_vals_file,
     sep) = args

    with open(in_dists_file, 'r') as dists_hdl, \
         open(in_vg_vals_file, 'r') as vg_vals_hdl:

        for (dists_txt, vg_vals_txt) in zip(
            dists_hdl.readlines(), vg_vals_hdl.readlines()):

            dist_strs = dists_txt.strip().split(sep)
            vg_val_strs = vg_vals_txt.strip().split(sep)

            dist_lab = dist_strs[0]
            vg_val_lab = vg_val_strs[0]

            assert dist_lab == vg_val_lab

            dists = np.array([float(dist_str) for dist_str in dist_strs[1:]])

            vg_vals = np.array(
                [float(vg_val_str) for vg_val_str in vg_val_strs[1:]])

            assert dists.size == vg_vals.size

            yield (dist_lab, dists, vg_vals)

    return


def get_vg(args):

    (dists,
     vg_vals,
     dist_lab,
     mix_vg_list,
     perm_r_list,
     opt_iters,
     wt_by_dist_flag,
     plot_flag,
     fig_size,
     out_dir,
     plt_at_zero_dist_flag,
     max_fit_dist) = args

    fit_idxs = dists <= max_fit_dist

    dists_fit = dists[fit_idxs]
    vg_val_fit = vg_vals[fit_idxs]

    perm_r_list = np.array(np.unique(perm_r_list), dtype=int)
    perm_r_list = perm_r_list[perm_r_list <= len(mix_vg_list)]

    all_mix_vg_ftns = {
       'Nug': nug_vg,
       'Sph': sph_vg,
       'Exp': exp_vg,
       'Lin': lin_vg,
       'Gau': gau_vg,
       'Pow': pow_vg,
       'Hol': hol_vg
       }

    min_obj = np.inf
    best_vg_name = ''
    best_vg_param = ''
    lb_thresh = 1e-8  # lower bound (used instead of zero)
    max_dist_thresh = max(1e6, dists.max())
    var_multpr = 2

    for perm_r in perm_r_list:
        vg_perms = combinations(mix_vg_list, int(perm_r))

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
                    sub_bounds = [(lb_thresh, 2),
                                  (lb_thresh, var_multpr * vg_vals.max())]

                else:
                    sub_bounds = [
                        (dists.min(), max_dist_thresh),
                        (lb_thresh, var_multpr * vg_vals.max())]

                [bounds.append(tuple(l)) for l in sub_bounds]

            opt = differential_evolution(
                vg_calib,
                tuple(bounds),
                args=(mix_vg_names, dists_fit, vg_val_fit, wt_by_dist_flag),
                maxiter=opt_iters,
                popsize=len(bounds) * 50)

            assert opt.success, 'Optimization did not succeed!'

            # Conditions for an optimization result to be selected:
            # 1: Obj ftn value less than the previous * fit_thresh
            # 2: Range of the variograms is in ascending order

            # minimize type optimization:
            rngs = opt.x[0::2].copy()
            sills = opt.x[1::2].copy()

            #  using Akaike Information Criterion (AIC) to select a model
            curr_AIC = (
                (vg_vals.size * np.log(opt.fun)) + (2 * opt.x.shape[0]))

            cond_1_fun = curr_AIC < min_obj * (1. - 1e-2)
            cond_2_fun = np.all(np.where(np.ediff1d(rngs) < 0, False, True))

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

    vg_str = ''  # final nested variogram string

    for i in range(len(best_vg_name)):
        prms = best_vg_param[(i * 2): (i * 2 + 2)]

        vg_str += (
            ' + %0.5f %s(%0.1f)' % (prms[1], best_vg_name[i][0], prms[0]))

    if vg_str:
        vg_str = vg_str[3:]

    print(dist_lab, vg_str)

    assert vg_str, 'No vg fitted!'

    if plt_at_zero_dist_flag:
        theo_dists = np.concatenate(([0.0], dists))

    else:
        theo_dists = dists

    theo_vg_vals = get_theo_vg_vals(vg_str, theo_dists)

    if plot_flag:
        plt.figure(figsize=fig_size)

        plt.plot(
            dists,
            vg_vals,
            label='empirical',
            lw=3,
            alpha=0.4,
            color='red')

        plt.plot(
            theo_dists,
            theo_vg_vals,
            label='theoretical',
            lw=1,
            alpha=0.6,
            color='blue')

        plt.legend()
        plt.xlabel('Distance')
        plt.ylabel('Semi-variogram')
        plt.title(f'{dist_lab}\n{vg_str}')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.savefig(str(out_dir / f'{dist_lab}_vg.png'), bbox_inches='tight')

        plt.close()

    return (dist_lab, vg_str, theo_dists, theo_vg_vals)


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\variograms\comb_vg\ppt_no_zeros_1961_2015')

    os.chdir(main_dir)

    in_dists_file = Path(r'M_dists.csv')
    in_vg_vals_file = Path(r'M_vg_vals.csv')

    sep = ';'

    fig_size = (10, 7)

    # TODO: Norm vgs ftn.
    out_dir = Path('vgs_tesst')

    wt_by_dist_flag = False
    max_fit_dist = 100e3

    plt_at_zero_dist_flag = True
    max_legend_vals = 15
    theo_vgs = ['Sph', 'Exp', 'Gau']  # , 'Pow', 'Lin', 'Hol']
    vg_n_perms = [1, 2]
    opt_iters = 1000
    plot_flag = True

    n_cpus = 12

    out_dir.mkdir(exist_ok=True)

    vg_inputs_args = (in_dists_file, in_vg_vals_file, sep)

    vg_fit_input_gen = ((
        dists,
        vg_vals,
        dist_lab,
        theo_vgs,
        vg_n_perms,
        opt_iters,
        wt_by_dist_flag,
        plot_flag,
        fig_size,
        out_dir,
        plt_at_zero_dist_flag,
        max_fit_dist)
        for (dist_lab, dists, vg_vals) in get_vg_fit_inputs(vg_inputs_args)
        )

    if n_cpus == 1:
        ress = []
        for vg_inputs in vg_fit_input_gen:
            ress.append(get_vg(vg_inputs))

    else:
        mp_pool = Pool(n_cpus)

        ress = list(mp_pool.imap_unordered(get_vg, vg_fit_input_gen))

    with open(out_dir / f'vgs.csv', 'w') as vgs_hdl:
        vgs_hdl.write(f'label{sep}vg\n')
        for dist_lab, vg_str, *_ in ress:
            vgs_hdl.write(f'{sep}'.join([dist_lab, vg_str]))
            vgs_hdl.write('\n')

    plt.figure(figsize=fig_size)
    for (dist_lab, vg_str, theo_dists, theo_vg_vals) in ress:

        plt.plot(
            theo_dists,
            theo_vg_vals,
            label=dist_lab,
            lw=1,
            alpha=0.6)

    if len(ress) <= max_legend_vals:
        plt.legend()

    plt.xlabel('Distance')
    plt.ylabel('Semi-variogram')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.savefig(
        str(out_dir / f'cmpr_theo_vgs.png'), bbox_inches='tight')

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
