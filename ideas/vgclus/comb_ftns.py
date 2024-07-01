'''
@author: Faizan3800X-Uni

Dec 11, 2020

4:15:24 PM
'''

from math import pi
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from spinterps.misc import get_theo_vg_vals

plt.ioff()


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
                popsize=len(bounds) * 10,
                polish=False)

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
