# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
import os
import timeit
import time
from math import pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()


def nug_vg(h_arr, arg):
    # arg = (range, sill)
    nug_vg = np.full(h_arr.shape, arg[1])
    return nug_vg


def sph_vg(h_arr, arg):
    # arg = (range, sill)
    a = (1.5 * h_arr) / arg[0]
    b = h_arr**3 / (2 * arg[0]**3)
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
    lin_vg = arg[1] * (h_arr/arg[0])
    lin_vg[h_arr > arg[0]] = arg[1]
    return lin_vg


def gau_vg(h_arr, arg):
    # arg = (range, sill)
    a = -3 * ((h_arr**2/arg[0]**2))
    gau_vg = (arg[1] * (1 - np.exp(a)))
    return gau_vg


def pow_vg(h_arr, arg):
    # arg = (range, sill)
    pow_vg = (arg[1] * (h_arr**arg[0]))
    return pow_vg


def hol_vg(h_arr, arg):
    # arg = (range, sill)
    hol_vg = np.zeros(h_arr.shape[0])  # do somethig about the zero
    idxs = np.where(h_arr > 0)
    a = (pi * h_arr[idxs]) / arg[0]
#    try:
    hol_vg[idxs] = (arg[1] * (1 - (np.sin(a)/a)))
#    except:
#            import IPython.core.debugger
#            dbg = IPython.core.debugger.Pdb()
#            dbg.set_trace()
#            tre = 1
    return hol_vg


all_vg_ftns = {}
all_vg_ftns['Nug'] = nug_vg
all_vg_ftns['Sph'] = sph_vg
all_vg_ftns['Exp'] = exp_vg
all_vg_ftns['Lin'] = lin_vg
all_vg_ftns['Gau'] = gau_vg
all_vg_ftns['Pow'] = pow_vg
all_vg_ftns['Hol'] = hol_vg

if __name__ == '__main__':
    print('Started on %s \n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = os.path.join(r'P:\Synchronize\Python3Codes\krig_idw_nebs\test_ppt')

    in_vgs_file = (r'ppt_fitted_variograms__nk_1.000__evg_name_'
                   r'robust__ngp_5__h_typ_var.csv')

    sep = ';'
    time_fmt = '%Y-%m-%d'
    max_range = 300000.0
    thresh_var = 1e-2

    os.chdir(main_dir)

    in_vgs_df = pd.read_csv(in_vgs_file, sep=sep, index_col=0)

    in_vgs_df = in_vgs_df.iloc[:, 0]

    h_arr = np.arange(0, max_range, 1000)

    for model in in_vgs_df.values[:1000]:
        if not isinstance(model, str):
            continue

        vg_models = model.split('+')
        vg_arr = []

        var = 0.0
        for submodel in vg_models:
            submodel = submodel.strip()
            sill, vg = submodel.split(' ')
            vg, rng = vg.split('(')
            rng = rng.split(')')[0]

            vg_arr.append(all_vg_ftns[vg](h_arr,
                                          (max(1e-5, float(rng)),
                                           float(sill))))
            var += float(sill)

        if var < thresh_var:
            continue

        print(model)
        plt.plot(h_arr, np.sum(vg_arr, axis=0), alpha=0.1, color='b')

    plt.show()
    #    import IPython.core.debugger
    #    dbg = IPython.core.debugger.Pdb()
    #    dbg.set_trace()
    #    tre = 1


    STOP = timeit.default_timer()  # Ending time
    print(('\nDone with everything on %s. Total run time was'
           ' about %0.4f seconds' % (time.asctime(), STOP-START)))
