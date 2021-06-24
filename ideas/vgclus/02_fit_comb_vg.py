'''
@author: Faizan-Uni-Stuttgart

Dec 10, 2020

3:33:35 PM

'''
import os
import time
import timeit
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

from comb_ftns import get_vg

plt.ioff()

DEBUG_FLAG = False


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


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\variograms\vgs_cmpr_monthly\ppt_monthly_1971_2010__no_0s')

    os.chdir(main_dir)

    cmpr_id = 'Y'

    in_dists_file = Path(r'%s_dists.csv' % cmpr_id)
    in_vg_vals_file = Path(r'%s_vg_vals.csv' % cmpr_id)

    sep = ';'

    fig_size = (10, 7)

    # TODO: Norm vgs ftn.
    out_dir = Path('vgs_%s' % cmpr_id)

    # More weight give to values that are closer.
    wt_by_dist_flag = False

    # VG values after this distance not considered for optimization.
    max_fit_dist = 120e3

    plt_at_zero_dist_flag = True
    max_legend_vals = 15
    theo_vgs = ['Sph', 'Exp']  # , 'Gau']  # , 'Pow', 'Lin', 'Hol']
    vg_n_perms = [1, 2]
    opt_iters = 1000
    plot_flag = True

    n_cpus = 8

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
