# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jan 6, 2024

7:11:17 PM

Keywords:

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    in_crds_path = Path(r'U:\TUM\projects\altoetting\tss\final_hydmod_data\ppt_1D_gkd_dwd_crds.csv')
    in_tss_path = Path(r'U:\TUM\projects\altoetting\tss\final_hydmod_data\ppt_1D_gkd_dwd_tss.pkl')

    # ref_col = 'DWD_D_P1036'
    # ref_col = 'DWD_D_P4480'
    # ref_col = 'DWD_D_P4333'
    # ref_col = 'DWD_D_P2302'
    # ref_col = 'DWD_D_P2877'
    # ref_col = 'DWD_D_P2162'
    # ref_col = 'DWD_D_P2847'
    # ref_col = 'DWD_D_P2088'

    # ref_cols = ['DWD_D_P1036', 'DWD_D_P4480', 'DWD_D_P2162', 'DWD_D_P2847', 'DWD_D_P2088', 'DWD_D_P3307']

    # in_crds_path = Path(r'U:\TUM\projects\altoetting\tss\final_hydmod_data\tem_1D_tg_gkd_dwd_crds.csv')
    # in_tss_path = Path(r'U:\TUM\projects\altoetting\tss\final_hydmod_data\tem_1D_tg_gkd_dwd_tss.pkl')

    # ref_col = 'GKD_D_T200011'
    #==========================================================================

    in_crds_path = Path(r'U:\dwd_meteo\daily\crds\daily_de\daily_fm_epsg32632.csv')
    in_tss_path = Path(r'U:\dwd_meteo\daily\dfs__merged_subset\daily_de_fm_Y1961_2022.pkl')

    ref_cols = ['FM5371', 'FM1987']
    #==========================================================================

    dist_buff_x = 20000
    dist_buff_y = 10000

    use_axis = 0

    min_valid_vals = 365 * 1
    min_valid_cols = 1
    #==========================================================================

    crds_df = pd.read_csv(in_crds_path, sep=';', index_col=0)
    crds_df.index = crds_df.index.astype(str)

    assert np.isfinite(crds_df.loc[:, ['X', 'Y']].values).all()

    tss_df = pd.read_pickle(in_tss_path)

    cmn_cols = crds_df.index.intersection(tss_df.columns)
    assert cmn_cols.size

    crds_df = crds_df.loc[cmn_cols]
    tss_df = tss_df.loc[:, cmn_cols]

    dist_lim = -np.inf

    for ref_col in ref_cols:

        print(ref_col)

        assert ref_col in cmn_cols

        ref_x, ref_y = crds_df.loc[ref_col, ['X', 'Y']]

        if use_axis == 0:

            buff_idxs = (
                (crds_df.loc[:, 'X'].values >= (ref_x - dist_buff_x)) &
                (crds_df.loc[:, 'X'].values <= (ref_x + dist_buff_x)))

        elif use_axis == 1:
            buff_idxs = (
                (crds_df.loc[:, 'Y'].values >= (ref_y - dist_buff_y)) &
                (crds_df.loc[:, 'Y'].values <= (ref_y + dist_buff_y)))

        else:
            raise NotImplementedError(use_axis)

        # buff_idxs[crds_df.index.get_loc(ref_col)] = False

        n_stns_in_buff = buff_idxs.sum()
        assert n_stns_in_buff

        ignr_cols = crds_df.index[~buff_idxs].tolist()

        print(n_stns_in_buff)

        for col in tss_df.columns:

            if tss_df[col].count() >= min_valid_vals:
                continue

            ignr_cols.append(col)

        assert ref_col not in ignr_cols

        assert (crds_df.shape[0] - len(ignr_cols)) > min_valid_cols

        print(tss_df.shape, crds_df.shape, len(ignr_cols))

        vg_cloud = get_vg_cloud(
            tss_df, ref_col, ref_x, ref_y, crds_df, ignr_cols, use_axis)

        plt.scatter(
            vg_cloud[:, 0], vg_cloud[:, 1], alpha=0.5, edgecolors='none')

        dist_lim = max(
            dist_lim,
            max(abs(vg_cloud[:, 0].min()), abs(vg_cloud[:, 0].max())))

    plt.ylim(0, None)
    plt.xlim(-dist_lim * 1.1, +dist_lim * 1.1)

    # plt.title(ref_col)

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.show()
    return


def get_vg_cloud(tss_df, ref_col, ref_x, ref_y, crds_df, ignr_cols, use_axis):

    vg_cloud = []
    for col in tss_df.columns:

        if col == ref_col: continue

        if col in ignr_cols: continue

        dist = (
            ((ref_x - crds_df.loc[col, 'X']) ** 2) +
            ((ref_y - crds_df.loc[col, 'Y']) ** 2)) ** 0.5

        if dist == 0: continue

        if use_axis == 0:
            if ref_y - crds_df.loc[col, 'Y'] < 0:
                dist *= -1

        elif use_axis == 1:
            if ref_x - crds_df.loc[col, 'X'] < 0:
                dist *= -1

        else:
            raise NotImplementedError(use_axis)

        vg_tss = ((tss_df.loc[:, ref_col].values -
                   tss_df.loc[:, ref_col].mean()) -
                  (tss_df.loc[:, col].values -
                   tss_df.loc[:, col].mean())) ** 2

        fnt_idxs = np.isfinite(vg_tss)

        fnt_idxs &= vg_tss > 0

        vg_tss = vg_tss[fnt_idxs]

        if not vg_tss.size: continue

        vg_val = np.mean(vg_tss)

        vg_cloud.append([dist, vg_val])

    vg_cloud = np.array(vg_cloud)

    return vg_cloud


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
