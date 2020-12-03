'''
@author: Faizan-Uni-Stuttgart

Nov 19, 2020

6:03:21 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr')
    os.chdir(main_dir)

    in_data_file = Path(r'precipitation.csv')
    in_crds_file = Path('precipitation_coords.csv')

    sep = ';'

    beg_time = '1991-01-01'
    end_time = '1991-12-30'
    time_fmt = '%Y-%m-%d'

    fig_size = (10, 6)
    fig_xlab = 'Distance (m)'
    fig_ylab = 'Variogram (-)'

    offset = 20000

    directions = [
        -np.pi, -np.pi * 0.75, -np.pi * 0.55, -np.pi * 0.25, 0.0,
        +np.pi, +np.pi * 0.75, +np.pi * 0.55, +np.pi * 0.25]

    out_dir = Path(r'anisotropic_vg_plots')

    out_dir.mkdir(exist_ok=True)

    data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
    data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

    data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)

    data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

    data_df = data_df.loc[beg_time:end_time]

    data_df.dropna(axis=1, how='any', inplace=True)

    crds_df = pd.read_csv(in_crds_file, sep=sep, index_col=0)[['X', 'Y']]

    crds_df = crds_df.loc[data_df.columns]

    probs_df = data_df.rank(axis=0) / (data_df.shape[0] + 1)

    norms_df = pd.DataFrame(
        data=norm.ppf(probs_df.values), columns=data_df.columns)

    ft_df = pd.DataFrame(
        data=np.fft.rfft(norms_df, axis=0), columns=data_df.columns)

#     mag_df = pd.DataFrame(data=np.abs(ft_df), columns=data_df.columns)

    phs_df = pd.DataFrame(data=np.angle(ft_df), columns=data_df.columns)

    phs_le_idxs = phs_df < 0

    phs_df[phs_le_idxs] = (2 * np.pi) + phs_df[phs_le_idxs]

    n_freqs = phs_df.shape[0]

    for i, direction in enumerate(directions):
        plt.figure(figsize=fig_size)

        for ref_stn in data_df.columns:
            print(ref_stn)
            dst_stns = data_df.columns.difference([ref_stn])

            # Distances.
            ref_x, ref_y = crds_df.loc[ref_stn, ['X', 'Y']].values

            dst_xs = crds_df.loc[dst_stns, 'X'].values
            dst_ys = crds_df.loc[dst_stns, 'Y'].values

            perp_dists = (ref_x - dst_xs) + (np.tan(direction) * (ref_y - dst_ys))
            perp_dists /= (1 + np.tan(direction) ** 2) ** 0.5

            up_idxs = (perp_dists < +offset) & (perp_dists > 0)
            dn_idxs = (perp_dists > -offset) & (perp_dists < 0)

            take_idxs = (up_idxs | dn_idxs)
#             other_idxs = ~take_idxs

            dst_stns = dst_stns[take_idxs]

            dst_xs = dst_xs[take_idxs]
            dst_ys = dst_ys[take_idxs]

#             plt.scatter(dst_xs[up_idxs], dst_ys[up_idxs], label='up', alpha=0.3, color='blue')
#             plt.scatter(dst_xs[dn_idxs], dst_ys[dn_idxs], label='dn', alpha=0.3, color='green')
#             plt.scatter(dst_xs[other_idxs], dst_ys[other_idxs], label='ot', alpha=0.1, color='black')
#             plt.scatter(ref_x, ref_y, label='ref', color='red', alpha=0.9)
#
#             plt.legend()
#
#             plt.show()
#
#             plt.draw()

#             angles < direction |

            dists = (ref_x - dst_xs) ** 2
            dists += (ref_y - dst_ys) ** 2

            dists **= 0.5

            phs_corrs = np.cos(
                phs_df.loc[:, ref_stn].values.reshape(-1, 1) -
                phs_df.loc[:, dst_stns].values).sum(axis=0) / n_freqs

            dists_vgs = np.array([dists, phs_corrs])

            sort_idxs = np.argsort(dists_vgs[0, :])

            dists_vgs = dists_vgs[:, sort_idxs]

            plt.scatter(
                dists_vgs[0, :],
                dists_vgs[1, :],
                color='k',
                alpha=0.1)

        plt.xlabel(fig_xlab)
        plt.ylabel(fig_ylab)

        plt.ylim(0.3, 1.0)

        plt.grid()

        plt.text(0, 1, f'Angle: {np.degrees(direction):0.2f}')

        plt.gca().set_axisbelow(True)

    #     plt.xlim(0, 50e3)
        plt.xlim(0, plt.xlim()[1])

        plt.savefig(out_dir / f'phs_corrs__dir_{i}.png', bbox_inches='tight')
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
