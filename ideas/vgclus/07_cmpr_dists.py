'''
@author: Faizan-Uni-Stuttgart

Nov 20, 2020

10:07:40 AM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\cluster_vg_tests\ppt')

    os.chdir(main_dir)

    in_data_files = [
        Path(r'each_day_vg/ts_EDK.csv'),
        Path(r'monthly_cluster_vg/ts_EDK.csv'), ]

    data_labels = ['EDK', 'EDKC']

    aggs = np.arange(1, 10 + 1, 1, dtype=np.int64)

    # First_rounded than threshold applied.
    round_vals = 1
    min_val_thresh = 0.01

    sep = ';'

    beg_time = '1971-01-01 00:00:00'
    end_time = '2010-12-31 23:59:00'

    fig_size = (15, 7)

    dpi = 200

    fig_xlabel = 'Precipitation (mm)'
    fig_ylabel = 'Probability (-)'

    replace_nan_with_zero_flag = True

    out_dir = Path(f'cmpr_figs__dists_edk')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    rows = 2
    cols = len(in_data_files)

    for agg in aggs:
        axes = plt.subplots(
            rows,
            cols,
            squeeze=False,
            figsize=fig_size,
            sharex=True,
            sharey=True)[1]

        max_val = -np.inf
        row = 0
        for i in range(len(in_data_files)):
            print('Going through:', in_data_files[i])

            if in_data_files[i].suffix == '.csv':
                data_df = pd.read_csv(in_data_files[i], sep=sep, index_col=0)
    #             data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

            elif in_data_files[i].suffix == '.pkl':
                data_df = pd.read_pickle(in_data_files[i])

            else:
                raise NotImplementedError(
                    f'Unknown extension of in_data_file: {in_data_files[i].suffix}!')

            data_df = data_df.loc[beg_time:end_time]

#             if i == 0:
#                 data_df.dropna(axis=1, how='any', inplace=True)
#
#             else:
#                 data_df.dropna(axis=0, how='any', inplace=True)

            if replace_nan_with_zero_flag:
                data_df.replace(float('nan'), 0.0, inplace=True)

            assert np.all(np.isfinite(data_df.values))

            data_df = data_df.round(round_vals)

            data_df[data_df <= min_val_thresh] = 0.0

            data_df = data_df.rolling(agg, axis=0).sum().iloc[agg - 1:,:]

            data_df[data_df <= min_val_thresh] = min_val_thresh

            data_max = np.max(data_df.values)
            if data_max > max_val:
                max_val = data_max

            probs_vals_all = []
            for stn in data_df:
                stn_vals = np.sort(data_df[stn].values)

                probs = rankdata(stn_vals, method='max').astype(float)
                probs /= (data_df.shape[0] + 1)

                probs_vals_all.extend((stn_vals, probs))

                axes[row, i].semilogx(
                    stn_vals,
                    probs,
                    lw=1,
                    alpha=0.1,
                    color='k')

            probs_vals_all = np.array(probs_vals_all)

            med_vals = np.median(
                probs_vals_all[0::2], axis=0)

            med_probs = np.median(
                probs_vals_all[1::2], axis=0)

            axes[row, i].semilogx(
                med_vals,
                med_probs,
                lw=2,
                alpha=0.75,
                color='red',
                label='median')

            axes[row, i].set_title(data_labels[i])

            axes[row, i].legend()
            axes[row, i].grid()
            axes[row, i].set_axisbelow(True)

            if i == 0:
                axes[row, i].set_ylabel(fig_ylabel)

            axes[1, 1].semilogx(
                med_vals,
                med_probs,
                lw=2,
                alpha=0.5,
                label=f'{data_labels[i]}')

        row = 1
        for i in range(len(in_data_files)):
            if i == 1:
                axes[row, i].legend()
                axes[row, i].grid()
                axes[row, i].set_axisbelow(True)
                axes[row, i].set_xlabel(fig_xlabel)

                continue

            axes[row, i].set_axis_off()

        ax_max_str = str(max_val).split('.')[0]
        n_zeros = len(ax_max_str)

        axes[0, 0].set_xlim(10 ** (-round_vals), 10 ** n_zeros)

        out_fig_name = f'cmpr_dists_agg{agg}.png'

        plt.savefig(str(out_dir / out_fig_name), bbox_inches='tight', dpi=dpi)
        plt.close()

#         plt.show()

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
