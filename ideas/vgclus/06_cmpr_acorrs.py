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

plt.ioff()

DEBUG_FLAG = False


def roll_real_2arrs(arr1, arr2, lag):

    assert isinstance(arr1, np.ndarray)
    assert isinstance(arr2, np.ndarray)

    assert arr1.ndim == 1
    assert arr2.ndim == 1

    assert arr1.size == arr2.size

    assert isinstance(lag, (int, np.int64))
    assert abs(lag) < arr1.size

    if lag > 0:
        # arr2 is shifted ahead
        arr1 = arr1[:-lag].copy()
        arr2 = arr2[+lag:].copy()

    elif lag < 0:
        # arr1 is shifted ahead
        arr1 = arr1[+lag:].copy()
        arr2 = arr2[:-lag].copy()

    else:
        pass

    assert arr1.size == arr2.size

    return arr1, arr2


def main():

    main_dir = Path(
        r'P:\cluster_vg_tests\ppt\monthly_cluster_vg')

    os.chdir(main_dir)

    in_data_files = [
        Path(r'T:\Synchronize_LDs\full_neckar_precipitation_interpolation\ts_EDK.csv'),
        Path(r'ts_EDK.csv'), ]

    data_labels = ['EDK', 'EDKC']

    lags = np.arange(1, 10 + 1, dtype=np.int64)

    ranks_flag = False

    sep = ';'

    fig_size = (15, 7)

    dpi = 200

    beg_time = '1971-01-01 00:00:00'
    end_time = '2010-12-31 23:59:00'

    replace_nan_with_zero_flag = True

    out_dir = Path(f'cmpr_figs__lag_corrs_edk')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    rows = 2
    cols = len(in_data_files)

    axes = plt.subplots(
        rows,
        cols,
        squeeze=False,
        figsize=fig_size,
        sharex=True,
        sharey=True)[1]

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

#         if i == 0:
#             data_df.dropna(axis=1, how='any', inplace=True)
#
#         else:
#             data_df.dropna(axis=0, how='any', inplace=True)

        if replace_nan_with_zero_flag:
            data_df.replace(float('nan'), 0.0, inplace=True)

        assert np.all(np.isfinite(data_df.values))

        if ranks_flag:
            data_df = data_df.rank(axis=0)

        lags_corrs_all = []
        for stn in data_df:
            stn_vals = data_df[stn].values.copy()

            lags_corrs = []
            for lag in lags:
                stn_no_lag_vals, stn_lag_vals = roll_real_2arrs(
                    stn_vals, stn_vals, lag)

                lag_corr = np.corrcoef(stn_no_lag_vals, stn_lag_vals)[0, 1]

                lags_corrs.append((lag, lag_corr))

            lags_corrs_all.extend(lags_corrs)

            lags_corrs = np.array(lags_corrs)

            axes[row, i].plot(
                lags_corrs[:, 0],
                lags_corrs[:, 1],
                lw=1,
                alpha=0.1,
                color='k')

        lags_corrs_all = np.array(lags_corrs_all)

        lags_corrs_med = []
        for lag in lags:
            med_lag_corr = np.median(
                lags_corrs_all[lags_corrs_all[:, 0] == lag, 1])

            lags_corrs_med.append((lag, med_lag_corr))

        lags_corrs_med = np.array(lags_corrs_med)

        axes[row, i].plot(
            lags_corrs_med[:, 0],
            lags_corrs_med[:, 1],
            lw=2,
            alpha=0.75,
            color='red',
            label='median')

        axes[row, i].set_title(data_labels[i])

        axes[row, i].legend()
        axes[row, i].grid()
        axes[row, i].set_axisbelow(True)

        if i == 0:
            axes[row, i].set_ylabel('Correlation (-)')

        axes[1, 1].plot(
            lags_corrs_med[:, 0],
            lags_corrs_med[:, 1],
            lw=2,
            alpha=0.75,
            label=f'{data_labels[i]}')

    row = 1
    for i in range(len(in_data_files)):
        if i == 1:
            axes[row, i].legend()
            axes[row, i].grid()
            axes[row, i].set_axisbelow(True)
            axes[row, i].set_xlabel('Lag (steps)')

            continue

        axes[row, i].set_axis_off()

    if ranks_flag:
        rank_lab = '__ranks'

    else:
        rank_lab = ''

    out_fig_name = f'lag_corrs{rank_lab}.png'

    plt.savefig(str(out_dir / out_fig_name), bbox_inches='tight', dpi=dpi)
    plt.close()

#     plt.show()

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
