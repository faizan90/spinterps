'''
@author: Faizan-Uni-Stuttgart

Jun 21, 2021

12:49:53 PM

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
from scipy.stats import rankdata

plt.ioff()

DEBUG_FLAG = False

DEBUG_FLAG = True


def main():

    main_dir = Path(
        r'P:\cluster_vg_tests\ppt')

    os.chdir(main_dir)

    in_data_files = [
        Path(r'vg_clus_test__old_code/ts_OK.csv'),
        Path(r'vg_clus_test__new_code3/ts_OK.csv'), ]

    data_labels = ['OK', 'OKC']

    sep = ';'

    beg_time = '1971-01-01 00:00:00'
    end_time = '2010-12-31 23:59:00'

    fig_size = (15, 7)

    fig_xlabel = 'Time (months)'
    fig_ylabel = 'Precipitation (mm)'

    replace_nan_with_zero_flag = True

    out_dir = Path(f'old_code__cmpr_figs__tss_ref_stns_ok')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    data_dfs = []

    cols = []

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

        if replace_nan_with_zero_flag:
            data_df.replace(float('nan'), 0.0, inplace=True)

        data_dfs.append(data_df)

        cols.append(set(data_df.columns.values))

    cmn_cols = cols[0]
    for i in range(1, len(cols)):
        cmn_cols = cmn_cols.intersection(cols[i])

    assert len(cmn_cols), 'No common stations!'

    n_dfs = len(data_dfs)

    stats = [[] for i in range(n_dfs)]

    for col in cmn_cols:
        print(col)

        plt.figure(figsize=fig_size)

        strs = []
        for i in range(len(data_dfs)):
            getattr(data_dfs[i], col).plot(
                label=data_labels[i], alpha=0.5, lw=0.75)

            vals = data_dfs[i][col].values

            minv = vals.min()
            meanv = vals.mean()
            maxv = vals.max()
            varv = vals.var()

            strs.append(f'{data_labels[i]}: ')
            strs.append(f'Min.: {minv:0.2f}, ')
            strs.append(f'Mean.: {meanv:0.2f}, ')
            strs.append(f'Max.: {maxv:0.2f}, ')
            strs.append(f'Var.: {varv:0.2f}\n')

            stats[i].append([minv, meanv, maxv, varv])

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.title(f'Station: {col}\n' + ''.join(strs))

        plt.legend()

        plt.xlabel(fig_xlabel)
        plt.ylabel(fig_ylabel)

        plt.savefig(str(out_dir / f'ts_cmpr_{col}.png'), bbox_inches='tight')

        plt.close()

    stats = [np.array(stat) for stat in stats]

    labs = ['Min.', 'Mean', 'Max.', 'Var.']

    for i, lab in enumerate(labs):
        axes = plt.subplots(1, 1, squeeze=False)[1]

        for j in range(n_dfs):

            vals = stats[j][:, i]
            vals.sort()

            probs = rankdata(vals) / (vals.size + 1.0)

            axes[0, 0].plot(vals, probs, label=data_labels[j])

        axes[0, 0].grid()
        axes[0, 0].set_axisbelow(True)
        axes[0, 0].legend()

        axes[0, 0].set_xlabel(lab)
        axes[0, 0].set_ylabel('Probability')

        plt.savefig(str(out_dir / f'ts_stat_{lab}.png'), bbox_inches='tight')
        plt.close()
    return


if __name__ == '__main__':
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
