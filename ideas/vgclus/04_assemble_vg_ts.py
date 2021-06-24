'''
@author: Faizan-Uni-Stuttgart

Jun 23, 2021

12:14:32 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import pandas as pd
from parse import search

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms\vgs_cmpr_monthly\ppt_monthly_1971_2010__no_0s\vgs_M')
    os.chdir(main_dir)

    clustered_vgs_file = Path(r'clustered_vgs.csv')

    match_patt = 'M{month:d}'
    cluster_basis = 'months'

#     match_patt = 'Y{year:d}'
#     cluster_basis = 'years'

    sep = ';'

    beg_time = '1971-01-01'
    end_time = '2010-12-31'

    freq = 'D'

    out_vg_ts_path = Path(r'clustered_vg_ts.csv')

    #==========================================================================

    cluster_vgs_df = pd.read_csv(
        clustered_vgs_file, sep=sep, index_col=0, dtype=str)

    date_range = pd.date_range(beg_time, end_time, freq=freq)

    out_vgs_ts_df = pd.Series(index=date_range, dtype=str)

    vgs_dict = {}

    if cluster_basis == 'months':

        for idx in cluster_vgs_df.index:
            parse_res = search(match_patt, idx)

            assert parse_res is not None, idx

            month = parse_res['month']

            vgs_dict[month] = cluster_vgs_df.loc[idx].values[0]

            assert not pd.isna(vgs_dict[month]), f'Missing entry at {idx}!'

        for month in vgs_dict.keys():
            month_idxs = out_vgs_ts_df.index.month == month
            assert month_idxs.sum()

            out_vgs_ts_df.loc[month_idxs] = vgs_dict[month]

    elif cluster_basis == 'years':

        for idx in cluster_vgs_df.index:
            parse_res = search(match_patt, idx)

            assert parse_res is not None, idx

            year = parse_res['year']

            vgs_dict[year] = cluster_vgs_df.loc[idx].values[0]

            assert not pd.isna(vgs_dict[year]), f'Missing entry at {idx}!'

        for year in vgs_dict.keys():
            year_idxs = out_vgs_ts_df.index.year == year
            assert year_idxs.sum()

            out_vgs_ts_df.loc[year_idxs] = vgs_dict[year]

    else:
        raise ValueError(f'Unknown cluster_basis: {cluster_basis}!')

    assert not pd.isna(out_vgs_ts_df.values).sum(), 'Some values not written!'

    out_vgs_ts_df.to_csv(out_vg_ts_path, sep=sep)
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
