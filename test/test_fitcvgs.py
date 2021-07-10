'''
@author: Faizan-Uni-Stuttgart

Jul 5, 2021

2:18:04 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd

from spinterps import ClusteredVariograms

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\hydmod_de')
    os.chdir(main_dir)

    in_data_file = Path(r'P:\dwd_meteo\daily_de_buff_100km_tem__merged__daily_hourly_dwd__daily_ecad\daily_de_tn_Y1961_2020__merged_data.pkl')
    in_crds_file = Path(r'P:\dwd_meteo\daily_de_buff_100km_tem__merged__daily_hourly_dwd__daily_ecad\daily_de_tn_Y1961_2020__merged_crds.csv')

    in_manual_file = Path(r'P:\Synchronize\IWS\Testings\variograms\cp_time_ser_neckar_1910_2014.csv')
    manual_df_col = 'cp'
    manual_nan_val = 99

    out_dir = Path(r'test_fitcvgs_months')

    sep = ';'
    time_fmt = '%Y-%m-%d'

    beg_time = '1961-01-01'
    end_time = '1965-12-31'

    clus_type = 'months'

    # Minimum values to form a mean value per pair.
    n_min_valid_values = 10

    stns_min_dist_thresh = 100

    # Number of value to compute the smoothed empirical variogram.
    # Works as a running statistic.
    smoothing = 100

    max_dist_thresh = 5e5

    n_cpus = 12

    evg_stat = 'median'

    ignore_zeros_flag = True
    pstv_dfnt_flag = True
    simplify_flag = True
    norm_flag = True

    plot_figs_flag = True

    theoretical_variograms = ['Sph', 'Exp']
    permutation_sizes = [1, 2]
    max_opt_iters = 1000
    max_variogram_range = 1e7
    apply_wts_flag = True

    n_distance_intervals = 1000
    ks_alpha = 0.8
    min_abs_kriging_weight = 1e-2
    max_nrst_nebs = 50

    if in_data_file.suffix == '.csv':
        data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
        data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

    elif in_data_file.suffix == '.pkl':
        data_df = pd.read_pickle(in_data_file)

    else:
        raise NotImplementedError(
            f'Unknown file extension: {in_data_file.suffix}!')

    data_df = data_df.loc[beg_time:end_time]

#     data_df = data_df.iloc[:,:100]

    crds_df = pd.read_csv(in_crds_file, sep=sep, index_col=0)

    crds_df = crds_df.loc[:, ['X', 'Y']]

    if clus_type == 'manual':
        manual_ser = pd.read_csv(
            in_manual_file, sep=sep, index_col=0)[manual_df_col]

        manual_ser.index = pd.to_datetime(manual_ser.index, format=time_fmt)

        manual_ser = manual_ser.loc[beg_time:end_time]

        manual_nan_val = np.array(
            [manual_nan_val], dtype=manual_ser.values.dtype)[0]

        manual_ser[manual_ser > manual_nan_val] = manual_nan_val

    else:
        manual_ser = None
        manual_nan_val = None
    #==========================================================================

    cvgs_cls = ClusteredVariograms(True)

    cvgs_cls.set_data(
        data_df,
        crds_df,
        stns_min_dist_thresh=stns_min_dist_thresh)

    cvgs_cls.set_empirical_variogram_clustering_parameters(
            clus_type,
            ignore_zeros_flag,
            n_min_valid_values,
            smoothing,
            evg_stat,
            pstv_dfnt_flag,
            simplify_flag,
            norm_flag,
            max_dist_thresh,
            clus_ts=manual_ser,
            clus_ts_nan=manual_nan_val)

    cvgs_cls.set_theoretical_variogram_parameters(
        theoretical_variograms,
        permutation_sizes,
        max_opt_iters,
        max_variogram_range,
        apply_wts_flag)

    cvgs_cls.set_theoretical_variogram_clustering_parameters(
        n_distance_intervals,
        ks_alpha,
        min_abs_kriging_weight,
        max_nrst_nebs)

    cvgs_cls.set_misc_settings(out_dir, n_cpus, plot_figs_flag)

    cvgs_cls.verify()

    cvgs_cls.cluster_evgs()

    cvgs_cls.fit_theoretical_variograms()

    cvgs_cls.cluster_theoretical_variograms()
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
