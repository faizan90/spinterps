'''
@author: Faizan-Uni-Stuttgart

Oct 27, 2020

2:38:08 PM

'''
import os
import time
import timeit
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\hydmod\input_hyd_data')
    os.chdir(main_dir)

    in_h5_file = Path(r'temp_edk_1961_to_2015_daily_1km_rockenau_six_cats.h5')

    data_grp = 'full_neckar_avg_temp_kriging_1961-01-01_to_2015-12-31_1km_all'

    out_file_name_pref = f'neckar_{data_grp}'

    variable_labels = ['EDK']
    rel_cell_area_label = 'rel_itsctd_area'

    time_label = 'time/time_strs'

    sep = ';'

    float_fmt = '%0.3f'

    out_time_fmt = '%Y-%m-%d'

    fig_size = (15, 7)

    fig_xlabel = 'Time (days)'
    fig_ylabel = 'Precipitation (mm)'

    set_na_to_zero_flag = False

    save_df_pkl_flag = True

    out_dir = main_dir

    with h5py.File(in_h5_file, mode='r', driver=None) as h5_hdl:
        h5_times = pd.to_datetime(
            h5_hdl[time_label][...], format='%Y%m%dT%H%M%S')

        data_ds = h5_hdl[data_grp]

        for variable_label in variable_labels:
            keys = [int(key) for key in data_ds[variable_label].keys()]

            out_df = pd.DataFrame(index=h5_times, columns=keys, dtype=float)

            for key in keys:
                itsctd_area = h5_hdl[f'{rel_cell_area_label}/{key}'][:]

                out_df[key][:] = (
                    data_ds[f'{variable_label}/{key}'] * itsctd_area.T
                    ).sum(axis=1)

                plt.figure(figsize=fig_size)

                plt.plot(
                    out_df.index, out_df[key].values, alpha=0.7, label=key)

                plt.legend()

                plt.xlabel(fig_xlabel)
                plt.ylabel(fig_ylabel)

                plt.grid()

                out_fig_path = (
                    out_dir /
                    Path(f'{out_file_name_pref}__{variable_label}_{key}.png'))

                plt.savefig(str(out_fig_path), bbox_inches='tight')

                plt.close()

                if set_na_to_zero_flag:
                    nan_ct = np.isnan(out_df[key].values).sum()

                    if nan_ct:
                        print('\n')
                        print('#' * 30)
                        print(
                            f'WARNING: Set {nan_ct} values to zero in dataset '
                            f'{key} in file: {os.path.basename(in_h5_file)}!')
                        print('#' * 30)
                        print('\n')

                        out_df[key].replace(np.nan, 0.0, inplace=True)

            out_file_path = (
                out_dir / Path(f'{out_file_name_pref}__{variable_label}.csv'))

            out_df.to_csv(
                out_file_path,
                sep=sep,
                date_format=out_time_fmt,
                float_format=float_fmt)

            if save_df_pkl_flag:
                out_pkl_path = (out_dir / Path(
                    f'{out_file_name_pref}__{variable_label}.pkl'))

                out_df.to_pickle(out_pkl_path)

            print('Done with:', out_file_path)

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
