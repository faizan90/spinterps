'''
Nov 25, 2018
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import pandas as pd

from spinterps import FitVariograms


def get_mean_temp_paths():

#     in_vals_df_loc = os.path.join(
#              r'Mulde_temperature_avg_norm_cop_infill_1950_to_2015_20190417',
#              r'02_combined_station_outputs',
#              r'infilled_var_df_infill_stns.csv')
#
#     in_stn_coords_df_loc = os.path.join(
#              r'Mulde_temperature_avg_norm_cop_infill_1950_to_2015_20190417',
#              r'02_combined_station_outputs',
#             r'infilled_var_df_infill_stns_coords.csv')
#
#     out_dir = r'Mulde_temperature_avg_kriging_20190417'

    in_vals_df_loc = os.path.join(
             r'temperature_avg.csv')

    in_stn_coords_df_loc = os.path.join(
             r'temperature_avg_coords.csv')

    out_dir = r'temperature_kriging/obs'

    return in_vals_df_loc, in_stn_coords_df_loc, out_dir


def get_min_temp_paths():

    in_vals_df_loc = os.path.join(
             r'Mulde_temperature_min_norm_cop_infill_1950_to_2015_20190417',
             r'02_combined_station_outputs',
             r'infilled_var_df_infill_stns.csv')

    in_stn_coords_df_loc = os.path.join(
             r'Mulde_temperature_min_norm_cop_infill_1950_to_2015_20190417',
             r'02_combined_station_outputs',
            r'infilled_var_df_infill_stns_coords.csv')

    out_dir = r'Mulde_temperature_min_kriging_20190417'

    return in_vals_df_loc, in_stn_coords_df_loc, out_dir


def get_max_temp_paths():

    in_vals_df_loc = os.path.join(
             r'Mulde_temperature_max_norm_cop_infill_1950_to_2015_20190417',
             r'02_combined_station_outputs',
             r'infilled_var_df_infill_stns.csv')

    in_stn_coords_df_loc = os.path.join(
             r'Mulde_temperature_max_norm_cop_infill_1950_to_2015_20190417',
             r'02_combined_station_outputs',
            r'infilled_var_df_infill_stns_coords.csv')

    out_dir = r'Mulde_temperature_max_kriging_20190417'

    return in_vals_df_loc, in_stn_coords_df_loc, out_dir


def get_ppt_paths():

#     in_vals_df_loc = os.path.join(
#              r'Mulde_preciptiation_infilling_1950_2015',
#              r'02_combined_station_outputs',
#              r'infilled_var_df_infill_stns.csv')
#
#     in_stn_coords_df_loc = os.path.join(
#              r'Mulde_preciptiation_infilling_1950_2015',
#              r'02_combined_station_outputs',
#             r'infilled_var_df_infill_stns_coords.csv')
#
#     out_dir = r'Mulde_precipitation_kriging_20190417'

#     in_vals_df_loc = os.path.join(
#         r'full_neckar_clim_data_2',
#         r'precipitation.csv')
#
#     in_stn_coords_df_loc = os.path.join(
#         r'full_neckar_clim_data_2',
#         r'precipitation_coords.csv')
#
#     out_dir = r'Neckar_precipitation_kriging_20201028'

    in_vals_df_loc = os.path.join(
             r'precipitation.csv')

    in_stn_coords_df_loc = os.path.join(
             r'precipitation_coords.csv')

    out_dir = r'precipitation_kriging/obs'

    return in_vals_df_loc, in_stn_coords_df_loc, out_dir


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\multisite_phs_spec_corr')
    os.chdir(main_dir)

    vg_vars = ['ppt']

    strt_date = '1991-01-01'
    end_date = '1991-12-30'
    min_valid_stns = 10

    drop_stns = []  # ['T3705', 'T1875', 'T5664', 'T1197']
#     drop_stns = ['P3733', 'P3315', 'P3713', 'P3454']

    mdr = 0.7
    perm_r_list = [1, 2]
    fit_vgs = ['Sph', 'Exp']
    fil_nug_vg = 'Nug'
    n_best = 1
    ngp = 5
    figs_flag = False

    n_cpus = 8

    sep = ';'

    for vg_var in vg_vars:
        if vg_var == 'mean_temp':
            (in_vals_df_loc,
             in_stn_coords_df_loc,
             out_dir) = get_mean_temp_paths()

        elif vg_var == 'min_temp':
            (in_vals_df_loc,
             in_stn_coords_df_loc,
             out_dir) = get_min_temp_paths()

        elif vg_var == 'max_temp':
            (in_vals_df_loc,
             in_stn_coords_df_loc,
             out_dir) = get_max_temp_paths()

        elif vg_var == 'ppt':
            (in_vals_df_loc,
             in_stn_coords_df_loc,
             out_dir) = get_ppt_paths()

        else:
            raise RuntimeError(f'Unknown vg_var: {vg_var}!')

        in_vals_df = pd.read_csv(
            in_vals_df_loc, sep=sep, index_col=0, encoding='utf-8')

        in_vals_df.index = pd.to_datetime(in_vals_df.index, format='%Y-%m-%d')
        in_vals_df = in_vals_df.loc[strt_date:end_date, :]

        if drop_stns:
            in_vals_df.drop(labels=drop_stns, axis=1, inplace=True)

        in_vals_df.dropna(how='all', axis=0, inplace=True)

        in_coords_df = pd.read_csv(
            in_stn_coords_df_loc, sep=sep, index_col=0, encoding='utf-8')

        in_coords_df.index = list(map(str, in_coords_df.index))

        if drop_stns:
            in_coords_df.drop(labels=drop_stns, axis=0, inplace=True)

        in_coords_df = in_coords_df[['X', 'Y', 'Z']].astype(float)

        fit_vg_cls = FitVariograms()

        fit_vg_cls.set_data(in_vals_df, in_coords_df)

        fit_vg_cls.set_vg_fitting_parameters(
            mdr,
            perm_r_list,
            fil_nug_vg,
            ngp,
            fit_vgs,
            n_best)

        fit_vg_cls.set_misc_settings(n_cpus, min_valid_stns)

        fit_vg_cls.set_output_settings(out_dir, figs_flag)

        fit_vg_cls.verify()

        fit_vg_cls.fit_vgs()

        fit_vg_cls.save_fin_vgs_df()
        fit_vg_cls = None

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
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
