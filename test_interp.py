'''
Nov 25, 2018
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import pandas as pd

from spinterps import SpInterpMain


def main():

    main_dir = Path(r'Q:\Synchronize_LDs')
    os.chdir(main_dir)

    in_data_file = os.path.join(
            r'P:\Synchronize\IWS\DWD_meteo_hist_pres',
            r'Nidersachsen_temperature_avg_norm_cop_infill_1961_to_2015',
            r'02_combined_station_outputs',
            r'infilled_var_df_infill_stns.csv')

    in_vgs_file = os.path.join(
        r'P:\Synchronize\IWS\DWD_meteo_hist_pres',
        r'Niedersachsen_avg_temperature_kriging_20181121',
        r'avg_temp_fitted_variograms__nk_1.000__evg_name_robust__ngp_5__h_typ_var.csv')

    in_stns_coords_file = os.path.join(
        os.path.dirname(in_data_file),
        r'infilled_var_df_infill_stns_coords.csv')
    out_dir = r'test_Niedersachsen_temperature_avg_interpolation'
    var_units = 'c'  # u'\u2103'  # 'centigrade'
    var_name = 'avg_temperature'

    out_krig_net_cdf_file = r'test_Niedersachsen_avg_temp_kriging_%s_to_%s_5km_all.nc'

    freq = 'D'
    strt_date = r'1961-01-01'
    end_date = r'1961-06-30'

    out_krig_net_cdf_file = out_krig_net_cdf_file % (strt_date, end_date)

    in_drift_rasters_list = (
        [r'P:\Synchronize\IWS\2016_DFG_SPATE\data\spate_engine_data\Niedersachsen\hydmod\raster\srtm_mosacis_niedersachsen_5km_gkz3.tif'])  # ,
    #     r'santa_rs_minerve_prep_june17/taudem_out/northings_drift_5km.tif',
    #     r'santa_rs_minerve_prep_june17/taudem_out/eastings_drift_5km.tif'])

#     in_bounds_shp_file = (
#         os.path.join(r'P:\Synchronize\IWS\QGIS_Neckar\raster',
#                      r'taudem_out_spate_rockenau\watersheds.shp'))

    in_bounds_shp_file = (
        os.path.join(r'P:\Synchronize\IWS\2016_DFG_SPATE\data\spate_engine_data\Niedersachsen\hydmod\raster',
                     r'taudem_out_nied\watersheds_test_krige.shp'))

    align_ras_file = in_drift_rasters_list[0]

    out_figs_dir = os.path.join(out_dir, 'krige_figs')

    select_vg_lab = r'best_fit_vg_no_01'

    x_coords_lab = 'X'
    y_coords_lab = 'Y'
    time_dim_lab = 'time'
    nc_time_units = 'days since 1900-01-01 00:00:00.0'
    nc_calendar = 'gregorian'

    min_ppt_thresh = -float('inf')

    min_var_val = None
    max_var_val = None

    idw_exps = [2, 3, 5]
    n_cpus = 1
    buffer_dist = 20e3
    sec_buffer_dist = 2e3

    in_sep = str(';')
    in_date_fmt = '%Y-%m-%d'

    ord_krige_flag = True
    sim_krige_flag = True
    edk_krige_flag = True
    idw_flag = True
    plot_figs_flag = True
    verbose = True
    interp_around_polys_flag = True

#     ord_krige_flag = False
#     sim_krige_flag = False
#     edk_krige_flag = False
#     idw_flag = False
#     plot_figs_flag = False
#     verbose = False
#     interp_around_polys_flag = False

    in_data_df = pd.read_csv(
        in_data_file,
        sep=in_sep,
        index_col=0,
        encoding='utf-8')

    in_vgs_df = pd.read_csv(
        in_vgs_file,
        sep=in_sep,
        index_col=0,
        encoding='utf-8')

    in_stns_coords_df = pd.read_csv(
        in_stns_coords_file,
        sep=in_sep,
        index_col=0,
        encoding='utf-8')

    in_data_df.index = pd.to_datetime(in_data_df.index, format=in_date_fmt)
    in_vgs_df.index = pd.to_datetime(in_vgs_df.index, format=in_date_fmt)

    spinterp_cls = SpInterpMain(verbose)

    spinterp_cls.set_data(in_data_df, in_stns_coords_df)
    spinterp_cls.set_vgs_ser(in_vgs_df.iloc[:, 0])
    spinterp_cls.set_out_dir(out_dir)

    spinterp_cls.set_netcdf4_parameters(
        out_krig_net_cdf_file,
        var_units,
        var_name,
        nc_time_units,
        nc_calendar)

    spinterp_cls.set_interp_time_parameters(strt_date, end_date, freq, in_date_fmt)
    spinterp_cls.set_cell_selection_parameters(
        in_bounds_shp_file,
        buffer_dist,
        interp_around_polys_flag,
        sec_buffer_dist)
    spinterp_cls.set_alignment_raster(align_ras_file)

    spinterp_cls.set_misc_settings(
        n_cpus,
        plot_figs_flag,
        None,
        min_ppt_thresh,
        min_var_val,
        max_var_val)

    if ord_krige_flag:
        spinterp_cls.turn_ordinary_kriging_on()

    if sim_krige_flag:
        spinterp_cls.turn_simple_kriging_on()

    if edk_krige_flag:
        spinterp_cls.turn_external_drift_kriging_on(in_drift_rasters_list)

    if idw_flag:
        spinterp_cls.turn_inverse_distance_weighting_on(idw_exps)

    spinterp_cls.verify()
    spinterp_cls.interpolate()
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
