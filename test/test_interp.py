'''
Nov 25, 2018
@author: Faizan-Uni-Stuttgart

'''

import sys

if ('P:\\Synchronize\\Python3Codes' not in sys.path):
    sys.path.append('P:\\Synchronize\\Python3Codes')

import os
import timeit
import time
from pathlib import Path

import pandas as pd

from spinterps import SpInterpMain


def main():

    main_dir = Path(
        r'P:\hydmod_de')

    os.chdir(main_dir)

    in_data_file = Path(
        r'P:\Synchronize\IWS\DWD_meteo_hist_pres\full_neckar_clim_data_2\precipitation.csv')

    in_stns_coords_file = Path(
        r'P:\Synchronize\IWS\DWD_meteo_hist_pres\full_neckar_clim_data_2\precipitation_coords.csv')

    in_vgs_file = Path(
        r'P:\hydmod_de\daily\ppt_daily_1961_2020__no_0s\tvgs_clus_txts\M_final_tvgs_clus_ts.csv')

    index_type = 'date'

    out_dir = Path(r'test_spinterps_grd_split_08')
    var_units = 'mm'  # 'C'  # u'\u2103'  # 'centigrade'
    var_name = 'precipitation'  # 'temperature'  #

    out_krig_net_cdf_file = r'kriging_%s_to_%s_5km.nc'

    freq = 'D'
    strt_date = r'1961-01-01'
    end_date = r'1961-12-31'

    drop_stns = []

    out_krig_net_cdf_file = out_krig_net_cdf_file % (strt_date, end_date)

    in_drift_rasters_list = (
        [Path(r'P:\Synchronize\IWS\QGIS_Neckar\raster\lower_de_gauss_z3_100m.tif')])

    in_bounds_shp_file = (
        Path(r'P:\Synchronize\IWS\QGIS_Neckar\raster\taudem_out_spate_rockenau\watersheds_all.shp'))
#     in_bounds_shp_file = None

    align_ras_file = in_drift_rasters_list[0]
#     align_ras_file = None

    nc_time_units = 'days since 1900-01-01 00:00:00.0'
    nc_calendar = 'gregorian'

    min_var_val_thresh = 0.1  # -float('inf')  #

    min_var_val = 0  # None #
    max_var_val = None

    min_vg_val = 1e-4

    max_steps_per_chunk = None

    # Used only when no drift or align raster given.
#     cell_size = 1000
    cell_size = None

    # Can be None or a string vg.
    # Replace all nan vgs with this.
    nan_vg = '0.0 Nug(0.0)'

    min_nebor_dist_thresh = 1

    idw_exps = [5]
    n_cpus = 8
    buffer_dist = 22e3
    sec_buffer_dist = 5e3  # Buffer around computed grid/polygons bounds.
    simplify_tolerance_ratio = 0.25  # Units same as polygons.

    neighbor_selection_method = 'nrst'
    n_neighbors = 1
    n_pies = 8

    in_sep = ';'
    in_date_fmt = '%Y-%m-%d'

    ord_krige_flag = True
    sim_krige_flag = True
    edk_krige_flag = True
    idw_flag = True
    plot_figs_flag = True
    verbose = True
    interp_around_polys_flag = True

    # ord_krige_flag = False
    # sim_krige_flag = False
    # edk_krige_flag = False
    # idw_flag = False
    # plot_figs_flag = False
    # verbose = False
    # interp_around_polys_flag = False

    if in_data_file.suffix == '.csv':
        in_data_df = pd.read_csv(
            in_data_file,
            sep=in_sep,
            index_col=0,
            encoding='utf-8')

    elif in_data_file.suffix == '.pkl':
        in_data_df = pd.read_pickle(in_data_file)

    else:
        raise NotImplementedError(
            f'Unknown file extension: {in_data_file.suffix}!')

    in_vgs_df = pd.read_csv(
        in_vgs_file,
        sep=in_sep,
        index_col=0,
        encoding='utf-8',
        dtype=str)

    in_stns_coords_df = pd.read_csv(
        in_stns_coords_file,
        sep=in_sep,
        index_col=0,
        encoding='utf-8')

    if drop_stns:
        assert all([
            validation_col in in_stns_coords_df.index
            for validation_col in drop_stns])

        in_stns_coords_df = in_stns_coords_df.loc[
            in_stns_coords_df.index.difference(pd.Index(drop_stns))]

    if nan_vg:
        assert isinstance(nan_vg, str), 'nan_vg can only be None or a string!'

        in_vgs_df.replace(float('nan'), nan_vg, inplace=True)
        in_vgs_df.replace('nan', nan_vg, inplace=True)

    else:
        assert nan_vg is None, 'nan_vg can only be None or a string!'

    if index_type == 'date':
        in_data_df.index = pd.to_datetime(in_data_df.index, format=in_date_fmt)

        in_vgs_df.index = pd.to_datetime(in_vgs_df.index, format=in_date_fmt)

    elif index_type == 'obj':
        in_data_df.index = pd.Index(in_data_df.index, dtype=object)

        in_vgs_df.index = pd.Index(in_vgs_df.index, dtype=object)

    else:
        raise ValueError(f'Incorrect index_type: {index_type}!')

    in_stns_coords_df = (in_stns_coords_df[['X', 'Y', 'Z']]).astype(float)

    spinterp_cls = SpInterpMain(verbose)

    spinterp_cls.set_data(
        in_data_df, in_stns_coords_df, index_type, min_nebor_dist_thresh)

    spinterp_cls.set_vgs_ser(in_vgs_df.iloc[:, 0], index_type)

    spinterp_cls.set_out_dir(out_dir)

    spinterp_cls.set_netcdf4_parameters(
        out_krig_net_cdf_file,
        var_units,
        var_name,
        nc_time_units,
        nc_calendar)

    spinterp_cls.set_interp_time_parameters(
        strt_date, end_date, freq, in_date_fmt)

    if in_bounds_shp_file is not None:
        spinterp_cls.set_cell_selection_parameters(
            in_bounds_shp_file,
            buffer_dist,
            interp_around_polys_flag,
            sec_buffer_dist,
            simplify_tolerance_ratio)

    if align_ras_file is not None:
        spinterp_cls.set_alignment_raster(align_ras_file)

    spinterp_cls.set_neighbor_selection_method(
        neighbor_selection_method, n_neighbors, n_pies)

    spinterp_cls.set_misc_settings(
        n_cpus,
        plot_figs_flag,
        cell_size,
        min_var_val_thresh,
        min_var_val,
        max_var_val,
        max_steps_per_chunk,
        min_vg_val)

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

    spinterp_cls = None
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
