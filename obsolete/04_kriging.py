"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""

import os
import timeit
import time
from math import ceil

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import ogr
import gdal
import numpy as np
import pandas as pd
import netCDF4 as nc
import shapefile as shp

import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessPool

from faizpy import get_aligned_shp_bds_and_cell_size, cnvt_to_pt, chk_cntmt
from krige_mp_ftns import (
    ordinary_kriging,
    simple_kriging,
    external_drift_kriging,
    inverse_distance_wtng)

plt.ioff()

if __name__ == '__main__':
    from datetime import datetime
    from std_logger import StdFileLoggerCtrl

    # save all console activity to out_log_file
    out_log_file = os.path.join(r'P:\\',
                                r'Synchronize',
                                r'python_script_logs',
                                ('krigings_log_%s.log' %
                                 datetime.now().strftime('%Y%m%d%H%M%S')))
    log_link = StdFileLoggerCtrl(out_log_file)

    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = r'P:\Synchronize_LDs'

    interp_var = 'ppt'

    #==========================================================================
    if interp_var == 'mean_temp':
        # MEAN TEMPERATURE
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
        out_dir = r'Niedersachsen_temperature_avg_interpolation'
        var_units = u'\u2103'  # 'centigrade'
        var_name = 'avg_temperature'
        out_krig_net_cdf_file = r'Niedersachsen_avg_temp_kriging_%s_to_%s_1km_all.nc'

        # interpolated values
        # can be int, float, 'min_in'/'max_in' or None
        # min_var_val = 'min_in'
        # max_var_val = 'max_in'
        min_var_val = None
        max_var_val = None

    #==========================================================================

    #==========================================================================
    elif interp_var == 'min_temp':
        # MIN TEMPERATURE
        in_data_file = os.path.join(
            r'P:\Synchronize\IWS\DWD_meteo_hist_pres',
            r'Nidersachsen_temperature_min_norm_cop_infill_1961_to_2015',
            r'02_combined_station_outputs',
            r'infilled_var_df_infill_stns.csv')

        in_vgs_file = os.path.join(
            r'P:\Synchronize\IWS\DWD_meteo_hist_pres',
            r'Niedersachsen_min_temperature_kriging_20181121',
           r'min_temp_fitted_variograms__nk_1.000__evg_name_robust__ngp_5__h_typ_var.csv')

        in_stns_coords_file = os.path.join(
            os.path.dirname(in_data_file),
            r'infilled_var_df_infill_stns_coords.csv')
        out_dir = r'Niedersachsen_temperature_min_interpolation'
        var_units = u'\u2103'  # 'centigrade'
        var_name = 'min_temperature'
        out_krig_net_cdf_file = r'Niedersachsen_min_temp_kriging_%s_to_%s_1km_all.nc'

        # interpolated values
        # can be int, float, 'min_in'/'max_in' or None
        # min_var_val = 'min_in'
        # max_var_val = 'max_in'
        min_var_val = None
        max_var_val = None

    #==========================================================================

    #==========================================================================
    elif interp_var == 'max_temp':
        # MAX TEMPERATURE
        in_data_file = os.path.join(
            r'P:\Synchronize\IWS\DWD_meteo_hist_pres',
            r'Nidersachsen_temperature_max_norm_cop_infill_1961_to_2015',
            r'02_combined_station_outputs',
            r'infilled_var_df_infill_stns.csv')

        in_vgs_file = os.path.join(
            r'P:\Synchronize\IWS\DWD_meteo_hist_pres',
            r'Niedersachsen_max_temperature_kriging_20181121',
            r'max_temp_fitted_variograms__nk_1.000__evg_name_robust__ngp_5__h_typ_var.csv')

        in_stns_coords_file = os.path.join(
            os.path.dirname(in_data_file),
            r'infilled_var_df_infill_stns_coords.csv')
        out_dir = r'Niedersachsen_temperature_max_interpolation'
        var_units = u'\u2103'  # 'centigrade'
        var_name = 'max_temperature'
        out_krig_net_cdf_file = r'Niedersachsen_max_temp_kriging_%s_to_%s_1km_all.nc'

        # interpolated values
        # can be int, float, 'min_in'/'max_in' or None
        # min_var_val = 'min_in'
        # max_var_val = 'max_in'
        min_var_val = None
        max_var_val = None

    #==========================================================================

    #==========================================================================
    elif interp_var == 'ppt':
        # PRECIPITATION
        in_data_file = os.path.join(
            r'P:\Synchronize\IWS\DWD_meteo_hist_pres',
            r'Nidersachsen_precipitation_norm_cop_infill_1961_to_2015',
            r'02_combined_station_outputs',
            r'infilled_var_df_infill_stns.csv')

        in_vgs_file = os.path.join(
            r'P:\Synchronize\IWS\DWD_meteo_hist_pres',
            r'Niedersachsen_precipitation_kriging_20181124',
            r'ppt_fitted_variograms__nk_1.000__evg_name_robust__ngp_5__h_typ_var.csv')

        in_stns_coords_file = os.path.join(
            os.path.dirname(in_data_file),
            r'infilled_var_df_infill_stns_coords.csv')
        out_dir = r'Niedersachsen_precipitation_interpolation'
        var_units = 'mm'
        var_name = 'precipitation'
        out_krig_net_cdf_file = r'Niedersachsen_precipitation_kriging_%s_to_%s_1km_all.nc'

        # interpolated values
        # can be int, float, 'min_in'/'max_in' or None
        # min_var_val = 'min_in'
        # max_var_val = 'max_in'
        min_var_val = 0.0
        max_var_val = None

    #==========================================================================
    else:
        raise ValueError(f'Invalid value for interp_var: {interp_var}!')

    freq = 'D'
    strt_date = r'1961-01-01'
    end_date = r'1962-12-31'

    out_krig_net_cdf_file = out_krig_net_cdf_file % (strt_date, end_date)

    # assuming in_drift_raster and in_stns_coords_file and in_bounds_shp_file
    # have the same coordinates system
    # assuming in_drift_rasters_list have the same cell sizes, bounds and NDVs
    # basically they are copies of each other except for the drift values
    in_drift_rasters_list = (
        [r'P:\Synchronize\IWS\2016_DFG_SPATE\data\spate_engine_data\Niedersachsen\hydmod\raster\srtm_mosacis_niedersachsen_1km_gkz3.tif'])  # ,
    #     r'santa_rs_minerve_prep_june17/taudem_out/northings_drift_5km.tif',
    #     r'santa_rs_minerve_prep_june17/taudem_out/eastings_drift_5km.tif'])

#     in_bounds_shp_file = (
#         os.path.join(r'P:\Synchronize\IWS\QGIS_Neckar\raster',
#                      r'taudem_out_spate_rockenau\watersheds.shp'))

    in_bounds_shp_file = (
        os.path.join(r'P:\Synchronize\IWS\2016_DFG_SPATE\data\spate_engine_data\Niedersachsen\hydmod\raster',
                     r'taudem_out_nied\watersheds.shp'))

    align_ras_file = in_drift_rasters_list[0]

    out_figs_dir = os.path.join(out_dir, 'krige_figs')

    select_vg_lab = r'best_fit_vg_no_01'

    x_coords_lab = 'X'
    y_coords_lab = 'Y'
    time_dim_lab = 'time'
    nc_time_units = 'days since 1900-01-01 00:00:00.0'
    nc_calendar = 'gregorian'
    nc_mode = 'w'

    min_ppt_thresh = 1.0

    idw_exp = 5
    n_cpus = 31
    n_cpus_scale = 1
    buffer_dist = 20e3
    sec_buffer_dist = 2e3

    in_sep = str(';')
    in_date_fmt = '%Y-%m-%d'

    ord_krige_flag = True
    sim_krige_flag = True
    edk_krige_flag = True
    idw_flag = True
    plot_figs_flag = True

#     ord_krige_flag = False
#     sim_krige_flag = False
#     edk_krige_flag = False
#     idw_flag = False
#     plot_figs_flag = False

    os.chdir(main_dir)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if (not os.path.exists(out_figs_dir)) and plot_figs_flag:
        os.mkdir(out_figs_dir)

    print('min_var_val:', min_var_val)
    print('max_var_val:', max_var_val)
    print('idw_exp:', idw_exp)
    print('n_cpus:', n_cpus)
    print('nc_mode:', nc_mode)
    print('strt_date:', strt_date)
    print('end_date:', end_date)
    print('var_name:', var_name)
    print('out_dir:', out_dir)
    print('in_bounds_shp_file:', in_bounds_shp_file)
    print('out_krig_net_cdf_file:', out_krig_net_cdf_file)

    assert any([ord_krige_flag, sim_krige_flag, edk_krige_flag, idw_flag])

    #==========================================================================
    # read the data frames
    #==========================================================================
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

    in_data_df.dropna(inplace=True, axis=0, how='all')
    in_vgs_df.dropna(inplace=True, axis=0, how='all')
    in_stns_coords_df.dropna(inplace=True)

    in_data_df.index = pd.to_datetime(in_data_df.index, format=in_date_fmt)
    in_vgs_df.index = pd.to_datetime(in_vgs_df.index, format=in_date_fmt)

    fin_date_range = pd.date_range(strt_date, end_date, freq=freq)
    in_data_df = in_data_df.reindex(fin_date_range)
    in_vgs_df = in_vgs_df.reindex(fin_date_range)

    all_stns = in_data_df.columns.intersection(in_stns_coords_df.index)
    assert all_stns.shape[0]

    in_data_df = in_data_df.loc[:, all_stns]
    in_stns_coords_df = in_stns_coords_df.loc[all_stns, :]

    #==========================================================================
    # Get stations that are around/in the bounds_shp only
    #==========================================================================

    bds_vec = ogr.Open(in_bounds_shp_file)
    assert bds_vec

    bds_lyr = bds_vec.GetLayer(0)

    feat_buffs_list = []
    feat_sec_buffs_list = []
    for feat in bds_lyr:  # just to get the names of the catchments
        geom = feat.GetGeometryRef().Clone()
        assert geom

        feat_buffs_list.append(geom.Buffer(buffer_dist))
        feat_sec_buffs_list.append(geom.Buffer(sec_buffer_dist))

    bds_vec.Destroy()

    assert feat_buffs_list and feat_sec_buffs_list

    print(len(feat_buffs_list), 'polygons in the in_bounds_shp_file...')

    fin_stns = []
    for poly in feat_buffs_list:
        for stn in all_stns:
            if stn in fin_stns:
                continue

            curr_pt = cnvt_to_pt(
                *in_stns_coords_df.loc[stn, ['X', 'Y']].values)

            if chk_cntmt(curr_pt, poly):
                fin_stns.append(stn)

    assert fin_stns

    print('%d stations out of %d within buffer zone of in_bounds_shp_file' %
          (len(fin_stns), in_stns_coords_df.shape[0]))

    fin_stns = np.unique(fin_stns)
    in_data_df = in_data_df.loc[:, fin_stns]
    in_stns_coords_df = in_stns_coords_df.loc[fin_stns, :]

    #==========================================================================
    # Read the DEM
    #==========================================================================

    if edk_krige_flag:
        in_drift_arr_list = []
        _rows_list = []
        _cols_list = []

        for in_drift_raster in in_drift_rasters_list:
            in_drift_ds = gdal.Open(in_drift_raster)

            assert in_drift_ds, 'GDAL cannot open %s' % in_drift_raster

            drift_rows = in_drift_ds.RasterYSize
            drift_cols = in_drift_ds.RasterXSize

            drift_geotransform = in_drift_ds.GetGeoTransform()

            _drift_x_min = drift_geotransform[0]
            _drift_y_max = drift_geotransform[3]

            drift_band = in_drift_ds.GetRasterBand(1)
            drift_ndv = drift_band.GetNoDataValue()

            cell_width = drift_geotransform[1]
            cell_height = abs(drift_geotransform[5])

            _drift_x_max = _drift_x_min + (drift_cols * cell_width)
            _drift_y_min = _drift_y_max - (drift_rows * cell_height)

            _arr = in_drift_ds.ReadAsArray()

            in_drift_arr_list.append(_arr)
            _rows_list.append(_arr.shape[0])
            _cols_list.append(_arr.shape[1])

        assert all(_ == _rows_list[0] for _ in _rows_list), (
            'Drift raster have unequal number of rows!')

        assert all(_ == _cols_list[0] for _ in _cols_list), (
            'Drift raster have unequal number of columns!')

    #==========================================================================
    # Read the bounding shapefile
    #==========================================================================
    sf = shp.Reader(in_bounds_shp_file)
    polys_list = [i.__geo_interface__ for i in sf.iterShapes()]

    ((fin_x_min,
      fin_x_max,
      fin_y_min,
      fin_y_max),
     cell_width) = get_aligned_shp_bds_and_cell_size(
         in_bounds_shp_file, align_ras_file)

    cell_height = cell_width

    fin_x_min -= 2 * cell_width
    fin_x_max += 2 * cell_width
    fin_y_min -= 2 * cell_height
    fin_y_max += 2 * cell_height

    if edk_krige_flag:
        assert fin_x_min > _drift_x_min
        assert fin_x_max < _drift_x_max
        assert fin_y_min > _drift_y_min
        assert fin_y_max < _drift_y_max

        min_col = int(max(0, (fin_x_min - _drift_x_min) / cell_width))
        max_col = int(ceil((fin_x_max - _drift_x_min) / cell_width))

        min_row = int(max(0, (_drift_y_max - fin_y_max) / cell_height))
        max_row = int(ceil((_drift_y_max - fin_y_min) / cell_height))

    else:
        min_col = 0
        max_col = int(ceil((fin_x_max - fin_x_min) / cell_width))

        min_row = 0
        max_row = int(ceil((fin_y_max - fin_y_min) / cell_height))

    #==========================================================================
    # Calculate coordinates at which to krige
    #==========================================================================

    assert 0 <= min_col <= max_col, (min_col, max_col)
    assert 0 <= min_row <= max_row, (min_row, max_row)

    strt_x_coord = fin_x_min + (0.5 * cell_width)
    end_x_coord = strt_x_coord + ((max_col - min_col) * cell_width)

    strt_y_coord = fin_y_max - (0.5 * cell_height)
    end_y_coord = strt_y_coord - ((max_row - min_row) * cell_height)

    krige_x_coords = np.linspace(
        strt_x_coord, end_x_coord, (max_col - min_col + 1))

    krige_y_coords = np.linspace(
        strt_y_coord, end_y_coord, (max_row - min_row + 1))

    krige_x_coords_mesh, krige_y_coords_mesh = np.meshgrid(
        krige_x_coords, krige_y_coords)

    krige_coords_orig_shape = krige_x_coords_mesh.shape

    if plot_figs_flag:
        # xy coords for pcolormesh
        pcolmesh_x_coords = np.linspace(
            fin_x_min, fin_x_max, (max_col - min_col + 1))

        pcolmesh_y_coords = np.linspace(
            fin_y_max, fin_y_min, (max_row - min_row + 1))

        krige_x_coords_plot_mesh, krige_y_coords_plot_mesh = (
            np.meshgrid(pcolmesh_x_coords, pcolmesh_y_coords))

    else:
        krige_x_coords_plot_mesh, krige_y_coords_plot_mesh = None, None

    krige_x_coords_mesh = krige_x_coords_mesh.ravel()
    krige_y_coords_mesh = krige_y_coords_mesh.ravel()

    print('\n\n')
    print('#' * 10)

    _beg_t = timeit.default_timer()

    print(krige_x_coords_mesh.shape[0],
          'cells to interpolate per step before intersection!')

    fin_cntn_idxs = np.zeros(krige_x_coords_mesh.shape[0], dtype=bool)
    ogr_pts = np.vectorize(cnvt_to_pt)(krige_x_coords_mesh, krige_y_coords_mesh)

    for poly in feat_sec_buffs_list:
        curr_cntn_idxs = np.vectorize(chk_cntmt)(ogr_pts, poly)
        fin_cntn_idxs = fin_cntn_idxs | curr_cntn_idxs

    print(fin_cntn_idxs.sum(),
          'cells to interpolate per step after intersection!')

    _end_t = timeit.default_timer()
    _tot_t = _end_t - _beg_t

    print(f'Took {_tot_t:0.4f} seconds!')
    print('#' * 10)

    krige_x_coords_mesh = krige_x_coords_mesh[fin_cntn_idxs]
    krige_y_coords_mesh = krige_y_coords_mesh[fin_cntn_idxs]

    if edk_krige_flag:
        drift_vals_list = []

        krige_cols = np.arange(min_col, max_col + 1, dtype=int)
        krige_rows = np.arange(min_row, max_row + 1, dtype=int)

        assert krige_x_coords.shape[0] == krige_cols.shape[0]
        assert krige_y_coords.shape[0] == krige_rows.shape[0]

        (krige_drift_cols_mesh,
         krige_drift_rows_mesh) = np.meshgrid(krige_cols, krige_rows)

        krige_drift_cols_mesh = krige_drift_cols_mesh.ravel()
        krige_drift_rows_mesh = krige_drift_rows_mesh.ravel()

        krige_drift_cols_mesh = krige_drift_cols_mesh[fin_cntn_idxs]
        krige_drift_rows_mesh = krige_drift_rows_mesh[fin_cntn_idxs]

        for _drift_arr in in_drift_arr_list:
            _drift_vals = _drift_arr[
                krige_drift_rows_mesh, krige_drift_cols_mesh]

            drift_vals_list.append(_drift_vals)

        drift_vals_arr = np.array(drift_vals_list, dtype=float)

        drift_df_cols = list(range(len(in_drift_rasters_list)))
        in_stns_drift_df = pd.DataFrame(
            index=in_stns_coords_df.index,
            columns=drift_df_cols,
            dtype=float)

        for stn in in_stns_drift_df.index:
            stn_x = in_stns_coords_df.loc[stn, x_coords_lab]
            stn_y = in_stns_coords_df.loc[stn, y_coords_lab]

            stn_col = int((stn_x - _drift_x_min) / cell_width)
            stn_row = int((_drift_y_max - stn_y) / cell_height)

            for col, _arr in zip(drift_df_cols, in_drift_arr_list):
                try:
                    _ = _arr[stn_row, stn_col]
                    if not np.isclose(drift_ndv, _):
                        in_stns_drift_df.loc[stn, col] = _

                except IndexError:
                    pass

        in_stns_drift_df.dropna(inplace=True)

    #==========================================================================
    # Open NC
    #==========================================================================
    out_nc = nc.Dataset(
        os.path.join(out_dir, out_krig_net_cdf_file), mode=str(nc_mode))

    if nc_mode == 'w':
        out_nc.set_auto_mask(False)
        out_nc.createDimension(x_coords_lab, krige_x_coords.shape[0])
        out_nc.createDimension(y_coords_lab, krige_y_coords.shape[0])
        out_nc.createDimension(time_dim_lab, fin_date_range.shape[0])

        x_coords_nc = out_nc.createVariable(
            x_coords_lab, 'd', dimensions=x_coords_lab)

        x_coords_nc[:] = krige_x_coords

        y_coords_nc = out_nc.createVariable(
            y_coords_lab, 'd', dimensions=y_coords_lab)

        y_coords_nc[:] = krige_y_coords

        time_nc = out_nc.createVariable(
            time_dim_lab, 'i8', dimensions=time_dim_lab)

        time_nc[:] = nc.date2num(
            fin_date_range.to_pydatetime(),
            units=nc_time_units,
            calendar=nc_calendar)

        time_nc.units = nc_time_units
        time_nc.calendar = nc_calendar

    else:
        raise RuntimeError('Not configured for this option!')

        time_nc = out_nc.variables[time_dim_lab]
        krige_y_coords = y_coords_nc[:]
        krige_x_coords = x_coords_nc[:]

    #==========================================================================
    # MP stuff
    #==========================================================================
    mp_cond = False

    if ((n_cpus > 1) and  (in_data_df.shape[0] > (n_cpus + 1))):
        idxs = pd.np.linspace(
            0,
            in_data_df.shape[0],
            (n_cpus * n_cpus_scale) + 1,
            endpoint=True,
            dtype=int)

        idxs = np.unique(idxs)
        print('MP idxs:', idxs)

        if idxs.shape[0] == 1:
            idxs = np.concatenate((np.array([0]), idxs))

        mp_cond = True

    else:
        idxs = [0, in_data_df.shape[0]]

    #==========================================================================
    # Krige
    #==========================================================================
    if ord_krige_flag:
        print('\n\n')
        print('#' * 10)

        _beg_t = timeit.default_timer()

        print('Ordinary Kriging...')

        if 'OK' not in out_nc.variables:
            ok_nc = out_nc.createVariable(
                'OK',
                'd',
                dimensions=(time_dim_lab, y_coords_lab, x_coords_lab),
                fill_value=False)

        else:
            ok_nc = out_nc.variables['OK']

        ok_vars_gen = ((in_data_df.iloc[idxs[i]:idxs[i + 1]],
                        in_stns_coords_df,
                        in_vgs_df.iloc[idxs[i]:idxs[i + 1]],
                        min_ppt_thresh,
                        var_name,
                        krige_x_coords_mesh,
                        krige_y_coords_mesh,
                        krige_coords_orig_shape,
                        min_var_val,
                        max_var_val,
                        (idxs[i], idxs[i + 1]),
                        plot_figs_flag,
                        krige_x_coords_plot_mesh,
                        krige_y_coords_plot_mesh,
                        var_units,
                        polys_list,
                        out_figs_dir,
                        fin_cntn_idxs) for i in range(n_cpus))

        if mp_cond:
            ok_krige_flds = np.full(
                (fin_date_range.shape[0],
                 krige_coords_orig_shape[0],
                 krige_coords_orig_shape[1]),
                np.nan,
                dtype=np.float32)

            mp_ress = []

            try:
                mp_pool = ProcessPool(n_cpus)
                mp_pool.restart(True)

                mp_ress = list(mp_pool.uimap(ordinary_kriging, ok_vars_gen))

                mp_pool.clear()

            except Exception as msg:
                mp_pool.close()
                mp_pool.join()
                print('Error in ordinary_kriging:', msg)

            for mp_res in mp_ress:
                if (len(mp_res) != 3) and (not isinstance(list)):
                    print('\n', mp_res, '\n')
                    continue

                [strt_index, end_index, sub_ok_krige_flds] = mp_res
                ok_krige_flds[strt_index:end_index] = sub_ok_krige_flds

                # free memory
                mp_res[2], sub_ok_krige_flds = None, None

            ok_nc[:] = ok_krige_flds

        else:
            [strt_index,
             end_index,
             ok_krige_flds] = ordinary_kriging(next(ok_vars_gen))

            ok_nc[:] = ok_krige_flds

        ok_nc.units = var_units
        ok_nc.standard_name = var_name + ' (ordinary kriging)'

        ok_krige_flds = None

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    if sim_krige_flag:
        print('\n\n')
        print('#' * 10)

        _beg_t = timeit.default_timer()

        print('Simple Kriging...')
        if 'SK' not in out_nc.variables:
            sk_nc = out_nc.createVariable(
                'SK',
                'd',
                dimensions=(time_dim_lab, y_coords_lab, x_coords_lab),
                fill_value=False)

        else:
            sk_nc = out_nc.variables['SK']

        sk_vars_gen = ((in_data_df.iloc[idxs[i]:idxs[i + 1]],
                        in_stns_coords_df,
                        in_vgs_df.iloc[idxs[i]:idxs[i + 1]],
                        min_ppt_thresh,
                        var_name,
                        krige_x_coords_mesh,
                        krige_y_coords_mesh,
                        krige_coords_orig_shape,
                        min_var_val,
                        max_var_val,
                        (idxs[i], idxs[i + 1]),
                        plot_figs_flag,
                        krige_x_coords_plot_mesh,
                        krige_y_coords_plot_mesh,
                        var_units,
                        polys_list,
                        out_figs_dir,
                        fin_cntn_idxs) for i in range(n_cpus))

        if mp_cond:
            sk_krige_flds = np.full(
                (fin_date_range.shape[0],
                 krige_coords_orig_shape[0],
                 krige_coords_orig_shape[1]),
                np.nan,
                dtype=np.float32)

            mp_ress = []

            try:
                mp_pool = ProcessPool(n_cpus)
                mp_pool.restart(True)

                mp_ress = list(mp_pool.uimap(simple_kriging, sk_vars_gen))

                mp_pool.clear()

            except Exception as msg:
                mp_pool.close()
                mp_pool.join()
                print('Error in simple_kriging:', msg)

            for mp_res in mp_ress:
                if (len(mp_res) != 3) and (not isinstance(list)):
                    print('\n', mp_res, '\n')
                    continue

                [strt_index, end_index, sub_sk_krige_flds] = mp_res
                sk_krige_flds[strt_index:end_index] = sub_sk_krige_flds

                # free memory
                mp_res[2], sub_sk_krige_flds = None, None

            sk_nc[:] = sk_krige_flds

        else:
            [strt_index,
             end_index,
             sk_krige_flds] = simple_kriging(next(sk_vars_gen))

            sk_nc[:] = sk_krige_flds

        sk_nc.units = var_units
        sk_nc.standard_name = var_name + ' (simple kriging)'

        sk_krige_flds = None

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    if edk_krige_flag:
        print('\n\n')
        print('#' * 10)

        _beg_t = timeit.default_timer()

        print('External Drift Kriging...')
        if 'EDK' not in out_nc.variables:
            edk_nc = out_nc.createVariable(
                'EDK',
                'd',
                dimensions=(time_dim_lab, y_coords_lab, x_coords_lab),
                fill_value=False)

        else:
            edk_nc = out_nc.variables['EDK']

        edk_vars_gen = ((in_data_df.iloc[idxs[i]:idxs[i + 1]],
                         in_stns_drift_df,
                         in_stns_coords_df,
                         in_vgs_df.iloc[idxs[i]:idxs[i + 1]],
                         min_ppt_thresh,
                         var_name,
                         krige_x_coords_mesh,
                         krige_y_coords_mesh,
                         drift_vals_arr,
                         krige_coords_orig_shape,
                         drift_ndv,
                         min_var_val,
                         max_var_val,
                         (idxs[i], idxs[i + 1]),
                         plot_figs_flag,
                         krige_x_coords_plot_mesh,
                         krige_y_coords_plot_mesh,
                         var_units,
                         polys_list,
                         out_figs_dir,
                         fin_cntn_idxs) for i in range(n_cpus))

        if mp_cond:
            edk_krige_flds = np.full(
                (fin_date_range.shape[0],
                 krige_coords_orig_shape[0],
                 krige_coords_orig_shape[1]),
                np.nan,
                dtype=np.float32)

            mp_ress = []

            try:
                mp_pool = ProcessPool(n_cpus)
                mp_pool.restart(True)

                mp_ress = list(mp_pool.uimap(
                    external_drift_kriging, edk_vars_gen))

                mp_pool.clear()

            except Exception as msg:
                mp_pool.close()
                mp_pool.join()
                print('Error in external_drift_kriging:', msg)

            for mp_res in mp_ress:
                if (len(mp_res) != 3) and (not isinstance(list)):
                    print('\n', mp_res, '\n')
                    continue

                [strt_index, end_index, sub_edk_krige_flds] = mp_res
                edk_krige_flds[strt_index:end_index] = sub_edk_krige_flds

                print('sub_min:', np.nanmin(sub_edk_krige_flds))
                print('sub_max:', np.nanmax(sub_edk_krige_flds))

                # free memory
                mp_res[2], sub_edk_krige_flds = None, None

        else:
            [strt_index,
             end_index,
             edk_krige_flds] = external_drift_kriging(next(edk_vars_gen))

        edk_nc[:] = edk_krige_flds

        edk_nc.units = var_units
        edk_nc.standard_name = var_name + ' (external drift kriging)'

        edk_krige_flds = None

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #==========================================================================
    # IDW
    #==========================================================================
    if idw_flag:
        print('\n\n')
        print('#' * 10)

        _beg_t = timeit.default_timer()

        print('Inverse Distance Weighting...')
        if 'IDW' not in out_nc.variables:
            idw_nc = out_nc.createVariable(
                'IDW',
                'd',
                 dimensions=(time_dim_lab, y_coords_lab, x_coords_lab),
                 fill_value=False)

        else:
            idw_nc = out_nc.variables['IDW']

        idw_vars_gen = ((in_data_df.iloc[idxs[i]:idxs[i + 1]],
                        in_stns_coords_df,
                        min_ppt_thresh,
                        idw_exp,
                        var_name,
                        krige_x_coords_mesh,
                        krige_y_coords_mesh,
                        krige_coords_orig_shape,
                        min_var_val,
                        max_var_val,
                        (idxs[i], idxs[i + 1]),
                        plot_figs_flag,
                        krige_x_coords_plot_mesh,
                        krige_y_coords_plot_mesh,
                        var_units,
                        polys_list,
                        out_figs_dir,
                        fin_cntn_idxs) for i in range(n_cpus))

        if mp_cond:
            idw_flds = np.full(
                (fin_date_range.shape[0],
                 krige_coords_orig_shape[0],
                 krige_coords_orig_shape[1]),
                np.nan,
                dtype=np.float32)

            mp_ress = []
            try:
                mp_pool = ProcessPool(n_cpus)
                mp_pool.restart(True)

                mp_ress = list(mp_pool.uimap(
                    inverse_distance_wtng, idw_vars_gen))

                mp_pool.clear()

            except Exception as msg:
                mp_pool.close()
                mp_pool.join()
                print('Error in inverse_distance_wtng:', msg)

            for mp_res in mp_ress:
                if (len(mp_res) != 3) and (not isinstance(list)):
                    print('\n', mp_res, '\n')
                    continue

                [strt_index, end_index, sub_idw_flds] = mp_res
                idw_flds[strt_index:end_index] = sub_idw_flds

                # free memory
                mp_res[2], sub_idw_flds = None, None

        else:
            [strt_index,
             end_index,
             idw_flds] = inverse_distance_wtng(next(idw_vars_gen))

        idw_nc[:] = idw_flds

        idw_nc.units = var_units
        idw_nc.standard_name = (
            var_name + ' (IDW (exp=%0.3f))' % float(idw_exp))

        idw_flds = None

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    out_nc.Author = 'Faizan IWS Uni-Stuttgart'
    out_nc.Source = out_nc.filepath()
    out_nc.close()

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s. Total run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
    log_link.stop()
