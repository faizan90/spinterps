'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

from spinterps import Extract


def main():

    main_dir = Path(r'P:\cmip6\ec-earth3-cc\bw')
    os.chdir(main_dir)

    path_to_shp = r'P:\Synchronize\IWS\QGIS_Neckar\raster\taudem_out_dem_2km\watersheds_cumm.shp'
    label_field = r'DN'

#     path_to_shp = r'P:\Synchronize\IWS\Colleagues_Students\Ehsan\cp_classi\data\Border_6MainBasins_sh\MainBasins_Iran__utm39n3.shp'
#     label_field = r'HOZEH6'

    # path_to_rass = [Path(f'{year}.nc') for year in range(2006, 2021)]
    # path_to_rass = main_dir.glob('pr_*bw.nc')
    path_to_rass = [Path(r'pr_1950_2014_bw.nc')]
    input_ras_type = 'nc'

    # path_to_ras = r'P:\Synchronize\IWS\QGIS_Neckar\raster\taudem_out_spate_rockenau\fil.tif'
    # input_ras_type = 'gtiff'

    nc_x_crds_label = 'x_utm32n'
    nc_y_crds_label = 'y_utm32n'
    nc_variable_labels = ['pr']
    nc_time_label = 'time'

    # nc_x_crds_label = 'lon'
    # nc_y_crds_label = 'lat'
    # nc_variable_labels = ['pr']
    # nc_time_label = 'time'

    out_ext = 'h5'  # h5 means time series of pts in h5, nc snips it.

    out_suff = 'neckar'

    src_epsg = None
    dst_epsg = None

    simplify_tol_ratio = 0.001
    minimum_cell_area_intersection_percentage = 1e-3
    n_cpus = 1  # 'auto'

#     src_epsg = 4326
#     dst_epsg = 31467

#     main_dir = Path(r'P:\Downloads\spinterp_2d_nc_crds_test')
#     os.chdir(main_dir)
#
#     path_to_shp = r'01Small.shp'
#
#     label_field = r'Id'
#
#     path_to_ras = r'pr_SAM-44_ICHEC-EC-EARTH_historical_r12i1p1_SMHI-RCA4_v3_day_19810101-19851231.nc'
#     input_ras_type = 'nc'
#
#     nc_x_crds_label = 'lon'
#     nc_y_crds_label = 'lat'
#     nc_variable_labels = ['pr']
#     nc_time_label = 'time'

    Ext = Extract(True)

    res = None

    if input_ras_type == 'gtiff':

        for path_to_ras in path_to_rass:

            path_to_output = Path(rf'{path_to_ras.stem}_{out_suff}.{out_ext}')

            res = Ext.extract_from_geotiff(
                path_to_shp,
                label_field,
                path_to_ras,
                path_to_output,
                minimum_cell_area_intersection_percentage,
                n_cpus,
                src_epsg,
                dst_epsg,
                simplify_tol_ratio)

    elif input_ras_type == 'nc':
        for path_to_ras in path_to_rass:

            path_to_output = Path(rf'{path_to_ras.stem}_{out_suff}.{out_ext}')

            res = Ext.extract_from_netCDF(
                path_to_shp,
                label_field,
                path_to_ras,
                path_to_output,
                nc_x_crds_label,
                nc_y_crds_label,
                nc_variable_labels,
                nc_time_label,
                minimum_cell_area_intersection_percentage,
                n_cpus,
                src_epsg,
                dst_epsg,
                simplify_tol_ratio)

    else:
        raise NotImplementedError

    print('\n')
    print('res:', res)

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
