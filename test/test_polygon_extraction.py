'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

from spinterps import Extract


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Projects\2016_DFG_SPATE\data\cp_classi_for_partners\ts')
    os.chdir(main_dir)

    path_to_shp = r'P:\Synchronize\IWS\Projects\2016_DFG_SPATE\data\cp_classi_for_partners\vector\danube_gkz3_merged_polys.shp'

    label_field = r'BASINCODE'

    path_to_ras = r'P:\Synchronize\IWS\Projects\2016_DFG_SPATE\data\cp_classi_for_partners\raster\tot_prec_1901_daysum.nc'
    input_ras_type = 'nc'

#     path_to_ras = r'P:\Synchronize\IWS\Colleagues_Students\Mischa\lulc_geohyd_ratio_rasters\lower_de_gauss_z3_1km_hydrogeol_einheit_nr_hydmod_lulc_ratios.tif'
#     input_ras_type = 'gtiff'

    nc_x_crds_label = 'lon'
    nc_y_crds_label = 'lat'
    nc_variable_labels = ['TOT_PREC']
    nc_time_label = 'time'

#     src_epsg = None
#     dst_epsg = None

    src_epsg = 4326
    dst_epsg = 31467

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

    path_to_output = Path(r'test_danube_cosmo2.h5')
#     path_to_output = 'lower_de_gauss_z3_1km_hydrogeol_einheit_nr_hydmod_lulc_ratios.h5'

    Ext = Extract(True)

    res = None

    if input_ras_type == 'gtiff':
        res = Ext.extract_from_geotiff(
            path_to_shp,
            label_field,
            path_to_ras,
            path_to_output,
            src_epsg,
            dst_epsg)

    elif input_ras_type == 'nc':
        res = Ext.extract_from_netCDF(
            path_to_shp,
            label_field,
            path_to_ras,
            path_to_output,
            nc_x_crds_label,
            nc_y_crds_label,
            nc_variable_labels,
            nc_time_label,
            src_epsg,
            dst_epsg)

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
