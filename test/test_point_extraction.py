'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

from spinterps import (
    ExtractPoints,
    ExtractGTiffCoords,
    ExtractNetCDFCoords,
    GeomAndCrdsItsctIdxs,
    ExtractGTiffValues,
    ExtractNetCDFValues)


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\ft_spatio_temporal_interps')
    os.chdir(main_dir)

    path_to_shp = 'temperature_validation_stns.shp'
    label_field = 'field_1'

    path_to_ras = r'temperature_interpolation_validation\obs\temperature_kriging_1989-01-01_to_1992-12-30_1km_obs.nc'
    input_ras_type = 'nc'

#     path_to_ras = (
#         r'lower_de_gauss_z3_2km_atkis_19_extended_hydmod_lulc_ratios.tif')
#     input_ras_type = 'gtiff'

    nc_x_crds_label = 'X'
    nc_y_crds_label = 'Y'
    nc_variable_labels = ['OK']
    nc_time_label = 'time'

    path_to_output = r'temperature_interpolation_validation\obs\ts_valid.h5'

    verbose = True

    EP = ExtractPoints(verbose=verbose)

    EP.set_input(path_to_shp, label_field)

    EP.extract_points()

#     print(EP.get_labels())
#     print(EP.get_points())
#     print(EP.get_x_coordinates())
#     print(EP.get_y_coordinates())

    GCII = GeomAndCrdsItsctIdxs(verbose=verbose)

    if input_ras_type == 'gtiff':
        gtiff_crds_cls = ExtractGTiffCoords(verbose=verbose)

        gtiff_crds_cls.set_input(path_to_ras)
        gtiff_crds_cls.extract_coordinates()

        GCII.set_geometries(EP.get_points(), EP._geom_type)

        GCII.set_coordinates(
            gtiff_crds_cls.get_x_coordinates(),
            gtiff_crds_cls.get_y_coordinates(),
            gtiff_crds_cls._raster_type_lab)

        GCII.verify()
        GCII.compute_intersect_indices()

        gtiff_vals_cls = ExtractGTiffValues(verbose=verbose)

        gtiff_vals_cls.set_input(path_to_ras)
        gtiff_vals_cls.set_output(path_to_output)

        gtiff_vals_cls.extract_values(GCII.get_intersect_indices())

    elif input_ras_type == 'nc':
        nc_crds_cls = ExtractNetCDFCoords(verbose=verbose)

        nc_crds_cls.set_input(path_to_ras, nc_x_crds_label, nc_y_crds_label)
        nc_crds_cls.extract_coordinates()

        GCII.set_geometries(EP.get_points(), EP._geom_type)

        GCII.set_coordinates(
            nc_crds_cls.get_x_coordinates(),
            nc_crds_cls.get_y_coordinates(),
            nc_crds_cls._raster_type_lab)

        GCII.verify()
        GCII.compute_intersect_indices()

        for nc_variable_label in nc_variable_labels:
            nc_vals_cls = ExtractNetCDFValues(verbose=verbose)

            nc_vals_cls.set_input(
                path_to_ras, nc_variable_label, nc_time_label)
            nc_vals_cls.set_output(path_to_output)

            nc_vals_cls.extract_values(GCII.get_intersect_indices())

    else:
        raise NotImplementedError

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
