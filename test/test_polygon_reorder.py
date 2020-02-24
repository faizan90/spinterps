'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import pandas as pd

from spinterps import (
    ExtractPolygons,
    ExtractNetCDFCoords,
    ExtractNetCDFValues,
    ExtractGTiffCoords,
    ExtractGTiffValues,
    GeomAndCrdsItsctIdxs,
    ReOrderIdxs)


def main():

    main_dir = Path(r'P:\Downloads\spinterp_nc_gtiff_test')
    os.chdir(main_dir)

    path_to_shp = r'watersheds.shp'

    label_field = r'DN'

    path_to_nc = r'tem_spinterp.nc'

    path_to_ras = (
        r'lower_de_gauss_z3_2km_atkis_19_extended_hydmod_lulc_ratios.tif')

    nc_x_crds_label = 'X'
    nc_y_crds_label = 'Y'
    nc_variable_labels = ['OK', 'SK']
    nc_time_label = 'time'

    reorder_sequence = 'nc_gtiff'
#     reorder_sequence = 'gtiff_nc'

    maximum_reorder_threshold_distance = 100

    path_to_h5_output = 'test.h5'

    save_xy_crds_txt_flag = True

    verbose = True

    ##########################################################################
    poly_cls = ExtractPolygons(verbose=verbose)

    poly_cls.set_input(path_to_shp, label_field)

    poly_cls.extract_polygons()

    assert isinstance(nc_variable_labels, (list, tuple)), (
        'variable_labels can only be a list or tuple having strings!')

    assert all([isinstance(x, str) for x in nc_variable_labels]), (
        'variable_labels can only be a list or tuple having strings!')

    ##########################################################################

    if reorder_sequence == 'nc_gtiff':
        nc_crds_cls = ExtractNetCDFCoords(verbose=verbose)

        nc_crds_cls.set_input(path_to_nc, nc_x_crds_label, nc_y_crds_label)

        nc_crds_cls.extract_coordinates()

        nc_itsct_cls = GeomAndCrdsItsctIdxs(verbose=verbose)

        nc_itsct_cls.set_geometries(
            poly_cls.get_polygons(), poly_cls._geom_type)

        nc_itsct_cls.set_coordinates(
            nc_crds_cls.get_x_coordinates(),
            nc_crds_cls.get_y_coordinates(),
            nc_crds_cls._raster_type_lab)

        nc_itsct_cls.verify()

        nc_itsct_cls.compute_intersect_indices()

        if save_xy_crds_txt_flag:
            for label in nc_itsct_cls.get_intersect_indices():
                x = nc_itsct_cls.get_intersect_indices()[label]['x_cen_crds']
                y = nc_itsct_cls.get_intersect_indices()[label]['y_cen_crds']

                pd.DataFrame(data={'x': x, 'y': y}).to_csv(
                    f'{label}.csv', index=False, sep=';')

#         for nc_variable_label in nc_variable_labels:
#             nc_vals_cls = ExtractNetCDFValues(verbose=verbose)
#
#             nc_vals_cls.set_input(path_to_nc, nc_variable_label, nc_time_label)
#             nc_vals_cls.set_output(path_to_h5_output)
#
#             nc_vals_cls.extract_values(nc_itsct_cls.get_intersect_indices())

        gtiff_crds_cls = ExtractGTiffCoords(verbose=verbose)

        gtiff_crds_cls.set_input(path_to_ras)

        gtiff_crds_cls.extract_coordinates()

        ras_itsct_cls = GeomAndCrdsItsctIdxs()

        ras_itsct_cls.set_geometries(
            poly_cls.get_polygons(), poly_cls._geom_type)

        ras_itsct_cls.set_coordinates(
            gtiff_crds_cls.get_x_coordinates(),
            gtiff_crds_cls.get_y_coordinates(),
            gtiff_crds_cls._raster_type_lab)

        ras_itsct_cls.verify()

        ras_itsct_cls.compute_intersect_indices()

        reorder_idxs_cls = ReOrderIdxs(verbose=verbose)

        reorder_idxs_cls.set_reference(nc_itsct_cls.get_intersect_indices())
        reorder_idxs_cls.set_destination(ras_itsct_cls.get_intersect_indices())

        reorder_idxs_cls.reorder(maximum_reorder_threshold_distance)

        # without rordering
#         ras_idxs = ras_itsct_cls.get_intersect_indices()
#         ignore_rows_cols_equality = False

        # with reordering
        ras_idxs = reorder_idxs_cls.get_reordered_destination()
        ignore_rows_cols_equality = True

        gtiff_vals_cls = ExtractGTiffValues(verbose=verbose)

        gtiff_vals_cls.set_input(path_to_ras)
        gtiff_vals_cls.set_output(path_to_h5_output)

        gtiff_vals_cls.extract_values(
            ras_idxs,
            ignore_rows_cols_equality=ignore_rows_cols_equality)

    ##########################################################################

    elif reorder_sequence == 'gtiff_nc':
        gtiff_crds_cls = ExtractGTiffCoords(verbose=verbose)

        gtiff_crds_cls.set_input(path_to_ras)

        gtiff_crds_cls.extract_coordinates()

        ras_itsct_cls = GeomAndCrdsItsctIdxs()

        ras_itsct_cls.set_geometries(
            poly_cls.get_polygons(), poly_cls._geom_type)

        ras_itsct_cls.set_coordinates(
            gtiff_crds_cls.get_x_coordinates(),
            gtiff_crds_cls.get_y_coordinates(),
            gtiff_crds_cls._raster_type_lab)

        ras_itsct_cls.verify()

        ras_itsct_cls.compute_intersect_indices()

        if save_xy_crds_txt_flag:
            for label in ras_itsct_cls.get_intersect_indices():
                x = ras_itsct_cls.get_intersect_indices()[label]['x_cen_crds']
                y = ras_itsct_cls.get_intersect_indices()[label]['y_cen_crds']

                pd.DataFrame(data={'x': x, 'y': y}).to_csv(
                    f'{label}.csv', index=False, sep=';')

        gtiff_vals_cls = ExtractGTiffValues(verbose=verbose)

        gtiff_vals_cls.set_input(path_to_ras)
        gtiff_vals_cls.set_output(path_to_h5_output)

        gtiff_vals_cls.extract_values(ras_itsct_cls.get_intersect_indices())

        nc_crds_cls = ExtractNetCDFCoords(verbose=verbose)

        nc_crds_cls.set_input(path_to_nc, nc_x_crds_label, nc_y_crds_label)

        nc_crds_cls.extract_coordinates()

        nc_itsct_cls = GeomAndCrdsItsctIdxs(verbose=verbose)

        nc_itsct_cls.set_geometries(
            poly_cls.get_polygons(), poly_cls._geom_type)

        nc_itsct_cls.set_coordinates(
            nc_crds_cls.get_x_coordinates(),
            nc_crds_cls.get_y_coordinates(),
            nc_crds_cls._raster_type_lab)

        nc_itsct_cls.verify()

        nc_itsct_cls.compute_intersect_indices()

        reorder_idxs_cls = ReOrderIdxs(verbose=verbose)

        reorder_idxs_cls.set_reference(ras_itsct_cls.get_intersect_indices())
        reorder_idxs_cls.set_destination(nc_itsct_cls.get_intersect_indices())

        reorder_idxs_cls.reorder(maximum_reorder_threshold_distance)

        # without rordering
#         ras_idxs = ras_itsct_cls.get_intersect_indices()
#         ignore_rows_cols_equality = False

        # with reordering
        nc_idxs = reorder_idxs_cls.get_reordered_destination()
        ignore_rows_cols_equality = True

        for nc_variable_label in nc_variable_labels:
            nc_vals_cls = ExtractNetCDFValues(verbose=verbose)

            nc_vals_cls.set_input(path_to_nc, nc_variable_label, nc_time_label)
            nc_vals_cls.set_output(path_to_h5_output)

            nc_vals_cls.extract_values(
                nc_idxs,
                ignore_rows_cols_equality=ignore_rows_cols_equality)

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
