'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spinterps import (
    ExtractPoints,
    ExtractGTiffCoords,
    ExtractNetCDFCoords,
    GeomAndCrdsItsctIdxs,
    ExtractGTiffValues,
    ExtractNetCDFValues)


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Colleagues_Students\Jochen\plettenberg')
    os.chdir(main_dir)

    path_to_shp = 'crds_wgs84.shp'
    label_field = 'stn'

    path_to_ras = r'U:\fradnc\2023.nc'
    input_ras_type = 'nc'

#     path_to_ras = (
#         r'lower_de_gauss_z3_2km_atkis_19_extended_hydmod_lulc_ratios.tif')
#     input_ras_type = 'gtiff'

    nc_x_crds_label = 'lon'
    nc_y_crds_label = 'lat'
    nc_variable_labels = ['RW']
    nc_time_label = 'time'

    src_epsg = None
    dst_epsg = None

    # It can be None, or have and extension of h5 or if save_as_txt_flag is
    # set then the directory. pref is used a along with the data
    # label as the file name.
    path_to_output = main_dir

    # For output as text.
    save_as_txt_flag = True
    pref = 'ts_radolan_hourly_2023'
    sep = ';'

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

        GCII.set_coordinate_system_transforms(src_epsg, dst_epsg)

        GCII.verify()
        GCII.compute_intersect_indices()

        gtiff_vals_cls = ExtractGTiffValues(verbose=verbose)

        gtiff_vals_cls.set_input(path_to_ras)

        if save_as_txt_flag:
            gtiff_vals_cls.set_output(None)

        else:
            gtiff_vals_cls.set_output(path_to_output)

        gtiff_vals_cls.extract_values(GCII.get_intersect_indices())

        if save_as_txt_flag:
            extd_vals = gtiff_vals_cls.get_values()

            pt_sers = []
            for key, value in extd_vals.items():
                pt_ser = pd.Series(
                    index=value.keys(), data=value.values(), name=key)

                pt_sers.append(pt_ser)

            pt_df = pd.concat(pt_sers, axis=1)
            pt_ser = None

            out_df_path = (
                path_to_output / f'{pref}.csv')

            pt_df.to_csv(out_df_path, sep=sep)

    elif input_ras_type == 'nc':
        nc_crds_cls = ExtractNetCDFCoords(verbose=verbose)

        nc_crds_cls.set_input(path_to_ras, nc_x_crds_label, nc_y_crds_label)
        nc_crds_cls.extract_coordinates()

        GCII.set_geometries(EP.get_points(), EP._geom_type)

        GCII.set_coordinates(
            nc_crds_cls.get_x_coordinates(),
            nc_crds_cls.get_y_coordinates(),
            nc_crds_cls._raster_type_lab)

        GCII.set_coordinate_system_transforms(src_epsg, dst_epsg)

        GCII.verify()
        GCII.compute_intersect_indices()

        for nc_variable_label in nc_variable_labels:
            nc_vals_cls = ExtractNetCDFValues(verbose=verbose)

            nc_vals_cls.set_input(
                path_to_ras, nc_variable_label, nc_time_label)

            if save_as_txt_flag:
                nc_vals_cls.set_output(None)

            else:
                nc_vals_cls.set_output(path_to_output)

            nc_vals_cls.extract_values(GCII.get_intersect_indices())

            if save_as_txt_flag:
                extd_vals = nc_vals_cls.get_values()

                pt_sers = []
                for key, value in extd_vals.items():
                    pt_ser = pd.Series(
                        index=value.keys(),
                        data=np.concatenate(list(value.values()), axis=0),
                        name=key)

                    pt_sers.append(pt_ser)

                pt_df = pd.concat(pt_sers, axis=1)

                out_df_path = (
                    path_to_output / f'{pref}_{nc_variable_label}.csv')

                pt_df.to_csv(out_df_path, sep=sep)

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
