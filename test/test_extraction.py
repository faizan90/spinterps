'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

from spinterps import (
    ExtractReferencePolygons, ExtractNetCDFCoords, ExtractGTiffCoords)


def main():

#     main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\raster\taudem_out_spate_rockenau')
#     os.chdir(main_dir)
#
#     path_to_shp = r'watersheds.shp'
#
#     label_field = 'DN'
#
#     ERP = ExtractReferencePolygons()
#
#     ERP.set_path_to_poly_shp(path_to_shp, label_field)
#
#     ERP.assemble_polygon_data()
#
#     poly_labels = ERP.get_labels()
#     poly_geoms = ERP.get_geometries()
#     poly_areas = ERP.get_areas()
#     poly_extents = ERP.get_extents()

#     main_dir = Path(r'Q:\Synchronize_LDs\Mulde_pet_kriging_20190417')
#     os.chdir(main_dir)
#
#     path_to_nc = r'mulde_pet_kriging_1950-01-01_to_2015-12-31_1km_all.nc'
#
#     ENCC = ExtractNetCDFCoords()
#
#     ENCC.set_netcdf_properties(path_to_nc, 'X', 'Y')
#
#     ENCC.assemble_netcdf_data()
#
#     nc_x_crds = ENCC.get_x_coordinates()
#     nc_y_crds = ENCC.get_y_coordinates()

#     main_dir = Path(r'Q:\Synchronize_LDs\Mulde_pet_kriging_20190417')
#     os.chdir(main_dir)
#
#     path_to_gtiff = r'mulde_pet_kriging_1950-01-01_to_2015-12-31_1km_all.nc'

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\raster\taudem_out_spate_rockenau')
    os.chdir(main_dir)

    path_to_gtiff = r'fil.tif'

    EGTC = ExtractGTiffCoords()

    EGTC.set_path_to_gtiff(path_to_gtiff)

    EGTC.assemble_gtiff_data()

    gtiff_x_crds = EGTC.get_x_coordinates()
    gtiff_y_crds = EGTC.get_y_coordinates()

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
