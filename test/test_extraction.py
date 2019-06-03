'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import pandas as pd

from spinterps import (
    ExtractReferencePolygons,
    ExtractNetCDFCoords,
    ExtractGTiffCoords,
    ExtractGTiffValues,
    PolyAndCrdsItsctIdxs)


def main():

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\raster\taudem_out_spate_rockenau')
    os.chdir(main_dir)

    path_to_shp = r'watersheds.shp'

    label_field = 'DN'

    ERP = ExtractReferencePolygons()

    ERP.set_path_to_poly_shp(path_to_shp, label_field)

    ERP.assemble_polygon_data()

    poly_labels = ERP.get_labels()
    poly_geoms = ERP.get_geometries()
    poly_areas = ERP.get_areas()
    poly_extents = ERP.get_extents()

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
#
    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\raster')
    os.chdir(main_dir)

    path_to_gtiff = r'lower_de_gauss_z3_2km.tif'

    EGTC = ExtractGTiffCoords()

    EGTC.set_path_to_gtiff(path_to_gtiff)

    EGTC.assemble_gtiff_data()

    x_crds = EGTC.get_x_coordinates()
    y_crds = EGTC.get_y_coordinates()

    PCII = PolyAndCrdsItsctIdxs()

    PCII.set_polygons(poly_geoms, poly_labels)
    PCII.set_coordinates(x_crds, y_crds, EGTC._raster_type_lab)

    PCII.verify()

    PCII.compute_intersect_idxs()

    itsct_idxs = PCII.get_intersect_idxs()

    EGTV = ExtractGTiffValues()

    EGTV.set_path_to_gtiff(path_to_gtiff)
    EGTV.assemble_gtiff_data()

    EGTV.extract_data_for_indices(itsct_idxs)

    extracted_values = EGTV.get_extracted_data()

    for label in itsct_idxs:
        x_crds_label = x_crds[itsct_idxs[label]['x']]
        y_crds_label = y_crds[itsct_idxs[label]['y']]

        crds_df = pd.DataFrame(data=
            {'x': x_crds_label,
             'y': y_crds_label,
             'area':itsct_idxs[label]['area'],
             'rel_area':itsct_idxs[label]['rel_area'],
             **extracted_values[label]})

        crds_df.to_csv(f'{label}.csv', sep=';', index=False)

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
