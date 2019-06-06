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
    ExtractNetCDFValues,
    ExtractGTiffCoords,
    ExtractGTiffValues,
    PolyAndCrdsItsctIdxs)


def main():

    main_dir = Path(r'Q:\Synchronize_LDs\full_neckar_precipitation_interpolation\spinterp_nc_gtiff_test')
    os.chdir(main_dir)

    path_to_h5 = r'test_nc_gtiff.h5'

    path_to_shp = r'watersheds.shp'

    label_field = 'DN'

    ERP = ExtractReferencePolygons()

    ERP.set_input(path_to_shp, label_field)

    ERP.extract_polygon_data()

    poly_labels = ERP.get_labels()[:1]

    poly_geoms = {
        poly_label: ERP.get_geometries()[poly_label]
        for poly_label in poly_labels}

#     poly_areas = ERP.get_areas()
#     poly_extents = ERP.get_extents()

    path_to_nc = r'full_neckar_ppt_interp__1961-01-01_to_2015-12-31_1km_all.nc'

    ENCC = ExtractNetCDFCoords()

    ENCC.set_input(path_to_nc, 'X', 'Y')

    ENCC.extract_coordinates()

    nc_x_crds = ENCC.get_x_coordinates()
    nc_y_crds = ENCC.get_y_coordinates()

    PCII = PolyAndCrdsItsctIdxs()

    PCII.set_polygons(poly_geoms, poly_labels)
    PCII.set_coordinates(nc_x_crds, nc_y_crds, ENCC._raster_type_lab)

    PCII.verify()

    PCII.compute_intersect_idxs()

    nc_itsct_idxs = PCII.get_intersect_idxs()

    ENCV = ExtractNetCDFValues()

    ENCV.set_input(path_to_nc, 'OK', 'time')
    ENCV.set_output(path_to_h5)

    ENCV.extract_data_for_indices_and_save(nc_itsct_idxs, True)

#     path_to_gtiff = (
#         r'lower_de_gauss_z3_1km_atkis_19_extended_hydmod_lulc_ratios.tif')
#
#     EGTC = ExtractGTiffCoords()
#
#     EGTC.set_input(path_to_gtiff)
#
#     EGTC.extract_coordinates()
#
#     gtiff_x_crds = EGTC.get_x_coordinates()
#     gtiff_y_crds = EGTC.get_y_coordinates()
#
#     PCII = PolyAndCrdsItsctIdxs()
#
#     PCII.set_polygons(poly_geoms, poly_labels)
#     PCII.set_coordinates(gtiff_x_crds, gtiff_y_crds, EGTC._raster_type_lab)
#
#     PCII.verify()
#
#     PCII.compute_intersect_idxs()
#
#     gtiff_itsct_idxs = PCII.get_intersect_idxs()
#
#     EGTV = ExtractGTiffValues()
#
#     EGTV.set_input(path_to_gtiff)
#     EGTV.set_output(path_to_h5)
# #     EGTV.set_output(None)
#
#     EGTV.extract_data_for_indices_and_save(gtiff_itsct_idxs)

#     extracted_values = EGTV.get_extracted_data()

#     ENCV = ExtractNetCDFValues()
#
#     ENCV.set_input(path_to_nc, 'SK', 'time')
#     ENCV.set_output('dfdf.h5')
# #     ENCV.set_output(None)
#
#     ENCV.extract_data_for_indices_and_save(itsct_idxs, True)

#     extracted_values = ENCV.get_extracted_data()
#
#     for label in itsct_idxs:
# #         x_crds_label = x_crds[itsct_idxs[label]['cols']]
# #         y_crds_label = y_crds[itsct_idxs[label]['rows']]
#
#         x_crds_label = x_crds[
#             itsct_idxs[label]['rows'], itsct_idxs[label]['cols']]
#
#         y_crds_label = y_crds[
#             itsct_idxs[label]['rows'], itsct_idxs[label]['cols']]
#
#         crds_df = pd.DataFrame(data=
#             {'x': x_crds_label,
#              'y': y_crds_label,
#              'rows':itsct_idxs[label]['rows'],
#              'cols':itsct_idxs[label]['cols'],
#              'itsctd_area':itsct_idxs[label]['itsctd_area'],
#              'rel_itsctd_area':itsct_idxs[label]['rel_itsctd_area'],
#              'x_cen_crds':itsct_idxs[label]['x_cen_crds'],
#              'y_cen_crds':itsct_idxs[label]['y_cen_crds'],
#              **extracted_values[label]})
#
#         crds_df.to_csv(f'{label}.csv', sep=';', index=False)

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
