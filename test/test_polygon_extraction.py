'''
@author: Faizan-Uni-Stuttgart

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

from spinterps import Extract

DEBUG_FLAG = True


def main():

    main_dir = Path(r'P:\spinterps\DEBY')
    os.chdir(main_dir)

    path_to_shp = r'watersheds.shp'
    label_field = r'DN'  # 'Subbasin'  # 'PolygonId'  #

    # RADOLAN.
    # paths_to_rass = [Path(f'{year}.nc') for year in range(2006, 2022 + 1)]

    # paths_to_rass = main_dir.glob('chirps-*.nc')
    paths_to_rass = [
        # Path(r'T:\TUM\projects\altoetting\spinterps\ppt_1D_gkd_dwd_1km_bay/kriging.nc'),
        # Path(r'T:\TUM\projects\altoetting\spinterps\tem_1D_tg_gkd_dwd_1km_bay/kriging.nc'),
        # Path(r'T:\TUM\projects\altoetting\spinterps\pet_1D_gkd_dwd_1km_bay/kriging.nc'),
        # Path(r'T:\TUM\projects\altoetting\spinterps\ppt_1H_gkd_dwd_1km_bay/kriging.nc'),
        # Path(r'T:\TUM\projects\altoetting\spinterps\tem_1H_gkd_dwd_1km_bay/kriging.nc'),
        # Path(r'T:\TUM\projects\altoetting\spinterps\pet_1H_gkd_dwd_1km_bay/kriging.nc'),
        # Path(r'kriging_regen_tem_1D_tg.nc'),
        # Path(r'ppt_1D_gkd_dwd_1km_bay/kriging.nc'),
        Path(r'ppt_1D_gkd_dwd_1km_bay_infilled/kriging.nc'),
        # Path(r'2016.nc'),
        ]

    input_ras_type = 'nc'

    # paths_to_rass = [Path(r'uebk25by20230116_utm32n.tif')]
    # input_ras_type = 'gtiff'

    nc_x_crds_label = 'X'
    nc_y_crds_label = 'Y'
    nc_variable_labels = ['EDK', ]
    nc_time_label = 'time'

    # nc_x_crds_label = 'x_utm32n'
    # nc_y_crds_label = 'y_utm32n'
    # nc_variable_labels = ['RW']
    # nc_time_label = 'time'

    # h5 means tss of pts in h5, nc snips it, csv/pkl lumps it in space.
    # NOTE: In case of invalid numerical values, use raw
    #       output format and post-process it outside.
    # In case of h5 or csv, if ncs are too big. Snip them as nc and
    # then produce the h5 files.
    out_ext = 'pkl'

    out_suff = 'deby2'

    src_epsg = None
    dst_epsg = None

    simplify_tol_ratio = 0.0  # 0.01
    minimum_cell_area_intersection_percentage = 1e-3
    buffer_distance = 10e3
    n_cpus = 'auto'

    out_dir = main_dir  # / 'regen_ecad'
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    Ext = Extract(True)

    res = None

    paths_to_rass = list(paths_to_rass)

    assert len(paths_to_rass)

    if input_ras_type == 'gtiff':

        for path_to_ras in paths_to_rass:

            if out_suff:
                path_to_output = out_dir / Path(
                    rf'{path_to_ras.stem}_{out_suff}.{out_ext}')

            else:
                path_to_output = out_dir / Path(
                    rf'{path_to_ras.stem}.{out_ext}')

            res = Ext.extract_from_geotiff(
                path_to_shp,
                label_field,
                path_to_ras,
                path_to_output,
                minimum_cell_area_intersection_percentage,
                n_cpus,
                src_epsg,
                dst_epsg,
                simplify_tol_ratio,
                buffer_distance)

    elif input_ras_type == 'nc':

        if False:
            for path_to_ras in paths_to_rass:

                # path_to_output = out_dir / Path(
                #     rf'{path_to_ras.stem}_{out_suff}.{out_ext}')

                if out_suff:
                    path_to_output = out_dir / Path(
                        rf'{path_to_ras.stem}_{out_suff}.{out_ext}')

                else:
                    path_to_output = out_dir / Path(
                        rf'{path_to_ras.stem}.{out_ext}')

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
                    simplify_tol_ratio,
                    buffer_distance)

        else:
            paths_to_outputs = []
            for path_to_ras in paths_to_rass:

                # path_to_output = path_to_ras.parents[0] / (
                #     rf'{path_to_ras.stem}_{out_suff}.{out_ext}')

                if out_suff:
                    path_to_output = path_to_ras.parents[0] / Path(
                        rf'{path_to_ras.stem}_{out_suff}.{out_ext}')

                else:
                    path_to_output = path_to_ras.parents[0] / Path(
                        rf'{path_to_ras.stem}.{out_ext}')

                paths_to_outputs.append(path_to_output)

            res = Ext.extract_from_netCDF_batch(
                path_to_shp,
                label_field,
                paths_to_rass,
                paths_to_outputs,
                nc_x_crds_label,
                nc_y_crds_label,
                nc_variable_labels,
                nc_time_label,
                minimum_cell_area_intersection_percentage,
                n_cpus,
                src_epsg,
                dst_epsg,
                simplify_tol_ratio,
                buffer_distance)

    else:
        raise NotImplementedError

    print('\n')
    print('res:', res)

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

