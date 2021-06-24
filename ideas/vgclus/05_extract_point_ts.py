'''
@author: Faizan-Uni-Stuttgart

Nov 17, 2020

6:58:25 PM

'''
import os
import time
import timeit
from pathlib import Path

import pandas as pd

from spinterps import (
    ExtractPoints,
    ExtractNetCDFCoords,
    GeomAndCrdsItsctIdxs,
    ExtractNetCDFValues)

DEBUG_FLAG = False


def main():

    main_dir = Path(r'T:\Synchronize_LDs\full_neckar_precipitation_interpolation')
    os.chdir(main_dir)

    path_to_shp = r'P:\Synchronize\IWS\Colleagues_Students\Bianca\neckar_interps\ppt\stns_within_ppt.shp'
    label_field = 'STN_ID'

    path_to_ras = Path(r'full_neckar_ppt_interp__1961-01-01_to_2015-12-31_1km_all.nc')

    nc_x_crds_label = 'X'
    nc_y_crds_label = 'Y'
    nc_variable_labels = ['OK', 'EDK']
    nc_time_label = 'time'

    sep = ';'
    float_fmt = '%0.2f'

    out_dir = path_to_ras.parents[0]

    verbose = True

    EP = ExtractPoints(verbose=verbose)

    EP.set_input(path_to_shp, label_field)

    EP.extract_points()

    GCII = GeomAndCrdsItsctIdxs(verbose=verbose)

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

    out_crds = GCII.get_intersect_indices()

    cols = list(out_crds.keys())
    out_crds_df = pd.DataFrame(columns=['X', 'Y'], dtype=float)
    for col in cols:
        out_crds_df.loc[col, ['X', 'Y']] = (
            out_crds[col]['x_crds'][0], out_crds[col]['y_crds'][0])

    out_crds_df.to_csv(
        out_dir / f'ts_crds.csv',
        sep=sep,
        float_format=float_fmt)

    for nc_variable_label in nc_variable_labels:
        nc_vals_cls = ExtractNetCDFValues(verbose=verbose)

        nc_vals_cls.set_input(
            path_to_ras, nc_variable_label, nc_time_label)
        nc_vals_cls.set_output(None)

        nc_vals_cls.extract_values(GCII.get_intersect_indices())

        out_vals = nc_vals_cls.get_values()

        cols = list(out_vals.keys())
        out_df = None
        for col in cols:
            out_ser = pd.Series(
                index=pd.to_datetime(
                    list(out_vals[col].keys()),
                    format=nc_vals_cls._time_strs_fmt),
                data=list(out_vals[col].values()),
                dtype=float)

            if out_df is None:
                out_df = pd.DataFrame(data={col: out_ser})

            else:
                out_df[col] = out_ser

        out_df.to_csv(
            out_dir / f'ts_{nc_variable_label}.csv',
            sep=sep,
            float_format=float_fmt)

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
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
