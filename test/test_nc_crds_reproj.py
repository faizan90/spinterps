# -*- coding: utf-8 -*-

'''
@author: Faizan-Uni-Stuttgart

Feb 9, 2023

1:23:27 PM

Keywords:

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

from spinterps import CrdsReProjNC

DEBUG_FLAG = False


def main():

    main_dir = Path(r'U:\fradnc')
    os.chdir(main_dir)

    path_to_ncs = main_dir.glob('2023.nc')

    src_crs = 'EPSG:4326'
    dst_crs = 'EPSG:32632'
    src_x_lab = 'lon'
    src_y_lab = 'lat'
    dst_x_lab = 'x_utm32n'
    dst_y_lab = 'y_utm32n'
    dim_x_lab = 'dim_lon'  # 'longitude'  # 'dimx'
    dim_y_lab = 'dim_lat'  # 'latitude'  # 'dimy'
    data_var = 'RW'

    verbose = True
    #==========================================================================

    for path_to_nc in path_to_ncs:

        print(path_to_nc)

        reproj_nc_cls = CrdsReProjNC(verbose)

        reproj_nc_cls.append_crds(
            path_to_nc,
            src_crs,
            dst_crs,
            src_x_lab,
            src_y_lab,
            dst_x_lab,
            dst_y_lab,
            dim_x_lab,
            dim_y_lab,
            data_var)

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
