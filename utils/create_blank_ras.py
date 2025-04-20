# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

17.04.2025

09:33:01

Description:

Keywords:

'''

import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import pyproj
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\DEBY_ISAR\dem')
    os.chdir(main_dir)

    # 100m
    ipt_xmn = 3925000
    ipt_xmx = 4300000
    ipt_ymn = 2155100
    ipt_ymx = 2592600

    cel_sze = 100

    cds_str = pyproj.crs.CRS.from_epsg('3034').to_wkt()

    ndv = np.nan

    otp_pth = Path(rf'v1_{cel_sze}m/blank_raster_{cel_sze}m.tif')
    #==========================================================================

    otp_pth.parents[0].mkdir(exist_ok=True, parents=True)

    assert cel_sze > 0, cel_sze

    assert (ipt_xmx - ipt_xmn) > cel_sze, (ipt_xmn, ipt_xmx, cel_sze)
    assert (ipt_ymx - ipt_ymn) > cel_sze, (ipt_ymn, ipt_ymx, cel_sze)

    assert (ipt_xmn % cel_sze) == 0, (ipt_xmn, cel_sze, ipt_xmn % cel_sze)
    assert (ipt_xmx % cel_sze) == 0, (ipt_xmx, cel_sze, ipt_xmx % cel_sze)
    assert (ipt_ymn % cel_sze) == 0, (ipt_ymn, cel_sze, ipt_ymn % cel_sze)
    assert (ipt_ymx % cel_sze) == 0, (ipt_ymx, cel_sze, ipt_ymx % cel_sze)

    cnt_col = (ipt_xmx - ipt_xmn) // cel_sze
    cnt_row = (ipt_ymx - ipt_ymn) // cel_sze

    otp_dvr = gdal.GetDriverByName('GTiff')

    otp_ras_hdl = otp_dvr.Create(
        str(otp_pth.absolute()),
        cnt_col,
        cnt_row,
        1,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW'])

    assert otp_ras_hdl, 'Creating new raster failed!'
    otp_ras_hdl.SetGeoTransform([ipt_xmn, cel_sze, 0, ipt_ymx, 0, -cel_sze])
    otp_ras_hdl.SetProjection(cds_str)

    bnd = otp_ras_hdl.GetRasterBand(1)
    bnd.WriteArray(np.zeros((cnt_row, cnt_col), dtype=np.float32))
    bnd.FlushCache()
    bnd.SetNoDataValue(ndv)
    otp_ras_hdl = None
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
