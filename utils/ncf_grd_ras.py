# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

31.01.2025

17:12:29

Description:

Keywords:

'''

import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

# import pyproj
import numpy as np
import netCDF4 as nc
from osgeo import gdal

gdal.UseExceptions()

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\TUM\Colleagues_Students\hadi\modis')
    os.chdir(main_dir)

    ncf_pth = Path(r'P:\Synchronize\TUM\Colleagues_Students\hadi\modis\v7_1\bk_merged_mod_myd_v7\MOD10A1.061_500m.nc')
    ncf_xcs_lbl = 'xdim'
    ncf_ycs_lbl = 'ydim'
    # ncf_cds_str = pyproj.crs.CRS.from_epsg('4326').to_wkt()
    ncf_cds_str = 'PROJCS["unknown",GEOGCS["unknown",DATUM["unknown",SPHEROID["unknown",6371007.181,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

    ras_pth = Path(r'modis_sin_grid.tif')
    #==========================================================================

    ncf_hdl = nc.Dataset(ncf_pth)
    xcs = ncf_hdl[ncf_xcs_lbl][:].data
    ycs = ncf_hdl[ncf_ycs_lbl][:].data

    assert xcs.ndim == ycs.ndim == 1, (xcs.ndim, ycs.ndim)

    assert xcs[0] < xcs[-1], (xcs[0], xcs[-1])
    assert ycs[0] > ycs[-1], (ycs[0], ycs[-1])

    assert np.isclose((xcs[1:] - xcs[:-1]), xcs[1] - xcs[0]).all()
    assert np.isclose((ycs[:-1] - ycs[1:]), ycs[0] - ycs[1]).all()
    #==========================================================================

    # ref_gtp = {
    #     np.float32: gdal.GDT_Float32,
    #     np.float64: gdal.GDT_Float64,
    #     np.int32: gdal.GDT_Int32,
    #     np.uint32: gdal.GDT_UInt32,
    #     np.int16: gdal.GDT_Int16,
    #     np.uint16: gdal.GDT_UInt16,
    #     np.int8: gdal.GDT_Byte,
    #     }[ras_arr.dtype.type]
    #==========================================================================

    out_dvr = gdal.GetDriverByName('GTiff')

    out_ras = out_dvr.Create(
        ras_pth,
        xcs.size,  # Columns.
        ycs.size,  # Rows.
        1,  # Bands.
        gdal.GDT_Float64,  # dtype.
        options=['COMPRESS=LZW'])

    assert out_ras is not None
    #==========================================================================

    ras_arr = np.ones((ycs.size, xcs.size), dtype=np.int8)

    out_bnd = out_ras.GetRasterBand(1)
    out_bnd.WriteArray(ras_arr)
    out_bnd.SetNoDataValue(0)
    #==========================================================================

    out_ras.SetGeoTransform(
        [xcs[0] - (0.5 * (xcs[1] - xcs[0])),
         xcs[1] - xcs[0],
         0,
         ycs[0] - (0.5 * (ycs[1] - ycs[0])),
         0,
         ycs[1] - ycs[0]])

    out_ras.SetProjection(ncf_cds_str)

    out_ras = None

    ncf_hdl.close()
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
