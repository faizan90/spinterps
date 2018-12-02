'''
Created on Nov 25, 2018

@author: Faizan
'''
import os

import ogr
import gdal
import numpy as np
import psutil as ps


def get_current_proc_size(mb=False):

    interpreter_size = ps.Process(os.getpid()).memory_info().vms

    if mb:
        megabytes = 1024 ** 2
        interpreter_size //= megabytes

    return interpreter_size


def ret_mp_idxs(n_vals, n_cpus):

    idxs = np.linspace(0, n_vals, n_cpus + 1, endpoint=True, dtype=np.int64)

    idxs = np.unique(idxs)

    assert idxs.shape[0]

    if idxs.shape[0] == 1:
        idxs = np.concatenate((np.array([0]), idxs))

    assert (idxs[0] == 0) & (idxs[-1] == n_vals), idxs
    return idxs


def cnvt_to_pt(x, y):

    """Convert x, y coordinates to a point string in POINT(x y) format"""

    return ogr.CreateGeometryFromWkt("POINT (%f %f)" % (x, y))


def chk_cntmt(pt, bbx):

    """Containment check of points in a given polygon"""

    return bbx.Contains(pt)


def get_ras_props(in_ras, in_band_no=1):

    """
    Purpose: To return a given raster's extents, number of rows and columns
    pixel size in x and y direction, projection, noData value
    band count and GDAL data type using GDAL as a list.

    Description of the arguments:
        in_ras (string): Full path to the input raster. If the raster cannot
            be read by GDAL then the function returns None.

        in_band_no (int): The band which we want to use (starting from 1).
            Defaults to 1. Used for getting NDV.
    """

    in_ds = gdal.Open(in_ras, 0)
    if in_ds is not None:
        rows = in_ds.RasterYSize
        cols = in_ds.RasterXSize

        geotransform = in_ds.GetGeoTransform()
        x_min = geotransform[0]
        y_max = geotransform[3]

        pix_width = geotransform[1]
        pix_height = abs(geotransform[5])

        x_max = x_min + (cols * pix_width)
        y_min = y_max - (rows * pix_height)

        proj = in_ds.GetProjectionRef()

        in_band = in_ds.GetRasterBand(in_band_no)
        if in_band is not None:
            NDV = in_band.GetNoDataValue()
            gdt_type = in_band.DataType
        else:
            NDV = None
            gdt_type = None
        band_count = in_ds.RasterCount

        ras_props = [x_min,  # 0
                     x_max,  # 1
                     y_min,  # 2
                     y_max,  # 3
                     cols,  # 4
                     rows,  # 5
                     pix_width,  # 6
                     pix_height,  # 7
                     proj,  # 8
                     NDV,  # 9
                     band_count,  # 10
                     gdt_type  # 11
                     ]

        in_ds = None
        return ras_props

    else:
        raise IOError(
            ('Could not read the input raster (%s). Check path and file!') %
            in_ras)
    return


def get_aligned_shp_bds_and_cell_size(bounds_shp_file, align_ras_file):

    in_ds = ogr.Open(str(bounds_shp_file))
    assert in_ds, f'Could not open {bounds_shp_file}!'

    lyr_count = in_ds.GetLayerCount()

    assert lyr_count, f'No layers in {bounds_shp_file}!'
    assert lyr_count == 1, f'More than one layer in {bounds_shp_file}!'

    in_lyr = in_ds.GetLayer(0)
    envelope = in_lyr.GetExtent()

    assert envelope, f'No envelope for {bounds_shp_file}!'
    in_ds.Destroy()

    raw_shp_x_min, raw_shp_x_max, raw_shp_y_min, raw_shp_y_max = envelope

    ras_props = get_ras_props(str(align_ras_file))
    ras_cell_size, _1 = ras_props[6:8]

    assert np.isclose(ras_cell_size, _1), (
        f'align_ras ({align_ras_file}) not square!')

    ras_min_x, ras_max_x = ras_props[:2]
    ras_min_y, ras_max_y = ras_props[2:4]

    assert all(
        ((raw_shp_x_min >= ras_min_x),
         (raw_shp_x_max <= ras_max_x),
         (raw_shp_y_min >= ras_min_y),
         (raw_shp_y_max <= ras_max_y))), (
             f'bounds_shp ({bounds_shp_file}) outside of '
             f'align_ras ({align_ras_file})!')

    rem_min_col_width = ((raw_shp_x_min - ras_min_x) / ras_cell_size) % 1
    rem_min_row_width = ((ras_max_y - raw_shp_y_max) / ras_cell_size) % 1

    x_min_adj = rem_min_col_width * ras_cell_size
    y_max_adj = rem_min_row_width * ras_cell_size

    adj_shp_x_min = raw_shp_x_min - x_min_adj
    adj_shp_y_max = raw_shp_y_max + y_max_adj

    rem_max_col_width = ((raw_shp_x_max - ras_min_x) / ras_cell_size) % 1
    rem_max_row_width = ((ras_max_y - raw_shp_y_min) / ras_cell_size) % 1

    x_max_adj = rem_max_col_width * ras_cell_size
    y_min_adj = rem_max_row_width * ras_cell_size

    adj_shp_x_max = raw_shp_x_max + (ras_cell_size - x_max_adj)
    adj_shp_y_min = raw_shp_y_min - (ras_cell_size - y_min_adj)

    return (
        (adj_shp_x_min, adj_shp_x_max, adj_shp_y_min, adj_shp_y_max),
        ras_cell_size)
