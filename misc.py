'''
Created on Nov 25, 2018

@author: Faizan
'''
import os
import sys
from functools import wraps
import traceback as tb

import ogr
import gdal
import numpy as np
import psutil as ps
import netCDF4 as nc
from netcdftime import utime, datetime

from .cyth import fill_dists_2d_mat

print_line_str = 40 * '#'


def print_sl():

    print(2 * '\n', print_line_str, sep='')
    return


def print_el():

    print(print_line_str)
    return


def traceback_wrapper(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        func_res = None

        try:
            func_res = func(*args, **kwargs)

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

        return func_res

    return wrapper


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

    """Containment check of a point in a given polygon"""

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

    n_round = 6

    in_ds = ogr.Open(str(bounds_shp_file))
    assert in_ds, f'Could not open {bounds_shp_file}!'

    lyr_count = in_ds.GetLayerCount()

    assert lyr_count, f'No layers in {bounds_shp_file}!'
    assert lyr_count == 1, f'More than one layer in {bounds_shp_file}!'

    in_lyr = in_ds.GetLayer(0)
    envelope = in_lyr.GetExtent()

    assert envelope, f'No envelope for {bounds_shp_file}!'
    in_ds.Destroy()

    raw_shp_x_min, raw_shp_x_max, raw_shp_y_min, raw_shp_y_max = np.round(
        envelope, n_round)

    ras_props = get_ras_props(str(align_ras_file))
    ras_cell_size, _1 = np.round(ras_props[6:8], n_round)

    assert np.isclose(ras_cell_size, _1), (
        f'align_ras ({align_ras_file}) not square!')

    ras_min_x, ras_max_x = np.round(ras_props[:2], n_round)
    ras_min_y, ras_max_y = np.round(ras_props[2:4], n_round)

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


class GdalErrorHandler:

    '''Because of some annoying geometry area operation warning.'''

    def __init__(self):
        self.err_level = gdal.CE_None
        self.err_no = 0
        self.err_msg = ''

    def handler(self, err_level, err_no, err_msg):
        self.err_level = err_level
        self.err_no = err_no
        self.err_msg = err_msg


gdal_err_hdl = GdalErrorHandler()
gdal_err_hdlr = gdal_err_hdl.handler


def add_month(date, months_to_add):

    """
    Finds the next month from date.

    :param netcdftime.datetime date: Accepts datetime or phony datetime
        from ``netCDF4.num2date``.
    :param int months_to_add: The number of months to add to the date
    :returns: The final date
    :rtype: *netcdftime.datetime*
    """

    years_to_add = int((
        date.month +
        months_to_add -
        np.mod(date.month + months_to_add - 1, 12) - 1) / 12)

    new_month = int(np.mod(date.month + months_to_add - 1, 12)) + 1

    new_year = date.year + years_to_add

    date_next = datetime(
        year=new_year,
        month=new_month,
        day=date.day,
        hour=date.hour,
        minute=date.minute,
        second=date.second)
    return date_next


def add_year(date, years_to_add):

    """
    Finds the next year from date.

    :param netcdftime.datetime date: Accepts datetime or phony datetime
        from ``netCDF4.num2date``.
    :param int years_to_add: The number of years to add to the date
    :returns: The final date
    :rtype: *netcdftime.datetime*
    """

    new_year = date.year + years_to_add

    date_next = datetime(
        year=new_year,
        month=date.month,
        day=date.day,
        hour=date.hour,
        minute=date.minute,
        second=date.second)
    return date_next


def num2date(num_axis, units, calendar):

    """
    A wrapper from ``nc.num2date`` able to handle "years since" and
        "months since" units.

    If time units are not "years since" or "months since", calls
    usual ``netcdftime.num2date``.

    :param numpy.array num_axis: The numerical time axis following units
    :param str units: The proper time units
    :param str calendar: The NetCDF calendar attribute
    :returns: The corresponding date axis
    :rtype: *array*
    """

    res = None
    if not units.split(' ')[0] in ['years', 'months']:
        res = nc.num2date(num_axis, units=units, calendar=calendar)

    else:
        units_as_days = 'days ' + ' '.join(units.split(' ')[1:])

        start_date = nc.num2date(0.0, units=units_as_days, calendar=calendar)

        num_axis_mod = np.atleast_1d(np.array(num_axis))

        if units.split(' ')[0] == 'years':
            max_years = np.floor(np.max(num_axis_mod)) + 1
            min_years = np.ceil(np.min(num_axis_mod)) - 1

            years_axis = np.array([
                add_year(start_date, years_to_add)
                for years_to_add in np.arange(min_years, max_years + 2)])

            cdftime = utime(units_as_days, calendar=calendar)
            years_axis_as_days = cdftime.date2num(years_axis)

            yind = np.vectorize(np.int)(np.floor(num_axis_mod))

            num_axis_mod_days = (
                years_axis_as_days[yind - int(min_years)] +
                (num_axis_mod - yind) *
                np.diff(years_axis_as_days)[yind - int(min_years)])

            res = nc.num2date(
                num_axis_mod_days, units=units_as_days, calendar=calendar)

        elif units.split(' ')[0] == 'months':
            max_months = np.floor(np.max(num_axis_mod)) + 1
            min_months = np.ceil(np.min(num_axis_mod)) - 1

            months_axis = np.array([
                add_month(start_date, months_to_add)
                for months_to_add in np.arange(min_months, max_months + 12)])

            cdftime = utime(units_as_days, calendar=calendar)
            months_axis_as_days = cdftime.date2num(months_axis)

            mind = np.vectorize(np.int)(np.floor(num_axis_mod))

            num_axis_mod_days = (
                months_axis_as_days[mind - int(min_months)] +
                (num_axis_mod - mind) *
                np.diff(months_axis_as_days)[mind - int(min_months)])

            res = nc.num2date(
                num_axis_mod_days, units=units_as_days, calendar=calendar)

        else:
            raise ValueError(units.split(' ')[0])

    assert res is not None
    return res


def check_full_nuggetness(in_model):

    in_model = str(in_model)

    nuggetness = False

    if in_model == 'nan':
        pass

    else:
        models = in_model.split('+')

        Sill = 0.0
        Range = 0.0
        model_names = []

        for submodel in models:
            submodel = submodel.strip()

            model = submodel.split(' ')[1].split('(')[0]
            model_names.append(model)

            Sill += float(submodel.split('(')[0].strip()[:-3].strip())
            Range = max(Range, float(submodel.split('(')[1].split(')')[0]))

        if np.any(np.isclose(np.array([Sill, Range]), 0.0)):
            nuggetness = True

        if (len(models) == 1) and models[0] == 'Nug':
            nuggetness = True

    return nuggetness


def get_dist_mat(x_arr, y_arr):

    assert x_arr.ndim == y_arr.ndim == 1, 'x_arr and y_arr can be 1D only!'

    assert x_arr.size == y_arr.size, (
        'x_arr and y_arr lengths should be equal!')

    assert np.all(np.isfinite(x_arr)) and np.all(np.isfinite(x_arr)), (
        'Invalid values in x_arr or y_arr!')

    dists_arr = np.full(
        (x_arr.shape[0], x_arr.shape[0]), np.nan, dtype=np.float64)

    fill_dists_2d_mat(
        x_arr.astype(np.float64, copy=False),
        y_arr.astype(np.float64, copy=False),
        x_arr.astype(np.float64, copy=False),
        y_arr.astype(np.float64, copy=False),
        dists_arr)

    assert np.all(np.isfinite(dists_arr)), 'Invalid values of distances!'
    return dists_arr
