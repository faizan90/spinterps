'''
Created on Nov 25, 2018

@author: Faizan
'''
import os
import sys
import traceback as tb
from queue import Queue
from math import pi, ceil
from functools import wraps
from pathos.multiprocessing import ProcessPool

import numpy as np
import psutil as ps
import netCDF4 as nc
import shapefile as shp
from osgeo import ogr, gdal
from cftime import utime, datetime

from .cyth import fill_dists_2d_mat, fill_theo_vg_vals

print_line_str = 40 * '#'


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


def linearize_sub_polys(poly, polys, simplify_tol):

    if poly is None:
        print('WARNING: A geometry is None!')

    else:
        assert isinstance(polys, Queue), 'polys not a queue.Queue object!'

        assert simplify_tol >= 0

        gct = poly.GetGeometryCount()
        gt = poly.GetGeometryType()

        assert gt in (2, 3, 6), 'Meant for polygons only!'

        if gt == 2:
            lin_ring = poly
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(lin_ring)

        if gct == 1:
            if simplify_tol:
                poly = poly.SimplifyPreserveTopology(simplify_tol)

            poly = poly.Buffer(0)

            n_pts = poly.GetGeometryRef(0).GetPointCount()

            if n_pts >= 3:
                polys.put_nowait(poly)

            else:
                print('WARNING: A polygon has less than 3 points!')

        elif gct > 1:
            for i in range(gct):
                linearize_sub_polys(
                    poly.GetGeometryRef(i), polys, simplify_tol)

        elif gct == 0:
            raise ValueError(
                'Encountered a geometry with a count of 0!')

    return


def linearize_sub_polys_with_labels(label, poly, labels_polys, simplify_tol):

    if poly is None:
        print(f'WARNING: A geometry is None for label {label}!')

    else:
        assert isinstance(labels_polys, Queue), (
            'labels_polys not a queue.Queue object!')

        assert simplify_tol >= 0

        gt = poly.GetGeometryType()
        gn = poly.GetGeometryName()

        assert gt  in (2, 3, 6), (
            f'Meant for polygons only and not {poly.GetGeometryName()} '
            f'(Type: {gt}), label {label}!')

        if gt == 2:
            lin_ring = poly
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(lin_ring)

            gt = poly.GetGeometryType()
            gn = poly.GetGeometryName()

        gct = poly.GetGeometryCount()

        if gct == 1:
            if simplify_tol:
                poly = poly.SimplifyPreserveTopology(simplify_tol)

            poly = poly.Buffer(0)

            assert poly.GetGeometryType() in (3, 6), (
                f'Geometry changed to {gn} '
                f'(Type: {gt}), label {label}!')

            n_pts = poly.GetGeometryRef(0).GetPointCount()

            if n_pts >= 3:
                labels_polys.put_nowait((label, poly))

            else:
                raise ValueError(
                    f'A polygon has less than 3 points for label '
                    f'{label}!')

        elif gct > 1:
            for i in range(gct):
                linearize_sub_polys_with_labels(
                    label, poly.GetGeometryRef(i), labels_polys, simplify_tol)

        elif gct == 0:
            raise ValueError(
                f'Encountered a geometry with a count of 0 '
                f'for label {label} with geometry name {gn} and type {gt}!')

    return


def get_all_polys_in_shp(path_to_shp, simplify_tol):

    assert isinstance(simplify_tol, (int, float))
    assert simplify_tol >= 0

    bds_vec = ogr.Open(str(path_to_shp))

    assert bds_vec is not None, (
        'Could not open the polygons_shapefile!')

    assert bds_vec.GetLayerCount() == 1, (
        'Only one layer allowed in the bounds shapefile!')

    bds_lyr = bds_vec.GetLayer(0)

    all_geoms = Queue()
    for feat in bds_lyr:
        geom = feat.GetGeometryRef().Clone()

        assert geom is not None, (
            'Something wrong with the geometries in the '
            'polygons_shapefile!')

        geom_type = geom.GetGeometryType()

        if geom_type in (3, 6):
            linearize_sub_polys(geom, all_geoms, simplify_tol)

        else:
            ValueError(f'Invalid geometry type: {geom_type}!')

    bds_vec.Destroy()
    return all_geoms


def chk_pt_cntmnt_in_poly(args):

    poly_or_pts, crds_df = args

    if not isinstance(poly_or_pts, ogr.Geometry):
        # Expecting that these are then x and y crds that we get
        # from GetPoints.
        pts = poly_or_pts

        n_poly_pts = len(pts)
        assert n_poly_pts >= 3, (
            f'Polygon not having enough points ({n_poly_pts})!')

        ring = ogr.Geometry(ogr.wkbLinearRing)
        for pt in pts:
            ring.AddPoint(*pt)

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

    else:
        poly = poly_or_pts

    poly_or_pts = None

    assert poly is not None, 'Corrupted polygon after buffering!'

    poly_area = poly.Area()
    assert poly_area > 0, f'Polygon has no area!'

    fin_stns = []
    for stn in crds_df.index:
        x, y = crds_df.loc[stn, ['X', 'Y']]

        if chk_cntmt(cnvt_to_pt(x, y), poly):
            fin_stns.append(stn)

    return fin_stns


def chk_pt_cntmnt_in_polys_mp(polys, crds_df, n_cpus):

    def get_sub_crds_dfs_for_polys(polys, crds_df, max_pts):

        stns = crds_df.index.values
        x = crds_df['X'].values
        y = crds_df['Y'].values

        sub_polys = []
        sub_crds_dfs = []
        for poly in polys:
            poly_xmin, poly_xmax, poly_ymin, poly_ymax = poly.GetEnvelope()

            assert poly.GetGeometryCount() == 1

            n_poly_pts = poly.GetGeometryRef(0).GetPointCount()

            assert n_poly_pts >= 3, (
                f'Polygon not having enough points ({n_poly_pts})!')

            cntmnt_idxs = (
                (x >= poly_xmin) &
                (x <= poly_xmax) &
                (y >= poly_ymin) &
                (y <= poly_ymax))

            if not cntmnt_idxs.sum():
                continue

            poly_stns = stns[cntmnt_idxs]

            n_poly_stns = len(poly_stns)

            # Break the number of coordinates into chunks so that later,
            # fewer coordinates have to be processed per thread. This allows
            # for distributing load more uniformly over the available threads.
            break_idxs = np.arange(
                0, (max_pts * ceil(n_poly_stns / max_pts)) + 1, max_pts)

            assert break_idxs[-1] >= max_pts

            for i in range(0, break_idxs.size - 1):

                break_poly_stns = poly_stns[break_idxs[i]:break_idxs[i + 1]]

                sub_polys.append(poly)
                sub_crds_dfs.append(crds_df.loc[break_poly_stns,:].copy())

        return sub_polys, sub_crds_dfs

    crds_df = crds_df.copy()

    max_pts_per_thread = int(max(1, crds_df.shape[0] // n_cpus))

    # Get sub crds_dfs for each poly such that the point are within the
    # extents of the respective polygon.
    sub_polys, sub_crds_dfs = get_sub_crds_dfs_for_polys(
        polys, crds_df, max_pts_per_thread)

    crds_df = None

    assert all([poly.GetGeometryCount() == 1 for poly in sub_polys])

    # Only do this if multiprocessing is used.
    # Because ogr.Geometry objects can't be pickled, apparently.
    if n_cpus > 1:
        sub_polys = [poly.GetGeometryRef(0).GetPoints() for poly in sub_polys]

    # This subsetting can be made on the relative number of points per poly.
    cntmnt_gen = (
        (sub_polys[i], sub_crds_dfs[i]) for i in range(len(sub_polys)))

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)

        ress = list(mp_pool.uimap(chk_pt_cntmnt_in_poly, cntmnt_gen))

        mp_pool.clear()
        mp_pool.close()
        mp_pool.join()
        mp_pool = None

    else:
        ress = []
        for args in cntmnt_gen:
            ress.append(chk_pt_cntmnt_in_poly(args))

    fin_stns = set()
    for res in ress:
        fin_stns |= set(res)

    fin_stns = list(fin_stns)
    return fin_stns


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


def print_sl():

    print(2 * '\n', print_line_str, sep='')
    return


def print_el():

    print(print_line_str)
    return


def get_n_cpus():

    phy_cores = ps.cpu_count(logical=False)
    log_cores = ps.cpu_count()

    if phy_cores < log_cores:
        n_cpus = phy_cores

    else:
        n_cpus = log_cores - 1

    n_cpus = max(n_cpus, 1)

    return n_cpus


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


def get_polygons_shp_extents(shp_path):

    '''
    Returns the maximum xy extents by going through all the polygons in a
    shapefile. This may yield different results than using the extents
    method in ogr.

    Output is tuple of (x_min, x_max, y_min, y_max)
    '''

    shp_hdl = shp.Reader(str(shp_path))

    assert any([
        shp_hdl.shapeTypeName == 'POLYGON',
        shp_hdl.shapeTypeName == 'POLYGONZ',
        shp_hdl.shapeTypeName == 'POLYGONM']), shp_hdl.shapeTypeName

    x_min_fin = +np.inf
    x_max_fin = -np.inf
    y_min_fin = +np.inf
    y_max_fin = -np.inf
    for shape in shp_hdl.shapes():
        pts = np.array(shape.points)

        x_crds = pts[:, 0]
        y_crds = pts[:, 1]

        x_min = x_crds.min()
        x_max = x_crds.max()

        if x_min < x_min_fin:
            x_min_fin = x_min

        if x_max > x_max_fin:
            x_max_fin = x_max

        y_min = y_crds.min()
        y_max = y_crds.max()

        if y_min < y_min_fin:
            y_min_fin = y_min

        if y_max > y_max_fin:
            y_max_fin = y_max

    return (x_min_fin, x_max_fin, y_min_fin, y_max_fin)


def get_aligned_shp_bds_and_cell_size(
        bounds_shp_file, align_ras_file, cell_bdist):

    # Error allowed in mismatch between extents of the shape file and raster
    # relative to the cell size of the raster. Keep this low, a high tolerance
    # here might translate to problems later. This is just to allow for
    # very small mismatchs in coordinates and cell sizes.
    rel_cell_err = 1e-5

    ras_props = get_ras_props(str(align_ras_file))
    ras_cell_size, _1 = ras_props[6:8]

    abs_cell_err = abs(ras_cell_size * rel_cell_err)

    cell_size_diff = abs(ras_cell_size - _1)
    assert (cell_size_diff <= abs_cell_err), (
        f'align_ras ({align_ras_file}) not square ({ras_cell_size}, {_1})!')

    ras_min_x, ras_max_x = ras_props[:2]
    ras_min_y, ras_max_y = ras_props[2:4]

    if bounds_shp_file != 'None':
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

        if cell_bdist:
            raw_shp_x_min -= cell_bdist
            raw_shp_x_max += cell_bdist
            raw_shp_y_min -= cell_bdist
            raw_shp_y_max += cell_bdist

    else:
        raw_shp_x_min = ras_min_x
        raw_shp_x_max = ras_max_x
        raw_shp_y_min = ras_min_y
        raw_shp_y_max = ras_max_y

    if raw_shp_x_min < ras_min_x:
        raw_shp_x_min_diff = abs(raw_shp_x_min - ras_min_x)
        assert (raw_shp_x_min_diff <= abs_cell_err), (
            f'bounds_shp x_min ({raw_shp_x_min}) < '
            f'align_raster x_min ({ras_min_x})!')

    if raw_shp_x_max > ras_max_x:
        raw_shp_x_max_diff = abs(raw_shp_x_max - ras_max_x)
        assert (raw_shp_x_max_diff <= abs_cell_err), (
            f'bounds_shp x_max ({raw_shp_x_max}) < '
            f'align_raster x_max ({ras_max_x})!')

    if raw_shp_y_min < ras_min_y:
        raw_shp_y_min_diff = abs(raw_shp_y_min - ras_min_y)
        assert (raw_shp_y_min_diff <= abs_cell_err), (
            f'bounds_shp y_min ({raw_shp_y_min}) < '
            f'align_raster y_min ({ras_min_y})!')

    if raw_shp_y_max > ras_max_y:
        raw_shp_y_max_diff = abs(raw_shp_y_max - ras_max_y)
        assert (raw_shp_y_max_diff <= abs_cell_err), (
            f'bounds_shp y_max ({raw_shp_y_max}) < '
            f'align_raster y_max ({ras_max_y})!')

    if not np.isclose(raw_shp_x_min, ras_min_x, rtol=0, atol=rel_cell_err):
        rem_min_col_width = ((raw_shp_x_min - ras_min_x) / ras_cell_size) % 1
        x_min_adj = rem_min_col_width * ras_cell_size
        adj_shp_x_min = raw_shp_x_min - x_min_adj

    else:
        adj_shp_x_min = ras_min_x

    if not np.isclose(raw_shp_y_max, ras_max_y, rtol=0, atol=rel_cell_err):
        rem_min_row_width = ((ras_max_y - raw_shp_y_max) / ras_cell_size) % 1
        y_max_adj = rem_min_row_width * ras_cell_size
        adj_shp_y_max = raw_shp_y_max + y_max_adj

    else:
        adj_shp_y_max = ras_max_y

    if not np.isclose(raw_shp_x_max, ras_max_x, rtol=0, atol=rel_cell_err):
        rem_max_col_width = ((raw_shp_x_max - ras_min_x) / ras_cell_size) % 1
        x_max_adj = rem_max_col_width * ras_cell_size
        adj_shp_x_max = raw_shp_x_max + (ras_cell_size - x_max_adj)

    else:
        adj_shp_x_max = ras_max_x

    if not np.isclose(raw_shp_y_min, ras_min_y, rtol=0, atol=rel_cell_err):
        rem_max_row_width = ((ras_max_y - raw_shp_y_min) / ras_cell_size) % 1
        y_min_adj = rem_max_row_width * ras_cell_size
        adj_shp_y_min = raw_shp_y_min - (ras_cell_size - y_min_adj)

    else:
        adj_shp_y_min = ras_min_y

    # Check remaining error after adjusting the rows and columns.
    allowed_err_rems = np.array([0.0, ras_cell_size])

    err_rem_cols = np.array(
        [(adj_shp_x_max - adj_shp_x_min) % ras_cell_size] *
        allowed_err_rems.size)

    assert np.isclose(err_rem_cols, allowed_err_rems, rtol=0, atol=rel_cell_err).any()

    err_rem_rows = np.array(
        [(adj_shp_y_max - adj_shp_y_min) % ras_cell_size] *
        allowed_err_rems.size)

    assert np.isclose(err_rem_rows, allowed_err_rems, rtol=0, atol=rel_cell_err).any()

    # Check adjusted bounds to be in within the alignment raster.
    assert (adj_shp_x_min >= ras_min_x), (
        f'Adjusted bounds_shp x_min ({adj_shp_x_min}) < '
        f'align_raster x_min ({ras_min_x})!')

    assert (adj_shp_x_max <= ras_max_x), (
        f'Adjusted bounds_shp x_max ({adj_shp_x_max}) < '
        f'align_raster x_max ({ras_max_x})!')

    assert (adj_shp_y_min >= ras_min_y), (
        f'Adjusted bounds_shp y_min ({adj_shp_y_min}) < '
        f'align_raster y_min ({ras_min_y})!')

    assert (adj_shp_y_max <= ras_max_y), (
        f'Adjusted bounds_shp y_max ({adj_shp_y_max}) < '
        f'align_raster y_max ({ras_max_y})!')

    return (
        (adj_shp_x_min, adj_shp_x_max, adj_shp_y_min, adj_shp_y_max),
        ras_cell_size)


def add_month(date, months_to_add):

    """
    Finds the next month from date.

    :param cftime.datetime date: Accepts datetime or phony datetime
        from ``netCDF4.num2date``.
    :param int months_to_add: The number of months to add to the date
    :returns: The final date
    :rtype: *cftime.datetime*
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

    :param cftime.datetime date: Accepts datetime or phony datetime
        from ``netCDF4.num2date``.
    :param int years_to_add: The number of years to add to the date
    :returns: The final date
    :rtype: *cftime.datetime*
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
    usual ``cftime.num2date``.

    :param numpy.array num_axis: The numerical time axis following units
    :param str units: The proper time units
    :param str calendar: The NetCDF calendar attribute
    :returns: The corresponding date axis
    :rtype: *array*
    """

    res = None
    if not units.split(' ')[0] in ['years', 'months']:
        res = nc.num2date(
            num_axis,
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=True)

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


def get_theo_vg_vals(in_model, h_arr):

    in_model = str(in_model)

    models = in_model.split('+')

    vg_vals = np.zeros_like(h_arr)

    for submodel in models:
        submodel = submodel.strip()

        sill, submodel = submodel.split(' ')
        submodel, rng = submodel.split('(')
        rng = rng.split(')')[0]

        sill = float(sill)
        rng = float(rng)

        fill_theo_vg_vals(submodel, h_arr, rng, sill, vg_vals)

    return vg_vals


def disagg_vg_str(in_vg_strs):

    assert in_vg_strs

    vg_strs = in_vg_strs.split('+')

    sills = []
    vgs = []
    rngs = []

    for vg_str in vg_strs:
        vg_str = vg_str.strip()

        sill, vg_rng = vg_str.split(' ')
        vg, rng = vg_rng.split('(')
        rng = rng.split(')')[0]

        sills.append(float(sill))
        vgs.append(vg)
        rngs.append(float(rng))

    return (sills, vgs, rngs)


def check_full_nuggetness(in_model, min_vg_val):

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

        if np.any(np.array([Sill, Range]) <= min_vg_val):
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


def nug_vg(h_arr, arg):

    # arg = (range, sill)
    nug_vg = np.full(h_arr.shape, arg[1])
    return nug_vg


def sph_vg(h_arr, arg):

    # arg = (range, sill)
    a = (1.5 * h_arr) / arg[0]
    b = h_arr ** 3 / (2 * arg[0] ** 3)
    sph_vg = (arg[1] * (a - b))
    sph_vg[h_arr > arg[0]] = arg[1]
    return sph_vg


def exp_vg(h_arr, arg):

    # arg = (range, sill)
    a = -3 * (h_arr / arg[0])
    exp_vg = (arg[1] * (1 - np.exp(a)))
    return exp_vg


def lin_vg(h_arr, arg):

    # arg = (range, sill)
    lin_vg = arg[1] * (h_arr / arg[0])
    lin_vg[h_arr > arg[0]] = arg[1]
    return lin_vg


def gau_vg(h_arr, arg):

    # arg = (range, sill)
    a = -3 * ((h_arr ** 2 / arg[0] ** 2))
    gau_vg = (arg[1] * (1 - np.exp(a)))
    return gau_vg


def pow_vg(h_arr, arg):

    # arg = (range, sill)
    pow_vg = (arg[1] * (h_arr ** arg[0]))
    return pow_vg


def hol_vg(h_arr, arg):

    # arg = (range, sill)
    hol_vg = np.zeros(h_arr.shape[0])  # do somethig about the zero
    idxs = np.where(h_arr > 0)
    a = (pi * h_arr[idxs]) / arg[0]
    hol_vg[idxs] = (arg[1] * (1 - (np.sin(a) / a)))
    return hol_vg


all_mix_vg_ftns = {
   'Nug': nug_vg,
   'Sph': sph_vg,
   'Exp': exp_vg,
   'Lin': lin_vg,
   'Gau': gau_vg,
   'Pow': pow_vg,
   'Hol': hol_vg
   }
