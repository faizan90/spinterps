'''
Created on May 27, 2019

@author: Faizan-Uni
'''
from math import ceil
from queue import Queue

import pyproj
import numpy as np
import pandas as pd
from osgeo import ogr, gdal
from pathos.multiprocessing import ProcessPool

from ..misc import (
    print_sl,
    print_el,
    gdal_err_hdlr,
    linearize_sub_polys_with_labels,
    get_n_cpus)

gdal.PushErrorHandler(gdal_err_hdlr)
ogr.UseExceptions()


class GeomAndCrdsItsctIdxs:

    '''Get row and column indices where geometries and raster intersect'''

    def __init__(self, verbose=True):

        self._vb = verbose

        self._geom_types = (1, 3, 6)

        self._geoms = None
        self._labels = None
        self._geom_type = None
        self._geom_sim_tol_ratio = None

        self._ras_type_labs = ('nc', 'gtiff')

        self._geom_sim_tol = 0.0
        self._x_crds_orig = None
        self._y_crds_orig = None
        self._x_crds = None
        self._y_crds = None
        self._ras_type_lab = None
        self._crds_ndims = None

        self._crds_src_epsg = None
        self._crds_dst_epsg = None
        self._crds_tfmr = None

        self._cell_size = None

        self._min_itsct_area_pct_thresh = None
        self._n_cpus = None

        self._itsct_idxs_dict = None

        self._set_itsct_idxs_geoms_flag = False
        self._set_itsct_idxs_crds_flag = False
        self._set_itsct_misc_flag = False
        self._set_itsct_idxs_vrfd_flag = False
        self._set_itsct_misc_flag = False
        self._itsct_idxs_cmptd_flag = False
        return

    def set_geometries(
            self, geometries, geometry_type, simplify_tol_ratio=0.0):

        '''Set the geometries for intersection.

        Parameters
        ----------
        geometries : dict
            A dictionary whose values are gdal/ogr point/polygon geometries.
        geometry_type : int
            An integer corresponding to the geometry types of ogr.
            Currently supported are points (geometry_type=1), polygons
            (geometry_type=3) and multipolygons (geometry_type=6).
        simplify_tol_ratio : float
            How much to simplify the geometries with respect to the mean
            cell_size. Should be greater than or equal to 0. If zero than
            no simplification is performed.
        '''

        if self._vb:
            print_sl()

            print(
                'Setting geometries as input for geometries\' and '
                'coordinates\' intersections...')

        assert isinstance(geometries, dict), (
            'geometries not a dictionary!')

        assert isinstance(geometry_type, int), (
            'geometry_type can only be an integer!')

        assert geometry_type in self._geom_types, (
            f'geometry_type can be one of {self._geom_types} only!')  #

        assert isinstance(simplify_tol_ratio, float), (
            'simplify_tol_ratio not a float!')

        assert simplify_tol_ratio >= 0, ('Invalid simplify_tol_ratio!')

        n_geoms = len(geometries)

        assert n_geoms, 'Empty geometries!'

        labels = tuple(geometries.keys())

        for label in labels:
            assert label in geometries, (
                f'Label: {label} not in geometries!')

            geom = geometries[label]

            assert isinstance(geom, ogr.Geometry), (
                f'Geometry: {label} not and ogr.Geometry object!')

            geom_type = geom.GetGeometryType()
            geom_name = geom.GetGeometryName()

            if geometry_type == 1:
                assert geom_type == 1, (
                    f'Unsupported geometry type, name: '
                    f'{geom_type}, {geom_name}!')

                assert geom.GetGeometryCount() == 0, (
                    'Only one point allowed per feature!')

            elif (geometry_type == 3) or (geometry_type == 6):
                assert (
                    (geom.GetGeometryType() == 3) or
                    (geom.GetGeometryType() == 6)), (
                        f'Geometry: {label} not a polygon!')

                assert geom.GetGeometryCount() >= 1, (
                    'Only one  or more polygon allowed per feature!')

                if geom_type == 3:
                    assert len(geom.GetGeometryRef(0).GetPoints()) >= 3, (
                        f'Polygon: {label} has less than 3 points!')

                elif geom_type == 6:
                    for sub_geom in geom:
                        assert len(
                            sub_geom.GetGeometryRef(0).GetPoints()) >= 3, (
                                f'A polygon has less than 3 points!')

                else:
                    raise NotImplementedError

                assert geom.Area() > 0, f'Polygon: {label} has no area!'

            else:
                raise NotImplementedError

        self._geoms = geometries
        self._labels = labels
        self._geom_type = geometry_type
        self._geom_sim_tol_ratio = simplify_tol_ratio

        if self._vb:
            print(
                f'INFO: Set {n_geoms} geometries for '
                f'intersection with coordinates')

            if self._geom_type == 1:
                print('Geometry type is POINT')

            elif self._geom_type == 3:
                print('Geometry type is POLYGON')

            else:
                raise NotImplementedError

            print(
                'Geometry simplification tolerance ratio is:',
                self._geom_sim_tol_ratio)

            print_el()

        self._set_itsct_idxs_geoms_flag = True
        return

    def set_coordinates(self, x_crds, y_crds, raster_type_label):

        '''Set X and Y coordinates for intersection.

        Parameters
        ----------
        x_crds : 1D / 2D numeric np.ndarray
            Array having the x-coordinates. These should all be monotonically
            ascending or descending. With all unique values in case of 1D.
        y_crds : 1D / 2D numeric np.ndarray
            Array having the y-coordinates. These should all be monotonically
            ascending or descending. With all unique values in case of 1D.
            Should have the same length as x_crds.
        raster_type_label : str
            The type of coordinates. If \'nc\' then coordinates are cell
            centroid coordinates. If \'gtiff\' then coordinates are cell
            corners.
        '''

        if self._vb:
            print_sl()

            print(
                'Setting coordinates as input for geometries\' and '
                'coordinates\' intersections...')

        self._verf_crds(x_crds)
        self._verf_crds(y_crds)

        assert x_crds.ndim == y_crds.ndim, (
            f'Unequal dimensions of X ({x_crds.ndim}) and '
            f'Y ({y_crds.ndim}) coordinates!')

        assert isinstance(raster_type_label, str), (
            f'raster_type_label not a string!')

        assert raster_type_label in self._ras_type_labs, (
            f'raster_type_label only allowed to be one of '
            f'{self._ras_type_labs}!')

        self._x_crds_orig = x_crds
        self._y_crds_orig = y_crds

        self._ras_type_lab = raster_type_label
        self._crds_ndims = x_crds.ndim

        self._x_crds_orig.flags.writeable = False
        self._y_crds_orig.flags.writeable = False

        if self._vb:
            print(f'INFO: Set the following raster coordinate properties:')
            print(f'Shape of X coordinates: {self._x_crds_orig.shape}')

            print(
                f'X coordinates\' min and max: '
                f'{self._x_crds_orig.min():0.3f}, '
                f'{self._x_crds_orig.max():0.3f}')

            print(f'Shape of Y coordinates: {self._y_crds_orig.shape}')

            print(
                f'Y coordinates\' min and max: '
                f'{self._y_crds_orig.min():0.3f}, '
                f'{self._y_crds_orig.max():0.3f}')

            print_el()

        self._set_itsct_idxs_crds_flag = True
        return

    def set_coordinate_system_transforms(self, src_epsg, dst_epsg):

        '''
        Transform coordinates from source to destination systems.

        This applies to x_crds and y_crds specified in set_coordinates.
        '''

        if self._vb:
            print_sl()

            print('Setting coordinate system transforms...')

        if (src_epsg is None) and (dst_epsg is None):
            pass

        else:
            assert isinstance(src_epsg, int), 'src_epsg must be an integer!'
            assert isinstance(dst_epsg, int), 'dst_epsg must be an integer!'

            assert src_epsg != dst_epsg, (
                'src_epsg and dst_epsg cannot be the same!')

            assert (src_epsg > 0) and (dst_epsg > 0), (
                'Both src_epsg and dst_epsg must be greater than zero!')

            try:
                pyproj.crs.CRS.from_epsg(src_epsg)

            except pyproj.exceptions.CRSError:
                raise ValueError(
                    f'Unrecognized src_epsg coordinate system: {src_epsg}!')
            try:
                pyproj.crs.CRS.from_epsg(dst_epsg)

            except pyproj.exceptions.CRSError:
                raise ValueError(
                    f'Unrecognized dst_epsg coordinate system: {dst_epsg}!')

            self._crds_src_epsg = src_epsg
            self._crds_dst_epsg = dst_epsg

            self._crds_tfmr = pyproj.Transformer.from_crs(
                f'EPSG:{src_epsg}', f'EPSG:{dst_epsg}', always_xy=True)

        if self._vb:
            print(f'Set the following coordinate system transforms:')
            print(f'Source coordinate system EPSG: {src_epsg}')
            print(f'Destination coordinate system EPSG: {dst_epsg}')

            print_el()
        return

    def set_intersect_misc_settings(
            self,
            minimum_cell_area_intersection_percentage,
            n_cpus):

        assert isinstance(
            minimum_cell_area_intersection_percentage, (int, float)), (
                'minimum_cell_area_intersection_percentage not a float '
                'or an integer!')

        assert 0 <= minimum_cell_area_intersection_percentage <= 100, (
            'minimum_cell_area_intersection_percentage can only be between '
            '0 and 100!')

        if isinstance(n_cpus, str):
            assert n_cpus == 'auto', 'Invalid n_cpus!'

            n_cpus = get_n_cpus()

        else:
            assert isinstance(n_cpus, int), 'n_cpus is not an integer!'

            assert n_cpus > 0, 'Invalid n_cpus!'

        self._min_itsct_area_pct_thresh = (
            minimum_cell_area_intersection_percentage)

        self._n_cpus = n_cpus

        if self._vb:
            print_sl()

            print(f'INFO: Set the following misc. parameters:')

            print(
                f'Minimum cell area intersection percentage: '
                f'{self._min_itsct_area_pct_thresh}')

            print(
                f'Number of maximum process(es) to use: '
                f'{self._n_cpus}')

            print_el()

        self._set_itsct_misc_flag = True
        return

    def verify(self):

        if self._vb:
            print_sl()

            print(
                f'Verifying all inputs for geometries\' and coordinates\' '
                f'intersection...')

        assert self._set_itsct_idxs_geoms_flag, (
            'Call the set_geometries method first!')

        assert self._set_itsct_idxs_crds_flag, (
            'Call the set_coordinates method first!')

        if self._geom_type == 1:
            pass

        elif self._geom_type in (3, 6):
            assert self._set_itsct_misc_flag, (
                'Call set_intersect_misc_settings first!')

        else:
            raise NotImplementedError

        x_min = x_max = y_min = y_max = None

        if ((self._ras_type_lab == 'nc') and
            (self._crds_ndims == 1) and
            (self._geom_type == 1)):

            x_crds, y_crds = (
                self._x_crds_orig.copy(), self._y_crds_orig.copy())

        elif ((self._ras_type_lab == 'gtiff') and
              (self._crds_ndims == 1) and
              (self._geom_type == 1)):

            x_crds = self._get_rect_crds_1d(self._x_crds_orig)
            y_crds = self._get_rect_crds_1d(self._y_crds_orig)

            # Here, limits have to be set separately because actual limits
            # are lost otherwise.
            x_min = self._x_crds_orig.min()
            x_max = self._x_crds_orig.max()

            y_min = self._y_crds_orig.min()
            y_max = self._y_crds_orig.max()

        elif ((self._ras_type_lab == 'nc') and
              (self._crds_ndims == 2) and
              (self._geom_type == 1)):

            x_crds, y_crds = (
                self._x_crds_orig.copy(), self._y_crds_orig.copy())

        elif ((self._ras_type_lab == 'nc') and
              (self._crds_ndims == 1) and
              (self._geom_type == 3)):

            x_crds = self._get_rect_crds_1d(self._x_crds_orig)
            y_crds = self._get_rect_crds_1d(self._y_crds_orig)

        elif ((self._ras_type_lab == 'gtiff') and
              (self._crds_ndims == 1) and
              (self._geom_type == 3)):

            x_crds, y_crds = (
                self._x_crds_orig.copy(), self._y_crds_orig.copy())

        elif ((self._ras_type_lab == 'nc') and
              (self._crds_ndims == 2) and
              (self._geom_type == 3)):

            x_crds, y_crds = self._get_rect_crds_2d(
                self._x_crds_orig, self._y_crds_orig)

        else:
            raise NotImplementedError(
                f'Not configured for raster_type_label: '
                f'{self._ras_type_lab}, {self._crds_ndims} '
                f'dimensions and geometry type: {self._geom_type}!')

        if self._crds_tfmr is not None:
            if self._crds_ndims == 1:
                print(
                    'INFO: Converting coordinates to 2D due to '
                    'transformation!')

                x_crds, y_crds = np.meshgrid(x_crds, y_crds)

                self._crds_ndims = 2

            x_crds, y_crds = self._reproject(x_crds, y_crds)

        self._x_crds = x_crds
        self._y_crds = y_crds

        self._x_crds.flags.writeable = False
        self._y_crds.flags.writeable = False

        if x_min is None:
            x_min = self._x_crds.min()
            x_max = self._x_crds.max()

            y_min = self._y_crds.min()
            y_max = self._y_crds.max()

        if self._geom_type == 3:
            self._cell_size = self._get_cell_size()

            if self._geom_sim_tol_ratio:
                self._geom_sim_tol = (
                    self._cell_size * self._geom_sim_tol_ratio)

                self._geoms = self._get_simplified_geoms()

                if self._vb:
                    print(
                        f'INFO: Simplified geometries with a tolerance of '
                        f'{self._geom_sim_tol} units.')

        if self._geom_type == 1:
            for label in self._labels:
                geom = self._geoms[label]

                pt_x = geom.GetX()
                pt_y = geom.GetY()

                assert np.isfinite(pt_x), (
                    f'Label: {label} has invalid X coordinate!')

                assert np.isfinite(pt_y), (
                    f'Label: {label} has invalid Y coordinate!')

                assert x_min <= pt_x <= x_max, (
                    f'X coordinate of the point: {label} is not '
                    f'within the bounds of the X coordinates '
                    f'({x_min}, {pt_x}, {x_max})!')

                assert y_min <= pt_y <= y_max, (
                    f'Y coordinate of the point: {label} is not '
                    f'within the bounds of the Y coordinates '
                    f'({y_min}, {pt_y}, {y_max})!')

        elif self._geom_type == 3:
            for label in self._labels:
                geom = self._geoms[label]

                gx_min, gx_max, gy_min, gy_max = geom.GetEnvelope()

                assert gx_min >= x_min, (
                    f'Minimum X coordinate of the polygon: {label} is less '
                    f'than the minimum X coordinate ({gx_min}, {x_min})!')

                assert gx_max <= x_max, (
                    f'Maximum X coordinate of the polygon: {label} is greater '
                    f'than the maximum X coordinate ({gx_max}, {x_max})!')

                assert gy_min >= y_min, (
                    f'Minimum Y coordinate of the polygon: {label} is less '
                    f'than the minimum Y coordinate ({gy_min}, {y_min})!')

                assert gy_max <= y_max, (
                    f'Maximum Y coordinate of the polygon: {label} is greater '
                    f'than the maximum Y coordinate ({gy_max}, {y_max})!')

            if not (self._crds_ndims == 2):
                self._x_crds, self._y_crds = np.meshgrid(
                    self._x_crds, self._y_crds)

                self._crds_ndims = 2

                print(
                    'INFO: Converting coordinates to 2D due to '
                    'transformation!')

        else:
            raise NotImplementedError

        if self._vb:
            print(
                f'INFO: All inputs for geometries\' and coordinates\' '
                f'intersection verified to be correct.')

            print_el()

        self._set_itsct_idxs_vrfd_flag = True
        return

    def compute_intersect_indices(self):

        if self._vb:
            print_sl()

            print(
                'Computing the intersection indices between geometries '
                'and coordinates...')

        assert self._set_itsct_idxs_vrfd_flag, 'Call the verify method first!'

        self._itsct_idxs_cmptd_flag = False

        assert self._crds_ndims <= 2, (
            'Intersection configured for 2 or less dimensions of '
            'coordinates!')

        itsct_idxs_dict = None

        if self._geom_type == 1:
            if self._crds_ndims == 1:
                itsct_idxs_dict = self._cmpt_1d_pt_idxs()

            elif self._crds_ndims == 2:
                itsct_idxs_dict = self._cmpt_2d_pt_idxs()

            else:
                raise NotImplementedError(
                    'Not implemented for more than two dimensions!')

        elif self._geom_type == 3:
            assert self._crds_ndims == 2, (
                'Coordinates should have been 2D by now!')

            itsct_idxs_dict = self._cmpt_polys_itsct_idxs()

        else:
            raise NotImplementedError

        assert itsct_idxs_dict

        self._itsct_idxs_dict = itsct_idxs_dict

        if self._vb:
            print(
                'Done computing the intersection indices between geometries '
                'and coordinates')

            print_el()

        self._itsct_idxs_cmptd_flag = True
        return

    def get_intersect_indices(self):

        assert self._itsct_idxs_cmptd_flag, (
            'Call the compute_intersect_idxs method first!')

        return self._itsct_idxs_dict

    def _get_cell_size(self):

        if self._crds_ndims == 1:
            cell_size = max(
                np.abs(self._x_crds[1:] - self._x_crds[:-1]).max(),
                np.abs(self._y_crds[1:] - self._y_crds[:-1]).max())

        elif self._crds_ndims == 2:
            cell_size = max(
                np.abs(self._x_crds[1:,:] - self._x_crds[:-1,:]).max(),
                np.abs(self._x_crds[:, 1:] - self._x_crds[:,:-1]).max(),
                np.abs(self._y_crds[:, 1:] - self._y_crds[:,:-1]).max(),
                np.abs(self._y_crds[1:,:] - self._y_crds[:-1,:]).max())

        else:
            raise NotImplementedError(
                f'Mean cell size nor implmented for given number of '
                f'dimensions ({self._crds_ndims})!')

        if self._vb:
            print('INFO: Cell size is:', cell_size)

        return cell_size

    def _get_simplified_geoms(self):

        sim_geoms = {
            label: geom.SimplifyPreserveTopology(self._geom_sim_tol).Buffer(0)
            for label, geom in self._geoms.items()}

        return sim_geoms

    def _reproject(self, x_crds, y_crds):

        if self._vb:
            print(
                f'Reprojecting from EPSG {self._crds_src_epsg} to '
                f'{self._crds_dst_epsg}...')

        x_crds_reproj, y_crds_reproj = self._crds_tfmr.transform(
            x_crds, y_crds)

        assert x_crds.shape == x_crds_reproj.shape
        assert y_crds.shape == y_crds_reproj.shape

        assert np.all(np.isfinite(x_crds_reproj))
        assert np.all(np.isfinite(y_crds_reproj))

        if self._vb:
            print('Done reprojecting.')

        return x_crds_reproj, y_crds_reproj

    def _verf_crds(self, crds):

        assert isinstance(crds, np.ndarray), f'crds not of np.ndarray type!'

        assert np.issubdtype(crds.dtype, np.number), (
            'Only numeric coordinates are allowed!')

        assert np.all(np.isfinite(crds)), 'crds has invalid values inside!'

        assert np.all(crds.shape), 'Shape of crds not allowed to have a zero!'

        if crds.ndim == 1:
            self._verf_crds_1d(crds)

        elif crds.ndim == 2:
            self._verf_crds_2d(crds)

        else:
            raise NotImplementedError
        return

    def _verf_crds_1d(self, crds):

        assert np.unique(crds).shape == crds.shape, 'Non-unique crds!'

        assert (
            np.all(np.ediff1d(crds) > 0) or
            np.all(np.ediff1d(crds[::-1]) > 0)), (
                'crds not monotonically increasing or decreasing!')
        return

    def _verf_crds_2d(self, crds):

        diffs_lr = np.diff(crds)
        diffs_ud = np.diff(crds.T)

        assert (
            np.all(diffs_lr > 0) or
            np.all(np.fliplr(diffs_lr) > 0) or
            np.all(diffs_ud > 0) or
            np.all(np.flipud(diffs_ud) > 0) or
            np.all(np.flipud(diffs_ud) < 0)), (
                'crds not monotonically increasing or decreasing '
                'in any direction!')
        return

    def _get_rect_crds_1d(self, crds):

        assert crds.ndim == 1, 'Configured for 1D coordinates only!'

        if self._ras_type_lab == 'nc':
            crds_rect = np.full(crds.shape[0] + 1, np.nan)

            crds_rect[1:-1] = (crds[:-1] + crds[1:]) * 0.5

            crds_rect[+0] = crds[+0] - (0.5 * (crds[+1] - crds[+0]))
            crds_rect[-1] = crds[-1] + (0.5 * (crds[-1] - crds[-2]))

        elif self._ras_type_lab == 'gtiff':
            crds_rect = np.full(crds.shape[0] - 1, np.nan)

            crds_rect[:] = (crds[:-1] + crds[1:]) * 0.5

        else:
            raise NotImplementedError

        assert np.all(np.isfinite(crds_rect)), 'Invalid values in crds_rect!'

        assert (
            np.all(np.ediff1d(crds_rect) > 0) or
            np.all(np.ediff1d(crds_rect[::-1]) > 0)), (
                'crds_rect not monotonically increasing or decreasing!')

        return crds_rect

    def _get_rect_crds_2d(self, x_crds, y_crds):

        assert x_crds.ndim == y_crds.ndim, (
            'Unequal dimensions of X and Y coordinates!')

        assert x_crds.shape == y_crds.shape, (
            'Unequal shape of X and Y coordinates!')

        if self._ras_type_lab == 'nc':
            crds_rect_shape = (y_crds.shape[0] + 1, y_crds.shape[1] + 1)

            y_crds_rect = np.full(crds_rect_shape, np.nan)
            x_crds_rect = np.full(crds_rect_shape, np.nan)

            for i in range(1, crds_rect_shape[0] - 1):
                for j in range(1, crds_rect_shape[1] - 1):
                    y_crds_rect[i, j] = y_crds[i, j] - (
                        0.5 * (y_crds[i, j] - y_crds[i - 1, j - 1]))

                    x_crds_rect[i, j] = x_crds[i, j] - (
                        0.5 * (x_crds[i, j] - x_crds[i - 1, j - 1]))

            y_crds_rect[+0, +0] = y_crds[+0, +0] + (
                0.5 * (y_crds[+0, +0] - y_crds[+1, +0]))

            y_crds_rect[-1, -1] = y_crds[-1, -1] + (
                0.5 * (y_crds[-1, -1] - y_crds[-2, -1]))

            y_crds_rect[-1, +0] = y_crds[-1, +0] + (
                0.5 * (y_crds[-1, +0] - y_crds[-2, +0]))

            y_crds_rect[+0, -1] = y_crds[+0, -1] + (
                0.5 * (y_crds[+0, -1] - y_crds[+1, -1]))

            y_crds_rect[+0, +1:-1] = y_crds[+0, +1:] - (
                0.5 * (y_crds[+1, +1:] - y_crds[+0,:-1]))

            y_crds_rect[-1, +1:-1] = y_crds[-1, +1:] + (
                0.5 * (y_crds[-1,:-1] - y_crds[-2, +1:]))

            y_crds_rect[+1:-1, +0] = y_crds[+1:, +0] - (
                0.5 * (y_crds[+1:, +0] - y_crds[:-1, +0]))

            y_crds_rect[+1:-1, -1] = y_crds[+1:, -1] - (
                0.5 * (y_crds[+1:, -1] - y_crds[:-1, -1]))

            x_crds_rect[+0, +0] = x_crds[+0, +0] + (
                0.5 * (x_crds[+0, +0] - x_crds[+0, +1]))

            x_crds_rect[-1, -1] = x_crds[-1, -1] + (
                0.5 * (x_crds[-1, -1] - x_crds[-1, -2]))

            x_crds_rect[-1, +0] = x_crds[-1, +0] + (
                0.5 * (x_crds[-1, +0] - x_crds[-1, +1]))

            x_crds_rect[+0, -1] = x_crds[+0, -1] + (
                0.5 * (x_crds[+0, -1] - x_crds[+0, -2]))

            x_crds_rect[+0, +1:-1] = x_crds[+0, +1:] - (
                0.5 * (x_crds[+0, +1:] - x_crds[+0,:-1]))

            x_crds_rect[-1, +1:-1] = x_crds[-1, +1:] - (
                0.5 * (x_crds[-1, +1:] - x_crds[-1,:-1]))

            x_crds_rect[+1:-1, +0] = x_crds[+1:, +0] + (
                0.5 * (x_crds[:-1, +0] - x_crds[+1:, +1]))

            x_crds_rect[+1:-1, -1] = x_crds[+1:, -1] + (
                0.5 * (x_crds[:-1, -1] - x_crds[+1:, -2]))

        else:
            raise NotImplementedError

        assert np.all(np.isfinite(x_crds_rect)), (
            'Invalid values in x_crds_rect!')

        assert np.all(np.isfinite(y_crds_rect)), (
            'Invalid values in y_crds_rect!')

        return x_crds_rect, y_crds_rect

    def _cmpt_1d_pt_idxs(self):

        assert self._crds_ndims == 1, 'Configured for 1D coordinates only!'
        assert self._geom_type == 1, 'Configured for point geometries only!'

        x_crds = self._x_crds
        y_crds = self._y_crds

        n_pts = len(self._labels)

        pt_crds = np.full((n_pts, 2), np.nan)

        gx_min, gx_max, gy_min, gy_max = np.inf, -np.inf, np.inf, -np.inf
        for i, label in enumerate(self._labels):
            pt_x = self._geoms[label].GetX()
            pt_y = self._geoms[label].GetY()

            if pt_x < gx_min:
                gx_min = pt_x

            if pt_x > gx_max:
                gx_max = pt_x

            if pt_y < gy_min:
                gy_min = pt_y

            if pt_y > gy_max:
                gy_max = pt_y

            pt_crds[i] = pt_x, pt_y

        assert np.all(np.isfinite([gx_min, gx_max, gy_min, gy_max])), (
            'Invalid spatial bounds of points!')

        assert np.all(np.isfinite(pt_crds)), 'Invalid points in pt_crds!'

        geom_buff = max(
            abs(x_crds[+1] - x_crds[+0]),
            abs(x_crds[-1] - x_crds[-2]),
            abs(y_crds[+1] - y_crds[+0]),
            abs(y_crds[-1] - y_crds[-2]),
            )

        max_cell_dist = 2 * ((((0.5 * geom_buff) ** 2) * 2) ** 0.5)

        gx_min, gx_max, gy_min, gy_max = (
            gx_min - geom_buff,
            gx_max + geom_buff,
            gy_min - geom_buff,
            gy_max + geom_buff)

        x_idxs = np.where(
            (self._x_crds >= gx_min) & (self._x_crds <= gx_max))[0]

        y_idxs = np.where(
            (self._y_crds >= gy_min) & (self._y_crds <= gy_max))[0]

        n_x_idxs = x_idxs.size
        n_y_idxs = y_idxs.size

        assert n_x_idxs, (
            f'No X coordinate selected for the point: {label}!')

        assert n_y_idxs, (
            f'No Y coordinate selected for the point: {label}!')
        #======================================================================

        # The 2D distance version.
        # This requires too much memory and operations.
#         x_crds = np.tile(self._x_crds[x_idxs], n_y_idxs)
#         y_crds = np.repeat(self._y_crds[y_idxs], n_x_idxs)
#
#         assert x_crds.size == y_crds.size, (
#             'x_crds and y_crds not having same length!')
        #======================================================================

        # The 1D distance verions. Way more efficient than the 2D version.
        # Hope that this is correct.
        x_crds = self._x_crds[x_idxs]
        y_crds = self._y_crds[y_idxs]
        #======================================================================

        show_crds_flag = False

        if show_crds_flag:
            print(
                '   i, Label    |      RX      |      RY      |'
                '      DX      |'
                '      DY      |   Distance')

        itsct_idxs_dict = {}
        for i, label in enumerate(self._labels):
            pt_x, pt_y = pt_crds[i]

            x_sq_diff = (pt_x - x_crds) ** 2
            y_sq_diff = (pt_y - y_crds) ** 2
            #==================================================================

            # The 2D distance version.
#             dists = (x_sq_diff + y_sq_diff) ** 0.5
#
#             min_dist_idx = np.argmin(dists)
#
#             min_x_crd_idx = x_idxs[(min_dist_idx % n_x_idxs)]
#             min_y_crd_idx = y_idxs[(min_dist_idx // n_x_idxs)]
#
#             min_dist = (
#                 ((pt_x - self._x_crds[min_x_crd_idx]) ** 2) +
#                 ((pt_y - self._y_crds[min_y_crd_idx]) ** 2)) ** 0.5
            #==================================================================

            # The 1D distance version.
            x_idx_new = np.argmin(x_sq_diff)
            y_idx_new = np.argmin(y_sq_diff)

            min_x_crd_idx = x_idxs[x_idx_new]
            min_y_crd_idx = y_idxs[y_idx_new]

            min_dist = (
                ((pt_x - self._x_crds[min_x_crd_idx]) ** 2) +
                ((pt_y - self._y_crds[min_y_crd_idx]) ** 2)) ** 0.5

            dist_new = (
                ((pt_x - x_crds[x_idx_new]) ** 2) +
                ((pt_y - y_crds[y_idx_new]) ** 2)) ** 0.5

            assert np.isclose(dist_new, min_dist), (
                f'{i}, {label}: {dist_new}, {min_dist}!')
            #==================================================================

            if show_crds_flag:
                print(
                    f'{i:4d}, {label:<9s}|{pt_x:^14.5f}|{pt_y:^14.5f}|'
                    f'{self._x_crds[min_x_crd_idx]:^14.5f}|'
                    f'{self._y_crds[min_y_crd_idx]:^14.5f}|   '
                    f'{min_dist}')

            assert min_dist <= max_cell_dist, (
                f'Label: {label} have a distance greater than the limit'
                f'to the centroid of the nearest cell!')

            itsct_idxs_dict[label] = {
                'cols':np.array([min_x_crd_idx], dtype=int),
                'rows': np.array([min_y_crd_idx], dtype=int),
                'itsctd_area': np.array([0.0], dtype=float),
                'rel_itsctd_area': np.array([0.0], dtype=float),
                'x_cen_crds': np.array(
                    [self._x_crds[min_x_crd_idx]], dtype=float),
                'y_cen_crds': np.array(
                    [self._y_crds[min_y_crd_idx]], dtype=float),
                'x_crds': np.array([pt_x], dtype=float),
                'y_crds': np.array([pt_y], dtype=float)}

        return itsct_idxs_dict

    def _cmpt_2d_pt_idxs(self):

        print_sl()

        print(
            'WARNING: Call to an untested method '
            '"GeomAndCrdsItsctIdxs._cmpt_2d_pt_idxs"!')

        print_el()

        assert self._crds_ndims == 2, 'Configured for 2D coordinates only!'
        assert self._geom_type == 1, 'Configured for point geometries only!'

        x_crds = self._x_crds
        y_crds = self._y_crds

        n_pts = len(self._labels)

        pt_crds = np.full((n_pts, 2), np.nan)

        gx_min, gx_max, gy_min, gy_max = np.inf, -np.inf, np.inf, -np.inf
        for i, label in enumerate(self._labels):
            pt_x = self._geoms[label].GetX()
            pt_y = self._geoms[label].GetY()

            if pt_x < gx_min:
                gx_min = pt_x

            if pt_x > gx_max:
                gx_max = pt_x

            if pt_y < gy_min:
                gy_min = pt_y

            if pt_y > gy_max:
                gy_max = pt_y

            pt_crds[i] = pt_x, pt_y

        assert np.all(np.isfinite([gx_min, gx_max, gy_min, gy_max])), (
            'Invalid spatial bounds of points!')

        assert np.all(np.isfinite(pt_crds)), 'Invalid points in pt_crds!'

        geom_buff = max(
            abs(x_crds[+1, +0] - x_crds[+0, +0]),
            abs(x_crds[-1, -1] - x_crds[-2, -1]),
            abs(y_crds[+1, +0] - y_crds[+0, 0]),
            abs(y_crds[-1, -1] - y_crds[-2, -1]),
            )

        max_cell_dist = 2 * ((((0.5 * geom_buff) ** 2) * 2) ** 0.5)

        gx_min, gx_max, gy_min, gy_max = (
            gx_min - geom_buff,
            gx_max + geom_buff,
            gy_min - geom_buff,
            gy_max + geom_buff)

        tot_idxs = np.vstack(np.where(
            (x_crds >= gx_min) &
            (x_crds <= gx_max) &
            (y_crds >= gy_min) &
            (y_crds <= gy_max))).T

        keep_idxs = ~(
            (tot_idxs[:, 0] >= (x_crds.shape[0] - 1)) |
            (tot_idxs[:, 1] >= (x_crds.shape[1] - 1)))

        tot_idxs = tot_idxs[keep_idxs].copy('c')

        n_tot_idxs = tot_idxs.size

        assert n_tot_idxs, (
            f'No XY coordinate selected for the point: {label}!')

        x_crds = x_crds[tot_idxs[:, 0], tot_idxs[:, 1]]
        y_crds = y_crds[tot_idxs[:, 0], tot_idxs[:, 1]]

        assert x_crds.size == y_crds.size, (
            'x_crds and y_crds not having same length!')

        show_crds_flag = False

        if show_crds_flag:
            print(
                '   i, Label   |      RX      |      RY      |      DX      |'
                '      DY      |   Distance')

        itsct_idxs_dict = {}
        for i, label in enumerate(self._labels):
            pt_x, pt_y = pt_crds[i]

            x_sq_diff = (pt_x - x_crds) ** 2
            y_sq_diff = (pt_y - y_crds) ** 2

            dists = (x_sq_diff + y_sq_diff) ** 0.5

            min_dist_idx = np.argmin(dists)

            min_row_crd_idx, min_col_crd_idx = tot_idxs[min_dist_idx]

            min_dist = 0.0

            min_dist += (
                pt_x - self._x_crds[min_row_crd_idx, min_col_crd_idx]) ** 2

            min_dist += (
                pt_y - self._y_crds[min_row_crd_idx, min_col_crd_idx]) ** 2

            min_dist **= 0.5

            if show_crds_flag:
                print(
                    f'{i:4d}, {label:<9s}|{pt_x:^14.5f}|{pt_y:^14.5f}|'
                    f'{self._x_crds[min_row_crd_idx, min_col_crd_idx]:^14.5f}|'
                    f'{self._y_crds[min_row_crd_idx, min_col_crd_idx]:^14.5f}|'
                    f'   {min_dist}')

            assert min_dist <= max_cell_dist, (
                f'Label: {label} have a distance greater than the limit'
                f'to the centroid of the nearest cell!')

            itsct_idxs_dict[label] = {
                'cols':np.array([min_col_crd_idx], dtype=int),
                'rows': np.array([min_row_crd_idx], dtype=int),
                'itsctd_area': np.array([0.0], dtype=float),
                'rel_itsctd_area': np.array([0.0], dtype=float),
                'x_cen_crds': np.array(
                    [self._x_crds[min_row_crd_idx, min_col_crd_idx]],
                    dtype=float),
                'y_cen_crds': np.array(
                    [self._y_crds[min_row_crd_idx, min_col_crd_idx]],
                    dtype=float),
                'x_crds': np.array([pt_x], dtype=float),
                'y_crds': np.array([pt_y], dtype=float), }

        return itsct_idxs_dict

    def _cmpt_polys_itsct_idxs(self):

        '''
        Summary:
        1. Each multipolygon is split into polygons.
        2. Incase of a multithreaded scenario, polygons that have too many
           points for in them are broken into chunks to distributed the load
           across threads uniformly.
        3. X and Y corner coordinates of cells are tested for containment.
        4. Each multipolygon that was split for a given label, their data
           is combined to form only one set of indices per label. It may
           happen that some indices are repeated in the case of a cell
           intersecting with multiple polygons. This is also taken care of.
        '''

        org_polys_area = sum(
            [self._geoms[label].GetArea() for label in self._labels])

        org_poly_types = [
            self._geoms[label].GetGeometryType() for label in self._labels]

        if all([gt == 3 for gt in org_poly_types]):
            print('INFO: All geometries are POLYGONs!')

            lin_labels = self._labels
            lin_polys = [self._geoms[label] for label in self._labels]
            lin_polys_area = org_polys_area

        else:
            print('INFO: Some geometries are MULTIPOLYGONs!')

            labels_polys = Queue()

            for label in self._labels:
                linearize_sub_polys_with_labels(
                    label, self._geoms[label], labels_polys, 0)

            lin_labels = []
            lin_polys = []

            [(lin_labels.append(label), lin_polys.append(poly))
             for label, poly in labels_polys.queue]

            assert all([label in lin_labels
                        for label in self._labels]), (
                'Missing labels after linearizing polygons!')

            lin_polys_area = sum([poly.GetArea() for poly in lin_polys])
        #======================================================================

        max_pts_per_thread = int(max(1, self._x_crds.size // self._n_cpus))

        (chnkd_polys,
         chnkd_labels,
         chnkd_idxs,
         chnkd_x_crds,
         chnkd_y_crds) = self._get_chnkd_crds_for_polys(
             lin_polys,
             lin_labels,
             max_pts_per_thread)

        assert all([poly.GetGeometryCount() == 1 for poly in chnkd_polys])

        assert all([label in chnkd_labels
                    for label in self._labels]), (
            'Missing labels after chunking polygons!')
        #======================================================================

        n_cpus = min(self._n_cpus, len(chnkd_polys))

        assert n_cpus > 0, n_cpus

        # Only do this if multiprocessing is used.
        # Because ogr.Geometry objects can't be pickled, apparently.
        if n_cpus > 1:
            chnkd_polys = [
                poly.GetGeometryRef(0).GetPoints() for poly in chnkd_polys]

        idxs_gen = ((
            chnkd_polys[i],
            chnkd_labels[i],
            chnkd_idxs[i],
            chnkd_x_crds[i],
            chnkd_y_crds[i],
            False,
            self._min_itsct_area_pct_thresh)
            for i in range(len(chnkd_polys)))

        if n_cpus > 1:
            mp_pool = ProcessPool(n_cpus)
            mp_pool.restart(True)

            ress = list(mp_pool.uimap(
                GeomAndCrdsItsctIdxs._cmpt_poly_itsct_idxs, idxs_gen))

            mp_pool.clear()
            mp_pool.close()
            mp_pool.join()
            mp_pool = None

        else:
            ress = []
            for args in idxs_gen:
                ress.append(GeomAndCrdsItsctIdxs._cmpt_poly_itsct_idxs(args))
        #======================================================================

        itsct_idxs_dict, cells_area_sum = self._assemble_itsct_idxs_dict(
            ress)

        print(
            f'INFO: Total area of original polygons: '
            f'{org_polys_area} units.')

        print(
            f'INFO: Total area of linearized polygons: '
            f'{lin_polys_area} units.')

        print(
            f'INFO: Total area of cells intersecting with the polygons: '
            f'{cells_area_sum} units.')

        pdiff = 100 * (cells_area_sum - lin_polys_area) / lin_polys_area

        print(
            f'INFO: Mismatch in total area: '
            f'{cells_area_sum - lin_polys_area} units ({pdiff:0.3f}%).')

        return itsct_idxs_dict

    def _get_chnkd_crds_for_polys(self, polys, labels, max_pts):

        '''
        Break the number of cells to test for containment in a polygon into
        smaller chunks in order to balance the load per thread.
        '''

        sub_polys = []
        sub_labels = []
        sub_idxs = []
        sub_x_crds = []
        sub_y_crds = []
        for poly, label in zip(polys, labels):
            extents = poly.GetEnvelope()
            poly_xmin, poly_xmax, poly_ymin, poly_ymax = extents

            # NOTE: After buffering, the extents can go out of that of the raster.
            poly_xmin -= self._cell_size
            poly_xmax += self._cell_size
            poly_ymin -= self._cell_size
            poly_ymax += self._cell_size

            assert poly.GetGeometryCount() == 1

            n_poly_pts = poly.GetGeometryRef(0).GetPointCount()

            assert n_poly_pts >= 3, (
                f'Polygon not having enough points ({n_poly_pts})!',
                label, poly.GetGeometryRef(0).GetPoints())

            poly_idxs = np.vstack(np.where(
                (self._x_crds >= poly_xmin) &
                (self._x_crds <= poly_xmax) &
                (self._y_crds >= poly_ymin) &
                (self._y_crds <= poly_ymax))).T

            assert poly_idxs.size, (
                f'Polygon {label} with extents :{extents} out of the '
                f'coordinates\' bounds!')

            n_poly_idxs = poly_idxs.shape[0]

            # Break the number of coordinates into chunks so that later,
            # fewer coordinates have to be processed per thread. This allows
            # for distributing load more uniformly over the available threads.
            break_idxs = np.arange(
                0, (max_pts * ceil(n_poly_idxs / max_pts)) + 1, max_pts)

            break_idxs[-1] = n_poly_idxs

            assert break_idxs[-2] < break_idxs[-1]
            assert break_idxs[-1] >= n_poly_idxs

            for i in range(0, break_idxs.size - 1):
                break_poly_idxs = poly_idxs[break_idxs[i]:break_idxs[i + 1],:]

                sub_polys.append(poly)
                sub_labels.append(label)
                sub_idxs.append(break_poly_idxs)

                r_mn, c_mn = break_poly_idxs.min(axis=0)
                r_mx, c_mx = break_poly_idxs.max(axis=0) + 2

                sub_x_crds.append(self._x_crds[r_mn:r_mx, c_mn:c_mx])
                sub_y_crds.append(self._y_crds[r_mn:r_mx, c_mn:c_mx])

        return sub_polys, sub_labels, sub_idxs, sub_x_crds, sub_y_crds

    def _assemble_itsct_idxs_dict(self, ress):

        n_x_crds = self._x_crds.size

        label_data_keys = [
            'cols',
            'rows',
            'itsctd_area',
            'x_cen_crds',
            'y_cen_crds']

        label_dtypes = [
            int, int, float, float, float]

        assert len(label_data_keys) == len(label_dtypes)

        itsct_idxs_dict = {}
        cells_area_sum = 0.0

        for label in self._labels:
            label_data = []

            for res in ress:
                if res[0] != label:
                    continue

                assert len(label_data_keys) == len(res[1]), (
                    'Mismatch in expected and present number of outputs!')

                label_data.append(res[1])

            assert label_data, f'No data for label: {label}!'

            label_dict = {}
            for key in label_data_keys:
                label_dict[key] = []

                [label_dict[key].extend(data[key]) for data in label_data]

            rows_cols = pd.Index([
                (row, col)
                for row, col in zip(label_dict['rows'], label_dict['cols'])],
                dtype=object)

            dupd_idxs = np.where(rows_cols.duplicated(False))[0]

            fin_idxs = None
            if dupd_idxs.size:  # Work out the repeated cells.
                fin_idxs = self._drop_repeated_idxs(
                    rows_cols, dupd_idxs, label_dict)

            for key, dtype in zip(label_data_keys, label_dtypes):
                label_dict[key] = np.array(label_dict[key], dtype=dtype)

                if fin_idxs is not None:
                    label_dict[key] = label_dict[key][fin_idxs]

            n_itsctd_cells = label_dict['itsctd_area'].size

            cells_area = label_dict['itsctd_area'].sum()

            assert cells_area > 0, (
                f'Intersected area of polygons of label {label} have an '
                f'area of zero!')

            cells_area_sum += cells_area

            label_dict['rel_itsctd_area'] = (
                label_dict['itsctd_area'] / cells_area)

            assert all([value.size for value in label_dict.values()]), (
                f'Polygons {label} has invalid intersection data!')

            itsct_idxs_dict[label] = label_dict

            if self._vb:
                print(
                    f'INFO: Area of {label}: {cells_area} units having a '
                    f'total of {n_itsctd_cells} cells contained or within '
                    f'proximity out of '
                    f'{n_x_crds} ({100 * n_itsctd_cells / n_x_crds:0.1f}%).')

        return itsct_idxs_dict, cells_area_sum

    def _drop_repeated_idxs(self, rows_cols, dupd_idxs, label_dict):

        fin_idxs = list(range(rows_cols.size))

        done_idxs = set()
        for row, col in rows_cols[dupd_idxs]:
            if (row, col) in done_idxs:
                continue

            idxs = rows_cols.get_loc((row, col))

            if isinstance(idxs, slice):
                cdupd_idxs = list(range(idxs.start, idxs.stop, idxs.step))

            elif isinstance(idxs, np.ndarray):
                cdupd_idxs = np.where(idxs)[0].tolist()

            elif isinstance(idxs, int):
                raise ValueError('Expected more than one index!')

            else:
                raise ValueError(
                    f'Don\'t know what to do with: {idxs}!')

            # All except the first are taken.
            itsctd_area = sum(
                [label_dict['itsctd_area'][idx] for idx in cdupd_idxs])

            label_dict['itsctd_area'][cdupd_idxs[0]] = itsctd_area

            for cdupd_idx in cdupd_idxs[1:]:
                del fin_idxs[fin_idxs.index(cdupd_idx)]

                label_dict['itsctd_area'][cdupd_idx] = np.nan

                assert np.isclose(
                    label_dict['x_cen_crds'][cdupd_idx],
                    label_dict['x_cen_crds'][cdupd_idxs[0]])

                assert np.isclose(
                    label_dict['y_cen_crds'][cdupd_idx],
                    label_dict['y_cen_crds'][cdupd_idxs[0]])

                label_dict['x_cen_crds'][cdupd_idx] = np.nan
                label_dict['y_cen_crds'][cdupd_idx] = np.nan

            done_idxs.add((row, col))

        assert len(fin_idxs)

        fin_idxs = np.array(fin_idxs, dtype=np.uint64)

        return fin_idxs

    @staticmethod
    def _cmpt_poly_itsct_idxs(args):

        (poly_or_pts,
         label,
         sub_idxs,  # Has the indices of these x_crds and y_crds on the grid.
         x_crds,  # Corner crds.
         y_crds,  # Corner crds.
         vb,
         min_itsct_area_pct_thresh) = args

        if not isinstance(poly_or_pts, ogr.Geometry):
            # Points passed instead of polygons due to multiprocessing.
            n_pts = len(poly_or_pts)

            assert n_pts >= 3, (
                f'Polygon not having enough points ({n_pts})!')

            ring = ogr.Geometry(ogr.wkbLinearRing)
            for pt in poly_or_pts:
                ring.AddPoint(*pt)

            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # Needed for a special case.
            mp_flag = True

        else:
            poly = poly_or_pts

            mp_flag = False

        poly_or_pts = None

        assert poly is not None, 'Corrupted polygon after buffering!'

        poly_area = poly.Area()
        assert poly_area > 0, f'Polygon has no area!'
        #======================================================================

        n_cells_acptd = 0
        x_crds_acptd_idxs = []
        y_crds_acptd_idxs = []
        itsct_areas = []
        x_crds_acptd = []
        y_crds_acptd = []

        min_row_idx, min_col_idx = sub_idxs.min(axis=0)

        # In those rare cases, when the intersection area of all cells is
        # below threshold that results in no acpted cells, even when there
        # was intersection.
        atleast_1cell_acpt_flag = False

        for row_idx, col_idx in sub_idxs:
            cell_ring = ogr.Geometry(ogr.wkbLinearRing)

            ridx = row_idx - min_row_idx
            cidx = col_idx - min_col_idx

            cell_ring.AddPoint_2D(x_crds[ridx, cidx], y_crds[ridx, cidx])

            cell_ring.AddPoint_2D(
                x_crds[ridx + 1, cidx], y_crds[ridx + 1, cidx])

            cell_ring.AddPoint_2D(
                x_crds[ridx + 1, cidx + 1], y_crds[ridx + 1, cidx + 1])

            cell_ring.AddPoint_2D(
                x_crds[ridx, cidx + 1], y_crds[ridx, cidx + 1])

            cell_ring.AddPoint_2D(x_crds[ridx, cidx], y_crds[ridx, cidx])

            cell_poly = ogr.Geometry(ogr.wkbPolygon)
            cell_poly.AddGeometry(cell_ring)

            cell_area = cell_poly.Area()

            assert cell_area > 0, 'Area of a cell is zero!'

            itsct_poly = cell_poly.Intersection(poly)
            itsct_cell_area = itsct_poly.Area()

            assert 0.0 <= itsct_cell_area < np.inf, (
                f'Intersection area between a cell and polygon: '
                f'{label} not between zero and infinity!')

            if itsct_cell_area == 0:
                continue

            atleast_1cell_acpt_flag = True

            min_area_thresh = (
                (min_itsct_area_pct_thresh / 100.0) * cell_area)

            if itsct_cell_area < min_area_thresh:
                continue

            n_cells_acptd += 1

            x_crds_acptd_idxs.append(col_idx)
            y_crds_acptd_idxs.append(row_idx)

            itsct_areas.append(itsct_cell_area)

            centroid = cell_poly.Centroid()
            x_crds_acptd.append(centroid.GetX())
            y_crds_acptd.append(centroid.GetY())
        #======================================================================

        if not mp_flag:
            # Multiple polygons of the same label may have the case that some
            # of them do not intersect enough with the raster but others do.
            # So, we just throw a warning, instead of stopping.
            # Total intersected cell area is checked again later for
            # having a non-zero value when data from all polygons of the same
            # label is consolidated.
            assert atleast_1cell_acpt_flag or (n_cells_acptd > 0), (
                f'Zero cells accepted for polygon: {label}!')

        assert n_cells_acptd == len(x_crds_acptd_idxs)
        assert n_cells_acptd == len(y_crds_acptd_idxs)
        assert n_cells_acptd == len(itsct_areas)
        assert n_cells_acptd == len(x_crds_acptd)
        assert n_cells_acptd == len(y_crds_acptd)

        if vb:
            print(
                f'{n_cells_acptd} cells contained or within proximity out of '
                f'{sub_idxs.shape[0]}.')

        if n_cells_acptd == 0:
            print(
                f'WARNING: Polygon {label} with point a count of '
                f'{poly.GetGeometryRef(0).GetPointCount()} has zero '
                f'accepted cells!')

        return (label, {
            'cols': x_crds_acptd_idxs,
            'rows': y_crds_acptd_idxs,
            'itsctd_area': itsct_areas,
            'x_cen_crds': x_crds_acptd,
            'y_cen_crds': y_crds_acptd, })

    __verify = verify


class ReOrderIdxs:

    '''
    Reorder destination cell values based on their proximity to reference
    '''

    def __init__(self, verbose=True):

        self._vb = verbose

        self.show_ref_dst_crds_flag = False

        self._cmn_keys = ('cols', 'rows', 'x_cen_crds', 'y_cen_crds')

        self._ref_idxs = None
        self._dst_idxs = None

        self._rord_dst_idxs = None

        self._set_ref_flag = False
        self._set_dst_flag = False
        self._reordd_flag = False
        return

    def set_reference(self, indices):

        '''
        Set the indices dictionary of the reference dataset

        Parameters
        ----------
        indices : dict
            A dictionary with labels whose values are also dictionaries.
            Its format and contents are the same as the dictionary
            returned by get_intersect_indices of PolyAndCrdsItsctIdxs.
            These are used as the reference values. Values for each label
            in the dictionary set in the set_destination method are
            moved from their positions such that they follow the
            same order as the x_cen_crds and y_cen_crds in indices set
            in this method. All arrays should be 1D.
        '''

        if self._vb:
            print_sl()
            print('Setting reference indices for reordering...')

        assert isinstance(indices, dict), 'indices not a dictionary!'
        assert indices, 'Empty indices dictionary!'

        self._ref_idxs = indices

        if self._vb:
            print(
                f'Set reference indices for reordering with '
                f'{len(self._ref_idxs)} labels')

            print_el()

        self._set_ref_flag = True
        return

    def set_destination(self, indices):

        '''
        Set the indices dictionary of the destination dataset

        Parameters
        ----------
        indices : dict
            A dictionary with labels whose values are also dictionaries.
            Its format and contents are the same as the dictionary
            returned by get_intersect_indices of PolyAndCrdsItsctIdxs.
            These are used as the destination values. Values for each label
            in the dictionary set in this method are
            moved from their positions such that they follow the
            same order as the x_cen_crds and y_cen_crds in indices set
            in the set_reference method. Each array in the corresponding
            reference array should have the same number of items with only
            one dimension.
        '''

        if self._vb:
            print_sl()
            print('Setting destination indices for reordering...')

        assert isinstance(indices, dict), 'indices not a dictionary!'
        assert indices, 'Empty indices dictionary!'

        self._dst_idxs = indices

        if self._vb:
            print(
                f'Set destination indices for reordering with '
                f'{len(self._dst_idxs)} labels')

            print_el()

        self._set_dst_flag = True
        return

    def reorder(self, maximum_distance):

        '''Reorder the entries in the destination array such that they follow
        the same order as the reference in space.

        For example, reference indices are coming from a netCDF and
        destination indices are coming from a GeoTiff for the same
        polygons. In this case the row and column indices of the netCDF
        and the GeoTiff are not the same even if the extents are the
        same because GeoTiffs start counting from the top left corner while
        the netCDF from the bottom left (usually). Assuming that the cells
        of both the raster align, with a consequence that centroids of each
        cell in both the datasets have a matching one in the other but only
        the order is different. Calling the reorder create a new indices
        dictionary that has the same values as the destination but with the
        correct order. This allows for writing this dictionary to the HDF5
        dataset that already has the netCDF values by passing the
        ignore_rows_cols_equality as True to the extract_values method in
        the ExtractGTiffValues class. Same can be done for the netCDF if
        GeoTiff is used as reference.

        Parameters
        ----------
        maximum_distance : int/float
            The maximum distance allowed between a point in reference
            and the nearest cell in the destination.
            An AssertionError is raised if at least one point\'s distance
            is greater than this.
        '''

        if self._vb:
            print_sl()
            print('Reordrering...')

        assert self._set_ref_flag, 'Call the set_reference method first!'
        assert self._set_dst_flag, 'Call the set_destination method first!'

        assert isinstance(maximum_distance, (int, float)), (
            'maximum_distance neither an integer nor a float!')

        assert 0 <= maximum_distance < np.inf, (
            'maximum_distance can only be between zero and infinity!')

        reordd_idxs = {}
        for label in self._ref_idxs:
            print(f'Going through label: {label}')

            assert label in self._dst_idxs, (
                f'Label: {label} not in destination!')

            ref_label_dict = self._ref_idxs[label]
            dst_label_dict = self._dst_idxs[label]

            assert all([key in ref_label_dict for key in self._cmn_keys]), (
                f'One of the required keys {self._cmn_keys} is missing '
                f'in the reference!')

            assert all([key in dst_label_dict for key in self._cmn_keys]), (
                f'One of the required keys {self._cmn_keys} is missing '
                f'in the destination!')

            ref_shape = None

            reordd_idxs[label] = {}

            reord_label_idxs = reordd_idxs[label]

            for key in ref_label_dict:
                assert isinstance(ref_label_dict[key], np.ndarray), (
                    f'Value of the key: {key} '
                    f'not a np.ndarray in reference!')

                assert isinstance(dst_label_dict[key], np.ndarray), (
                    f'Value of the key: {key} '
                    f'not a np.ndarray in destination!')

                if ref_shape is None:
                    ref_shape = ref_label_dict[key].shape

                assert ref_label_dict[key].shape == ref_shape, (
                    f'Shape of array of key: {key} '
                    f'in reference not matching the first\'s shape!')

                assert dst_label_dict[key].shape == ref_shape, (
                    f'Shape of array of key: {key} '
                    f'in destination not matching the reference\'s shape!')

                assert ref_label_dict[key].ndim == 1, (
                    f'Reference array with key: {key} not 1D!')
                assert dst_label_dict[key].ndim == 1, (
                    f'Destination array with key: {key} not 1D!')

            print(f'Label: {label} has {ref_shape[0]} points')

            reord_idxs_arr = np.full(ref_shape, np.nan, dtype=float)
            min_dists = np.full(ref_shape, np.nan, dtype=float)

            ref_x_crds = ref_label_dict['x_cen_crds']
            ref_y_crds = ref_label_dict['y_cen_crds']

            dst_x_crds = dst_label_dict['x_cen_crds']
            dst_y_crds = dst_label_dict['y_cen_crds']

            if self.show_ref_dst_crds_flag:
                print(
                    f'i   |      RX      |      RY      |      DX      |'
                    f'      DY      |   Distance')

            for i in range(ref_shape[0]):
                ref_x_crd = ref_x_crds[i]
                ref_y_crd = ref_y_crds[i]

                x_sq_diff = (ref_x_crd - dst_x_crds) ** 2
                y_sq_diff = (ref_y_crd - dst_y_crds) ** 2

                ref_dst_dists = (x_sq_diff + y_sq_diff) ** 0.5

                min_dist_idx = np.argmin(ref_dst_dists)

                min_dists[i] = ref_dst_dists[min_dist_idx]

                reord_idxs_arr[i] = min_dist_idx

                if self.show_ref_dst_crds_flag:
                    print(
                        f'{i:<4d}|{ref_x_crd:^14.5f}|{ref_y_crd:^14.5f}|'
                        f'{dst_x_crds[min_dist_idx]:^14.5f}|'
                        f'{dst_x_crds[min_dist_idx]:^14.5f}|   '
                        f'{min_dists[i]}')

            assert np.all(np.isfinite(reord_idxs_arr)), (
                'Invalid values in the reordered index array!')

            reord_idxs_arr = reord_idxs_arr.astype(int)

            assert np.unique(reord_idxs_arr).shape == reord_idxs_arr.shape, (
                'Non-unique indices in the reordered indices array!')

            assert np.all(np.isfinite(min_dists)), (
                'Invalid values in the minimum distance array!')

            ge_dist_idxs = min_dists > maximum_distance
            n_ge_dist_idxs = ge_dist_idxs.sum()

            if n_ge_dist_idxs:
                raise AssertionError(
                    f'{n_ge_dist_idxs} have minimum distance greater '
                    f'than the limit: {maximum_distance}!')

            for key in dst_label_dict:
                reord_label_idxs[key] = dst_label_dict[key][reord_idxs_arr]

        assert reordd_idxs, 'This should not have happend!'

        self._rord_dst_idxs = reordd_idxs

        if self._vb:
            print('Done reordrering')
            print_el()

        self._reordd_flag = True
        return

    def get_reordered_destination(self):

        '''Get the reorder destination indices

        Returns
        -------
        _rord_dst_idxs : dict
            A dictionary same as the one specifed in the set_destination
            method but with the order changed for every label in
            accordance with the indices set in the set_reference method.
        '''

        assert self._reordd_flag, 'Call the reorder method first!'

        assert self._rord_dst_idxs is not None, 'This should not have happend!'

        return self._rord_dst_idxs

