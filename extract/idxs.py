'''
Created on May 27, 2019

@author: Faizan-Uni
'''

import ogr
import gdal
import pyproj
import numpy as np

from ..misc import print_sl, print_el, gdal_err_hdlr

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

        self._ras_type_labs = ('nc', 'gtiff')

        self._x_crds_orig = None
        self._y_crds_orig = None
        self._x_crds = None
        self._y_crds = None
        self._ras_type_lab = None
        self._crds_ndims = None

        self._crds_src_epsg = None
        self._crds_dst_epsg = None
        self._crds_tfmr = None

        self._min_itsct_area_pct_thresh = 1
        self._max_itct_cells_thresh = 5000

        self._itsct_idxs_dict = None

        self._set_itsct_idxs_geoms_flag = False
        self._set_itsct_idxs_crds_flag = False
        self._set_itsct_idxs_vrfd_flag = False
        self._itsct_idxs_cmptd_flag = False
        return

    def set_geometries(self, geometries, geometry_type):

        '''Set the geometries for intersection.

        Parameters
        ----------
        geometries : dict
            A dictionary whose values are gdal/ogr point/polygon geometries.
        geometry_type : int
            An integer corresponding to the geometry types of ogr.
            Currently supported are points (geometry_type=1), polygons
            (geometry_type=3) and multipolygons (geometry_type=6).
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
            f'geometry_type can be one of {self._geom_types} only!')

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
            maximum_cells_threshold_per_polygon):

        assert isinstance(
            minimum_cell_area_intersection_percentage, (int, float)), (
                'minimum_cell_area_intersection_percentage not a float '
                'or an integer!')

        assert 0 <= minimum_cell_area_intersection_percentage <= 100, (
            'minimum_cell_area_intersection_percentage can only be between '
            '0 and 100!')

        assert isinstance(maximum_cells_threshold_per_polygon, int), (
            'maximum_cells_threshold_per_polygon not an integer!')

        assert 0 <= maximum_cells_threshold_per_polygon < np.inf, (
            'maximum_cells_threshold_per_polygon can be between 0 '
            'and infinity only!')

        self._min_itsct_area_pct_thresh = (
            minimum_cell_area_intersection_percentage)

        self._max_itct_cells_thresh = maximum_cells_threshold_per_polygon

        if self._vb:
            print_sl()

            print(f'INFO: Set the following misc. parameters:')

            print(
                f'Minimum cell area intersection percentage: '
                f'{minimum_cell_area_intersection_percentage}')

            print(
                f'Maximum cells threshold per polygon: '
                f'{maximum_cells_threshold_per_polygon}')

            print_el()
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
            x_crds, y_crds = self._reproject(x_crds, y_crds)

        self._x_crds = x_crds
        self._y_crds = y_crds

        self._x_crds.flags.writeable = False
        self._y_crds.flags.writeable = False

        x_min = self._x_crds.min()
        x_max = self._x_crds.max()

        y_min = self._y_crds.min()
        y_max = self._y_crds.max()

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
                    f'within the bounds of the X coordinates!')

                assert y_min <= pt_y <= y_max, (
                    f'Y coordinate of the point: {label} is not '
                    f'within the bounds of the Y coordinates!')

        elif self._geom_type == 3:
            for label in self._labels:
                geom = self._geoms[label]

                gx_min, gx_max, gy_min, gy_max = geom.GetEnvelope()

                assert gx_min >= x_min, (
                    f'Minimum X coordinate of the polygon: {label} is less '
                    f'than the minimum X coordinate!')

                assert gx_max <= x_max, (
                    f'Maximum X coordinate of the polygon: {label} is greater '
                    f'than the maximum X coordinate!')

                assert gy_min >= y_min, (
                    f'Minimum Y coordinate of the polygon: {label} is less '
                    f'than the minimum Y coordinate!')

                assert gy_max <= y_max, (
                    f'Maximum Y coordinate of the polygon: {label} is greater '
                    f'than the maximum Y coordinate!')

        else:
            raise NotImplementedError

        if self._vb:
            print(
                f'INFO: All inputs for geometries\' and coordinates\' '
                f'intersection verified to be correct')

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

        if self._geom_type == 1:
            if self._crds_ndims == 1:
                itsct_idxs_dict = self._cmpt_1d_pt_idxs()

            else:
                raise NotImplementedError

        elif self._geom_type == 3:
            itsct_idxs_dict = {}

            for label in self._labels:
                geom = self._geoms[label]

                if self._vb:
                    print('\n')
                    print(f'Going through polygon: {label}...')

                if self._crds_ndims == 1:
                    res = self._cmpt_1d_poly_idxs(geom, label)

                elif self._crds_ndims == 2:
                    res = self._cmpt_2d_poly_idxs(geom, label)

                else:
                    raise NotImplementedError

                itsct_idxs_dict[label] = res

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
            np.all(np.flipud(diffs_ud) > 0)), (
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
                0.5 * (y_crds[+1, +1:] - y_crds[+0, :-1]))

            y_crds_rect[-1, +1:-1] = y_crds[-1, +1:] + (
                0.5 * (y_crds[-1, :-1] - y_crds[-2, +1:]))

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
                0.5 * (x_crds[+0, +1:] - x_crds[+0, :-1]))

            x_crds_rect[-1, +1:-1] = x_crds[-1, +1:] - (
                0.5 * (x_crds[-1, +1:] - x_crds[-1, :-1]))

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

        x_crds = np.tile(self._x_crds[x_idxs], n_y_idxs)
        y_crds = np.repeat(self._y_crds[y_idxs], n_x_idxs)

        assert x_crds.size == y_crds.size, (
            'x_crds and y_crds not having same length!')

        idxs = np.full((n_pts, 2), np.nan)

        show_crds_flag = False

        if show_crds_flag:
            print(
                f'   i, Label    |      RX      |      RY      |'
                f'      DX      |'
                f'      DY      |   Distance')

        itsct_idxs_dict = {}
        for i, label in enumerate(self._labels):
            pt_x, pt_y = pt_crds[i]

            x_sq_diff = (pt_x - x_crds) ** 2
            y_sq_diff = (pt_y - y_crds) ** 2

            dists = (x_sq_diff + y_sq_diff) ** 0.5

            min_dist_idx = np.argmin(dists)

            min_x_crd_idx = x_idxs[(min_dist_idx % n_x_idxs)]
            min_y_crd_idx = y_idxs[(min_dist_idx // n_x_idxs)]

            idxs[i] = min_x_crd_idx, min_y_crd_idx

            min_dist = (
                ((pt_x - self._x_crds[min_x_crd_idx]) ** 2) +
                ((pt_y - self._y_crds[min_y_crd_idx]) ** 2)) ** 0.5

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
                    [self._y_crds[min_y_crd_idx]], dtype=float), }

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

        idxs = np.full((n_pts, 2), np.nan)

        show_crds_flag = True

        if show_crds_flag:
            print(
                f'   i, Label   |      RX      |      RY      |      DX      |'
                f'      DY      |   Distance')

        itsct_idxs_dict = {}
        for i, label in enumerate(self._labels):
            pt_x, pt_y = pt_crds[i]

            x_sq_diff = (pt_x - x_crds) ** 2
            y_sq_diff = (pt_y - y_crds) ** 2

            dists = (x_sq_diff + y_sq_diff) ** 0.5

            min_dist_idx = np.argmin(dists)

            min_row_crd_idx, min_col_crd_idx = tot_idxs[min_dist_idx]

            idxs[i] = min_row_crd_idx, min_col_crd_idx

            min_dist = (
                ((pt_x - self._x_crds[min_row_crd_idx, min_col_crd_idx]) ** 2) +
                ((pt_y - self._y_crds[min_row_crd_idx, min_col_crd_idx]) ** 2)) ** 0.5

            if show_crds_flag:

                print(
                    f'{i:4d}, {label:<9s}|{pt_x:^14.5f}|{pt_y:^14.5f}|'
                    f'{self._x_crds[min_row_crd_idx, min_col_crd_idx]:^14.5f}|'
                    f'{self._y_crds[min_row_crd_idx, min_col_crd_idx]:^14.5f}|   '
                    f'{min_dist}')

            assert min_dist <= max_cell_dist, (
                f'Label: {label} have a distance greater than the limit'
                f'to the centroid of the nearest cell!')

            itsct_idxs_dict[label] = {
                'cols':np.array([min_col_crd_idx], dtype=int),
                'rows': np.array([min_row_crd_idx], dtype=int),
                'itsctd_area': np.array([0.0], dtype=float),
                'rel_itsctd_area': np.array([0.0], dtype=float),
                'x_cen_crds': np.array(
                    [self._x_crds[min_row_crd_idx, min_col_crd_idx]], dtype=float),
                'y_cen_crds': np.array(
                    [self._y_crds[min_row_crd_idx, min_col_crd_idx]], dtype=float), }

        return itsct_idxs_dict

    def _cmpt_1d_poly_idxs(self, geom, label):

        assert self._crds_ndims == 1, 'Configured for 1D coordinates only!'
        assert self._geom_type == 3, 'Configured for polygon geometries only!'

        x_crds = self._x_crds
        y_crds = self._y_crds

        geom_area = geom.Area()
        assert geom_area > 0, f'Polygon: {label} has no area!'

        geom_buff = geom.Buffer(max(
            abs(x_crds[+1] - x_crds[+0]),
            abs(x_crds[-1] - x_crds[-2]),
            abs(y_crds[+1] - y_crds[+0]),
            abs(y_crds[-1] - y_crds[-2]),
            ))

        assert geom_buff is not None, (
            f'Buffer operation failed on polygon: {label}!')

        geom_buff_area = geom_buff.Area()
        assert geom_buff_area > 0, f'Buffered polygon: {label} has no area!'

        assert geom_buff_area >= geom_area, (
            f'Buffered polygon: {label} area less than the original one!')

        extents = geom_buff.GetEnvelope()

        assert len(extents) == 4, 'Configured for 2D extents only!'

        gx_min, gx_max, gy_min, gy_max = extents

        tot_x_idxs = np.where((x_crds >= gx_min) & (x_crds <= gx_max))[0]
        tot_y_idxs = np.where((y_crds >= gy_min) & (y_crds <= gy_max))[0]

        assert tot_x_idxs.size, (
            f'No X coordinate selected for the polygon: {label}!')

        assert tot_y_idxs.size, (
            f'No Y coordinate selected for the polygon: {label}!')

        n_cells_acptd = 0
        x_crds_acptd_idxs = []
        y_crds_acptd_idxs = []
        itsct_areas = []
        itsct_rel_areas = []
        x_crds_acptd = []
        y_crds_acptd = []

        if self._vb:
            print(
                f'Testing {tot_x_idxs.size * tot_y_idxs.size} cells '
                f'for containment/proximity...')

        for x_idx in tot_x_idxs:
            for y_idx in tot_y_idxs:
                ring = ogr.Geometry(ogr.wkbLinearRing)

                ring.AddPoint(x_crds[x_idx], y_crds[y_idx])
                ring.AddPoint(x_crds[x_idx + 1], y_crds[y_idx])
                ring.AddPoint(x_crds[x_idx + 1], y_crds[y_idx + 1])
                ring.AddPoint(x_crds[x_idx], y_crds[y_idx + 1])
                ring.AddPoint(x_crds[x_idx], y_crds[y_idx])

                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                poly_area = poly.Area()

                assert poly_area > 0, 'Area of a cell is zero!'

                itsct_poly = poly.Intersection(geom)
                itsct_cell_area = itsct_poly.Area()

                assert 0.0 <= itsct_cell_area < np.inf, (
                    f'Intersection area between a cell and polygon: '
                    f'{label} not between zero and infinity!')

                min_area_thresh = (
                    (self._min_itsct_area_pct_thresh / 100.0) * poly_area)

                if itsct_cell_area < min_area_thresh:
                    continue

                n_cells_acptd += 1

                x_crds_acptd_idxs.append(x_idx)
                y_crds_acptd_idxs.append(y_idx)

                itsct_areas.append(itsct_cell_area)
                itsct_rel_areas.append(itsct_cell_area / geom_area)

                centroid = poly.Centroid()
                x_crds_acptd.append(centroid.GetX())
                y_crds_acptd.append(centroid.GetY())

        assert n_cells_acptd <= (tot_x_idxs.size * tot_y_idxs.size), (
            'This should not have happend!')

        assert n_cells_acptd > 0, f'Zero cells accepted for polygon: {label}!'
        assert n_cells_acptd == len(x_crds_acptd_idxs)
        assert n_cells_acptd == len(y_crds_acptd_idxs)
        assert n_cells_acptd == len(itsct_areas)
        assert n_cells_acptd == len(itsct_rel_areas)
        assert n_cells_acptd == len(x_crds_acptd)
        assert n_cells_acptd == len(y_crds_acptd)

        if self._vb:
            print(f'{n_cells_acptd} cells contained/in proximity')

        return {
            'cols':np.array(x_crds_acptd_idxs, dtype=int),
            'rows': np.array(y_crds_acptd_idxs, dtype=int),
            'itsctd_area': np.array(itsct_areas, dtype=float),
            'rel_itsctd_area': np.array(itsct_rel_areas, dtype=float),
            'x_cen_crds': np.array(x_crds_acptd, dtype=float),
            'y_cen_crds': np.array(y_crds_acptd, dtype=float), }

    def _cmpt_2d_poly_idxs(self, geom, label):

        assert self._crds_ndims == 2, 'Configured for 2D coordinates only!'

        geom_area = geom.Area()
        assert geom_area > 0, f'Polygon: {label} has no area!'

        x_crds = self._x_crds
        y_crds = self._y_crds

        geom_buff = geom.Buffer(max(
            abs(x_crds[+1, +0] - x_crds[+0, +0]),
            abs(x_crds[-1, -1] - x_crds[-2, -1]),
            abs(y_crds[+1, +0] - y_crds[+0, 0]),
            abs(y_crds[-1, -1] - y_crds[-2, -1]),
            ))

        assert geom_buff is not None, (
            f'Buffer operation failed on polygon: {label}!')

        geom_buff_area = geom_buff.Area()
        assert geom_buff_area > 0, f'Buffered Polygon: {label} has no area!'

        assert geom_buff_area >= geom_area, (
            f'Buffered polygon: {label} area less than the original one!')

        extents = geom_buff.GetEnvelope()

        assert len(extents) == 4, 'Configured for 2D extents only!'

        gx_min, gx_max, gy_min, gy_max = extents

        tot_idxs = np.vstack(np.where(
            (x_crds >= gx_min) &
            (x_crds <= gx_max) &
            (y_crds >= gy_min) &
            (y_crds <= gy_max))).T

        keep_idxs = ~(
            (tot_idxs[:, 0] >= (x_crds.shape[0] - 1)) |
            (tot_idxs[:, 1] >= (x_crds.shape[1] - 1)))

        tot_idxs = tot_idxs[keep_idxs].copy('c')

        assert tot_idxs.size, (
            f'No cell selected for the polygon: {label}!')

        assert np.all(x_crds.shape), (
            'Shape of X coordinates not allowed to have a zero!')

        assert np.all(y_crds.shape), (
            'Shape of Y coordinates not allowed to have a zero!')

        n_cells_acptd = 0
        x_crds_acptd_idxs = []
        y_crds_acptd_idxs = []
        itsct_areas = []
        itsct_rel_areas = []
        x_crds_acptd = []
        y_crds_acptd = []

        for row_idx, col_idx in tot_idxs:
            ring = ogr.Geometry(ogr.wkbLinearRing)

            ring.AddPoint(
                x_crds[row_idx, col_idx],
                y_crds[row_idx, col_idx])

            ring.AddPoint(
                x_crds[row_idx + 1, col_idx],
                y_crds[row_idx + 1, col_idx])

            ring.AddPoint(
                x_crds[row_idx + 1, col_idx + 1],
                y_crds[row_idx + 1, col_idx + 1])

            ring.AddPoint(
                x_crds[row_idx, col_idx + 1],
                y_crds[row_idx, col_idx + 1])

            ring.AddPoint(
                x_crds[row_idx, col_idx],
                y_crds[row_idx, col_idx])

            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            poly_area = poly.Area()

            assert poly_area > 0, 'Area of a cell is zero!'

            itsct_poly = poly.Intersection(geom)
            itsct_cell_area = itsct_poly.Area()

            assert 0.0 <= itsct_cell_area < np.inf, (
                f'Intersection area between a cell and polygon: '
                f'{label} not between zero and infinity!')

            min_area_thresh = (
                (self._min_itsct_area_pct_thresh / 100.0) * poly_area)

            if itsct_cell_area < min_area_thresh:
                continue

            n_cells_acptd += 1

            x_crds_acptd_idxs.append(col_idx)
            y_crds_acptd_idxs.append(row_idx)

            itsct_areas.append(itsct_cell_area)
            itsct_rel_areas.append(itsct_cell_area / geom_area)

            centroid = poly.Centroid()
            x_crds_acptd.append(centroid.GetX())
            y_crds_acptd.append(centroid.GetY())

        assert n_cells_acptd <= tot_idxs.size, 'This should not have happend!'

        assert n_cells_acptd > 0, f'Zero cells accepted for polygon: {label}!'
        assert n_cells_acptd == len(x_crds_acptd_idxs)
        assert n_cells_acptd == len(y_crds_acptd_idxs)
        assert n_cells_acptd == len(itsct_areas)
        assert n_cells_acptd == len(itsct_rel_areas)
        assert n_cells_acptd == len(x_crds_acptd)
        assert n_cells_acptd == len(y_crds_acptd)

        return {
            'cols':np.array(x_crds_acptd_idxs, dtype=int),
            'rows': np.array(y_crds_acptd_idxs, dtype=int),
            'itsctd_area': np.array(itsct_areas, dtype=float),
            'rel_itsctd_area': np.array(itsct_rel_areas, dtype=float),
            'x_cen_crds': np.array(x_crds_acptd, dtype=float),
            'y_cen_crds': np.array(y_crds_acptd, dtype=float), }

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
