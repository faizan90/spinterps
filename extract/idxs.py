'''
Created on May 27, 2019

@author: Faizan-Uni
'''

import ogr
import gdal
import numpy as np

from ..misc import print_sl, print_el, gdal_err_hdlr

gdal.PushErrorHandler(gdal_err_hdlr)
ogr.UseExceptions()


class PolyAndCrdsItsctIdxs:

    def __init__(self, verbose=True):

        self._vb = verbose

        self._poly_geoms = None
        self._poly_labels = None

        self._ras_type_labs = ('nc', 'gtiff')

        self._x_crds_orig = None
        self._y_crds_orig = None
        self._x_crds = None
        self._y_crds = None
        self._ras_type_lab = None
        self._crds_ndims = None

        self._min_itsct_area_pct_thresh = 1
        self._max_itct_cells_thresh = 5000

        self._itsct_idxs_dict = None

        self._set_itsct_idxs_polys_flag = False
        self._set_itsct_idxs_crds_flag = False
        self._set_itsct_idxs_vrfd_flag = False
        self._itsct_idxs_cmptd_flag = False
        return

    def set_polygons(self, polygon_geometries, labels):

        assert isinstance(polygon_geometries, dict)
        assert isinstance(labels, tuple)

        n_polys = len(polygon_geometries)

        assert n_polys

        assert n_polys == len(labels)

        for label in labels:
            assert label in polygon_geometries

            geom = polygon_geometries[label]

            assert isinstance(geom, ogr.Geometry)

            assert geom.GetGeometryType() == 3

            assert len(geom.GetGeometryRef(0).GetPoints()) > 2

            assert geom.Area() > 0

        self._poly_geoms = polygon_geometries
        self._poly_labels = labels

        if self._vb:
            print_sl()

            print(
                f'INFO: Set {n_polys} polygons for '
                f'intersection with coordinates')

            print_el()

        self._set_itsct_idxs_polys_flag = True
        return

    def set_coordinates(self, x_crds, y_crds, raster_type_label):

        self._verf_crds(x_crds)
        self._verf_crds(y_crds)

        assert x_crds.ndim == y_crds.ndim

        assert isinstance(raster_type_label, str)
        assert raster_type_label in self._ras_type_labs

        self._x_crds_orig = x_crds
        self._y_crds_orig = y_crds

        self._ras_type_lab = raster_type_label
        self._crds_ndims = x_crds.ndim

        self._x_crds_orig.flags.writeable = False
        self._y_crds_orig.flags.writeable = False

        if (self._ras_type_lab == 'nc') and (self._crds_ndims == 1):
            self._x_crds = self._get_rect_crds_1d(self._x_crds_orig)
            self._y_crds = self._get_rect_crds_1d(self._y_crds_orig)

        elif (self._ras_type_lab == 'gtiff') and (self._crds_ndims == 1):
            self._x_crds, self._y_crds = (
                self._x_crds_orig.copy(), self._y_crds_orig.copy())

        elif (self._ras_type_lab == 'nc') and (self._crds_ndims == 2):
            self._x_crds, self._y_crds = self._get_rect_crds_2d(
                self._x_crds_orig, self._y_crds_orig)

        else:
            raise NotImplementedError

        if self._vb:
            print_sl()

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

    def _verf_crds(self, crds):

        assert isinstance(crds, np.ndarray)
        assert np.all(np.isfinite(crds))
        assert np.all(crds.shape)

        if crds.ndim == 1:
            self._verf_crds_1d(crds)

        elif crds.ndim == 2:
            self._verf_crds_2d(crds)

        else:
            raise NotImplementedError
        return

    def _verf_crds_1d(self, crds):

        assert np.unique(crds).shape == crds.shape

        assert (
            np.all(np.ediff1d(crds) > 0) or
            np.all(np.ediff1d(crds[::-1]) > 0))
        return

    def _verf_crds_2d(self, crds):

        diffs_lr = np.diff(crds)
        diffs_ud = np.diff(crds.T)

        assert (
            np.all(diffs_lr > 0) or
            np.all(np.fliplr(diffs_lr) > 0) or
            np.all(diffs_ud > 0) or
            np.all(np.flipud(diffs_ud) > 0))
        return

    def _get_rect_crds_1d(self, crds):

        assert crds.ndim == 1

        crds_rect = np.full(crds.shape[0] + 1, np.nan)

        crds_rect[1:-1] = (crds[:-1] + crds[1:]) * 0.5

        crds_rect[+0] = crds[+0] - (0.5 * (crds[+1] - crds[+0]))
        crds_rect[-1] = crds[-1] + (0.5 * (crds[-1] - crds[-2]))

        assert np.all(np.isfinite(crds_rect))

        assert (
            np.all(np.ediff1d(crds_rect) > 0) or
            np.all(np.ediff1d(crds_rect[::-1]) > 0))

        return crds_rect

    def _get_rect_crds_2d(self, x_crds, y_crds):

        assert x_crds.ndim == y_crds.ndim
        assert x_crds.shape == y_crds.shape

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

        assert np.all(np.isfinite(x_crds_rect))
        assert np.all(np.isfinite(y_crds_rect))

        return x_crds_rect, y_crds_rect

    def set_intersect_misc_settings(
            self,
            minimum_cell_area_intersection_percentage,
            maximum_cells_threshold_per_polygon):

        assert isinstance(
            minimum_cell_area_intersection_percentage, (int, float))

        assert 0 <= minimum_cell_area_intersection_percentage <= 100

        assert isinstance(maximum_cells_threshold_per_polygon, int)

        assert 0 <= maximum_cells_threshold_per_polygon < np.inf

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

        assert self._set_itsct_idxs_polys_flag
        assert self._set_itsct_idxs_crds_flag

        x_min = self._x_crds.min()
        x_max = self._x_crds.max()

        y_min = self._y_crds.min()
        y_max = self._y_crds.max()

        for label in self._poly_labels:
            geom = self._poly_geoms[label]

            gx_min, gx_max, gy_min, gy_max = geom.GetEnvelope()

            assert gx_min >= x_min
            assert gx_max <= x_max

            assert gy_min >= y_min
            assert gy_max <= y_max

        if self._vb:
            print_sl()

            print(
                f'INFO: All inputs for polygons\' and coordinates\' '
                f'intersection verified to be correct')

            print_el()

        self._set_itsct_idxs_vrfd_flag = True
        return

    def _cmpt_1d_idxs(self, geom):

        assert self._crds_ndims == 1

        geom_area = geom.Area()
        assert geom_area > 0

        x_crds = self._x_crds
        y_crds = self._y_crds

        geom_buff = geom.Buffer(max(
            abs(x_crds[+1] - x_crds[+0]),
            abs(x_crds[-1] - x_crds[-2]),
            abs(y_crds[+1] - y_crds[+0]),
            abs(y_crds[-1] - y_crds[-2]),
            ))

        assert geom_buff is not None

        geom_buff_area = geom_buff.Area()
        assert geom_buff_area > 0

        gx_min, gx_max, gy_min, gy_max = geom_buff.GetEnvelope()

        tot_x_idxs = np.where((x_crds >= gx_min) & (x_crds <= gx_max))[0]
        tot_y_idxs = np.where((y_crds >= gy_min) & (y_crds <= gy_max))[0]

        assert tot_x_idxs.sum() > 1
        assert tot_y_idxs.sum() > 1

        assert np.all(x_crds.shape)
        assert np.all(y_crds.shape)

        n_cells_acptd = 0
        x_crds_acptd_idxs = []
        y_crds_acptd_idxs = []
        itsct_areas = []
        itsct_rel_areas = []
        x_crds_acptd = []
        y_crds_acptd = []

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

                assert poly_area > 0

                itsct_poly = poly.Intersection(geom)
                itsct_cell_area = itsct_poly.Area()

                assert 0.0 <= itsct_cell_area < np.inf

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

        assert n_cells_acptd > 0
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

    def _cmpt_2d_idxs(self, geom):

        assert self._crds_ndims == 2

        geom_area = geom.Area()
        assert geom_area > 0

        x_crds = self._x_crds
        y_crds = self._y_crds

        geom_buff = geom.Buffer(max(
            abs(x_crds[+1, +0] - x_crds[+0, +0]),
            abs(x_crds[-1, -1] - x_crds[-2, -1]),
            abs(y_crds[+1, +0] - y_crds[+0, 0]),
            abs(y_crds[-1, -1] - y_crds[-2, -1]),
            ))

        assert geom_buff is not None

        geom_buff_area = geom_buff.Area()
        assert geom_buff_area > 0

        gx_min, gx_max, gy_min, gy_max = geom_buff.GetEnvelope()

        tot_idxs = np.vstack(np.where(
            (x_crds >= gx_min) &
            (x_crds <= gx_max) &
            (y_crds >= gy_min) &
            (y_crds <= gy_max))).T

        keep_idxs = ~(
            (tot_idxs[:, 0] >= (x_crds.shape[0] - 1)) |
            (tot_idxs[:, 1] >= (x_crds.shape[1] - 1)))

        tot_idxs = tot_idxs[keep_idxs].copy('c')

        assert tot_idxs.size > 1

        assert np.all(x_crds.shape)
        assert np.all(y_crds.shape)

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

            assert poly_area > 0

            itsct_poly = poly.Intersection(geom)
            itsct_cell_area = itsct_poly.Area()

            assert 0.0 <= itsct_cell_area < np.inf

            min_area_thresh = (
                (self._min_itsct_area_pct_thresh / 100.0) * poly_area)

            if itsct_cell_area < min_area_thresh:
                continue

            print(
                x_crds[row_idx, col_idx],
                y_crds[row_idx, col_idx])

            print(
                x_crds[row_idx + 1, col_idx],
                y_crds[row_idx + 1, col_idx])

            print(
                x_crds[row_idx + 1, col_idx + 1],
                y_crds[row_idx + 1, col_idx + 1])

            print(
                x_crds[row_idx, col_idx + 1],
                y_crds[row_idx, col_idx + 1])

            print(
                x_crds[row_idx, col_idx],
                y_crds[row_idx, col_idx])

            n_cells_acptd += 1

            x_crds_acptd_idxs.append(col_idx)
            y_crds_acptd_idxs.append(row_idx)

            itsct_areas.append(itsct_cell_area)
            itsct_rel_areas.append(itsct_cell_area / geom_area)

            centroid = poly.Centroid()
            x_crds_acptd.append(centroid.GetX())
            y_crds_acptd.append(centroid.GetY())

        assert n_cells_acptd > 0
        assert n_cells_acptd == len(x_crds_acptd_idxs)
        assert n_cells_acptd == len(y_crds_acptd_idxs)
        assert n_cells_acptd == len(itsct_areas)
        assert n_cells_acptd == len(itsct_rel_areas)
        assert n_cells_acptd == len(x_crds_acptd)
        assert n_cells_acptd == len(y_crds_acptd)

        print(f'n_cells_acptd: {n_cells_acptd}')

        return {
            'cols':np.array(x_crds_acptd_idxs, dtype=int),
            'rows': np.array(y_crds_acptd_idxs, dtype=int),
            'itsctd_area': np.array(itsct_areas, dtype=float),
            'rel_itsctd_area': np.array(itsct_rel_areas, dtype=float),
            'x_cen_crds': np.array(x_crds_acptd, dtype=float),
            'y_cen_crds': np.array(y_crds_acptd, dtype=float), }

    def compute_intersect_idxs(self):

        assert self._set_itsct_idxs_vrfd_flag

        self._itsct_idxs_cmptd_flag = False

        assert self._crds_ndims < 3

        itsct_idxs_dict = {}

        for label in self._poly_labels:
            geom = self._poly_geoms[label]

            if self._crds_ndims == 1:
                res = self._cmpt_1d_idxs(geom)

            elif self._crds_ndims == 2:
                res = self._cmpt_2d_idxs(geom)

            else:
                raise NotImplementedError

            itsct_idxs_dict[label] = res

        self._itsct_idxs_dict = itsct_idxs_dict

        self._itsct_idxs_cmptd_flag = True
        return

    def get_intersect_idxs(self):

        assert self._itsct_idxs_cmptd_flag

        return self._itsct_idxs_dict

    __verify = verify
