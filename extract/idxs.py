'''
Created on May 27, 2019

@author: Faizan-Uni
'''

import ogr
import numpy as np

from ..misc import print_sl, print_el


class PolyAndCrdsItsctIdxs:

    def __init__(self, verbose=True):

        self._vb = verbose

        self._poly_geoms = None
        self._poly_labels = None

        self._ras_type_labs = ('nc', 'gtiff')

        self._x_crds_orig = None
        self._y_crds_orig = None
        self._ras_type_lab = None

        self._set_itsct_idxs_polys_flag = False
        self._set_itsct_idxs_crds_flag = False
        self._set_itsct_idxs_vrfd_flag = False
        return

    def set_polygons(self, polygon_geometries, labels):

        assert isinstance(polygon_geometries, dict)
        assert isinstance(labels, tuple)

        assert len(polygon_geometries)

        assert len(polygon_geometries) == len(labels)

        for label in labels:
            assert label in polygon_geometries

            geom = polygon_geometries[label]

            assert isinstance(geom, ogr.Geometry)

            assert geom.GetGeometryType() == 3

            assert len(geom.GetGeometryRef(0).GetPoints()) > 2

        self._poly_geoms = polygon_geometries
        self._poly_labels = labels

        if self._vb:
            print_sl()

            print(
                f'INFO: Set {len(self._poly_labels)} polygons for '
                f'intersection with coordinates')

            print_el()

        self._set_itsct_idxs_polys_flag = True
        return

    def set_coordinates(self, x_crds, y_crds, raster_type_label):

        assert isinstance(x_crds, np.ndarray)
        assert 1 <= x_crds.ndim <= 2
        assert np.all(np.isfinite(x_crds))
        assert np.all(x_crds.shape)

        assert isinstance(y_crds, np.ndarray)
        assert 1 <= y_crds.ndim <= 2
        assert np.all(np.isfinite(y_crds))
        assert np.all(y_crds.shape)

        assert isinstance(raster_type_label, str)
        assert raster_type_label in self._ras_type_labs

        self._x_crds_orig = x_crds
        self._y_crds_orig = y_crds
        self._ras_type_lab = raster_type_label

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

    def verify(self):

        assert self._set_itsct_idxs_polys_flag
        assert self._set_itsct_idxs_crds_flag

        x_min = self._x_crds_orig.min()
        x_max = self._x_crds_orig.max()

        y_min = self._y_crds_orig.min()
        y_max = self._y_crds_orig.max()

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

    __verify = verify
