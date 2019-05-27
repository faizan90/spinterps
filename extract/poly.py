'''
Created on May 27, 2019

@author: Faizan-Uni
'''

from pathlib import Path

import ogr

from ..misc import print_sl, print_el


class ExtractReferencePolygons:

    def __init__(self, verbose=True):

        self._vb = verbose

        self._poly_shp_path = None
        self._poly_label_field = None

        self._poly_labels = None
        self._poly_geoms = None
        self._poly_areas = None
        self._poly_extents = None

        self._set_poly_shp_path_flag = False
        self._set_poly_asm_flag = False
        return

    def set_path_to_poly_shp(self, path_to_shp, label_field):

        assert isinstance(path_to_shp, (str, Path))
        assert isinstance(label_field, str)

        path_to_shp = Path(path_to_shp).absolute()

        assert path_to_shp.exists()

        assert label_field

        self._poly_shp_path = path_to_shp
        self._poly_label_field = label_field

        if self._vb:
            print_sl()

            print(f'INFO: Set the following polygon(s) shapefile variables:')
            print(f'Path: {self._poly_shp_path}')
            print(f'Label field: {self._poly_label_field}')

            print_el()

        self._set_poly_shp_path_flag = True
        return

    def assemble_polygon_data(self):

        assert self._set_poly_shp_path_flag

        shp_hdl = ogr.Open(str(self._poly_shp_path))

        assert shp_hdl.GetLayerCount() == 1

        spt_ref = shp_hdl.GetSpatialRef()

        assert 'PROJCS' in spt_ref.ExportToWkt()

        poly_lyr = shp_hdl.GetLayer(0)

        assert poly_lyr.GetFeatureCount() > 0

        polygon = poly_lyr.GetNextFeature()

        labels = []
        geoms = {}
        areas = {}
        extents = {}

        while polygon:
            geom = polygon.GetGeometryRef().Clone()
            assert geom is not None

            geom_type = geom.GetGeometryType()
            geom_name = geom.GetGeometryName()

            assert geom_type == 3, (
                f'Unsupported geometry type, name: {geom_type}, {geom_name}!')

            label = polygon.GetFieldAsInterger(self._poly_label_field)
            assert label is not None
            assert isinstance(label, int)

            area = geom.Area()
            assert area is not None
            assert isinstance(area, float)
            assert area > 0

            extent = geom.GetEnvelope()
            assert extent is not None
            assert isinstance(extent, tuple)

            labels.append(label)
            geoms[label] = geom
            areas[label] = area
            extents[label] = extent

        n_polys = len(labels)

        assert n_polys == len(geoms)
        assert n_polys == len(areas)
        assert n_polys == len(extents)

        self._poly_labels = labels
        self._poly_geoms = geoms
        self._poly_areas = areas
        self._poly_extents = extents

        if self._vb:
            print_sl()

            print(f'INFO: Found {n_polys} polygons!')

            print(f'Label   |       Area       |   Extents')

            for label in self._poly_labels:
                area = self._poly_areas[label]
                extent = self._poly_extents[label]
                print(f'{label:<8d}|{area:^18f}|   {extent}')

            print_el()

        self._set_poly_asm_flag = True
        return

    def get_labels(self):

        assert self._set_poly_asm_flag

        return self._poly_labels

    def get_geometries(self):

        assert self._set_poly_asm_flag

        return self._poly_geoms

    def get_areas(self):

        assert self._set_poly_asm_flag

        return self._poly_areas

    def get_extents(self):

        assert self._set_poly_asm_flag

        return self._poly_extents
