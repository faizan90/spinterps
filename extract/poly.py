'''
Created on May 27, 2019

@author: Faizan-Uni
'''

from pathlib import Path

import ogr

from ..misc import print_sl, print_el


class ExtractPolygons:

    '''Get Polygons, their areas, extents and field values from an ESRI
    Shapefile.
    '''

    _geom_type = 3

    def __init__(self, verbose=True):

        self._vb = verbose

        self._poly_shp_path = None
        self._poly_label_field = None

        self._poly_labels = None
        self._poly_geoms = None
        self._poly_areas = None
        self._poly_extents = None

        self._set_in_flag = False
        self._set_poly_data_extrt_flag = False
        return

    def set_input(self, path_to_shp, label_field):

        '''Set the path to input ESRI Shapefile and a field name.

        Parameters
        ----------
        path_to_shp : str, pathlib.Path
            Path to the input shapefile.
        label_field : str
            Name of the field whose values will serve as labels for the
            outputs.
        '''

        if self._vb:
            print_sl()

            print('Setting polygons\' extraction input...')

        assert isinstance(path_to_shp, (str, Path)), (
            f'Specified path to input ({path_to_shp}) is not a string or '
            f'path-like object!')

        path_to_shp = Path(path_to_shp).absolute()

        assert path_to_shp.exists(), (
            f'Specified input file ({path_to_shp}) does not exist!')

        assert isinstance(label_field, str), 'label_field not a string!'
        assert label_field, 'label_field is an empty string!'

        self._poly_shp_path = path_to_shp
        self._poly_label_field = label_field

        if self._vb:
            print(f'INFO: Set the following polygon shapefile variables:')
            print(f'Path: {self._poly_shp_path}')
            print(f'Label field: {self._poly_label_field}')

            print_el()

        self._set_in_flag = True
        return

    def extract_polygons(self):

        '''Run the polygon extraction algorithm'''

        if self._vb:
            print_sl()

            print('Extracting polygons...')

        assert self._set_in_flag, 'Call the set_input method first!'

        shp_hdl = ogr.Open(str(self._poly_shp_path))

        assert shp_hdl is not None, (
            f'Unable to open the input file: {self._poly_shp_path} as an '
            f'ogr vector file!')

        assert shp_hdl.GetDriver().GetName() == 'ESRI Shapefile', (
            'Specified input vector file is not an ESRI Shapefile!')

        assert shp_hdl.GetLayerCount() == 1, (
            'Specified input vector file only allowed to have one layer!')

        poly_lyr = shp_hdl.GetLayer(0)

        assert poly_lyr.GetFeatureCount() > 0, (
            'Zero vectors inside the input shapefile!')

        labels = []
        geoms = {}
        areas = {}
        extents = {}

        polygon = poly_lyr.GetNextFeature()

        assert polygon.GetFieldCount() > 0, (
            'Zero fields inside the input shapefile!')

        assert self._poly_label_field in polygon.keys(), (
            f'label_field: {self._poly_label_field} not in input shapefile!')

        while polygon:
            geom = polygon.GetGeometryRef().Clone()

            assert geom is not None, 'Could not read/clone a geometry!'

            geom_type = geom.GetGeometryType()
            geom_name = geom.GetGeometryName()

            assert (geom_type == 3) or (geom_type == 6), (
                f'Unsupported geometry type, name: {geom_type}, {geom_name}!')

            assert geom.GetGeometryCount() >= 1, (
                'Only one polygon allowed per feature!')

            if geom_type == 3:
                assert len(geom.GetGeometryRef(0).GetPoints()) >= 3, (
                    f'A polygon has less than 3 points!')

            elif geom_type == 6:
                for sub_geom in geom:
                    assert len(sub_geom.GetGeometryRef(0).GetPoints()) >= 3, (
                        f'A polygon has less than 3 points!')

            else:
                raise NotImplementedError

            label = polygon.GetFieldAsString(self._poly_label_field)

            assert label is not None, (
                f'Could not read the field: {label} as a string!')

            area = geom.Area()
            assert area is not None, (
                f'Could not read area of the polygon: {label}!')

            assert isinstance(area, float), f'Area of polygon not a float!'
            assert area > 0, f'Polygon: {label} has no area!'

            extent = geom.GetEnvelope()
            assert extent is not None, (
                f'Could not read spatial bounds of the polygon: {label}!')

            assert isinstance(extent, tuple), (
                'Polygon: {label}\'s extents not a tuple!')

            labels.append(label)
            geoms[label] = geom
            areas[label] = area
            extents[label] = extent

            polygon = poly_lyr.GetNextFeature()

        shp_hdl.Destroy()
        shp_hdl = None

        n_polys = len(labels)

        assert n_polys, 'This should not have happend!'

        labels_set = set(labels)

        assert n_polys == len(labels_set), (
            'Non-unique field values of polygons!')

        assert n_polys == len(geoms), 'This should not have happend!'
        assert n_polys == len(areas), 'This should not have happend!'
        assert n_polys == len(extents), 'This should not have happend!'

        self._poly_labels = tuple(labels)
        self._poly_geoms = geoms
        self._poly_areas = areas
        self._poly_extents = extents

        if self._vb:
            print(f'INFO: Found {n_polys} polygons!')

            print(f'Label   |       Area       |   Extent')

            for label in self._poly_labels:
                area = self._poly_areas[label]
                extent = self._poly_extents[label]
                print(f'{label:<8s}|{area:^18.3f}|   {extent}')

            print('Done extracting polygons')
            print_el()

        self._set_poly_data_extrt_flag = True
        return

    def get_labels(self):

        '''Get the field values corresponding to the extracted polygons

        Returns
        -------
        _poly_labels : tuple
            Field values as strings corresponding to the extracted polygons.
        '''

        assert self._set_poly_data_extrt_flag, (
            'Call the extract_polygon_data method first!')

        assert self._poly_labels is not None, 'This should not have happend!'

        return self._poly_labels

    def get_polygons(self):

        '''Get the extracted polygons

        Returns
        -------
        _poly_geoms : dict
            A dictionary whose values are the extracted polygons and keys
            are the corresponding field values as strings that was
            specified in the set_input method i.e. label_field.
        '''

        assert self._set_poly_data_extrt_flag, (
            'Call the extract_polygon_data method first!')

        assert self._poly_geoms is not None, 'This should not have happend!'

        return self._poly_geoms

    def get_areas(self):

        '''Get the areas corresponding to the extracted polygons

        Returns
        -------
        _poly_areas : dict
            A dictionary whose values are the extracted polygons\' areas
            and keys are the corresponding field values as strings that was
            specified in the set_input method i.e. label_field.
        '''

        assert self._set_poly_data_extrt_flag, (
            'Call the extract_polygon_data method first!')

        assert self._poly_areas is not None, 'This should not have happend!'

        return self._poly_areas

    def get_extents(self):

        '''Get the extents corresponding to the extracted polygons

        Returns
        -------
        _poly_extents : dict
            A dictionary whose values are the extracted polygons\' extents
            as a tuple with the format (x_min, x_max, y_min, y_max)
            and keys are the corresponding field values as strings that was
            specified in the set_input method i.e. label_field.
        '''

        assert self._set_poly_data_extrt_flag, (
            'Call the extract_polygon_data method first!')

        assert self._poly_extents is not None, 'This should not have happend!'

        return self._poly_extents
