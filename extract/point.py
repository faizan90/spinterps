'''
Created on Jun 12, 2019

@author: Faizan-Uni
'''

from pathlib import Path

from osgeo import ogr

from ..misc import print_sl, print_el


class ExtractPoints:

    '''Get points and their field values from an ESRI Shapefile'''

    _geom_type = 1

    def __init__(self, verbose=True):

        self._vb = verbose

        self._point_shp_path = None
        self._point_label_field = None

        self._point_labels = None
        self._point_geoms = None
        self._point_x_crds = None
        self._point_y_crds = None

        self._set_in_flag = False
        self._set_point_data_extrt_flag = False
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

            print('Setting points\' extraction input...')

        assert isinstance(path_to_shp, (str, Path)), (
            f'Specified path to input ({path_to_shp}) is not a string or '
            f'path-like object!')

        path_to_shp = Path(path_to_shp).absolute()

        assert path_to_shp.exists(), (
            f'Specified input file ({path_to_shp}) does not exist!')

        assert isinstance(label_field, str), 'label_field not a string!'
        assert label_field, 'label_field is an empty string!'

        self._point_shp_path = path_to_shp
        self._point_label_field = label_field

        if self._vb:
            print(f'INFO: Set the following point shapefile variables:')
            print(f'Path: {self._point_shp_path}')
            print(f'Label field: {self._point_label_field}')

            print_el()

        self._set_in_flag = True
        return

    def extract_points(self):

        '''Run the point extraction algorithm'''

        if self._vb:
            print_sl()

            print('Extracting points...')

        assert self._set_in_flag, 'Call the set_input method first!'

        shp_hdl = ogr.Open(str(self._point_shp_path))

        assert shp_hdl is not None, (
            f'Unable to open the input file: {self._point_shp_path} as an '
            f'ogr vector file!')

        assert shp_hdl.GetDriver().GetName() == 'ESRI Shapefile', (
            'Specified input vector file is not an ESRI Shapefile!')

        assert shp_hdl.GetLayerCount() == 1, (
            'Specified input vector file only allowed to have one layer!')

        point_lyr = shp_hdl.GetLayer(0)

        assert point_lyr.GetFeatureCount() > 0, (
            'Zero vectors inside the input shapefile!')

        labels = []
        geoms = {}
        x_crds = {}
        y_crds = {}

        point = point_lyr.GetNextFeature()

        assert point.GetFieldCount() > 0, (
            'Zero fields inside the input shapefile!')

        assert self._point_label_field in point.keys(), (
            f'label_field: {self._point_label_field} not in input shapefile!')

        while point:
            geom = point.GetGeometryRef().Clone()

            assert geom is not None, 'Could not read/clone a geometry!'

            geom_type = geom.GetGeometryType()
            geom_name = geom.GetGeometryName()

            assert geom_type == 1, (
                f'Unsupported geometry type, name: {geom_type}, {geom_name}!')

            assert geom.GetGeometryCount() == 0, (
                'Only one point allowed per feature!')

            label = point.GetFieldAsString(self._point_label_field)

            assert label is not None, (
                f'Could not read the field: {label} as a string!')

            labels.append(label)
            geoms[label] = geom

            x_crds[label] = geom.GetX()
            y_crds[label] = geom.GetY()

            point = point_lyr.GetNextFeature()

        shp_hdl.Destroy()
        shp_hdl = None

        n_points = len(labels)

        assert n_points, 'This should not have happend!'

        labels_set = set(labels)

        assert n_points == len(labels_set), (
            'Non-unique field values of points!')

        assert n_points == len(geoms), 'This should not have happend!'
        assert n_points == len(x_crds), 'This should not have happend!'
        assert n_points == len(y_crds), 'This should not have happend!'

        self._point_labels = tuple(labels)
        self._point_geoms = geoms
        self._point_x_crds = x_crds
        self._point_y_crds = y_crds

        if self._vb:
            print(f'INFO: Found {n_points} points!')

            print(f'Label   |        X         |       Y      ')

            for label in self._point_labels:
                print(
                    f'{label:<8s}|'
                    f'{x_crds[label]:^18.3f}|'
                    f'{y_crds[label]:^18.3f}')

            print('Done extracting points')
            print_el()

        self._set_point_data_extrt_flag = True
        return

    def get_labels(self):

        '''Get the field values corresponding to the extracted points

        Returns
        -------
        _point_labels : tuple
            Field values as strings corresponding to the extracted points.
        '''

        assert self._set_point_data_extrt_flag, (
            'Call the extract_point_data method first!')

        assert self._point_labels is not None, 'This should not have happend!'

        return self._point_labels

    def get_points(self):

        '''Get the extracted points

        Returns
        -------
        _point_geoms : dict
            A dictionary whose values are the extracted points and keys
            are the corresponding field values as strings that was
            specified in the set_input method i.e. label_field.
        '''

        assert self._set_point_data_extrt_flag, (
            'Call the extract_point_data method first!')

        assert self._point_geoms is not None, 'This should not have happend!'

        return self._point_geoms

    def get_x_coordinates(self):

        '''Get the X coordinates corresponding to the extracted points

        Returns
        -------
        _point_x_crds : dict
            A dictionary whose values are the extracted points\' coordinates
            and keys are the corresponding field values as strings that was
            specified in the set_input method i.e. label_field.
        '''

        assert self._set_point_data_extrt_flag, (
            'Call the extract_point_data method first!')

        assert self._point_x_crds is not None, 'This should not have happend!'

        return self._point_x_crds

    def get_y_coordinates(self):

        '''Get the Y coordinates corresponding to the extracted points

        Returns
        -------
        _point_y_crds : dict
            A dictionary whose values are the extracted points\' coordinates
            and keys are the corresponding field values as strings that was
            specified in the set_input method i.e. label_field.
        '''

        assert self._set_point_data_extrt_flag, (
            'Call the extract_point_data method first!')

        assert self._point_y_crds is not None, 'This should not have happend!'

        return self._point_y_crds
