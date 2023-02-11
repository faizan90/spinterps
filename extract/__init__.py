'''
Created on May 27, 2019

@author: Faizan-Uni
'''

from .poly import ExtractPolygons
from .point import ExtractPoints
from .nc import ExtractNetCDFCoords, ExtractNetCDFValues
from .gtiff import ExtractGTiffCoords, ExtractGTiffValues
from .idxs import GeomAndCrdsItsctIdxs, ReOrderIdxs


class Extract:

    '''A convenience class for using extraction modules in one place.

    See the documentation of individual classes for input and output
    specification.
    '''

    def __init__(self, verbose=True):

        assert isinstance(verbose, bool), 'verbose not a boolean!'

        self._vb = verbose

        self._poly_cls = None

        return

    def _get_poly_cls(self, path_to_shp, label_field):

        poly_cls = ExtractPolygons(verbose=self._vb)

        poly_cls.set_input(path_to_shp, label_field)

        poly_cls.extract_polygons()

        return poly_cls

    def extract_from_geotiff(
            self,
            path_to_shp,
            label_field,
            path_to_gtiff,
            path_to_output,
            minimum_cell_area_intersection_percentage,
            n_cpus,
            src_epsg=None,
            dst_epsg=None,
            simplify_tol_ratio=0.0):

        poly_cls = self._get_poly_cls(path_to_shp, label_field)

        gtiff_crds_cls = ExtractGTiffCoords(verbose=self._vb)

        gtiff_crds_cls.set_input(path_to_gtiff)

        gtiff_crds_cls.extract_coordinates()

        itsct_cls = GeomAndCrdsItsctIdxs()

        itsct_cls.set_geometries(
            poly_cls.get_polygons(), poly_cls._geom_type, simplify_tol_ratio)

        itsct_cls.set_coordinates(
            gtiff_crds_cls.get_x_coordinates(),
            gtiff_crds_cls.get_y_coordinates(),
            gtiff_crds_cls._raster_type_lab)

        itsct_cls.set_coordinate_system_transforms(src_epsg, dst_epsg)

        itsct_cls.set_intersect_misc_settings(
            minimum_cell_area_intersection_percentage,
            n_cpus)

        itsct_cls.verify()

        itsct_cls.compute_intersect_indices()

        gtiff_vals_cls = ExtractGTiffValues(verbose=self._vb)

        gtiff_vals_cls.set_input(path_to_gtiff)
        gtiff_vals_cls.set_output(path_to_output)

        gtiff_vals_cls.extract_values(itsct_cls.get_intersect_indices())

        if path_to_output is None:
            res = gtiff_vals_cls.get_values()

        else:
            res = None

        return res

    def extract_from_netCDF(
            self,
            path_to_shp,
            label_field,
            path_to_nc,
            path_to_output,
            x_crds_label,
            y_crds_label,
            variable_labels,
            time_label,
            minimum_cell_area_intersection_percentage,
            n_cpus,
            src_epsg=None,
            dst_epsg=None,
            simplify_tol_ratio=0.0):

        assert isinstance(variable_labels, (list, tuple)), (
            'variable_labels can only be a list or tuple having strings!')

        assert all([isinstance(x, str) for x in variable_labels]), (
            'variable_labels can only be a list or tuple having strings!')

        poly_cls = self._get_poly_cls(path_to_shp, label_field)

        nc_crds_cls = ExtractNetCDFCoords(verbose=self._vb)

        nc_crds_cls.set_input(path_to_nc, x_crds_label, y_crds_label)

        nc_crds_cls.extract_coordinates()

        itsct_cls = GeomAndCrdsItsctIdxs(verbose=self._vb)

        itsct_cls.set_geometries(
            poly_cls.get_polygons(), poly_cls._geom_type, simplify_tol_ratio)

        itsct_cls.set_coordinates(
            nc_crds_cls.get_x_coordinates(),
            nc_crds_cls.get_y_coordinates(),
            nc_crds_cls._raster_type_lab)

        itsct_cls.set_coordinate_system_transforms(src_epsg, dst_epsg)

        itsct_cls.set_intersect_misc_settings(
            minimum_cell_area_intersection_percentage,
            n_cpus)

        itsct_cls.verify()

        itsct_cls.compute_intersect_indices()

        ress = {}
        for variable_label in variable_labels:
            nc_vals_cls = ExtractNetCDFValues(verbose=self._vb)

            nc_vals_cls.set_input(path_to_nc, variable_label, time_label)
            nc_vals_cls.set_output(path_to_output)

            if nc_vals_cls._out_fmt == 'h5':
                nc_vals_cls.extract_values(itsct_cls.get_intersect_indices())

            elif nc_vals_cls._out_fmt == 'nc':
                nc_vals_cls.snip_values(
                    itsct_cls.get_intersect_indices(), nc_crds_cls)

            elif nc_vals_cls._out_fmt == 'raw':
                pass

            else:
                raise ValueError(
                    f'Unknown output format: {nc_vals_cls._out_fmt}!')

            if path_to_output is None:
                ress[variable_label] = nc_vals_cls.get_values()
            else:
                ress = None

        return ress
