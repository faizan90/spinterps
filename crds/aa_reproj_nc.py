# -*- coding: utf-8 -*-

'''
Created on Feb 9, 2023

@author: Faizan3800X-Uni
'''

from pathlib import Path

import pyproj
import numpy as np
import netCDF4 as nc


class CrdsReProjNC:

    '''
    Create new coordinates inside a given netCDF file.
    '''

    def __init__(self, verbose):

        assert isinstance(verbose, bool), type(verbose)

        self._vb = verbose
        return

    def append_crds(
            self,
            path_to_nc,
            src_crs,
            dst_crs,
            src_x_lab,
            src_y_lab,
            dst_x_lab,
            dst_y_lab,
            dim_x_lab,
            dim_y_lab):

        assert isinstance(path_to_nc, Path), type(path_to_nc)
        assert path_to_nc.exists(), 'File path_to_nc does not exist!'

        with nc.Dataset(path_to_nc, 'a') as nc_hdl:

            assert src_x_lab in nc_hdl.variables, (
                f'Variable {src_x_lab} does not exist!')

            assert src_y_lab in nc_hdl.variables, (
                f'Variable {src_y_lab} does not exist!')

            assert dim_x_lab in nc_hdl.dimensions, (
                f'Dimensions {dim_x_lab} does not exist!')

            assert dim_y_lab in nc_hdl.dimensions, (
                f'Dimensions {dim_y_lab} does not exist!')

            tfmr = pyproj.Transformer.from_crs(
                src_crs, dst_crs, always_xy=True)

            src_xs = nc_hdl[src_x_lab][:]
            src_ys = nc_hdl[src_y_lab][:]

            assert np.all(np.isfinite(src_xs)), (
                'Invalid values in source X coordinates!')

            assert np.all(np.isfinite(src_ys)), (
                'Invalid values in source Y coordinates!')

            assert src_xs.ndim == src_ys.ndim, (
                'X ({src_xs.ndim}) and Y ({src_ys.ndim}) coordinates should '
                'have the same nuber of dimensions!')

            if (src_xs.ndim == 1) and (src_ys.ndim == 1):

                src_xs, src_ys = np.meshgrid(src_xs, src_ys)

            elif (src_xs.ndim == 2) and (src_ys.ndim == 2):

                pass

            else:
                raise ValueError(
                    f'Source coordinates can only have 1 or 2 dimensions '
                    f'and not {(src_xs.ndim, src_ys.ndim)}!')

            dst_xs, dst_ys = tfmr.transform(src_xs, src_ys)

            assert np.all(np.isfinite(dst_xs)), (
                'Invalid values in destination X coordinates!')

            assert np.all(np.isfinite(dst_ys)), (
                'Invalid values in destination Y coordinates!')

            if dst_x_lab not in nc_hdl.variables:
                dst_x_nc = nc_hdl.createVariable(
                    dst_x_lab,
                    'd',
                    dimensions=(dim_y_lab, dim_x_lab))

            else:
                dst_x_nc = nc_hdl[dst_x_lab]

                assert dst_x_nc.shape == src_xs.shape, (
                    'Existing destination X coordinates cannot be '
                    'overwritten as they have differtent shapes: '
                    f'({dst_x_nc.shape, src_xs.shape})!')

            if dst_y_lab not in nc_hdl.variables:
                dst_y_nc = nc_hdl.createVariable(
                    dst_y_lab,
                    'd',
                    dimensions=(dim_y_lab, dim_x_lab))

            else:
                dst_y_nc = nc_hdl[dst_y_lab]

                assert dst_y_nc.shape == src_ys.shape, (
                    'Existing destination Y coordinates cannot be '
                    'overwritten as they have differtent shapes: '
                    f'({dst_y_nc.shape, src_ys.shape})!')

            dst_x_nc[:] = dst_xs
            dst_y_nc[:] = dst_ys

        return
