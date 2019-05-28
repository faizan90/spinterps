'''
Created on May 27, 2019

@author: Faizan-Uni
'''

from pathlib import Path

import numpy as np
import netCDF4 as nc

from ..misc import print_sl, print_el


class ExtractNetCDFCoords:

    _raster_type_lab = 'nc'

    def __init__(self, verbose=True):

        self._vb = verbose

        self._nc_path = None
        self._nc_x_crds_lab = None
        self._nc_y_crds_lab = None
        self._nc_x_crds = None
        self._nc_y_crds = None

        self._set_nc_props_flag = False
        self._set_nc_data_asm_flag = False
        return

    def set_netcdf_properties(
            self, path_to_nc, x_crds_lab='X', y_crds_lab='Y'):

        assert isinstance(path_to_nc, (str, Path))
        assert isinstance(x_crds_lab, str)
        assert isinstance(y_crds_lab, str)

        path_to_nc = Path(path_to_nc).absolute()

        assert path_to_nc.exists()

        self._nc_path = path_to_nc
        self._nc_x_crds_lab = x_crds_lab
        self._nc_y_crds_lab = y_crds_lab

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the netCDF:')
            print(f'Path: {self._nc_path}')
            print(f'X coordinates\' label: {self._nc_x_crds_lab}')
            print(f'Y coordinates\' label: {self._nc_y_crds_lab}')

            print_el()

        self._set_nc_props_flag = True
        return

    def assemble_netcdf_data(self):

        assert self._set_nc_props_flag

        nc_hdl = nc.Dataset(str(self._nc_path))

        assert self._nc_x_crds_lab in nc_hdl.variables
        assert self._nc_y_crds_lab in nc_hdl.variables

        self._nc_x_crds = nc_hdl[self._nc_x_crds_lab][...]
        self._nc_y_crds = nc_hdl[self._nc_y_crds_lab][...]

        assert np.all(self._nc_x_crds.shape)
        assert np.all(self._nc_y_crds.shape)

        if np.ma.is_masked(self._nc_x_crds):
            self._nc_x_crds = self._nc_x_crds.data
            self._nc_y_crds = self._nc_y_crds.data

            if self._vb:
                print_sl()

                print(
                    f'INFO: X and coordinates array were masked. '
                    f'Took the "data" attribute!')

                print_el()

        elif isinstance(self._nc_x_crds, np.ndarray):
            pass

        else:
            raise AssertionError

        assert np.all(np.isfinite(self._nc_x_crds))
        assert np.all(np.isfinite(self._nc_y_crds))

        assert len(self._nc_x_crds.shape) == len(self._nc_y_crds.shape)

        if self._vb:
            print_sl()

            print(f'INFO: netCDF coordinates\' properties:')
            print(f'Dimensions of coordinates: {len(self._nc_x_crds.shape)}')
            print(f'Shape of X coordinates: {self._nc_x_crds.shape}')
            print(f'Shape of Y coordinates: {self._nc_y_crds.shape}')

            print_el()

        nc_hdl.close()

        self._set_nc_data_asm_flag = True
        return

    def get_x_coordinates(self):

        assert self._set_nc_data_asm_flag

        return self._nc_x_crds

    def get_y_coordinates(self):

        assert self._set_nc_data_asm_flag

        return self._nc_y_crds
