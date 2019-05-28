'''
Created on May 27, 2019

@author: Faizan-Uni
'''

from pathlib import Path

import gdal
import numpy as np

from ..misc import print_sl, print_el


class ExtractGTiffCoords:

    _raster_type_lab = 'gtiff'

    def __init__(self, verbose=True):

        self._vb = verbose

        self._gtiff_path = None
        self._gtiff_x_crds = None
        self._gtiff_y_crds = None

        self._set_gtiff_path_flag = False
        self._set_gtiff_data_asm_flag = False
        return

    def set_path_to_gtiff(self, path_to_gtiff):

        assert isinstance(path_to_gtiff, (str, Path))

        path_to_gtiff = Path(path_to_gtiff).absolute()

        assert path_to_gtiff.exists()

        self._gtiff_path = path_to_gtiff

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the GTiff:')
            print(f'Path: {self._gtiff_path}')

            print_el()

        self._set_gtiff_path_flag = True
        return

    def assemble_gtiff_data(self):

        assert self._set_gtiff_path_flag

        gtiff_hdl = gdal.Open(str(self._gtiff_path))

        assert gtiff_hdl is not None

        assert gtiff_hdl.GetDriver().ShortName == 'GTiff'

        n_rows = gtiff_hdl.RasterYSize
        n_cols = gtiff_hdl.RasterXSize

        assert n_rows
        assert n_cols

        geotransform = gtiff_hdl.GetGeoTransform()

        gtiff_hdl = None

        x_min = geotransform[0]
        y_max = geotransform[3]

        pix_width = geotransform[1]
        pix_height = abs(geotransform[5])

        assert pix_width > 0
        assert pix_height > 0

        x_max = x_min + (n_cols * pix_width)
        y_min = y_max - (n_rows * pix_height)

        assert x_min < x_max
        assert y_min < y_max

        assert np.all(np.isfinite([x_min, x_max, y_min, y_max]))

        self._gtiff_x_crds = np.linspace(x_min, x_max, n_cols + 1)
        self._gtiff_y_crds = np.linspace(y_min, y_max, n_rows + 1)

        assert self._gtiff_x_crds.shape[0] == (n_cols + 1)
        assert self._gtiff_y_crds.shape[0] == (n_rows + 1)

        assert np.all(np.isfinite(self._gtiff_x_crds))
        assert np.all(np.isfinite(self._gtiff_y_crds))

        if self._vb:
            print_sl()

            print(f'INFO: GTiff coordinates\' properties:')
            print(f'Shape of X coordinates: {self._gtiff_x_crds.shape}')
            print(f'Shape of Y coordinates: {self._gtiff_y_crds.shape}')

            print_el()

        self._set_gtiff_data_asm_flag = True
        return

    def get_x_coordinates(self):

        assert self._set_gtiff_data_asm_flag

        return self._gtiff_x_crds

    def get_y_coordinates(self):

        assert self._set_gtiff_data_asm_flag

        return self._gtiff_y_crds
