'''
Created on Nov 25, 2018

@author: Faizan
'''

import gdal
import numpy as np


class KrigingDrift:

    def __init__(self):

        self._drft_asm_flag = False
        return

    def _assemble_drift_data(self):

        '''
        Given the drift rasters, check if all of them have similar properties
        such as cell size, coordinates' minima and maxima and no data values.

        If the alignment raster is specified then the cell size of the
        drift rasters should match that of the alignment raster.

        Manually specifed cell size is ignored and the cell size of the
        drift rasters is taken if no alignment raster is specified.
        '''

        self._drft_oarrs = []

        check_valss = [
            [],  # x_mins
            [],  # y_maxs
            [],  # n_rows,
            [],  # n_cols,
            [],  # ndvs
            [],  # cell_widths
            [],  # cell_heights
            ]

        if self._vb:
            print('\n', '#' * 10, sep='')
            print('Assembling drift data...')

        for drift_raster in self._drft_rass:
            drift_ds = gdal.Open(str(drift_raster))

            if self._vb:
                print('Going through:', drift_raster.name)

            assert drift_ds is not None, 'Could not read drift raster!'

            drift_gt = drift_ds.GetGeoTransform()

            drift_x_min = drift_gt[0]
            drift_y_max = drift_gt[3]

            cell_width = drift_gt[1]
            cell_height = abs(drift_gt[5])

            assert np.isclose(cell_width, self._cell_size), (
                f'Drift raster\'s cell width {cell_width} unequal to '
                f'the one used {self._cell_size}!')

            assert np.isclose(cell_height, self._cell_size), (
                f'Drift raster\'s cell height {cell_height} unequal to '
                f'the one used {self._cell_size}!')

            drift_rows = drift_ds.RasterYSize
            drift_cols = drift_ds.RasterXSize

            drift_band = drift_ds.GetRasterBand(1)
            drift_arr = drift_band.ReadAsArray()

            self._drft_oarrs.append(drift_arr)

            drift_ndv = drift_band.GetNoDataValue()

            if self._vb:
                print('x_min:', drift_x_min)
                print('y_max:', drift_y_max)
                print('cell_width and height:', cell_width)
                print('No. rows:', drift_rows)
                print('No. columns:', drift_cols)
                print('NDV:', drift_ndv)
                print('\n')

            check_valss[0].append(drift_x_min)
            check_valss[1].append(drift_y_max)
            check_valss[2].append(drift_rows)
            check_valss[3].append(drift_cols)
            check_valss[4].append(drift_ndv)
            check_valss[5].append(cell_width)
            check_valss[6].append(cell_height)

        for check_vals in check_valss:
            assert np.all(np.isclose(check_vals, check_vals[0]))

        self._drft_x_min = check_valss[0][0]
        self._drft_y_max = check_valss[1][0]

        self._drft_x_max = self._drft_x_min + (
            check_valss[3][0] * self._cell_size)

        self._drft_y_min = self._drft_y_max - (
            check_valss[2][0] * self._cell_size)

        self._drft_ndv = check_valss[4][0]

        self._drft_ras_props = tuple(check_valss)  # just in case

        if self._vb:
            print('Drift rasters assembled without problems.')
            print('#' * 10)

        self._drft_asm_flag = True
        return
