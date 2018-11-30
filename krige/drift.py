'''
Created on Nov 25, 2018

@author: Faizan
'''

import gdal
import numpy as np
import pandas as pd


class KrigingDrift:

    def __init__(self):

        self._drft_ndv = None
        self._drft_arrs = None
        self._stns_drft_df = None

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

        self._drft_oarrs = []  #  the original ones, ndvs are not NaNs

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

    def _prepare_stns_drift(self):

        '''
        Bring the drift data to a useable form. Also, for every station,
        extract the drift values from each drift raster and put them in
        a seperate dataframe.
        '''

        self._drft_arrs = []  # the clipped ones, ndvs are set to NaNs

        krige_cols = np.arange(self._min_col, self._max_col + 1, dtype=int)
        krige_rows = np.arange(self._min_row, self._max_row + 1, dtype=int)

        assert self._nc_x_crds.shape[0] == krige_cols.shape[0]
        assert self._nc_y_crds.shape[0] == krige_rows.shape[0]

        (krige_drift_cols_mesh,
         krige_drift_rows_mesh) = np.meshgrid(krige_cols, krige_rows)

        krige_drift_cols_mesh = krige_drift_cols_mesh.ravel()
        krige_drift_rows_mesh = krige_drift_rows_mesh.ravel()

        if self._cell_sel_prms_set:
            krige_drift_cols_mesh = krige_drift_cols_mesh[self._cntn_idxs]
            krige_drift_rows_mesh = krige_drift_rows_mesh[self._cntn_idxs]

        for drift_arr in self._drft_oarrs:
            drift_vals = drift_arr[
                krige_drift_rows_mesh, krige_drift_cols_mesh]

            drift_vals[np.isclose(self._drft_ndv, drift_vals)] = np.nan

            self._drft_arrs.append(drift_vals)

        self._drft_arrs = np.array(self._drft_arrs, dtype=float)

        drift_df_cols = np.arange(len(self._drft_oarrs))

        self._stns_drft_df = pd.DataFrame(
            index=self._crds_df.index,
            columns=drift_df_cols,
            dtype=float)

        for stn in self._stns_drft_df.index:
            stn_x = self._crds_df.loc[stn, 'X']
            stn_y = self._crds_df.loc[stn, 'Y']

            stn_col = int((stn_x - self._drft_x_min) / self._cell_size)
            stn_row = int((self._drft_y_max - stn_y) / self._cell_size)

            for col, drft_oarr in zip(drift_df_cols, self._drft_oarrs):
                drft = drft_oarr[stn_row, stn_col]

                if np.isclose(self._drft_ndv, drft):
                    drft = np.nan

                self._stns_drft_df.loc[stn, col] = drft

        # TODO: check for repeating drift, that can produce singular matrix
        self._stns_drft_df.dropna(inplace=True)
        return
