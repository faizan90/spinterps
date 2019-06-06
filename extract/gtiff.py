'''
Created on May 27, 2019

@author: Faizan-Uni
'''

from pathlib import Path

import gdal
import h5py
import numpy as np

from ..misc import print_sl, print_el


class ExtractGTiffCoords:

    _raster_type_lab = 'gtiff'

    def __init__(self, verbose=True):

        self._vb = verbose

        self._in_path = None
        self._x_crds = None
        self._y_crds = None

        self._set_in_flag = False
        return

    def set_input(self, path_to_gtiff):

        assert isinstance(path_to_gtiff, (str, Path)), (
            f'Specified path to input ({path_to_gtiff}) is not a string or '
            f'path-like object!')

        path_to_gtiff = Path(path_to_gtiff).absolute()

        assert path_to_gtiff.exists(), (
            f'Specified input file ({path_to_gtiff}) does not exist!')

        self._in_path = path_to_gtiff

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the input:')
            print(f'Path: {self._in_path}')

            print_el()

        self._set_in_flag = True
        return

    def extract_coordinates(self):

        assert self._set_in_flag, (
            f'Call the set_input method first!')

        in_hdl = gdal.Open(str(self._in_path))

        assert in_hdl is not None, (
            f'Could not open the input ({self._in_path}) as a '
            f'GDAL raster for reading!')

        in_drv_shrt_name = in_hdl.GetDriver().ShortName

        assert in_drv_shrt_name == 'GTiff', (
            f'Input not a GDAL GeoTiff raster rather {in_drv_shrt_name}!')

        n_rows = in_hdl.RasterYSize
        n_cols = in_hdl.RasterXSize

        assert n_rows, 'The input raster cannot have zero rows!'
        assert n_cols, 'The input raster cannot have zero columns!'

        geotransform = in_hdl.GetGeoTransform()

        in_hdl = None

        x_min = geotransform[0]
        y_max = geotransform[3]

        pix_width = abs(geotransform[1])
        pix_height = abs(geotransform[5])

        assert pix_width > 0, 'Raster cell width cannot be zero or less!'
        assert pix_height > 0, 'Raster cell height cannot be zero or less!'

        x_max = x_min + (n_cols * pix_width)
        y_min = y_max - (n_rows * pix_height)

        assert x_min < x_max, 'Incorrect order of x_min and x_max!'
        assert y_min < y_max, 'Incorrect order of y_min and y_max!'

        assert np.all(np.isfinite([x_min, x_max, y_min, y_max])), (
            f'Input raster bounds {(x_min, x_max, y_min, y_max)} '
            f'are invalid!')

        self._x_crds = np.linspace(x_min, x_max, n_cols + 1)
        self._y_crds = np.linspace(y_max, y_min, n_rows + 1)

        assert self._x_crds.shape[0] == (n_cols + 1), (
            'Number of columns is not as required!')

        assert self._y_crds.shape[0] == (n_rows + 1), (
            'Number of rows is not as required!')

        assert np.all(np.isfinite(self._x_crds)), (
            'X coordinates are not finite!')

        assert np.all(np.isfinite(self._y_crds)), (
            'Y coordinates are not finite!')

        assert self._x_crds.ndim == self._y_crds.ndim, (
            'Unequal dimensions of X and Y coordinates!')

        assert (
            np.all(np.ediff1d(self._x_crds) > 0) or
            np.all(np.ediff1d(self._x_crds[::-1]) > 0)), (
                'X coordinates not monotonically increasing or decreasing!')

        assert (
            np.all(np.ediff1d(self._y_crds) > 0) or
            np.all(np.ediff1d(self._y_crds[::-1]) > 0)), (
                'Y coordinates not monotonically increasing or decreasing!')

        if self._vb:
            print_sl()

            print(f'INFO: GTiff coordinates\' properties:')
            print(f'Shape of X coordinates: {self._x_crds.shape}')
            print(f'Shape of Y coordinates: {self._y_crds.shape}')

            print_el()

        self._set_crds_extrt_flag = True
        return

    def get_x_coordinates(self):

        assert self._set_crds_extrt_flag, (
            'Call the extract_coordinates method first!')

        return self._x_crds

    def get_y_coordinates(self):

        assert self._set_crds_extrt_flag, (
            'Call the extract_coordinates method first!')

        return self._y_crds


class ExtractGTiffValues:

    def __init__(self, verbose=True):

        self._vb = verbose

        self._in_path = None
        self._extrtd_data = None

        self._out_path = None
        self._out_fmt = None

        self._set_in_flag = False
        self._set_out_flag = False
        self._set_data_extrt_flag = False
        return

    def set_input(self, path_to_gtiff):

        assert isinstance(path_to_gtiff, (str, Path)), (
            f'Specified path to input {path_to_gtiff} is not a string or '
            f'path-like object!')

        path_to_gtiff = Path(path_to_gtiff).absolute()

        assert path_to_gtiff.exists(), (
            f'Specified input file ({path_to_gtiff}) does not exist!')

        self._in_path = path_to_gtiff

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the input:')
            print(f'Path: {self._in_path}')

            print_el()

        self._set_in_flag = True
        return

    def set_output(self, path_to_output=None):

        if path_to_output is None:
            self._out_fmt = 'raw'

        else:
            assert isinstance(path_to_output, (str, Path)), (
                f'Specified path to output ({path_to_output}) is not '
                f'a string or path-like object!')

            path_to_output = Path(path_to_output).absolute()

            assert path_to_output.parents[0].exists(), (
                f'Parent directory of the output file does not exist!')

            fmt = path_to_output.suffix

            if fmt in ('.h5', '.hdf5'):
                self._out_fmt = 'h5'

            else:
                raise NotImplementedError(
                    'Only configured for file extensions ending in '
                    'h5 and hdf5 only!')

        self._out_path = path_to_output

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the output:')
            print(f'Path: {self._out_path}')
            print(f'Format: {self._out_fmt}')

            print_el()

        self._set_out_flag = True
        return

    def _verf_idxs(self, indicies, save_add_vars_flag):

        add_var_labels_main = set()

        for crds_idxs in indicies.values():

            assert 'cols' in crds_idxs, (
                f'\'cols\' is not in the indices dictionary!')

            cols_idxs = crds_idxs['cols']

            assert cols_idxs.ndim == 1, (
                'Column indices allowed to be 1D only!')

            assert cols_idxs.size > 0, (
                'Column indices must have at least one value!')

            assert np.issubdtype(cols_idxs.dtype, np.integer), (
                'Column indices array values not of integer type!')

            assert 'rows' in crds_idxs, (
                f'\'rows\' is not in the indices dictionary!')

            rows_idxs = crds_idxs['rows']
            assert rows_idxs.ndim == 1, (
                'Row indices allowed to be 1D only!')

            assert rows_idxs.size > 0, (
                'Row indices must have at least one value!')

            assert np.issubdtype(rows_idxs.dtype, np.integer), (
                'Row indices array values not of integer type!')

            assert cols_idxs.shape == rows_idxs.shape, (
                'Unequal number of row and column indices!')

            if not save_add_vars_flag:
                continue

            if not add_var_labels_main:
                add_var_labels_main = set(
                    crds_idxs.keys()) - set(('rows', 'cols'))

            else:
                # "of values" is supposed to be twice :)
                assert not (
                    add_var_labels_main -
                    set(crds_idxs.keys()) -
                    set(('rows', 'cols'))), (
                        'Non-matching keys of values of values!')

        return add_var_labels_main

    def extract_data_for_indices_and_save(
            self, indicies, save_add_vars_flag=True):

        assert self._set_in_flag, 'Call the set_input method first!'
        assert self._set_out_flag, 'Call the set_ouput method first!'

        assert isinstance(indicies, dict), 'indices not a dictionary!'
        assert indicies, 'Empty indices dictionary!'

        assert isinstance(save_add_vars_flag, bool), (
            'save_add_vars_flag not a boolean!')

        add_var_labels = self._verf_idxs(indicies, save_add_vars_flag)

        in_hdl = gdal.Open(str(self._in_path))

        assert in_hdl is not None, (
            f'Could not open the input file ({self._in_path}) as a '
            f'GDAL raster for reading!')

        in_drv_shrt_name = in_hdl.GetDriver().ShortName

        assert in_drv_shrt_name == 'GTiff', (
            f'Input not a GDAL GeoTiff raster rather {in_drv_shrt_name}!')

        n_bnds = in_hdl.RasterCount

        assert n_bnds > 0, 'Input raster must have at least one band!'

        gtiff_bnds = {}
        for i in range(1, n_bnds + 1):
            bnd = in_hdl.GetRasterBand(i)

            ndv = bnd.GetNoDataValue()

            data = bnd.ReadAsArray().astype(float)

            assert data.ndim == 2, 'Raster bands allowed to be 2D only!'
            assert data.size > 0, 'Raster band has zero values!'

            if not np.isnan(ndv):
                data[np.isclose(data, ndv)] = np.nan

            assert np.any(np.isfinite(data)), (
                f'All values in band {i} are invalid!')

            gtiff_bnds[f'B{i:02d}'] = data

        in_hdl = None

        assert gtiff_bnds, 'This should not have happend!'

        if self._vb:
            print_sl()

            print(f'INFO: Read {n_bnds} from the input raster')

            print_el()

        path_stem = self._in_path.stem
        assert path_stem, 'Input file has no name?'

        misc_grp_labs = add_var_labels | set(('rows', 'cols'))

        extrtd_data = {}

        if self._out_fmt == 'h5':
            out_hdl = h5py.File(str(self._out_path), mode='a', driver=None)

            if 'rows' not in out_hdl:
                for grp in misc_grp_labs:
                    assert grp not in out_hdl, (
                        f'Variable {grp} is not supposed to exist in the '
                        f'output file!')

                    out_hdl.create_group(grp)

            else:
                for grp in misc_grp_labs:
                    assert grp in out_hdl, (
                        f'Variable {grp} was supposed to exist in the '
                        f'output file!')

            assert path_stem not in out_hdl, (
                f'Specified output variable {path_stem} exists in the output '
                f'already!')

            out_var_grp = out_hdl.create_group(path_stem)

        elif self._out_fmt == 'raw':
            pass

        else:
            raise NotImplementedError

        for label, crds_idxs in indicies.items():

            cols_idxs = crds_idxs['cols']
            cols_idxs_min = cols_idxs.min()
            cols_idxs_max = cols_idxs.max()
            assert (cols_idxs_max >= 0) & (cols_idxs_min >= 0), (
                'Column indices are not allowed to be negative!')

            rows_idxs = crds_idxs['rows']
            rows_idxs_min = rows_idxs.min()
            rows_idxs_max = rows_idxs.max()
            assert (rows_idxs_max >= 0) & (rows_idxs_min >= 0), (
                'Row indices are not allowed to be negative!')

            crds_set = set([(x, y) for x, y in zip(cols_idxs, rows_idxs)])

            assert len(crds_set) == cols_idxs.size, (
                'Repeating row and column index combinations not allowed!')

            bnds_data = {}
            for bnd, data in gtiff_bnds.items():
                assert rows_idxs_max < data.shape[0], (
                    'Row index is out of bounds!')

                assert cols_idxs_max < data.shape[1], (
                    'Column index is out of bounds!')

                bnd_data = data[rows_idxs, cols_idxs]

                bnds_data[bnd] = bnd_data

            assert bnds_data, 'This should not have happend!'

            extrtd_data[label] = bnds_data

            if self._out_fmt == 'raw':
                pass

            elif self._out_fmt == 'h5':
                label_str = str(label)

                assert label_str not in out_var_grp, (
                    f'Dataset {label_str} exists in the output file already!')

                for add_var_lab in add_var_labels:
                    grp_lnk = f'{add_var_lab}/{label_str}'

                    if label_str in out_hdl[add_var_lab]:
                        assert np.all(np.isclose(
                            out_hdl[grp_lnk][...],
                            crds_idxs[add_var_lab])), (
                                f'Existing values at {grp_lnk} in the HDF5 '
                                f'and current ones are unequal!')

                    else:
                        out_hdl[grp_lnk] = crds_idxs[add_var_lab]

                if label_str in out_hdl['rows']:
                    assert label_str in out_hdl['cols'], (
                        f'Dataset {label_str} supposed to exist in '
                        f'the \'cols\' group!')

                    assert (out_hdl[f'rows/{label_str}'].shape ==
                        rows_idxs.shape), (
                            'Shape of existing row indices inside the '
                            'HDF5 and current ones is unequal!')

                    assert np.all(
                        out_hdl[f'rows/{label_str}'][...] == rows_idxs), (
                            f'Existing values of rows in the HDF5 '
                            f'and the current ones are unequal!')

                    assert (out_hdl[f'cols/{label_str}'].shape ==
                        cols_idxs.shape), (
                            'Shape of existing column indices inside the '
                            'HDF5 and current ones is unequal!')

                    assert np.all(
                        out_hdl[f'cols/{label_str}'][...] == cols_idxs), (
                            f'Existing values of columns in the HDF5 '
                            f'and the current ones are unequal!')

                else:
                    out_hdl[f'rows/{label_str}'] = rows_idxs
                    out_hdl[f'cols/{label_str}'] = cols_idxs

                out_var_grp[label_str] = np.vstack(list(bnds_data.values()))

                out_hdl.flush()

            else:
                raise NotImplementedError

        gtiff_bnds = None

        if self._out_fmt == 'h5':
            out_hdl.flush()
            out_hdl.close()
            out_hdl = None

        elif self._out_fmt == 'raw':
            assert extrtd_data, 'This should not have happend!'
            self._extrtd_data = extrtd_data

        else:
            raise NotImplementedError

        self._set_data_extrt_flag = True
        return

    def get_extracted_data(self):

        assert self._out_fmt == 'raw', (
            'Call to this function allowed only when path_to_output is None!')

        assert self._set_data_extrt_flag, (
            'Call the extract_data_for_indices_and_save method first!')

        assert self._extrtd_data is not None, 'This should not have happend!'

        return self._extrtd_data
