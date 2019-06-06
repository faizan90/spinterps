'''
Created on May 27, 2019

@author: Faizan-Uni
'''

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import netCDF4 as nc

from ..misc import print_sl, print_el, num2date


class ExtractNetCDFCoords:

    _raster_type_lab = 'nc'

    def __init__(self, verbose=True):

        self._vb = verbose

        self._in_path = None
        self._x_crds_lab = None
        self._y_crds_lab = None
        self._x_crds = None
        self._y_crds = None

        self._set_in_flag = False
        self._set_crds_extrt_flag = False
        return

    def set_input(self, path_to_nc, x_crds_lab, y_crds_lab):

        assert isinstance(path_to_nc, (str, Path)), (
            f'Specified path to input ({path_to_nc}) is not a string or '
            f'path-like object!')

        assert isinstance(x_crds_lab, str), 'x_crds_lab not a string!'
        assert x_crds_lab, 'x_crds_lab is an empty string!'

        assert isinstance(y_crds_lab, str), 'y_crds_lab not a string!'
        assert y_crds_lab, 'y_crds_lab is an empty string!'

        path_to_nc = Path(path_to_nc).absolute()

        assert path_to_nc.exists(), (
            f'Specified input file ({path_to_nc}) does not exist!')

        self._in_path = path_to_nc
        self._x_crds_lab = x_crds_lab
        self._y_crds_lab = y_crds_lab

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the netCDF:')
            print(f'Path: {self._in_path}')
            print(f'X coordinates\' label: {self._x_crds_lab}')
            print(f'Y coordinates\' label: {self._y_crds_lab}')

            print_el()

        self._set_in_flag = True
        return

    def extract_coordinates(self):

        assert self._set_in_flag, 'Call the set_input method first!'

        in_hdl = nc.Dataset(str(self._in_path))

        assert self._x_crds_lab in in_hdl.variables, (
            f'X coordinates variable: {self._x_crds_lab} does not exist '
            f'in the input!')

        assert self._y_crds_lab in in_hdl.variables, (
            f'Y coordinates variable: {self._y_crds_lab} does not exist '
            f'in the input!')

        self._x_crds = in_hdl[self._x_crds_lab][...]
        self._y_crds = in_hdl[self._y_crds_lab][...]

        assert np.all(self._x_crds.shape), (
            'Shape of X coordinates not allowed to have a zero!')

        assert np.all(self._y_crds.shape), (
            'Shape of Y coordinates not allowed to have a zero!')

        if isinstance(self._x_crds, np.ma.MaskedArray):
            self._x_crds = self._x_crds.data
            self._y_crds = self._y_crds.data

            if self._vb:
                print_sl()

                print(
                    f'INFO: X and Y coordinates array were masked. '
                    f'Took the "data" attribute!')

                print_el()

        elif (isinstance(self._x_crds, np.ndarray) and
              isinstance(self._y_crds, np.ndarray)):
            pass

        else:
            raise NotImplementedError(
                'X and Y coordinates arrays allowed to be np.ndarray or '
                'np.ma.MaskedArray instances only!')

        assert np.all(np.isfinite(self._x_crds)), (
            'Invalid values in X coordinates!')

        assert np.all(np.isfinite(self._y_crds)), (
            'Invalid values in Y coordinates!')

        assert self._x_crds.ndim == self._y_crds.ndim, (
            'Unequal dimensions of X and Y coordinates!')

        if self._vb:
            print_sl()

            print(f'INFO: netCDF coordinates\' properties:')
            print(f'Dimensions of coordinates: {self._x_crds.ndim}')
            print(f'Shape of X coordinates: {self._x_crds.shape}')
            print(f'Shape of Y coordinates: {self._y_crds.shape}')

            print_el()

        in_hdl.close()
        in_hdl = None

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


class ExtractNetCDFValues:

    def __init__(self, verbose=True):

        self._vb = verbose

        self._in_path = None
        self._in_var_lab = None
        self._in_time_lab = None

        self._out_path = None
        self._out_fmt = None

        self._time_strs_fmt = '%Y%m%dT%H%M%S'

        self._extrtd_data = None

        self._set_in_flag = False
        self._set_out_flag = False
        self._set_data_extrt_flag = False
        return

    def set_input(self, path_to_nc, variable_label, time_label):

        assert isinstance(path_to_nc, (str, Path)), (
            f'Specified path to input ({path_to_nc}) is not a string or '
            f'path-like object!')

        assert isinstance(variable_label, str), 'variable_label not a string!'
        assert variable_label, 'variable_label is an empty string!'

        assert isinstance(time_label, str), 'time_label not a string!'
        assert time_label, 'time_label is an empty string!'

        path_to_nc = Path(path_to_nc).absolute()

        assert path_to_nc.exists(), (
            f'Specified input file ({path_to_nc}) does not exist!')

        self._in_path = path_to_nc
        self._in_var_lab = variable_label
        self._in_time_lab = time_label

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the netCDF:')
            print(f'Path: {self._in_path}')
            print(f'Extraction variable label: {self._in_var_lab}')
            print(f'Time label: {self._in_time_lab}')

            print_el()

        self._set_in_flag = True
        return

    def set_output(self, path_to_output=None):

        out_fmt = None

        if path_to_output is None:
            out_fmt = 'raw'

        else:
            assert isinstance(path_to_output, (str, Path)), (
                f'Specified path to output ({path_to_output}) is not '
                f'a string or path-like object!')

            path_to_output = Path(path_to_output).absolute()

            assert path_to_output.parents[0].exists(), (
                f'Parent directory of the output file does not exist!')

            fmt = path_to_output.suffix

            if fmt in ('.h5', '.hdf5'):
                out_fmt = 'h5'

            else:
                raise NotImplementedError(
                    'Only configured for output file extensions of '
                    'h5 and hdf5 only!')

        assert out_fmt is not None

        self._out_fmt = out_fmt

        self._out_path = path_to_output

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the output:')
            print(f'Path: {self._out_path}')
            print(f'Format: {self._out_fmt}')

            print_el()

        self._set_out_flag = True
        return

    def extract_data_for_indices_and_save(
            self, indicies, save_add_vars_flag=True):

        assert self._set_in_flag, 'Call the set_input method first!'
        assert self._set_out_flag, 'Call the set_ouput method first!'

        assert isinstance(indicies, dict), 'indices not a dictionary!'
        assert indicies, 'Empty indices dictionary!'

        assert isinstance(save_add_vars_flag, bool), (
            'save_add_vars_flag not a boolean!')

        add_var_labels = self._verf_idxs(indicies, save_add_vars_flag)

        in_hdl = nc.Dataset(str(self._in_path))

        assert self._in_var_lab in in_hdl.variables, (
            f'Extraction variable: {self._in_var_lab} not in the input!')

        # memory manage?
        in_var = in_hdl[self._in_var_lab]
        in_var.set_always_mask(False)

        assert in_var.ndim == 3, (
            f'Variable: {self._in_var_lab} in the input must have '
            f'3 dimensions!')

        assert in_var.size > 0, (
            f'Zero values in the extraction variable: {self._in_var_lab}!')

        assert all(in_var.shape), (
            f'Shape of the extraction variable: {self._in_var_lab} not '
            f'allowed to have a zero!')

        assert np.issubdtype(in_var.dtype, np.number), (
            f'Data type of the extraction variable: {self._in_var_lab} not a '
            f'subtype of np.number!')

        assert self._in_time_lab in in_hdl.variables, (
            f'Time variable: {self._in_time_lab} not in the input!')

        in_time = in_hdl[self._in_time_lab]
        in_time.set_always_mask(False)

        assert np.issubdtype(in_time.dtype, np.number), (
            f'Data type of the input time variable: {self._in_time_lab} '
            f'not a subtype of np.number!')

        assert in_time.ndim == 1, (
            f'Input time variable: {self._in_time_lab} can only be 1D!')

        assert in_time.size > 0, (
            f'Zero values in the input time variable: {self._in_time_lab}!')

        assert in_time.shape[0] == in_var.shape[0], (
            f'Unequal first axis lengths of the input variables: '
            f'{self._in_var_lab} and {self._in_time_lab}!')

        if hasattr(in_time, 'units') and hasattr(in_time, 'calendar'):
            in_time_strs = pd.DatetimeIndex(num2date(
                in_time[...],
                in_time.units,
                in_time.calendar)).strftime(self._time_strs_fmt)

            h5_str_dt = h5py.special_dtype(vlen=str)

        else:
            in_time_strs = None

        path_stem = self._in_path.stem
        assert path_stem, 'Input file has no name?'

        misc_grp_labs = add_var_labels | set(('rows', 'cols', path_stem))

        if self._out_fmt == 'h5':
            out_hdl = h5py.File(str(self._out_path), mode='a', driver=None)

            if self._in_time_lab not in out_hdl:
                out_time_grp = out_hdl.create_group(self._in_time_lab)
                out_time_grp[self._in_time_lab] = in_time[...]

                if hasattr(in_time, 'units'):
                    out_time_grp.attrs['units'] = in_time.units

                if hasattr(in_time, 'calendar'):
                    out_time_grp.attrs['calendar'] = in_time.calendar

                for grp in misc_grp_labs:
                    assert grp not in out_hdl, (
                        f'Variable {grp} is not supposed to exist in the '
                        f'HDF5 file!')

                    out_hdl.create_group(grp)

                if in_time_strs is not None:
                    in_time_strs_ds = out_time_grp.create_dataset(
                        'time_strs',
                        (in_time_strs.shape[0],),
                        dtype=h5_str_dt)

                    in_time_strs_ds[:] = in_time_strs

            else:
                out_time_grp = out_hdl[self._in_time_lab]

                assert (
                    out_time_grp[self._in_time_lab].shape == in_time.shape), (
                        f'Unequal number of values of the time variable: '
                        f'{self._in_time_lab} in input and output files!')

                assert np.all(
                    out_time_grp[self._in_time_lab][...] == in_time[...]), (
                        f'Unequal corresponding values of the time variable: '
                        f'{self._in_time_lab} in input and output files!')

                if hasattr(in_time, 'units'):
                    assert out_time_grp.attrs['units'] == in_time.units

                if hasattr(in_time, 'calendar'):
                    assert out_time_grp.attrs['calendar'] == in_time.calendar

                for grp in misc_grp_labs:
                    assert grp in out_hdl, (
                        f'Variable {grp} was supposed to exist in the '
                        f'output file!')

                if (('time_strs' in out_time_grp) and
                    (in_time_strs is not None)):

                    assert (
                        out_time_grp['time_strs'].shape ==
                        in_time_strs.shape), (
                            f'Unequal number of values of the variable: '
                            f'time_strs in input and output files!')

                    assert np.all(
                        out_time_grp['time_strs'][...] == in_time_strs), (
                            f'Unequal corresponding values of the variable: '
                            f'time_strs in input and output files!')

            assert self._in_var_lab not in out_hdl[path_stem], (
                f'Specified output variable {path_stem} exists in the output '
                f'already!')

            out_var_grp = out_hdl[path_stem].create_group(self._in_var_lab)

            if hasattr(in_var, 'units'):
                out_var_grp.attrs['units'] = in_var.units

        elif self._out_fmt == 'raw':
            extrtd_data = {}

        else:
            raise NotImplementedError

        if self._vb:
            print_sl()

            print(f'INFO: Input netCDF variable\'s properties:')
            print(f'Dimensions of extraction variable: {in_var.ndim}')
            print(f'Shape of extraction variable: {in_var.shape}')
            print(f'Shape of time variable: {in_time.shape}')

            print_el()

        in_var_data = in_var[...]

        if in_time_strs is not None:
            in_time_data = in_time_strs

        else:
            in_time_data = in_time[...]

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

            assert rows_idxs_max < in_var_data.shape[1], (
                'One or more row indices are out of bounds!')

            assert cols_idxs_max < in_var_data.shape[2], (
                'One or more column indices are out of bounds!')

            crds_set = set([(x, y) for x, y in zip(cols_idxs, rows_idxs)])

            assert len(crds_set) == cols_idxs.size, (
                'Repeating row and column index combinations not allowed!')

            if self._out_fmt == 'raw':

                steps_data = {}
                for i in range(in_time.shape[0]):
                    step = in_time_data[i]
                    step_data = in_var_data[i, rows_idxs, cols_idxs]

                    steps_data[step] = step_data

                assert steps_data, 'This should not have happend!'

                extrtd_data[label] = steps_data

            elif self._out_fmt == 'h5':
                label_str = str(label)

                assert label_str not in out_var_grp, (
                    f'Dataset {label_str} exists in the output file already!')

                for add_var_lab in add_var_labels:
                    grp_lnk = f'{add_var_lab}/{label_str}'

                    assert isinstance(crds_idxs[add_var_lab], np.ndarray), (
                        'Additonal variables can only be numeric arrays!')

                    assert np.issubdtype(
                        crds_idxs[add_var_lab], np.number), (
                            'Only numeric datatypes allowed for the '
                            'additional variables!')

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

                out_var_grp[label_str] = in_var_data[
                    :, rows_idxs, cols_idxs]

                out_hdl.flush()

            else:
                raise NotImplementedError

        in_hdl.close()
        in_hdl = None

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
