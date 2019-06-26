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

    '''Extract the X and Y coordinates from netCDF'''

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

        '''Set the path to input netCDF file.

        Parameters
        ----------
        path_to_nc : str, pathlib.Path
            Path to the input netCDF file.
        x_crds_lab : str
            Label of the variable representing the X coordinates array.
        y_crds_lab : str
            Label of the variable representing the Y coordinates array.
        '''

        if self._vb:
            print_sl()

            print('Setting netCDF coordinates\' extraction input...')

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

            print(f'INFO: Set the following parameters for the netCDF:')
            print(f'Path: {self._in_path}')
            print(f'X coordinates\' label: {self._x_crds_lab}')
            print(f'Y coordinates\' label: {self._y_crds_lab}')

            print_el()

        self._set_in_flag = True
        return

    def extract_coordinates(self):

        '''Run the coordinates extraction algorithm'''

        if self._vb:
            print_sl()

            print('Extracting netCDF coordinates...')

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
                print(
                    f'INFO: X and Y coordinates arrays were masked. '
                    f'Took the "data" attribute!')

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

        assert np.issubdtype(self._x_crds.dtype, np.number), (
            'X coordinates are non-numeric!')

        assert np.issubdtype(self._y_crds.dtype, np.number), (
            'Y coordinates are non-numeric!')

        if self._vb:
            print('Done extracting GeoTiff coordinates')

            print_el()

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

        '''Return the X coordinates extracted from the specified netCDF file
        in the set_input method.

        Returns
        -------
        _x_crds : nD numeric np.ndarray
            X coordinates from the netCDF file.
            Dimensions depend on the input.
        '''

        assert self._set_crds_extrt_flag, (
            'Call the extract_coordinates method first!')

        assert self._x_crds is not None, 'This should not have happend!'

        return self._x_crds

    def get_y_coordinates(self):

        '''Return the Y coordinates extracted from the specified netCDF file
        in the set_input method.

        Returns
        -------
        _y_crds : nD numeric np.ndarray
            Y coordinates from the netCDF file.
            Dimensions depend on the input.
        '''

        assert self._set_crds_extrt_flag, (
            'Call the extract_coordinates method first!')
        assert self._y_crds is not None, 'This should not have happend!'

        return self._y_crds


class ExtractNetCDFValues:

    '''Extract values from a netCDF at given indices'''

    def __init__(self, verbose=True):

        self._vb = verbose

        self._in_path = None
        self._in_var_lab = None
        self._in_time_lab = None

        self._out_path = None
        self._out_fmt = None

        self._time_strs_fmt = '%Y%m%dT%H%M%S'

        self._extrtd_data = None

        self.raise_on_duplicate_row_col_flag = True

        self._set_in_flag = False
        self._set_out_flag = False
        self._set_data_extrt_flag = False
        return

    def set_input(self, path_to_nc, variable_label, time_label):

        '''Set the path to input netCDF file.

        Parameters
        ----------
        path_to_nc : str, pathlib.Path
            Path to the input netCDF file.
        variable_label : str
            Label of the variable that has to be extracted. The variable has
            to be 3D and numeric. with axes 0, 1, 2 representing time, X and
            Y coordinates respectively.
        time_label : str
            Label of the variable that represents time. It should be 1D and
            numeric with the same length as the first axis of the extraction
            variable.
        '''

        if self._vb:
            print_sl()

            print('Setting netCDF values\' extraction input...')

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
            print(f'INFO: Set the following parameters for the netCDF:')
            print(f'Path: {self._in_path}')
            print(f'Extraction variable label: {self._in_var_lab}')
            print(f'Time label: {self._in_time_lab}')

            print_el()

        self._set_in_flag = True
        return

    def set_output(self, path_to_output=None):

        '''Set the path to output file.

        Parameters
        ----------
        path_to_output : None, str, pathlib.Path
            Path to the output file. If None, then extracted values can be
            returned by a call to the get_extracted_data method as a
            dictionary. See the documentation of the get_extracted_data
            method for the output format. If the output file extension is
            h5 or hdf5 then the outputs are written to an HDF5 file.
            No other output format is defined yet.

            Structure of the output HDF5 is as follows:
            - X : N 2D numeric np.ndarrays
                Where X is the stem of the input netCDF file name.
                N is the number of keys of the indices variable passed to
                the extract_data_for_indices_and_save method. Every key is
                a dataset.
                Every dataset holds the extracted values as a 2D array at
                the given indices from the input netCDF. First dimension
                is time and the second is the cell value at corresponding
                row and column indices at a given time step.
            - cols : 1D int np.ndarray
                Column indices of the extracted values in the input netCDF.
                Shape of this variable is equal to the length of the
                second axis of the variable X.
            - rows : 1D int np.ndarray
                Row indices of the extracted values in the input GeoTiff.
                Shape of this variable is equal to the length of the
                second axis of the variable X.
            - time : 1D numeric np.ndarray
                The time variable as is in the netCDF. Values of the
                extraction variable is taken at all time steps.
            - time_strs : 1D object np.ndarray
                String representation of the time variable. If time has units
                and a calendar. The format is taken from the variable
                self._time_strs_fmt.

            Additional variables that might exist based on the value of the
            save_add_vars_flag variable passed to the
            extract_data_for_indices_and_save method come from the indices
            dictionary. There are written as they are to the output file but
            should be all numpy numeric dtype arrays.
        '''

        if self._vb:
            print_sl()

            print('Setting netCDF values\' extraction output...')

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
            print(f'INFO: Set the following parameters for the output:')
            print(f'Path: {self._out_path}')
            print(f'Format: {self._out_fmt}')

            print_el()

        self._set_out_flag = True
        return

    def extract_values(
            self,
            indices,
            save_add_vars_flag=True,
            ignore_rows_cols_equality=False):

        '''Extract the values at given indices.

        Parameters
        ----------
        indices : dict
            A dictionary whose keys are labels of polygons they represent.
            The values are also dictionaries that must have the
            keys \'cols\' and \'rows\' representing the columns and rows
            in the netCDF array for each label respectively.
            Values of keys other than these are written to an HDF5 file
            specified in the set_output method if path_to_output is not None
            and save_add_vars_flag is True. All additonal values should be
            numpy numeric arrays and are written as they are.
        save_add_vars_flag : bool
            Whether to write variables other than \'cols\' and \'rows\' in the
            items of the indices dictionary to the output HDF5 file.
        ignore_rows_cols_equality : bool
            Whether to ignore equality of rows/cols array in case it
            exists already. This happens when the output HDF5 was written
            to before using another input whose extents were different than
            the current raster but cell sizes are equal and hence the number
            of cells. The other variables in indices should match the
            previously written values. save_add_vars_flag is set to False is
            ignore_rows_cols_equality is True.
        '''

        if self._vb:
            print_sl()

            print('Extracting netCDF values...')

        assert self._set_in_flag, 'Call the set_input method first!'
        assert self._set_out_flag, 'Call the set_ouput method first!'

        assert isinstance(indices, dict), 'indices not a dictionary!'
        assert indices, 'Empty indices dictionary!'

        assert isinstance(save_add_vars_flag, bool), (
            'save_add_vars_flag not a boolean!')

        assert isinstance(ignore_rows_cols_equality, bool), (
            'ignore_rows_cols_equality not a boolean!')

        if ignore_rows_cols_equality and save_add_vars_flag:
            save_add_vars_flag = False

            if self._vb:
                print(
                    'INFO: save_add_vars_flag set to False because '
                    'ignore_rows_cols_equality is True!')

        add_var_labels = self._verf_idxs(indices, save_add_vars_flag)

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
            try:
                in_time_strs = pd.DatetimeIndex(num2date(
                    in_time[...],
                    in_time.units,
                    in_time.calendar)).strftime(self._time_strs_fmt)

                h5_str_dt = h5py.special_dtype(vlen=str)

            except Exception as msg:
                if self._vb:
                    print(
                        f'WARNING: Error while converting dates to '
                        f'strings ({msg})!')

                in_time_strs = None
                h5_str_dt = None

        else:
            in_time_strs = None
            h5_str_dt = None

        path_stem = self._in_path.stem
        assert path_stem, 'Input file has no name?'

        misc_grp_labs = add_var_labels | set(('rows', 'cols'))

        if self._out_fmt == 'h5':
            out_hdl = h5py.File(str(self._out_path), mode='a', driver=None)

            if self._in_time_lab not in out_hdl:
                out_time_grp = out_hdl.create_group(self._in_time_lab)
                out_time_grp[self._in_time_lab] = in_time[...]

                if hasattr(in_time, 'units'):
                    out_time_grp.attrs['units'] = in_time.units

                if hasattr(in_time, 'calendar'):
                    out_time_grp.attrs['calendar'] = in_time.calendar

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

            misc_grps_flags = [grp in out_hdl for grp in misc_grp_labs]
            if any(misc_grps_flags):
                for grp in misc_grp_labs:
                    assert grp in out_hdl, (
                        f'Variable {grp} was supposed to exist in the '
                        f'output file!')

            else:
                for grp in misc_grp_labs:
                    assert grp not in out_hdl, (
                        f'Variable {grp} was not supposed to exist in the '
                        f'HDF5 file!')

                    out_hdl.create_group(grp)

            if path_stem not in out_hdl:
                out_stem_grp = out_hdl.create_group(path_stem)

            else:
                out_stem_grp = out_hdl[path_stem]

            assert self._in_var_lab not in out_stem_grp, (
                f'Specified output variable {self._in_var_lab} exists '
                f'in the {path_stem} group of the output '
                f'already!')

            out_var_grp = out_stem_grp.create_group(self._in_var_lab)

            if hasattr(in_var, 'units'):
                out_var_grp.attrs['units'] = in_var.units

        elif self._out_fmt == 'raw':
            extrtd_data = {}

        else:
            raise NotImplementedError

        if self._vb:
            print(f'INFO: Input netCDF variable\'s properties:')
            print(f'Dimensions of extraction variable: {in_var.ndim}')
            print(f'Shape of extraction variable: {in_var.shape}')
            print(f'Shape of time variable: {in_time.shape}')

        in_var_data = in_var[...]

        if in_time_strs is not None:
            in_time_data = in_time_strs

        else:
            in_time_data = in_time[...]

        for label, crds_idxs in indices.items():

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

            if self.raise_on_duplicate_row_col_flag:
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
                        crds_idxs[add_var_lab].dtype, np.number), (
                            'Only numeric datatypes allowed for the '
                            'additional variables!')

                    if label_str in out_hdl[add_var_lab]:
                        assert (out_hdl[grp_lnk].shape ==
                            crds_idxs[add_var_lab].shape), (
                                f'Shape of existing variable: {add_var_lab} '
                                f'inside the HDF5 and current ones is '
                                f'unequal!')

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

                    assert (out_hdl[f'cols/{label_str}'].shape ==
                        cols_idxs.shape), (
                            'Shape of existing column indices inside the '
                            'HDF5 and current ones is unequal!')

                    if not ignore_rows_cols_equality:
                        assert np.all(
                            out_hdl[f'rows/{label_str}'][...] == rows_idxs), (
                                f'Existing values of rows in the HDF5 '
                                f'and the current ones are unequal!')

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

        if self._vb:
            print('Done extracting netCDF values')

            print_el()

        self._set_data_extrt_flag = True
        return

    def get_values(self):

        '''Get the data that was extracted by a call to the
        extract_data_for_indices_and_save method.

        Will only work if the path_to_output was None in the set_output method.

        Return
        ------
        _extrtd_data : dict
            A dictionary with keys as labels of the indices dictionary
            passed to the extract_data_for_indices_and_save method and
            values that are also dictionaries with keys in the time format
            given by the _time_strs_fmt variable if the netCDF time had units
            and a calendar. Otherwise it is the value in the time array.
            The values are 1D np.float64 arrays having the values at
            corresponding row and column indices in that band taken from the
            indices dictionary passed to the extract_data_for_indices_and_save
            method.
        '''

        assert self._out_fmt == 'raw', (
            'Call to this function allowed only when path_to_output is None!')

        assert self._set_data_extrt_flag, (
            'Call the extract_data_for_indices_and_save method first!')

        assert self._extrtd_data is not None, 'This should not have happend!'

        return self._extrtd_data

    def _verf_idxs(self, indices, save_add_vars_flag):

        add_var_labels_main = set()

        for crds_idxs in indices.values():

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
