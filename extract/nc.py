'''
Created on May 27, 2019

@author: Faizan-Uni
'''

from math import ceil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import psutil as ps
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

        self._h5nc_comp_level = 1

        self._set_in_flag = False
        self._set_out_flag = False
        self._set_data_extrt_flag = False
        self._set_data_snip_flag = False
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
            If the output file extension is nc then the outputs
            are written to an netCDF4 file. The extents of the nc file
            are the same as those of the shape file.
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

            Structure of the output netCDF4 is as follows:
            - Y : 3D numeric np.ndarray
                Y is the extracted variable in the variables to extract.
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

            elif fmt in ('.nc',):
                out_fmt = 'nc'

            elif fmt in ('.csv',):
                out_fmt = 'csv'

            elif fmt in ('.pkl',):
                out_fmt = 'pkl'

            else:
                raise NotImplementedError(
                    'Only configured for output file extensions of '
                    'h5, hdf5, nc, csv and pkl only!')

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

        assert in_var.size != 0, (
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

        assert in_time.size != 0, (
            f'Zero values in the input time variable: {self._in_time_lab}!')

        assert in_time.shape[0] == in_var.shape[0], (
            f'Unequal first axis lengths of the input variables: '
            f'{self._in_var_lab} and {self._in_time_lab}!')

        if hasattr(in_time, 'units') and hasattr(in_time, 'calendar'):
            try:
                in_time_strs = np.array([
                    date.strftime(self._time_strs_fmt)
                    for date in
                    num2date(in_time[...], in_time.units, in_time.calendar)])

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

        if in_time_strs is not None:
            in_time_data = in_time_strs

        else:
            in_time_data = in_time[...]

        misc_grp_labs = add_var_labels | set(('rows', 'cols'))

        extrtd_data = None

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
                        out_time_grp['time_strs'][:].astype(str) ==
                        in_time_strs), (
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

        elif self._out_fmt in ('raw',):
            extrtd_data = pd.DataFrame(
                index=in_time_data,
                columns=list(indices.keys()),
                dtype=object)

        elif self._out_fmt in ('csv',):
            extrtd_data = pd.DataFrame(
                index=in_time_data,
                columns=list(indices.keys()),
                dtype=np.float64)

        elif self._out_fmt in ('pkl',):
            extrtd_data = pd.DataFrame(
                index=pd.to_datetime(in_time_data, format=self._time_strs_fmt),
                columns=list(indices.keys()),
                dtype=np.float64)

        else:
            raise NotImplementedError

        if self._vb:
            print(f'INFO: Input netCDF variable\'s properties:')
            print(f'Dimensions of extraction variable: {in_var.ndim}')
            print(f'Shape of extraction variable: {in_var.shape}')
            print(f'Shape of time variable: {in_time.shape}')

        # Memory management.
        # NOTE: A slice can be taken out first like that in nc.
        #       Then from that slice relevant data can be extracted.
        #       It is much more efficient in terms of reading the whole array.
        in_var_nbytes = in_var.dtype.itemsize * np.prod(
            in_var.shape, dtype=np.uint64)

        tot_avail_mem = int(ps.virtual_memory().free * 0.35)

        n_mem_time_prds = ceil(in_var_nbytes / tot_avail_mem)

        assert n_mem_time_prds >= 1, n_mem_time_prds

        if n_mem_time_prds > 1:
            if self._vb: print('Memory not enough to read in one go!')

            in_var_mem_idxs = np.linspace(
                0,
                in_var.shape[0],
                n_mem_time_prds,
                endpoint=True,
                dtype=np.int64)

            assert in_var_mem_idxs[+0] == 0, in_var_mem_idxs
            assert in_var_mem_idxs[-1] == in_var.shape[0], in_var_mem_idxs

            assert np.unique(in_var_mem_idxs).size == in_var_mem_idxs.size, (
                np.unique(in_var_mem_idxs).size, in_var_mem_idxs.size)

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

            assert rows_idxs_max < in_var.shape[1], (
                'One or more row indices are out of bounds!')

            assert cols_idxs_max < in_var.shape[2], (
                'One or more column indices are out of bounds!')

            if self.raise_on_duplicate_row_col_flag:
                crds_set = set([(x, y) for x, y in zip(cols_idxs, rows_idxs)])

                assert len(crds_set) == cols_idxs.size, (
                    'Repeating row and column index combinations not allowed!')

            if self._out_fmt in ('raw', 'csv', 'pkl'):

                if n_mem_time_prds == 1:
                    steps_data = in_var[:][:, rows_idxs, cols_idxs]

                else:
                    steps_data = np.full(
                        (in_time.shape[0], rows_idxs.size),
                        np.nan,
                        dtype=in_var.dtype)

                    # print('Reading...')
                    for i, j in zip(
                        in_var_mem_idxs[:-1], in_var_mem_idxs[1:]):

                        # print(i, j)

                        otp_var_slc = in_var[i:j]
                        for k, l in enumerate(range(i, j)):
                            steps_data[l] = (
                                otp_var_slc[k][(rows_idxs, cols_idxs)])

                        otp_var_slc = None

                assert steps_data.size, 'This should not have happend!'

                if self._out_fmt == 'raw':
                    extrtd_data.loc[:, [label]] = steps_data

                elif self._out_fmt in ('csv', 'pkl'):

                    # NOTE: In case of invalid numerical values, use raw
                    #       output format and post process it outside.

                    nan_idxs = np.isnan(steps_data)
                    if nan_idxs.any():

                        for i in range(nan_idxs.shape[0]):

                            if nan_idxs[i].all(): continue

                            if not nan_idxs[i].any():
                                wtd_sum = (
                                    steps_data[i] *
                                    crds_idxs['rel_itsctd_area']).sum()

                            else:
                                rel_ara = crds_idxs[
                                    'rel_itsctd_area'][~nan_idxs[i]]

                                wtd_prd = rel_ara * steps_data[i, ~nan_idxs[i]]

                                rel_ara_sum = rel_ara.sum()

                                wtd_sum = wtd_prd.sum() / rel_ara_sum

                            assert np.isfinite(wtd_sum), wtd_sum

                            extrtd_data.loc[extrtd_data.index[i], label] = (
                                wtd_sum)

                    else:
                        extrtd_data.loc[:, label] = (
                            steps_data * crds_idxs['rel_itsctd_area']
                            ).sum(axis=1)

                else:
                    raise NotImplementedError

                steps_data = None

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

                out_lab_ds = out_var_grp.create_dataset(
                    label_str,
                    (in_var.shape[0], rows_idxs.size),
                    in_var.dtype,
                    compression='gzip',
                    compression_opts=self._h5nc_comp_level,
                    chunks=(1, rows_idxs.size))

                if n_mem_time_prds == 1:
                    out_lab_ds[:] = in_var[:][:, rows_idxs, cols_idxs]

                else:
                    # NOTE: Reading writing slices like that in nc does not
                    #       work!

                    # print('Reading...')
                    for i, j in zip(in_var_mem_idxs[:-1], in_var_mem_idxs[1:]):

                        # print(i, j)

                        otp_var_slc = in_var[i:j]
                        for k, l in enumerate(range(i, j)):
                            out_lab_ds[l] = (
                                otp_var_slc[k][(rows_idxs, cols_idxs)])

                        otp_var_slc = None
                        out_hdl.flush()

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
            assert extrtd_data is not None, 'This should not have happend!'
            self._extrtd_data = extrtd_data

        elif self._out_fmt == 'csv':
            assert extrtd_data is not None, 'This should not have happend!'

            extrtd_data.to_csv(self._out_path, sep=';')

        elif self._out_fmt == 'pkl':
            assert extrtd_data is not None, 'This should not have happend!'

            extrtd_data.to_pickle(self._out_path)

        else:
            raise NotImplementedError

        if self._vb:
            print('Done extracting netCDF values')

            print_el()

        self._set_data_extrt_flag = True
        return

    def snip_values(self, indices, nc_crds_cls):

        '''Snip the input netCDF file to a smaller one based on indicies
        in space. All timesteps are extracted. Existing file is replaced.

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
        '''

        if self._vb:
            print_sl()

            print('Snipping netCDF values...')

        assert self._set_in_flag, 'Call the set_input method first!'
        assert self._set_out_flag, 'Call the set_ouput method first!'
        assert self._out_fmt in ('nc', 'raw'), (
            'Output format not set to nc or raw!')

        assert isinstance(indices, dict), 'indices not a dictionary!'
        assert indices, 'Empty indices dictionary!'

        assert isinstance(nc_crds_cls, ExtractNetCDFCoords), (
            'nc_crds_cls not an ExtractNetCDFCoords object!')

        assert nc_crds_cls._set_crds_extrt_flag, (
            'Coordinates not extracted in nc_crds_cls!')

        self._verf_idxs(indices, False)

        in_hdl = nc.Dataset(str(self._in_path))

        assert self._in_var_lab in in_hdl.variables, (
            f'Extraction variable: {self._in_var_lab} not in the input!')

        # memory manage?
        in_var = in_hdl[self._in_var_lab]
        in_var.set_always_mask(False)

        assert in_var.ndim == 3, (
            f'Variable: {self._in_var_lab} in the input must have '
            f'3 dimensions!')

        assert in_var.size != 0, (
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

        in_x_lab = nc_crds_cls._x_crds_lab
        in_y_lab = nc_crds_cls._y_crds_lab

        assert np.issubdtype(in_time.dtype, np.number), (
            f'Data type of the input time variable: {self._in_time_lab} '
            f'not a subtype of np.number!')

        assert in_time.ndim == 1, (
            f'Input time variable: {self._in_time_lab} can only be 1D!')

        assert in_time.size != 0, (
            f'Zero values in the input time variable: {self._in_time_lab}!')

        assert in_time.shape[0] == in_var.shape[0], (
            f'Unequal first axis lengths of the input variables: '
            f'{self._in_var_lab} and {self._in_time_lab}!')

        if hasattr(in_time, 'units') and hasattr(in_time, 'calendar'):
            try:
                in_time_strs = np.array([
                    date.strftime(self._time_strs_fmt)
                    for date in
                    num2date(in_time[...], in_time.units, in_time.calendar)])

            except Exception as msg:
                if self._vb:
                    print(
                        f'WARNING: Error while converting dates to '
                        f'strings ({msg})!')

                in_time_strs = None

        else:
            in_time_strs = None

        path_stem = self._in_path.stem
        assert path_stem, 'Input file has no name?'

        if self._out_fmt == 'nc':

            snip_row_slice, snip_col_slice = self._get_snip_objs(
                indices, in_var)

            snip_col_min = snip_col_slice.start
            snip_col_max = snip_col_slice.stop - 1

            snip_row_min = snip_row_slice.start
            snip_row_max = snip_row_slice.stop - 1

            # NOTE: Mode 'a' stopped working on non-existing files for some
            # reason.
            if self._out_path.exists():
                out_hdl = nc.Dataset(str(self._out_path), mode='a')

            else:
                out_hdl = nc.Dataset(str(self._out_path), mode='w')

            # Time dim.
            in_time_dim = in_hdl[self._in_time_lab].get_dims()[0]
            in_time_dim_lab = in_time_dim.name

            if in_time_dim_lab not in out_hdl.dimensions:

                if in_time_dim.isunlimited():
                    time_size = None

                else:
                    time_size = in_hdl[self._in_time_lab].shape[0]

                out_hdl.createDimension(in_time_dim_lab, time_size)

            # X dim.
            assert (
                in_hdl[in_x_lab].ndim ==
                in_hdl[in_y_lab].ndim), (
                    in_hdl[in_x_lab].ndim,
                    in_hdl[in_y_lab].ndim)

            if in_hdl[in_x_lab].ndim == 1:
                in_x_dim = in_hdl[in_x_lab].get_dims()[0]

            elif in_hdl[in_x_lab].ndim == 2:
                in_x_dim = in_hdl[in_x_lab].get_dims()[1]

            else:
                raise ValueError(
                    'Dimensions of coordinates cannot be more than 2!')

            in_x_dim_lab = in_x_dim.name

            # Y dim.
            if in_hdl[in_y_lab].ndim == 1:
                in_y_dim = in_hdl[in_y_lab].get_dims()[0]

            elif in_hdl[in_y_lab].ndim == 2:
                in_y_dim = in_hdl[in_y_lab].get_dims()[0]

            else:
                raise ValueError(
                    'Dimensions of coordinates cannot be more than 2!')

            in_y_dim_lab = in_y_dim.name

            # Time.
            if self._in_time_lab not in out_hdl.variables:
                out_time = out_hdl.createVariable(
                    self._in_time_lab,
                    in_hdl[self._in_time_lab].dtype,
                    in_time_dim_lab)

                out_time[:] = in_hdl[self._in_time_lab][:]

                if hasattr(in_time, 'units'):
                    out_time.units = in_time.units

                if hasattr(in_time, 'calendar'):
                    out_time.calendar = in_time.calendar

                if in_time_strs is not None:
                    out_time_strs = out_hdl.createVariable(
                        f'{self._in_time_lab}_strs', str, in_time_dim_lab)

                    out_time_strs[:] = in_time_strs

            else:
                out_time = out_hdl[self._in_time_lab]

                assert (out_hdl[self._in_time_lab].size ==
                        in_hdl[self._in_time_lab].size), (
                            'Unequal length of time of inputs and outputs!')

                assert np.all(
                    out_hdl[self._in_time_lab][:] ==
                    in_hdl[self._in_time_lab][:]), (
                        'Existing time values in input and output not '
                        'the same!')

                if hasattr(in_time, 'units'):
                    assert out_time.units == in_time.units, (
                        'Units in input and output not the same!')

                if hasattr(in_time, 'calendar'):
                    assert out_time.calendar == in_time.calendar, (
                        'Calendar of input and output not the same!')

                if in_time_strs is not None:

                    assert np.all(
                        out_hdl[f'{self._in_time_lab}_strs'][:] ==
                        in_time_strs), (
                            f'{self._in_time_lab}_strs not the same in '
                            f'input and output!')

            # X.
            if in_x_dim_lab not in out_hdl.dimensions:
                out_hdl.createDimension(
                    in_x_dim_lab, snip_col_max - snip_col_min + 1)

            # Y.
            if in_y_dim_lab not in out_hdl.dimensions:
                out_hdl.createDimension(
                    in_y_dim_lab, snip_row_max - snip_row_min + 1)

            if in_x_lab not in out_hdl.variables:
                if in_hdl[in_x_lab].ndim == 1:
                    out_x = out_hdl.createVariable(
                        in_x_lab,
                        in_hdl[in_x_lab].dtype,
                        in_x_dim_lab)

                    out_x[:] = in_hdl[in_x_lab][
                        snip_col_min:snip_col_max + 1]

                elif in_hdl[in_x_lab].ndim == 2:
                    out_x = out_hdl.createVariable(
                        in_x_lab,
                        in_hdl[in_x_lab].dtype,
                        (in_y_dim_lab, in_x_dim_lab))

                    out_x[:] = in_hdl[in_x_lab][
                        snip_row_min:snip_row_max + 1,
                        snip_col_min:snip_col_max + 1]

                else:
                    raise ValueError(
                        'Dimensions of coordinates cannot be more than 2!')

                if hasattr(in_hdl[in_x_lab], 'units'):
                    out_x.units = in_hdl[in_x_lab].units

            else:
                if in_hdl[in_x_lab].ndim == 1:

                    assert ((snip_col_max - snip_col_min + 1) ==
                            out_hdl[in_x_lab].size), (
                                'Size of x coordinates not equal for inputs '
                                'and outputs!')

                    assert np.all(np.isclose(
                        in_hdl[in_x_lab][
                            snip_col_min:snip_col_max + 1],
                        out_hdl[in_x_lab][:])), (
                            'Values of x coordinates do not match for input '
                            'and output!')

                elif in_hdl[in_x_lab].ndim == 2:

                    assert ((snip_col_max - snip_col_min + 1) ==
                            out_hdl[in_x_lab].shape[1]), (
                                'Size of x coordinates not equal for inputs '
                                'and outputs!')

                    assert ((snip_row_max - snip_row_min + 1) ==
                            out_hdl[in_x_lab].shape[0]), (
                                'Size of x coordinates not equal for inputs '
                                'and outputs!')

                    assert np.all(np.isclose(
                        in_hdl[in_x_lab][
                            snip_row_min:snip_row_max + 1,
                            snip_col_min:snip_col_max + 1],
                        out_hdl[in_x_lab][:])), (
                            'Values of x coordinates do not match for input '
                            'and output!')

                else:
                    raise ValueError(
                        'Dimensions of coordinates cannot be more than 2!')

            if in_y_lab not in out_hdl.variables:

                if in_hdl[in_y_lab].ndim == 1:
                    out_y = out_hdl.createVariable(
                        in_y_lab,
                        in_hdl[in_y_lab].dtype,
                        in_y_dim_lab)

                    out_y[:] = in_hdl[in_y_lab][
                        snip_row_min:snip_row_max + 1]

                elif in_hdl[in_y_lab].ndim == 2:
                    out_y = out_hdl.createVariable(
                        in_y_lab,
                        in_hdl[in_y_lab].dtype,
                        (in_y_dim_lab, in_x_dim_lab))

                    out_y[:] = in_hdl[in_y_lab][
                        snip_row_min:snip_row_max + 1,
                        snip_col_min:snip_col_max + 1]

                else:
                    raise ValueError(
                        'Dimensions of coordinates cannot be more than 2!')

                if hasattr(in_hdl[in_y_lab], 'units'):
                    out_y.units = in_hdl[in_y_lab].units

            else:
                if in_hdl[in_y_lab].ndim == 1:

                    assert ((snip_row_max - snip_row_min + 1) ==
                            out_hdl[in_y_lab].size), (
                                'Size of y coordinates not equal for inputs '
                                'and outputs!')

                    assert np.all(np.isclose(
                        in_hdl[in_y_lab][
                            snip_row_min:snip_row_max + 1],
                        out_hdl[in_y_lab][:])), (
                            'Values of y coordinates do not match for input '
                            'and output!')

                elif in_hdl[in_y_lab].ndim == 2:

                    assert ((snip_row_max - snip_row_min + 1) ==
                            out_hdl[in_y_lab].shape[0]), (
                                'Size of y coordinates not equal for inputs '
                                'and outputs!')

                    assert ((snip_col_max - snip_col_min + 1) ==
                            out_hdl[in_y_lab].shape[1]), (
                                'Size of y coordinates not equal for inputs '
                                'and outputs!')

                    assert np.all(np.isclose(
                        in_hdl[in_y_lab][
                            snip_row_min:snip_row_max + 1,
                            snip_col_min:snip_col_max + 1],
                        out_hdl[in_y_lab][:])), (
                            'Values of y coordinates do not match for input '
                            'and output!')

                else:
                    raise ValueError(
                        'Dimensions of coordinates cannot be more than 2!')

            # Data.
            if self._in_var_lab not in out_hdl.variables:
                out_data = out_hdl.createVariable(
                    self._in_var_lab,
                    in_var[[0], [0], [0]].dtype,
                    (in_time_dim_lab, in_y_dim_lab, in_x_dim_lab),
                    compression='zlib',
                    complevel=self._h5nc_comp_level,
                    chunksizes=(1,
                                out_hdl.dimensions[in_y_dim_lab].size,
                                out_hdl.dimensions[in_x_dim_lab].size))

            else:
                out_data = out_hdl[self._in_var_lab]

                assert out_data.shape[0] == out_hdl[
                    self._in_time_lab].shape[0], (
                        'Size of time of existing output data not as '
                        'expected!')

                if out_hdl[in_x_lab].ndim == 1:

                    assert (out_data.shape[2] ==
                            out_hdl[in_x_lab].shape[0]), (
                                'Size of x coordinates of existing output '
                                'data not as expected!')

                    assert (out_data.shape[1] ==
                            out_hdl[in_y_lab].shape[0]), (
                                'Size of y coordinates of existing output '
                                'data not as expected!')

                elif out_hdl[in_x_lab].ndim == 2:

                    assert (out_data.shape[1:] ==
                            out_hdl[in_x_lab].shape), (
                                'Size of x coordinates of existing output '
                                'data not as expected!')

                    assert (out_data.shape[1:] ==
                            out_hdl[in_y_lab].shape), (
                                'Size of y coordinates of existing output '
                                'data not as expected!')

                else:
                    raise ValueError(
                        'Dimensions of coordinates cannot be more than 2!')

        elif self._out_fmt == 'raw':
            steps_data = {}

        else:
            raise NotImplementedError

        if self._vb:
            print(f'INFO: Input netCDF variable\'s properties:')
            print(f'Dimensions of extraction variable: {in_var.ndim}')
            print(f'Shape of extraction variable: {in_var.shape}')
            print(f'Shape of time variable: {in_time.shape}')

        if in_time_strs is not None:
            in_time_data = in_time_strs

        else:
            in_time_data = in_time[...]

        # Memory management.
        # Actually, the entire array is not read when slicing.
        # The limits with max mem can be higher. But I think, it does not
        # matter.
        in_var_nbytes = in_var.dtype.itemsize * np.prod(
            in_var.shape, dtype=np.uint64)

        tot_avail_mem = int(ps.virtual_memory().free * 0.35)

        n_mem_time_prds = ceil(in_var_nbytes / tot_avail_mem)

        assert n_mem_time_prds >= 1, n_mem_time_prds

        if n_mem_time_prds > 1:
            if self._vb: print('Memory not enough to read in one go!')

            in_var_mem_idxs = np.linspace(
                0,
                in_var.shape[0],
                n_mem_time_prds,
                endpoint=True,
                dtype=np.int64)

            assert in_var_mem_idxs[+0] == 0, in_var_mem_idxs
            assert in_var_mem_idxs[-1] == in_var.shape[0], in_var_mem_idxs

            assert np.unique(in_var_mem_idxs).size == in_var_mem_idxs.size, (
                np.unique(in_var_mem_idxs).size, in_var_mem_idxs.size)

        if self._out_fmt == 'raw':

            if n_mem_time_prds == 1:
                steps_slc = in_var[:, snip_row_slice, snip_col_slice]

                for i in range(in_time.shape[0]):
                    step = in_time_data[i]
                    step_data = steps_slc[i]

                    steps_data[step] = step_data

                steps_slc = step_data = step = None

            else:
                # print('Reading...')
                for i, j in zip(in_var_mem_idxs[:-1], in_var_mem_idxs[1:]):

                    # print(i, j)
                    steps_slc = in_var[i:j, snip_row_slice, snip_col_slice]

                    for k, l in enumerate(range(i, j)):
                        step = in_time_data[l]
                        step_data = steps_slc[k]

                        steps_data[step] = step_data

                    steps_slc = step_data = step = None

            assert steps_data, 'This should not have happend!'

        elif self._out_fmt == 'nc':

            if n_mem_time_prds == 1:
                out_data[:] = in_var[:, snip_row_slice, snip_col_slice]

            else:
                # print('Reading...')
                for i, j in zip(in_var_mem_idxs[:-1], in_var_mem_idxs[1:]):

                    # print(i, j)
                    out_data[i:j] = in_var[i:j, snip_row_slice, snip_col_slice]

                    out_hdl.sync()

            out_hdl.sync()

        else:
            raise NotImplementedError

        in_hdl.close()
        in_hdl = None

        if self._out_fmt == 'nc':
            out_hdl.sync()
            out_hdl.close()
            out_hdl = None

        elif self._out_fmt == 'raw':
            self._extrtd_data = steps_data

        else:
            raise NotImplementedError

        if self._vb:
            print('Done snipping netCDF values')

            print_el()

        self._set_data_snip_flag = True
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

        assert self._set_data_extrt_flag or self._set_data_snip_flag, (
            'Call the extract_values or snip_values method first!')

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

    def _get_snip_objs(self, indices, in_var):

        snip_col_min = in_var.shape[2]
        snip_col_max = 0

        snip_row_min = in_var.shape[1]
        snip_row_max = 0

        for crds_idxs in indices.values():

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

            assert rows_idxs_max < in_var.shape[1], (
                'One or more row indices are out of bounds!')

            assert cols_idxs_max < in_var.shape[2], (
                'One or more column indices are out of bounds!')

            if cols_idxs.min() < snip_col_min:
                snip_col_min = cols_idxs.min()

            if cols_idxs.max() > snip_col_max:
                snip_col_max = cols_idxs.max()

            if rows_idxs.min() < snip_row_min:
                snip_row_min = rows_idxs.min()

            if rows_idxs.max() > snip_row_max:
                snip_row_max = rows_idxs.max()

        snip_row_slice = slice(snip_row_min, snip_row_max + 1, 1)
        snip_col_slice = slice(snip_col_min, snip_col_max + 1, 1)

        return snip_row_slice, snip_col_slice
