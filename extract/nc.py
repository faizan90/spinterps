'''
Created on May 27, 2019

@author: Faizan-Uni
'''

from pathlib import Path

import h5py
import numpy as np
import netCDF4 as nc

from ..misc import print_sl, print_el


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

        assert isinstance(path_to_nc, (str, Path))

        assert isinstance(x_crds_lab, str)
        assert x_crds_lab

        assert isinstance(y_crds_lab, str)
        assert y_crds_lab

        path_to_nc = Path(path_to_nc).absolute()

        assert path_to_nc.exists()

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

        assert self._set_in_flag

        in_hdl = nc.Dataset(str(self._in_path))

        assert self._x_crds_lab in in_hdl.variables
        assert self._y_crds_lab in in_hdl.variables

        self._x_crds = in_hdl[self._x_crds_lab][...]
        self._y_crds = in_hdl[self._y_crds_lab][...]

        assert np.all(self._x_crds.shape)
        assert np.all(self._y_crds.shape)

        if isinstance(self._x_crds, np.ma.MaskedArray):
            self._x_crds = self._x_crds.data
            self._y_crds = self._y_crds.data

            if self._vb:
                print_sl()

                print(
                    f'INFO: X and coordinates array were masked. '
                    f'Took the "data" attribute!')

                print_el()

        elif (isinstance(self._x_crds, np.ndarray) and
              isinstance(self._y_crds, np.ndarray)):
            pass

        else:
            raise NotImplementedError

        assert np.unique(self._x_crds.size) == self._x_crds.size
        assert np.unique(self._y_crds.size) == self._y_crds.size

        assert np.all(np.isfinite(self._x_crds))
        assert np.all(np.isfinite(self._y_crds))

        assert self._x_crds.ndim == 1
        assert self._x_crds.ndim == self._y_crds.ndim

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

        assert self._set_crds_extrt_flag

        return self._x_crds

    def get_y_coordinates(self):

        assert self._set_crds_extrt_flag

        return self._y_crds


class ExtractNetCDFValues:

    def __init__(self, verbose=True):

        self._vb = verbose

        self._extrtd_data = None

        self._in_path = None
        self._in_var_lab = None
        self._in_time_lab = None

        self._out_path = None
        self._out_fmt = None

        self._set_in_flag = False
        self._set_out_flag = False
        self._set_data_extrt_flag = False
        return

    def set_input(self, path_to_nc, variable_label, time_label):

        assert isinstance(path_to_nc, (str, Path))

        assert isinstance(variable_label, str)
        assert variable_label

        assert isinstance(time_label, str)
        assert time_label

        path_to_nc = Path(path_to_nc).absolute()

        assert path_to_nc.exists()

        self._in_path = path_to_nc
        self._in_var_lab = variable_label
        self._in_time_lab = time_label

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the netCDF:')
            print(f'Path: {self._in_path}')
            print(f'Variable label: {self._in_var_lab}')
            print(f'Time label: {self._in_time_lab}')

            print_el()

        self._set_in_flag = True
        return

    def set_output(self, path_to_output):

        if path_to_output is None:
            self._out_fmt = 'raw'

        else:
            assert isinstance(path_to_output, (str, Path))

            path_to_output = Path(path_to_output).absolute()

            assert path_to_output.parents[0].exists()

            fmt = path_to_output.suffix

            if fmt in ('.h5', '.hdf5'):
                self._out_fmt = 'h5'

            else:
                raise NotImplementedError

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

#         for label, crds_idxs in indicies.items():
        for crds_idxs in indicies.values():
            assert 'cols' in crds_idxs
            cols_idxs = crds_idxs['cols']
            assert cols_idxs.ndim == 1
            assert cols_idxs.size > 0
            assert np.issubdtype(cols_idxs.dtype, np.integer)

            assert 'rows' in crds_idxs
            rows_idxs = crds_idxs['rows']
            assert rows_idxs.ndim == 1
            assert rows_idxs.size > 0
            assert np.issubdtype(rows_idxs.dtype, np.integer)

            if not save_add_vars_flag:
                continue

            if not add_var_labels_main:
                add_var_labels_main = set(
                    crds_idxs.keys()) - set(('rows', 'cols'))

            else:
                assert not (
                    add_var_labels_main -
                    set(crds_idxs.keys()) -
                    set(('rows', 'cols')))

        return add_var_labels_main

    def extract_data_for_indicies_and_save(
            self, indicies, save_add_vars_flag=True):

        assert self._set_in_flag
        assert self._set_out_flag

        assert isinstance(indicies, dict)
        assert indicies

        assert isinstance(save_add_vars_flag, bool)

        add_var_labels = self._verf_idxs(indicies, save_add_vars_flag)

        in_hdl = nc.Dataset(str(self._in_path))

        assert self._in_var_lab in in_hdl.variables
        assert self._in_time_lab in in_hdl.variables

        # memory manage?
        in_var = in_hdl[self._in_var_lab]
        in_var.set_always_mask(False)

        assert in_var.ndim == 3
        assert in_var.size > 0
        assert all(in_var.shape)
        assert np.issubdtype(in_var.dtype, np.number)

        in_time = in_hdl[self._in_time_lab]
        in_time.set_always_mask(False)
        assert np.issubdtype(in_time.dtype, np.number)

        assert in_time.ndim == 1
        assert in_time.size > 0

        assert in_time.shape[0] == in_var.shape[0]

        misc_grp_labs = add_var_labels | set(('rows', 'cols', 'data'))

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
                    assert grp not in out_hdl
                    out_hdl.create_group(grp)

            else:
                out_time_grp = out_hdl[self._in_time_lab]

                assert np.all(
                    out_time_grp[self._in_time_lab][...] == in_time[...])

                if hasattr(in_time, 'units'):
                    assert out_time_grp.attrs['units'] == in_time.units

                if hasattr(in_time, 'calendar'):
                    assert out_time_grp.attrs['calendar'] == in_time.calendar

                for grp in misc_grp_labs:
                    assert grp in out_hdl

            assert self._in_var_lab not in out_hdl['data']

            out_var_grp = out_hdl['data'].create_group(self._in_var_lab)

            if hasattr(in_var, 'units'):
                out_var_grp.attrs['units'] = in_var.units

        elif self._out_fmt == 'raw':
            extrt_data = {}

        else:
            raise NotImplementedError

        if self._vb:
            print_sl()

            print(f'INFO: Input netCDF variable\'s properties:')
            print(f'Dimensions of variable: {in_var.ndim}')
            print(f'Shape of variable: {in_var.shape}')
            print(f'Shape of time: {in_time.shape}')

            print_el()

        in_var_data = in_var[...]
        for label, crds_idxs in indicies.items():

            cols_idxs = crds_idxs['cols']
            cols_idxs_min = cols_idxs.min()
            cols_idxs_max = cols_idxs.max()
            assert (cols_idxs_max > 0) & (cols_idxs_min > 0)

            rows_idxs = crds_idxs['rows']
            rows_idxs_min = rows_idxs.min()
            rows_idxs_max = rows_idxs.max()
            assert (rows_idxs_max > 0) & (rows_idxs_min > 0)

            assert cols_idxs.shape == rows_idxs.shape

            assert rows_idxs_max < in_var_data.shape[1]
            assert cols_idxs_max < in_var_data.shape[2]

            crds_set = set([(x, y) for x, y in zip(cols_idxs, rows_idxs)])

            assert len(crds_set) == cols_idxs.size

            if self._out_fmt == 'raw':
                steps_data = {}

                for i in range(in_time.shape[0]):
                    step = in_time[i].item()
                    step_data = in_var_data[i, rows_idxs, cols_idxs]

                    steps_data[step] = step_data

                assert steps_data

                extrt_data[label] = steps_data

            elif self._out_fmt == 'h5':
                label_str = str(label)

                for add_var_lab in add_var_labels:
                    grp_lnk = f'{add_var_lab}/{label_str}'

                    if label_str in out_hdl[add_var_lab]:
                        assert np.all(np.isclose(
                            out_hdl[grp_lnk][...],
                            crds_idxs[add_var_lab]))

                    else:
                        out_hdl[grp_lnk] = crds_idxs[add_var_lab]

                if label_str in out_hdl['rows']:
                    assert np.all(
                        out_hdl[f'rows/{label_str}'][...] == rows_idxs)

                    assert np.all(
                        out_hdl[f'cols/{label_str}'][...] == cols_idxs)

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
            assert extrt_data
            self._extrtd_data = extrt_data

        else:
            raise NotImplementedError

        self._set_data_extrt_flag = True
        return

    def get_extracted_data(self):

        assert self._out_fmt == 'raw'

        assert self._set_data_extrt_flag

        assert self._extrtd_data is not None

        return self._extrtd_data
