'''
Created on Nov 25, 2018

@author: Faizan
'''

from pathlib import Path

import numpy as np
import pandas as pd

from ..variograms.vgsinput import VariogramsData as VD


class KrigingData(VD):

    def __init__(self, verbose=True):

        VD.__init__(self, verbose)

        self._vg_ser_set_flag = False
        self._out_dir_set_flag = False
        self._nc_set_flag = False
        self._time_prms_set_flag = False
        self._cell_sel_prms_set = False
        self._algn_ras_set_flag = False
        self._misc_settings_set_flag = False
        self._data_vrfd_flag = False
        return

    def set_vgs_ser(self, vgs_ser):

        assert isinstance(vgs_ser, pd.Series), (
            'vgs_ser has to be a pd.Series object!')

        assert isinstance(vgs_ser.index, pd.DatetimeIndex), (
            'Index of vgs_ser has to be a pd.DatetimeIndex object!')

        assert all(vgs_ser.shape), 'Empty vgs_ser!'

        self._vgs_ser = vgs_ser

        miss_steps_ctr = 0
        for date in self._vgs_ser.index:
            if len(self._vgs_ser.loc[date]) >= 6:
                continue

            self._vgs_ser[date] = 'nan'
            miss_steps_ctr += 1

        if self._vb:
            print('\n', '#' * 10, sep='')
            print(
                f'vgs_ser set with a shape of: {self._vgs_ser.shape}. '
                f'Out of which {miss_steps_ctr} have no variograms.')
            print('#' * 10)

        # TODO: do an intersect with data df index
        self._vg_ser_set_flag = True
        return

    def set_out_dir(self, out_dir):

        assert isinstance(out_dir, (str, Path))

        out_dir = Path(out_dir).absolute()

        assert out_dir.parents[0].exists()

        self._out_dir = out_dir

        if self._vb:
            print('\n', '#' * 10, sep='')
            print('Outputs directory set to:', str(self._out_dir))
            print('#' * 10)

        self._out_dir_set_flag = True
        return

    def set_netcdf4_parameters(
            self,
            out_file_name,
            var_units,
            var_label,
            time_units,
            time_calendar):

        assert isinstance(out_file_name, str)
        assert out_file_name

        self._nc_out = out_file_name

        assert isinstance(var_units, str)
        assert var_units

        self._nc_vunits = var_units

        assert isinstance(var_label, str)
        assert var_label

        self._nc_vlab = var_label

        assert isinstance(time_units, str)
        assert time_units

        self._nc_tunits = time_units

        assert isinstance(time_calendar, str)
        assert time_calendar

        self._nc_tcldr = time_calendar

        self._nc_tlab = 'time'
        self._nc_xlab = 'X'
        self._nc_ylab = 'Y'

        if self._vb:
            print('\n', '#' * 10, sep='')
            print('Set the following output netCDF4 file parameters:')
            print(f'File name: {self._nc_out}')
            print(f'Variable units: {self._nc_vunits}')
            print(f'Variable label: {self._nc_vlab}')
            print(f'Time units: {self._nc_tunits}')
            print(f'Time calendar: {self._nc_tcldr}')
            print(f'Time label: {self._nc_tlab}')
            print(f'X-coordinates label: {self._nc_xlab}')
            print(f'Y-coordinates label: {self._nc_ylab}')
            print('#' * 10)

        self._nc_set_flag = True
        return

    def set_interp_time_parameters(
            self,
            beg_time,
            end_time,
            time_freq,
            time_fmt=None):

        assert isinstance(time_fmt, str)
        assert isinstance(time_freq, str)

        self._tfreq = time_freq

        if isinstance(beg_time, str):
            self._tbeg = pd.to_datetime(beg_time, format=time_fmt)

        elif isinstance(beg_time, pd.Timestamp):
            self._tbeg = beg_time

        else:
            raise AssertionError(
                'beg_time can only be an str or a pd.Timestamp object!')

        if isinstance(end_time, str):
            self._tend = pd.to_datetime(end_time, format=time_fmt)

        elif isinstance(end_time, pd.Timestamp):
            self._tend = end_time

        else:
            raise AssertionError(
                'end_time can only be an str or a pd.Timestamp object!')

        if self._vb:
            print('\n', '#' * 10, sep='')
            print('Set the following time interpolation parameters:')
            print('Begin time:', self._tbeg)
            print('End time:', self._tend)
            print('Time frequency:', self._tfreq)
            print('#' * 10)

        self._time_prms_set_flag = True
        return

    def set_cell_selection_parameters(
            self,
            polygons_shapefile,
            station_select_buffer_distance,
            polygon_cell_buffer_distance=None,
            interp_full_grid_flag=False):

        assert isinstance(polygons_shapefile, (str, Path))

        polygons_shapefile = Path(polygons_shapefile).absolute()

        assert polygons_shapefile.exists()
        assert polygons_shapefile.is_file()

        self._poly_shp = polygons_shapefile

        assert isinstance(station_select_buffer_distance, (float, int))
        assert 0 <= station_select_buffer_distance < np.inf

        self._stn_bdist = float(station_select_buffer_distance)

        assert isinstance(interp_full_grid_flag, bool)

        self._ifull_grd_flag = interp_full_grid_flag

        if not self._ifull_grd_flag:
            assert isinstance(polygon_cell_buffer_distance, (float, int))
            assert 0 <= polygon_cell_buffer_distance < np.inf

            self._cell_bdist = float(polygon_cell_buffer_distance)

        if self._vb:
            print('\n', '#' * 10, sep='')
            print('Set the following cell selection parameters:')
            print('Polygons shapefile:', str(self._poly_shp))
            print('interp_full_grid_flag:', self._ifull_grd_flag)
            print('Buffer distance to select a station:', self._stn_bdist)
            print('Buffer distance to select a cell:', self._cell_bdist)
            print('#' * 10)

        self._cell_sel_prms_set = True
        return

    def set_alignment_raster(self, align_raster):

        assert isinstance(align_raster, (str, Path))

        align_raster = Path(align_raster).absolute()

        assert align_raster.exists()
        assert align_raster.is_file()

        self._algn_ras = align_raster

        if self._vb:
            print('\n', '#' * 10, sep='')
            print('Alignment raster set to:', str(self._algn_ras))
            print('#' * 10)

        self._algn_ras_set_flag = True
        return

    def set_misc_settings(
            self,
            n_cpus,
            n_cpus_scale=1,
            plot_figs_flag=False,
            cell_size=None,
            min_value_to_krige_thresh=None,
            min_cutoff_value=None,
            max_cutoff_value=None):

        assert isinstance(n_cpus, int)
        assert 0 < n_cpus

        self._n_cpus = n_cpus

        assert isinstance(n_cpus_scale, int)
        assert 0 < n_cpus_scale

        self._n_cpus_scale = n_cpus_scale

        assert isinstance(plot_figs_flag, bool)
        self._plot_figs_flag = plot_figs_flag

        if cell_size is not None:
            assert isinstance(cell_size, (int, float))
            assert 0 < cell_size < np.inf

            self._cell_size = float(cell_size)

        else:
            self._cell_size = cell_size

        if min_value_to_krige_thresh is not None:
            assert isinstance(min_value_to_krige_thresh, (int, float))
            assert -np.inf <= min_value_to_krige_thresh < np.inf

            self._min_var_thr = float(min_value_to_krige_thresh)

        else:
            self._min_var_thr = min_value_to_krige_thresh

        if min_cutoff_value is not None:
            assert isinstance(min_cutoff_value, (int, float))
            assert -np.inf <= min_cutoff_value <= np.inf

            self._min_var_cut = float(min_cutoff_value)

        else:
            self._min_var_cut = min_cutoff_value

        if max_cutoff_value is not None:
            assert isinstance(max_cutoff_value, (int, float))
            assert -np.inf <= max_cutoff_value <= np.inf

            self._max_var_cut = float(max_cutoff_value)

        else:
            self._max_var_cut = max_cutoff_value

        if (self._min_var_thr is not None) and (self._min_var_cut is not None):

            assert self._min_var_thr >= self._min_var_cut

        if (self._min_var_cut is not None) and (self._max_var_cut is not None):

            assert self._min_var_cut < self._max_var_cut

        if (self._min_var_thr is not None) and (self._max_var_cut is not None):

            assert self._min_var_thr < self._max_var_cut

        if self._vb:
            print('\n', '#' * 10, sep='')
            print('Set the following misc. settings:')
            print('n_cpus:', self._n_cpus)
            print('n_cpus_scale:', self._n_cpus_scale)
            print('plot_figs_flag:', self._plot_figs_flag)
            print('cell_size:', self._cell_size)
            print('min_value_to_krige_thresh:', self._min_var_thr)
            print('min_cutoff_value:', self._min_var_cut)
            print('max_cutoff_value:', self._max_var_cut)
            print('#' * 10)

        self._misc_settings_set_flag = True
        return

    def verify(self):

        assert self._data_set_flag
        assert self._out_dir_set_flag
        assert self._nc_set_flag
        assert self._time_prms_set_flag
        assert self._misc_settings_set_flag

        if not self._cell_sel_prms_set:
            if self._vb:
                print('\n', '#' * 10, sep='')
                print('Cell selection parameters were not set!')
                print('#' * 10)

        if not self._algn_ras_set_flag:
            if self._vb:
                print('\n', '#' * 10, sep='')
                print('Alignment raster was not set!')
                print('#' * 10)

        self._data_vrfd_flag = True
        return

    __verify = verify
