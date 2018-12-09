'''
Created on Nov 25, 2018

@author: Faizan
'''

from pathlib import Path

import numpy as np
import pandas as pd

from ..variograms.vgsinput import VariogramsData as VD


class SpInterpData(VD):

    def __init__(self, verbose=True):

        VD.__init__(self, verbose)

        self._vgs_ser = None

        self._cell_bdist = None

        self._n_cpus = 1
        self._mp_flag = False

        self._plot_figs_flag = False

        self._cell_size = None

        self._min_var_thr = None
        self._min_var_cut = None
        self._max_var_cut = None

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

        '''
        Set the time series of variograms

        Parameters
        ----------
        vg_ser : object pd.Series
            A series object having one variogram for each time step
            for kriging. If this function is not called then only IDW
            is possible. Its datatype should be object.
        '''

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

        self._vg_ser_set_flag = True
        return

    def set_out_dir(self, out_dir):

        '''Set the main outputs directory'''

        assert isinstance(out_dir, (str, Path)), (
            'out_dir can only be a string or pathlib.Path object!')

        out_dir = Path(out_dir).absolute()

        assert out_dir.parents[0].exists(), (
            'Parent directory of out_dir does not exist!')

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

        '''
        Set some properties of the output netCDF file.

        Parameters
        ----------
        out_file_name : str
            Name of the output netCDF4 file. This file will hold the time
            series of interpolated grids for each type of interpolation
            that is specified.
        var_units : str
            Units of the variable that is interpolated. Can be an empty
            string as well. This is not important for the interpolation but
            may help others to know what the units of the variable are.
        var_label : str
            Name of the interpolated variable. Can be anything. Again this
            is not important for the interpolation but may help others to '
            know what the name of the variable is.
        time_units : str
            Starting date that is used as a reference by the netCDF4 module
            to convert datetime objects to integers and the other way around.
            It generally looks like days since 1900-00-00. Read the netCDF4
            documentation to get a better idea.
        time_calendar : str
            A calendar name that is compatible with the netCDF4 library.
            It is typically gregorian or something. Read the netCDF4
            documentation to get a better idea.
        '''

        assert isinstance(out_file_name, str), (
            'out_file_name file not a string!')

        assert out_file_name, 'Empty out_file_name!'

        self._nc_out = out_file_name

        assert isinstance(var_units, str), 'var_units not a string!'

        self._nc_vunits = var_units

        assert isinstance(var_label, str), 'var_label not a string!'

        self._nc_vlab = var_label

        assert isinstance(time_units, str), 'time_units not a string!'
        assert time_units, 'Empty time_units string!'

        self._nc_tunits = time_units

        assert isinstance(time_calendar, str), 'time_calendar not a string!'
        assert time_calendar, 'Empty time_calendar string!'

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

        '''
        Set the starting and ending time for the interpolation.

        Parameters
        ----------
        beg_time : str or pandas.Timestamp
            Start time of the interpolation. If a string, then time_fmt
            should also be a string containing the format.
        end_time : str or pandas.Timestamp
            End time of the interpolation. If a string, then time_fmt
            should also be a string containing the format.
        time_freq : str
            Time frequency between the starting and ending time. Can be
            D, A etc. It has to be a valid pandas time frequency string.
            Input station data and variogram series will be reindexed
            based on beg_time, end_time and time_freq.
        time_fmt : None or str
            A string representing the time format of beg_time and end_time
            if they are string objects.
        '''

        assert isinstance(time_fmt, str) or time_fmt is None, (
            'time_fmt not a string or None!')

        assert isinstance(time_freq, str), 'time-_freq not a string!'

        self._tfreq = time_freq

        if isinstance(beg_time, str):
            assert isinstance(time_fmt, str), (
                'time_fmt has to be a string if beg_time is a string!')

            self._tbeg = pd.to_datetime(beg_time, format=time_fmt)

        elif isinstance(beg_time, pd.Timestamp):
            self._tbeg = beg_time

        else:
            raise AssertionError(
                'beg_time can only be an str or a pd.Timestamp object!')

        if isinstance(end_time, str):
            assert isinstance(time_fmt, str), (
                'time_fmt has to be a string if end_time is a string!')

            assert isinstance(time_fmt, str), 'time_fmt not a string!'

            self._tend = pd.to_datetime(end_time, format=time_fmt)

        elif isinstance(end_time, pd.Timestamp):
            self._tend = end_time

        else:
            raise AssertionError(
                'end_time can only be an str or a pd.Timestamp object!')

        assert self._tend >= self._tbeg, (
            'Begining time of interpolation cannot be less than the ending '
            'time!')

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
            interp_around_polys_flag=True,
            polygon_cell_buffer_distance=None):

        '''
        Set some interpolation parameters.

        Parameters
        ----------
        polygons_shapefile : str or pathlib.Path
            Path to the shapefile that holds the polygons that may
            be used to select the nearest stations to the polygons in
            the cordinates dataframe based on the value of
            station_select_buffer_distance. This can make the
            interpolation faster by droping the stations that have no
            effect on the interpolation. These polygons can also be used
            to interpolate only those cells that are inside or near
            the polygons. It is up to the user to use consistent units
            for every parameter.
        station_select_buffer_distance : int or float
            Maximum farthest distance that a station might have from the
            nearest polygon, in order for it to considered a neighbor
            in the interpolation computation. It is up to the user to use
            consistent units for every parameter.
        interp_around_polys_flag : bool
            A flag to signal if cells within polygon_cell_buffer_distance
            of the nearest polygon are interpolated only. This saves
            time because the farther polygons are not needed anyway.
            If False, the entire grid between minimum and maximum station
            coordinates is interpolated.
        polygon_cell_buffer_distance : None or int or float
            The maximum distance away from the nearest polygon so that
            a cell is considered for interpolation. None by default but if
            interp_around_polys_flag is True then should be a float or
            an int.  It is up to the user to use consistent units for
            every parameter.
        '''

        assert isinstance(polygons_shapefile, (str, Path)), (
            'polygons_shapefile has to be a string or a pathlib.Path object!')

        polygons_shapefile = Path(polygons_shapefile).absolute()

        assert polygons_shapefile.exists(), (
            'polygons_shapefile does not exist!')

        assert polygons_shapefile.is_file(), (
            'polygons_shapefile is not a file!')

        self._poly_shp = polygons_shapefile

        assert isinstance(station_select_buffer_distance, (float, int)), (
            'station_select_buffer_distance has to be a float or an int!')

        assert 0 <= station_select_buffer_distance < np.inf, (
            'station_select_buffer_distance has to be between zero '
            'and infinity!')

        self._stn_bdist = float(station_select_buffer_distance)

        assert isinstance(interp_around_polys_flag, bool), (
            'interp_around_polys_flag not a boolean!')

        self._ipoly_flag = interp_around_polys_flag

        if self._ipoly_flag:
            assert isinstance(polygon_cell_buffer_distance, (float, int)), (
                'polygon_cell_buffer_distance should be a float or an int '
                'if interp_around_polys_flag is True!')

            assert 0 <= polygon_cell_buffer_distance < np.inf, (
                'polygon_cell_buffer_distance has to be in between zero and '
                'infinity!')

            self._cell_bdist = float(polygon_cell_buffer_distance)

        if self._vb:
            print('\n', '#' * 10, sep='')
            print('Set the following cell selection parameters:')
            print('Polygons shapefile:', str(self._poly_shp))
            print('interp_around_polys_flag:', self._ipoly_flag)
            print('Buffer distance to select a station:', self._stn_bdist)
            print('Buffer distance to select a cell:', self._cell_bdist)
            print('#' * 10)

        self._cell_sel_prms_set = True
        return

    def set_alignment_raster(self, align_raster):

        '''
        Set a raster to which our interpolation grid's minimum and maximum
        x and y coordinates are aligned.

        Parameters
        ----------
        align_raster : str or pathlib.Path
            Path to the alignment raster. It is up to the user to have
            a coordinate system that is consistent with the station
            coordinates. Cell size of the align_raster is used in the
            interpolation if this function is called.
        '''

        assert isinstance(align_raster, (str, Path)), (
            'align_raster has to be a string or a pathlib.Path object!')

        align_raster = Path(align_raster).absolute()

        assert align_raster.exists(), 'align_raster does not exist!'
        assert align_raster.is_file(), 'align_raster is not a file!'

        self._algn_ras = align_raster

        if self._vb:
            print('\n', '#' * 10, sep='')
            print('Alignment raster set to:', str(self._algn_ras))
            print('#' * 10)

        self._algn_ras_set_flag = True
        return

    def set_misc_settings(
            self,
            n_cpus=1,
            plot_figs_flag=False,
            cell_size=None,
            min_value_to_krige_thresh=None,
            min_cutoff_value=None,
            max_cutoff_value=None):

        '''
        Set some more parameters

        Default values defined in KrigingMain

        Parameters
        ----------
        n_cpus : int
            Number of threads to use for interpolation
        plot_figs_flag : bool
            Plot the interpolated grid as a pcolormesh if True.
        cell_size : int or float
            Cell size in the units that are consistent with the station
            coordinates system. It is used only if no drift or alignment
            rasters are specified.
        min_value_to_krige_thresh : int or float
            If all the stations have a value less than or equal to this
            for a given time step then all the cells are assigned the mean
            value of all the station values. This is good in cases like
            precipitation. Where interpolating the grid with really
            small values does not make sense.
        min_cutoff_value : None or int or float
            If not None, interpolated values below this are set
            equal to it. If not None it should be less than
            min_value_to_krige_thresh.
        max_cutoff_value : None or int or float
            If not None, interpolated values above this are set
            equal to it. If not None it should be greater than
            min_value_to_krige_thresh
        '''

        assert isinstance(n_cpus, int), 'n_cpus not an integer!'
        assert 0 < n_cpus, 'n_cpus less than one!'

        self._n_cpus = n_cpus

        if self._n_cpus > 1:
            self._mp_flag = True

        assert isinstance(plot_figs_flag, bool), (
            'plot_figs_flag not a boolean!')

        self._plot_figs_flag = plot_figs_flag

        if cell_size is not None:
            assert isinstance(cell_size, (int, float)), (
                'cell_size can only be a float or an int if not None!')

            assert 0 < cell_size < np.inf, (
                'cell_size not in between zero and infinity!')

            self._cell_size = float(cell_size)

        if min_value_to_krige_thresh is not None:
            assert isinstance(min_value_to_krige_thresh, (int, float)), (
                'min_value_to_krige_thresh can be a float or an int if '
                'not None!')

            assert -np.inf <= min_value_to_krige_thresh < np.inf, (
                'min_value_to_krige_thresh has to be in between -infinity '
                'and +infinity!')

            self._min_var_thr = float(min_value_to_krige_thresh)

        if min_cutoff_value is not None:
            assert isinstance(min_cutoff_value, (int, float)), (
                'min_cutoff_value can only be a float or an int if not None!')

            assert -np.inf <= min_cutoff_value <= np.inf, (
                'min_cutoff_value has to be in between -infinity '
                'and +infinity!')

            self._min_var_cut = float(min_cutoff_value)

        if max_cutoff_value is not None:
            assert isinstance(max_cutoff_value, (int, float)), (
                'max_cutoff_value can only be a float or an int if not None!')

            assert -np.inf <= max_cutoff_value <= np.inf, (
                'max_cutoff_value has to be in between -infinity '
                'and +infinity!')

            self._max_var_cut = float(max_cutoff_value)

        if (self._min_var_thr is not None) and (self._min_var_cut is not None):

            assert self._min_var_thr >= self._min_var_cut, (
                'min_value_to_krige_thresh has to be greater than or equal to'
                'min_cutoff_value!')

        if (self._min_var_cut is not None) and (self._max_var_cut is not None):

            assert self._min_var_cut < self._max_var_cut, (
                'max_cutoff_value has to be greater than min_cutoff_value!')

        if (self._min_var_thr is not None) and (self._max_var_cut is not None):

            assert self._min_var_thr < self._max_var_cut, (
                'min_value_to_krige_thresh has to be less than '
                'max_cutoff_value!')

        if self._vb:
            print('\n', '#' * 10, sep='')
            print('Set the following misc. settings:')
            print('n_cpus:', self._n_cpus)
            print('plot_figs_flag:', self._plot_figs_flag)
            print('cell_size:', self._cell_size)
            print('min_value_to_krige_thresh:', self._min_var_thr)
            print('min_cutoff_value:', self._min_var_cut)
            print('max_cutoff_value:', self._max_var_cut)
            print('#' * 10)

        self._misc_settings_set_flag = True
        return

    def _verify(self):

        assert self._data_set_flag
        assert self._out_dir_set_flag
        assert self._nc_set_flag
        assert self._time_prms_set_flag

        if (not self._vg_ser_set_flag) and self._vb:

            print('\n', '#' * 10, sep='')
            print(
                'No variograms series were set by the user. '
                'Only non-variogram type interpolations will be possible!')
            print('#' * 10)

        if (not self._cell_sel_prms_set) and self._vb:

            print('\n', '#' * 10, sep='')
            print('Cell selection parameters were not set by the user!')
            print('#' * 10)

        if (not self._algn_ras_set_flag) and self._vb:

            print('\n', '#' * 10, sep='')
            print('Alignment raster was not set by the user!')
            print('#' * 10)

        if (not self._misc_settings_set_flag) and self._vb:

            print('\n', '#' * 10, sep='')
            print('Using default misc. parameters!')
            print('#' * 10)

        self._data_vrfd_flag = True
        return

    __verify = _verify
