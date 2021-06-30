'''
Created on Nov 25, 2018

@author: Faizan
'''

from pathlib import Path

import numpy as np
import pandas as pd

from ..variograms.vgsinput import VariogramsData as VD
from ..misc import print_sl, print_el


class SpInterpData(VD):

    def __init__(self, verbose=True):

        VD.__init__(self, verbose)

        self._vgs_ser = None
        self._out_dir = None

        self._n_cpus = 1
        self._mp_flag = False
        self._cell_size = None

        self._plot_figs_flag = False

        self._nc_out = None
        self._nc_vunits = None
        self._nc_vlab = None
        self._nc_tunits = None
        self._nc_tcldr = None
        self._nc_tlab = None
        self._nc_xlab = None
        self._nc_ylab = None

        self._tbeg = None
        self._tend = None
        self._tfreq = None

        self._algn_ras = None

        self._poly_shp = None
        self._ipoly_flag = False
        self._cell_bdist = 0

        self._min_var_thr = None
        self._min_var_cut = None
        self._max_var_cut = None
        self._max_steps_per_chunk = None
        self._min_vg_val = None

        self._neb_sel_mthds = ('all', 'nrst', 'pie')

        self._neb_sel_mthd = 'all'
        self._n_nebs = None
        self._n_pies = None

        self._vg_ser_set_flag = False
        self._out_dir_set_flag = False
        self._nc_set_flag = False
        self._time_prms_set_flag = False
        self._cell_sel_prms_set = False
        self._algn_ras_set_flag = False
        self._neb_sel_mthd_set_flag = False
        self._misc_settings_set_flag = False
        self._data_vrfd_flag = False
        return

    def set_vgs_ser(self, vgs_ser, index_type='date'):

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

        if self._index_type is not None:
            assert index_type == self._index_type, (
                'Given and previously set index_type do not match!')

        if index_type == 'date':
            assert isinstance(vgs_ser.index, pd.DatetimeIndex), (
                'Data type of index of vg_ser does not match index_type!')

        elif index_type == 'obj':
            assert isinstance(vgs_ser.index, object), (
                'Data type of index of vg_ser does not match index_type!')

            vgs_ser.index = vgs_ser.index.astype(str)

        else:
            raise AssertionError(
                'index_type can only be \'obj\' or \'date\'!')

        assert all(vgs_ser.shape), 'Empty vgs_ser!'

        self._vgs_ser = pd.Series(vgs_ser, dtype=str)

        miss_steps_ctr = (self._vgs_ser.values == 'nan').sum()

        if self._vb:
            print_sl()
            print(
                f'vgs_ser set with a shape of: {self._vgs_ser.shape}. '
                f'Out of which {miss_steps_ctr} have no variograms.')
            print_el()

        self._vg_ser_set_flag = True
        return

    def set_out_dir(self, out_dir):

        '''Set the main outputs directory'''

        assert isinstance(out_dir, (str, Path)), (
            'out_dir can only be a string or pathlib.Path object!')

        out_dir = Path(out_dir).absolute()

        assert out_dir.parents[0].exists(), (
            'Parent directory of out_dir does not exist!')

        self._out_dir = Path(out_dir)

        if self._vb:
            print_sl()
            print('Outputs directory set to:', str(self._out_dir))
            print_el()

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
        time_units : str or None
            Starting date that is used as a reference by the netCDF4 module
            to convert datetime objects to integers and the other way around.
            It generally looks like days since 1900-00-00. Read the netCDF4
            documentation to get a better idea. Can be None to represent
            non-datetime like index.
        time_calendar : str or None
            A calendar name that is compatible with the netCDF4 library.
            It is typically gregorian or something. Read the netCDF4
            documentation to get a better idea. Can be None to represent
            non-datetime like index.
        '''

        assert isinstance(out_file_name, str), (
            'out_file_name file not a string!')

        assert out_file_name, 'Empty out_file_name!'

        self._nc_out = out_file_name

        assert isinstance(var_units, str), 'var_units not a string!'

        self._nc_vunits = var_units

        assert isinstance(var_label, str), 'var_label not a string!'

        self._nc_vlab = var_label

        if time_units is not None:
            assert isinstance(time_units, str), 'time_units not a string!'
            assert time_units, 'Empty time_units string!'

        self._nc_tunits = time_units

        if time_calendar is not None:
            assert isinstance(time_calendar, str), (
                'time_calendar not a string!')

            assert time_calendar, 'Empty time_calendar string!'

        self._nc_tcldr = time_calendar

        self._nc_tlab = 'time'
        self._nc_xlab = 'X'
        self._nc_ylab = 'Y'

        if self._vb:
            print_sl()
            print('Set the following output netCDF4 file parameters:')
            print(f'File name: {self._nc_out}')
            print(f'Variable units: {self._nc_vunits}')
            print(f'Variable label: {self._nc_vlab}')
            print(f'Time units: {self._nc_tunits}')
            print(f'Time calendar: {self._nc_tcldr}')
            print(f'Time label: {self._nc_tlab}')
            print(f'X-coordinates label: {self._nc_xlab}')
            print(f'Y-coordinates label: {self._nc_ylab}')
            print_el()

        self._nc_set_flag = True
        return

    def set_interp_time_parameters(
            self,
            beg_time,
            end_time,
            time_freq,
            time_fmt):

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
            if they are string objects. None means the index is a
            non-datetime like. In such a case no checks are done for the
            previous parameters and everything is set to None.
        '''

        if time_fmt is not None:
            assert isinstance(time_fmt, str), 'time_fmt not a string!'

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
            print_sl()
            print('Set the following time interpolation parameters:')
            print('Begin time:', self._tbeg)
            print('End time:', self._tend)
            print('Time frequency:', self._tfreq)
            print_el()

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
            nearest polygon, in order for it to be considered as a neighbor
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
            interp_around_polys_flag is True then it should be a float or
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
            print_sl()
            print('Set the following cell selection parameters:')
            print('Polygons shapefile:', str(self._poly_shp))
            print('interp_around_polys_flag:', self._ipoly_flag)
            print('Buffer distance to select a station:', self._stn_bdist)
            print('Buffer distance to select a cell:', self._cell_bdist)
            print_el()

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
            print_sl()
            print('Alignment raster set to:', str(self._algn_ras))
            print_el()

        self._algn_ras_set_flag = True
        return

    def set_neighbor_selection_method(
            self,
            selection_method,
            n_neighbors=None,
            n_pies=None):

        '''
        Set the type of method to select neighboring points while
        interpolating.

        Parameters
        ----------
        selection_method : str
            The type of method to use while selecting neighbors.
            Three are described below.
            Default is all, even if this method is not called.
            all : take all present neighbors values to interpolate at
            a given point at a given time step.
            nrst : take exactly n_neighbors nearest neighbors while
            interpolating. An error is raised if at any given step
            available neighbors are less than this. Slow as compared
            to the all method. But will probably outperform it if
            neighbors are too many e.g. 500.
            pie : same as the nrst method but have point sampled
            almost uniformly from n_pies parts of a circle. Almost
            because if n_neighbors is not a multiple of n_pies. In
            such cases the nearest ones are taken. Slow as compared
            to the nrst method. Recommend using it while
            interpolating massive grids with a lot of neighbors.
        n_neighbors : int
            The numbers of neighbors to take at a given time step
            if the selection method is nrst or pie. Error is raised
            if at any time step the given number of neighbors with
            valid values is less than it. Should be greater than or
            equal to n_pies (if specified). Has to be specified if
            selection method is nrst or pie.
        n_pies : int
            If selection method is nrst of pie, how many pies to
            divide a circle in to take a uniform number of neighbors
            from each pie. This can be due to non uniform distribution
            of neighbors in plane. Has to be specified if selection
            method is pie. Always less than or equal to n_neighbors.
        '''

        assert isinstance(selection_method, str), (
            'selection_method not a string!')

        assert selection_method in self._neb_sel_mthds, (
            f'selection_method can only be one of the '
            f'{self._neb_sel_mthds} methods!')

        if (selection_method == 'nrst') or (selection_method == 'pie'):
            assert isinstance(n_neighbors, int), (
                'n_neighbors not an integer!')

            assert n_neighbors > 0, 'n_neighbors less than or equal to zero!'

        elif (selection_method == 'all'):
            pass

        else:
            raise NotImplementedError

        if selection_method == 'pie':
            assert isinstance(n_pies, int), 'n_pies not an integer!'

            assert 0 < n_pies <= n_neighbors, (
                'n_pies can be in between zero and n_neighbors only!')

        elif (selection_method == 'all') or (selection_method == 'nrst'):
            pass

        else:
            raise NotImplementedError

        self._neb_sel_mthd = selection_method

        if selection_method in ('nrst', 'pie'):
            self._n_nebs = n_neighbors

        if selection_method == 'pie':
            self._n_pies = n_pies

        if self._vb:
            print_sl()
            print('Set the following neighbor selection parameters:')
            print(f'Neighbor selection method: {self._neb_sel_mthd}')
            print(f'Number of neighbors to use: {self._n_nebs}')
            print(f'Number of pies: {self._n_pies}')
            print_el()

        self._neb_sel_mthd_set_flag = True
        return

    def set_misc_settings(
            self,
            n_cpus=1,
            plot_figs_flag=False,
            cell_size=None,
            min_value_to_interp_thresh=-np.inf,
            min_cutoff_value=None,
            max_cutoff_value=None,
            max_steps_per_chunk=None,
            min_vg_val=0):

        '''
        Set some more parameters

        Default values defined in SpInterpMain

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
        min_value_to_interp_thresh : int or float
            If all the stations have a value less than or equal to this
            for a given time step then all the cells are assigned the mean
            value of all the station values. This is good in cases like
            precipitation. Where interpolating the grid with really
            small values does not make sense.
        min_cutoff_value : None or int or float
            If not None, interpolated values below this are set
            equal to it. If not None it should be less than
            min_value_to_interp_thresh.
        max_cutoff_value : None or int or float
            If not None, interpolated values above this are set
            equal to it. If not None it should be greater than
            min_value_to_interp_thresh
        max_steps_per_chunk : None or int
            Maximum number of steps that can be interpolated per thread.
            Final number of steps is the minimum based on available memory
            and max_steps_per_chunk.
        min_vg_val : float
            Variogram values below or equal to this are set to zero.
            Should be >= 0 and < infinity.
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

        assert isinstance(min_value_to_interp_thresh, (int, float)), (
            'min_value_to_interp_thresh can be a float or an int if '
            'not None!')

        assert -np.inf <= min_value_to_interp_thresh < np.inf, (
            'min_value_to_interp_thresh has to be in between -infinity '
            'and +infinity!')

        self._min_var_thr = float(min_value_to_interp_thresh)

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

        if max_steps_per_chunk is not None:
            assert isinstance(max_steps_per_chunk, int), (
                'max_steps_per_chunk can only be an integer if not None!')

            assert max_steps_per_chunk > 0, (
                'max_steps_per_chunk should be greeater than zero!')

            self._max_steps_per_chunk = max_steps_per_chunk

        if (self._min_var_thr is not None) and (self._min_var_cut is not None):

            assert self._min_var_thr >= self._min_var_cut, (
                'min_value_to_interp_thresh has to be greater than or equal to '
                'min_cutoff_value!')

        if (self._min_var_cut is not None) and (self._max_var_cut is not None):

            assert self._min_var_cut < self._max_var_cut, (
                'max_cutoff_value has to be greater than min_cutoff_value!')

        if (self._min_var_thr is not None) and (self._max_var_cut is not None):

            assert self._min_var_thr < self._max_var_cut, (
                'min_value_to_interp_thresh has to be less than '
                'max_cutoff_value!')

        assert isinstance(min_vg_val, float), 'min_vg_val must be a float!'

        assert 0 <= min_vg_val < np.inf, 'Invalid value of min_vg_vals!'

        self._min_vg_val = min_vg_val

        if self._vb:
            print_sl()
            print('Set the following misc. settings:')
            print('n_cpus:', self._n_cpus)
            print('plot_figs_flag:', self._plot_figs_flag)
            print('cell_size:', self._cell_size)
            print('min_value_to_interp_thresh:', self._min_var_thr)
            print('min_cutoff_value:', self._min_var_cut)
            print('max_cutoff_value:', self._max_var_cut)
            print('max_steps_per_chunk:', self._max_steps_per_chunk)
            print('min_vg_val:', self._min_vg_val)
            print_el()

        self._misc_settings_set_flag = True
        return

    def _verify(self):

        assert self._data_set_flag, 'Call the set_data method first!'

        assert self._out_dir_set_flag, 'Call the set_out_dir method first!'

        assert self._nc_set_flag, (
            'Call the set_netcdf4_parameters method first!')

        assert self._time_prms_set_flag, (
            'Call the set_interp_time_parameters method first!')

        assert self._neb_sel_mthd_set_flag, (
            'Call set_neighbor_selection_method method first!')

        if self._index_type == 'date':
            pass

        elif self._index_type == 'obj':
            idx_union = self._data_df.index.difference(self._vgs_ser.index)

            assert not idx_union.size, (
                'For object type index, data and variograms must have the '
                'index!')

        if self._vb:
            if not self._vg_ser_set_flag:

                print_sl()
                print(
                    'No variograms series were set by the user. '
                    'Only non-variogram type interpolations will be possible!')
                print_el()

            if not self._cell_sel_prms_set:

                print_sl()
                print('Cell selection parameters were not set by the user!')
                print_el()

            if not self._algn_ras_set_flag:

                print_sl()
                print('Alignment raster was not set by the user!')
                print_el()

            if not self._misc_settings_set_flag:

                print_sl()
                print('Using default misc. parameters!')
                print_el()

        self._data_vrfd_flag = True
        return

    __verify = _verify
