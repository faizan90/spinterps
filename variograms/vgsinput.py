'''
Created on 24 Nov 2018

@author: Faizan_TR
'''
from pathlib import Path

import numpy as np
import pandas as pd

from ..misc import get_dist_mat, print_sl, print_el


class VariogramsData:

    def __init__(self, verbose):

        self._vb = verbose

        self._data_df = None
        self._crds_df = None
        self._index_type = None
        self._stns_min_dist_thrsh = 0.0

        self._data_set_flag = False
        return

    def set_data(
            self,
            stns_time_ser_df,
            stns_crds_df,
            index_type='date',
            stns_min_dist_thresh=0):

        '''Set the data time series and coordinates dataframes.

        Parameters
        ----------
        stns_time_ser_df : pd.DataFrame with floating dtype
            Input time series dataframe of stations. Index should be
            pd.DatetimeIndex object and columns should be stations labels.

        stns_crds_df : pd.DataFrame with integral dtype
            A dataframe holding the coordinates of the stations. Index
            should be station labels and two columns 'X' and 'Y'
            representing the x- and y-coordinates.
        index_type : str
            The type of stns_time_ser_df.index. Currently, allowed
            ones are obj and date. When index_type is obj, then the entries
            in indicies of stns_time_ser_df and vgs_ser (set later) must be
            the same.
        stns_min_dist_thresh : int or float
            The minimum distance that each station must have with its
            neighbors. Distances that are zero or too small may result
            in SingularMatrix errors while kriging.
        '''

        if self._vb:
            print_sl()

            print('Setting data...')

        assert isinstance(stns_time_ser_df, pd.DataFrame), (
            'stns_time_ser_df has to be a pd.DataFrame object!')

        assert isinstance(stns_crds_df, pd.DataFrame), (
            'stns_crds_df has to be a pd.DataFrame object!')

        assert all(stns_time_ser_df.shape), 'Empty stns_time_ser_df!'

        assert all(stns_crds_df.shape), 'Empty stns_crds_df!'

        assert np.issubdtype(stns_time_ser_df.values.dtype, np.floating), (
            'dtype of stns_time_ser_df should be a subtype of np.floating!')

        assert np.issubdtype(stns_crds_df.values.dtype, np.number), (
            'dtype of stns_crds_df should be a subtype of np.number!')

        if self._index_type is not None:
            assert index_type == self._index_type, (
                'Given and previously set index_type do not match!')

        if index_type == 'date':
            assert isinstance(stns_time_ser_df.index, pd.DatetimeIndex), (
                'Data type of index of stns_time_ser_df does not match '
                'index_type!')

        elif index_type == 'obj':
            assert isinstance(stns_time_ser_df.index, object), (
                'Data type of index of stns_time_ser_df does not match '
                'index_type!')

            stns_time_ser_df.index = stns_time_ser_df.index.astype(str)

        else:
            raise AssertionError(
                'index_type can only be \'obj\' or \'date\'!')

        assert not np.any(np.isinf(stns_time_ser_df.values)), (
            'Infinity not allowed in stns_time_ser_df!')

        assert all([
            'X' in stns_crds_df.columns,
            'Y' in stns_crds_df.columns]), (
                'stns_crds_df has a missing \'X\' or \'Y\' column!')

        assert isinstance(stns_min_dist_thresh, (int, float)), (
            'stns_min_dist_thresh must be an integer or a float!')

        assert 0 <= stns_min_dist_thresh < np.inf, (
            'stns_min_dist_thresh must be inbetween 0 and infinity!')

        if self._vb:
            print(
                f'Original shape of stns_time_ser_df: '
                f'{stns_time_ser_df.shape}.')

            print(
                f'Original shape of stns_crds_df: '
                f'{stns_crds_df.shape}.')

        stns_time_ser_df.columns = map(str, stns_time_ser_df.columns)
        stns_crds_df.index = map(str, stns_crds_df.index)

        stns_crds_df = stns_crds_df.loc[:, ['X', 'Y']]
        stns_crds_df.dropna(axis=0, how='any', inplace=True)

        stns_time_ser_df.dropna(axis=1, how='all', inplace=True)

        if self._vb:
            print(
                f'stns_time_ser_df shape after dropping NaN '
                f'columns: {stns_time_ser_df.shape}')

        cmn_stns = stns_time_ser_df.columns.intersection(stns_crds_df.index)

        assert cmn_stns.shape[0] > 1, (
            'Less than 2 common stations between stns_time_ser_df and '
            'stns_crds_df!')

        stns_time_ser_df = stns_time_ser_df[cmn_stns]
        stns_crds_df = stns_crds_df.loc[cmn_stns]

        dists_mat = get_dist_mat(
            stns_crds_df['X'].values, stns_crds_df['Y'].values)

        if ((dists_mat <= stns_min_dist_thresh).sum() <=
            stns_crds_df.shape[0]):
                (f'Warning: Stations have neighbors with distances less '
                 f'than the threshold {stns_min_dist_thresh}!')

        if self._vb:
            print(
                f'{cmn_stns.shape[0]} out of {stns_time_ser_df.shape[1]} '
                f'stations are common in stns_time_ser_df and stns_crds_df.')

            print(
                f'Adjusted shape of stns_time_ser_df after taking common '
                f'stations: {stns_time_ser_df.shape}.')

            print(
                f'Adjusted shape of stns_crds_df after taking common '
                f'stations: {stns_crds_df.shape}.')

            print(
                f'Minimum station distance threshold: {stns_min_dist_thresh}')

        assert np.all(np.isfinite(stns_crds_df['X'].values)), (
            'Invalid values in x-coordinates!')

        assert np.all(np.isfinite(stns_crds_df['Y'].values)), (
            'Invalid values in y-coordinates!')

        self._data_df = stns_time_ser_df
        self._crds_df = stns_crds_df
        self._index_type = index_type
        self._stns_min_dist_thrsh = stns_min_dist_thresh

        if self._vb:
            print('Data set successfully!')

            print_el()

        self._data_set_flag = True
        return


class VariogramsInput(VariogramsData):

    def __init__(self, verbose=True):

        VariogramsData.__init__(self, verbose)

        self._vgs_prms_set_flag = False
        self._misc_settings_set_flag = False
        self._output_settings_set_flag = False
        self._inputs_vrfd_flag = False
        return

    def set_vg_fitting_parameters(
            self,
            maximum_distance_ratio,
            n_vgs_perms,
            nugget_vg,
            n_opt_grid_points,
            vg_names,
            n_best_vgs,
            nk=50):

        '''Set some of the inputs to the Variogram class

        Parameters
        ----------
        maximum_distance_ratio : int
            Maximum allowed distance between pairs while computing variogram
            as a ratio of the maximum possible distance between any pair.
        n_vgs_perms : list-like
            A list-like object having integers that show the number of
            variogram types that are used to make combinations.
        nugget_vg : str
            Type of the nugget variogram. It can be any of the allowed ones.
        n_opt_grid_points : int
            The number of grid points between zero and variance (and zero
            and range) to choose as initial points for optimization. The
            more it is the slower the optimization and higher the chance of
            finding the global optimum.
        vg_names : list-like
            A list-like object containing the string lables of variograms
            that can tried for fitting to the empirical variogram.
        n_best_vgs : int
            The number of maximum variogram strings to store in the output
            variogram series. These will be in ascending value of the
            objective function value for each step i.e. the best is the first
            one.
        nk : int
            Number of values per bin. Should be greater than zero and less
            than the maximum number of possible pairs.
        '''

        assert isinstance(maximum_distance_ratio, float), (
            'maximum_distance_ratio not a floating value!')

        assert 0 < maximum_distance_ratio <= 1, (
            'maximum_distance_ratio must be inbetween zero and one!')

        assert hasattr(n_vgs_perms, '__iter__'), (
            'n_vgs_perms not an iterable!')

        assert all([isinstance(i, int) for i in n_vgs_perms]), (
            'n_vgs_perms can only have integer values inside!')

        assert all([i > 0 for i in n_vgs_perms]), (
            'n_vgs_perms have values less than or equal to zero!')

        assert np.unique(n_vgs_perms).size == len(n_vgs_perms), (
            'Non-unique values in n_vgs_perms!')

        assert isinstance(nugget_vg, str), 'nugget_vg not an str object!'

        assert isinstance(n_opt_grid_points, int), (
            'n_opt_grid_points not an integer!')

        assert 0 < n_opt_grid_points < np.inf, (
            'n_opt_grid_points should be greater than zero and less '
            'than infinity!')

        assert hasattr(vg_names, '__iter__'), 'vg_names not an iterable!'

        assert all([isinstance(i, str) for i in vg_names]), (
            'Only str objects allowed inside vg_names!')

        assert all([len(i) for i in vg_names]), 'Empty strings in vg_names!'

        assert isinstance(n_best_vgs, int), 'n_best_vgs not an integer!'

        assert 0 < n_best_vgs < np.inf, (
            'n_best_vgs should be greater than zero and less than infinity!')

        assert isinstance(nk, int), 'nk not an integer!'
        assert nk > 0, 'Invalid nk!'

        self._mdr = maximum_distance_ratio
        self._n_vgs_perms = n_vgs_perms
        self._nug_vg = nugget_vg
        self._n_gps = n_opt_grid_points
        self._vg_names = vg_names
        self._n_best_vgs = n_best_vgs
        self._nk = nk

        if self._vb:
            print('Variogram parameters set successfully!')

        self._vgs_prms_set_flag = True
        return

    def set_misc_settings(
            self,
            n_cpus,
            min_valid_stations_per_step=1):

        '''Set some misc. settings

        Parameters
        ----------
        n_cpus : int
            The number of threads to use while fitting the variograms.

        min_valid_stations_per_step : int
            The number of minimum valid stations that a given time step
            should have in order to fit variogram(s) to that step's data.
        '''

        assert isinstance(n_cpus, int), 'n_cpus not an integer!'
        assert 0 < n_cpus, 'n_cpus must be greater than zero!'

        self._n_cpus = n_cpus

        assert isinstance(min_valid_stations_per_step, int), (
            'min_valid_stations_per_step not an integer!')

        assert 0 < min_valid_stations_per_step, (
            'min_valid_stations_per_step must be greater than zero!')

        self._min_vld_stns = min_valid_stations_per_step

        if self._vb:
            print('Misc. settings set successfully!')

        self._misc_settings_set_flag = True
        return

    def set_output_settings(
            self,
            out_dir,
            out_figs_flag=False):

        '''Set the outputs settings

        out_dir : str of Path-like
            Path to output directory where all the outputs are stored.
        out_figs_flag : bool
            Whether to save the fitted varigrams' figures.
        '''

        assert isinstance(out_dir, (str, Path)), (
            'out_dir not an str or a Path object!')

        out_dir = Path(out_dir)

        assert out_dir.parents[0].exists(), (
            'Parent directory of out_dir does not exist!')

        assert isinstance(out_figs_flag, bool), 'out_figs_flag not a boolean!'

        self._out_dir = out_dir

        if out_figs_flag:
            self._out_figs_path = out_dir / 'vg_figs'

        else:
            self._out_figs_path = None

        if self._vb:
            print('Output settings set successfully!')

        self._output_settings_set_flag = True
        return

    def verify(self):

        assert self._data_set_flag, 'Data not set!'
        assert self._vgs_prms_set_flag, 'Variogram parameters not set!'
        assert self._misc_settings_set_flag, 'Misc. settings not set!'
        assert self._output_settings_set_flag, 'Output directory not set!'

        assert np.any(self._data_df.count(axis=1) >= self._min_vld_stns), (
            'No time steps have available station greater then the '
            'minimum threshold!')

        self._inputs_vrfd_flag = True
        return

    __verify = verify
