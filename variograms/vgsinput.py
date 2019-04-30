'''
Created on 24 Nov 2018

@author: Faizan_TR
'''
from pathlib import Path

import numpy as np
import pandas as pd


class VariogramsData:

    def __init__(self, verbose):

        self._vb = verbose

        self._index_type = None

        self._data_set_flag = False
        return

    def set_data(self, stns_time_ser_df, stns_crds_df, index_type='date'):

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
            The datatype of stns_time_ser_df.index. Currently, allowed
            ones are obj and date.
        '''

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

        else:
            raise AssertionError(
                'index_type can only be \'obj\' or \'date\'!')

        assert all([
            'X' in stns_crds_df.columns,
            'Y' in stns_crds_df.columns]), (
                'stns_crds_df has a missing \'X\' or \'Y\' column!')

        if self._vb:
            print('\n', '#' * 10, sep='')
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

        assert cmn_stns.shape[0], (
            'No common stations between stns_time_ser_df and stns_crds_df!')

        stns_time_ser_df = stns_time_ser_df[cmn_stns]
        stns_crds_df = stns_crds_df.loc[cmn_stns]

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

        assert not np.isnan(stns_crds_df['X'].values).sum(), (
            'NaN values in x-coordinates!')

        assert not np.isnan(stns_crds_df['Y'].values).sum(), (
            'NaN values in y-coordinates!')

        self._data_df = stns_time_ser_df
        self._crds_df = stns_crds_df
        self._index_type = index_type

        if self._vb:
            print('Data set successfully!')
            print('#' * 10)

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
            n_best_vgs):

        '''Set some of the inputs to the Variogram class here'''

        assert isinstance(maximum_distance_ratio, float)
        assert 0 < maximum_distance_ratio <= 1

        assert hasattr(n_vgs_perms, '__iter__')
        assert all([isinstance(i, int) for i in n_vgs_perms])
        assert all([i > 0 for i in n_vgs_perms])

        assert isinstance(nugget_vg, str)

        assert isinstance(n_opt_grid_points, int)
        assert 0 < n_opt_grid_points <= 100

        assert hasattr(vg_names, '__iter__')
        assert all([isinstance(i, str) for i in vg_names])
        assert all([len(i) for i in vg_names])

        assert isinstance(n_best_vgs, int)
        assert 0 < n_best_vgs < 100

        self._mdr = maximum_distance_ratio
        self._n_vgs_perms = n_vgs_perms
        self._nug_vg = nugget_vg
        self._n_gps = n_opt_grid_points
        self._vg_names = vg_names
        self._n_best_vgs = n_best_vgs

        if self._vb:
            print('Variogram parameters set successfully!')

        self._vgs_prms_set_flag = True
        return

    def set_misc_settings(
            self,
            n_cpus,
            min_valid_stations_per_step=1):

        assert isinstance(n_cpus, int)
        assert 0 < n_cpus

        self._n_cpus = n_cpus

        assert isinstance(min_valid_stations_per_step, int)
        assert 0 < min_valid_stations_per_step

        self._min_vld_stns = min_valid_stations_per_step

        if self._vb:
            print('Misc. settings set successfully!')

        self._misc_settings_set_flag = True
        return

    def set_output_settings(
            self,
            out_dir,
            out_figs_flag=False):

        assert isinstance(out_dir, (str, Path))

        out_dir = Path(out_dir)

        assert out_dir.parents[0].exists(), (
            'Parent directory of out_dir does not exist!')

        assert isinstance(out_figs_flag, bool)

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

        assert np.any(self._data_df.count(axis=1) >= self._min_vld_stns)

        self._inputs_vrfd_flag = True
        return

    __verify = verify
