'''
Created on Nov 25, 2018

@author: Faizan
'''

import os
import timeit
from math import ceil
from pathlib import Path
from multiprocessing import Pool, Manager, Lock

import numpy as np
import psutil as ps

from .steps import SpInterpSteps as SIS
from .data import SpInterpData as SID
from .prepare import SpInterpPrepare as SIP
from ..misc import ret_mp_idxs, get_current_proc_size

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)


class SpInterpMain(SID, SIP):

    def __init__(self, verbose=True):

        SID.__init__(self, verbose)
        SIP.__init__(self)

        self._drft_rass = None
        self._n_drft_rass = None

        self._idw_exps = None
        self._n_idw_exps = None

        self._ork_flag = False
        self._spk_flag = False
        self._edk_flag = False
        self._idw_flag = False

        self._max_mem_usage_ratio = 0.8

        self._main_vrfd_flag = False
        return

    def verify(self):

        SID._SpInterpData__verify(self)

        self._prepare()

        assert self._prpd_flag

        self._main_vrfd_flag = True
        return

    def interpolate(self):

        assert self._main_vrfd_flag

        interp_steps_idxs = self._get_thread_steps_idxs()

        if self._mp_flag:
            mp_pool = Pool(self._n_cpus)

            spi_map = mp_pool.map  # has to be a blocking one

            self._lock = Manager().Lock()

        else:
            spi_map = map

            self._lock = Lock()  #  the manager might slow things down

        interp_steps_cls = SIS(self)

        if self._vb:
            print('\n', '#' * 10, sep='')
            print('Started interpolation...')
            print('Parent process ID:', os.getpid())
            print('Interpolation step indices:\n', interp_steps_idxs, sep='')

        for interp_arg in self._interp_args:
            ivar_name = interp_arg[2]

            if self._vb:
                interp_beg_time = timeit.default_timer()

                t_passes = int((interp_steps_idxs.shape[0] - 1) / self._n_cpus)

                print('\n', '#' * 10, sep='')
                print('Current interpolation method:', ivar_name)
                print(f'Using {t_passes} pass(es) to interpolate.')
                print('\n')

            for j in range(0, interp_steps_idxs.shape[0] - 1, self._n_cpus):
                max_rng = min(self._n_cpus, interp_steps_idxs[j:].shape[0])
                assert max_rng > 0

                if self._vb:
                    print(
                        f'Pass number {(j // max_rng) + 1} out of '
                        f'{t_passes} pass(es).')

                    print(f'Using {max_rng} worker(s) for the current pass.')

                interp_gen = (
                    self._get_interp_gen_data(
                        interp_steps_idxs[j + i],
                        interp_steps_idxs[j + i + 1],
                        interp_arg,
                        max_rng)

                    for i in range(max_rng))

                if self._mp_flag:
                    if self._vb:
                        print('Sending jobs to all workers...')

                maped_obj = spi_map(
                    interp_steps_cls.get_interp_flds, interp_gen)

                if not self._mp_flag:
                    list(maped_obj)

                if self._vb:
                    print('Done writing for the current pass.')
                    print('\n')

            if self._vb:
                tot_beg_time = timeit.default_timer() - interp_beg_time

                print(
                    f'Done with the interpolation method: {ivar_name} in '
                    f'{tot_beg_time:0.3f} seconds.',)

                print('#' * 10)

        if self._vb:
            print('\n')
            print('Done interpolating!')
            print('#' * 10)
        return

    def _get_interp_gen_data(
            self, beg_idx, end_idx, interp_arg, max_rng):

        if interp_arg[0] in ['OK', 'SK', 'EDK']:
            vgs_ser = self._vgs_ser

        else:
            vgs_ser = None

        if interp_arg[0] == 'EDK':
            drft_arrs = self._drft_arrs
            stns_drft_df = self._stns_drft_df

        else:
            drft_arrs, stns_drft_df = 2 * [None]

        return (
            self._data_df.iloc[beg_idx:end_idx],
            beg_idx,
            end_idx,
            max_rng,
            interp_arg,
            self._lock,
            drft_arrs,
            stns_drft_df,
            vgs_ser)

    def _get_thread_steps_idxs(self):

        '''
        All sizes in bytes
        '''
        megabytes = 1024 ** 2

        bytes_per_number = 4

        # actual interpreter size will probably be much smaller than this
        interpreter_size = get_current_proc_size()

        assert 0 < self._max_mem_usage_ratio <= 1

        tot_avail_mem = int(
            ps.virtual_memory().free * self._max_mem_usage_ratio)

        avail_threads_mem = tot_avail_mem - (self._n_cpus * interpreter_size)

        assert avail_threads_mem > 0, 'Memory too little or too many threads!'

        max_mem_per_thread = avail_threads_mem // self._n_cpus

        assert max_mem_per_thread > 0, 'Memory too little or too many threads!'

        tot_interp_arr_size = bytes_per_number * (
            self._data_df.shape[0] *
            self._interp_crds_orig_shape[0] *
            self._interp_crds_orig_shape[1])

        bytes_per_step = (
            bytes_per_number * np.prod(self._interp_crds_orig_shape))

        assert max_mem_per_thread > bytes_per_step, (
                'Interpolation grid too fine or too many threads!')

        if tot_interp_arr_size < avail_threads_mem:
            step_idxs = ret_mp_idxs(self._data_df.shape[0], self._n_cpus)

        else:
            max_concurrent_steps = avail_threads_mem // (
                bytes_per_number *
                self._interp_crds_orig_shape[0] *
                self._interp_crds_orig_shape[1])

            steps_scale_cnst = ceil(
                self._data_df.shape[0] / max_concurrent_steps)

            assert steps_scale_cnst > 1, 'This should not happen!'

            step_idxs = ret_mp_idxs(
                self._data_df.shape[0], self._n_cpus * steps_scale_cnst)

        steps_per_thread = step_idxs[1] - step_idxs[0]

        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Memory management stuff...')

            print(f'Main interpreter size: {interpreter_size // megabytes} MB')

            print(
                f'Total memory available for use: '
                f'{tot_avail_mem // megabytes} MB')

            print(
                f'Memory available per thread: '
                f'{max_mem_per_thread // megabytes} MB')

            print(
                f'Total size of the interpolated array: '
                f'{tot_interp_arr_size / megabytes:0.6f} MB')

            print(
                f'Size per step: {bytes_per_step / megabytes:0.6f} MB')

            print(f'No. of steps interpolated per thread: {steps_per_thread}')

            print('#' * 10)

        return step_idxs

    def turn_ordinary_kriging_on(self):

        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set ordinary kriging flag to True.')

            print('#' * 10)

        self._ork_flag = True
        return

    def turn_ordinary_kriging_off(self):

        assert self._ork_flag
        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set ordinary kriging flag to False.')

            print('#' * 10)

        self._ork_flag = False
        return

    def turn_simple_kriging_on(self):

        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set simple kriging flag to True.')

            print('#' * 10)

        self._spk_flag = True
        return

    def turn_simple_kriging_off(self):

        assert self._spk_flag

        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set simple kriging flag to False.')

            print('#' * 10)

        self._spk_flag = False
        return

    def turn_external_drift_kriging_on(self, drift_rasters):

        '''
        Signal to do external drift kriging.

        Parameters
        ----------
        drift_rasters : iterable of paths
            An iterable (list or tuple) holding paths to the rasters that
            should be used as drifts. All rasters should have the same spatial
            properties and coordinate systems that are compatible with the
            station coordinates.
        '''

        assert hasattr(drift_rasters, '__iter__')

        self._drft_rass = []

        for drift_raster in drift_rasters:
            assert isinstance(drift_raster, (str, Path)), (
                'Supplied drift raster path is not a string or a '
                'pathlib.Path object!')

            drift_raster_path = Path(drift_raster).absolute()

            assert drift_raster_path.exists(), (
                'Supplied drift raster path does not point to a file!')

            assert drift_raster_path.is_file(), (
                'Supplied drift raster path does not point to a file!')

            self._drft_rass.append(drift_raster_path)

        self._drft_rass = tuple(self._drft_rass)

        self._n_drft_rass = len(self._drft_rass)

        assert self._n_drft_rass, 'Zero drift rasters were supplied!'

        if self._vb:
            print('\n', '#' * 10, sep='')

            print(
                'Set external drift kriging flag to True with the '
                'following drift rasters:')

            for drift_raster in self._drft_rass:
                print(str(drift_raster))

            print('#' * 10)

        self._edk_flag = True
        return

    def turn_external_drift_kriging_off(self):

        assert self._edk_flag

        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set external_drift kriging flag to False.')

            print('#' * 10)

        self._drft_rass = None
        self._n_drft_rass = None

        self._edk_flag = False
        return

    def turn_inverse_distance_weighting_on(self, idw_exps):

        '''
        Signal to do inverse distance weighting.

        Parameters
        ----------
        idw_exps : iterable of ints or floats
            The exponents to use in inverse distance weighting.
            Seperate interpolation grids are computed for each exponent.
        '''

        assert hasattr(idw_exps, '__iter__')

        self._idw_exps = []

        for idw_exp in idw_exps:
            assert isinstance(idw_exp, (int, float)), (
                'IDW exponent not a float or an int!')

            self._idw_exps.append(float(idw_exp))

        self._idw_exps = tuple(self._idw_exps)

        self._n_idw_exps = len(self._idw_exps)

        assert self._n_idw_exps, 'Zero IDW exponents given!'

        if self._vb:
            print('\n', '#' * 10, sep='')

            print(
                'Set inverse distance weighting flag to True with the '
                'following exponents:')

            print(self._idw_exps)

            print('#' * 10)

        self._idw_flag = True
        return

    def turn_inverse_distance_weighting_off(self):

        assert self._idw_flag

        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set inverse distance weighting flag to False.')

            print('#' * 10)

        self._idw_exps = None
        self._n_idw_exps = None

        self._idw_flag = False
        return
