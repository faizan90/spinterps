'''
Created on Nov 25, 2018

@author: Faizan
'''

import os
import timeit
from pathlib import Path
from functools import partial
from multiprocessing import Pool, Manager, Lock

import numpy as np
import psutil as ps

from .data import SpInterpData as SID
from .steps import SpInterpSteps as SIS
from .prepare import SpInterpPrepare as SIP
from ..misc import ret_mp_idxs, get_current_proc_size, print_sl, print_el

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

        self._max_mem_usage_ratio = 0.80

        self._main_vrfd_flag = False
        return

    def verify(self):

        SID._SpInterpData__verify(self)

        self._prepare()

        assert self._prpd_flag, 'Preparing data for interpolation failed!'

        self._main_vrfd_flag = True
        return

    def interpolate(self):

        assert self._main_vrfd_flag, 'Call the verify method first!'

        interp_steps_idxs = self._get_thread_steps_idxs()

        if self._mp_flag:
            mp_pool = Pool(self._n_cpus)

            # has to be a blocking one
            spi_map = partial(mp_pool.map, chunksize=1)

            self._lock = Manager().Lock()

        else:
            spi_map = map

            self._lock = Lock()  #  the manager might slow things down

        interp_steps_cls = SIS(self)

        if self._vb:
            print_sl()
            print('Started interpolation...')
            print('Parent process ID:', os.getpid())
            print('Interpolation step indices:\n', interp_steps_idxs, sep='')

        ivar_names = [interp_arg[2] for interp_arg in self._interp_args]

        if self._vb:
            interp_beg_time = timeit.default_timer()

            t_passes = int((interp_steps_idxs.shape[0] - 1) / self._n_cpus)

            print_sl()
            print('Current interpolation methods:', ivar_names)
            print(f'Using {t_passes} pass(es) to interpolate.')
            print('\n')

        interp_gen = (
            self._get_interp_gen_data(
                interp_steps_idxs[i],
                interp_steps_idxs[i + 1],
                interp_steps_idxs.size - 1)

            for i in range(interp_steps_idxs.size - 1))

        maped_obj = spi_map(
            interp_steps_cls.interpolate_subset, interp_gen)

        if not self._mp_flag:
            list(maped_obj)

        else:
            mp_pool.close()
            mp_pool.join()
            mp_pool = None

        if self._vb:
            tot_beg_time = timeit.default_timer() - interp_beg_time

            print(
                f'Done with the interpolation methods: {ivar_names} in '
                f'{tot_beg_time:0.3f} seconds.',)

            print_el()
        return

    def turn_ordinary_kriging_on(self):

        if self._vb:
            print_sl()

            print('Set ordinary kriging flag to True.')

            print_el()

        self._ork_flag = True
        return

    def turn_ordinary_kriging_off(self):

        assert self._ork_flag
        if self._vb:
            print_sl()

            print('Set ordinary kriging flag to False.')

            print_el()

        self._ork_flag = False
        return

    def turn_simple_kriging_on(self):

        if self._vb:
            print_sl()

            print('Set simple kriging flag to True.')

            print_el()

        self._spk_flag = True
        return

    def turn_simple_kriging_off(self):

        assert self._spk_flag

        if self._vb:
            print_sl()

            print('Set simple kriging flag to False.')

            print_el()

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
            print_sl()

            print(
                'Set external drift kriging flag to True with the '
                'following drift rasters:')

            for drift_raster in self._drft_rass:
                print(str(drift_raster))

            print_el()

        self._edk_flag = True
        return

    def turn_external_drift_kriging_off(self):

        assert self._edk_flag

        if self._vb:
            print_sl()

            print('Set external_drift kriging flag to False.')

            print_el()

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
            print_sl()

            print(
                'Set inverse distance weighting flag to True with the '
                'following exponents:')

            print(self._idw_exps)

            print_el()

        self._idw_flag = True
        return

    def turn_inverse_distance_weighting_off(self):

        assert self._idw_flag

        if self._vb:
            print_sl()

            print('Set inverse distance weighting flag to False.')

            print_el()

        self._idw_exps = None
        self._n_idw_exps = None

        self._idw_flag = False
        return

    def _get_interp_gen_data(self, beg_idx, end_idx, max_rng):

        data_df = self._data_df.iloc[beg_idx:end_idx]

        if any(
            [interp_arg[0] in ['OK', 'SK', 'EDK']
             for interp_arg in self._interp_args]):

            vgs_ser = self._vgs_ser.loc[data_df.index]
            vgs_rord_tidxs_ser = self._vgs_rord_tidxs_ser.loc[data_df.index]

        else:
            vgs_ser = None
            vgs_rord_tidxs_ser = None

        if 'EDK' in [interp_arg[0] for interp_arg in self._interp_args]:
            drft_arrs = self._drft_arrs
            stns_drft_df = self._stns_drft_df

        else:
            drft_arrs, stns_drft_df = 2 * [None]

        return (
            data_df,
            beg_idx,
            end_idx,
            max_rng,
            self._interp_args,
            self._lock,
            drft_arrs,
            stns_drft_df,
            vgs_ser,
            vgs_rord_tidxs_ser)

    def _get_thread_steps_idxs(self):

        '''
        All sizes in bytes
        '''
        megabytes = 1024 ** 2

        bytes_per_number = 8

        # Actual interpreter size will probably be much smaller than this.
        interpreter_size = get_current_proc_size()

        assert 0 < self._max_mem_usage_ratio <= 1

        tot_avail_mem = int(
            ps.virtual_memory().free * self._max_mem_usage_ratio)

#         avail_threads_mem = tot_avail_mem - (self._n_cpus * interpreter_size)
        avail_threads_mem = tot_avail_mem - interpreter_size

        assert avail_threads_mem > 0, 'Memory too little or too many threads!'

        max_mem_per_thread = avail_threads_mem // self._n_cpus

        assert max_mem_per_thread > 0, 'Memory too little or too many threads!'

        # Size of the netcdf array to write.
        tot_interp_arr_size = bytes_per_number * len(self._interp_args) * (
            self._data_df.shape[0] *
            self._interp_crds_orig_shape[0] *
            self._interp_crds_orig_shape[1])

        bytes_per_step = (
            bytes_per_number *
            np.prod(self._interp_crds_orig_shape) *
            len(self._interp_args))

        dst_ref_2d_dists_size = bytes_per_number * (
            self._interp_x_crds_msh.size *
            self._crds_df.shape[0])

        if self._vgs_ser is not None:
            ref_ref_2d_dists_size = bytes_per_number * (
                self._crds_df.shape[0] ** 2)

            ref_ref_2d_vars_size = ref_ref_2d_dists_size * (
                1)

            dst_ref_2d_vars_size = dst_ref_2d_dists_size * (
                1)

        else:
            ref_ref_2d_dists_size = 0
            ref_ref_2d_vars_size = 0
            dst_ref_2d_vars_size = 0
            max_vgs_per_thread = 0

        assert max_mem_per_thread > bytes_per_step, (
                'Interpolation grid too fine!')

        max_cpus_ctr = self._n_cpus
        max_chunks_ctr = self._n_cpus
        cpu_chunk_flag = 0
        while True:

            assert max_cpus_ctr > 0, 'Not enough memory for the given grid!'

            assert max_chunks_ctr < self._data_df.shape[0], (
                'Not enough memory for the given grid!')

            step_idxs = ret_mp_idxs(self._data_df.shape[0], max_chunks_ctr)

            if self._vgs_ser is not None:
                mult_arrs_ct = 0

                max_vgs_per_thread = max([
                    np.unique(self._vgs_unq_ids.iloc[
                        step_idxs[i]:step_idxs[i + 1]].values).size
                    for i in range(step_idxs.size - 1)])

                if any([self._ork_flag, self._edk_flag]):
                    mult_arrs_ct += 1

                if self._spk_flag:
                    mult_arrs_ct += 1

                max_vgs_per_thread *= mult_arrs_ct

            else:
                max_vgs_per_thread = 0

            misc_size = 0

            misc_size += dst_ref_2d_dists_size * max_cpus_ctr

            misc_size += ref_ref_2d_dists_size * max_cpus_ctr

            misc_size += (
                ref_ref_2d_vars_size * max_vgs_per_thread * max_cpus_ctr)

            misc_size += (
                dst_ref_2d_vars_size * max_vgs_per_thread * max_cpus_ctr)

            has_size = ((tot_interp_arr_size / max_chunks_ctr) + misc_size)

            if has_size < tot_avail_mem:
                self._n_cpus = max_cpus_ctr

                max_mem_per_thread = avail_threads_mem // self._n_cpus

                break

            if cpu_chunk_flag and (max_cpus_ctr > 1):
                max_cpus_ctr -= 1

                cpu_chunk_flag = 0

            else:
                max_chunks_ctr += 1
                cpu_chunk_flag = 1

        steps_per_thread = (step_idxs[1:] - step_idxs[0:-1]).max()

        if ((self._max_steps_per_chunk is not None) and
            (steps_per_thread > self._max_steps_per_chunk)):

            step_idxs = np.arange(
                0, self._data_df.shape[0], self._max_steps_per_chunk)

            step_idxs = np.concatenate((step_idxs, [self._data_df.shape[0]]))

            steps_per_thread = self._max_steps_per_chunk

        if self._vb:
            print_sl()

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

            print(f'Size per step: {bytes_per_step / megabytes:0.6f} MB')

            print(f'No. of steps interpolated per thread: {steps_per_thread}')

            print(f'Misc. memory required: {misc_size / megabytes:0.6f} MB')

            print('Final number of threads to use:', self._n_cpus)

            if max_vgs_per_thread:
                print(
                    'Maximum variogram strings per thread:',
                    max_vgs_per_thread)

            print_el()

        return step_idxs
