'''
Created on Nov 25, 2018

@author: Faizan
'''

import os
import timeit
from pathlib import Path
from functools import partial
from multiprocessing import Pool, Manager, Lock, Process

import numpy as np
import pandas as pd
import psutil as ps
import netCDF4 as nc
import matplotlib.pyplot as plt

from .data import SpInterpData as SID
from .steps import SpInterpSteps as SIS
from .prepare import SpInterpPrepare as SIP
from .plot import SpInterpPlot as SIPLT
from ..misc import (
    ret_mp_idxs,
    get_current_proc_size,
    print_sl,
    print_el,
    get_proc_pid,
    monitor_memory)

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
        self._nnb_flag = False
        self._interp_flag_est_vars = False

        self._max_mem_usage_ratio = 1.0

        self._prc_mem_mon_flag = False

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

        (interp_steps_idxs,
         grid_chunk_idxs,
         max_n_cpus) = self._get_thread_steps_idxs()

        proc_pids = [(self._n_cpus, os.getpid())]

        if self._mp_flag:
            try:
                mp_pool = Pool(self._n_cpus)

            except Exception as msg:
                print(f'Error creating Pool for interpolation: {msg}',)

            # Has to be a blocking one.
            spi_map = partial(mp_pool.map, chunksize=1)

            self._lock = Manager().Lock()

            proc_pids.extend(
                list(spi_map(get_proc_pid,
                             [(i,) for i in range(self._n_cpus)])))

        else:
            spi_map = map

            self._lock = Lock()  # The manager might slow things down.

        if self._prc_mem_mon_flag:
            mem_mon_proc = Process(
                target=monitor_memory, args=((proc_pids, self._out_dir, 0.5),))

            mem_mon_proc.start()

        interp_steps_cls = SIS(self)

        if self._vb:
            print_sl()
            print('Started interpolation...')
            print('Parent process ID:', os.getpid())
            print('\n')

            print('Interpolation step indices:\n', interp_steps_idxs, sep='')
            print('\n')

            print('Grid chunk indices:\n', grid_chunk_idxs, sep='')

        ivar_names = [interp_arg[2] for interp_arg in self._interp_args]

        if self._vb:
            interp_beg_time = timeit.default_timer()

            t_passes = int(np.ceil(
                (interp_steps_idxs.shape[0] - 1) / max_n_cpus))

            t_passes *= (grid_chunk_idxs.shape[0] - 1)

            print_sl()

            print('Current interpolation methods:', ivar_names)
            print(f'Using {t_passes} pass(es) to interpolate.')

            print_el()

        interp_gen = (
            self._get_interp_gen_data(
                interp_steps_idxs[i],
                interp_steps_idxs[i + 1],
                interp_steps_idxs.size - 1,
                grid_chunk_idxs[j],
                grid_chunk_idxs[j + 1],
                )

            for i in range(interp_steps_idxs.size - 1)
            for j in range(grid_chunk_idxs.shape[0] - 1))

        maped_obj = spi_map(interp_steps_cls.interpolate_subset, interp_gen)

        if not self._mp_flag:
            list(maped_obj)

        if self._vb:
            interp_time = timeit.default_timer() - interp_beg_time

            print_sl()

            print(
                f'Done with the interpolation methods: {ivar_names} in '
                f'{interp_time:0.1f} seconds.',)

            print_el()

        if True:
            if self._vb:
                print_sl()

                bt = timeit.default_timer()

                print('Computing final time series...')

            self._save_stats_sers()

            et = timeit.default_timer()

            if self._vb:

                print(f'Took: {et - bt:0.3f} seconds!')

                print_el()

        if self._plot_figs_flag:

            if self._vb:
                plot_beg_time = timeit.default_timer()

                print_sl()

                print('Plotting figures...')

            interp_plot_cls = SIPLT(self)

            plot_data_idxs = ret_mp_idxs(self._data_df.shape[0], self._n_cpus)

            plot_gen = (
                (self._data_df.iloc[plot_data_idxs[i]:plot_data_idxs[i + 1],:],
                 plot_data_idxs[i],
                 plot_data_idxs[i + 1],
                 self._interp_args,
                 self._vgs_ser,
                 )
                for i in range(plot_data_idxs.size - 1))

            maped_obj = spi_map(interp_plot_cls.plot_interp_steps, plot_gen)

            if not self._mp_flag:
                list(maped_obj)

            if self._vb:
                plot_time = timeit.default_timer() - plot_beg_time

                print(f'Done with plotting in {plot_time:0.1f} seconds.')

                print_el()

        if not self._mp_flag:
            pass

        else:
            mp_pool.close()
            mp_pool.join()
            mp_pool = None

        if self._prc_mem_mon_flag:
            try:
                mem_mon_proc.terminate()

            except Exception as msg:
                print(f'Error upon terminating mem_mon_proc: {msg}')

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

        raise NotImplementedError('SK is deprecated!')

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

    def turn_nearest_neighbor_on(self):

        if self._vb:
            print_sl()

            print('Set nearest neighbor flag to True.')

            print_el()

        self._nnb_flag = True
        return

    def turn_nearest_neighbor_off(self):

        assert self._nnb_flag
        if self._vb:
            print_sl()

            print('Set nearest neighbor flag to False.')

            print_el()

        self._nnb_flag = False
        return

    def turn_ordinary_kriging_est_var_on(self):

        assert self._ork_flag

        if self._vb:
            print_sl()

            print('Set ordinary kriging estimation variance flag to True.')

            print_el()

        self._interp_flag_est_vars = True
        return

    def turn_ordinary_kriging_est_var_off(self):

        assert self._interp_flag_est_vars

        if self._vb:
            print_sl()

            print('Set ordinary kriging estimation variance flag to False.')

            print_el()

        self._interp_flag_est_vars = False
        return

    def _save_stats_sers(self):

        data_df = self._data_df.sort_index()

        interp_labels = [
            interp_arg[2]
            for interp_arg in self._interp_args
            if interp_arg[2] != 'EST_VARS_OK']

        time_steps = data_df.index

        # Count is handeld separately.
        stats = ['min', 'mean', 'max', 'std', 'count']

        stats_df = pd.DataFrame(
            index=time_steps,
            columns=(
                [f'data_{stat}' for stat in stats] +
                [f'{ilab}_{stat}'
                 for ilab in interp_labels for stat in stats]),
            dtype=np.float32)

        for stat in stats:
            stats_df.loc[self._data_df.index, f'data_{stat}'] = getattr(
                self._data_df, stat)(axis=1).astype(np.float32)

        nc_hdl = nc.Dataset(str(self._nc_file_path), mode='r')

        for ilab in interp_labels:

            for i in range(time_steps.shape[0]):

                interp_fld = nc_hdl[ilab][i].data

                for stat in stats:

                    if stat == 'count':
                        stat_val = np.isfinite(interp_fld).sum(dtype=np.uint64)

                    else:
                        stat_val = getattr(np, f'nan{stat}')(interp_fld)

                    stats_df.loc[
                        time_steps[i], f'{ilab}_{stat}'] = np.float32(stat_val)

        stats_df.to_csv(
            self._out_dir / 'stats.csv', sep=';', float_format='%0.6f')

        nc_hdl.close()

        self._plot_stats_sers(stats, stats_df, interp_labels)
        return

    def _plot_stats_sers(self, stats, stats_df, interp_labels):

        plt.figure(figsize=(8.0, 6.5))

        for stat in stats:

            min_val = +np.inf
            max_val = -np.inf

            obs_vls = stats_df[f'data_{stat}'].values

            # Taking mean value wherever not finite.
            obs_nft_ixs = ~np.isfinite(obs_vls)

            obs_avg = obs_vls[~obs_nft_ixs].mean()

            if obs_nft_ixs.sum():
                obs_vls[obs_nft_ixs] = obs_avg

            obs_cum_sum = obs_vls.cumsum()
            ref_cum_sum = np.repeat(obs_avg, obs_vls.size).cumsum()

            if stat == 'count':
                obs_cum_sum /= obs_cum_sum[-1]
                ref_cum_sum /= ref_cum_sum[-1]

            min_val = min(min_val, obs_cum_sum[+0])
            min_val = min(min_val, ref_cum_sum[+0])

            max_val = max(max_val, obs_cum_sum[-1])
            max_val = max(max_val, ref_cum_sum[-1])

            plt.plot(ref_cum_sum, obs_cum_sum, label='DATA')

            for ilab in interp_labels:

                sim_vls = stats_df[f'{ilab}_{stat}'].values

                sim_nft_ixs = ~np.isfinite(sim_vls)
                if sim_nft_ixs.sum():
                    sim_vls[sim_nft_ixs] = sim_vls[~sim_nft_ixs].mean()

                sim_cum_sum = sim_vls.cumsum()

                if stat == 'count':
                    sim_cum_sum /= sim_cum_sum[-1]

                min_val = min(min_val, sim_cum_sum[+0])
                max_val = max(max_val, sim_cum_sum[-1])

                plt.plot(ref_cum_sum, sim_cum_sum, label=ilab)

            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                ls='--',
                c='k',
                lw=2.0,
                alpha=0.75)

            plt.xlabel('REF')
            plt.ylabel('SIM')

            plt.grid()
            plt.gca().set_axisbelow(True)
            plt.gca().set_aspect('equal')

            plt.legend(loc='upper left')

            plt.savefig(
                self._out_dir / f'stats__{stat}.png', bbox_inches='tight')

            plt.clf()

        plt.close()
        return

    def _get_interp_gen_data(
            self,
            beg_steps_idx,
            end_steps_idx,
            max_steps_rng,
            beg_grid_idx,
            end_grid_idx,
            ):

        data_df = self._data_df.iloc[beg_steps_idx:end_steps_idx]

        if any(
            [interp_arg[0] in ['OK', 'SK', 'EDK']
             for interp_arg in self._interp_args]):

            vgs_ser = self._vgs_ser.loc[data_df.index]
            vgs_rord_tidxs_ser = self._vgs_rord_tidxs_ser.loc[data_df.index]

            assert np.all(vgs_ser.values != 'nan'), (
                'NaN VGs not allowed! '
                'Use Nugget or any other appropriate one!')

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
            beg_steps_idx,
            end_steps_idx,
            max_steps_rng,
            self._interp_args,
            self._lock,
            drft_arrs,
            stns_drft_df,
            vgs_ser,
            vgs_rord_tidxs_ser,
            beg_grid_idx,
            end_grid_idx,
            )

    def _get_thread_steps_idxs(self):

        '''
        All sizes in bytes.

        To find the optimum number of steps to interpolate per thread per pass,
        it is first checked if the entire thing can be executed in one go i.e.
        having the user specified number of threads. If the memory won't be
        enough, then first, time step groups per thread are reduced and then
        the number of running threads at any given instance. This is done
        alternately, till a solution is found. If not, the program stops.
        The maximum number of running threads can be dropped to one and the
        maximum number of steps per thread per pass can go as low as one.
        '''
        megabytes = 1024 ** 2

        bytes_per_number = self._intrp_dtype(1).itemsize

        # Actual interpreter size will probably be much smaller than this.
        interpreter_size = get_current_proc_size()

        assert 0 < self._max_mem_usage_ratio <= 1

        tot_avail_mem = int(
            ps.virtual_memory().available * self._max_mem_usage_ratio)

#         avail_threads_mem = tot_avail_mem - (self._n_cpus * interpreter_size)
        avail_threads_mem = tot_avail_mem  # - interpreter_size

        assert avail_threads_mem > 0, 'Memory too little or too many threads!'

        max_mem_per_thread = avail_threads_mem // self._n_cpus

        assert max_mem_per_thread > 0, 'Memory too little or too many threads!'

        # Size of the netcdf array to write.
        tot_interp_arr_size = bytes_per_number * len(self._interp_args) * (
            self._data_df.shape[0] *
            np.prod(self._interp_crds_orig_shape).astype(np.uint64))

        bytes_per_step = (
            bytes_per_number *
            np.prod(self._interp_crds_orig_shape) *
            len(self._interp_args))

        dst_ref_2d_dists_size = 8 * (
            self._interp_x_crds_msh.size * self._crds_df.shape[0])

        if self._vgs_ser is not None:
            ref_ref_2d_dists_size = 8 * (
                self._crds_df.shape[0] ** 2)

            ref_ref_2d_vars_size = ref_ref_2d_dists_size * 1

            # It is increased in size based on the type of interpolation used.
            dst_ref_2d_vars_size = dst_ref_2d_dists_size * 1

        else:
            ref_ref_2d_dists_size = 0
            ref_ref_2d_vars_size = 0
            dst_ref_2d_vars_size = 0
            max_vgs_per_thread = 0

        assert max_mem_per_thread > bytes_per_step, (
                'Interpolation grid too fine!')

        step_idxs = ret_mp_idxs(self._data_df.shape[0], self._n_cpus)

        steps_per_thread = (step_idxs[+1:] - step_idxs[:-1]).max()

        # This should take place before the while loop.
        if ((self._max_steps_per_chunk is not None) and
            (steps_per_thread > self._max_steps_per_chunk)):

            step_idxs = np.arange(
                0, self._data_df.shape[0], self._max_steps_per_chunk)

            step_idxs = np.concatenate((step_idxs, [self._data_df.shape[0]]))

            steps_per_thread = self._max_steps_per_chunk

        grid_chunks = 1
        grid_idxs = ret_mp_idxs(self._interp_crds_orig_shape[0], grid_chunks)

        # TODOO: The case when the number of interpolated steps is less
        # than the number of threads. It happens rarely. No big worries though.
        max_cpus_ctr = self._n_cpus
        max_chunks_ctr = step_idxs.size - 1
        cpu_chunk_flag = 0

        while True:
            if self._vgs_ser is not None:
                mult_arrs_ct = 0

                unq_vgs_cts = [
                    len(set(self._vgs_unq_ids.iloc[
                        step_idxs[i]:step_idxs[i + 1]].values))
                    for i in range(step_idxs.size - 1)]

                max_vgs_per_thread = sum(unq_vgs_cts) / len(unq_vgs_cts)

                if any([self._ork_flag, self._edk_flag]):
                    mult_arrs_ct += 1

                if self._spk_flag:
                    mult_arrs_ct += 1

                max_vgs_per_thread *= mult_arrs_ct

            else:
                max_vgs_per_thread = 0

            misc_size = interpreter_size * max_cpus_ctr

            misc_size += (dst_ref_2d_dists_size / grid_chunks) * max_cpus_ctr

            misc_size += ref_ref_2d_dists_size * max_cpus_ctr

            misc_size += (
                ref_ref_2d_vars_size * max_vgs_per_thread * max_cpus_ctr)

            misc_size += (
                (dst_ref_2d_vars_size / grid_chunks) *
                max_vgs_per_thread *
                max_cpus_ctr)

            # The extra percents are for other variables that are
            # created while interpolating.
            has_size = 1.3 * (
                np.ceil(tot_interp_arr_size /
                        grid_chunks /
                        (max_chunks_ctr / max_cpus_ctr)) +
                misc_size)

            if has_size < tot_avail_mem:
                max_mem_per_thread = avail_threads_mem // max_cpus_ctr

                break

            if cpu_chunk_flag:
                grid_chunks += 1
                # max_cpus_ctr -= 1
                cpu_chunk_flag = 0

            else:
                max_chunks_ctr += max_cpus_ctr
                cpu_chunk_flag = 1

            assert max_cpus_ctr > 0, 'Not enough memory for the given grid!'

            assert max_chunks_ctr < self._data_df.shape[0], (
                'Not enough memory for the given grid!')

            assert grid_chunks < self._interp_crds_orig_shape[0], (
                'Not enough memory for the given grid!')

            step_idxs = ret_mp_idxs(self._data_df.shape[0], max_chunks_ctr)

            grid_idxs = ret_mp_idxs(
                self._interp_crds_orig_shape[0], grid_chunks)

        steps_per_thread = (step_idxs[+1:] - step_idxs[:-1]).max()

        if self._vb:
            print_sl()

            print('Memory management stuff...')

            print(f'Main interpreter size: {interpreter_size // megabytes} MB.')

            print(
                f'Total memory available for use: '
                f'{tot_avail_mem // megabytes} MB.')

            print(
                f'Memory available per thread: '
                f'{max_mem_per_thread // megabytes} MB.')

            print(
                f'Total size of the interpolated array: '
                f'{tot_interp_arr_size / megabytes:0.1f} MB.')

            print(f'Size per step: {bytes_per_step / megabytes:0.1f} MB.')

            print(f'No. of steps interpolated per thread: {steps_per_thread}.')

            print(f'No. of time step chunks: {max_chunks_ctr}.')

            print(f'No. of grid chunks: {grid_chunks}.')

            print(
                f'Misc. memory required by all threads: '
                f'{misc_size / megabytes:0.1f} MB.')

            print(
                f'Approximate maximum size of RAM needed at any instant: '
                f'{has_size / megabytes:0.1f} MB.')

            print(f'Final number of threads to use: {max_cpus_ctr}.')

            if max_vgs_per_thread:
                print(
                    f'Mean variogram strings per thread: '
                    f'{round(max_vgs_per_thread, 1)}.')

            print_el()

        return step_idxs, grid_idxs, max_cpus_ctr
