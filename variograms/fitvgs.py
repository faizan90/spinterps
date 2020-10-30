'''
Created on Nov 25, 2018

@author: Faizan
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

from ..misc import ret_mp_idxs, traceback_wrapper
from .vgs import Variogram as VG
from .vgsinput import VariogramsInput as VI

plt.ioff()


class FitVariograms(VI):

    def __init__(self, verbose=True):

        VI.__init__(self, verbose)

        self._vgs_fitted_flag = False
        return

    def verify(self):

        VI._VariogramsInput__verify(self)
        return

    def fit_vgs(self):

        assert self._inputs_vrfd_flag, 'Call the verify method first!'

        fit_vgs_steps_cls = FitVariogramsSteps(self)

        if self._out_figs_path is not None:
            self._out_dir.mkdir(exist_ok=True)
            self._out_figs_path.mkdir(exist_ok=True)

        if self._n_cpus == 1:
            self._vg_strs_df = fit_vgs_steps_cls.get_vgs_df(self._data_df)

        else:
            n_steps = self._data_df.shape[0]

            mp_idxs = ret_mp_idxs(n_steps, self._n_cpus)

            self._vg_strs_df = pd.DataFrame(
                index=self._data_df.index,
                columns=np.arange(self._n_best_vgs),
                dtype=object)

            # Time step indices are reordered in such a way that each chunk
            # sent to a different thread gets a subset of indices such that
            # the indices are uniformly sampled from all the time
            # periods. This helps in keeping all the threads busy for as
            # long as possible. For example, if the indices in the begining
            # have too few stations, then the thread will finish fast and
            # would have nothing to do. Reordering tries to avoid this
            # scenario.
            time_idxs = self._data_df.index

            time_idxs_reshuff = []
            for i in range(mp_idxs[1]):
                time_idxs_reshuff.extend(time_idxs[mp_idxs[:-1] + i].tolist())

            time_idxs_reshuff = pd.DatetimeIndex(time_idxs_reshuff)

            time_idxs_reshuff = time_idxs_reshuff[
                ~time_idxs_reshuff.duplicated(keep='first')]

            time_idxs_diff = time_idxs.difference(time_idxs_reshuff)

            time_idxs_reshuff = time_idxs_reshuff.append(time_idxs_diff)

            assert time_idxs_reshuff.size == n_steps

            sub_data_dfs_gen = (
                self._data_df.loc[time_idxs_reshuff[mp_idxs[i]:mp_idxs[i + 1]]]
                for i in range(self._n_cpus))

            mp_pool = Pool(self._n_cpus)

            vg_strs_dfs = mp_pool.map(
                fit_vgs_steps_cls.get_vgs_df, sub_data_dfs_gen)

            for vg_strs_df in vg_strs_dfs:
                self._vg_strs_df.update(vg_strs_df)

        self._vgs_fitted_flag = True
        return

    def get_fin_vgs_df(self):

        assert self._vgs_fitted_flag, 'Call the fit_vgs method first!'

        return self._vg_strs_df

    def save_fin_vgs_df(self):

        assert self._vgs_fitted_flag, 'Call the fit_vgs method first!'

        self._out_dir.mkdir(exist_ok=True)

        self._vg_strs_df.to_csv(str(self._out_dir / 'vg_strs.csv'), sep=';')
        return


class FitVariogramsSteps:

    '''Intended for use by the FitVariograms class'''

    def __init__(self, fit_vg_cls):

        self._vb = fit_vg_cls._vb

        self._crds_df = fit_vg_cls._crds_df

        self._mdr = fit_vg_cls._mdr
        self._n_vgs_perms = fit_vg_cls._n_vgs_perms
        self._nug_vg = fit_vg_cls._nug_vg
        self._n_gps = fit_vg_cls._n_gps
        self._vg_names = fit_vg_cls._vg_names
        self._n_best_vgs = fit_vg_cls._n_best_vgs

        self._index_type = fit_vg_cls._index_type

        self._min_vld_stns = fit_vg_cls._min_vld_stns

        self._out_figs_path = fit_vg_cls._out_figs_path
        return

    @traceback_wrapper
    def _fit_vgs_step(self, date, sub_stns_time_ser_df, vg_strs_df):

        fig_size = (13, 7)

        in_vals_ser = sub_stns_time_ser_df.loc[date].copy()
        in_vals_ser.dropna(inplace=True)

        aval_stns = in_vals_ser.index.intersection(self._crds_df.index)

        if aval_stns.shape[0] < self._min_vld_stns:
            if self._vb:
                print(f'No VG on {date}, too few stations!')

            return

        x_crds = self._crds_df.loc[aval_stns]['X'].values
        y_crds = self._crds_df.loc[aval_stns]['Y'].values

        z_vals = in_vals_ser.loc[aval_stns].values

        vg = VG(
            x=x_crds,
            y=y_crds,
            z=z_vals,
            mdr=self._mdr,
            nk=10,
            typ='cnst',
            perm_r_list=self._n_vgs_perms,
            fil_nug_vg=self._nug_vg,
            ld=None,
            uh=None,
            h_itrs=100,
            opt_meth='L-BFGS-B',
            opt_iters=1000,
            fit_vgs=self._vg_names,
            n_best=self._n_best_vgs,
            evg_name='classic',
            use_wts=False,
            ngp=self._n_gps,
            fit_thresh=0.01)

        vg.fit()

        fit_vg_list = vg.vg_str_list
        fit_vgs_no = len(fit_vg_list) - 1

        if self._vb:
            if fit_vg_list:
                print(f'{date}: {fit_vg_list[-1]}')

            else:
                print(f'Could not fit any VG on date: {date}')
                return

        for i, vg_str in enumerate(fit_vg_list):
            vg_strs_df.loc[date][fit_vgs_no - i] = vg_str

        if self._out_figs_path is not None:
            vg_names = vg.best_vg_names
            if not vg_names:
                return

            evg = vg.vg_vg_arr
            h_arr = vg.vg_h_arr
            vg_fit = vg.vg_fit

            if self._index_type == 'date':
                date_str = '%0.4d-%0.2d-%0.2d' % (
                    date.year, date.month, date.day)

            elif self._index_type == 'obj':
                date_str = date

            else:
                raise ValueError(
                    f'Not programmed to handle given index_type: '
                    f'{self._index_type}!')

            plt.figure(figsize=fig_size)

            plt.plot(h_arr, evg, 'bo', alpha=0.3)

            for m in range(len(vg_names)):
                plt.plot(
                    vg_fit[m][:, 0],
                    vg_fit[m][:, 1],
                    c=pd.np.random.rand(3,),
                    linewidth=4,
                    zorder=m,
                    label=fit_vg_list[m],
                    alpha=0.6)

            plt.grid()

            plt.xlabel('Distance')
            plt.ylabel('Variogram')

            plt.title(
                'Event date: %s' % (date_str), fontdict={'fontsize':15})

            if vg_names:
                plt.legend(loc=4, framealpha=0.7)

            plt.savefig(
                str(self._out_figs_path / f'{date_str}.png'),
                bbox_inches='tight')

            plt.close()

        return

    def get_vgs_df(self, sub_stns_time_ser_df):

        vg_strs_df = pd.DataFrame(
            index=sub_stns_time_ser_df.index,
            columns=np.arange(self._n_best_vgs),
            dtype=object)

        for date in sub_stns_time_ser_df.index:
            self._fit_vgs_step(date, sub_stns_time_ser_df, vg_strs_df)

        return vg_strs_df
