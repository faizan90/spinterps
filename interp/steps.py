'''
Created on Nov 26, 2018

@author: Faizan-Uni
'''
import sys
import traceback as tb

import numpy as np

import matplotlib.pyplot as plt
from descartes import PolygonPatch

from ..cyth import (
    OrdinaryKriging,
    SimpleKriging,
    ExternalDriftKriging_MD,
    get_idw_arr)

plt.ioff()


class SpInterpSteps:

    def __init__(self, krig_main_cls):

        read_labs = [
            '_n_cpus',
            '_mp_flag',
            '_crds_df',
            '_min_var_thr',
            '_min_var_cut',
            '_max_var_cut',
            '_nc_vlab',
            '_nc_vunits',
            '_plot_polys',
            '_cntn_idxs',
            '_plot_figs_flag',
            '_interp_crds_orig_shape',
            '_interp_x_crds_plt_msh',
            '_interp_y_crds_plot_msh',
            '_interp_x_crds_msh',
            '_interp_y_crds_msh',
            '_qu_timeout_secs'
            ]

        self._debug = False

        for read_lab in read_labs:
            setattr(self, read_lab, getattr(krig_main_cls, read_lab))

        return

    def get_interp_flds(self, all_args):

        ret = None

        try:
            ret = self._get_interp_flds(all_args)

        except Exception as msg:
            print('Error while interpolating:', msg)

            * _, exc_traceback = sys.exc_info()
            tb.print_tb(exc_traceback, limit=None, file=sys.stdout)

        return ret

    def _get_interp_flds(self, all_args):

        (data_df,
         t_idx,
         beg_idx,
         end_idx,
         max_rng,
         interp_arg,
         lock,
         qu_data,
         qu_barr,
         qu_done,
         drft_arrs,
         stns_drft_df,
         vgs_ser) = all_args

        interp_type = interp_arg[0]

        if interp_type == 'IDW':
            idw_exp = interp_arg[3]

            krg_flag = False

        else:
            krg_flag = True

        if self._plot_figs_flag:
            out_figs_dir = interp_arg[1]

        curr_drift_vals = None

        fin_date_range = data_df.index

        interp_flds = np.full(
            (fin_date_range.shape[0],
             self._interp_crds_orig_shape[0],
             self._interp_crds_orig_shape[1]),
            np.nan,
            dtype=np.float32)

        if self._mp_flag:
            if t_idx < (max_rng - 1):
                qu_done.put(
                    (2222,), block=True, timeout=self._qu_timeout_secs)

            qu_barr.put((1,), block=True, timeout=self._qu_timeout_secs)

        for i, interp_time in enumerate(fin_date_range):
            curr_stns = data_df.loc[interp_time, :].dropna().index

            assert curr_stns.shape == np.unique(curr_stns).shape

            curr_data_vals = data_df.loc[interp_time, curr_stns].values
            curr_x_coords = self._crds_df.loc[curr_stns, 'X'].values
            curr_y_coords = self._crds_df.loc[curr_stns, 'Y'].values

            if interp_type == 'EDK':
                curr_drift_vals = (
                    np.atleast_2d(stns_drft_df.loc[curr_stns].values.T))

            if krg_flag:
                model = str(vgs_ser.loc[interp_time])

            else:
                model = None

            interp_vals = None

            if (model == 'nan') or (not curr_stns.shape[0]):

                print('No interpolation on:', interp_time)
                continue

            # TODO: another ratio for this.
            if np.all(curr_data_vals <= self._min_var_thr):
                interp_vals = np.full(
                    self._interp_crds_orig_shape, curr_data_vals.mean())

            if krg_flag:
                try:
                    interp_vals = self._get_krgd_fld(
                        curr_x_coords,
                        curr_y_coords,
                        curr_data_vals,
                        curr_drift_vals,
                        model,
                        interp_type,
                        drft_arrs)

                except Exception as msg:
                    time_str = interp_time.strftime('%Y-%m-%dT%H:%M:%S')
                    print('Error on %s:' % time_str, msg)

            else:
                interp_vals = get_idw_arr(
                    self._interp_x_crds_msh,
                    self._interp_y_crds_msh,
                    curr_x_coords,
                    curr_y_coords,
                    curr_data_vals,
                    idw_exp)

            if interp_vals is None:
                continue

            if self._cntn_idxs is not None:
                interp_flds[i].ravel()[self._cntn_idxs] = interp_vals

            else:
                interp_flds[i] = interp_vals.reshape(
                    self._interp_crds_orig_shape)

            self._mod_min_max(interp_flds[i])

            if self._plot_figs_flag:
                self._plot_interp(
                    interp_flds[i],
                    curr_x_coords,
                    curr_y_coords,
                    interp_time,
                    model,
                    interp_type,
                    out_figs_dir)

        with lock:
            qu_data.put(
                [interp_flds, beg_idx, end_idx],
                block=True,
                timeout=self._qu_timeout_secs)

            interp_flds = None

            if self._mp_flag:
                qu_get = qu_done.get(
                    block=True,
                    timeout=self._qu_timeout_secs)[0]

                assert qu_get == 1111
        return

    def _plot_interp(
            self,
            interp_fld,
            curr_x_coords,
            curr_y_coords,
            interp_time,
            model,
            interp_type,
            out_figs_dir):

        time_str = interp_time.strftime('%Y_%m_%d_T_%H_%M')

        out_fig_name = f'{interp_type.lower()}_{time_str}.png'

        fig, ax = plt.subplots()

        grd_min = np.nanmin(interp_fld)
        grd_max = np.nanmax(interp_fld)

        pclr = ax.pcolormesh(
            self._interp_x_crds_plt_msh,
            self._interp_y_crds_plot_msh,
            interp_fld,
            vmin=grd_min,
            vmax=grd_max)

        cb = fig.colorbar(pclr)

        cb.set_label(self._nc_vlab + ' (' + self._nc_vunits + ')')

        ax.scatter(
            curr_x_coords,
            curr_y_coords,
            label='obs. pts.',
            marker='+',
            c='r',
            alpha=0.7)

        ax.legend(framealpha=0.5)

        if self._plot_polys is not None:
            for poly in self._plot_polys:
                ax.add_patch(PolygonPatch(poly, alpha=1, fc='None', ec='k'))

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

        title = (
            f'Time: {time_str}\n(VG: {model})\n'
            f'Min.: {grd_min:0.4f}, Max.: {grd_max:0.4f}')
        ax.set_title(title)

        plt.setp(ax.get_xmajorticklabels(), rotation=70)
        ax.set_aspect('equal', 'datalim')
        plt.savefig(str(out_figs_dir / out_fig_name), bbox_inches='tight')
        plt.close()
        return

    def _mod_min_max(self, interp_fld):

        if self._min_var_cut is not None:
            interp_fld[interp_fld < self._min_var_cut] = self._min_var_cut

        if self._max_var_cut is not None:
            interp_fld[interp_fld > self._max_var_cut] = self._max_var_cut

        return

    def _get_krgd_fld(
            self,
            curr_x_coords,
            curr_y_coords,
            curr_data_vals,
            curr_drift_vals,
            model,
            interp_type,
            drft_arrs):

        if interp_type == 'OK':
            krige_cls = OrdinaryKriging(
                xi=curr_x_coords,
                yi=curr_y_coords,
                zi=curr_data_vals,
                xk=self._interp_x_crds_msh,
                yk=self._interp_y_crds_msh,
                model=model)

        elif interp_type == 'SK':
            krige_cls = SimpleKriging(
                xi=curr_x_coords,
                yi=curr_y_coords,
                zi=curr_data_vals,
                xk=self._interp_x_crds_msh,
                yk=self._interp_y_crds_msh,
                model=model)

        elif interp_type == 'EDK':
            krige_cls = ExternalDriftKriging_MD(
                xi=curr_x_coords,
                yi=curr_y_coords,
                zi=curr_data_vals,
                si=curr_drift_vals,
                xk=self._interp_x_crds_msh,
                yk=self._interp_y_crds_msh,
                sk=drft_arrs,
                model=model)

        krige_cls.krige()

        return krige_cls.zk
