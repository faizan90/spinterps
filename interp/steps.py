'''
Created on Nov 26, 2018

@author: Faizan-Uni
'''
import warnings

import numpy as np
import netCDF4 as nc

import matplotlib.pyplot as plt
from descartes import PolygonPatch

from .grps import SpInterpNeighborGrouping as SIG
from ..misc import traceback_wrapper, check_full_nuggetness
from ..cyth import (
    fill_dists_one_pt,
    fill_wts_and_sum,
    get_mults_sum,
    fill_dists_2d_mat,
    fill_vg_var_arr)

plt.ioff()


class SpInterpSteps:

    def __init__(self, spinterp_main_cls):

        read_labs = [
            '_vb',
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
            '_interp_y_crds_plt_msh',
            '_interp_x_crds_msh',
            '_interp_y_crds_msh',
            '_index_type',
            '_nc_file_path',
            '_neb_sel_mthd',
            '_n_nebs',
            '_n_pies',
            ]

        for read_lab in read_labs:
            setattr(self, read_lab, getattr(spinterp_main_cls, read_lab))

        self._n_dst_pts = self._interp_x_crds_msh.shape[0]
        return

    @traceback_wrapper
    def interpolate_subset(self, all_args):

        (data_df,
         beg_idx,
         end_idx,
         max_rng,
         interp_args,
         lock,
         drft_arrs,
         stns_drft_df,
         vgs_ser) = all_args

        if vgs_ser is not None:
            assert not data_df.index.difference(vgs_ser.index).shape[0]

        interp_types = [interp_arg[0] for interp_arg in interp_args]
        interp_labels = [interp_arg[2] for interp_arg in interp_args]

        if self._plot_figs_flag:
            out_figs_dirs = {
                interp_arg[2]: interp_arg[1] for interp_arg in interp_args}

        krg_flag = any(
            [krg_type in interp_types
             for krg_type in ('OK', 'SK', 'EDK')])

        edk_flag = 'EDK' in [interp_arg[0] for interp_arg in interp_args]

        time_steps = data_df.index

        interp_flds_dict = {interp_label :np.full(
            (time_steps.shape[0], np.prod(self._interp_crds_orig_shape)),
            np.nan,
            dtype=np.float32)
            for interp_label in interp_labels}

        cntn_idxs_wh = np.where(self._cntn_idxs)[0]

        grp_cls = SIG(
            self._neb_sel_mthd,
            self._n_nebs,
            self._n_pies,
            self._interp_x_crds_msh,
            self._interp_y_crds_msh,
            verbose=False)

        grps_in_time = grp_cls.get_grps_in_time(data_df)

        for time_stn_grp, cmn_time_stn_grp_idxs in grps_in_time:
            assert time_stn_grp.size == np.unique(time_stn_grp).size

            time_stn_grp_ref_xs = self._crds_df.loc[time_stn_grp, 'X'].values
            time_stn_grp_ref_ys = self._crds_df.loc[time_stn_grp, 'Y'].values

            time_stn_grp_data_vals = data_df.loc[
                cmn_time_stn_grp_idxs, time_stn_grp].values

            sub_time_steps = time_steps[cmn_time_stn_grp_idxs]

            if edk_flag:
                time_stn_grp_drifts = stns_drft_df.loc[time_stn_grp].values

            else:
                time_stn_grp_drifts = None

            if krg_flag:
                vg_models = vgs_ser.loc[cmn_time_stn_grp_idxs].values
                nuggetness_flags = np.zeros(vg_models.shape[0], dtype=bool)

                for i, vg_model in enumerate(vg_models):
                    nuggetness_flags[i] = check_full_nuggetness(vg_model)

                    if nuggetness_flags[i] and self._vb:
                        print('Full nugget at time step:', sub_time_steps[i])

            else:
                vg_models = None
                nuggetness_flags = None

            time_neb_idxs, time_neb_idxs_grps = grp_cls.get_neb_idxs_and_grps(
                time_stn_grp_ref_xs, time_stn_grp_ref_ys)

            interp_flds_time_idxs = np.where(cmn_time_stn_grp_idxs)[0]

            pts_done_flags = np.zeros(self._n_dst_pts, dtype=bool)
            sub_pts_flags = np.zeros(self._n_dst_pts, dtype=bool)
            prblm_time_steps = []

            for time_neb_idxs_grp in time_neb_idxs_grps:
                sub_dst_xs = self._interp_x_crds_msh[time_neb_idxs_grp]
                sub_dst_ys = self._interp_y_crds_msh[time_neb_idxs_grp]

                sub_ref_idxs = time_neb_idxs[time_neb_idxs_grp[0]]

                sub_ref_xs = time_stn_grp_ref_xs[sub_ref_idxs]
                sub_ref_ys = time_stn_grp_ref_ys[sub_ref_idxs]

                sub_ref_data = time_stn_grp_data_vals[
                    :, sub_ref_idxs].copy('c')

                interp_steps_flags = np.ones(sub_ref_data.shape[0], dtype=bool)
                for j in range(sub_ref_data.shape[0]):
                    if np.any(sub_ref_data[j] >= self._min_var_thr):
                        continue

                    interp_steps_flags[j] = False

                if edk_flag:
                    sub_ref_drifts = time_stn_grp_drifts[sub_ref_idxs, :]
                    sub_dst_drifts = drft_arrs[:, time_neb_idxs_grp]

                else:
                    sub_ref_drifts = None
                    sub_dst_drifts = None

                sub_pts_flags[:] = False
                sub_pts_flags[time_neb_idxs_grp] = True

                for i, interp_type in enumerate(interp_types):
                    if interp_type == 'IDW':
                        idw_exp = interp_args[i][3]

                    else:
                        idw_exp = None

                    interp_vals = self._get_interp(
                        sub_dst_xs,
                        sub_dst_ys,
                        sub_ref_xs,
                        sub_ref_ys,
                        sub_ref_data,
                        interp_type,
                        sub_ref_drifts,
                        sub_dst_drifts,
                        idw_exp,
                        vg_models,
                        interp_steps_flags,
                        nuggetness_flags,
                        sub_time_steps,
                        prblm_time_steps)

                    self._mod_min_max(interp_vals)

                    interp_flds = interp_flds_dict[interp_labels[i]]

                    if self._cntn_idxs is not None:
                        for j, time_idx in enumerate(interp_flds_time_idxs):
                            interp_flds[
                                time_idx, cntn_idxs_wh[sub_pts_flags]] = (
                                    interp_vals[j])

                    else:
                        for j, time_idx in enumerate(interp_flds_time_idxs):
                            interp_flds[time_idx, sub_pts_flags] = (
                                interp_vals[j])

                pts_done_flags[time_neb_idxs_grp] = True

            assert np.all(pts_done_flags)

            if prblm_time_steps:
                print(
                    'WARNING: There were problems while interpolating '
                    'at the following steps:')

                for prblm_stp in prblm_time_steps:
                    print(prblm_stp)

        if self._plot_figs_flag:
            for interp_type, interp_label in zip(interp_types, interp_labels):
                out_figs_dir = out_figs_dirs[interp_label]
                interp_flds = interp_flds_dict[interp_label]

                for i in range(time_steps.shape[0]):
                    if interp_type == 'IDW':
                        model = None

                    elif interp_type in ('OK', 'SK', 'EDK'):
                        model = vgs_ser.iloc[i]

                    else:
                        raise NotImplementedError

                    self._plot_interp(
                        interp_flds[i].reshape(self._interp_crds_orig_shape),
                        self._crds_df.loc[:, 'X'].values,
                        self._crds_df.loc[:, 'Y'].values,
                        time_steps[i],
                        model,
                        interp_type,
                        out_figs_dir)

        with lock:
            nc_is = np.linspace(beg_idx, end_idx, max_rng + 1, dtype=int)
            ar_is = nc_is - beg_idx

            nc_hdl = nc.Dataset(str(self._nc_file_path), mode='r+')

            for interp_label in interp_labels:
                interp_flds = interp_flds_dict[interp_label]

                for i in range(max_rng):
                    nc_hdl[interp_label][
                        nc_is[i]:nc_is[i + 1], :, :] = (
                            interp_flds[ar_is[i]:ar_is[i + 1]])

                nc_hdl.sync()
                interp_flds_dict[interp_label] = None

            nc_hdl.close()
        return

    def _get_interp(
            self,
            dst_xs,
            dst_ys,
            ref_xs,
            ref_ys,
            ref_data,
            interp_type,
            ref_drfts,
            dst_drfts,
            idw_exp,
            models,
            interp_steps_flags,
            nuggetness_flags,
            sub_time_steps,
            prblm_time_steps):

        n_dsts = dst_xs.size
        n_refs = ref_xs.size
        n_time = ref_data.shape[0]

        dst_data = np.full((n_time, n_dsts), np.nan)

        ref_means = ref_data.mean(axis=1)

        if interp_type == 'IDW':
            wts = np.full(n_refs, np.nan)
            ref_dst_dists = np.full(n_refs, np.nan)

            for i in range(n_dsts):
                fill_dists_one_pt(
                    dst_xs[i], dst_ys[i], ref_xs, ref_ys, ref_dst_dists)

                wts_sum = fill_wts_and_sum(ref_dst_dists, wts, idw_exp)

                for j in range(n_time):
                    if interp_steps_flags[j]:
                        mults_sum = get_mults_sum(wts, ref_data[j])
                        dst_data[j, i] = mults_sum / wts_sum

                    else:
                        dst_data[j, i] = ref_means[j]

        else:
            if interp_type == 'OK':
                add_rc = 1
                covar_flag = 0

            elif interp_type == 'SK':
                add_rc = 0
                covar_flag = 1

            elif interp_type == 'EDK':
                add_rc = 1 + ref_drfts.shape[1]
                covar_flag = 0

            else:
                raise NotImplementedError

            ref_2d_vars = np.zeros((n_refs + add_rc, n_refs + add_rc))
            ref_2d_dists = np.full((n_refs, n_refs), np.nan)
            dst_ref_2d_dists = np.full((n_dsts, n_refs), np.nan)
            dst_ref_2d_vars = np.zeros((n_dsts, n_refs + add_rc))

            fill_dists_2d_mat(ref_xs, ref_ys, ref_xs, ref_ys, ref_2d_dists)
            fill_dists_2d_mat(dst_xs, dst_ys, ref_xs, ref_ys, dst_ref_2d_dists)

            for j in range(n_time):
                model = models[j]

                if (not interp_steps_flags[j]) or (nuggetness_flags[j]):
                    dst_data[j, :] = ref_means[j]
                    continue

                if model == 'nan':
                    continue

                fill_vg_var_arr(
                    ref_2d_dists, ref_2d_vars, covar_flag, 1, model)

                fill_vg_var_arr(
                    dst_ref_2d_dists, dst_ref_2d_vars, covar_flag, 0, model)

                if interp_type == 'OK':
                    ref_2d_vars[n_refs, :n_refs] = 1.0
                    ref_2d_vars[:, n_refs] = 1.0
                    ref_2d_vars[n_refs, n_refs] = 0.0
                    dst_ref_2d_vars[:, n_refs] = 1.0

                elif interp_type == 'SK':
                    pass

                elif interp_type == 'EDK':
                    ref_2d_vars[n_refs, :n_refs] = 1.0
                    ref_2d_vars[:n_refs, n_refs] = 1.0
                    dst_ref_2d_vars[:, n_refs] = 1.0

                    for k in range(ref_drfts.shape[1]):
                        ref_2d_vars[n_refs + 1 + k, :n_refs] = ref_drfts[:, k]
                        ref_2d_vars[:n_refs, n_refs + 1 + k] = ref_drfts[:, k]

                        dst_ref_2d_vars[:, n_refs + 1 + k] = dst_drfts[k, :]

                else:
                    raise NotImplementedError

                try:
                    ref_2d_vars_inv = np.linalg.inv(ref_2d_vars)

                except Exception:
                    if sub_time_steps[j] not in prblm_time_steps:
                        prblm_time_steps.append(sub_time_steps[j])

                    continue

                for i in range(n_dsts):
                    lmds = np.matmul(ref_2d_vars_inv, dst_ref_2d_vars[i])
                    dst_data[j, i] = (lmds[:n_refs] * ref_data[j]).sum()

        return dst_data

    def _plot_interp(
            self,
            interp_fld,
            curr_x_coords,
            curr_y_coords,
            interp_time,
            model,
            interp_type,
            out_figs_dir):

        if self._index_type == 'date':
            time_str = interp_time.strftime('%Y_%m_%d_T_%H_%M')

        elif self._index_type == 'obj':
            time_str = interp_time

        else:
            raise NotImplementedError(
                f'Unknown index_type: {self._index_type}!')

        out_fig_name = f'{interp_type.lower()}_{time_str}.png'

        fig, ax = plt.subplots()

#         with np.errstate(invalid='ignore'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            grd_min = np.nanmin(interp_fld)
            grd_max = np.nanmax(interp_fld)

        pclr = ax.pcolormesh(
            self._interp_x_crds_plt_msh,
            self._interp_y_crds_plt_msh,
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

        with np.errstate(invalid='ignore'):
            if self._min_var_cut is not None:
                interp_fld[interp_fld < self._min_var_cut] = self._min_var_cut

            if self._max_var_cut is not None:
                interp_fld[interp_fld > self._max_var_cut] = self._max_var_cut
        return
