'''
Created on Nov 26, 2018

@author: Faizan-Uni
'''
import numpy as np
import netCDF4 as nc

import matplotlib.pyplot as plt
from descartes import PolygonPatch

from .grps import SpInterpNeighborGrouping as SIG
from ..misc import traceback_wrapper
from ..cyth import (
    OrdinaryKriging,
    SimpleKriging,
    ExternalDriftKriging_MD,
    fill_dists_one_pt,
    fill_wts_and_sum,
    get_mults_sum,
    sel_equidist_refs,
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
            '_n_dst_pts',
            ]

        for read_lab in read_labs:
            setattr(self, read_lab, getattr(spinterp_main_cls, read_lab))

        return

    @traceback_wrapper
    def get_interp_flds(self, all_args):

        (data_df,
         beg_idx,
         end_idx,
         max_rng,
         interp_arg,
         lock,
         drft_arrs,
         stns_drft_df,
         vgs_ser) = all_args

        interp_type = interp_arg[0]

        if interp_type == 'IDW':
            idw_exp = interp_arg[3]
            krg_flag = False

        else:
            krg_flag = True
            idw_exp = None

        if self._plot_figs_flag:
            out_figs_dir = interp_arg[1]

        time_stn_grp_drifts = None

        fin_date_range = data_df.index

        interp_flds = np.full(
            (fin_date_range.shape[0],
             np.prod(self._interp_crds_orig_shape)),
            np.nan,
            dtype=np.float32)

        grp_cls = SIG(
            self._neb_sel_mthd,
            self._n_nebs,
            self._n_pies,
            self._n_dst_pts,
            self._interp_x_crds_msh,
            self._interp_y_crds_msh,
            verbose=self._vb)

        grp_cls._vb_debug = True

        grps_in_time = grp_cls._get_grps_in_time(data_df)

        for time_stn_grp, cmn_time_stn_grp_idxs in grps_in_time:
            if self._vb:
                print('Time stn group:', time_stn_grp)

            assert time_stn_grp.size == np.unique(time_stn_grp).size

            time_stn_grp_ref_xs = self._crds_df.loc[time_stn_grp, 'X'].values
            time_stn_grp_ref_ys = self._crds_df.loc[time_stn_grp, 'Y'].values

            time_stn_grp_data_vals = data_df.loc[
                cmn_time_stn_grp_idxs, time_stn_grp].values

            if interp_type == 'EDK':
                time_stn_grp_drifts = stns_drft_df.loc[time_stn_grp].values

            if krg_flag:
                models = vgs_ser.loc[cmn_time_stn_grp_idxs].values

            else:
                models = None

            time_neb_idxs, time_neb_idxs_grps = grp_cls._get_neb_idxs_and_grps(
                time_stn_grp_ref_xs, time_stn_grp_ref_ys)

            interp_flds_time_idxs = np.where(cmn_time_stn_grp_idxs)[0]

            pts_done_flags = np.zeros(self._n_dst_pts, dtype=bool)
            sub_pts_flags = np.zeros(self._n_dst_pts, dtype=bool)
            for time_neb_idxs_grp in time_neb_idxs_grps:
                sub_dst_xs = self._interp_x_crds_msh[time_neb_idxs_grp]
                sub_dst_ys = self._interp_y_crds_msh[time_neb_idxs_grp]

                sub_ref_idxs = time_neb_idxs[time_neb_idxs_grp[0]]
                sub_ref_xs = time_stn_grp_ref_xs[sub_ref_idxs]
                sub_ref_ys = time_stn_grp_ref_ys[sub_ref_idxs]
                sub_ref_data = time_stn_grp_data_vals[
                    :, sub_ref_idxs].copy('c')

                if interp_type == 'EDK':
                    sub_ref_drifts = time_stn_grp_drifts[sub_ref_idxs, :]
                    sub_dst_drifts = drft_arrs[:, time_neb_idxs_grp]

                else:
                    sub_ref_drifts = None
                    sub_dst_drifts = None

                sub_pts_flags[:] = False
                interp_vals = self._interp2(
                    sub_dst_xs,
                    sub_dst_ys,
                    sub_ref_xs,
                    sub_ref_ys,
                    sub_ref_data,
                    interp_type,
                    sub_ref_drifts,
                    sub_dst_drifts,
                    idw_exp,
                    krg_flag,
                    models)

                pts_done_flags[time_neb_idxs_grp] = True
                sub_pts_flags[time_neb_idxs_grp] = True

                if self._cntn_idxs is not None:
                    for j, time_idx in enumerate(interp_flds_time_idxs):
                        interp_flds[
                            time_idx, self._cntn_idxs[sub_pts_flags]] = (
                                interp_vals[j])

                else:
                    for j, time_idx in enumerate(interp_flds_time_idxs):
                        interp_flds[time_idx, sub_pts_flags] = interp_vals[j]

            assert np.all(pts_done_flags)
#         raise Exception

        if self._plot_figs_flag:
            for i in range(fin_date_range.shape[0]):
                if interp_type == 'IDW':
                    model = None

                else:
                    model = vgs_ser.iloc[i]

                self._plot_interp(
                    interp_flds[i].reshape(self._interp_crds_orig_shape),
                    self._crds_df.loc[:, 'X'].values,
                    self._crds_df.loc[:, 'Y'].values,
                    fin_date_range[i],
                    model,
                    interp_type,
                    out_figs_dir)
#
#         with lock:
#             nc_is = np.linspace(beg_idx, end_idx, max_rng + 1, dtype=int)
#             ar_is = nc_is - beg_idx
#
#             nc_hdl = nc.Dataset(str(self._nc_file_path), mode='r+')
#
#             for i in range(max_rng):
#                 nc_hdl[interp_arg[2]][
#                     nc_is[i]:nc_is[i + 1], :, :] = (
#                         interp_flds[ar_is[i]:ar_is[i + 1]])
#
#             nc_hdl.sync()
#             nc_hdl.close()
#
#             interp_flds = None
        return

    def _interp(
            self,
            curr_data_vals,
            model,
            curr_stns,
            interp_time,
            krg_flag,
            curr_x_coords,
            curr_y_coords,
            curr_drift_vals,
            interp_type,
            drft_arrs,
            idw_exp,
            interp_flds,
            interp_fld_idx):

        interp_vals = None

        # TODO: another ratio for this.
        if np.all(curr_data_vals < self._min_var_thr):
            interp_vals = np.full(
                self._interp_x_crds_msh.shape[0], curr_data_vals.mean())

        elif (model == 'nan') or (not curr_stns.shape[0]):
            print('No interpolation on:', interp_time)
            return

        elif krg_flag:
            try:
                interp_vals = self._get_krgd_fld(
                    curr_x_coords,
                    curr_y_coords,
                    curr_data_vals,
                    curr_drift_vals,
                    self._interp_x_crds_msh,
                    self._interp_y_crds_msh,
                    drft_arrs,
                    model,
                    interp_type)

            except Exception as msg:
                if hasattr(interp_time, 'strftime'):
                    time_str = interp_time.strftime('%Y-%m-%dT%H:%M:%S')

                else:
                    time_str = interp_time

                print('Error in kriging on %s:' % time_str, msg)

                raise Exception

        else:
            raise Exception
#             interp_vals = get_idw_arr(
#                 self._interp_x_crds_msh,
#                 self._interp_y_crds_msh,
#                 curr_x_coords,
#                 curr_y_coords,
#                 curr_data_vals,
#                 idw_exp)

        if self._cntn_idxs is not None:
            interp_flds[interp_fld_idx].ravel()[self._cntn_idxs] = interp_vals

        else:
            interp_flds[interp_fld_idx] = interp_vals.reshape(
                self._interp_crds_orig_shape)

        self._mod_min_max(interp_flds[interp_fld_idx])
        return

    def _interp2(
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
            krg_flag,
            models):

        n_dsts = dst_xs.size
        n_refs = ref_xs.size
        n_time = ref_data.shape[0]

        dst_data = np.full((n_time, n_dsts), np.nan)

        if not krg_flag:
            wts = np.full(n_refs, np.nan)
            ref_dst_dists = np.full(n_refs, np.nan)

            for i in range(n_dsts):
                fill_dists_one_pt(
                    dst_xs[i], dst_ys[i], ref_xs, ref_ys, ref_dst_dists)

                wts_sum = fill_wts_and_sum(ref_dst_dists, wts, idw_exp)

                for j in range(n_time):
                    mults_sum = get_mults_sum(wts, ref_data[j])

                    dst_data[j, i] = mults_sum / wts_sum

        else:
            if interp_type in ('OK', 'SK'):
                add_rc = 1

            elif interp_type == 'EDK':
                add_rc = 1 + ref_drfts.shape[1]

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
                if model == 'nan':
                    continue

                fill_vg_var_arr(ref_2d_dists, ref_2d_vars, model)
                fill_vg_var_arr(dst_ref_2d_dists, dst_ref_2d_vars, model)

                if interp_type in ('OK', 'SK'):
                    ref_2d_vars[n_refs, :n_refs] = 1.0
                    ref_2d_vars[:, n_refs] = 1.0
                    ref_2d_vars[n_refs, n_refs] = 0.0
                    dst_ref_2d_vars[:, n_refs] = 1.0

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
                    continue
#                     print(j, ref_2d_vars)
#                     raise Exception

                for i in range(n_dsts):
                    lmds = np.matmul(ref_2d_vars_inv, dst_ref_2d_vars[i])
                    dst_data[j, i] = (lmds[:n_refs] * ref_data[j]).sum()

        return dst_data

    def _interp_new(
            self,
            curr_data_vals,
            model,
            curr_stns,
            interp_time,
            krg_flag,
            curr_x_coords,
            curr_y_coords,
            curr_drift_vals,
            interp_type,
            drft_arrs,
            idw_exp,
            interp_flds,
            interp_fld_idx):

        interp_vals = None

        # TODO: another ratio for this.
        if np.all(curr_data_vals < self._min_var_thr):
            interp_vals = np.full(
                self._interp_x_crds_msh.shape[0], curr_data_vals.mean())

        elif (model == 'nan') or (not curr_stns.shape[0]):
            print('No interpolation on:', interp_time)
            return

        else:
            interp_vals = self._get_interp_fld_new(
                curr_x_coords,
                curr_y_coords,
                curr_data_vals,
                curr_drift_vals,
                model,
                interp_type,
                drft_arrs,
                krg_flag,
                idw_exp)

        if self._cntn_idxs is not None:
            interp_flds[interp_fld_idx].ravel()[self._cntn_idxs] = interp_vals

        else:
            interp_flds[interp_fld_idx] = interp_vals.reshape(
                self._interp_crds_orig_shape)

        self._mod_min_max(interp_flds[interp_fld_idx])
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

        if self._index_type == 'date':
            time_str = interp_time.strftime('%Y_%m_%d_T_%H_%M')

        elif self._index_type == 'obj':
            time_str = interp_time

        else:
            raise NotImplementedError(
                f'Unknown index_type: {self._index_type}!')

        out_fig_name = f'{interp_type.lower()}_{time_str}.png'

        fig, ax = plt.subplots()

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

    def _get_interp_fld_new(
            self,
            curr_x_coords,
            curr_y_coords,
            curr_data_vals,
            curr_drift_vals,
            model,
            interp_type,
            drft_arrs,
            krg_flag,
            idw_exp):

        # configured for n_nebs only!
        n_nebs = 5

        full_neb_idxs, grps_ctr, full_neb_idxs_grps = self._get_pt_neb_idxs(
            curr_x_coords, curr_y_coords, n_nebs=n_nebs, neb_range=None)

        interp_vals = np.full(self._interp_x_crds_msh.size, np.nan)

        for grp_idx in range(grps_ctr):
            same_grp_idxs = np.where(full_neb_idxs_grps[:, 0] == grp_idx)[0]

#             if same_grp_idxs.size != 1:
#                 print(f'grp_idx: {grp_idx}, N: {same_grp_idxs.size}')

            grp_dst_x_crds = self._interp_x_crds_msh[same_grp_idxs]
            grp_dst_y_crds = self._interp_y_crds_msh[same_grp_idxs]

            grp_ref_x_crds = curr_x_coords[full_neb_idxs[same_grp_idxs[0]]]
            grp_ref_y_crds = curr_y_coords[full_neb_idxs[same_grp_idxs[0]]]
            grp_ref_z_crds = curr_data_vals[full_neb_idxs[same_grp_idxs[0]]]

            if drft_arrs is None:
                grp_dst_drift_crds = None
                grp_ref_drift_crds = None

            else:
                grp_dst_drift_crds = drft_arrs[:, same_grp_idxs]
                grp_ref_drift_crds = (
                    curr_drift_vals[:, full_neb_idxs[same_grp_idxs[0]]])

            if krg_flag:
                grp_interp_vals = self._get_krgd_fld(
                    grp_ref_x_crds,
                    grp_ref_y_crds,
                    grp_ref_z_crds,
                    grp_ref_drift_crds,
                    grp_dst_x_crds,
                    grp_dst_y_crds,
                    grp_dst_drift_crds,
                    model,
                    interp_type)

            else:
                raise Exception
#                 grp_interp_vals = get_idw_arr(
#                     grp_dst_x_crds,
#                     grp_dst_y_crds,
#                     grp_ref_x_crds,
#                     grp_ref_y_crds,
#                     grp_ref_z_crds,
#                     idw_exp)

            assert np.all(np.isnan(interp_vals[same_grp_idxs])), (
                'All pts should be NaN!')

            interp_vals[same_grp_idxs] = grp_interp_vals

        assert np.all(np.isfinite(interp_vals)), 'Some points not interpolated!'
        return interp_vals

    def _get_krgd_fld(
            self,
            curr_x_coords,
            curr_y_coords,
            curr_data_vals,
            curr_drift_vals,
            dst_x_crds,
            dst_y_crds,
            dst_drift_crds,
            model,
            interp_type
            ):

        if interp_type == 'OK':
            krige_cls = OrdinaryKriging(
                xi=curr_x_coords,
                yi=curr_y_coords,
                zi=curr_data_vals,
                xk=dst_x_crds,
                yk=dst_y_crds,
                model=model)

        elif interp_type == 'SK':
            krige_cls = SimpleKriging(
                xi=curr_x_coords,
                yi=curr_y_coords,
                zi=curr_data_vals,
                xk=dst_x_crds,
                yk=dst_y_crds,
                model=model)

        elif interp_type == 'EDK':
            krige_cls = ExternalDriftKriging_MD(
                xi=curr_x_coords,
                yi=curr_y_coords,
                zi=curr_data_vals,
                si=curr_drift_vals,
                xk=dst_x_crds,
                yk=dst_y_crds,
                sk=dst_drift_crds,
                model=model)

        else:
            raise ValueError(f'Unknown interpolation type: {interp_type}!')

        krige_cls.krige()

        return krige_cls.zk

#     def _get_pt_neb_idxs(
#             self, curr_x_coords, curr_y_coords, n_nebs=None, neb_range=None):
#
#         # this can be done in two ways:
#             # First: fixed number of nebors
#             # Second: Number of nebors depends on range
#             # In the second case the value of -1 will show where
#             # to stop while reading the nebor indices.
#
#         # write more checks
#         assert any([n_nebs is None, neb_range is None])
#         assert any([n_nebs is not None, neb_range is not None])
#
#         if n_nebs is None:
#             raise NotImplementedError
#
#         n_neb_idxs_cols = None
#
#         if n_nebs is not None:
#             n_neb_idxs_cols = n_nebs
#
#         elif neb_range is not None:
#             n_neb_idxs_cols = curr_x_coords.size
#
#         else:
#             raise NotImplementedError
#
#         full_neb_idxs = np.full(
#             (self._interp_x_crds_msh.size, n_neb_idxs_cols),
#             -1,
#             dtype=int)
#
#         for i in range(self._interp_x_crds_msh.size):
#             interp_x_crd = self._interp_x_crds_msh[i]
#             interp_y_crd = self._interp_y_crds_msh[i]
#
#             dists = (
#                 ((interp_x_crd - curr_x_coords) ** 2) +
#                 ((interp_y_crd - curr_y_coords) ** 2)) ** 0.5
#
#             if n_nebs is not None:
#                 full_neb_idxs[i, :] = np.argsort(np.argsort(dists)[:n_nebs])
#
#             elif neb_range is not None:
#                 neb_idxs = np.where(dists <= neb_range)[0]
#                 full_neb_idxs[i, :neb_idxs.size] = neb_idxs
#
#             else:
#                 raise NotImplementedError
#
#         assert np.all(np.any(full_neb_idxs > -1, axis=0))
#
#         if n_nebs is not None:
#             assert np.all(full_neb_idxs > -1)
#
#         elif neb_range is not None:
#             pass
#
#         else:
#             raise NotImplementedError
#
#         # col 1: grp index
#         # col 2: index in all similar config grps
#         # col 3: N grps that share the same config
#         full_neb_idxs_grps = np.full(
#             (self._interp_x_crds_msh.size, 3), -1, dtype=int)
#
#         grps_ctr = 0
#         for i in range(self._interp_x_crds_msh.size):
#             if full_neb_idxs_grps[i, 0] != -1:
#                 continue
#
#             neb_idxs = full_neb_idxs[i]
#
#             equal_idxs = np.all(full_neb_idxs == neb_idxs, axis=1)
#
#             n_equal_idxs = int(equal_idxs.sum())
#
#             full_neb_idxs_grps[equal_idxs, 0] = grps_ctr
#             full_neb_idxs_grps[equal_idxs, 1] = np.arange(1, n_equal_idxs + 1)
#             full_neb_idxs_grps[equal_idxs, 2] = n_equal_idxs
#
#             grps_ctr += 1
#
#         if n_nebs is not None:
#             assert np.all(full_neb_idxs_grps > -1)
#
#         elif neb_range is not None:
#             pass
#
#         else:
#             raise NotImplementedError
#
#         return full_neb_idxs, grps_ctr, full_neb_idxs_grps

    def _get_pt_neb_idxs(
            self, curr_x_coords, curr_y_coords, n_nebs=None, neb_range=None):

        # this can be done in two ways:
            # First: fixed number of nebors
            # Second: Number of nebors depends on range
            # In the second case the value of -1 will show where
            # to stop while reading the nebor indices.

        # write more checks
        assert any([n_nebs is None, neb_range is None])
        assert any([n_nebs is not None, neb_range is not None])

        n_nebs_tot = curr_x_coords.size

        if n_nebs is None:
            raise NotImplementedError

        n_neb_idxs_cols = None

        if n_nebs is not None:
            n_neb_idxs_cols = n_nebs

        elif neb_range is not None:
            n_neb_idxs_cols = curr_x_coords.size

        else:
            raise NotImplementedError

        if n_nebs is not None:
            n_pies = 16
            n_refs_per_pie = 30
            min_dist_thresh = 0
            dists = np.zeros(n_nebs_tot, dtype=np.float64)
            ref_pie_idxs = np.zeros(n_nebs_tot, dtype=np.uint32)
            tem_ref_sel_dists = np.zeros(n_nebs_tot, dtype=np.float64)
#             srtd_tem_ref_sel_dist_idxs = np.zeros(n_nebs_tot, dtype=np.int64)
            ref_sel_pie_idxs = np.zeros(n_nebs_tot, dtype=np.int64)
            ref_pie_cts = np.zeros(n_pies, dtype=np.uint32)

        full_neb_idxs = np.full(
            (self._interp_x_crds_msh.size, n_neb_idxs_cols),
            -1,
            dtype=int)

        for i in range(self._interp_x_crds_msh.size):
            interp_x_crd = self._interp_x_crds_msh[i]
            interp_y_crd = self._interp_y_crds_msh[i]

#             dists = (
#                 ((interp_x_crd - curr_x_coords) ** 2) +
#                 ((interp_y_crd - curr_y_coords) ** 2)) ** 0.5

            if n_nebs is not None:
                sel_equidist_refs(
                    interp_x_crd,
                    interp_y_crd,
                    curr_x_coords,
                    curr_y_coords,
                    n_pies,
#                     n_refs_per_pie,
                    min_dist_thresh,
                    -1,
#                     srtd_tem_ref_sel_dist_idxs,
                    dists,
                    tem_ref_sel_dists,
                    ref_sel_pie_idxs,
                    ref_pie_idxs,
                    ref_pie_cts)

#                 print(i, ref_pie_cts)
#                 print(i, ref_sel_pie_idxs)

                ref_sel_pie_idxs_wh = np.where(ref_sel_pie_idxs > -1)[0]
#                 sort_ref_sel_pie_idxs_wh_idxs = np.argsort(ref_sel_pie_idxs[ref_sel_pie_idxs_wh])
#                 fin_nebs_idxs_wh_arr = ref_sel_pie_idxs_wh[sort_ref_sel_pie_idxs_wh_idxs]

                uniq_ref_sel_pie_idxs = np.unique(ref_sel_pie_idxs[ref_sel_pie_idxs_wh])

                full_sorted_nebs_idxs = []
                for uniq_ref_sel_pie_idx in uniq_ref_sel_pie_idxs:
                    same_pie_idxs = np.where(ref_sel_pie_idxs == uniq_ref_sel_pie_idx)[0]
                    same_pie_dist_sorted_idxs = same_pie_idxs[np.argsort(dists[same_pie_idxs])]

                    full_sorted_nebs_idxs.extend(same_pie_dist_sorted_idxs.tolist())

                full_sorted_nebs_idxs = np.array(full_sorted_nebs_idxs)

                full_neb_idxs[i, :] = full_sorted_nebs_idxs[:n_nebs]
#                 full_neb_idxs[i, :] = np.sort(fin_nebs_idxs_wh_arr[:n_nebs])

#                 dists = (
#                     ((interp_x_crd - curr_x_coords) ** 2) +
#                     ((interp_y_crd - curr_y_coords) ** 2)) ** 0.5
#
#                 full_neb_idxs[i, :] = np.sort(np.argsort(dists)[:n_nebs])

#                 print(i, full_neb_idxs[i])

            elif neb_range is not None:
                dists = (
                    ((interp_x_crd - curr_x_coords) ** 2) +
                    ((interp_y_crd - curr_y_coords) ** 2)) ** 0.5

                neb_idxs = np.where(dists <= neb_range)[0]
                full_neb_idxs[i, :neb_idxs.size] = neb_idxs

            else:
                raise NotImplementedError

        assert np.all(np.any(full_neb_idxs > -1, axis=0))

        if n_nebs is not None:
            assert np.all(full_neb_idxs > -1)

        elif neb_range is not None:
            pass

        else:
            raise NotImplementedError

        # col 1: grp index
        # col 2: index in all similar config grps
        # col 3: N grps that share the same config
        full_neb_idxs_grps = np.full(
            (self._interp_x_crds_msh.size, 3), -1, dtype=int)

        grps_ctr = 0
        for i in range(self._interp_x_crds_msh.size):
            if full_neb_idxs_grps[i, 0] != -1:
                continue

            neb_idxs = full_neb_idxs[i]

            equal_idxs = np.all(full_neb_idxs == neb_idxs, axis=1)

            n_equal_idxs = int(equal_idxs.sum())

            full_neb_idxs_grps[equal_idxs, 0] = grps_ctr
            full_neb_idxs_grps[equal_idxs, 1] = np.arange(1, n_equal_idxs + 1)
            full_neb_idxs_grps[equal_idxs, 2] = n_equal_idxs

            grps_ctr += 1

        if n_nebs is not None:
            assert np.all(full_neb_idxs_grps > -1)

        elif neb_range is not None:
            pass

        else:
            raise NotImplementedError

        return full_neb_idxs, grps_ctr, full_neb_idxs_grps
