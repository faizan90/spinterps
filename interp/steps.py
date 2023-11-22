'''
Created on Nov 26, 2018

@author: Faizan-Uni
'''
import timeit

import numpy as np
import netCDF4 as nc

from .vgclus import VariogramCluster as VC
from .grps import SpInterpNeighborGrouping as SIG
from ..misc import traceback_wrapper, check_full_nuggetness  # , print_sl, print_el
from ..cyth import (
    fill_wts_and_sum,
    get_mults_sum,
    fill_dists_2d_mat,
    fill_vg_var_arr,
    copy_2d_arr_at_idxs)


def print_sl():
    return


print_el = print_sl


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
            '_cntn_idxs',
            '_interp_crds_orig_shape',
            '_interp_x_crds_msh',
            '_interp_y_crds_msh',
            '_nc_file_path',
            '_neb_sel_mthd',
            '_n_nebs',
            '_n_pies',
            '_min_vg_val',
            '_interp_flag_est_vars',
            ]

        for read_lab in read_labs:
            setattr(self, read_lab, getattr(spinterp_main_cls, read_lab))

        # self._n_dst_pts = self._interp_x_crds_msh.shape[0]
        return

    @traceback_wrapper
    def interpolate_subset(self, args_for_interp):

        # Calling like this allows for proper garbage collection,
        # in case I forgot to dereference some variables that require
        # a lot of RAM.

        args_for_disk = self._get_all_interp_outputs(args_for_interp)

        self._write_to_disk(args_for_disk)
        return

    def _get_dists_arr(self, x1s, y1s, x2s, y2s):

        assert x1s.ndim == y1s.ndim == x2s.ndim == y2s.ndim == 1

        assert x1s.size == y1s.size
        assert x2s.size == y2s.size

        n1s, n2s = x1s.size, x2s.size

        dists_arr = np.full((n1s, n2s), np.nan)

        fill_dists_2d_mat(x1s, y1s, x2s, y2s, dists_arr)

        return dists_arr

    def _get_svars_arr(self, dists_arr, vgs, interp_types, diag_mat_flag):

        assert isinstance(interp_types, (list, tuple))

        diag_mat_flag = int(diag_mat_flag)

        assert diag_mat_flag >= 0

        var_flag = 0
        if any([interp_type in interp_types for interp_type in ['OK', 'EDK']]):
            var_flag = 1

        covar_flag = 0
        if 'SK' in interp_types:
            covar_flag = 1

        # Keys are (vg and covar_flag).
        # covar_flag is 1 for SK and 0 for OK and EDK.
        svars_dict = {}
        for vg in vgs:

            assert vg != 'nan'

            if var_flag:
                svars_arr = np.full_like(dists_arr, np.nan)

                fill_vg_var_arr(
                    dists_arr,
                    svars_arr,
                    0,
                    diag_mat_flag,
                    vg,
                    self._min_vg_val)

                svars_dict[(vg, 0)] = svars_arr

            if covar_flag:
                # With covar being 1.
                svars_arr = np.full_like(dists_arr, np.nan)

                fill_vg_var_arr(
                    dists_arr,
                    svars_arr,
                    covar_flag,
                    diag_mat_flag,
                    vg,
                    self._min_vg_val)

                svars_dict[(vg, 1)] = svars_arr

        return svars_dict

    def _get_2d_arr_subset(
            self, arr, row_idxs, col_idxs, rows_add_rc=0, cols_add_rc=0):

        assert arr.ndim == 2

        assert row_idxs.ndim == 1
        assert col_idxs.ndim == 1

        assert arr.size
        assert row_idxs.size
        assert col_idxs.size

        assert row_idxs.min() >= 0
        assert row_idxs.max() < arr.shape[0]

        assert col_idxs.min() >= 0
        assert col_idxs.max() < arr.shape[1]

        assert rows_add_rc >= 0
        assert cols_add_rc >= 0

        subset_arr = np.full(
            (row_idxs.size + rows_add_rc, col_idxs.size + cols_add_rc),
            np.nan,
            dtype=np.float64)

        copy_2d_arr_at_idxs(arr, row_idxs, col_idxs, subset_arr)

        return subset_arr

    def _get_vars_arr_subset(
            self,
            model,
            interp_type,
            ref_drfts,
            ref_ref_2d_vars_arr_all,
            dst_ref_2d_vars_arr_all,
            ref_ref_sub_idxs,
            time_neb_idxs_grp,
            dst_drfts):

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

        ref_ref_2d_vars_arr_sub = self._get_2d_arr_subset(
            ref_ref_2d_vars_arr_all[(model, covar_flag)],
            ref_ref_sub_idxs,
            ref_ref_sub_idxs,
            add_rc,
            add_rc)

        dst_ref_2d_vars_arr_sub = self._get_2d_arr_subset(
            dst_ref_2d_vars_arr_all[(model, covar_flag)],
            time_neb_idxs_grp,
            ref_ref_sub_idxs,
            0,
            add_rc)

        n_refs_sub = ref_ref_sub_idxs.size

        if interp_type == 'OK':
            ref_ref_2d_vars_arr_sub[n_refs_sub,:n_refs_sub] = 1.0
            ref_ref_2d_vars_arr_sub[:, n_refs_sub] = 1.0
            ref_ref_2d_vars_arr_sub[n_refs_sub, n_refs_sub] = 0.0
            dst_ref_2d_vars_arr_sub[:, n_refs_sub] = 1.0

        elif interp_type == 'SK':
            pass

        elif interp_type == 'EDK':
            ref_ref_2d_vars_arr_sub[n_refs_sub,:n_refs_sub] = 1.0
            ref_ref_2d_vars_arr_sub[:n_refs_sub, n_refs_sub] = 1.0
            dst_ref_2d_vars_arr_sub[:, n_refs_sub] = 1.0
            ref_ref_2d_vars_arr_sub[n_refs_sub:, n_refs_sub:] = 0.0

            for k in range(ref_drfts.shape[1]):
                ref_ref_2d_vars_arr_sub[n_refs_sub + 1 + k,:n_refs_sub] = (
                    ref_drfts[:, k])

                ref_ref_2d_vars_arr_sub[:n_refs_sub, n_refs_sub + 1 + k] = (
                    ref_drfts[:, k])

                dst_ref_2d_vars_arr_sub[:, n_refs_sub + 1 + k] = dst_drfts[k,:]

        else:
            raise NotImplementedError

        if False:
            assert np.all(np.isfinite(ref_ref_2d_vars_arr_sub))
            assert np.all(np.isfinite(dst_ref_2d_vars_arr_sub))

        return ref_ref_2d_vars_arr_sub, dst_ref_2d_vars_arr_sub

    def _get_interp(
            self,
            ref_data,
            interp_type,
            ref_drfts,
            dst_drfts,
            idw_exp,
            models,
            interp_steps_flags,
            nuggetness_flags,
            sub_time_steps,
            prblm_time_steps,
            dst_ref_2d_dists_arr_sub,
            ref_ref_2d_vars_arr_all,
            dst_ref_2d_vars_arr_all,
            ref_ref_sub_idxs,
            time_neb_idxs_grp):

        n_dsts = dst_ref_2d_dists_arr_sub.shape[0]
        n_refs = ref_data.shape[1]
        n_time = ref_data.shape[0]

        dst_data = np.full((n_time, n_dsts), np.nan)

        if self._interp_flag_est_vars:
            est_vars = np.full(
                (n_time, n_dsts), np.nan, dtype=np.float64)

        else:
            est_vars = None

        ref_means = ref_data.mean(axis=1)

        wts = np.full(n_refs, np.nan)

        # If only one neighbor is present, then the result is the neighbor
        # itself. No need for interpolation.
        if n_refs == 1:
            for j in range(n_time): dst_data[j,:] = ref_data[j,:]

        elif interp_type == 'NNB':

            for i in range(n_dsts):
                nnb_idx = np.argmin(dst_ref_2d_dists_arr_sub[i])

                for j in range(n_time):
                    dst_data[j, i] = ref_data[j, nnb_idx]

        elif interp_type == 'IDW':

            for i in range(n_dsts):
                wts_sum = fill_wts_and_sum(
                    dst_ref_2d_dists_arr_sub[i], wts, idw_exp)

                assert (wts_sum >= 1e-14), wts_sum

                for j in range(n_time):
                    if interp_steps_flags[j]:
                        mults_sum = get_mults_sum(wts, ref_data[j])
                        dst_data[j, i] = mults_sum / wts_sum

                    else:
                        dst_data[j, i] = ref_means[j]

        elif interp_type in ('OK', 'SK', 'EDK'):
            old_model = ''
            last_failed_flag = False

            # Sorting is better to avoid flipping between same strings.
            models_sort_idxs = np.argsort(models)

            for j in models_sort_idxs:
                model = models[j]

                if (not interp_steps_flags[j]) or (nuggetness_flags[j]):

                    dst_data[j,:] = ref_means[j]
                    continue

                if model == 'nan':
                    raise ValueError('NaN VG!')

                if model != old_model:
                    (ref_ref_2d_vars_arr_sub,
                     dst_ref_2d_vars_arr_sub) = self._get_vars_arr_subset(
                        model,
                        interp_type,
                        ref_drfts,
                        ref_ref_2d_vars_arr_all,
                        dst_ref_2d_vars_arr_all,
                        ref_ref_sub_idxs,
                        time_neb_idxs_grp,
                        dst_drfts)

                    old_model = model

                    try:
                        ref_2d_vars_inv = np.linalg.pinv(
                            ref_ref_2d_vars_arr_sub)

                        # # This comes after inversion is successful.
                        # eig_vals = np.linalg.eigvals(
                        #     ref_ref_2d_vars_arr_sub).round(14)
                        #
                        # if np.any(eig_vals) <= 0:
                        #     raise ValueError

                        last_failed_flag = False

                    except Exception:
                        last_failed_flag = True

                        if last_failed_flag:
                            if (sub_time_steps[j] not in prblm_time_steps):
                                prblm_time_steps.append(sub_time_steps[j])

                            ref_2d_vars_inv = np.nan

                if last_failed_flag:
                    self._fill_idw_dst_data(
                        n_dsts,
                        dst_ref_2d_dists_arr_sub,
                        wts,
                        ref_data[j],
                        dst_data,
                        j,
                        2.0)

                else:
                    self._fill_krig_idw_dst_data(
                        n_refs,
                        n_dsts,
                        ref_2d_vars_inv,
                        dst_ref_2d_vars_arr_sub,
                        dst_ref_2d_dists_arr_sub,
                        wts,
                        ref_data,
                        j,
                        dst_data,
                        est_vars)

        else:
            raise NotImplementedError(interp_type)

        return dst_data, est_vars

    def _fill_krig_idw_dst_data(
            self,
            n_refs,
            n_dsts,
            ref_2d_vars_inv,
            dst_ref_2d_vars_arr_sub,
            dst_ref_2d_dists_arr_sub,
            wts,
            ref_data,
            j,
            dst_data,
            est_vars):

        for i in range(n_dsts):
            lmds = np.matmul(ref_2d_vars_inv, dst_ref_2d_vars_arr_sub[i])

            if not np.isclose(lmds[:n_refs].sum(), 1.0):

                # IDW used when kriging fails.
                # TODO: have a flag for this.

                self._fill_idw_dst_data(
                    n_dsts,
                    dst_ref_2d_dists_arr_sub,
                    wts,
                    ref_data[j],
                    dst_data,
                    j,
                    2.0)

            else:
                dst_data[j, i] = (lmds[:n_refs] * ref_data[j]).sum()

            if self._interp_flag_est_vars:
                est_vars[j, i] = (
                    (lmds * dst_ref_2d_vars_arr_sub[i]).sum() +
                    lmds[n_refs])
        return

    def _fill_idw_dst_data(
            self,
            n_dsts,
            dst_ref_2d_dists_arr_sub,
            wts,
            ref_data_j,
            dst_data,
            j,
            idw_exp):

        for i in range(n_dsts):
            wts_sum = fill_wts_and_sum(dst_ref_2d_dists_arr_sub[i], wts, idw_exp)

            mults_sum = get_mults_sum(wts, ref_data_j)

            dst_data[j, i] = mults_sum / wts_sum
        return

    @np.errstate(invalid='ignore')
    def _mod_min_max(self, interp_fld):

        # with np.errstate(invalid='ignore'):
        if self._min_var_cut is not None:
            interp_fld[interp_fld < self._min_var_cut] = self._min_var_cut

        if self._max_var_cut is not None:
            interp_fld[interp_fld > self._max_var_cut] = self._max_var_cut

        return

    def _get_all_interp_outputs(self, args):

        (data_df,
         beg_idx,
         end_idx,
         max_rng,
         interp_args,
         lock,
         drft_arrs,
         stns_drft_df,
         vgs_ser,
         vgs_rord_tidxs_ser,
         fld_beg_row,
         fld_end_row,
        ) = args

        interp_beg_time = timeit.default_timer()

        interp_types = [interp_arg[0] for interp_arg in interp_args]
        interp_labels = [interp_arg[2] for interp_arg in interp_args]

        krg_flag = any(
            [krg_type in interp_types for krg_type in ('OK', 'SK', 'EDK')])

        edk_flag = 'EDK' in interp_types

        if krg_flag:
            assert np.all(vgs_ser != 'nan'), 'NaN VGs not allowed!'

        time_steps = data_df.index
        #======================================================================

        fld_n_cols = self._interp_crds_orig_shape[1]
        fld_beg_idx = fld_beg_row * fld_n_cols
        fld_end_idx = fld_end_row * fld_n_cols

        fld_grd_shape = (
            (fld_end_row - fld_beg_row), self._interp_crds_orig_shape[1])

        if self._cntn_idxs is not None:
            fld_idxs_range = np.arange(fld_beg_idx, fld_end_idx)

            cntn_idxs_whr = np.where(self._cntn_idxs)[0]
            cntn_idxs_wr_range = np.arange(cntn_idxs_whr.size)

            assert cntn_idxs_whr.size

            msh_idxs = cntn_idxs_wr_range[
                (cntn_idxs_whr >= fld_beg_idx) &
                (cntn_idxs_whr < fld_end_idx)]

            cntn_idxs_whr = cntn_idxs_whr[
                (cntn_idxs_whr >= fld_beg_idx) &
                (cntn_idxs_whr < fld_end_idx)]

            assert fld_idxs_range[0] <= cntn_idxs_whr[0], (
                fld_idxs_range[0], cntn_idxs_whr[0])

            # Shift cntn_idxs_whr values backwards to match the sub field
            # indices.
            cntn_idxs_whr = cntn_idxs_whr - fld_idxs_range[0]

            self._n_dst_pts = msh_idxs.size

            self._interp_x_crds_msh = self._interp_x_crds_msh[msh_idxs]
            self._interp_y_crds_msh = self._interp_y_crds_msh[msh_idxs]

            if drft_arrs is not None:
                assert edk_flag

                drft_arrs = drft_arrs[:, msh_idxs]

            msh_idxs = None
            fld_idxs_range = None
            cntn_idxs_wr_range = None

        else:
            self._n_dst_pts = np.prod(fld_grd_shape)

            self._interp_x_crds_msh = self._interp_x_crds_msh[
                fld_beg_idx:fld_end_idx]

            self._interp_y_crds_msh = self._interp_y_crds_msh[
                fld_beg_idx:fld_end_idx]

            if drft_arrs is not None:
                assert edk_flag

                drft_arrs = drft_arrs[:, fld_beg_idx:fld_end_idx]
        #======================================================================

        assert np.all(data_df.columns == self._crds_df.index)

        # Check uniqueness elsewhere?
        assert np.unique(data_df.columns.values).size == data_df.shape[1]

        if vgs_ser is not None:
            assert not data_df.index.difference(vgs_ser.index).shape[0], (
                'Data and variogram series have non-intersecting indices!')

            vc_cls = VC(vgs_ser)

            vgs_clus_dict, vgs_ser = vc_cls.get_vgs_cluster()

        grp_cls = SIG(
            self._neb_sel_mthd,
            self._n_nebs,
            self._n_pies,
            self._interp_x_crds_msh,
            self._interp_y_crds_msh,
            verbose=self._vb)

        grps_in_time = grp_cls.get_grps_in_time(data_df)

        # TODO: min_nebor_dist_thresh should be related to time and not just
        # proximity. Because, if a very close nebor is active at a time
        # when others are inactive than there is no need to drop one.

        # TODO: Have a threshold of min and max stations for which the
        # neighbors are considered the same.
        #======================================================================

        if vgs_ser is not None:
            ref_ref_2d_dists_arr_all = self._get_dists_arr(
                self._crds_df.loc[:, 'X'].values,
                self._crds_df.loc[:, 'Y'].values,
                self._crds_df.loc[:, 'X'].values,
                self._crds_df.loc[:, 'Y'].values)

            ref_ref_2d_vars_arr_all = self._get_svars_arr(
                ref_ref_2d_dists_arr_all,
                list(vgs_clus_dict.keys()),
                interp_types,
                1)

        else:
            ref_ref_2d_vars_arr_all = None

        # Set to None, to save on space.
        ref_ref_2d_dists_arr_all = None

        dst_ref_2d_dists_arr_all = self._get_dists_arr(
            self._interp_x_crds_msh,
            self._interp_y_crds_msh,
            self._crds_df.loc[:, 'X'].values,
            self._crds_df.loc[:, 'Y'].values)

        if vgs_ser is not None:
            dst_ref_2d_vars_arr_all = self._get_svars_arr(
                dst_ref_2d_dists_arr_all,
                list(vgs_clus_dict.keys()),
                interp_types,
                0)

        else:
            dst_ref_2d_vars_arr_all = None

        self._interp_x_crds_msh = None
        self._interp_y_crds_msh = None
        #======================================================================

        interp_flds_dict = {interp_label:np.full(
            (time_steps.shape[0], np.prod(fld_grd_shape)),
            np.nan,
            dtype=np.float64)
            for interp_label in interp_labels}

        if self._interp_flag_est_vars:
            est_vars_flds_dict = {'EST_VARS_OK':np.full(
                (time_steps.shape[0], np.prod(fld_grd_shape)),
                np.nan,
                dtype=np.float64)}

        else:
            est_vars_flds_dict = None
        #======================================================================

        prblm_time_steps = []

        # time_stn_grp: The stations.
        # cmn_time_stn_grp_idxs: Time steps at which time_stn_grp stations are
        # active.

        for time_stn_grp, cmn_time_stn_grp_idxs in grps_in_time:

            sub_time_steps = time_steps[cmn_time_stn_grp_idxs]

            if not time_stn_grp.size:
                print(sub_time_steps.size, 'step(s) have no station(s)!')

                # There are no stations for these time steps. Hence, no
                # interpolation at all.
                for sub_time_step in sub_time_steps:
                    if sub_time_step in prblm_time_steps:
                        continue

                    prblm_time_steps.append(sub_time_step)

                continue

            assert time_stn_grp.size, 'No stations in time_stn_grp!'

            assert time_stn_grp.size == np.unique(time_stn_grp).size, (
                'Non-unique stations in step group!')

            # These are used to take out a subset of values from the distances
            # and variances arrays.
            time_stn_grp_idxs = self._crds_df.index.get_indexer_for(
                time_stn_grp)

            # Coordinates of time_stn_grp stations.
            # Remember, the order of station labels in data_df and crds_df
            # should be the same.
            time_stn_grp_ref_xs = self._crds_df.loc[time_stn_grp, 'X'].values
            time_stn_grp_ref_ys = self._crds_df.loc[time_stn_grp, 'Y'].values

            # Values of time_stn_grp stations at cmn_time_stn_grp_idxs time
            # steps.
            time_stn_grp_data_vals = data_df.loc[
                cmn_time_stn_grp_idxs, time_stn_grp].values

            if edk_flag:
                # Drift of time_stn_grp stations.
                time_stn_grp_drifts = stns_drft_df.loc[time_stn_grp].values

            else:
                time_stn_grp_drifts = None

            if krg_flag:
                vg_models = vgs_ser.loc[cmn_time_stn_grp_idxs].values
                nuggetness_flags = np.zeros(vg_models.shape[0], dtype=bool)

                for i, vg_model in enumerate(vg_models):
                    nuggetness_flags[i] = check_full_nuggetness(
                        vg_model, self._min_vg_val)

            else:
                vg_models = None
                nuggetness_flags = None

            # time_neb_idxs: time_stn_grp stations that are close to points
            # at time_neb_idxs_grps indices of the interpolation grid.
            time_neb_idxs, time_neb_idxs_grps = grp_cls.get_neb_idxs_and_grps(
                time_stn_grp_ref_xs, time_stn_grp_ref_ys)

            interp_flds_time_idxs = np.where(cmn_time_stn_grp_idxs)[0]

            pts_done_flags = np.zeros(self._n_dst_pts, dtype=bool)
            sub_pts_flags = np.zeros(self._n_dst_pts, dtype=bool)

            for time_neb_idxs_grp in time_neb_idxs_grps:
                # Any index from time_neb_idxs_grp will do as
                # time_neb_idxs have same indices for each time step.
                sub_ref_idxs = time_neb_idxs[time_neb_idxs_grp[0]]

                sub_ref_data = time_stn_grp_data_vals[:, sub_ref_idxs].copy(
                    'c')
                #==============================================================

                ref_ref_sub_idxs = time_stn_grp_idxs[sub_ref_idxs]

                dst_ref_2d_dists_arr_sub = self._get_2d_arr_subset(
                    dst_ref_2d_dists_arr_all,
                    time_neb_idxs_grp,
                    ref_ref_sub_idxs,
                    0,
                    0)
                #==============================================================

                # Take mean when all values are too low.
                interp_steps_flags = np.ones(sub_ref_data.shape[0], dtype=bool)
                for j in range(sub_ref_data.shape[0]):
                    if np.any(sub_ref_data[j] >= self._min_var_thr):
                        continue

                    interp_steps_flags[j] = False

                if edk_flag:
                    sub_ref_drifts = time_stn_grp_drifts[sub_ref_idxs,:]
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
                        idw_exp = 5  # In case inversion fails.

                    interp_vals, est_vars = self._get_interp(
                        sub_ref_data,
                        interp_type,
                        sub_ref_drifts,
                        sub_dst_drifts,
                        idw_exp,
                        vg_models,
                        interp_steps_flags,
                        nuggetness_flags,
                        sub_time_steps,
                        prblm_time_steps,
                        dst_ref_2d_dists_arr_sub,
                        ref_ref_2d_vars_arr_all,
                        dst_ref_2d_vars_arr_all,
                        ref_ref_sub_idxs,
                        time_neb_idxs_grp)

                    self._mod_min_max(interp_vals)

                    interp_flds = interp_flds_dict[interp_labels[i]]

                    if self._interp_flag_est_vars:
                        est_vars_flds = est_vars_flds_dict[
                            f'EST_VARS_{interp_labels[i]}']

                    if self._cntn_idxs is not None:
                        for j, time_idx in enumerate(interp_flds_time_idxs):
                            interp_flds[
                                time_idx, cntn_idxs_whr[sub_pts_flags]] = (
                                    interp_vals[j])

                            if self._interp_flag_est_vars: est_vars_flds[
                                time_idx, cntn_idxs_whr[sub_pts_flags]] = (
                                    est_vars[j])

                    else:
                        for j, time_idx in enumerate(interp_flds_time_idxs):
                            interp_flds[time_idx, sub_pts_flags] = (
                                interp_vals[j])

                            if self._interp_flag_est_vars: est_vars_flds[
                                time_idx, sub_pts_flags] = est_vars[j]

                pts_done_flags[time_neb_idxs_grp] = True

            assert np.all(pts_done_flags), 'Some points not interpolated!'
        #======================================================================

        # Release memory.
        ref_ref_2d_vars_arr_all = None
        dst_ref_2d_dists_arr_all = None
        dst_ref_2d_vars_arr_all = None

        dst_ref_2d_dists_arr_sub = None

        grp_cls = None
        #======================================================================

        if prblm_time_steps:
            print_sl()

            with lock:
                print(
                    'WARNING: There were problems while interpolating '
                    'at the following steps:')

                if len(prblm_time_steps) <= 10:
                    for prblm_stp in prblm_time_steps:
                        print(prblm_stp)

                else:
                    print(prblm_time_steps)

            print_el()

        return (
            lock,
            beg_idx,
            end_idx,
            data_df,
            vgs_ser,
            max_rng,
            interp_labels,
            interp_flds_dict,
            fld_beg_row,
            fld_end_row,
            vgs_rord_tidxs_ser,
            time_steps,
            interp_beg_time,
            est_vars_flds_dict)

    def _write_to_disk(self, args):

        (lock,
         beg_idx,
         end_idx,
         data_df,
         vgs_ser,
         max_rng,
         interp_labels,
         interp_flds_dict,
         fld_beg_row,
         fld_end_row,
         vgs_rord_tidxs_ser,
         time_steps,
         interp_beg_time,
         est_vars_flds_dict) = args

        with lock:
            if self._vb:
                print_sl()

                print(
                    f'Writing data between {beg_idx} and {end_idx} indices '
                    f'to disk...')

                print(
                    'Start and end step of this part:',
                    data_df.index[0],
                    data_df.index[-1])

            nc_hdl = nc.Dataset(str(self._nc_file_path), mode='r+')

            if vgs_ser is None:
                nc_is = np.linspace(beg_idx, end_idx, max_rng + 1, dtype=int)
                ar_is = nc_is - beg_idx

                for interp_label in interp_labels:
                    interp_flds = interp_flds_dict[interp_label]

                    for i in range(max_rng):
                        nc_hdl[interp_label][
                            nc_is[i]:nc_is[i + 1],
                            fld_beg_row:fld_end_row,:] = (
                                interp_flds[ar_is[i]:ar_is[i + 1]])

                    interp_flds_dict[interp_label] = None

                    # est_var_flds = est_vars_flds_dict[f'EST_VARS_{interp_label}']
                    #
                    # for i in range(max_rng):
                    #     nc_hdl[f'EST_VARS_{interp_label}'][
                    #         nc_is[i]:nc_is[i + 1],
                    #         fld_beg_row:fld_end_row,:] = (
                    #             est_var_flds[ar_is[i]:ar_is[i + 1]])
                    #
                    # est_vars_flds_dict[f'EST_VARS_{interp_label}'] = None

                    nc_hdl.sync()

            else:
                for interp_label in interp_labels:
                    interp_flds = interp_flds_dict[interp_label]

                    for i in range(vgs_ser.shape[0]):

                        nc_idx = vgs_rord_tidxs_ser.loc[time_steps[i]]

                        nc_hdl[interp_label][
                            nc_idx, fld_beg_row:fld_end_row,:] = interp_flds[i]

                    interp_flds_dict[interp_label] = None

                    if self._interp_flag_est_vars: est_vars_flds = est_vars_flds_dict[
                        f'EST_VARS_{interp_label}']

                    for i in range(vgs_ser.shape[0]):

                        nc_idx = vgs_rord_tidxs_ser.loc[time_steps[i]]

                        if self._interp_flag_est_vars:
                            nc_hdl[f'EST_VARS_{interp_label}'][
                                nc_idx,
                                fld_beg_row:fld_end_row,:] = est_vars_flds[i]

                    if self._interp_flag_est_vars: est_vars_flds_dict[
                        f'EST_VARS_{interp_label}'] = None

                    nc_hdl.sync()

            interp_flds = None
            est_vars_flds = None

            nc_hdl.close()

            interp_end_time = timeit.default_timer()

            if self._vb:
                print(
                    f'Done writing data between time indices of '
                    f'{beg_idx} and {end_idx} and grid row indices between '
                    f'{fld_beg_row} and {fld_end_row} to disk...')

                print(
                    f'Took {interp_end_time - interp_beg_time:0.1f} '
                    f'seconds to interpolate for this thread.')

                print_el()
        return
