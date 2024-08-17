'''
Created on Nov 26, 2018

@author: Faizan-Uni
'''
import timeit

import numpy as np
import netCDF4 as nc

# from .vgclus import VariogramCluster as VC
# from .grps import SpInterpNeighborGrouping as SIG
from ..mpg import fill_shm_arrs
from ..misc import traceback_wrapper  # , check_full_nuggetness  # , print_sl, print_el
from .za_nnb import NNB
from ..cyth import (
    fill_wts_and_sum,
    get_mults_sum,
    fill_dists_2d_mat,
    fill_vg_var_arr,
    copy_2d_arr_at_idxs)


def print_sl():
    return


print_el = print_sl

'''
What is supposed to happen...

Needed New:
    - A grouping object, that tell which cells at which time steps have same
      config (data, vgs, etc.).
    - vg_var arrays can be float32.
    - Drop Simple Kriging!
    - A dict like object can be used to fill the variogram array and hold the
      results up to a given length like rsmp.
    - It is more straight forward to interpolate a point in time.

Case: NNB
---------

    INPUTS
    ------
    - Array of interp crds as shm.
    - Array of data   crds as shm.
    - SIG with all the configs.
    - Array of ref-to-ref distances as shm.
    - Array of dst-to-dst distances as shm.
    - Array of interp values as shm.
    - Array with indices for nearest neigbors.

    PROCESS
    -------
    - Assign nearest nebors.
    - Fill interp values array.

    OUTPUT
    ------
    - Nothing.
    - Other procs write to the output netCDF file, plot, etc.

Case: IDW
---------

    INPUTS
    ------
    - Array of interp crds as shm.
    - Array of data   crds as shm.
    - SIG with all the configs.
    - Array of ref-to-ref distances as shm.
    - Array of dst-to-dst distances as shm.
    - Array of interp values as shm.
    - Based on max nebors, for each cell, indices of data at each time step.

    PROCESS
    -------
    - For a given group of data values, normalize distance based on maximum.
    - Then compute weights.
    - Fill interp values array.

    OUTPUT
    ------
    - Nothing.
    - Other procs write to the output netCDF file, plot, etc.
'''


class SpInterpSteps:

    def __init__(self, spinterp_main_cls):

        read_labs = [
            '_vb',
            '_mp_flag',
            '_crds_df',
            '_min_var_thr',
            '_min_var_cut',
            '_max_var_cut',
            '_cntn_idxs',
            '_interp_crds_orig_shape',
            # '_interp_x_crds_msh',
            # '_interp_y_crds_msh',
            '_nc_file_path',
            '_neb_sel_mthd',
            '_n_nebs',
            '_n_pies',
            '_min_vg_val',
            '_interp_flag_est_vars',
            '_intrp_dtype',
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

    def _get_all_interp_outputs(self, args):

        (data_df,
         beg_idx,
         end_idx,
         max_rng,
         interp_args,
         lock,
         fld_beg_row,
         fld_end_row,
         sis_shm_ags,
        ) = args

        interp_beg_time = timeit.default_timer()

        if self._mp_flag: fill_shm_arrs(sis_shm_ags)

        interp_labels = [interp_arg[2] for interp_arg in interp_args]

        assert data_df.shape[1] == self._crds_df.shape[0], (
            data_df.shape, self._crds_df.shape)

        assert self._cntn_idxs.size == np.prod(
            self._interp_crds_orig_shape, dtype=np.uint64), (
                self._cntn_idxs.size, self._interp_crds_orig_shape)

        assert (sis_shm_ags.sim_ref_dts.shape[0] ==
                self._cntn_idxs.sum(dtype=np.uint64)), (
            sis_shm_ags.sim_ref_dts.shape,
            self._cntn_idxs.sum(dtype=np.uint64))

        dte_fnt_ixs = np.isfinite(data_df.values)

        print('Start...')
        for ipn_lab in interp_labels:

            ipn_ary = getattr(sis_shm_ags, ipn_lab)

            l = 0
            for i in range(fld_beg_row, fld_end_row, 1):
                for j in range(self._interp_crds_orig_shape[1]):

                    k = (i * self._interp_crds_orig_shape[1]) + j

                    if not self._cntn_idxs[k]: continue

                    nnb_obj = NNB(
                        sis_shm_ags.sim_ref_dts[l], data_df, dte_fnt_ixs)

                    ipn_ary[beg_idx:end_idx, i, j] = nnb_obj.get_ipn_vls_fst()

                    l += 1

                print(i, beg_idx, end_idx)

        print('Done.')

        return (
            lock,
            beg_idx,
            end_idx,
            data_df,
            max_rng,
            interp_labels,
            sis_shm_ags,
            fld_beg_row,
            fld_end_row,
            interp_beg_time)

    def _write_to_disk(self, args):

        (lock,
         beg_idx,
         end_idx,
         data_df,
         max_rng,
         interp_labels,
         sis_shm_ags,
         fld_beg_row,
         fld_end_row,
         interp_beg_time) = args

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

            nc_is = np.linspace(beg_idx, end_idx, max_rng + 1, dtype=int)
            ar_is = nc_is - beg_idx

            for interp_label in interp_labels:
                interp_flds = getattr(sis_shm_ags, interp_label)

                nc_ds = nc_hdl[interp_label]

                for i in range(max_rng):
                    nc_ds[nc_is[i]:nc_is[i + 1],
                          fld_beg_row:fld_end_row,:] = (
                            interp_flds[ar_is[i]:ar_is[i + 1]])

                nc_hdl.sync()

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
                (n_time, n_dsts), np.nan, dtype=self._intrp_dtype)

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

                    if self._interp_flag_est_vars: est_vars[j,:] = 0.0

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
                    for i in range(n_dsts):
                        # self._fill_nnb_dst_data(
                        #     dst_ref_2d_dists_arr_sub,
                        #     ref_data[j],
                        #     dst_data,
                        #     j,
                        #     i)

                        dst_data[j , i] = ref_data[
                            j, np.argmin(dst_ref_2d_dists_arr_sub[i])]

                    if self._interp_flag_est_vars: est_vars[j,:] = 0.0

                else:
                    self._fill_krig_idw_dst_data(
                        n_refs,
                        n_dsts,
                        ref_2d_vars_inv,
                        dst_ref_2d_vars_arr_sub,
                        dst_ref_2d_dists_arr_sub,
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
            ref_data,
            j,
            dst_data,
            est_vars):

        for i in range(n_dsts):
            lmds = np.matmul(ref_2d_vars_inv, dst_ref_2d_vars_arr_sub[i])

            if not np.isclose(lmds[:n_refs].sum(), 1.0):

                # NNB used when kriging fails.

                dst_data[j, i] = ref_data[
                    j, np.argmin(dst_ref_2d_dists_arr_sub[i])]

                if self._interp_flag_est_vars:
                    est_vars[j, i] = 0.0

            else:
                dst_data[j, i] = (lmds[:n_refs] * ref_data[j]).sum()

                if self._interp_flag_est_vars:
                    est_vars[j, i] = (
                        (lmds * dst_ref_2d_vars_arr_sub[i]).sum() +
                        lmds[n_refs])
        return

    # def _fill_nnb_dst_data(
    #         self,
    #         dst_ref_2d_dists_arr_sub,
    #         ref_data_j,
    #         dst_data,
    #         j,
    #         i):
    #
    #     dst_data[j, i] = ref_data_j[np.argmin(dst_ref_2d_dists_arr_sub[i])]
    #     return

    # def _fill_idw_dst_data(
    #         self,
    #         n_dsts,
    #         dst_ref_2d_dists_arr_sub,
    #         wts,
    #         ref_data_j,
    #         dst_data,
    #         j,
    #         idw_exp):
    #
    #     for i in range(n_dsts):
    #         wts_sum = fill_wts_and_sum(dst_ref_2d_dists_arr_sub[i], wts, idw_exp)
    #
    #         mults_sum = get_mults_sum(wts, ref_data_j)
    #
    #         dst_data[j, i] = mults_sum / wts_sum
    #     return

    @np.errstate(invalid='ignore')
    def _mod_min_max(self, interp_fld):

        # with np.errstate(invalid='ignore'):
        if self._min_var_cut is not None:
            interp_fld[interp_fld < self._min_var_cut] = self._min_var_cut

        if self._max_var_cut is not None:
            interp_fld[interp_fld > self._max_var_cut] = self._max_var_cut

        return

