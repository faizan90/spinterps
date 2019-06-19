'''
Created on Nov 26, 2018

@author: Faizan-Uni
'''
import numpy as np
import netCDF4 as nc

import matplotlib.pyplot as plt
from descartes import PolygonPatch

from ..misc import traceback_wrapper
from ..cyth import (
    OrdinaryKriging,
    SimpleKriging,
    ExternalDriftKriging_MD,
    get_idw_arr,
    slct_nebrs_cy)

plt.ioff()


class SpInterpSteps:

    def __init__(self, spinterp_main_cls):

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
            '_index_type',
            '_nc_file_path',
            ]

        self._debug = False

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

        curr_drift_vals = None

        fin_date_range = data_df.index

        interp_flds = np.full(
            (fin_date_range.shape[0],
             self._interp_crds_orig_shape[0],
             self._interp_crds_orig_shape[1]),
            np.nan,
            dtype=np.float32)

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

#             self._interp(
#                 curr_data_vals,
#                 model,
#                 curr_stns,
#                 interp_time,
#                 krg_flag,
#                 curr_x_coords,
#                 curr_y_coords,
#                 curr_drift_vals,
#                 interp_type,
#                 drft_arrs,
#                 idw_exp,
#                 interp_flds,
#                 i)

            self._interp_new(
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
                i)

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
            nc_is = np.linspace(beg_idx, end_idx, max_rng + 1, dtype=int)
            ar_is = nc_is - beg_idx

            nc_hdl = nc.Dataset(str(self._nc_file_path), mode='r+')

            for i in range(max_rng):
                nc_hdl[interp_arg[2]][
                    nc_is[i]:nc_is[i + 1], :, :] = (
                        interp_flds[ar_is[i]:ar_is[i + 1]])

            nc_hdl.sync()
            nc_hdl.close()

            interp_flds = None
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
            interp_vals = get_idw_arr(
                self._interp_x_crds_msh,
                self._interp_y_crds_msh,
                curr_x_coords,
                curr_y_coords,
                curr_data_vals,
                idw_exp)

        if self._cntn_idxs is not None:
            interp_flds[interp_fld_idx].ravel()[self._cntn_idxs] = interp_vals

        else:
            interp_flds[interp_fld_idx] = interp_vals.reshape(
                self._interp_crds_orig_shape)

        self._mod_min_max(interp_flds[interp_fld_idx])
        return

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
                grp_interp_vals = get_idw_arr(
                    grp_dst_x_crds,
                    grp_dst_y_crds,
                    grp_ref_x_crds,
                    grp_ref_y_crds,
                    grp_ref_z_crds,
                    idw_exp)

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

    @traceback_wrapper
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

        if n_nebs is None:
            raise NotImplementedError

        n_neb_idxs_cols = None

        if n_nebs is not None:
            n_neb_idxs_cols = n_nebs

        elif neb_range is not None:
            n_neb_idxs_cols = curr_x_coords.size

        else:
            raise NotImplementedError

        full_neb_idxs = np.full(
            (self._interp_x_crds_msh.size, n_neb_idxs_cols),
            -1,
            dtype=int)

        for i in range(self._interp_x_crds_msh.size):
            interp_x_crd = self._interp_x_crds_msh[i]
            interp_y_crd = self._interp_y_crds_msh[i]

            dists = (
                ((interp_x_crd - curr_x_coords) ** 2) +
                ((interp_y_crd - curr_y_coords) ** 2)) ** 0.5

            if n_nebs is not None:
                full_neb_idxs[i, :] = np.argsort(np.argsort(dists)[:n_nebs])

            elif neb_range is not None:
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
