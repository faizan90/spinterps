# -*- coding: utf-8 -*-

'''
Created on Aug 17, 2024

@author: Faizan-TU Munich
'''

import numpy as np

from ..cyth import (
    # fill_wts_and_sum,
    # get_mults_sum,
    fill_dists_2d_mat,
    # fill_vg_var_arr,
    # copy_2d_arr_at_idxs,
    )

from ..mpg import SHMARY


class SpInterpsStepsSHM:

    def __init__(self, sim_obj):

        # krg_flg = any([sim_obj._ork_flag, sim_obj._edk_flag])

        # This one needs to be taken care of in prepare.py.
        sim_shp = sim_obj._interp_x_crds_msh.shape  #  A ravel.
        # sim_xcs_msh_shm = SHMARY(sim_shp, np.float64, 'c', 'sim_xcs')
        # sim_ycs_msh_shm = SHMARY(sim_shp, np.float64, 'c', 'sim_ycs')

        ref_shp = (sim_obj._crds_df.shape[0],)
        # No need to shm ref cds. They are too small. Maybe later.
        # ref_xcs_shm = SHMARY(ref_shp, np.float64, 'c', 'ref_xcs')
        # ref_ycs_shm = SHMARY(ref_shp, np.float64, 'c', 'ref_ycs')
        #======================================================================

        # if krg_flg:
        #     ref_ref_shp = (ref_shp[0], ref_shp[0])
        #     ref_ref_dts_shm = SHMARY(
        #         ref_ref_shp, np.float64, 'c', 'ref_ref_dts')
        #
        #     self._fill_dists_arr(
        #         sim_obj._crds_df[:, 'X'].values,
        #         sim_obj._crds_df[:, 'Y'].values,
        #         sim_obj._crds_df[:, 'X'].values,
        #         sim_obj._crds_df[:, 'Y'].values,
        #         ref_ref_dts_shm.shr,
        #         )
        #======================================================================

        sim_ref_shp = (sim_shp[0], ref_shp[0])
        sim_ref_dts_shm = SHMARY(sim_ref_shp, np.float64, 'c', 'sim_ref_dts')

        # _interp_x_crds_msh is not shm for now.
        # self._fill_dists_arr(
        #     sim_obj._sim_xcs_msh_shm.shr,
        #     sim_obj._sim_ycs_msh_shm.shr,
        #     sim_obj._crds_df[:, 'X'].values,
        #     sim_obj._crds_df[:, 'Y'].values,
        #     sim_ref_dts_shm.shr,
        #     )

        self._fill_dists_arr(
            sim_obj._interp_x_crds_msh,
            sim_obj._interp_y_crds_msh,
            sim_obj._crds_df[:, 'X'].values,
            sim_obj._crds_df[:, 'Y'].values,
            sim_ref_dts_shm.shr,
            )
        #======================================================================

        ipn_fld_shp = (
            sim_obj._data_df.shape[0], * sim_obj._interp_crds_orig_shape)

        ipn_fds_shm_dct = {}
        for ipn_lab in [interp_arg[0] for interp_arg in sim_obj._interp_args]:

            ipn_fld_shm = SHMARY(
                ipn_fld_shp, sim_obj._intrp_dtype, 'c', ipn_lab)

            ipn_fds_shm_dct[ipn_lab] = ipn_fld_shm
        #======================================================================
        return

    def _fill_dists_arr(self, x1s, y1s, x2s, y2s, dists_arr):

        assert x1s.ndim == y1s.ndim == x2s.ndim == y2s.ndim == 1

        assert x1s.size == y1s.size
        assert x2s.size == y2s.size

        fill_dists_2d_mat(x1s, y1s, x2s, y2s, dists_arr)
        return
