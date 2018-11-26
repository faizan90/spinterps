# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
import os
import timeit
import time

import numpy as np

import pyximport
pyximport.install()
from .idw_nebs import slct_nebrs_cy, get_idw

from krig_single_pts import (OrdinaryKriging,
                             SimpleKriging,
                             ExternalDriftKriging_MD)

def get_all_grid(interp_x_coords_arr,
                 interp_y_coords_arr,
                 nebor_x_coords_arr,
                 nebor_y_coords_arr,
                 nebor_vals_arr,
                 idw_exp,
                 n_quads,
                 n_per_quad,
                 min_dist_thresh,
                 n_nebs,
                 interp_types,
                 vg_model=None,
                 use_close_nebs=True):

    n_interp_pts = interp_x_coords_arr.shape[0]
    n_interp_types = len(interp_types)
    interp_vals_arr = np.full((n_interp_pts, n_interp_types), np.nan)

    if 'idw' in interp_types:
        idw_idx = interp_types.index('idw')

    if 'ok' in interp_types:
        assert vg_model
        ok_idx = interp_types.index('ok')

#    if 'sk' in in_interp_types:
#        assert vg_model
#        sk_idx = interp_types.index('sk')
#
#    if 'edk' in in_interp_types:
#        assert vg_model
#        edk_idx = interp_types.index('edk')

    if vg_model is not None:
        vg_model = str(vg_model)

    if use_close_nebs:
        # arrays to hold temporary values
        dists_arr = np.zeros(n_nebs, dtype=np.float64)
        prcssed_nebrs_arr = np.zeros(n_nebs, dtype=np.uint32)
        slctd_nebrs_arr = np.zeros(n_nebs, dtype=np.uint32)
        slctd_nebrs_dists_arr = np.zeros(n_nebs, dtype=np.float64)
        nebs_idxs_arr = np.zeros(n_nebs, dtype=np.int64)
        fin_nebs_idxs_arr = np.zeros(n_nebs, dtype=np.uint32)

    for i in range(n_interp_pts):
        x = interp_x_coords_arr[i]
        y = interp_y_coords_arr[i]

        if use_close_nebs:
            slct_nebrs_cy(x,
                          y,
                          nebor_x_coords_arr,
                          nebor_y_coords_arr,
                          n_quads,
                          n_per_quad,
                          min_dist_thresh,
                          n_nebs,
                          prcssed_nebrs_arr,
                          slctd_nebrs_arr,
                          nebs_idxs_arr,
                          dists_arr,
                          slctd_nebrs_dists_arr,
                          fin_nebs_idxs_arr)

            fin_nebs_idxs_bool_arr = fin_nebs_idxs_arr.astype(bool)
            fin_nebrs_x_crds = nebor_x_coords_arr[fin_nebs_idxs_bool_arr]
            fin_nebrs_y_crds = nebor_y_coords_arr[fin_nebs_idxs_bool_arr]
            fin_nebrs_vals = nebor_vals_arr[fin_nebs_idxs_bool_arr]

        else:
            fin_nebrs_x_crds = nebor_x_coords_arr
            fin_nebrs_y_crds = nebor_y_coords_arr
            fin_nebrs_vals = nebor_vals_arr

        if fin_nebrs_x_crds.shape[0]:
            if 'idw' in interp_types:
                idw_val = get_idw(x,
                                  y,
                                  fin_nebrs_x_crds,
                                  fin_nebrs_y_crds,
                                  fin_nebrs_vals,
                                  idw_exp)
                interp_vals_arr[i, idw_idx] = idw_val

            if 'ok' in interp_types:
                _x = np.array([x])
                _y = np.array([y])

#                import IPython.core.debugger
#                dbg = IPython.core.debugger.Pdb()
#                dbg.set_trace()

                ord_krig = OrdinaryKriging(xi=fin_nebrs_x_crds,
                                           yi=fin_nebrs_y_crds,
                                           zi=fin_nebrs_vals,
                                           xk=_x,
                                           yk=_y,
                                           model=vg_model)

                ord_krig.krige()

                interp_vals_arr[i, ok_idx] = ord_krig.zk[0]

        else:
            print('No nebors!')

    return interp_vals_arr

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.ioff()

#    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
#    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = os.path.join(r'P:\Synchronize\Python3Codes',
                            r'krig_idw_nebs\test_ppt')

    in_data_file = r'infilled_var_df_infill_stns.csv'
    in_coords_file = r'infilled_var_df_infill_stns_coords.csv'

    in_vgs_file = (r'ppt_fitted_variograms__nk_1.000__evg_name_'
                   r'robust__ngp_5__h_typ_var.csv')

    sep = ';'
    time_fmt = '%Y-%m-%d'

    interp_date = '2014-06-05'

    interp_types = ['idw', 'ok']#, 'sk', 'edk']

    idw_exp = 3
    cell_size = 5000
    buff_cells = 2
#    n_nebs = 10
    n_quads = 20
    n_per_quad = 1
    min_dist_thresh = 100.0

    os.chdir(main_dir)

    in_data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
    in_coords_df = pd.read_csv(in_coords_file, sep=sep, index_col=0)
    in_vgs_df = pd.read_csv(in_vgs_file, sep=sep, index_col=0)

    in_data_df.index = pd.to_datetime(in_data_df.index, format=time_fmt)
    in_coords_df = in_coords_df[['X', 'Y']]
    in_vgs_df = in_vgs_df.iloc[:, 0]

    curr_vals_ser = in_data_df.loc[interp_date].dropna().iloc[:100].copy()
    curr_coords_df = in_coords_df.loc[curr_vals_ser.index].dropna().copy()
    vg_model = in_vgs_df.loc[interp_date]

    comm_stns = curr_coords_df.index
    comm_stns = comm_stns.intersection(curr_vals_ser.index)

    curr_vals_ser = curr_vals_ser.loc[comm_stns]
    curr_coords_df = curr_coords_df.loc[comm_stns]

    curr_x_vals = curr_coords_df['X'].values.astype(np.float64, order='C')
    curr_y_vals = curr_coords_df['Y'].values.astype(np.float64, order='C')
    curr_vals = curr_vals_ser.values.astype(np.float64, order='C')

    assert np.all(curr_x_vals)
    assert np.all(curr_y_vals)
    assert np.all(np.logical_not(np.isnan(curr_vals)))

    min_x = curr_x_vals.min()
    max_x = curr_x_vals.max()

    min_y = curr_y_vals.min()
    max_y = curr_y_vals.max()

    min_val = curr_vals.min()
    max_val = curr_vals.max()

    n_nebs = curr_vals.shape[0]

    x_interp_vec = np.arange(min_x - (buff_cells * cell_size),
                             max_x + (buff_cells * cell_size), cell_size)

    y_interp_vec = np.arange(min_y - (buff_cells * cell_size),
                             max_y + (buff_cells * cell_size), cell_size)

    interp_x_coords, interp_y_coords = np.meshgrid(x_interp_vec, y_interp_vec)

    orig_grid_shape = interp_x_coords.shape

    interp_x_coords_arr = interp_x_coords.ravel()
    interp_y_coords_arr = interp_y_coords.ravel()

    pcol_x_coords = np.linspace(x_interp_vec[0],
                                x_interp_vec[-1],
                                x_interp_vec.shape[0] + 1)

    pcol_y_coords = np.linspace(y_interp_vec[0],
                                y_interp_vec[-1],
                                y_interp_vec.shape[0] + 1)

    (pcol_x_mesh_coords,
     pcol_y_mesh_coords) = np.meshgrid(pcol_x_coords, pcol_y_coords)

    atleast_one = 0

    if 'idw' in interp_types:
        idw_idx = interp_types.index('idw')
        atleast_one += 1

    if 'ok' in interp_types:
        assert vg_model
        ok_idx = interp_types.index('ok')
        atleast_one += 1

    if 'sk' in interp_types:
        assert vg_model
        sk_idx = interp_types.index('sk')
        atleast_one += 1

    if 'edk' in interp_types:
        assert vg_model
        edk_idx = interp_types.index('edk')
        atleast_one += 1

    assert atleast_one

    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    with_neb_sel_interps = get_all_grid(interp_x_coords_arr,
                                        interp_y_coords_arr,
                                        curr_x_vals,
                                        curr_y_vals,
                                        curr_vals,
                                        idw_exp,
                                        n_quads,
                                        n_per_quad,
                                        min_dist_thresh,
                                        n_nebs,
                                        interp_types,
                                        vg_model,
                                        use_close_nebs=True)

    wo_neb_sel_interps = get_all_grid(interp_x_coords_arr,
                                      interp_y_coords_arr,
                                      curr_x_vals,
                                      curr_y_vals,
                                      curr_vals,
                                      idw_exp,
                                      n_quads,
                                      n_per_quad,
                                      min_dist_thresh,
                                      n_nebs,
                                      interp_types,
                                      vg_model,
                                      use_close_nebs=False)

    fig, axes = plt.subplots(atleast_one, atleast_one)

    if 'idw' in interp_types:
        idw_neb_sel_interp_grid = \
            with_neb_sel_interps[:, idw_idx].reshape(orig_grid_shape)
        idw_wo_neb_sel_interp_grid = \
            wo_neb_sel_interps[:, idw_idx].reshape(orig_grid_shape)

        axes[idw_idx, 0].pcolormesh(pcol_x_mesh_coords,
                                    pcol_y_mesh_coords,
                                    idw_neb_sel_interp_grid,
                                    vmin=min_val,
                                    vmax=max_val)
        axes[idw_idx, 0].scatter(curr_x_vals, curr_y_vals)

        axes[idw_idx, 1].pcolormesh(pcol_x_mesh_coords,
                                    pcol_y_mesh_coords,
                                    idw_wo_neb_sel_interp_grid,
                                    vmin=min_val,
                                    vmax=max_val)
        axes[idw_idx, 1].scatter(curr_x_vals, curr_y_vals)

    if 'ok' in interp_types:
        ok_neb_sel_interp_grid = \
            with_neb_sel_interps[:, ok_idx].reshape(orig_grid_shape)
        ok_wo_neb_sel_interp_grid = \
            wo_neb_sel_interps[:, ok_idx].reshape(orig_grid_shape)

        axes[ok_idx, 0].pcolormesh(pcol_x_mesh_coords,
                                   pcol_y_mesh_coords,
                                   ok_neb_sel_interp_grid,
                                   vmin=min_val,
                                   vmax=max_val)
        axes[ok_idx, 0].scatter(curr_x_vals, curr_y_vals)

        axes[ok_idx, 1].pcolormesh(pcol_x_mesh_coords,
                                   pcol_y_mesh_coords,
                                   ok_wo_neb_sel_interp_grid,
                                   vmin=min_val,
                                   vmax=max_val)
        axes[ok_idx, 1].scatter(curr_x_vals, curr_y_vals)

    plt.show()
    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s. Total run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START)))
