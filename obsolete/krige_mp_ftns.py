'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)

import numpy as np
import matplotlib.pyplot as plt
from descartes import PolygonPatch

import pyximport
pyximport.install()

from spinterps import get_idw_arr
from krigings import (OrdinaryKriging, SimpleKriging, ExternalDriftKriging_MD)

plt.ioff()


def mod_min_max(min_var_val, max_var_val, curr_data_vals, krige_fld):

    if min_var_val == 'min_in':
        min_in = curr_data_vals.min()
        krige_fld[krige_fld < min_in] = min_in

    elif min_var_val is None:
        pass

    elif (isinstance(min_var_val, float) or isinstance(min_var_val, int)):
        krige_fld[krige_fld < min_var_val] = min_var_val

    else:
        raise ValueError('Incorrect min_var_val specified!')

    if max_var_val == 'max_in':
        max_in = curr_data_vals.max()
        krige_fld[krige_fld > max_in] = max_in

    elif max_var_val is None:
        pass

    elif (isinstance(max_var_val, float) or isinstance(max_var_val, int)):
        krige_fld[krige_fld > max_var_val] = max_var_val

    else:
        raise ValueError('Incorrect max_var_val specified!')
    return


def plot_krige(krige_x_coords_plot_mesh,
               krige_y_coords_plot_mesh,
               krige_fld,
               var_name,
               var_units,
               curr_x_coords,
               curr_y_coords,
               polys_list,
               date,
               model,
               out_fig_path):

    fig, ax = plt.subplots()
    pclr = ax.pcolormesh(krige_x_coords_plot_mesh,
                         krige_y_coords_plot_mesh,
                         krige_fld,
                         vmin=np.nanmin(krige_fld),
                         vmax=np.nanmax(krige_fld))
    cb = fig.colorbar(pclr)
    cb.set_label(var_name + ' (' + var_units + ')')
    ax.scatter(curr_x_coords,
               curr_y_coords,
               label='obs. pts.',
               marker='+',
               c='r',
               alpha=0.7)
    ax.legend(framealpha=0.5)

    for poly in polys_list:
        ax.add_patch(PolygonPatch(poly,
                                  alpha=1,
                                  fc='None',
                                  ec='k'))

    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    title = 'Date: %s\n(vg: %s)\n' % (date.strftime('%Y-%m-%d'), model)
    ax.set_title(title)

    plt.setp(ax.get_xmajorticklabels(), rotation=70)
    ax.set_aspect('equal', 'datalim')
    plt.savefig(str(out_fig_path), bbox_inches='tight')
    plt.close()
    return


def ordinary_kriging(ok_vars):
    (in_data_df,
     in_stns_coords_df,
     in_vgs_df,
     min_ppt_thresh,
     var_name,
     krige_x_coords_mesh,
     krige_y_coords_mesh,
     krige_coords_orig_shape,
     min_var_val,
     max_var_val,
     (strt_index, end_index),
     plot_figs_flag,
     krige_x_coords_plot_mesh,
     krige_y_coords_plot_mesh,
     var_units,
     polys_list,
     out_figs_dir,
     fin_cntn_idxs) = ok_vars

    fin_date_range = in_data_df.index

    ord_krige_flds = np.full((fin_date_range.shape[0],
                              krige_coords_orig_shape[0],
                              krige_coords_orig_shape[1]),
                             np.nan,
                             dtype=np.float32)

    for i, date in enumerate(fin_date_range):
        curr_stns = in_data_df.loc[date, :].dropna().index

        assert curr_stns.shape == np.unique(curr_stns).shape

        curr_data_vals = in_data_df.loc[date, curr_stns].values
        curr_x_coords = in_stns_coords_df.loc[curr_stns, 'X'].values
        curr_y_coords = in_stns_coords_df.loc[curr_stns, 'Y'].values

        model = str(in_vgs_df.loc[date][0])

        if not (curr_stns.shape[0] and (model != 'nan')):
            print('No OK interpolation on %s for %s' % (str(date), var_name))
            continue

        if (np.all(curr_data_vals <= min_ppt_thresh) and
            (var_name == 'precipitation')):
            _ = np.full(krige_x_coords_mesh.shape[0], 0.0)

        else:

            try:
                ord_krig = OrdinaryKriging(xi=curr_x_coords,
                                           yi=curr_y_coords,
                                           zi=curr_data_vals,
                                           xk=krige_x_coords_mesh,
                                           yk=krige_y_coords_mesh,
                                           model=model)
                ord_krig.krige()
                _ = ord_krig.zk

            except Exception as msg:
                print('Error on %s:' % date.strftime('%Y-%m-%d'), msg)
                _ = np.full(krige_x_coords_mesh.shape[0], np.nan)

        nan_fld = np.full(fin_cntn_idxs.shape[0], np.nan, dtype=np.float32)
        nan_fld[fin_cntn_idxs] = _
        ord_krige_flds[i] = (
            nan_fld.reshape(krige_coords_orig_shape))

        mod_min_max(min_var_val,
                    max_var_val,
                    curr_data_vals,
                    ord_krige_flds[i])

        if plot_figs_flag:
            out_fig_path = (
                os.path.join(out_figs_dir,
                             'ok_%s.png' % date.strftime('%Y-%m-%d')))

            plot_krige(krige_x_coords_plot_mesh,
                       krige_y_coords_plot_mesh,
                       ord_krige_flds[i],
                       var_name,
                       var_units,
                       curr_x_coords,
                       curr_y_coords,
                       polys_list,
                       date,
                       model,
                       out_fig_path)

    return [strt_index, end_index, ord_krige_flds]


def simple_kriging(sk_vars):
    (in_data_df,
     in_stns_coords_df,
     in_vgs_df,
     min_ppt_thresh,
     var_name,
     krige_x_coords_mesh,
     krige_y_coords_mesh,
     krige_coords_orig_shape,
     min_var_val,
     max_var_val,
     (strt_index, end_index),
     plot_figs_flag,
     krige_x_coords_plot_mesh,
     krige_y_coords_plot_mesh,
     var_units,
     polys_list,
     out_figs_dir,
     fin_cntn_idxs) = sk_vars

    fin_date_range = in_data_df.index
    sim_krige_flds = np.full((fin_date_range.shape[0],
                              krige_coords_orig_shape[0],
                              krige_coords_orig_shape[1]),
                             np.nan,
                             dtype=np.float32)

    for i, date in enumerate(fin_date_range):
        curr_stns = in_data_df.loc[date, :].dropna().index

        curr_data_vals = in_data_df.loc[date, curr_stns].values
        curr_x_coords = in_stns_coords_df.loc[curr_stns, 'X'].values
        curr_y_coords = in_stns_coords_df.loc[curr_stns, 'Y'].values

        model = str(in_vgs_df.loc[date][0])

        if not (curr_stns.shape[0] and (model != 'nan')):
            print('No SK interpolation on %s for %s' % (str(date), var_name))
            continue

        if (np.all(curr_data_vals <= min_ppt_thresh) and
            (var_name == 'precipitation')):

            _ = np.full(krige_x_coords_mesh.shape[0], 0.0)

        else:
            try:
                sim_krig = SimpleKriging(xi=curr_x_coords,
                                         yi=curr_y_coords,
                                         zi=curr_data_vals,
                                         xk=krige_x_coords_mesh,
                                         yk=krige_y_coords_mesh,
                                         model=model)
                sim_krig.krige()
                _ = sim_krig.zk

            except Exception as msg:
                print('Error on %s:' % date.strftime('%Y-%m-%d'), msg)
                _ = np.full(krige_x_coords_mesh.shape[0], np.nan)

        nan_fld = np.full(fin_cntn_idxs.shape[0], np.nan, dtype=np.float32)
        nan_fld[fin_cntn_idxs] = _
        sim_krige_flds[i] = (
            nan_fld.reshape(krige_coords_orig_shape))

        mod_min_max(min_var_val,
                    max_var_val,
                    curr_data_vals,
                    sim_krige_flds[i])

        if plot_figs_flag:
            out_fig_path = (
                os.path.join(out_figs_dir,
                             'sk_%s.png' % date.strftime('%Y-%m-%d')))

            plot_krige(krige_x_coords_plot_mesh,
                       krige_y_coords_plot_mesh,
                       sim_krige_flds[i],
                       var_name,
                       var_units,
                       curr_x_coords,
                       curr_y_coords,
                       polys_list,
                       date,
                       model,
                       out_fig_path)

    return [strt_index, end_index, sim_krige_flds]


def external_drift_kriging(edk_vars):
    (in_data_df,
     in_stns_drift_df,
     in_stns_coords_df,
     in_vgs_df,
     min_ppt_thresh,
     var_name,
     krige_x_coords_mesh,
     krige_y_coords_mesh,
     drift_vals_arr,
     krige_coords_orig_shape,
     drift_ndv,
     min_var_val,
     max_var_val,
     (strt_index, end_index),
     plot_figs_flag,
     krige_x_coords_plot_mesh,
     krige_y_coords_plot_mesh,
     var_units,
     polys_list,
     out_figs_dir,
     fin_cntn_idxs) = edk_vars

    fin_date_range = in_data_df.index
    edk_krige_flds = np.full((fin_date_range.shape[0],
                              krige_coords_orig_shape[0],
                              krige_coords_orig_shape[1]),
                             np.nan,
                             dtype=np.float32)

    for i, date in enumerate(fin_date_range):
        _ = in_data_df.loc[date, :].dropna().index
        curr_stns = _.intersection(in_stns_drift_df.index)

        curr_data_vals = in_data_df.loc[date, curr_stns].values
        curr_x_coords = in_stns_coords_df.loc[curr_stns, 'X'].values
        curr_y_coords = in_stns_coords_df.loc[curr_stns, 'Y'].values
        curr_drift_vals = (
            np.atleast_2d(in_stns_drift_df.loc[curr_stns].values.T))

        model = str(in_vgs_df.loc[date][0])

        if not (curr_stns.shape[0] and (model != 'nan')):
            print('No EDK interpolation on %s for %s' % (str(date), var_name))
            continue

        if (np.all(curr_data_vals <= min_ppt_thresh) and
            (var_name == 'precipitation')):
            _ = np.full(krige_x_coords_mesh.shape[0], 0.0)

        else:
            try:
                edk_krig = ExternalDriftKriging_MD(xi=curr_x_coords,
                                                   yi=curr_y_coords,
                                                   zi=curr_data_vals,
                                                   si=curr_drift_vals,
                                                   xk=krige_x_coords_mesh,
                                                   yk=krige_y_coords_mesh,
                                                   sk=drift_vals_arr,
                                                   model=model)
                edk_krig.krige()

                _ = edk_krig.zk.copy()

            except Exception as msg:
                print('Error on %s:' % date.strftime('%Y-%m-%d'), msg)
                _ = np.full(krige_x_coords_mesh.shape[0], np.nan)

        _[np.isclose(drift_ndv, drift_vals_arr[0])] = np.nan

        nan_fld = np.full(fin_cntn_idxs.shape[0], np.nan, dtype=np.float32)
        nan_fld[fin_cntn_idxs] = _
        edk_krige_flds[i] = nan_fld.reshape(krige_coords_orig_shape)

        mod_min_max(min_var_val,
                    max_var_val,
                    curr_data_vals,
                    edk_krige_flds[i])

        if plot_figs_flag:
            out_fig_path = (
                os.path.join(out_figs_dir,
                             'edk_%s.png' % date.strftime('%Y-%m-%d')))

            plot_krige(krige_x_coords_plot_mesh,
                       krige_y_coords_plot_mesh,
                       edk_krige_flds[i],
                       var_name,
                       var_units,
                       curr_x_coords,
                       curr_y_coords,
                       polys_list,
                       date,
                       model,
                       out_fig_path)

    return [strt_index, end_index, edk_krige_flds]


def inverse_distance_wtng(idw_vars):
    (in_data_df,
     in_stns_coords_df,
     min_ppt_thresh,
     idw_exp,
     var_name,
     krige_x_coords_mesh,
     krige_y_coords_mesh,
     krige_coords_orig_shape,
     min_var_val,
     max_var_val,
     (strt_index, end_index),
     plot_figs_flag,
     krige_x_coords_plot_mesh,
     krige_y_coords_plot_mesh,
     var_units,
     polys_list,
     out_figs_dir,
     fin_cntn_idxs) = idw_vars

    fin_date_range = in_data_df.index
    idw_flds = np.full((fin_date_range.shape[0],
                        krige_coords_orig_shape[0],
                        krige_coords_orig_shape[1]),
                       np.nan,
                       dtype=np.float32)

    for i, date in enumerate(fin_date_range):
        curr_stns = in_data_df.loc[date, :].dropna().index

        curr_data_vals = in_data_df.loc[date, curr_stns].values
        curr_x_coords = in_stns_coords_df.loc[curr_stns, 'X'].values
        curr_y_coords = in_stns_coords_df.loc[curr_stns, 'Y'].values

        if not curr_stns.shape[0]:
            print('No IDW interpolation on %s for %s' % (str(date), var_name))
            continue

        if (np.all(curr_data_vals <= min_ppt_thresh) and
            (var_name == 'precipitation')):
            curr_idw_arr = np.full(krige_x_coords_mesh.shape[0], 0.0)

        else:
            curr_idw_arr = get_idw_arr(krige_x_coords_mesh,
                                       krige_y_coords_mesh,
                                       curr_x_coords,
                                       curr_y_coords,
                                       curr_data_vals,
                                       idw_exp)

        nan_fld = np.full(fin_cntn_idxs.shape[0], np.nan, dtype=np.float32)
        nan_fld[fin_cntn_idxs] = curr_idw_arr

        idw_flds[i] = nan_fld.reshape(krige_coords_orig_shape)

        mod_min_max(min_var_val,
                    max_var_val,
                    curr_data_vals,
                    idw_flds[i])

        if plot_figs_flag:
            out_fig_path = (
                os.path.join(out_figs_dir,
                             'idw_%s.png' % date.strftime('%Y-%m-%d')))
            plot_krige(krige_x_coords_plot_mesh,
                       krige_y_coords_plot_mesh,
                       idw_flds[i],
                       var_name,
                       var_units,
                       curr_x_coords,
                       curr_y_coords,
                       polys_list,
                       date,
                       'IDW',
                       out_fig_path)

    return [strt_index, end_index, idw_flds]


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    os.chdir(main_dir)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
