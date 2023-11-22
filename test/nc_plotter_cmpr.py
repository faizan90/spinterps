'''
@author: Faizan-Uni-Stuttgart

Jul 10, 2021

10:45:32 AM

'''
import os
import sys
import time
import timeit
import warnings
import traceback as tb
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib as mpl
import matplotlib.pyplot as plt

DEBUG_FLAG = False


def main():

    main_dir = Path(r'U:\TUM\projects\altoetting\spinterps\ppt_1H_gkd_dwd_1km_G3_edk')

    os.chdir(main_dir)

    in_nc_path_1 = Path(r'U:\TUM\projects\altoetting\spinterps\ppt_1H_gkd_dwd_1km_G3_edk\kriging.nc')
    var_label_1 = 'EDK'
    x_label_1 = 'X'
    y_label_1 = 'Y'
    time_label_1 = 'time'
    sclr_1 = None

    in_nc_path_2 = Path(r'U:\TUM\projects\altoetting\spinterps\ppt_1H_gkd_dwd_1km_G2\kriging.nc')
    var_label_2 = 'IDW_000'
    x_label_2 = 'X'
    y_label_2 = 'Y'
    time_label_2 = 'time'
    sclr_2 = None

    cbar_label = 'Precipitation (mm)'
    # cbar_label = 'PET (mm)'
    # cbar_label = 'Temperature (C)'

    cmap = 'viridis'  # 'Blues'  #

    # var_min_val = 0
    # var_max_val = 40

    # var_min_val = 0
    # var_max_val = 2

    show_title_flag = True
    # show_title_flag = False

    # beg_time, end_time = pd.to_datetime(
    #     ['2019-06-10 12:00:00', '2019-06-11 02:00:00'],
    #     format='%Y-%m-%d %H:%M:%S')

    # beg_time, end_time = pd.to_datetime(
    #     ['2011-01-13 01:00:00', '2011-01-13 20:00:00'],
    #     format='%Y-%m-%d %H:%M:%S')

    # beg_time, end_time = pd.to_datetime(
    #     ['2021-01-28 14:00:00', '2021-01-29 05:00:00'],
    #     format='%Y-%m-%d %H:%M:%S')

    # beg_time, end_time = pd.to_datetime(
    #     ['2009-05-26 14:00:00', '2009-05-27 04:00:00'],
    #     format='%Y-%m-%d %H:%M:%S')

    beg_time, end_time = pd.to_datetime(
        ['2010-05-18 00', '2010-06-07 23'],
        format='%Y-%m-%d %H')

    # Catchments shapefile.
    in_cat_file = Path(r'U:\TUM\projects\altoetting\hydmod\dem_ansys_1km_G3\watersheds.shp')

    cat_col = 'DN'

    drop_stns = []  # [420, 3421, 427, 3465, 3470]

    out_figs_dir = Path(r'cmpr_grids')
    #==========================================================================

    with nc.Dataset(in_nc_path_1, 'r') as nc_hdl:
        x_crds_1 = nc_hdl[x_label_1][...].data
        y_crds_1 = nc_hdl[y_label_1][...].data

        times_1 = nc.num2date(
            nc_hdl[time_label_1][:].data,
            units=nc_hdl[time_label_1].units,
            calendar=nc_hdl[time_label_1].calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True)

        times_1 = pd.DatetimeIndex(times_1)

        take_idxs_beg_1 = times_1.get_loc(beg_time)
        take_idxs_end_1 = times_1.get_loc(end_time)

        times_1 = times_1[take_idxs_beg_1:take_idxs_end_1]
        data_1 = nc_hdl[var_label_1][take_idxs_beg_1:take_idxs_end_1].data

    if x_crds_1.ndim == 1:
        x_crds_plt_msh_1, y_crds_plt_msh_1 = np.meshgrid(x_crds_1, y_crds_1)

    else:
        x_crds_plt_msh_1, y_crds_plt_msh_1 = (x_crds_1, y_crds_1)
    #==========================================================================

    with nc.Dataset(in_nc_path_2, 'r') as nc_hdl:
        x_crds_2 = nc_hdl[x_label_2][...].data
        y_crds_2 = nc_hdl[y_label_2][...].data

        times_2 = nc.num2date(
            nc_hdl[time_label_2][:].data,
            units=nc_hdl[time_label_2].units,
            calendar=nc_hdl[time_label_2].calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True)

        times_2 = pd.DatetimeIndex(times_2)

        take_idxs_beg_2 = times_2.get_loc(beg_time)
        take_idxs_end_2 = times_2.get_loc(end_time)

        times_2 = times_2[take_idxs_beg_2:take_idxs_end_2]
        data_2 = nc_hdl[var_label_2][take_idxs_beg_2:take_idxs_end_2].data

    if x_crds_2.ndim == 1:
        x_crds_plt_msh_2, y_crds_plt_msh_2 = np.meshgrid(x_crds_2, y_crds_2)

    else:
        x_crds_plt_msh_2, y_crds_plt_msh_2 = (x_crds_2, y_crds_2)
    #==========================================================================

    cats_hdl = fiona.open(str(in_cat_file))
    #==========================================================================

    out_var_figs_dir = out_figs_dir / f'{var_label_1}__{2}'

    out_figs_dir.mkdir(exist_ok=True)
    out_var_figs_dir.mkdir(exist_ok=True)
    #==========================================================================

    for i in range(data_1.shape[0]):

        fig, axs = plt.subplots(
            1, 3, width_ratios=(1, 1, 0.1), figsize=(10, 5))

        time_str_1 = f'{times_1[i].strftime("%Y%m%dT%H%M")}'

        time_str_2 = f'{times_2[i].strftime("%Y%m%dT%H%M")}'

        print(f'Plotting {time_str_1} vs. {time_str_2}')

        interp_fld_1 = data_1[i]

        if sclr_1 is not None:
            interp_fld_1 *= sclr_1

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            grd_min_1 = np.nanmin(interp_fld_1)
            grd_max_1 = np.nanmax(interp_fld_1)
        #======================================================================

        interp_fld_2 = data_2[i]

        if sclr_2 is not None:
            interp_fld_2 *= sclr_2

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            grd_min_2 = np.nanmin(interp_fld_2)
            grd_max_2 = np.nanmax(interp_fld_2)
        #======================================================================

        var_min_val = np.nanmin([grd_min_1, grd_min_2])
        var_max_val = np.nanmax([grd_max_1, grd_max_2])
        #======================================================================

        pclr = axs[0].pcolormesh(
            x_crds_plt_msh_1,
            y_crds_plt_msh_1,
            interp_fld_1,
            vmin=var_min_val,
            vmax=var_max_val,
            shading='auto',
            cmap=cmap)

        axs[0].set_xlabel('Easting')
        axs[0].set_ylabel('Northing')

        if show_title_flag:
            title = (
                f'Time: {time_str_1}\n'
                f'Min.: {grd_min_1:0.4f}, Max.: {grd_max_1:0.4f}')

            axs[0].set_title(title)
        #======================================================================

        plt.setp(axs[0].get_xmajorticklabels(), rotation=70)
        axs[0].set_aspect('equal', 'datalim')
        #======================================================================

        axs[1].pcolormesh(
            x_crds_plt_msh_2,
            y_crds_plt_msh_2,
            interp_fld_2,
            vmin=var_min_val,
            vmax=var_max_val,
            shading='auto',
            cmap=cmap)

        for geom in cats_hdl:

            cat_id = geom['properties'][cat_col]

            if cat_id in drop_stns:
                continue

            geom_type = geom['geometry']['type']
            if geom_type == 'Polygon':
                pts = [np.array(geom['geometry']['coordinates'][0]), ]

            elif geom_type == 'MultiPolygon':
                pts = [
                    np.array(sub_geom[0])
                    for sub_geom in geom['geometry']['coordinates']]

            else:
                raise ValueError(f'Unknown geometry type: {geom_type}!')

            for geom_i in range(len(pts)):
                axs[0].plot(pts[geom_i][:, 0], pts[geom_i][:, 1], c='white')
                axs[1].plot(pts[geom_i][:, 0], pts[geom_i][:, 1], c='white')
        #======================================================================

        # cb = fig.colorbar(pclr)
        #
        # cb.set_label(cbar_label)

        axs[1].set_xlabel('Easting')
        # axs[1].set_ylabel('Northing')
        # axs[1].yaxis.set_tick_params(labelright=True, labelleft=False)

        if show_title_flag:
            title = (
                f'Time: {time_str_2}\n'
                f'Min.: {grd_min_2:0.4f}, Max.: {grd_max_2:0.4f}')

            axs[1].set_title(title)

        plt.setp(axs[1].get_xmajorticklabels(), rotation=70)
        axs[1].set_aspect('equal', 'datalim')
        #======================================================================

        # axs[-1].set_axis_off()

        cb_norm = mpl.colors.Normalize(
            vmin=var_min_val, vmax=var_max_val)

        mpl.colorbar.ColorbarBase(
            axs[-1],
            cmap=cmap,
            norm=cb_norm,
            label=cbar_label,
            orientation='vertical')
        #======================================================================

        out_fig_name = f'{var_label_1.lower()}_{time_str_1}.png'

        plt.savefig(str(out_var_figs_dir / out_fig_name), bbox_inches='tight')

        # cb.remove()

        for ax in axs:
            ax.cla()
        # plt.cla()
        # break

        # fig.clear()

        plt.close()
    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
