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
import netCDF4 as nc
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(r'U:\cmip6\ec-earth3-cc\bw_historical')

    os.chdir(main_dir)

    in_nc_path_1 = Path(r'D:\hydmod\neckar\daily_1961_2020_spinterp_2km_ppt\kriging.nc')
    var_label_1 = 'EDK'
    x_label_1 = 'X'
    y_label_1 = 'Y'
    time_label_1 = 'time'
    sclr_1 = None

    in_nc_path_2 = Path(r'pr_day_EC-Earth3-CC_historical_r1i1p1f1_gr_19500101-20141231_v20210113_bw.nc')
    var_label_2 = 'pr'
    x_label_2 = 'x_utm32n'
    y_label_2 = 'y_utm32n'
    time_label_2 = 'time'
    sclr_2 = 24 * 3600.0

    cbar_label = 'Precipitation (mm)'
    # cbar_label = 'PET (mm)'
    # cbar_label = 'Temperature (C)'

    cmap = 'viridis'

    var_min_val = 0
    var_max_val = 25

    # var_min_val = 0
    # var_max_val = 2

    show_title_flag = True
    # show_title_flag = False

    take_idxs_beg_1 = 15834
    take_idxs_end_1 = 15835

    take_idxs_beg_2 = 15834 + 4018
    take_idxs_end_2 = 15835 + 4018

    # Catchments shapefile.
    in_cat_file = Path(r'D:\Synchronize\IWS\QGIS_Neckar\raster\taudem_out_dem_2km\watersheds_cumm.shp')

    cat_col = 'DN'

    drop_stns = [420, 3421, 427, 3465, 3470]

    out_figs_dir = Path(r'cmpr_grids')
    #==========================================================================

    with nc.Dataset(in_nc_path_1, 'r') as nc_hdl:
        x_crds_1 = nc_hdl[x_label_1][...].data
        y_crds_1 = nc_hdl[y_label_1][...].data

        data_1 = nc_hdl[var_label_1][take_idxs_beg_1:take_idxs_end_1].data
        times_1 = nc_hdl[time_label_1][take_idxs_beg_1:take_idxs_end_1].data

        time_strs_1 = nc.num2date(
            times_1, nc_hdl[time_label_1].units, nc_hdl[time_label_1].calendar)

    if x_crds_1.ndim == 1:
        x_crds_plt_msh_1, y_crds_plt_msh_1 = np.meshgrid(x_crds_1, y_crds_1)

    else:
        x_crds_plt_msh_1, y_crds_plt_msh_1 = (x_crds_1, y_crds_1)
    #==========================================================================

    with nc.Dataset(in_nc_path_2, 'r') as nc_hdl:
        x_crds_2 = nc_hdl[x_label_2][...].data
        y_crds_2 = nc_hdl[y_label_2][...].data

        data_2 = nc_hdl[var_label_2][take_idxs_beg_2:take_idxs_end_2].data
        times_2 = nc_hdl[time_label_2][take_idxs_beg_2:take_idxs_end_2].data

        time_strs_2 = nc.num2date(
            times_2, nc_hdl[time_label_2].units, nc_hdl[time_label_2].calendar)

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

    fig, axs = plt.subplots(1, 3, width_ratios=(1, 1, 0.1), figsize=(10, 5))

    for i in range(data_1.shape[0]):
        time_str_1 = f'{time_strs_1[i].strftime("%Y%m%dT%H%M")}'

        time_str_2 = f'{time_strs_2[i].strftime("%Y%m%dT%H%M")}'

        print(f'Plotting {time_str_1} vs. {time_str_2}')

        interp_fld_1 = data_1[i]

        if sclr_1 is not None:
            interp_fld_1 *= sclr_1

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            grd_min_1 = np.nanmin(interp_fld_1)
            grd_max_1 = np.nanmax(interp_fld_1)
        #======================================================================

        pclr = axs[0].pcolormesh(
            x_crds_plt_msh_1,
            y_crds_plt_msh_1,
            interp_fld_1,
            vmin=var_min_val,
            vmax=var_max_val,
            shading='auto',
            cmap=cmap)

        # cb = plt.colorbar(pclr)
        #
        # cb.set_label(cbar_label)

        axs[-1].set_axis_off()

        plt.colorbar(
            pclr,
            ax=axs[-1],
            label=cbar_label,
            orientation='vertical',
            fraction=0.8)

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

        interp_fld_2 = data_2[i]

        if sclr_2 is not None:
            interp_fld_2 *= sclr_2

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            grd_min_2 = np.nanmin(interp_fld_2)
            grd_max_2 = np.nanmax(interp_fld_2)
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

        out_fig_name = f'{var_label_1.lower()}_{time_str_1}.png'

        plt.savefig(str(out_var_figs_dir / out_fig_name), bbox_inches='tight')

        # cb.remove()

        plt.cla()
        break
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
