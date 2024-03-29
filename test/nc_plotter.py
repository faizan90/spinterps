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
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'U:\hydmod\neckar\daily_1961_2020_spinterp_2km_ppt')

    os.chdir(main_dir)

    in_nc_path = Path(r'kriging.nc')

    var_label = 'EDK'  # 'IDW_000'
    x_label = 'X'
    y_label = 'Y'
    time_label = 'time'

    cbar_label = 'Precipitation (mm)'
    # cbar_label = 'PET (mm)'
    # cbar_label = 'Temperature (C)'

    # cbar_label = 'Estimation variance'

    cmap = 'Blues'  # 'viridis'

    var_min_val = None
    var_max_val = None

    # var_min_val = 0
    # var_max_val = 80

    x_llim = None
    x_ulim = None

    y_llim = None
    y_ulim = None

    # x_llim = 480000
    # x_ulim = 870000

    # y_llim = 5.61e6
    # y_ulim = 5.23e6

    show_title_flag = True
    # show_title_flag = False

    plot_sum_flag = True
    plot_sum_flag = False

    beg_time, end_time = pd.to_datetime(
        ['1970-01-20 00:00:00', '1970-02-10 00:00:00'],
        format='%Y-%m-%d %H:%M:%S')

    # beg_time, end_time = pd.to_datetime(
    #     ['20190520T070000', '20190521T070000'],
    #     format='%Y%m%dT%H%M%S')

    in_cat_file = Path(r'P:\Synchronize\IWS\QGIS_Neckar\vector\427_2km.shp')
    # in_cat_file = None

    out_figs_dir = Path(r'interp_plots')
    #==========================================================================

    with nc.Dataset(in_nc_path, 'r') as nc_hdl:
        x_crds = nc_hdl[x_label][...].data
        y_crds = nc_hdl[y_label][...].data

        times = nc.num2date(
            nc_hdl[time_label][:].data,
            units=nc_hdl[time_label].units,
            calendar=nc_hdl[time_label].calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True)

        times = pd.DatetimeIndex(times)

        take_idxs_beg = times.get_loc(beg_time)
        take_idxs_end = times.get_loc(end_time)

        times = times[take_idxs_beg:take_idxs_end]
        data = nc_hdl[var_label][take_idxs_beg:take_idxs_end].data

    if x_crds.ndim == 1:
        x_crds_plt_msh, y_crds_plt_msh = np.meshgrid(x_crds, y_crds)

    else:
        x_crds_plt_msh, y_crds_plt_msh = (x_crds, y_crds)
    #==========================================================================

    out_var_figs_dir = out_figs_dir / var_label

    out_figs_dir.mkdir(exist_ok=True)
    out_var_figs_dir.mkdir(exist_ok=True)

    if in_cat_file is not None:
        cats_hdl = fiona.open(str(in_cat_file))
    #==========================================================================

    fig, ax = plt.subplots()

    for i in range(data.shape[0]):
        time_str = f'{times[i]}'

        print(f'Plotting {time_str}')

        if plot_sum_flag:
            interp_fld = data.sum(axis=0)

        else:
            interp_fld = data[i]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            grd_min = np.nanmin(interp_fld)
            grd_max = np.nanmax(interp_fld)
            grd_men = np.nanmean(interp_fld)
        #======================================================================

        pclr = ax.pcolormesh(
            x_crds_plt_msh,
            y_crds_plt_msh,
            interp_fld,
            vmin=var_min_val,
            vmax=var_max_val,
            shading='auto',
            cmap=cmap)

        cb = fig.colorbar(pclr)

        cb.set_label(cbar_label)

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

        if show_title_flag:
            title = (
                f'Time: {time_str}\n'
                f'Min.: {grd_min:0.4f}, Mean: {grd_men:0.4f}, '
                f'Max.: {grd_max:0.4f}')

            ax.set_title(title)

        plt.setp(ax.get_xmajorticklabels(), rotation=70)
        ax.set_aspect('equal', 'datalim')
        #======================================================================

        if in_cat_file is not None:

            for geom in cats_hdl:

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
                    # poly = plt.Polygon(
                    #     pts[geom_i],
                    #     closed=False,
                    #     color='',
                    #     alpha=0.95,
                    #     ec=None)
                    #
                    # ax.add_patch(poly)

                    ax.plot(pts[geom_i][:, 0], pts[geom_i][:, 1], c='k')
        #======================================================================

        ax.set_xlim(x_llim, x_ulim)
        ax.set_ylim(y_ulim, y_llim)

        out_fig_name = f'{var_label.lower()}_{time_str}.png'

        out_fig_name = out_fig_name.replace(':', '_')

        plt.savefig(str(out_var_figs_dir / out_fig_name), bbox_inches='tight')

        cb.remove()

        plt.cla()

        if plot_sum_flag:
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
