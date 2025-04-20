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

    main_dir = Path(r'P:\DEBY_ISAR\spinterps\tg_1D_100m_20250418')
    # main_dir = Path(r'P:\dwd_meteo\gridded\extract_hyras\d_lmtd_and_infld_tst')
    # main_dir = Path(r'P:\DEBY\spinterps\ppt_1D_1km_20240906')
    os.chdir(main_dir)

    # in_nc_path = Path(r'EMO-1arcmin-ws_1990_2022.nc')
    # in_nc_path = Path(r'tas_1hr_HOSTRADA-v1-0_BE_gn_2001030100-2001033123.nc')
    # in_nc_path = Path(r'RW_2007_2023.nc')
    in_nc_path = Path(r'kriging.nc')

    nme_sfx = ''

    var_label = 'EDK'  # 'NNB'  # 'pr'  # 'pptn'  # 'tas'  # 'RR'  # 'petn'  # 'pr'  # 'tasmin'  #
    # var_label = 'ws'  # 'RW'  # 'OK__CTD'  # 'OK__DIF'  # 'OK'  # 'EST_VARS_OK'  # 'IDW_000'

    x_label = 'X'  # 'lon'  # 'x'  # 'X1D'  #  'longitude'  #
    y_label = 'Y'  # 'lat'  # 'y'  # 'Y1D'  # 'latitude'  #
    # x_label = 'x_utm32n'  # 'X2D'  #
    # y_label = 'y_utm32n'  # 'Y2D'  #

    time_label = 'time'

    # cbar_label = 'Precipitation [mm]'
    # cbar_label = 'Snow depth [mm]'
    # cbar_label = 'Snow melt [mm]'
    # cbar_label = 'PET [mm]'
    cbar_label = 'Temperature [°C]'
    # cbar_label = 'Vapor pressure [hPa]'
    # cbar_label = 'Radiation [j/m$^2$]'
    # cbar_label = 'Wind speed [m/s]'

    # cbar_label = 'Estimation variance'

    cmap = 'viridis'  # 'Blues'  #

    dpi = 150

    var_min_val = None
    var_max_val = None

    # var_min_val = 0
    # var_max_val = 10

    var_llm = None
    var_ulm = None

    # var_llm = 0.0
    # var_ulm = 100

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

    # plot_sum_flag = True
    plot_sum_flag = False

    # nan_val = 9999
    nan_val = None

    beg_time, end_time = pd.to_datetime(
        ['2022-01-02', '2022-01-30'],
        format='%Y-%m-%d')

    # beg_time, end_time = pd.to_datetime(
    #     ['1990-01-02 06:00:00', '1990-01-30 06:00:00'],
    #     format='%Y-%m-%d %H:%M:%S')

    # beg_time, end_time = pd.to_datetime(
    #     ['20190520T070000', '20190521T070000'],
    #     format='%Y%m%dT%H%M%S')

    # in_cat_file = Path(r'P:\DEBY\dem_ansys_1km\watersheds.shp')
    # in_cat_file = Path(r'P:\DEBY\bayern_epsg32632.shp')
    # in_cat_file = Path(r'P:\TUM\projects\altoetting\vector\bayern_epsg32632.shp')
    # in_cat_file = Path(r'P:\DEBY_ISAR\dem\v2_500m\dem_ansys__daily\watersheds.shp')
    in_cat_file = Path(r'P:\DEBY_ISAR\inputs\catchment_epsg_3034.shp')

    # in_cat_file = None

    in_crds_file = None
    # in_crds_file = Path(r'../ppt_1D_gkd_dwd_crds.csv')

    out_figs_dir = Path(r'vbe_gds__plots')
    #==========================================================================

    if in_crds_file is not None:
        crds_df = pd.read_csv(in_crds_file, sep=';', index_col=0)[['X', 'Y']]
        crds_df.index = crds_df.index.astype(str)

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
        take_idxs_end = times.get_loc(end_time) + 1

        times = times[take_idxs_beg:take_idxs_end]
        data = nc_hdl[var_label][take_idxs_beg:take_idxs_end].data

        if nan_val is not None:
            data[data == nan_val] = np.nan

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

            if var_llm is not None:
                interp_fld[interp_fld < var_llm] = np.nan

            if var_ulm is not None:
                interp_fld[interp_fld > var_ulm] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            grd_min = np.nanmin(interp_fld)
            grd_max = np.nanmax(interp_fld)
            grd_men = np.nanmean(interp_fld)
            grd_std = np.nanstd(interp_fld)
        #======================================================================

        pclr = ax.pcolormesh(
            x_crds_plt_msh,
            y_crds_plt_msh,
            interp_fld,
            vmin=var_min_val,
            vmax=var_max_val,
            shading='auto',  # 1D CRDS.
            # shading='flat',  # 2D CRDS.
            # shading='nearest',  # 1D CRDS.
            cmap=cmap)

        cb = fig.colorbar(pclr)

        cb.set_label(cbar_label)

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

        if show_title_flag:
            title = (
                f'Time: {time_str}\n'
                f'Mean: {grd_men:0.4f}, Std.: {grd_std:0.4f}\n'
                f'Min.: {grd_min:0.4f}, Max.: {grd_max:0.4f}')

            ax.set_title(title)

        plt.setp(ax.get_xmajorticklabels(), rotation=90)
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

                    ax.plot(pts[geom_i][:, 0], pts[geom_i][:, 1], c='k', lw=1)
        #======================================================================

        if in_crds_file is not None:
            plt.scatter(
                crds_df['X'].values,
                crds_df['Y'].values,
                c='k',
                s=0.2,
                alpha=0.75,
                edgecolors='none')
        #======================================================================

        ax.set_xlim(x_llim, x_ulim)
        ax.set_ylim(y_ulim, y_llim)

        # plt.show()
        # break

        out_fig_name = f'{var_label.lower()}{nme_sfx}_{time_str}.png'

        out_fig_name = out_fig_name.replace(':', '_')

        plt.savefig(
            out_var_figs_dir / out_fig_name, bbox_inches='tight', dpi=dpi)

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
