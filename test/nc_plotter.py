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

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\hydmod_de\hourly\pet_hourly_2008_2020_interp_5km')

    os.chdir(main_dir)

    in_nc_path = Path(r'kriging_5km.nc')

    var_label = 'PET'
    x_label = 'X'
    y_label = 'Y'
    time_label = 'time'

    # cbar_label = 'Precipitation (mm)'
    cbar_label = 'PET (mm)'

    cmap = 'viridis'

    var_min_val = None
    var_max_val = None

    # var_min_val = 0
    # var_max_val = 2

    show_title_flag = True
    # show_title_flag = False

    take_idxs_beg = 500
    take_idxs_end = 600

    out_figs_dir = Path(r'interp_plots')
    #==========================================================================

    with nc.Dataset(in_nc_path, 'r') as nc_hdl:
        x_crds = nc_hdl[x_label][...].data
        y_crds = nc_hdl[y_label][...].data

        data = nc_hdl[var_label][take_idxs_beg:take_idxs_end].data
        times = nc_hdl[time_label][take_idxs_beg:take_idxs_end].data

    x_crds_plt_msh, y_crds_plt_msh = np.meshgrid(x_crds, y_crds)
    #==========================================================================

    out_var_figs_dir = out_figs_dir / var_label

    out_figs_dir.mkdir(exist_ok=True)
    out_var_figs_dir.mkdir(exist_ok=True)
    #==========================================================================

    fig, ax = plt.subplots()

    for i in range(data.shape[0]):
        time_str = f'{times[i]}'

        print(f'Plotting {time_str}')

        interp_fld = data[i]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            grd_min = np.nanmin(interp_fld)
            grd_max = np.nanmax(interp_fld)
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
                f'Min.: {grd_min:0.4f}, Max.: {grd_max:0.4f}')

            ax.set_title(title)

        plt.setp(ax.get_xmajorticklabels(), rotation=70)
        ax.set_aspect('equal', 'datalim')

        out_fig_name = f'{var_label.lower()}_{time_str}.png'

        plt.savefig(str(out_var_figs_dir / out_fig_name), bbox_inches='tight')

        cb.remove()

        plt.cla()
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
