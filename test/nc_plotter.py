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
        r'P:\Synchronize\IWS\Colleagues_Students\Bianca\santa_ppt_interps\tp_ctvg\TM_00')

    os.chdir(main_dir)

    in_nc_path = Path(r'TM_00_itfm.nc')

    var_label = 'OK'
    x_label = 'X'
    y_label = 'Y'

    var_type = 'Probability (-)'

    cmap = 'jet'

    out_figs_dir = Path(r'interp_plots_itfm')
    #==========================================================================

    with nc.Dataset(in_nc_path, 'r') as nc_hdl:
        x_crds = nc_hdl[x_label][...].data
        y_crds = nc_hdl[y_label][...].data

        data = nc_hdl[var_label][...].data

    x_crds_plt_msh, y_crds_plt_msh = np.meshgrid(x_crds, y_crds)
    #==========================================================================

    out_var_figs_dir = out_figs_dir / var_label

    out_figs_dir.mkdir(exist_ok=True)
    out_var_figs_dir.mkdir(exist_ok=True)
    #==========================================================================

    for i in range(data.shape[0]):
        time_str = f'{i}'

        print(f'Plotting {time_str}')

        interp_fld = data[i]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            grd_min = np.nanmin(interp_fld)
            grd_max = np.nanmax(interp_fld)
        #======================================================================

        fig, ax = plt.subplots()

        pclr = ax.pcolormesh(
            x_crds_plt_msh,
            y_crds_plt_msh,
            interp_fld,
            vmin=grd_min,
            vmax=grd_max,
            shading='auto',
            cmap=cmap)

        cb = fig.colorbar(pclr)

        cb.set_label(var_type)

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

        title = (
            f'Time: {time_str}\n'
            f'Min.: {grd_min:0.4f}, Max.: {grd_max:0.4f}')

        ax.set_title(title)

        plt.setp(ax.get_xmajorticklabels(), rotation=70)
        ax.set_aspect('equal', 'datalim')

        out_fig_name = f'{var_label.lower()}_{time_str}.png'

        plt.savefig(str(out_var_figs_dir / out_fig_name), bbox_inches='tight')
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
