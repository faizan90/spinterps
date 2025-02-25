# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

09.07.2024

08:28:46

Description:

Keywords:

'''

import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

from spinterps import ResampleRasToRas

DEBUG_FLAG = True


def main():

    main_dir = Path(r'P:\Synchronize\TUM\Colleagues_Students\hadi\modis')
    os.chdir(main_dir)

    # src_pth = Path(r'lower_de_gauss_z3.tif')
    # dst_pth = Path(r'lower_de_gauss_z3_1km.tif')
    # out_pth = Path(r'ref_aln1.tif')

    src_pth = Path(r'P:\Synchronize\IWS\Hydrological_Modeling\dems\srtm_de_mosaic_utm32N_100m_bayern.tif')
    dst_pth = Path(r'modis_hmg3d_itsct_grid.tif')
    out_pth = Path(r'modis_hmg3d_itsct_srtm_test.tif')

    # src_pth = Path(r'srtm_de_mosaic_utm32N_25m_bayern.tif')
    # dst_pth = Path(r'srtm_de_mosaic_utm32N_25m_bayern_to_1km_hydr.tif')
    # out_pth = Path(r'ref_aln3.tif')

    # src_pth = Path(r'dem_1km_raw.tif')
    # dst_pth = Path(r'AWC_eu23_utm32N.tif')
    # out_pth = Path(r'ref_aln4.tif')

    n_cpus = 4  # 'auto'
    #==========================================================================

    rsp_obj = ResampleRasToRas(True)

    rsp_obj.set_inputs(src_pth, dst_pth, n_cpus)
    rsp_obj.set_outputs(out_pth)

    rsp_obj.resample()
    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
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
