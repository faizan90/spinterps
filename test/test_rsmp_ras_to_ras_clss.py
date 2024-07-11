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

from spinterps import ResampleRasToRasClss

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\spinterps\rsmp\ras_to_ras')
    os.chdir(main_dir)

    src_pth = Path(r'vils_rott_isen_corine_2018_domain.tif')  #  Path(r'U:\TUM\projects\altoetting\landuse\DATA\U2018_CLC2018_V2020_20u1.tif')  # Path(r'vils_rott_isen_fil_1km.tif')  # Path(r'U2018_CLC2018_V2020_20u1_utm32N__bayern_epsg32632.tif')  #
    dst_pth = Path(r'vils_rott_isen_fil_1km.tif')

    out_pth = Path(r'ref_aln_clss.tif')

    n_cpus = 'auto'
    #==========================================================================

    rsp_obj = ResampleRasToRasClss(True)

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
