# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jan 12, 2024

1:40:50 PM

Keywords:

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    x0, y0 = 0, 0
    x1, y1 = 1, 1

    ang = (180 - 45) * np.pi / 180
    #==========================================================================

    xr, yr = rotate_pt_2d(x0, x1, y0, y1, ang)

    plt.plot([x0, x1], [y0, y1])
    plt.plot([x0, xr], [y0, yr])

    plt.gca().set_aspect('equal')

    plt.grid()
    plt.show()
    return


def rotate_pt_2d(x0, x1, y0, y1, ang):

    dx = x1 - x0
    dy = y1 - y0

    xr = (dx * np.cos(ang)) - (dy * np.sin(ang)) + x0
    yr = (dx * np.sin(ang)) + (dy * np.cos(ang)) + y0

    # dist = ((dx ** 2) + (dy ** 2)) ** 0.5
    # xr = (dist * np.cos(ang)) + x0
    # yr = (dist * np.sin(ang)) + y0

    return xr, yr


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
