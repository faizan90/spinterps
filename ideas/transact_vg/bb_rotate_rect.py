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

    x0, y0 = 1, 1
    x1, y1 = 1, 2

    ang = (43) * np.pi / 180

    ofst_x = 0.1
    #==========================================================================

    x1 -= x0
    y1 -= y0

    poly_ref = np.array([
        [+ofst_x, +0.],
        [+ofst_x, +y1],
        [-ofst_x, +y1],
        [-ofst_x, +0.],
        [-ofst_x, -y1],
        [+ofst_x, -y1],
        [+ofst_x, +0.],
        ])

    poly_r = rotate_poly(poly_ref, ang)

    poly_r[:, 0] += x0
    poly_r[:, 1] += y0

    plt.plot(poly_r[:, 0], poly_r[:, 1])

    plt.gca().set_aspect('equal')

    plt.grid()
    plt.show()
    return


def rotate_poly(poly_ref, ang):

    xr0, yr0 = rotate_pt_2d(poly_ref[0, 0], poly_ref[0, 1], ang)
    xr1, yr1 = rotate_pt_2d(poly_ref[1, 0], poly_ref[1, 1], ang)
    xr2, yr2 = rotate_pt_2d(poly_ref[2, 0], poly_ref[2, 1], ang)
    xr3, yr3 = rotate_pt_2d(poly_ref[3, 0], poly_ref[3, 1], ang)
    xr4, yr4 = rotate_pt_2d(poly_ref[4, 0], poly_ref[4, 1], ang)
    xr5, yr5 = rotate_pt_2d(poly_ref[5, 0], poly_ref[5, 1], ang)

    poly_r = np.array([
        [xr0, yr0],
        [xr1, yr1],
        [xr2, yr2],
        [xr3, yr3],
        [xr4, yr4],
        [xr5, yr5],
        [xr0, yr0],
        ])

    return poly_r


def rotate_pt_2d(dx, dy, ang):

    xr = (dx * np.cos(ang)) - (dy * np.sin(ang))
    yr = (dx * np.sin(ang)) + (dy * np.cos(ang))

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
