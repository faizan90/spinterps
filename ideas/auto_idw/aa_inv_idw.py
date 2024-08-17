# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

09.08.2024

17:54:02

Description:

Keywords:

'''

import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    ref_cds = np.array([10.0, 14.0])
    ref_vls = np.array([0.80, 0.40])

    dst_cde = 19.00
    dst_vlw_hde = 0.01
    #==========================================================================

    idw_dst, exp_dst = get_idw_exp(
        -2.0, 2.0, dst_vlw_hde, ref_cds, dst_cde, ref_vls)

    idw_dst_tst = get_idw_vle(ref_cds, dst_cde, exp_dst, ref_vls)

    print(idw_dst_tst, idw_dst, exp_dst)
    return


def get_idw_vle(ref_cds, dst_cde, idw_exp, ref_vls):

    dst_nrm = np.abs(ref_cds - dst_cde).max()

    ref_wts = 1.0 / (((((ref_cds - dst_cde) ** 2) ** 0.5) / dst_nrm) ** idw_exp)

    ref_wts /= ref_wts.sum()

    dst_vle = (ref_wts * ref_vls).sum()

    return dst_vle


def get_idw_exp(exp_llm, exp_ulm, dst_vle, ref_cds, dst_cde, ref_vls):

    '''
    Bisection algorithm.
    '''

    sgn = -np.sign(np.corrcoef(ref_cds, ref_vls))[0, 1]

    min_tol = 1e-15

    tol = np.inf

    exp_lft = exp_llm
    exp_rht = exp_ulm

    idw_lft = get_idw_vle(ref_cds, dst_cde, exp_lft, ref_vls)
    idw_rht = get_idw_vle(ref_cds, dst_cde, exp_rht, ref_vls)

    idw_mde_pre = np.inf

    itr_ctr = 0
    while tol > min_tol:

        exp_mde = 0.5 * (exp_lft + exp_rht)

        idw_mde = get_idw_vle(ref_cds, dst_cde, exp_mde, ref_vls)

        print(itr_ctr, idw_lft, idw_mde, idw_rht)
        print(exp_lft, exp_mde, exp_rht)
        print('\n')

        if np.isclose(idw_mde, dst_vle):

            break

        elif np.isclose(idw_lft, dst_vle):

            idw_mde = idw_lft
            break

        elif np.isclose(idw_rht, dst_vle):

            idw_mde = idw_rht
            break

        elif (dst_vle < idw_lft < idw_rht):

            exp_rht = exp_lft
            idw_rht = idw_lft

            exp_lft = exp_lft - sgn
            idw_lft = get_idw_vle(ref_cds, dst_cde, exp_lft, ref_vls)

        elif (dst_vle < idw_rht < idw_lft):

            exp_lft = exp_rht
            idw_lft = idw_rht

            exp_rht = exp_rht - sgn
            idw_rht = get_idw_vle(ref_cds, dst_cde, exp_rht, ref_vls)

        elif (dst_vle > idw_lft > idw_rht):

            exp_rht = exp_lft
            idw_rht = idw_lft

            exp_lft = exp_lft + sgn
            idw_lft = get_idw_vle(ref_cds, dst_cde, exp_lft, ref_vls)

        elif (dst_vle > idw_rht > idw_lft):

            exp_lft = exp_rht
            idw_lft = idw_rht

            exp_rht = exp_rht + sgn
            idw_rht = get_idw_vle(ref_cds, dst_cde, exp_rht, ref_vls)

        elif ((idw_lft <= dst_vle <= idw_mde) or
              (idw_lft >= dst_vle >= idw_mde)):

            exp_rht = exp_mde
            idw_rht = idw_mde

        elif ((idw_mde <= dst_vle <= idw_rht) or
              ((idw_mde >= dst_vle >= idw_rht))):

            exp_lft = exp_mde
            idw_lft = idw_mde

        else:
            print('Failed!')
            break

        tol = abs(idw_mde_pre - idw_mde)

        if tol < min_tol:
            print('Min. Tol.')

        itr_ctr += 1

        idw_mde_pre = idw_mde

    idw_fnl = get_idw_vle(ref_cds, dst_cde, exp_mde, ref_vls)

    tol = abs(idw_mde - idw_fnl)

    return idw_fnl, exp_mde


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
