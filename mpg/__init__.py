# -*- coding: utf-8 -*-

'''
Created on Jul 6, 2024

@author: Faizan-TU Munich
'''

import os
import sys
import time
import traceback as tb
from functools import wraps
from datetime import datetime
from timeit import default_timer

from .aa_lcks import DummyLock, DummyManager
from .ab_shms import SHMARY2, fre_shm_ars


def wait_for_pool_init(mp_pool, pol_sze, pol_nme, vb):

    trs = 0
    tmr_bgn = default_timer()
    while True:
        pol_res = list(mp_pool.map(get_pid, range(pol_sze), chunksize=1))

        time.sleep(0.01 * pol_sze)

        trs += 1

        if len(pol_res) == len(set(pol_res)): break

    tmr_end = default_timer()

    if vb:

        if len(pol_res) > 8: pol_res = '[...]'

        print(
            f'{pol_nme}', '|',
            f'{pol_sze}', '|',
            os.getppid(), '|',
            pol_res, '|',
            trs, '|',
            f'{tmr_end - tmr_bgn:0.2f} secs')
    return


def get_pid(args):

    _ = args

    return os.getpid()
