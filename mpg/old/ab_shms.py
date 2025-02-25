# -*- coding: utf-8 -*-

'''
Created on Jul 6, 2024

@author: Faizan-TU Munich
'''

from multiprocessing import shared_memory

import numpy as np


def get_shm_arr(args, var_lab):

    shm = shared_memory.SharedMemory(
        name=getattr(args, f'shm__name__{var_lab}'))

    var_val = np.ndarray(
        shape=getattr(args, f'shm__shape__{var_lab}'),
        dtype=getattr(args, f'shm__dtype__{var_lab}'),
        order=getattr(args, f'shm__order__{var_lab}'),
        buffer=shm.buf)

    # This copy seems to be very important. Don't remove.
    return var_val.copy(order=getattr(args, f'shm__order__{var_lab}'))


def init_shm_arrs(
        args_cls, shm_args_cls, shm_vars_dict, ignore_exist_flag=False):

    for var_lab, var_val in shm_vars_dict.items():

        if var_val.flags.f_contiguous:
            order = 'f'

        else:
            order = 'c'

        shm = shared_memory.SharedMemory(create=True, size=var_val.nbytes)

        shr = np.ndarray(
            var_val.shape, dtype=var_val.dtype, buffer=shm.buf, order=order)

        shr[:] = var_val[:]

        if ignore_exist_flag and hasattr(shm_args_cls, f'shm__{var_lab}'):
            continue

        else:
            assert not hasattr(shm_args_cls, f'shm__{var_lab}')

        setattr(shm_args_cls, f'shm__{var_lab}', shm)
        setattr(shm_args_cls, f'shr__{var_lab}', shr)

        setattr(args_cls, f'shm__name__{var_lab}', shm.name)
        setattr(args_cls, f'shm__shape__{var_lab}', var_val.shape)
        setattr(args_cls, f'shm__dtype__{var_lab}', var_val.dtype)
        setattr(args_cls, f'shm__order__{var_lab}', order)

    args_cls.vrs = list(shm_vars_dict.keys())
    return


def fill_shm_arrs(args):

    for var_lab in args.vrs:
        shm = shared_memory.SharedMemory(
            name=getattr(args, f'shm__name__{var_lab}'))

        var_val = np.ndarray(
            shape=getattr(args, f'shm__shape__{var_lab}'),
            dtype=getattr(args, f'shm__dtype__{var_lab}'),
            order=getattr(args, f'shm__order__{var_lab}'),
            buffer=shm.buf,)

        setattr(args, var_lab, var_val)
        setattr(args, f'shm__buffer__{var_lab}', shm)

    args.init_shms_flag = True
    return


def free_shm_arrs(shm_args):

    for attr in dir(shm_args):

        if 'shm__' not in attr: continue

        getattr(shm_args, attr).close()
        getattr(shm_args, attr).unlink()

    return


class SHMARGS:

    def __init__(self):

        return
