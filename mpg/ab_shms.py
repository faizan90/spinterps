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

    '''
    Sync with attach_args_cls_vrs upon changes!
    '''

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


def attach_args_cls_vrs(args_cls, shm_arr_objs):

    '''
    Sync with init_shm_arrs upon changes!
    '''

    lbs = []
    for shm_arr_obj in shm_arr_objs:

        lbl = shm_arr_obj.lbl

        setattr(args_cls, f'shm__name__{lbl}', shm_arr_obj.nme)
        setattr(args_cls, f'shm__shape__{lbl}', shm_arr_obj.shp)
        setattr(args_cls, f'shm__dtype__{lbl}', shm_arr_obj.dte)
        setattr(args_cls, f'shm__order__{lbl}', shm_arr_obj.odr)

        lbs.append(lbl)
    #==========================================================================

    assert len(lbs) == len(set(lbs)), lbs

    # A list is needed.
    if not hasattr(args_cls, 'vrs'):
        args_cls.vrs = []

    else:
        assert len(set(lbs).intersection(set(args_cls.vrs))) == 0, (
            lbs, args_cls.vrs)

    args_cls.vrs.extend(lbs)
    return


def fill_shm_arrs(args):

    '''
    As far as I understand, reading from a buffer does not use new
    memory.
    '''

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


class SHMARY:

    '''
    Difference in syntax helps with not mixing this with regular arrays.
    '''

    def __init__(self, shape, dtype, order, label):

        # uint64 returns a float after the product with itemsize!
        nbs = int(dtype(1).itemsize) * int(np.prod(shape, dtype=np.uint64))

        # Unique names are better against really difficult bugs. Worth the
        # risk of more meory usage in case of duplication.
        shm = shared_memory.SharedMemory(create=True, size=nbs)

        shr = np.ndarray(shape, dtype=dtype, buffer=shm.buf, order=order)

        self.shm = shm
        self.shr = shr
        self.lbl = label
        self.odr = order
        self.shp = shape
        self.dte = dtype
        self.nme = shm.name
        return
