# -*- coding: utf-8 -*-

'''
Created on Jul 6, 2024

@author: Faizan-TU Munich
'''

from multiprocessing import shared_memory

import numpy as np


class SHMARY2:

    '''
    - This array will behave as a numpy array in normal cases.
    - Upon pickling, it will not allow the main array (shr) to be pickled.
    - When shr is accessed in a another process, it is read from the buffer
      created by the main process and then it behaves as a numpy array.
    '''

    def __init__(self, shape, dtype, order, label):

        # uint64 returns a float after the product with itemsize!
        nbs = int(dtype(1).itemsize) * int(np.prod(shape, dtype=np.uint64))

        # Size in GigaBytes.
        # print(nbs / (1024 ** 3.))

        # Unique names are better against really difficult bugs. Worth the
        # risk of more memory usage in case of duplication.
        shm = shared_memory.SharedMemory(create=True, size=nbs)

        shr = np.ndarray(shape, dtype=dtype, buffer=shm.buf, order=order)

        self.shm = shm
        self.lbl = label
        self.odr = order
        self.shp = shape
        self.dte = dtype
        self.nme = shm.name

        # shr is a property that uses _shr.
        self._shr = shr
        return

    def _init_shr(self):

        # Don't use hasattr. That triggers a recursion for some reason.
        if '_shr' not in self.__dict__:

            self._shr = np.ndarray(
                self.shp,
                dtype=self.dte,
                buffer=self.shm.buf,
                order=self.odr)

        return

    @property
    def shr(self):

        self._init_shr()

        return self._shr

    def __getstate__(self):

        '''
        Result of this what will get pickled at least during mprg.

        Large objects such as ndarray should be kept out!
        '''

        stt = self.__dict__.copy()
        if '_shr' in stt: del stt['_shr']

        return stt

    def __setstate__(self, stt):

        '''
        This is for after pickling.
        '''

        self.__dict__ = stt
        return

    def __getitem__(self, key):

        '''
        This allows for Numpy like indexing on the SHMARY2 object.
        '''

        self._init_shr()

        return self._shr.__getitem__(key)

    def __setitem__(self, key, vle):

        '''
        This allows for Numpy like indexing on the SHMARY object.
        '''

        self._init_shr()

        return self._shr.__setitem__(key, vle)

    def __copy__(self):

        shy = SHMARY2(self.shp, self.dte, self.odr, self.lbl)

        if '_shr' in self.__dict__: shy[:] = self._shr

        return shy

    def __getattr__(self, key):

        '''
        This is called when default access method fails. When it fails, then
        the method of the array is called.
        '''

        self._init_shr()

        return getattr(self._shr, key)

    def close(self):

        '''
        For a process that did not create this.
        '''

        self.shm.close()
        return

    def unlink(self):

        '''
        For a process that created this.
        '''

        self.shm.unlink()
        return

    @staticmethod
    def frm_npy_ary(ary):

        if ary.flags.f_contiguous:
            order = 'f'

        else:
            order = 'c'

        shy = SHMARY2(ary.shape, ary.dtype.type, order, None)

        shy[:] = ary
        return shy


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
