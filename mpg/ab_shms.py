# -*- coding: utf-8 -*-

'''
Created on Feb 25, 2025

@author: Faizan-TU Munich
'''

from multiprocessing import shared_memory

import numpy as np


class SHMARY:

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

        self._init_shr_flg = True
        return

    def _init_shr(self):

        # Don't use hasattr. That triggers a recursion for some reason.
        if '_shr' not in self.__dict__:

            self._shr = np.ndarray(
                self.shp,
                dtype=self.dte,
                buffer=self.shm.buf,
                order=self.odr)

            self._init_shr_flg = False

        return

    @property
    def shr(self):

        if self._init_shr_flg: self._init_shr()

        return self._shr

    def __getstate__(self):

        '''
        Result of this what will get pickled at least during mprg.

        Large objects such as ndarray should be kept out!
        '''

        stt = self.__dict__.copy()

        if '_shr' in stt:
            self._init_shr_flg = True
            del stt['_shr']

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

        if self._init_shr_flg: self._init_shr()

        return self._shr.__getitem__(key)

    def __setitem__(self, key, vle):

        '''
        This allows for Numpy like indexing on the SHMARY object.
        '''

        if self._init_shr_flg: self._init_shr()

        return self._shr.__setitem__(key, vle)

    def __copy__(self):

        shy = SHMARY(self.shp, self.dte, self.odr, self.lbl)

        if '_shr' in self.__dict__: shy[:] = self._shr

        return shy

    def __getattr__(self, key):

        '''
        This is called when default access method fails. When it fails, then
        the method of the array is called.
        '''

        if self._init_shr_flg: self._init_shr()

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

        shy = SHMARY(ary.shape, ary.dtype.type, order, None)

        shy[:] = ary
        return shy


def fre_shm_ars(ags):

    for attr in dir(ags):

        obj = getattr(ags, attr)

        if not isinstance(obj, SHMARY): continue

        obj.close(); obj.unlink()

    return
