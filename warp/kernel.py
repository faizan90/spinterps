'''
Created on Nov 29, 2019

@author: Faizan-Uni
'''

import numpy as np

from .base import Base

from .kernels import kernel_triangular
from .kernels import kernel_triangular_test

from ..misc import print_sl, print_el


class KernelOneDim(Base):

    def __init__(self, verbose=True):

        Base.__init__(self, verbose)

        self._kernel_ref_labels = ('TRI',)
        self._kernel_ref_prms_count = (4,)  # do manually
        self._kernel_ref_ftns = (kernel_triangular,)
        self._kernel_ref_test_ftns = (kernel_triangular_test,)

        assert (
            len(self._kernel_ref_labels) ==
            len(self._kernel_ref_prms_count) ==
            len(self._kernel_ref_ftns) ==
            len(self._kernel_ref_test_ftns))

        self._kernel_one_dim_labels = None
        self._kernel_one_dim_unq_labels = None

        self._kernel_one_dim_prms = None

        self._kernel_one_dim_transs = None

        self._kernel_one_dim_n_ftns = None

        self._set_kernel_one_dim_labels_flag = False
        self._set_kernel_one_dim_prms_flag = False
        self._set_kernel_one_dim_trans_flag = False

        self._kernel_one_dim_vrfd_flag = False
        return

    def set_kernel_one_dim_labels(self, kernel_labels):

        assert not self._set_kernel_one_dim_labels_flag, (
            'kernel_labels set already!')

        assert isinstance(kernel_labels, np.ndarray), (
            'kernel_labels not a numpy.ndarray object!')

        assert kernel_labels.ndim == 1, (
            'kernel_labels not a 1D array!')

        assert kernel_labels.shape[0], (
            'Zero elements in kernel_labels!')

        assert kernel_labels.dtype == '<U16', (
            'Data type of kernel_labels is not <U16!')

        unq_labels = np.unique(kernel_labels)

        assert all(
            [x in self._kernel_ref_labels for x in unq_labels]), (
                f'Unknown strength kernel labels in '
                f'kernel_labels.\nDefined ones are: '
                f'{self._kernel_ref_labels}, Given: {unq_labels}')

        self._kernel_one_dim_labels = kernel_labels
        self._kernel_one_dim_unq_labels = unq_labels

        if self._vb:
            print_sl()

            print('Set 1D strength kernel labels successfully!')

            print(
                f'Number of labels: '
                f'{self._kernel_one_dim_labels.shape[0]}')

            print(
                f'Unique kernel labels: '
                f'{self._kernel_one_dim_unq_labels}')

            print_el()

        self._set_kernel_one_dim_labels_flag = True
        return

    def set_kernel_one_dim_prms(self, kernel_prms):

        assert not self._set_kernel_one_dim_prms_flag, (
            'kernel_prms set already!')

        assert isinstance(kernel_prms, np.ndarray), (
            'kernel_prms not a numpy.ndarray object!')

        assert kernel_prms.ndim == 2, (
            'kernel_prms not a 2D array!')

        assert all(kernel_prms.shape), (
            'Zero elements in kernel_prms!')

        assert kernel_prms.dtype == np.float64, (
            'Data type of kernel_prms is not np.float!')

        assert np.all(~np.isinf(kernel_prms)), (
            'Infinity not allowed in kernel_prms!')

        nan_flags = np.isfinite(kernel_prms).astype(int)

        assert (nan_flags[:, :-1] - nan_flags[:, 1:]).min() >= 0, (
            'NaNs in between valid values in kernel_prms is not '
            'allowed!')

        del nan_flags

        self._kernel_one_dim_prms = kernel_prms

        if self._vb:
            print_sl()

            print('Set 1D strength kernel parameters successfully!')

            print(
                f'Number of parameter vectors: '
                f'{self._kernel_one_dim_prms.shape[0]}')

            print(
                f'Maximum possible parameters: '
                f'{self._kernel_one_dim_prms.shape[1]}')

            print_el()

        self._set_kernel_one_dim_prms_flag = True
        return

    def set_kernel_one_dim_translations(self, kernel_transs):

        assert not self._set_kernel_one_dim_trans_flag, (
            'kernel_transs set already!')

        assert isinstance(kernel_transs, np.ndarray), (
            '1D kernel_transs not a numpy.ndarray object!')

        assert kernel_transs.ndim == 1, (
            '1D kernel_transs not a 1D array!')

        assert kernel_transs.shape[0], (
            'Zero elements in 1D kernel_transs!')

        assert kernel_transs.dtype == np.float64, (
            'Data type of 1D kernel_transs is not np.float64!')

        assert np.all(np.isfinite(kernel_transs)), (
            'Invalid values in 1D kernel_transs!')

        self._kernel_one_dim_transs = kernel_transs

        if self._vb:
            print_sl()

            print('Set 1D kernel translations successfully!')

            print(
                f'Number of translations: '
                f'{self._kernel_one_dim_transs.shape[0]}')

            print(
                f'Minimum translation: {self._kernel_one_dim_transs.min()}')

            print(
                f'Maximum translation: {self._kernel_one_dim_transs.max()}')

            print_el()

        self._set_kernel_one_dim_trans_flag = True
        return

    def _verify(self):

        assert not self._kernel_one_dim_vrfd_flag, (
            'Kernel data verified already!')

        assert self._set_kernel_one_dim_labels_flag, (
            'Kernel labels not set!')

        assert self._set_kernel_one_dim_prms_flag, (
            'Kernel parameters not set!')

        assert self._set_kernel_one_dim_trans_flag, (
            'Kernel translations not set!')

        n_kern_labs = self._kernel_one_dim_labels.shape[0]
        n_kern_prms = self._kernel_one_dim_prms.shape[0]
        n_kern_transs = self._kernel_one_dim_transs.shape[0]

        self._kernel_one_dim_n_ftns = n_kern_labs

        assert n_kern_labs == n_kern_prms, (
            f'Unequal number of items in 1D kernel labels '
            f'({n_kern_labs}) and 1D kernel parameters ({n_kern_prms})!')

        assert n_kern_labs == n_kern_transs, (
            f'Unequal number of items in 1D kernel labels '
            f'({n_kern_labs}) and 1D kernel translations ({n_kern_transs})!')

        for kern_lab in self._kernel_one_dim_unq_labels:
            kern_idxs = np.where(
                self._kernel_one_dim_labels == kern_lab)[0]

            # test to have the appropriate number of arguments to the kernel
            # functions
            prms_cts = (
                np.isfinite(
                    self._kernel_one_dim_prms[kern_idxs])
                    ).sum(axis=1)

            kern_lab_idx = self._kernel_ref_labels.index(kern_lab)

            kern_prm_ct = self._kernel_ref_prms_count[kern_lab_idx]

            ct_test_pass_flag = True
            prm_ct_equal_flags = (prms_cts == kern_prm_ct)
            for i in range(prm_ct_equal_flags.shape[0]):
                if prm_ct_equal_flags[i]:
                    continue

                ct_test_pass_flag = False

                print(
                    f'Unequal number of parameters ({prms_cts[i]}) specified '
                    f'for the kernel function {kern_lab} that requires '
                    f'{kern_prm_ct} parameter(s) at index {kern_idxs[i]}!')

                continue

            if not ct_test_pass_flag:
                raise AssertionError(
                    f'Testing kernel function parameters failed for '
                    f'the kernel {kern_lab}!')

            # test for validity of input values to the kernel functions
            prm_test_pass_flag = True
            prm_test_ftn = self._kernel_ref_test_ftns[kern_lab_idx]
            for kern_idx in kern_idxs:
                try:
                    prm_test_ftn(
                        *self._kernel_one_dim_prms[
                            kern_idx][:kern_prm_ct])

                except Exception as msg:
                    print(msg, f'at index {kern_idx}!')

                    prm_test_pass_flag = False

                    continue

            if not prm_test_pass_flag:
                raise AssertionError(
                    f'Testing kernel function parameters failed for '
                    f'the kernel {kern_lab}!')

        self._kernel_one_dim_vrfd_flag = True
        return

    __verify = _verify
