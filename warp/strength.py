'''
Created on Nov 29, 2019

@author: Faizan-Uni
'''

import numpy as np

from .crds_src import CrdsSrcOneDim as CSOD
from .kernel import KernelOneDim as KOD

from ..misc import print_sl, print_el


class StrengthOneDim(CSOD, KOD):

    def __init__(self, verbose=True):

        CSOD.__init__(self, verbose)
        KOD.__init__(self, verbose)

        self._strength_one_dim_raw_values = None
        self._strength_one_dim_normed_values = None

        self._set_strength_one_dim_flags_flag = False
        self._strength_one_dim_vrfd_flag = False
        self._strength_one_dim_cmptd_flag = False
        return

    def set_strength_flags(self, strength_flags):

        assert not self._set_strength_one_dim_flags_flag, (
            'Strength flags set already!')

        assert isinstance(strength_flags, np.ndarray), (
            '1D strength_flags not a numpy.ndarray object!')

        assert strength_flags.ndim == 1, (
            '1D strength_flags not a 1D array!')

        assert strength_flags.shape[0], (
            'Zero elements in 1D strength_flags!')

        assert strength_flags.dtype == np.int8, (
            'Data type of 1D strength_flags is not np.int8!')

        assert np.all(np.isfinite(strength_flags)), (
            'Invalid values in 1D strength_flags!')

        assert strength_flags.min() >= 0, (
            f'1D strength_flags allowed to have zeros and ones only!')

        assert strength_flags.max() == 1, (
            f'1D strength_flags allowed to have zeros and ones only!')

        self._strength_one_dim_cmpt_flags = strength_flags

        if self._vb:
            print_sl()

            print('Set 1D strength computation flags successfully!')

            print(
                f'Number of flags: '
                f'{self._strength_one_dim_cmpt_flags.shape[0]}')

            print_el()

        self._set_strength_one_dim_flags_flag = True
        return

    def compute_strength_one_dim_values(self):

        assert not self._strength_one_dim_cmptd_flag, (
            'Strength computed already!')

        assert self._strength_one_dim_vrfd_flag, (
            'Strength 1D data unverified!')

        self._strength_one_dim_raw_values = np.full(
            (self._kernel_one_dim_n_ftns,
             self._crds_src_one_dim_n_crds),
            np.nan,
            dtype=np.float64)

        self._strength_one_dim_normed_values = (
            self._strength_one_dim_raw_values.copy())

        for i in range(self._kernel_one_dim_n_ftns):
            kern_lab = self._kernel_one_dim_labels[i]

            kern_ftn_idx = self._kernel_ref_labels.index(kern_lab)

            kern_prms_ct = self._kernel_ref_prms_count[kern_ftn_idx]

            kern_ftn = self._kernel_ref_ftns[kern_ftn_idx]
            kern_prms = self._kernel_one_dim_prms[i, :kern_prms_ct]

            for j in range(self._crds_src_one_dim_n_crds):
                if self._strength_one_dim_cmpt_flags[j]:
                    self._strength_one_dim_raw_values[i, j] = kern_ftn(
                        self._crds_src_one_dim_crds[j], *kern_prms)

                else:
                    self._strength_one_dim_raw_values[i, j] = 0.0

        assert np.all(np.isfinite(self._strength_one_dim_raw_values)), (
            'Invalid values in 1D strength raw values!')

        assert np.all(
            (self._strength_one_dim_raw_values >= 0) &
            (self._strength_one_dim_raw_values <= 1)), (
                '1D strength raw values out of range!')

        for j in range(self._crds_src_one_dim_n_crds):
            strengths_sum = self._strength_one_dim_raw_values[:, j].sum()

            if not strengths_sum:
                strengths_normed = 0.0

            else:
                strengths_normed = (
                    self._strength_one_dim_raw_values[:, j] / strengths_sum)

            self._strength_one_dim_normed_values[:, j] = strengths_normed

        assert np.all(np.isfinite(self._strength_one_dim_normed_values)), (
            'Invalid values in 1D strength normed values!')

        assert np.all(
            (self._strength_one_dim_normed_values >= 0) &
            (self._strength_one_dim_normed_values <= 1)), (
                '1D strength normed values out of range!')

        self._strength_one_dim_cmptd_flag = True
        return

    def _verify(self):

        assert not self._strength_one_dim_vrfd_flag, (
            'Strength data verified already!')

        CSOD._CrdsSrcOneDim__verify(self)

        assert self._crds_src_one_dim_vrfd_flag, (
            '1D source coordinates\' data in an unverified state!')

        KOD._KernelOneDim__verify(self)

        assert self._kernel_one_dim_vrfd_flag, (
            '1D kernel data in an unverified state!')

        assert self._set_strength_one_dim_flags_flag, (
            'Strength flags not set!')

        self._strength_one_dim_vrfd_flag = True
        return

    __verify = _verify
