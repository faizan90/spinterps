'''
Created on Nov 29, 2019

@author: Faizan-Uni
'''

import numpy as np

from .crds_src import CrdsSrcOneDim as CSOD
from .kernel import KernelOneDim as KOD


class StrengthOneDim(CSOD, KOD):

    def __init__(self, verbose=True):

        CSOD.__init__(self, verbose)
        KOD.__init__(self, verbose)

        self._strength_one_dim_raw_values = None
        self._strength_one_dim_normed_values = None

        self._strength_one_dim_vrfd_flag = False
        self._strength_one_dim_cmptd_flag = False
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
                self._strength_one_dim_raw_values[i, j] = kern_ftn(
                    self._crds_src_one_dim_crds[j], *kern_prms)

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

        self._strength_one_dim_vrfd_flag = True
        return

    __verify = _verify
