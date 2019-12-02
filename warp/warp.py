'''
Created on Nov 29, 2019

@author: Faizan-Uni
'''

import numpy as np

from .strength import StrengthOneDim as SOD


class WarpOneDim(SOD):

    def __init__(self, verbose=True):

        SOD.__init__(self, verbose)

        self._warp_one_dim_diff_vec = None
        self._warp_one_dim_n_crds = None

        self._warp_one_dim_dst_crds = None

        self._warp_one_dim_vrfd_flag = False
        self._warp_one_dim_cmptd_flag = False
        return

    def compute_warp_one_dim_values(self):

        assert not self._warp_one_dim_cmptd_flag, (
            'Warp computed already!')

        assert self._warp_one_dim_vrfd_flag, (
            'Warp 1D data unverified!')

        self._warp_one_dim_dst_crds = np.zeros(
            (self._warp_one_dim_n_crds,), dtype=np.float64)

        for j in range(self._warp_one_dim_n_crds):
            warp_crd = 0.0

            for i in range(self._kernel_one_dim_n_ftns):
                warp_crd += (
                    self._kernel_one_dim_transs[i] *
                    self._strength_one_dim_raw_values[i, j] *
                    self._strength_one_dim_normed_values[i, j])

            self._warp_one_dim_dst_crds[j] = (
                warp_crd + self._crds_src_one_dim_crds[j])

        self._warp_one_dim_cmptd_flag = True
        return

    @property
    def dst_crds(self):

        assert self._warp_one_dim_cmptd_flag, (
            f'Warp not computed!')

        return self._warp_one_dim_dst_crds

    def _verify(self):

        assert not self._warp_one_dim_vrfd_flag, (
            'Warp data verified already!')

        SOD._StrengthOneDim__verify(self)

        assert self._strength_one_dim_vrfd_flag, (
            '1D strength\'s data in an unverified state!')

        self._warp_one_dim_n_crds = self._crds_src_one_dim_n_crds

        self._warp_one_dim_vrfd_flag = True
        return

    __verify = _verify
