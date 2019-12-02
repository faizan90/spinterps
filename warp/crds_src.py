'''
Created on Nov 29, 2019

@author: Faizan-Uni
'''

import numpy as np

from .base import Base

from ..misc import print_sl, print_el


class CrdsSrcOneDim(Base):

    def __init__(self, verbose=True):

        assert isinstance(verbose, bool), 'verbose can only be a boolean!'

        Base.__init__(self, verbose)

        self._crds_src_one_dim_crds = None
        self._crds_src_one_dim_label = None
        self._crds_src_one_dim_n_crds = None

        self._set_crds_src_one_dim_flag = False
        self._crds_src_one_dim_vrfd_flag = False
        return

    def set_src_one_dim_crds(self, coordinates, label):

        assert not self._set_crds_src_one_dim_flag, (
            '1D source coordinates set already!')

        assert isinstance(coordinates, np.ndarray), (
            '1D source coordinates not a numpy.ndarray object!')

        assert coordinates.ndim == 1, (
            '1D source coordinates not a 1D array!')

        assert coordinates.shape[0], (
            'Zero elements in 1D source coordinates!')

        assert coordinates.dtype == np.float64, (
            'Data type of 1D source coordinates is not np.float!')

        assert np.all(np.isfinite(coordinates)), (
            'Invalid values in 1D source coordinates!')

        assert isinstance(label, str), (
            '1D source coordinates\' label not a string!')

        assert len(label), (
            '1D source coordinates\' label has no characters!')

        self._crds_src_one_dim_crds = coordinates
        self._crds_src_one_dim_label = label
        self._crds_src_one_dim_n_crds = self._crds_src_one_dim_crds.shape[0]

        if self._vb:
            print_sl()

            print(f'Set 1D source coordinates successfully!')

            print(f'Label: {self._crds_src_one_dim_label}')

            print(
                f'Number of elements: {self._crds_src_one_dim_n_crds}')

            print(f'Minimum coordinate: {self._crds_src_one_dim_crds.min()}')

            print(f'Maximum coordinate: {self._crds_src_one_dim_crds.max()}')

            print_el()

        self._set_crds_src_one_dim_flag = True
        return

    def _verify(self):

        assert not self._crds_src_one_dim_vrfd_flag, (
            '1D source coordinates\' data verfied already!')

        assert self._set_crds_src_one_dim_flag, (
            '1D source coordinates not set!')

        self._crds_src_one_dim_vrfd_flag = True
        return

    __verify = _verify
