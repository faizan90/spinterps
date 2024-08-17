# -*- coding: utf-8 -*-

'''
Created on Aug 17, 2024

@author: Faizan-TU Munich
'''

import numpy as np


class NNB:

    '''
    Interpolate a point in time using the Nearest Neighbors
    (Thiessen Polygons) method.
    '''

    # Used as an index when all values at a time step are not finite.
    nan_int = (2 ** 64) - 1

    def __init__(self, sim_ref_dts, ref_tss_dfe):

        '''
        sim_ref_dts: Distances between each point in ref to sim point.
        ref_tss_dfe: Dataframe of time series of reference points.
        '''

        assert sim_ref_dts.ndim == 1, sim_ref_dts.ndim
        assert ref_tss_dfe.values.ndim == 2, ref_tss_dfe.values.ndim

        assert ref_tss_dfe.shape[1] == sim_ref_dts.shape[0], (
            ref_tss_dfe.shape[1], sim_ref_dts.shape[0])

        self._sim_ref_dts = sim_ref_dts
        self._ref_tss_dfe = ref_tss_dfe
        return

    def get_ipn_vls(self):

        nst_nbr_ixs = self.get_nst_nbr_ixs()

        ipn_vls = np.empty(self._ref_tss_dfe.shape[0], dtype=np.float32)

        for i in range(ipn_vls.size):

            j = nst_nbr_ixs[i]

            if j == self.nan_int: continue

            ipn_vls[i] = self._ref_tss_dfe.iloc[i, j]

        return ipn_vls

    def get_nst_nbr_ixs(self):

        '''
        All time steps.
        '''

        nst_nbr_ixs = np.full(
            self._ref_tss_dfe.shape[0], self.nan_int, dtype=np.uint64)

        for i in range(nst_nbr_ixs.size):
            nst_nbr_ixs[i] = self.get_nst_nbr_idx(i)

        return nst_nbr_ixs

    def get_nst_nbr_idx(self, i):

        '''
        Just for one time step.
        '''

        vls_ser = self._ref_tss_dfe.iloc[i,:]

        vls_fnt_ixs = np.isfinite(vls_ser.values)

        if vls_fnt_ixs.any():
            nst_nbr_idx = np.argmin(self._sim_ref_dts[vls_fnt_ixs])
            nst_nbr_idx = np.where(vls_fnt_ixs)[0][nst_nbr_idx]

        else:
            nst_nbr_idx = self.nan_int

        return nst_nbr_idx
