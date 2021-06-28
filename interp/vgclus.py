'''
@author: Faizan3800X-Uni

Jun 24, 2021

3:31:23 PM
'''
import numpy as np
import pandas as pd


class VariogramCluster:

    def __init__(self, vgs_ser, verbose=True):

        self._vb = verbose

        self._vb_debug = False

        assert isinstance(vgs_ser, pd.Series)

        assert vgs_ser.size

        assert np.all(vgs_ser.notna().values)

        assert vgs_ser.dtype == object

        self._vgs = vgs_ser.values
        self._tidxs = vgs_ser.index
        self._n_vgs = vgs_ser.size
        return

    def _same_vgs_strs_idxs(self):

        hashes = []
        for i in range(self._n_vgs):
            hashes.append(hash(self._vgs[i]))

        assert 0 < len(hashes) <= self._n_vgs

        hashes = np.array(hashes)

        assert np.all(np.isfinite(hashes))

        n_unique_hashes = np.unique(hashes).size

        if self._vb_debug:
            print('Total vgs unique hashes:', n_unique_hashes)

        same_idxs_dict = {}
        vgs_ser = pd.Series(index=self._tidxs, dtype=str)
        sel_flags = np.zeros(self._n_vgs, dtype=bool)

        for i in range(self._n_vgs):
            if sel_flags[i]:
                continue

            equal_idxs = np.where((hashes == hashes[i]))[0]

            sel_flags[equal_idxs] = True

            same_idxs_dict[self._vgs[i]] = self._tidxs[equal_idxs]
            vgs_ser.loc[self._tidxs[equal_idxs]] = self._vgs[i]

        assert len(same_idxs_dict) == n_unique_hashes
        assert np.all(sel_flags)

        return same_idxs_dict, vgs_ser

    def get_vgs_cluster(self):

        '''
        Return a dictionary with time step indices as values for each
        variogram string.
        '''

        vgs_clus_dict, vgs_ser = self._same_vgs_strs_idxs()

        return vgs_clus_dict, vgs_ser
