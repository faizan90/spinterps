'''
Created on Jun 22, 2019

@author: Faizan
'''
import numpy as np
import pandas as pd

from ..cyth import sel_equidist_refs
from ..misc import print_sl, print_el


class SpInterpNeighborGrouping:

    def __init__(
            self,
            neb_sel_method,
            n_nebs,
            n_pies,
            n_dst_pts,
            dst_xs,
            dst_ys,
            verbose=True):

        self._vb = verbose

        self._vb_debug = False

        self._neb_sel_mthd = neb_sel_method
        self._n_nebs = n_nebs
        self._n_pies = n_pies
        self._n_dst_pts = n_dst_pts
        self._dst_xs = dst_xs
        self._dst_ys = dst_ys
        self._not_neb_flag = -1
        return

    def _get_grps_in_time(self, data_df):

        assert isinstance(data_df, pd.DataFrame)
        assert np.all(data_df.shape)
        assert isinstance(data_df.columns, pd.Index)

        hashes = []
        n_steps = data_df.shape[0]
        for i in range(n_steps):
            avail_stns = data_df.iloc[i].dropna().index.values
            hashes.append(hash(avail_stns.tobytes()))

        assert 0 < len(hashes) <= n_steps

        hashes = np.array(hashes)

        n_uniq_hashes = np.unique(hashes).size

        assert np.all(np.isfinite(hashes))

        time_neb_grps = []
        sel_flags = np.zeros(n_steps, dtype=bool)
        for i in range(n_steps):
            if sel_flags[i]:
                continue

            equal_idxs = (hashes == hashes[i])
            sel_flags[equal_idxs] = True

            time_neb_grps.append(
                (data_df.iloc[i].dropna().index, equal_idxs))

        assert np.all(sel_flags)
        assert len(time_neb_grps) == n_uniq_hashes

        if self._vb or self._vb_debug:
            print_sl()

            print(
                f'INFO: {n_uniq_hashes} station configurations over '
                f'{n_steps} steps!')

            print_el()

        return time_neb_grps

    def _get_neb_idxs_grps(self, all_neb_idxs):

        assert np.all(all_neb_idxs.shape)
        assert all_neb_idxs.ndim == 2
        assert np.issubdtype(all_neb_idxs.dtype, np.integer)

        hashes = []
        for i in range(all_neb_idxs.shape[0]):
            hashes.append(hash(all_neb_idxs[i].tobytes()))

        assert 0 < len(hashes) <= all_neb_idxs.shape[0]

        hashes = np.array(hashes)

        assert np.all(np.isfinite(hashes))

        n_unique_hashes = np.unique(hashes).size

        if self._vb_debug:
            print('N crds unique hashes:', n_unique_hashes)

        neb_idxs_grps = []
        sel_flags = np.zeros(self._n_dst_pts, dtype=bool)
        for i in range(self._n_dst_pts):
            if sel_flags[i]:
                continue

            equal_idxs = np.where((hashes == hashes[i]))[0]

            sel_flags[equal_idxs] = True

            neb_idxs_grps.append(equal_idxs)

        assert len(neb_idxs_grps) == n_unique_hashes
        assert np.all(sel_flags)

        return neb_idxs_grps

    def _get_all_neb_idxs(self, n_refs):

        all_neb_idxs = np.tile(np.arange(n_refs), (self._n_dst_pts, 1))

        return all_neb_idxs

    def _get_nrst_neb_idxs(self, ref_xs, ref_ys):

        all_neb_idxs = np.full(
            (self._n_dst_pts, self._n_nebs),
            self._not_neb_flag,
            dtype=int)

        for i in range(self._n_dst_pts):
            dst_x = self._dst_xs[i]
            dst_y = self._dst_ys[i]

            dists = (
                ((dst_x - ref_xs) ** 2) +
                ((dst_y - ref_ys) ** 2)) ** 0.5

            all_neb_idxs[i, :] = np.sort(np.argsort(dists)[:self._n_nebs])

        assert np.all(all_neb_idxs != self._not_neb_flag)

        return all_neb_idxs

    def _get_pie_neb_idxs(self, ref_xs, ref_ys, n_refs):

        min_dist_thresh = -1  # keep it -1

        dists = np.zeros(n_refs, dtype=np.float64)
        ref_pie_idxs = np.zeros(dists.size, dtype=np.uint32)
        tem_ref_sel_dists = np.zeros(dists.size, dtype=np.float64)
        ref_sel_pie_idxs = np.zeros(dists.size, dtype=np.int64)
        ref_pie_cts = np.zeros(self._n_pies, dtype=np.uint32)

        all_neb_idxs = np.full(
            (self._n_dst_pts, self._n_nebs),
            self._not_neb_flag,
            dtype=int)

        for i in range(self._dst_xs.size):
            dst_x = self._dst_xs[i]
            dst_y = self._dst_ys[i]

            sel_equidist_refs(
                dst_x,
                dst_y,
                ref_xs,
                ref_ys,
                self._n_pies,
                min_dist_thresh,
                self._not_neb_flag,
                dists,
                tem_ref_sel_dists,
                ref_sel_pie_idxs,
                ref_pie_idxs,
                ref_pie_cts)

            if min_dist_thresh < 0:
                assert np.all(ref_sel_pie_idxs != self._not_neb_flag)

            else:
                raise NotImplementedError

                assert (
                    np.all(ref_sel_pie_idxs != self._not_neb_flag) or
                    ((ref_sel_pie_idxs == self._not_neb_flag).sum() == (
                    self._n_nebs - 1)))

            ref_sel_pie_idxs_wh = np.where(
                ref_sel_pie_idxs != self._not_neb_flag)[0]

            uniq_ref_sel_pie_idxs = np.unique(
                ref_sel_pie_idxs[ref_sel_pie_idxs_wh])

            full_sorted_nebs_idxs = []
            for uniq_ref_sel_pie_idx in uniq_ref_sel_pie_idxs:
                same_pie_idxs = np.where(
                    ref_sel_pie_idxs == uniq_ref_sel_pie_idx)[0]

                same_pie_dist_sorted_idxs = (
                    same_pie_idxs[np.argsort(dists[same_pie_idxs])])

                full_sorted_nebs_idxs.extend(
                    same_pie_dist_sorted_idxs.tolist())

            full_sorted_nebs_idxs = np.array(full_sorted_nebs_idxs)

            if min_dist_thresh < 0:
                assert full_sorted_nebs_idxs.size == dists.size

            else:
                # only works if self._n_nebs is 1, which probably never
                # happens
                raise NotImplementedError

                assert (
                    (full_sorted_nebs_idxs.size == 1) or
                    (full_sorted_nebs_idxs.size == dists.size))

            all_neb_idxs[i, :] = full_sorted_nebs_idxs[:self._n_nebs]

        assert np.all(np.any(all_neb_idxs != self._not_neb_flag, axis=1))

        return all_neb_idxs

    def _get_neb_idxs_and_grps(self, ref_xs, ref_ys):

        assert ref_xs.ndim == ref_ys.ndim
        assert ref_xs.ndim == 1
        assert ref_xs.size == ref_ys.size

        n_refs = ref_xs.size

        if self._vb_debug:
            print('n_refs:', n_refs)

        assert n_refs

        if self._neb_sel_mthd == 'all':
            all_neb_idxs = self._get_all_neb_idxs(n_refs)

        elif self._neb_sel_mthd == 'nrst':
            all_neb_idxs = self._get_nrst_neb_idxs(ref_xs, ref_ys)

        elif self._neb_sel_mthd == 'pie':
            all_neb_idxs = self._get_pie_neb_idxs(ref_xs, ref_ys, n_refs)

        else:
            raise NotImplementedError

        neb_idxs_grps = self._get_neb_idxs_grps(all_neb_idxs)

        return all_neb_idxs, neb_idxs_grps

