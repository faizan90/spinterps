# -*- coding: utf-8 -*-

'''
Created on 08.07.2026

@author: Faizan-TU Munich
'''

import sys
import traceback as tb
from multiprocessing import Pool as MPPool

import numpy as np
import pandas as pd

from ..mpg import wait_for_pool_init

REF_XCL = 'X'
REF_YCL = 'Y'
DST_XCL = 'X'
DST_YCL = 'Y'

OTT_XCL = 'X'
OTT_YCL = 'Y'


class PointToPointDisAgg:

    def __init__(self, verbose=False, pol_sze=1):

        '''
        Ref is coarse and dst is fine.
        '''

        assert isinstance(verbose, bool)
        assert isinstance(pol_sze, int)
        assert 1 <= pol_sze <= 60

        self._ref_dfe_tss = None
        self._ref_dfe_cds = None
        self._ref_tfq = None

        self._dst_dfe_tss = None
        self._dst_dfe_cds = None
        self._dst_tfq = None

        self._ott_dfe_tss = None
        self._ott_dfe_cds = None

        self._ppr_flg = False
        self._dsg_flg = False

        self._vb = verbose

        self._mpg_ncs = 1  # pol_sze
        return

    def set_ref(self, ref_dfe_tss, ref_dfe_cds, ref_tfq):

        self._ref_dfe_tss = ref_dfe_tss
        self._ref_dfe_cds = ref_dfe_cds
        self._ref_tfq = ref_tfq
        return

    def set_dst(self, dst_dfe_tss, dst_dfe_cds, dst_tfq):

        self._dst_dfe_tss = dst_dfe_tss
        self._dst_dfe_cds = dst_dfe_cds
        self._dst_tfq = dst_tfq
        return

    def prepare(self):

        if self._vb: print('Preparing....')

        self._chk_vbs(
             self._ref_dfe_tss, self._ref_dfe_cds, self._ref_tfq, 'ref')

        self._chk_vbs(
             self._dst_dfe_tss, self._dst_dfe_cds, self._dst_tfq, 'dst')

        self._ppr_flg = True
        return

    def disaggregate(self):

        assert self._ppr_flg

        self._ppr_flg = False
        self._dsg_flg = False

        self._ott_dfe_tss = None
        self._ott_dfe_cds = None
        #======================================================================

        # Distance dataframe.
        ref_dst_dfe_dts = pd.DataFrame(
            index=self._ref_dfe_cds.index,
            columns=self._dst_dfe_cds.index,
            dtype=np.float64)

        # slt_sts = ('dtc', 'pcr', 'scr',)
        # slt_sts = ('scr',)
        slt_sts = ('dtc',)  # Distance is needed to get nebors that have zero
        #                     but different name.

        dts_dfs = {key:ref_dst_dfe_dts.copy() for key in slt_sts}

        assert len(slt_sts)

        if self._vb: print('Calculating distances....')

        self._dtc_max = 20e3

        dtc_mtx = get_dtc_mtx(
            self._ref_dfe_cds[REF_XCL].values,
            self._ref_dfe_cds[REF_YCL].values,
            self._dst_dfe_cds[DST_XCL].values,
            self._dst_dfe_cds[DST_YCL].values)

        # dtc_mtx[dtc_mtx > self._dtc_max] = np.nan

        #  Only distance for now.
        dts_dfs['dtc'][:] = dtc_mtx

        # dtc_mtx = None

        dtc_dtn = []
        # for ref_lbl in ref_dst_dfe_dts.index:
        #
        #     (ref_xcd,
        #      ref_ycd) = self._ref_dfe_cds.loc[ref_lbl, [REF_XCL, REF_YCL]]
        #
        #     ref_tss = self._ref_dfe_tss.loc[:, ref_lbl].copy()
        #
        #     for dst_lbl in ref_dst_dfe_dts.columns:
        #
        #         if 'dtc' in slt_sts:
        #
        #             (dst_xcd,
        #              dst_ycd) = self._dst_dfe_cds.loc[
        #                  dst_lbl, [DST_XCL, DST_YCL]]
        #
        #             dtc_dtc = self._get_dtc(ref_xcd, ref_ycd, dst_xcd, dst_ycd)
        #
        #             dts_dfs['dtc'].loc[ref_lbl, dst_lbl] = dtc_dtc
        #
        #         if 'pcr' in slt_sts:
        #
        #             dst_tss = self._dst_dfe_tss.loc[:, dst_lbl].copy()
        #
        #             dtc_pcr = self._get_pcr(ref_tss, dst_tss)
        #
        #             dts_dfs['pcr'].loc[ref_lbl, dst_lbl] = dtc_pcr
        #
        #         if 'scr' in slt_sts:
        #
        #             dst_tss = self._dst_dfe_tss.loc[:, dst_lbl].copy()
        #
        #             dtc_scr = self._get_scr(ref_tss, dst_tss)
        #
        #             dts_dfs['scr'].loc[ref_lbl, dst_lbl] = dtc_scr
        #
        #     if False:
        #         print(ref_lbl)
        #         # print(1. - dts_dfs['scr'].loc[ref_lbl,:].min())
        #         print(dts_dfs['dtc'].loc[ref_lbl,:].min())
        #         print('\n')

            # dtc_dtn.append(1. - dts_dfs['scr'].loc[ref_lbl,:].min())

        if False:
            dtc_dtn = np.array(dtc_dtn)
            dtc_dtn.sort()
            dtn_pbs = np.arange(1., dtc_dtn.size + 1.0) / (1.0 + dtc_dtn.size)

            import matplotlib.pyplot as plt
            plt.plot(dtc_dtn, dtn_pbs)
            plt.grid()
            plt.gca().set_axisbelow(True)
            plt.xlim(0.5, 1)
            plt.show()

        assert all([np.isfinite(dts_dfs[key].values).all() for key in slt_sts])

        dts_dfs_rks = {key:dts_dfs[key].rank(axis=0) for key in slt_sts}

        ref_dst_dfe_dts[:] = 0.0
        for dts_dfe in dts_dfs_rks.values(): ref_dst_dfe_dts += dts_dfe
        ref_dst_dfe_dts /= len(slt_sts)
        #======================================================================

        # Points having no distance in ref and dst need considered from ref
        # only.
        min_dts = dts_dfs['dtc'].values.min(axis=1)

        min_dts[np.isclose(min_dts, 0.0)] = 0.0

        tke_ref_ixs = self._ref_dfe_cds.index[min_dts > 0]

        ott_lbs = tke_ref_ixs.union(self._dst_dfe_cds.index)
        #======================================================================

        # This could be better. Maybe later.
        ref_dst_ixs_ist = self._ref_dfe_tss.index.intersection(
            self._dst_dfe_tss.index)

        assert ref_dst_ixs_ist.size >= 3

        ott_dfe_tss = pd.DataFrame(
            index=self._dst_dfe_tss.index, columns=ott_lbs, dtype=np.float32)

        ott_dfe_tss.update(self._dst_dfe_tss)

        ott_dfe_cds = pd.DataFrame(
            index=ott_lbs, columns=[OTT_XCL, OTT_YCL], dtype=np.float64)

        ott_dfe_cds.update(self._ref_dfe_cds)
        ott_dfe_cds.update(self._dst_dfe_cds)

        assert np.isfinite(ott_dfe_cds.values).all()
        #======================================================================

        if self._vb: print('Disaggregating....')

        dsg_ags = []
        for ott_lbl in ott_dfe_tss.columns:

            if ott_lbl in self._dst_dfe_tss.columns: continue

            ref_lbl_srs_tss = self._ref_dfe_tss.loc[:, ott_lbl]

            ref_dst_lbl_srs_dts = ref_dst_dfe_dts.loc[ott_lbl,:]

            dsg_ags.append((
                self._vb,
                ott_lbl,
                self._dst_tfq,
                ref_lbl_srs_tss,
                self._dst_dfe_tss,
                ref_dst_lbl_srs_dts))
        #======================================================================

        if len(dsg_ags):

            print('dsg_ags:', len(dsg_ags))

            if self._mpg_ncs > 1:

                if self._vb: print('Initiating multiprocessing pool...')

                pol_sze = min(len(dsg_ags), self._mpg_ncs)

                mpg_pol = MPPool(pol_sze)

                wait_for_pool_init(mpg_pol, pol_sze, 'DSG', True)

                ress = mpg_pol.imap_unordered(_dsg_sgl, dsg_ags, chunksize=1)

            else:
                mpg_pol = None

                ress = map(_dsg_sgl, dsg_ags)

            ress = list(ress)
            #==================================================================

            for res in ress:

                if res is None: continue

                ott_dfe_tss.update(res)

            ress = None
            #==================================================================

            if mpg_pol is not None:

                if self._vb: print('Terminating multiprocessing pool...')

                mpg_pol.close()
                mpg_pol.join()
                mpg_pol = None
            #==================================================================
        #======================================================================

        self._ott_dfe_tss = ott_dfe_tss
        self._ott_dfe_cds = ott_dfe_cds
        self._dsg_flg = True
        return

    def get_ott(self):

        assert self._dsg_flg

        return self._ott_dfe_tss, self._ott_dfe_cds

    def _chk_vbs(self, dfe_tss, dfe_cds, tfq, vll):

        assert dfe_tss is not None
        assert dfe_cds is not None
        assert tfq is not None

        assert isinstance(dfe_tss, pd.DataFrame)
        assert all(dfe_tss.shape)
        assert dfe_tss.shape[0] >= 2
        assert isinstance(dfe_tss.index, pd.DatetimeIndex)
        assert np.isfinite(dfe_tss.values).any()

        assert isinstance(dfe_cds, pd.DataFrame)
        assert all(dfe_cds.shape)
        assert dfe_cds.shape[1] == 2

        if vll == 'ref':
            assert REF_XCL in dfe_cds.columns
            assert REF_YCL in dfe_cds.columns

        elif vll == 'dst':
            assert DST_XCL in dfe_cds.columns
            assert DST_YCL in dfe_cds.columns

        else:
            raise NotImplementedError(vll)

        assert np.isfinite(dfe_cds.values).all()

        assert isinstance(tfq, pd.Timedelta)
        assert tfq > pd.Timedelta(0)

        cmn_lbs = dfe_tss.columns.intersection(dfe_cds.index)

        assert cmn_lbs.size

        assert cmn_lbs.size == dfe_tss.shape[1]
        assert cmn_lbs.size == dfe_cds.shape[0]

        tds = dfe_tss.index[1:] - dfe_tss.index[:-1]

        assert (tds > pd.Timedelta(0)).all()
        assert (tds == tfq).all()

        ixs = pd.date_range(dfe_tss.index[+0], dfe_tss.index[-1], freq=tfq)

        assert ixs.size

        # dfe_tss = dfe_tss.reindex(ixs)

        assert all(dfe_tss.shape)
        #======================================================================

        return

    def _get_pcr(self, ref_tss, dst_tss):

        dst_tss = dst_tss.resample(
            self._ref_tfq, closed='right', label='right').sum()

        cmn_ixs = dst_tss.index.intersection(ref_tss.index)

        if cmn_ixs.size:

            with np.errstate(divide='ignore', invalid='ignore'):
                dtc = np.corrcoef(
                    dst_tss.loc[cmn_ixs].values,
                    ref_tss.loc[cmn_ixs].values)[0, 1]

            dtc = 1. - dtc

            if np.isnan(dtc): dtc = 3.0

        else:
            dtc = 3.0

        return dtc

    def _get_scr(self, ref_tss, dst_tss):

        dst_tss = dst_tss.resample(
            self._ref_tfq, closed='right', label='right').sum()

        cmn_ixs = dst_tss.index.intersection(ref_tss.index)

        if cmn_ixs.size:

            with np.errstate(divide='ignore', invalid='ignore'):
                dtc = np.corrcoef(
                    dst_tss.loc[cmn_ixs].rank().values,
                    ref_tss.loc[cmn_ixs].rank().values)[0, 1]

            dtc = 1. - dtc

            if np.isnan(dtc): dtc = 3.0

        else:
            dtc = 3.0

        return dtc

    def _get_dtc(self, ref_xcd, ref_ycd, dst_xcd, dst_ycd):

        dtc = 0.0
        dtc += (ref_xcd - dst_xcd) ** 2
        dtc += (ref_ycd - dst_ycd) ** 2
        dtc **= 0.5
        # dtc /= 1000.

        return dtc


def get_dtc_mtx(ref_xcs, ref_ycs, dst_xcs, dst_ycs):

    assert ref_xcs.ndim == 1
    assert ref_ycs.ndim == 1

    assert dst_xcs.ndim == 1
    assert dst_ycs.ndim == 1

    ref_xcs = ref_xcs[:, None]
    ref_ycs = ref_ycs[:, None]

    dst_xcs = dst_xcs[None,:]
    dst_ycs = dst_ycs[None,:]

    dtc_mtx = (ref_xcs - dst_xcs) ** 2.
    dtc_mtx += (ref_ycs - dst_ycs) ** 2.
    dtc_mtx **= 0.5
    return dtc_mtx


def _dsg_sgl(ags):

    try:
        res = _dsg_sgl_org(ags)

    except:
        pre_stack = tb.format_stack()[:-1]

        err_tb = list(tb.TracebackException(*sys.exc_info()).format())

        lines = [err_tb[0]] + pre_stack + err_tb[2:]

        for line in lines: print(line, file=sys.stderr, end='')

        res = None

    return res


def _dsg_sgl_org(ags):

    vb, ott_lbl, dst_tfq, ref_srs_tss, dst_dfe_tss, ref_dst_srs_dts = ags

    if vb: print('Started:', ott_lbl)

    nbs = []
    ref_sms = 0.0
    dst_sms = 0.0
    ref_tke_ixs = []

    ott_srs = pd.Series(index=dst_dfe_tss.index)
    ott_srs.name = ref_srs_tss.name
    for t in range(1, ref_srs_tss.shape[0]):

        if False and vb and (not (t % 10000)): print('Timestep:', t)

        ref_idx_bgn = ref_srs_tss.index[t - 1]
        ref_idx_end = ref_srs_tss.index[t - 0]

        ref_sum = ref_srs_tss.loc[ref_idx_end]
        if np.isnan(ref_sum): continue

        if False:
            print('\t', ref_sum, ref_idx_bgn, ref_idx_end)
        #==================================================================

        dst_ixs_fgs = (
            (dst_dfe_tss.index > ref_idx_bgn) &
            (dst_dfe_tss.index <= ref_idx_end))

        if not dst_ixs_fgs.any(): continue

        dst_ixs = dst_dfe_tss.index[dst_ixs_fgs]

        ref_scs = (ref_idx_end - ref_idx_bgn).total_seconds()

        dst_scs = (
            dst_ixs[-1] - dst_ixs[+0] + dst_tfq).total_seconds()

        assert (dst_scs <= ref_scs) | np.isclose(ref_scs, dst_scs)

        if not np.isclose(ref_scs, dst_scs): continue
        #==================================================================

        ref_tke_ixs.append(ref_idx_end)

        # If sum of ref is zero then till here.
        if np.isclose(ref_sum, 0.0):
            ott_srs.loc[dst_ixs] = 0.0
            continue
        #==================================================================

        dst_dfe_tss_tsp = dst_dfe_tss.loc[dst_ixs_fgs].copy()
        dst_dfe_tss_tsp.dropna(axis=1, how='any', inplace=True)

        vld_ixs = (dst_dfe_tss_tsp.sum(axis=0).values > 0)

        if vld_ixs.any(): dst_dfe_tss_tsp = dst_dfe_tss_tsp.loc[:, vld_ixs]

        if not dst_dfe_tss_tsp.size: continue

        ref_dst_srs_dts_tsp = ref_dst_srs_dts.loc[
            dst_dfe_tss_tsp.columns].copy()
        #==================================================================

        # Look for a nebor with a value larger than 0.
        nst_flg = False
        for nst_idx in np.argsort(ref_dst_srs_dts_tsp.values):

            nst_lbl = ref_dst_srs_dts_tsp.index[nst_idx]

            dst_tss = dst_dfe_tss_tsp.loc[:, nst_lbl].values.copy()

            assert np.isfinite(dst_tss).all()

            dst_sum = dst_tss.sum()

            if np.isclose(dst_sum, 0.0): continue

            nst_flg = True
            break

        if nst_flg:

            assert dst_sum > 0

            if nst_lbl not in nbs: nbs.append(nst_lbl)

            dst_sms += dst_sum
            ref_sms += ref_sum

            dst_wts = dst_tss / dst_sum

        else:
            # If no nebor than distribute uniformly.
            dst_wts = dst_tss.copy()
            dst_wts[:] = 1. / dst_wts.size

        dsg_vls = dst_wts * ref_sum

        ott_srs.loc[dst_dfe_tss_tsp.index] = dsg_vls
        #==================================================================
    #======================================================================

    if False:
        print(
            # '\t',
            ref_srs_tss.name,
            round(ott_srs.sum(), 1),
            round(ref_srs_tss.loc[ref_tke_ixs].sum(), 1),
            round(ref_srs_tss.sum(), 1),
            round(ref_sms, 1),
            round(dst_sms, 1),
            set(nbs),
            )

    assert ((ott_srs.sum() <= ref_srs_tss.sum()) or
            np.isclose(ott_srs.sum(), ref_srs_tss.sum()))

    if vb: print('Finished:', ott_lbl)
    return ott_srs
