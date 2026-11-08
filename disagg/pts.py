# -*- coding: utf-8 -*-

'''
Created on 08.07.2026

@author: Faizan-TU Munich
'''

import sys
import traceback as tb
from timeit import default_timer as dt
from multiprocessing import Pool as MPPool

import numpy as np
import pandas as pd

from ..mpg import wait_for_pool_init, SHMARY2, fre_shm_ars

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

        self._mpg_ncs = pol_sze
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

        self._dtc_max = 50e3

        dtc_mtx = get_dtc_mtx(
            self._ref_dfe_cds[REF_XCL].values,
            self._ref_dfe_cds[REF_YCL].values,
            self._dst_dfe_cds[DST_XCL].values,
            self._dst_dfe_cds[DST_YCL].values)

        # dtc_mtx[dtc_mtx > self._dtc_max] = np.nan

        #  Only distance for now.
        dts_dfs['dtc'][:] = dtc_mtx

        # dtc_mtx = None
        #======================================================================

        if False:

            dtc_dtn = []
            for ref_lbl in ref_dst_dfe_dts.index:

                (ref_xcd,
                 ref_ycd) = self._ref_dfe_cds.loc[ref_lbl, [REF_XCL, REF_YCL]]

                if ('pcr' in slt_sts) or ('scr' in slt_sts):
                    ref_tss = self._ref_dfe_tss.loc[:, ref_lbl].copy()

                for dst_lbl in ref_dst_dfe_dts.columns:

                    if 'dtc' in slt_sts:

                        (dst_xcd,
                         dst_ycd) = self._dst_dfe_cds.loc[
                             dst_lbl, [DST_XCL, DST_YCL]]

                        dtc_dtc = self._get_dtc(
                            ref_xcd, ref_ycd, dst_xcd, dst_ycd)

                        dts_dfs['dtc'].loc[ref_lbl, dst_lbl] = dtc_dtc

                    if 'pcr' in slt_sts:

                        dst_tss = self._dst_dfe_tss.loc[:, dst_lbl].copy()

                        dtc_pcr = self._get_pcr(ref_tss, dst_tss)

                        dts_dfs['pcr'].loc[ref_lbl, dst_lbl] = dtc_pcr

                    if 'scr' in slt_sts:

                        dst_tss = self._dst_dfe_tss.loc[:, dst_lbl].copy()

                        dtc_scr = self._get_scr(ref_tss, dst_tss)

                        dts_dfs['scr'].loc[ref_lbl, dst_lbl] = dtc_scr

                if False:
                    print(ref_lbl)
                    # print(1. - dts_dfs['scr'].loc[ref_lbl,:].min())
                    print(dts_dfs['dtc'].loc[ref_lbl,:].min())
                    print('\n')

                # dtc_dtn.append(1. - dts_dfs['scr'].loc[ref_lbl,:].min())

            if False:
                dtc_dtn = np.array(dtc_dtn)
                dtc_dtn.sort()
                dtn_pbs = np.arange(
                    1., dtc_dtn.size + 1.0) / (1.0 + dtc_dtn.size)

                import matplotlib.pyplot as plt
                plt.plot(dtc_dtn, dtn_pbs)
                plt.grid()
                plt.gca().set_axisbelow(True)
                plt.xlim(0.5, 1)
                plt.show()
        #======================================================================

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

        ref_tss = SHMARY2.frm_npy_ary(self._ref_dfe_tss.values)
        ref_ixs = self._ref_dfe_tss.index
        ref_cls = self._ref_dfe_tss.columns

        dst_tss = SHMARY2.frm_npy_ary(self._dst_dfe_tss.values)
        dst_ixs = self._dst_dfe_tss.index
        dst_cls = self._dst_dfe_tss.columns

        ref_dst_dts = SHMARY2.frm_npy_ary(ref_dst_dfe_dts.values)
        #======================================================================

        if self._vb: print('Disaggregating....')

        dsg_ags = []
        for ott_lbl in ott_dfe_tss.columns:

            if ott_lbl in self._dst_dfe_tss.columns: continue

            ref_lbl_srs_tss = self._ref_dfe_tss.loc[:, ott_lbl]

            assert ref_lbl_srs_tss.min() >= 0.0

            dsg_ags_sgl = DSGAGS()
            dsg_ags_sgl.vb = self._vb
            dsg_ags_sgl.ott_lbl = ott_lbl
            dsg_ags_sgl.dst_tfq = self._dst_tfq

            dsg_ags_sgl.ref_tss = ref_tss
            dsg_ags_sgl.ref_ixs = ref_ixs
            dsg_ags_sgl.ref_cls = ref_cls

            dsg_ags_sgl.dst_tss = dst_tss
            dsg_ags_sgl.dst_ixs = dst_ixs
            dsg_ags_sgl.dst_cls = dst_cls

            dsg_ags_sgl.ref_dst_dts = ref_dst_dts

            dsg_ags.append(dsg_ags_sgl)
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

            #==================================================================
            for dsg_ags_sgl in dsg_ags: fre_shm_ars(dsg_ags_sgl)
            dsg_ags = None
            #==================================================================

            if self._vb: print('Updating output DataFrame...')

            for res in ress:

                if res is None: continue

                ott_lbl, ott_tss = res

                ott_dfe_tss[ott_lbl] = ott_tss

            ress = None
            #==================================================================

            if mpg_pol is not None:

                if self._vb: print('Terminating multiprocessing pool...')

                mpg_pol.close()
                mpg_pol.join()
                mpg_pol = None
            #==================================================================
        #======================================================================

        if self._vb: print('Writing to disk...')

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

    tmr_bgn = dt()

    vb = ags.vb
    ott_lbl = ags.ott_lbl
    dst_tfq = ags.dst_tfq

    ref_tss_ful = ags.ref_tss
    ref_ixs_ful = ags.ref_ixs
    ref_cls_ful = ags.ref_cls

    dst_tss_ful = ags.dst_tss
    dst_ixs_ful = ags.dst_ixs
    dst_cls_ful = ags.dst_cls

    ref_dst_dts = ags.ref_dst_dts

    ref_srs_tss = pd.Series(
        index=ref_ixs_ful,
        data=ref_tss_ful[:, ref_cls_ful.get_loc(ott_lbl)],
        name=ott_lbl)

    ref_dst_srs_dts = pd.Series(
        index=dst_cls_ful,
        data=ref_dst_dts[ref_cls_ful.get_loc(ott_lbl)],
        name=ott_lbl)

    # dtc_thd = 20e3
    #
    # cos_nbr_ixs = ref_dst_srs_dts.values <= dtc_thd
    #
    # dst_tss_ful = dst_tss_ful[:, cos_nbr_ixs]
    # dst_cls_ful = dst_cls_ful[cos_nbr_ixs]

    if vb: print('Started:', ott_lbl)

    nbs = []
    ref_sms = 0.0
    dst_sms = 0.0
    ref_tke_ixs = []

    tps_ctr = 0

    ott_srs = pd.Series(index=dst_ixs_ful, dtype=np.float32)
    ott_srs.name = ott_lbl

    ott_dne_fgs = np.zeros(ref_ixs_ful.size, dtype=bool)

    # Hopefully, time res is smaller than a month.
    lop_vbs_ful = get_lop_vbs_mhs(ref_ixs_ful)

    for lop_vbs in lop_vbs_ful:

        lop_lbl, lop_idx_min, lop_idx_max = lop_vbs

        dst_tke_fgs = (
            (dst_ixs_ful >= ref_ixs_ful[lop_idx_min]) &
            (dst_ixs_ful <= ref_ixs_ful[lop_idx_max]))

        if not dst_tke_fgs.any(): continue

        dst_tke_tss = dst_tss_ful[dst_tke_fgs,:]
        dst_tke_ixs = dst_ixs_ful[dst_tke_fgs]

        if False and vb: print(lop_lbl)
        #======================================================================

        for t in range(lop_idx_min, lop_idx_max + 1, 1):

            if ott_dne_fgs[t]: continue

            if False and vb and (not (t % 10000)): print('Timestep:', t)

            # Select ref.
            ref_idx_bgn = ref_srs_tss.index[t - 1]
            ref_idx_end = ref_srs_tss.index[t - 0]

            ref_sum = ref_srs_tss.loc[ref_idx_end]
            if np.isnan(ref_sum):
                ott_dne_fgs[t] = True
                continue

            assert ref_sum >= 0.0, (
                t, ref_idx_end, ref_sum)

            if False:
                print('\t', ref_sum, ref_idx_bgn, ref_idx_end)
            #==================================================================

            # Select dst ixs.
            ref_scs = (ref_idx_end - ref_idx_bgn).total_seconds()

            dst_ixs_fgs = (
                (dst_tke_ixs > ref_idx_bgn) &
                (dst_tke_ixs <= ref_idx_end))

            if not dst_ixs_fgs.any():
                ott_dne_fgs[t] = True
                continue

            dst_ixs = dst_tke_ixs[dst_ixs_fgs]

            dst_scs = (
                dst_ixs[-1] - dst_ixs[+0] + dst_tfq).total_seconds()

            assert (dst_scs <= ref_scs) | np.isclose(ref_scs, dst_scs)

            if not np.isclose(ref_scs, dst_scs):
                ott_dne_fgs[t] = True
                continue
            #==================================================================

            ref_tke_ixs.append(ref_idx_end)

            # If sum of ref is zero then till here.
            if np.isclose(ref_sum, 0.0):
                ott_srs.loc[dst_ixs] = 0.0
                ott_dne_fgs[t] = True
                continue
            #==================================================================

            # Select dst columns.
            dst_dfe_tss_tsp = pd.DataFrame(
                index=dst_ixs,
                columns=dst_cls_ful,
                data=dst_tke_tss[dst_ixs_fgs])

            dst_dfe_tss_tsp = dst_dfe_tss_tsp.dropna(axis=1, how='any')

            vld_ixs = (dst_dfe_tss_tsp.sum(axis=0).values > 0)

            if vld_ixs.any(): dst_dfe_tss_tsp = dst_dfe_tss_tsp.loc[:, vld_ixs]

            ref_dst_srs_dts_tsp = ref_dst_srs_dts.loc[
                dst_dfe_tss_tsp.columns].copy()
            #==================================================================

            # Look for a nebor in dst with sum larger than 0.
            nst_flg = False

            if vld_ixs.any():

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
                dst_wts = np.full(dst_ixs.size, 1. / dst_ixs.size)

            dsg_vls = dst_wts * ref_sum

            ott_srs.loc[dst_dfe_tss_tsp.index] = dsg_vls.astype(np.float32)

            tps_ctr += 1
            ott_dne_fgs[t] = True
            #==================================================================
    #==========================================================================

    ott_sum = np.nansum(ott_srs)
    ref_sum = np.nansum(ref_srs_tss)

    assert ott_sum >= 0, (
        'ott_sum:', ott_sum)

    assert ref_sum >= 0, (
        'ref_sum:', ref_sum)

    assert ((ott_sum <= ref_sum) or np.isclose(ott_sum, ref_sum)), (
        ott_sum, ref_sum)

    tmr_end = dt()

    if vb:
        print(
            'Finished:',
            ott_lbl,
            ott_srs.shape[0],
            tps_ctr,
            round(ott_sum, 2),
            round(ref_sum, 2),
            f'{tmr_end - tmr_bgn:0.2f} secs.')

    ott_tss = np.array(ott_srs.values)
    return ott_lbl, ott_tss


def get_lop_vbs_yer(ref_ixs_ful):

    yrs = np.unique(ref_ixs_ful.year)

    vbs = []
    for yer in yrs:

        ref_yer_fgs = ref_ixs_ful.year == yer

        if not ref_yer_fgs.any(): continue

        lop_tke_ixs = np.where(ref_yer_fgs)[0]

        lop_idx_min = max(0, lop_tke_ixs[+0] - 1)
        lop_idx_max = min(ref_ixs_ful.size - 1, lop_tke_ixs[-1] + 1)

        vbs.append(((yer,), lop_idx_min, lop_idx_max))

    assert len(vbs)
    return vbs


def get_lop_vbs_mhs(ref_ixs_ful):

    yrs = np.unique(ref_ixs_ful.year)
    mhs = np.unique(ref_ixs_ful.month)

    vbs = []
    for yer in yrs:

        ref_yer_fgs = ref_ixs_ful.year == yer

        for mth in mhs:

            ref_mth_fgs = ref_yer_fgs & (ref_ixs_ful.month == mth)

            if not ref_mth_fgs.any(): continue

            lop_tke_ixs = np.where(ref_mth_fgs)[0]

            lop_idx_min = max(0, lop_tke_ixs[+0] - 1)
            lop_idx_max = min(ref_ixs_ful.size - 1, lop_tke_ixs[-1] + 1)

            vbs.append(((yer, mth), lop_idx_min, lop_idx_max))

    assert len(vbs)
    return vbs


class DSGAGS:

    def __init__(self):

        return
