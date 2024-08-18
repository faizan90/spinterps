# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Aug 10, 2024

6:10:35 PM

Description:

Keywords:

'''

import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from multiprocessing import Pool, Manager

import numpy as np
import pandas as pd
import netCDF4 as nc
from osgeo import ogr
import matplotlib.pyplot as plt
from scipy.stats import rankdata

ogr.UseExceptions()

DEBUG_FLAG = True


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\spinterps\krig_sclg\naab')
    os.chdir(main_dir)

    ncf_pth = Path(r'ppt_1D_spinterps_1km/kriging.nc')

    # When None, then linear scaling is used.
    elb = 'EST_VARS_OK'  # None  #
    tlb = 'time'

    vbs = ['OK', 'EDK', 'IDW_000', 'IDW_001', 'NNB']

    beg_tme, end_tme = pd.to_datetime(
        ['1971-01-01', '2022-12-30'], format='%Y-%m-%d')

    ply_pth = Path(r'naab_ezg_1km.shp')
    ply_fld = 'DN'
    ply_bfr_dst = 10e3

    cds_pth = Path(r'ppt_1D_gkd_dwd_crds.csv')
    tss_pth = Path(r'ppt_1D_tss.pkl')

    nan_vle = None

    non_neg_flg = True  # In case of ppt-type variables.
    men_ccn_flg = True  # It is difficult to keep both mean and variance.

    cmp_lvl = 1

    pol_sze = 8

    fgs_dir = Path('ppt_1D_spinterps_1km/std_ctd_ots18')
    #==========================================================================

    assert len(vbs) == len(set(vbs)), vbs

    fgs_dir.mkdir(exist_ok=True)

    with nc.Dataset(ncf_pth, 'r') as ncf_hdl:

        tme_sps = nc.num2date(
            ncf_hdl[tlb][:].data,
            units=ncf_hdl[tlb].units,
            calendar=ncf_hdl[tlb].calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True)

        tme_sps = pd.DatetimeIndex(tme_sps)

        take_idxs_beg = tme_sps.get_loc(beg_tme)
        take_idxs_end = tme_sps.get_loc(end_tme) + 1

        tme_sps = tme_sps[take_idxs_beg:take_idxs_end]

    tke_nms = get_ctd_cds_nms(
        cds_pth,
        ply_pth,
        ply_fld,
        0.0,
        ply_bfr_dst)

    tss_dfe = pd.read_pickle(tss_pth)

    cmn_nms = tke_nms.intersection(tss_dfe.columns)

    tss_dfe = tss_dfe.loc[tme_sps, cmn_nms]
    #==========================================================================

    mgr = Manager()

    crt_ags = CRTAGS()

    crt_ags.ncf_pth = ncf_pth
    crt_ags.fgs_dir = fgs_dir

    crt_ags.tlb = tlb
    crt_ags.elb = elb

    crt_ags.lck = mgr.Lock()

    crt_ags.nan_vle = nan_vle
    crt_ags.cmp_lvl = cmp_lvl

    crt_ags.tss_dfe = tss_dfe

    crt_ags.beg_tme = beg_tme
    crt_ags.end_tme = end_tme

    crt_ags.non_neg_flg = non_neg_flg
    crt_ags.men_ccn_flg = men_ccn_flg
    #==========================================================================

    mpg_ags = ((vlb, crt_ags) for vlb in vbs)

    pol_sze = min(len(vbs), pol_sze)

    if pol_sze == 1:
        ress = []
        for ags in mpg_ags:
            res = crt_grd_ipn_var(ags)

            res.append(ress)

    else:
        mpg_pol = Pool(pol_sze)

        ress = mpg_pol.imap_unordered(crt_grd_ipn_var, mpg_ags, chunksize=1)
        ress = list(ress)
    #==========================================================================

    sas_dfe = [get_dta_sas_dfe(tss_dfe)]

    for res in ress: sas_dfe.extend(res)

    sas_dfe = pd.concat(sas_dfe, axis=1)

    sas_dfe.to_csv(fgs_dir / 'CTD_stats.csv', sep=';', float_format='%0.6f')

    for vlb in vbs:
        plot_stats_sers(
            fgs_dir / f'CTD_stats__{vlb}.png',
            sas_dfe,
            [vlb, f'{vlb}__CTD'])
    #==========================================================================
    return


def crt_grd_ipn_var(ags):

    vb = True

    vlb, ags = ags

    ncf_pth = ags.ncf_pth
    fgs_dir = ags.fgs_dir

    tlb = ags.tlb
    elb = ags.elb

    lck = ags.lck

    nan_vle = ags.nan_vle
    cmp_lvl = ags.cmp_lvl

    tss_dfe = ags.tss_dfe

    beg_tme = ags.beg_tme
    end_tme = ags.end_tme

    non_neg_flg = ags.non_neg_flg
    men_ccn_flg = ags.men_ccn_flg
    #==========================================================================

    with lck:
        if vb: print(f'Start {vlb}...')

        with nc.Dataset(ncf_pth, 'r') as ncf_hdl:

            tme_sps = nc.num2date(
                ncf_hdl[tlb][:].data,
                units=ncf_hdl[tlb].units,
                calendar=ncf_hdl[tlb].calendar,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True)

            tme_sps = pd.DatetimeIndex(tme_sps)

            take_idxs_beg = tme_sps.get_loc(beg_tme)
            take_idxs_end = tme_sps.get_loc(end_tme) + 1

            tme_sps = tme_sps[take_idxs_beg:take_idxs_end]

            vbe_gds = ncf_hdl[vlb][take_idxs_beg:take_idxs_end].data

            if elb is not None:
                # Statistics look better when variance is used.
                vbe_scg_gds = ncf_hdl[elb][
                    take_idxs_beg:take_idxs_end].data  # ** 0.5

            else:
                vbe_scg_gds = None

    vbe_gds_ctd = np.full(vbe_gds.shape, np.nan, dtype=vbe_gds.dtype)

    if nan_vle is not None:
        vbe_gds[vbe_gds == nan_vle] = np.nan
    #==========================================================================

    std_dns = np.full((tme_sps.size, 4), np.nan)
    for i, tme_stp in enumerate(tme_sps):

        ref_std = tss_dfe.loc[tme_stp].std()
        tst_std = np.nanstd(vbe_gds[i,:,:])
        tst_mean = np.nanmean(vbe_gds[i,:,:])

        if vbe_scg_gds is not None:
            vbe_scg_grd = vbe_scg_gds[i,:,:]

        else:
            vbe_scg_grd = None

        tst_grd_ctd, tst_std_ctd, tst_std_slr = get_slr_mlt(
            tst_mean,
            tst_std,
            ref_std,
            vbe_gds[i,:,:],
            non_neg_flg,
            men_ccn_flg,
            vbe_scg_grd,)

        vbe_gds_ctd[i,:,:] = tst_grd_ctd

        std_dns[i,:] = ref_std, tst_std, tst_std_ctd, tst_std_slr
    #==========================================================================

    std_dns_dfe = pd.DataFrame(
        index=tme_sps,
        dtype=np.float32,
        data=std_dns,
        columns=('dta_std', 'ipn_std', 'ctd_std', 'std_slr'))

    std_dns_dfe.to_csv(
        fgs_dir / f'stds__{vlb}.csv', sep=';', float_format='%0.6f')

    if vb: print(f'Done scaling {vlb}.')
    #==========================================================================

    plt.figure()

    min_lmt = np.nanmin(std_dns[:,:3])
    max_lmt = np.nanmax(std_dns[:,:3])

    min_lmt -= 0.025 * (max_lmt - min_lmt)
    max_lmt += 0.025 * (max_lmt - min_lmt)

    plt.plot(
        [min_lmt, max_lmt], [min_lmt, max_lmt], alpha=0.75, ls='--', c='b')

    plt.scatter(
        std_dns[:, 0],
        std_dns[:, 1],
        c='k',
        alpha=0.8,
        edgecolors='none',
        label='OLD',
        s=5 ** 2)

    plt.scatter(
        std_dns[:, 0],
        std_dns[:, 2],
        c='b',
        alpha=0.8,
        edgecolors='none',
        label='NEW',
        s=3 ** 2)

    plt.grid()
    plt.gca().set_axisbelow(True)
    plt.gca().set_aspect('equal')

    plt.legend(loc='lower right')

    plt.xlabel('REF [-]')
    plt.ylabel('SIM [-]')

    plt.savefig(
        fgs_dir / f'std_ctd_sctr__{vlb}.png', bbox_inches='tight', dpi=150)

    plt.close()
    #==========================================================================

    plt.figure()

    std_srs = std_dns[:, 3].copy()
    std_srs.sort()

    pbs = rankdata(std_srs) / (std_srs.size + 1.0)

    plt.plot(std_srs, pbs, c='k', alpha=0.8)

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.ylim(-0.025, 1 + 0.025)

    plt.xlabel('STD SLR [-]')
    plt.ylabel('Grade [-]')

    plt.savefig(
        fgs_dir / f'std_slr_dst__{vlb}.png', bbox_inches='tight', dpi=150)

    plt.close()
    #==========================================================================

    if vb: print(f'Plotting histogram for {vlb}...')
    plot_histogram(std_srs, 20, 10, fgs_dir / f'std_slr_hst__{vlb}.png')
    #==========================================================================

    if vb: print(f'Computing stats for {vlb}...')
    ncf_vre_org_sas_dfe = get_ary_sas_dfe(vlb, vbe_gds, tme_sps)
    ncf_vre_ctd_sas_dfe = get_ary_sas_dfe(f'{vlb}__CTD', vbe_gds_ctd, tme_sps)
    #==========================================================================

    with lck:
        if vb: print(f'Writing results for {vlb}...')

        with nc.Dataset(ncf_pth, 'r+') as ncf_hdl:

            # Corrected/scaled.
            vbe_ctd_lbl = f'{vlb}__CTD'

            if vbe_ctd_lbl in ncf_hdl.variables:
                vbe_ctd_dst = ncf_hdl[vbe_ctd_lbl]

            else:
                vbe_dst = ncf_hdl[vlb]

                vbe_ctd_dst = ncf_hdl.createVariable(
                    vbe_ctd_lbl,
                    vbe_gds_ctd.dtype,
                    vbe_dst.dimensions,
                    fill_value=False,
                    compression='zlib',
                    complevel=cmp_lvl,
                    chunksizes=(1,
                                vbe_gds_ctd.shape[1],
                                vbe_gds_ctd.shape[2]))

                vbe_dst = None

            vbe_ctd_dst[take_idxs_beg:take_idxs_end] = vbe_gds_ctd

            if False:
                # Difference absolute.
                vbe_ctd_lbl = f'{vlb}__DIF'

                if vbe_ctd_lbl in ncf_hdl.variables:
                    vbe_ctd_dst = ncf_hdl[vbe_ctd_lbl]

                else:
                    vbe_dst = ncf_hdl[vlb]

                    vbe_ctd_dst = ncf_hdl.createVariable(
                        vbe_ctd_lbl,
                        vbe_gds_ctd.dtype,
                        vbe_dst.dimensions,
                        fill_value=False,
                        compression='zlib',
                        complevel=cmp_lvl,
                        chunksizes=(1,
                                    vbe_gds_ctd.shape[1],
                                    vbe_gds_ctd.shape[2]))

                    vbe_dst = None

                vbe_ctd_dst[take_idxs_beg:take_idxs_end] = (
                    vbe_gds - vbe_gds_ctd)

            if False:
                # Difference relative.
                vbe_ctd_lbl = f'{vlb}__RDF'

                if vbe_ctd_lbl in ncf_hdl.variables:
                    vbe_ctd_dst = ncf_hdl[vbe_ctd_lbl]

                else:
                    vbe_dst = ncf_hdl[vlb]

                    vbe_ctd_dst = ncf_hdl.createVariable(
                        vbe_ctd_lbl,
                        vbe_gds_ctd.dtype,
                        vbe_dst.dimensions,
                        fill_value=False,
                        compression='zlib',
                        complevel=cmp_lvl,
                        chunksizes=(1,
                                    vbe_gds_ctd.shape[1],
                                    vbe_gds_ctd.shape[2]))

                    vbe_dst = None

                vbe_ctd_dst[take_idxs_beg:take_idxs_end] = (
                    (vbe_gds - vbe_gds_ctd) / vbe_gds)
    #==========================================================================

    return [ncf_vre_org_sas_dfe, ncf_vre_ctd_sas_dfe]


def get_slr_mlt(
        tst_men,
        tst_std,
        dst_std,
        tst_grd,
        non_neg_flg,
        men_ccn_flg,
        tst_scg_grd):

    '''
    Bisection algorithm.

    Multiplicative type of std correction.
    '''

    if np.isclose(tst_std, 0.0) and np.isclose(dst_std, 0.0):

        slr_fnl = 1.0

        std_fnl = tst_std

        grd_fnl = tst_grd

        return grd_fnl, std_fnl, slr_fnl

    vb = False

    # A scaling grid.
    dtn = np.full_like(tst_grd, np.nan, dtype=np.float32)
    dtn[np.isfinite(tst_grd)] = 1.0

    if tst_scg_grd is not None:

        # Station cells do not get any correction.
        etn_std_lsr_ixs = np.where((tst_grd < tst_men) & (tst_scg_grd > 0))
        etn_std_gtr_ixs = np.where((tst_grd > tst_men) & (tst_scg_grd > 0))

        # This reveals the station locations.
        dtn_lsr = np.zeros_like(tst_scg_grd)
        dtn_lsr[etn_std_lsr_ixs] = tst_scg_grd[etn_std_lsr_ixs]
        dtn_lsr[np.isnan(tst_grd)] = np.nan

        dtn_gtr = np.zeros_like(tst_scg_grd)
        dtn_gtr[etn_std_gtr_ixs] = tst_scg_grd[etn_std_gtr_ixs]
        dtn_gtr[np.isnan(tst_grd)] = np.nan

    else:
        # Station cells do not get any correction.
        etn_std_lsr_ixs = np.where((tst_grd < tst_men))
        etn_std_gtr_ixs = np.where((tst_grd > tst_men))

        # This does not reveal station locations.
        dtn_lsr = np.zeros_like(tst_grd)
        dtn_lsr[etn_std_lsr_ixs] = 1.0
        dtn_lsr[np.isnan(tst_grd)] = np.nan

        dtn_gtr = np.zeros_like(tst_grd)
        dtn_gtr[etn_std_gtr_ixs] = 1.0
        dtn_gtr[np.isnan(tst_grd)] = np.nan

    min_tol = 1e-6

    tol = np.inf

    slr_lft = 0.0

    slr_rts = []
    if not np.isclose(dst_std, 0.0): slr_rts.append(tst_std / dst_std)
    if not np.isclose(tst_std, 0.0): slr_rts.append(dst_std / tst_std)

    slr_rht = max(slr_rts) * 2.0

    lft_std = np.nanstd(get_sld_grd(
        tst_grd, dtn_lsr, dtn_gtr, slr_lft, non_neg_flg, men_ccn_flg, tst_men))

    rht_std = np.nanstd(get_sld_grd(
        tst_grd, dtn_lsr, dtn_gtr, slr_rht, non_neg_flg, men_ccn_flg, tst_men))

    std_mde_pre = np.inf

    fal_flg = False

    itr_ctr = 0
    while tol > min_tol:

        slr_mde = 0.5 * (slr_lft + slr_rht)

        mde_std = np.nanstd(get_sld_grd(
            tst_grd,
            dtn_lsr,
            dtn_gtr,
            slr_mde,
            non_neg_flg,
            men_ccn_flg,
            tst_men))

        if ((lft_std <= dst_std <= mde_std) or
            (lft_std >= dst_std >= mde_std)):

            slr_rht = slr_mde
            rht_std = mde_std

            # slr_lft = slr_mde
            # lft_std = mde_std

        elif ((mde_std <= dst_std <= rht_std) or
              ((mde_std >= dst_std >= rht_std))):

            slr_lft = slr_mde
            lft_std = mde_std

            # slr_rht = slr_mde
            # rht_std = mde_std

        else:
            if vb: print(
                f'Failed! ({lft_std:0.3f}, {mde_std:0.3f}, {rht_std:0.3f}, '
                f'{dst_std:0.3f}, {itr_ctr})')

            fal_flg = True
            break

        tol = abs(std_mde_pre - mde_std)

        if tol < min_tol:
            if vb: print(f'Min. Tol. ({itr_ctr})')

        itr_ctr += 1

        std_mde_pre = mde_std

    if fal_flg:
        slr_fnl = 1.0

        std_fnl = tst_std

        grd_fnl = tst_grd

    else:
        slr_fnl = [slr_lft, slr_mde, slr_rht][np.argmin(np.abs(
            np.array([dst_std - lft_std,
                      dst_std - mde_std,
                      dst_std - rht_std])))]

        tst_grd_fnl = get_sld_grd(
            tst_grd,
            dtn_lsr,
            dtn_gtr,
            slr_fnl,
            non_neg_flg,
            men_ccn_flg,
            tst_men)

        std_fnl = np.nanstd(tst_grd_fnl)

        tol = abs(mde_std - std_fnl)

        grd_fnl = tst_grd_fnl

    return grd_fnl, std_fnl, slr_fnl


def get_sld_grd(grd, lsr, gtr, slr, non_neg_flg, men_ccn_flg, ref_men):

    sld_grd = np.zeros_like(grd)

    sld_grd += lsr * max(0, (1 - slr))
    sld_grd += gtr * (0 + slr)

    sld_grd *= grd

    sld_grd = (sld_grd * max(0, (1 - slr))) + (slr * grd)

    if True and (not np.isclose(slr, 0)):
        sld_grd /= abs(1 - slr)  # Is this right?

    # Non-negative values.
    if non_neg_flg: sld_grd[sld_grd < 0] = 0

    # Correction for the mean.
    if men_ccn_flg:
        grd_men = np.nanmean(sld_grd)

        if not np.isclose(grd_men, 0.0):
            sld_grd *= ref_men / grd_men

    return sld_grd


def plot_histogram(vls, bns_cnt, dtn_cnt, fgr_pth):

    vls = vls.copy()
    vls.sort()

    vls = vls[np.isfinite(vls)]

    assert vls.size

    min_vle = vls.min() - (0.025 * (vls.max() - vls.min()))
    max_vle = vls.max() + (0.025 * (vls.max() - vls.min()))

    bns_sgl = np.linspace(min_vle, max_vle, bns_cnt, endpoint=False)

    dtn_wdh = (bns_sgl[1] - bns_sgl[0]) / dtn_cnt

    hts = []
    bns = []
    for i in range(dtn_cnt):

        bns_dtn = (bns_sgl + dtn_wdh * i)

        htm = np.histogram(
            vls, bins=bns_dtn, density=False)[0].astype(np.float64)

        htm /= htm.sum()

        hts.append(htm)
        bns.append(bns_dtn)
    #==========================================================================

    plt.figure()

    for i in range(dtn_cnt):
        plt.bar(
            bns[i][:-1] + ((bns[i][1:] - bns[i][:-1]) * 0.5),
            hts[i],
            bns[i][1:] - bns[i][:-1],
            alpha=0.3,
            color='k')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('STD SLR [-]')
    plt.ylabel('Density [-]')

    plt.savefig(fgr_pth, bbox_inches='tight', dpi=150)

    plt.close()
    return


def plot_stats_sers(plt_pth, stats_df, interp_labels):

    stats = ['min', 'mean', 'max', 'std', 'count']

    plt.figure(figsize=(8.0, 6.5))

    for stat in stats:

        max_val = -np.inf

        obs_vls = stats_df[f'data_{stat}'].values

        # Taking mean value wherever not finite.
        obs_nft_ixs = ~np.isfinite(obs_vls)

        obs_avg = obs_vls[~obs_nft_ixs].mean()

        if obs_nft_ixs.sum():
            obs_vls[obs_nft_ixs] = obs_avg

        obs_cum_sum = obs_vls.cumsum()
        ref_cum_sum = np.repeat(obs_avg, obs_vls.size).cumsum()

        if stat == 'count':
            obs_cum_sum /= obs_cum_sum[-1]
            ref_cum_sum /= ref_cum_sum[-1]

        max_val = max(max_val, obs_cum_sum[-1])
        max_val = max(max_val, ref_cum_sum[-1])

        plt.plot(ref_cum_sum, obs_cum_sum, label='DATA')

        for ilab in interp_labels:

            sim_vls = stats_df[f'{ilab}_{stat}'].values

            sim_nft_ixs = ~np.isfinite(sim_vls)
            if sim_nft_ixs.sum():
                sim_vls[sim_nft_ixs] = sim_vls[~sim_nft_ixs].mean()

            sim_cum_sum = sim_vls.cumsum()

            if stat == 'count':
                sim_cum_sum /= sim_cum_sum[-1]

            max_val = max(max_val, sim_cum_sum[-1])

            plt.plot(ref_cum_sum, sim_cum_sum, label=ilab)

        plt.plot(
            [0, max_val],
            [0, max_val],
            ls='--',
            c='k',
            lw=2.0,
            alpha=0.75)

        plt.xlabel('REF')
        plt.ylabel('SIM')

        plt.grid()
        plt.gca().set_axisbelow(True)
        plt.gca().set_aspect('equal')

        plt.legend(loc='upper left')

        plt.savefig(
            plt_pth.with_name(f'{plt_pth.stem}__{stat}{plt_pth.suffix}'),
            bbox_inches='tight')

        plt.clf()

    plt.close()
    return


def get_ary_sas_dfe(vlb, ary, tme_sps):

    # Count is handeld separately.
    sas = ['min', 'mean', 'max', 'std', 'count']

    sas_dfe = pd.DataFrame(
        index=tme_sps,
        columns=[f'{vlb}_{stt}' for stt in sas],
        dtype=np.float32)

    for i in range(ary.shape[0]):

        ipn_fld = ary[i,:,:]

        for stt in sas:

            if stt == 'count':
                stt_vle = np.isfinite(ipn_fld).sum(dtype=np.uint64)

            else:
                stt_vle = getattr(np, f'nan{stt}')(ipn_fld)

            sas_dfe.loc[tme_sps[i], f'{vlb}_{stt}'] = np.float32(stt_vle)

    return sas_dfe


def get_dta_sas_dfe(dta_dfe):

    dta_dfe = dta_dfe.sort_index()

    tme_sps = dta_dfe.index

    # Count is handeld separately.
    sas = ['min', 'mean', 'max', 'std', 'count']

    sas_dfe = pd.DataFrame(
        index=tme_sps,
        columns=[f'data_{stt}' for stt in sas],
        dtype=np.float32)

    for stt in sas:
        sas_dfe.loc[dta_dfe.index, f'data_{stt}'] = getattr(
            dta_dfe, stt)(axis=1).astype(np.float32)

    return sas_dfe

# def get_ncf_vre_sas_dfe(vlb, tlb, ncf_pth, beg_tme, end_tme):
#
#     ncf_hdl = nc.Dataset(ncf_pth, mode='r')
#
#     tme_sps = nc.num2date(
#         ncf_hdl[tlb][:].data,
#         units=ncf_hdl[tlb].units,
#         calendar=ncf_hdl[tlb].calendar,
#         only_use_cftime_datetimes=False,
#         only_use_python_datetimes=True)
#
#     tme_sps = pd.DatetimeIndex(tme_sps)
#
#     take_idxs_beg = tme_sps.get_loc(beg_tme)
#     take_idxs_end = tme_sps.get_loc(end_tme) + 1
#     #==========================================================================
#
#     # Count is handeld separately.
#     sas = ['min', 'mean', 'max', 'std', 'count']
#
#     sas_dfe = pd.DataFrame(
#         index=tme_sps,
#         columns=[f'{vlb}_{stt}' for stt in sas],
#         dtype=np.float32)
#
#     for i in range(take_idxs_beg, take_idxs_end):
#
#         ipn_fld = ncf_hdl[vlb][i].data
#
#         for stt in sas:
#
#             if stt == 'count':
#                 stt_vle = np.isfinite(ipn_fld).sum(dtype=np.uint64)
#
#             else:
#                 stt_vle = getattr(np, f'nan{stt}')(ipn_fld)
#
#             sas_dfe.loc[tme_sps[i], f'{vlb}_{stt}'] = np.float32(stt_vle)
#
#     ncf_hdl.close()
#     return sas_dfe


def get_ctd_cds_nms(
        in_crds_file,
        subset_shp_file,
        subset_shp_fld,
        simplyify_tol,
        shp_buff_dist):

    in_crds_df = pd.read_csv(
        in_crds_file,
        sep=';',
        index_col=0,
        engine='python',
        encoding='latin1')

    cat_poly = get_merged_poly(subset_shp_file, subset_shp_fld, simplyify_tol)
    cat_buff = cat_poly.Buffer(shp_buff_dist)
    assert cat_buff

    in_crds_x_fnt = np.isfinite(in_crds_df['X'].values)
    in_crds_y_fnt = np.isfinite(in_crds_df['Y'].values)

    in_crds_fnt = in_crds_x_fnt & in_crds_y_fnt

    in_crds_df = in_crds_df.loc[in_crds_fnt].copy()

    # Remove duplicate, the user can also implement proper selection
    # because a change station location means a new time series normally.
    keep_crds_stns_steps = ~in_crds_df.index.duplicated(keep='last')
    in_crds_df = in_crds_df.loc[keep_crds_stns_steps]

    in_crds_df.sort_index(inplace=True)

    contain_crds = get_stns_in_poly(in_crds_df, cat_buff)

    subset_crds_stns = in_crds_df.index.intersection(contain_crds)

    if False: print(subset_crds_stns.size, 'stations selected in crds_df!')

    return subset_crds_stns


def get_stns_in_poly(crds_df, poly):

    contain_stns = []

    for i, stn in enumerate(crds_df.index):

        x, y = crds_df.iloc[i].loc[['X', 'Y']]

        pt = ogr.CreateGeometryFromWkt("POINT (%f %f)" % (float(x), float(y)))

        if pt is None:
            print(f'Station {stn} returned a Null point!')
            continue

        if poly.Contains(pt):
            contain_stns.append(stn)

    return contain_stns


def get_merged_poly(in_shp, field='DN', simplify_tol=0):

    '''Merge all polygons with the same ID in the 'field' (from TauDEM)

    Because sometimes there are some polygons from the same catchment,
    this is problem because there can only one cathcment with one ID,
    it is an artifact of gdal_polygonize.
    '''

    cat_ds = ogr.Open(str(in_shp))
    lyr = cat_ds.GetLayer(0)

    feat_dict = {}
    fid_to_field_dict = {}

    feat = lyr.GetNextFeature()
    while feat:
        fid = feat.GetFID()
        f_val = feat.GetFieldAsString(field)
        feat_dict[fid] = feat.Clone()
        fid_to_field_dict[fid] = f_val
        feat = lyr.GetNextFeature()

    fid_list = []
    for fid in list(fid_to_field_dict.keys()):
        fid_list.append(fid)

    if len(fid_list) > 1:
        cat_feat = feat_dict[fid_list[0]]
        merged_cat = cat_feat.GetGeometryRef().Buffer(0)
        for fid in fid_list[1:]:
            # The buffer with zero seems to fix invalid geoms somehow.
            curr_cat_feat = feat_dict[fid].Clone()
            curr_cat = curr_cat_feat.GetGeometryRef().Buffer(0)

            merged_cat = merged_cat.Union(curr_cat)
    else:
        cat = feat_dict[fid_list[0]].Clone()
        merged_cat = cat.GetGeometryRef().Buffer(0)

    if simplify_tol:
        merged_cat = merged_cat.Simplify(simplify_tol)

    cat_ds.Destroy()
    return merged_cat


class CRTAGS:

    def __init__(self):
        return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
