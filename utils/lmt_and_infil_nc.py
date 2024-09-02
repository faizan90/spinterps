# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jul 2, 2024

11:23:21 AM

Description:

Keywords:

'''

import os
import sys
import time
import timeit
from math import ceil
import traceback as tb
from pathlib import Path
from shutil import copy2
from multiprocessing import Pool, Lock, Manager

import numpy as np
import psutil as ps
import pandas as pd
import netCDF4 as nc
from osgeo import ogr

ogr.UseExceptions()

DEBUG_FLAG = True


def main():

    main_dir = Path(r'P:\dwd_meteo\gridded\extract_hyras')
    os.chdir(main_dir)

    if True:
        ref_tss_pth = Path(
            r'P:\DEBY\observed_point_time_series\dwd_gkd_merged_crctd_subset\daily\ppt_1D_gkd_dwd_tss.pkl')

        ref_crd_pth = Path(
            r'P:\DEBY\observed_point_time_series\dwd_gkd_merged_crctd_subset\daily\ppt_1D_gkd_dwd_crds.csv')

        ifl_pth = Path(r'a_snip/pr_hyras_1_1931_2020_v5-0_de.nc')
        ifl_vlb = 'pr'
        ifl_xlb = 'x_utm32n'
        ifl_ylb = 'y_utm32n'
        ifl_tlb = 'time'

        # Beyond limits are set to NaN.
        var_llm = 0.00
        var_ulm = 500.  # Actual max. of 186.

    elif False:
        ref_tss_pth = Path(
            r'P:\DEBY\observed_point_time_series\dwd_gkd_merged_crctd_subset\daily\tem_tg_1D_gkd_dwd_tss.pkl')

        ref_crd_pth = Path(
            r'P:\DEBY\observed_point_time_series\dwd_gkd_merged_crctd_subset\daily\tem_tg_1D_gkd_dwd_crds.csv')

        ifl_pth = Path(r'a_snip/tas_hyras_5_1951_2020_v5-0_de.nc')
        ifl_vlb = 'tas'
        ifl_xlb = 'x_utm32n'
        ifl_ylb = 'y_utm32n'
        ifl_tlb = 'time'

        # Beyond limits are set to NaN.
        var_llm = -40.  # Actual min of -27.5.
        var_ulm = +60.  # Actual max of -30.5.

    elif False:
        ref_tss_pth = Path(
            r'P:\DEBY\observed_point_time_series\dwd_gkd_merged_crctd_subset\daily\tem_tx_1D_gkd_dwd_tss.pkl')

        ref_crd_pth = Path(
            r'P:\DEBY\observed_point_time_series\dwd_gkd_merged_crctd_subset\daily\tem_tx_1D_gkd_dwd_crds.csv')

        ifl_pth = Path(r'a_snip/tasmax_hyras_5_1951_2020_v5-0_de.nc')
        ifl_vlb = 'tasmax'
        ifl_xlb = 'x_utm32n'
        ifl_ylb = 'y_utm32n'
        ifl_tlb = 'time'

        # Beyond limits are set to NaN.
        var_llm = -40.  # Actual min -20.
        var_ulm = +70.  # Acual max. 40.

    elif False:
        ref_tss_pth = Path(
            r'P:\DEBY\observed_point_time_series\dwd_gkd_merged_crctd_subset\daily\tem_tn_1D_gkd_dwd_tss.pkl')

        ref_crd_pth = Path(
            r'P:\DEBY\observed_point_time_series\dwd_gkd_merged_crctd_subset\daily\tem_tn_1D_gkd_dwd_crds.csv')

        ifl_pth = Path(r'a_snip/tasmin_hyras_5_1951_2020_v5-0_de.nc')
        ifl_vlb = 'tasmin'
        ifl_xlb = 'x_utm32n'
        ifl_ylb = 'y_utm32n'
        ifl_tlb = 'time'

        # Beyond limits are set to NaN.
        var_llm = -40.  # Actual min of -33.8.
        var_ulm = +50.  # Actual max of 23.4.

    else:
        raise ValueError
    #==========================================================================

    mpg_pol_sze = 16

    ifl_bfr = 30e3  # Nebors from ref selected in this buffer.

    # Points outside poly buffer set to NaN.
    ply_fld = 'DN'
    ply_bfr_dst = 3e3  # Should be at least cell size, as centroids are tested.
    ply_pth = Path(
        r'P:\TUM\projects\altoetting\hydmod\dem_ansys_r4_1km6\watersheds.shp')

    inp_cnk_sze = 2 * (1024 ** 3)  # In bytes and per thread!

    ovr_wie_flg = False

    ott_dir = Path(rf'd_lmtd_and_infld')  # Ovrwrt if same.
    #==========================================================================

    mpg_ags = [(
        ifl_pth,
        ifl_vlb,
        ifl_xlb,
        ifl_ylb,
        ifl_tlb,
        ifl_bfr,
        ott_dir,
        var_llm,
        var_ulm,
        ply_pth,
        ply_fld,
        ovr_wie_flg,
        ply_bfr_dst,
        ref_tss_pth,
        ref_crd_pth,
        inp_cnk_sze,
        mpg_pol_sze,
        )]

    for ags in mpg_ags: lmt_ifl_ncf(ags)

    return


def lmt_ifl_ncf(ags):

    (ifl_pth,
     ifl_vlb,
     ifl_xlb,
     ifl_ylb,
     ifl_tlb,
     ifl_bfr,
     ott_dir,
     var_llm,
     var_ulm,
     ply_pth,
     ply_fld,
     ovr_wie_flg,
     ply_bfr_dst,
     ref_tss_pth,
     ref_crd_pth,
     inp_cnk_sze,
     mpg_pol_sze,
    ) = ags

    # NOTE: All reading done from input file and all writing
    #       done to output file.

    try: ott_dir.mkdir(exist_ok=True, parents=True)
    except: pass

    otp_pth = ott_dir / ifl_pth.name
    #==========================================================================

    print(otp_pth)
    print('')

    vb = False
    #==========================================================================

    assert ifl_pth != otp_pth

    if (not otp_pth.exists()) or ovr_wie_flg:
        if vb: print('')
        if vb: print('Copying...')

        copy2(ifl_pth, otp_pth)
    #==========================================================================

    with nc.Dataset(ifl_pth, 'r') as ncf_hdl:

        inp_dte = ncf_hdl[ifl_vlb].dtype

        if vb: print('')
        if vb: print('Reading coordinates...')

        xcs = ncf_hdl[ifl_xlb][:].data
        ycs = ncf_hdl[ifl_ylb][:].data

        xcs, ycs = get_ctd_cds(xcs, ycs, ncf_hdl[ifl_vlb].shape)
        #======================================================================

        if vb: print('Reading time...')
        tcs_dts = ncf_hdl[ifl_tlb]

        tcs_vls = tcs_dts[:].data

        tcs_tim = pd.DatetimeIndex(nc.num2date(
            tcs_vls,
            tcs_dts.units,
            tcs_dts.calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True))
    #==========================================================================

    if vb: print('')
    if vb: print('Reading TSS and CRD files...')

    ref_tss_dfe, ref_crd_dfe = get_tss_ncf_cds_cmn_dfs(
        ref_tss_pth,
        ref_crd_pth,
        ply_pth,
        ply_fld,
        ifl_bfr,
        tcs_tim)

    tcs_tim = ref_tss_dfe.index

    if vb: print('REF TSS Shape:', ref_tss_dfe.shape)
    if vb: print('REF CRD Shape:', ref_crd_dfe.shape)
    #==========================================================================

    if vb: print('')
    if vb: print('Flagging cells that are outside polygon(s) range...')

    bgn_tmr = timeit.default_timer()

    if mpg_pol_sze != 1:
        cds_fgs_ixs = np.linspace(
            0,
            xcs.shape[0],
            (mpg_pol_sze * 2) + 1,
            endpoint=True,
            dtype=np.int64)

        cds_fgs_ixs = np.unique(cds_fgs_ixs)

        assert cds_fgs_ixs[+0] == 0, cds_fgs_ixs
        assert cds_fgs_ixs[-1] == xcs.shape[0], cds_fgs_ixs

        assert np.unique(cds_fgs_ixs).size == cds_fgs_ixs.size, (
            np.unique(cds_fgs_ixs).size, cds_fgs_ixs.size)

    else:
        cds_fgs_ixs = np.array([0, xcs.shape[0]])

    mpg_ags = ((
        i,
        xcs[ri:rj,:],
        ycs[ri:rj,:],
        ply_pth,
        ply_fld,
        ply_bfr_dst)
        for i, (ri, rj) in enumerate(zip(cds_fgs_ixs[:-1], cds_fgs_ixs[1:])))

    if (mpg_pol_sze == 1) or (cds_fgs_ixs.size == 2):

        mpg_pol = None

        for ags in mpg_ags:
            not_ifl_fgs_ary = get_fgs_cls_sgl(ags)[1]
            break

    else:
        mpg_pol = Pool(mpg_pol_sze)

        mpg_res = mpg_pol.imap_unordered(get_fgs_cls_sgl, mpg_ags, chunksize=1)
        mpg_res = list(mpg_res)

        ixs_rdm = [res[0] for res in mpg_res]

        not_ifl_fgs_ary = np.concatenate(
            [mpg_res[idx][1] for idx in np.argsort(ixs_rdm)], axis=0)

    assert not_ifl_fgs_ary.shape == xcs.shape

    end_tmr = timeit.default_timer()

    if vb: print(
        f'Flagging cells for containment took {end_tmr - bgn_tmr:0.2f} secs.')

    assert not_ifl_fgs_ary.any()

    if vb: print(
        f'Out of {np.prod(not_ifl_fgs_ary.shape, dtype=np.uint64)} cells, '
        f'{not_ifl_fgs_ary.sum(dtype=np.uint64)} are out of the polygon(s) '
        f'range while {(~not_ifl_fgs_ary).sum(dtype=np.uint64)} are to be '
        f'infilled.')
    #==========================================================================

    # Memory management.
    inp_vle_cts = tcs_tim.size * np.prod(xcs.shape, dtype=np.uint64)

    tot_mmy = min(inp_cnk_sze, int(ps.virtual_memory().available * 0.5))

    mmy_cns_cnt = ceil((inp_vle_cts / tot_mmy) * inp_dte.itemsize)

    assert mmy_cns_cnt >= 1, mmy_cns_cnt

    if mmy_cns_cnt > 1:
        if vb: print('Memory not enough to read in one go!')

        mmy_ixs = np.linspace(
            0,
            tcs_tim.size,
            (mmy_cns_cnt + 1),
            endpoint=True,
            dtype=np.int64)

        assert mmy_ixs[+0] == 0, mmy_ixs
        assert mmy_ixs[-1] == tcs_tim.size, mmy_ixs

        assert np.unique(mmy_ixs).size == mmy_ixs.size, (
            np.unique(mmy_ixs).size, mmy_ixs.size)

    else:
        mmy_ixs = np.array([0, tcs_tim.size])

    mmy_ixs_sze = mmy_ixs.size

    if vb: print(f'Total chunks to read: {mmy_ixs_sze - 1}')
    #==========================================================================

    if (mpg_pol_sze == 1) or (mmy_ixs_sze == 2):

        lck = Lock()

    else:
        mgr = Manager()
        lck = mgr.Lock()
    #==========================================================================

    mpg_ags = [(
        xcs,
        ycs,
        lck,
        var_llm,
        var_ulm,
        ifl_pth,
        otp_pth,
        ifl_tlb,
        ifl_vlb,
        ref_tss_dfe.iloc[mmy_idx_bgn:mmy_idx_end],
        ref_crd_dfe,
        not_ifl_fgs_ary)
        for mmy_idx_bgn, mmy_idx_end in zip(mmy_ixs[:-1], mmy_ixs[1:])]

    if (mpg_pol_sze == 1) or (mmy_ixs_sze == 2):

        for ags in mpg_ags: lmt_ifl_ncf_sgl(ags)

    else:

        if mpg_pol is None:
            mpg_pol = Pool(min(len(mpg_ags), mpg_pol_sze))

        list(mpg_pol.imap_unordered(lmt_ifl_ncf_sgl, mpg_ags, chunksize=1))

        mpg_pol.close()
        mpg_pol.join()
        mpg_pol = None

    return
#==============================================================================
#
#==============================================================================


def lmt_ifl_ncf_sgl(ags):

    (xcs,
     ycs,
     lck,
     var_llm,
     var_ulm,
     ifl_pth,
     otp_pth,
     ifl_tlb,
     ifl_vlb,
     ref_tss_dfe,
     ref_crd_dfe,
     not_ifl_fgs_ary) = ags

    vb = False

    if vb: print('')
    if vb: print(f'Reading netCDF data...')

    bgn_tmr = timeit.default_timer()

    with nc.Dataset(ifl_pth, 'r') as ncf_hdl:

        tcs_dts = ncf_hdl[ifl_tlb]

        red_tcs_ixs = nc.date2num(
            ref_tss_dfe.index.to_pydatetime(),
            units=tcs_dts.units,
            calendar=tcs_dts.calendar).astype(np.int64)

        if vb: print('Ixs to read:', red_tcs_ixs[0], red_tcs_ixs[-1])

        var_dts = ncf_hdl[ifl_vlb]
        var_vls = var_dts[red_tcs_ixs,:,:].data

        var_vls_shp = var_vls.shape

        var_vls_sts = var_vls_shp[0]

        assert var_vls_shp[1:] == xcs.shape, (var_vls_shp[1:], xcs.shape)

        if vb: print('Shape of input grid:', var_vls_shp)

    tcs_tim = ref_tss_dfe.index

    end_tmr = timeit.default_timer()

    if vb: print(f'Reading netCDF data took {end_tmr - bgn_tmr:0.2f} secs.')
    #==========================================================================

    if vb: print('')
    if vb: print(f'Setting cells outside polygon(s) range to NaN...')

    bgn_tmr = timeit.default_timer()

    for i in range(var_vls.shape[1]):
        for j in range(var_vls.shape[2]):
            if not_ifl_fgs_ary[i, j]: var_vls[:, i, j] = np.nan

    end_tmr = timeit.default_timer()

    if vb: print(f'Setting cells to NaN took {end_tmr - bgn_tmr:0.2f} secs.')
    #==========================================================================

    if vb: print('')
    if vb: print('Lower limits...')
    if vb: print('NaN Min Before:', np.nanmin(var_vls))

    llm_ctr = 0
    var_llm_ixs = var_vls < var_llm

    var_llm_ixs_any = var_llm_ixs.any(axis=(1, 2))

    if var_llm_ixs_any.any():
        for t in range(var_vls_sts):

            if not var_llm_ixs_any[t]: continue

            llm_ctr += var_llm_ixs[t].sum(dtype=np.uint64)

            var_stp = var_vls[t,:,:]

            var_stp[var_llm_ixs[t]] = np.nan

            var_vls[t,:,:] = var_stp

    if vb: print(f'{llm_ctr} were below lower limit.')
    #==========================================================================

    if vb: print('')
    if vb: print('Upper limits...')
    if vb: print('NaN Max Before:', np.nanmax(var_vls))

    ulm_ctr = 0
    var_ulm_ixs = var_vls > var_ulm

    var_ulm_ixs_any = var_ulm_ixs.any(axis=(1, 2))

    if var_ulm_ixs_any.any():
        for t in range(var_vls_sts):

            if not var_ulm_ixs_any[t]: continue

            ulm_ctr += var_ulm_ixs[t].sum(dtype=np.uint64)

            var_stp = var_vls[t,:,:]

            var_stp[var_ulm_ixs[t]] = np.nan

            var_vls[t,:,:] = var_stp

    if vb: print(f'{ulm_ctr} were above upper limit.')
    #==========================================================================

    if vb: print('')
    if vb: print('Infilling...')

    bgn_tmr = timeit.default_timer()

    nan_ctr = 0
    ifl_ctr = 0
    for i in range(var_vls.shape[1]):

        # if vb: print('PNT:', i)

        for j in range(var_vls.shape[2]):

            if not_ifl_fgs_ary[i, j]: continue

            nan_tss = np.isnan(var_vls[:, i, j])

            if not nan_tss.any(): continue

            nan_sum = nan_tss.sum(dtype=np.uint64)

            nan_ctr += nan_sum

            pnt_tss = var_vls[:, i, j]

            ref_crd_dst = (
                ((ref_crd_dfe['X'].values - xcs[i, j]) ** 2) +
                ((ref_crd_dfe['Y'].values - ycs[i, j]) ** 2)) ** 0.5

            ref_crd_dst_ixs = np.argsort(ref_crd_dst)

            for k in ref_crd_dst_ixs:
                if nan_sum == 0: break

                fnt_ixs = tcs_tim[nan_tss].intersection(
                    ref_tss_dfe.iloc[:, k].dropna().index)

                if not fnt_ixs.size: continue

                nan_sum -= fnt_ixs.size

                ifl_ctr += fnt_ixs.size

                fnt_ixs_int = tcs_tim.get_indexer(fnt_ixs)

                nan_tss[fnt_ixs_int] = False

                pnt_tss[fnt_ixs_int] = (
                    ref_tss_dfe.loc[fnt_ixs, ref_tss_dfe.columns[k]])

            var_vls[:, i, j] = pnt_tss

    end_tmr = timeit.default_timer()

    if vb: print(f'Infilling took {end_tmr - bgn_tmr:0.2f} secs.')

    if vb: print(
        'Total NaNs before and after fixing:', (nan_ctr, nan_ctr - ifl_ctr))

    if vb: print('NaN Min After:', np.nanmin(var_vls))
    if vb: print('NaN Max After:', np.nanmax(var_vls))
    #==========================================================================

    with lck:
        bgn_tmr = timeit.default_timer()

        if vb: print('')
        if vb: print('Updating values on disk...')

        with nc.Dataset(otp_pth, 'r+') as ncf_hdl:

            var_dts = ncf_hdl[ifl_vlb]

            var_dts[red_tcs_ixs,:,:] = var_vls

        end_tmr = timeit.default_timer()

        if vb: print(
            f'Updating values on disk took {end_tmr - bgn_tmr:0.2f} secs.')
    #==========================================================================

    return


def get_fgs_cls_sgl(ags):

    (idx,
     xcs,
     ycs,
     ply_pth,
     ply_fld,
     ply_bfr_dst,
    ) = ags

    vb = False

    if vb: print('Buffering polygon(s)...')
    ctn_ply = get_merged_poly(ply_pth, ply_fld, 0.0, ply_bfr_dst)
    assert ctn_ply

    not_ifl_fgs_ary = np.zeros_like(xcs, dtype=np.bool_)

    ply_xmn, ply_xmx, ply_ymn, ply_ymx = ctn_ply.GetEnvelope()

    for i in range(xcs.shape[0]):
        for j in range(xcs.shape[1]):

            if not (ply_xmn < xcs[i, j] < ply_xmx):
                not_ifl_fgs_ary[i, j] = True

            elif not (ply_ymn < ycs[i, j] < ply_ymx):
                not_ifl_fgs_ary[i, j] = True

            elif not chk_ctn(ctn_ply, xcs[i, j], ycs[i, j]):
                not_ifl_fgs_ary[i, j] = True

    return (idx, not_ifl_fgs_ary,)


def chk_ctn(ply, xcd, ycd):

    pt = ogr.CreateGeometryFromWkt("POINT (%f %f)" % (float(xcd), float(ycd)))

    assert pt is not None, f'Point returned a Null point!'

    return ply.Contains(pt)


def get_ctd_cds(xcs, ycs, var_vls_shp):

    assert all(xcs.shape), xcs.shape
    assert all(ycs.shape), ycs.shape
    assert var_vls_shp[1:], var_vls_shp

    assert np.all(np.isfinite(xcs))
    assert np.all(np.isfinite(ycs))

    # Needs centroids in 2D!
    if xcs.ndim == 1:

        assert np.all(xcs[+1:] > xcs[:-1]), xcs

        assert (np.all(ycs[:-1] > ycs[+1:]) or
                np.all(ycs[+1:] > ycs[:-1])), ycs

        if var_vls_shp[1:] == (ycs.shape[0], xcs.shape[0]):
            pass

        elif var_vls_shp[1:] == (ycs.shape[0] - 1, xcs.shape[0] - 1):

            xcs = xcs[:-1] + (0.5 * (xcs[+1:] - xcs[:-1]))
            ycs = ycs[:-1] - (0.5 * (ycs[:-1] - ycs[+1:]))

        else:
            raise NotImplementedError((var_vls_shp, xcs.shape, ycs.shape))

        xcs, ycs = np.meshgrid(xcs, ycs)

    else:
        assert xcs.ndim == 2, xcs.ndim

        assert np.all(xcs[:, +1:] > xcs[:,:-1]), xcs

        assert(np.all(ycs[:-1,:] > ycs[+1:,:]) or
               np.all(ycs[+1:,:] > ycs[:-1,:])), ycs

        if var_vls_shp[1:] == xcs.shape:

            pass

        elif var_vls_shp[1:] == (xcs.shape[0] - 1, xcs.shape[1] - 1):

            xcs = xcs[:,:-1] + (0.5 * (xcs[+1:,:] - xcs[:,:-1]))
            ycs = ycs[:-1,:] - (0.5 * (ycs[:-1,:] - ycs[+1:,:]))

            xcs = xcs[:-1,:]
            ycs = ycs[:,:-1]

        else:
            raise NotImplementedError((var_vls_shp, xcs.shape, ycs.shape))

    assert xcs.shape == ycs.shape, (xcs.shape, ycs.shape)
    assert var_vls_shp[1:] == xcs.shape, (var_vls_shp, xcs.shape, ycs.shape)

    assert np.all(np.isfinite(xcs))
    assert np.all(np.isfinite(ycs))

    assert np.all(xcs[:, +1:] > xcs[:,:-1]), xcs

    assert(np.all(ycs[:-1,:] > ycs[+1:,:]) or
           np.all(ycs[+1:,:] > ycs[:-1,:])), ycs
    return xcs, ycs


def get_tss_ncf_cds_cmn_dfs(
        tss_pth,
        cds_pth,
        ply_pth,
        ply_fld,
        pln_bfr,
        ncf_tme_ixs):

    # Based on polygon.
    cds_dfe = get_std_cds_dfe(
        cds_pth,
        ply_pth,
        ply_fld,
        0.0,
        pln_bfr)

    # Based on time.
    tss_dfe = pd.read_pickle(tss_pth)
    ncf_tss_cmn_ixs = tss_dfe.index.intersection(ncf_tme_ixs)

    assert ncf_tss_cmn_ixs.size

    tss_dfe = tss_dfe.loc[ncf_tss_cmn_ixs,:]

    tss_cts = tss_dfe.count(axis=0).values

    tss_kep_lbs = tss_dfe.columns[tss_cts > 0]

    assert tss_kep_lbs.size

    # Based on crds.
    tss_cds_cmn_ixs = tss_kep_lbs.intersection(cds_dfe.index)

    assert tss_cds_cmn_ixs.size

    tss_dfe = tss_dfe.loc[:, tss_cds_cmn_ixs].copy()
    cds_dfe = cds_dfe.loc[tss_cds_cmn_ixs,:].copy()

    return tss_dfe, cds_dfe


def get_std_cds_dfe(
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

    in_crds_df.index = in_crds_df.index.astype(str)

    cat_buff = get_merged_poly(
        subset_shp_file, subset_shp_fld, simplyify_tol, shp_buff_dist)

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

    assert subset_crds_stns.size

    if False: print(subset_crds_stns.size, 'stations selected in crds_df!')

    return in_crds_df.loc[subset_crds_stns,:].copy()


def get_stns_in_poly(crds_df, poly):

    contain_stns = []

    for i, stn in enumerate(crds_df.index):

        x, y = crds_df.iloc[i].loc[['X', 'Y']]

        pt = ogr.CreateGeometryFromWkt("POINT (%f %f)" % (float(x), float(y)))

        if pt is None:
            print(f'Station {stn} returned a Null point!')
            continue

        if poly.Contains(pt): contain_stns.append(stn)

    return contain_stns


def get_merged_poly(in_shp, field='DN', simplify_tol=0, ply_bfr_dst=0):

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

    if ply_bfr_dst:
        merged_cat = merged_cat.Buffer(ply_bfr_dst)

    return merged_cat


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
