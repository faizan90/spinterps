# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jul 13, 2024

2:32:01 PM

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
from multiprocessing import Pool, Lock, Manager

import numpy as np
import psutil as ps
import pandas as pd
import netCDF4 as nc

DEBUG_FLAG = True


def main():

    main_dir = Path(r'P:\dwd_meteo\gridded\extract_radolan')
    os.chdir(main_dir)

    mpg_pol_sze = 16

    inp_ncf_phs = Path('b_merged_in_time').glob('*.nc')

    rmp_res = 'D'
    min_cnt = 24
    rmp_typ = 'sum'

    inp_vlb = 'RW'
    inp_xlb = 'x_utm32n'
    inp_ylb = 'y_utm32n'
    inp_tlb = 'time'

    otp_tus = 'days since 2006-01-01'
    otp_cmp_lvl = 1

    # Applied to shift the entire time series by this offset.
    tme_dla = pd.Timedelta(-7, unit='h')
    tme_bfr_sps = 24
    inp_cnk_sze = 2 * (1024 ** 3)  # In bytes and per thread!

    ovr_wte_flg = True

    ott_dir = Path(r'c_hourly_rsmp_to_daily')
    #==========================================================================

    # Don't MP!
    mpg_ags = ((
        rmp_res,
        min_cnt,
        rmp_typ,
        inp_vlb,
        inp_xlb,
        inp_ylb,
        inp_tlb,
        otp_tus,
        tme_dla,
        ott_dir,
        mpg_pol_sze,
        inp_cnk_sze,
        tme_bfr_sps,
        otp_cmp_lvl,
        inp_ncf_pth,
        ovr_wte_flg,
        ) for inp_ncf_pth in inp_ncf_phs)

    for ags in mpg_ags: rsmp_ncf_tme(ags)
    return


def rsmp_ncf_tme(ags):

    (rmp_res,
     min_cnt,
     rmp_typ,
     inp_vlb,
     inp_xlb,
     inp_ylb,
     inp_tlb,
     otp_tus,
     tme_dla,
     ott_dir,
     mpg_pol_sze,
     inp_cnk_sze,
     tme_bfr_sps,
     otp_cmp_lvl,
     inp_ncf_pth,
     ovr_wte_flg,
    ) = ags

    try: ott_dir.mkdir(exist_ok=True, parents=True)
    except: pass

    otp_ncf_pth = ott_dir / inp_ncf_pth.name

    if (not ovr_wte_flg) and otp_ncf_pth.exists(): return

    vb = True

    otp_dte = np.float32

    print(f'Processing {inp_ncf_pth.name}...')

    if vb: print('Initiating emtpy nc file...')

    int_ncf((
        inp_vlb,
        inp_xlb,
        inp_ylb,
        inp_tlb,
        tme_dla,
        otp_tus,
        rmp_res,
        min_cnt,
        otp_dte,
        inp_ncf_pth,
        otp_cmp_lvl,
        otp_ncf_pth))
    #==========================================================================

    # Memory management.
    with nc.Dataset(inp_ncf_pth) as ncf_hdl:
        inp_shp = ncf_hdl[inp_vlb].shape
        inp_dte = ncf_hdl[inp_vlb].dtype

    inp_vle_cts = np.prod(inp_shp, dtype=np.uint64)

    tot_mmy = min(inp_cnk_sze, int(ps.virtual_memory().available * 0.5))

    mmy_cns_cnt = ceil((inp_vle_cts / tot_mmy) * inp_dte.itemsize)

    assert mmy_cns_cnt >= 1, mmy_cns_cnt

    if mmy_cns_cnt > 1:
        if vb: print('Memory not enough to read in one go!')

        mmy_ixs = np.linspace(
            0,
            inp_shp[0],
            (mmy_cns_cnt + 1),
            endpoint=True,
            dtype=np.int64)

        assert mmy_ixs[+0] == 0, mmy_ixs
        assert mmy_ixs[-1] == inp_shp[0], mmy_ixs

        assert np.unique(mmy_ixs).size == mmy_ixs.size, (
            np.unique(mmy_ixs).size, mmy_ixs.size)

    else:
        mmy_ixs = np.array([0, inp_shp[0]])

    mmy_ixs_sze = mmy_ixs.size

    if vb: print(f'Total chunks to read: {mmy_ixs_sze - 1}')
    #==========================================================================

    if (mpg_pol_sze == 1) or (mmy_ixs_sze == 2):

        lck = Lock()

    else:
        mgr = Manager()
        lck = mgr.Lock()

    mpg_ags = [(
        lck,
        inp_vlb,
        inp_tlb,
        tme_dla,
        otp_dte,
        rmp_res,
        rmp_typ,
        min_cnt,
        inp_shp,
        otp_tus,
        lop_idx,
        mmy_idx_bgn,
        mmy_idx_end,
        mpg_pol_sze,
        mmy_ixs_sze,
        inp_ncf_pth,
        tme_bfr_sps,
        otp_ncf_pth)
        for lop_idx, (mmy_idx_bgn, mmy_idx_end) in enumerate(
            zip(mmy_ixs[:-1], mmy_ixs[1:]))]

    if (mpg_pol_sze == 1) or (mmy_ixs_sze == 2):

        for ags in mpg_ags: rsmp_ncf_tme_sgl(ags)

    else:
        mpg_pol = Pool(min(len(mpg_ags), mpg_pol_sze))

        list(mpg_pol.imap_unordered(rsmp_ncf_tme_sgl, mpg_ags, chunksize=1))

        mpg_pol.close()
        mpg_pol.join()
        mpg_pol = None
    return


def rsmp_ncf_tme_sgl(ags):

    (lck,
     inp_vlb,
     inp_tlb,
     tme_dla,
     otp_dte,
     rmp_res,
     rmp_typ,
     min_cnt,
     inp_shp,
     otp_tus,
     lop_idx,
     mmy_idx_bgn,
     mmy_idx_end,
     mpg_pol_sze,
     mmy_ixs_sze,
     inp_ncf_pth,
     tme_bfr_sps,
     otp_ncf_pth) = ags

    vb = False

    if vb: print('')

    if lop_idx == 0: bgn_tmr = timeit.default_timer()

    if vb: print(
        f'Reading input data from {inp_ncf_pth.name} | '
        f'{lop_idx + 1} out of {mmy_ixs_sze - 1} | '
        f'{mmy_idx_bgn} till {mmy_idx_end} in {inp_shp[0]}...')

    with nc.Dataset(inp_ncf_pth) as ncf_hdl:

        inp_dta = ncf_hdl[inp_vlb][
            mmy_idx_bgn:mmy_idx_end + tme_bfr_sps].data.astype(
                np.float32, order='c')

        inp_tds = ncf_hdl[inp_tlb]
        inp_tcs = inp_tds[mmy_idx_bgn:mmy_idx_end + tme_bfr_sps].data

        inp_tus = inp_tds.units
        inp_tcr = inp_tds.calendar

        inp_tds = None

    ncf_hdl = None
    #======================================================================

    inp_tix = pd.DatetimeIndex(nc.num2date(
        inp_tcs,
        inp_tus,
        inp_tcr,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True))

    if tme_dla:
        inp_tix += tme_dla
    #======================================================================

    inp_tus = None

    if vb: print('Converting netCDF to DataFrame...')

    inp_dfe = pd.DataFrame(
        index=inp_tix,
        columns=[
            str((i, j))
            for i in range(inp_dta.shape[1])
            for j in range(inp_dta.shape[2])],
        data=inp_dta.reshape(-1, inp_dta.shape[1] * inp_dta.shape[2]),
        dtype=otp_dte)

    # for ij in inp_dfe.columns:
    #     i, j = eval(ij)
    #     inp_dfe.loc[:, ij] = inp_dta[:, i, j].astype(otp_dte)

    inp_dta = None
    #======================================================================

    if True:
        if vb: print('Counting timesteps...')

        cts_dfe = inp_dfe.resample(rmp_res).count().astype(otp_dte)

        if rmp_res == 'm':
            assert min_cnt is None, 'For months, min_count must be None!'

            min_cnt_ser = cts_dfe.index.days_in_month.values.reshape(-1, 1)

        else:
            min_cnt_ser = min_cnt

        cts_dfe[cts_dfe < min_cnt_ser] = np.nan
        cts_dfe[cts_dfe >= min_cnt_ser] = 1.0

        assert cts_dfe.max().max() <= 1.0, cts_dfe.max().max()

        min_cnt_ser = None
    else:
        cts_dfe = None
    #======================================================================

    tme_srs = pd.Series(index=inp_tix, dtype=np.int8)
    tme_srs[:] = 1

    tme_sts_srs = tme_srs.resample(rmp_res).count()

    if rmp_res == 'm':
        assert min_cnt is None, 'For months, min_count must be None!'

        min_cnt_srs = tme_sts_srs.index.days_in_month.values.reshape(-1, 1)

    else:
        min_cnt_srs = min_cnt

    tme_tke_ixs = np.where((tme_sts_srs >= min_cnt_srs).values)[0]
    tme_tke_ixs_bgn, tme_tke_ixs_end = tme_tke_ixs[0], tme_tke_ixs[-1]
    #======================================================================

    if vb: print('Resampling...')

    rmp_dfe = getattr(inp_dfe.resample(rmp_res), rmp_typ)()

    inp_dfe = None

    if cts_dfe is not None:
        rmp_dfe *= cts_dfe
        cts_dfe = None

    rmp_dfe = rmp_dfe.astype(otp_dte)
    #======================================================================

    if vb: print('Preparing to write output...')

    otp_dta = np.empty(
        (tme_tke_ixs_end - tme_tke_ixs_bgn + 1, inp_shp[1], inp_shp[2]),
        dtype=otp_dte)

    for ij in rmp_dfe.columns:
        i, j = eval(ij)

        otp_dta[:, i, j] = rmp_dfe.loc[:, ij].values[
            tme_tke_ixs_bgn:tme_tke_ixs_end + 1]
    #======================================================================

    tme_tke_ixs_bgn_otp, tme_tke_ixs_end_otp = nc.date2num(
        (rmp_dfe.index[tme_tke_ixs_bgn],
         rmp_dfe.index[tme_tke_ixs_end]),
        units=otp_tus,
        calendar=inp_tcr).astype(np.int64)

    rmp_dfe = None
    #======================================================================

    if vb: print(
        f'Writing output data '
        f'{tme_tke_ixs_bgn_otp, tme_tke_ixs_end_otp}...')

    with lck:
        with nc.Dataset(otp_ncf_pth, 'r+') as otp_ncf_hdl:

            otp_vds = otp_ncf_hdl[inp_vlb]

            otp_vds[tme_tke_ixs_bgn_otp:tme_tke_ixs_end_otp + 1,:,:] = otp_dta

    otp_dta = None
    #======================================================================

    if lop_idx == 0:

        end_tmr = timeit.default_timer()

        lop_tme = end_tmr - bgn_tmr

        tot_tme = (lop_tme * (mmy_ixs_sze - 1)) / mpg_pol_sze

        print(f'Took {lop_tme:0.2f} secs for one loop!')

        print(f'Need ca. {tot_tme:0.2f} secs for resampling in total.\n')

    if vb: print('')

    return


def int_ncf(ags):

    (inp_vlb,
     inp_xlb,
     inp_ylb,
     inp_tlb,
     tme_dla,
     otp_tus,
     rmp_res,
     min_cnt,
     otp_dte,
     inp_ncf_pth,
     otp_cmp_lvl,
     otp_ncf_pth) = ags

    with nc.Dataset(inp_ncf_pth) as inp_ncf_hdl, nc.Dataset(
        otp_ncf_pth, 'w') as otp_ncf_hdl:

        inp_vds = inp_ncf_hdl[inp_vlb]
        inp_xds = inp_ncf_hdl[inp_xlb]
        inp_yds = inp_ncf_hdl[inp_ylb]
        inp_tds = inp_ncf_hdl[inp_tlb]

        inp_tcs = inp_tds[:].data

        inp_tus = inp_tds.units
        inp_tcr = inp_tds.calendar
        #======================================================================

        inp_tix = pd.DatetimeIndex(nc.num2date(
            inp_tcs,
            inp_tus,
            inp_tcr,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True))

        if tme_dla:
            inp_tix += tme_dla

        tme_srs = pd.Series(index=inp_tix, dtype=np.int8)
        tme_srs[:] = 1

        tme_sts_srs = tme_srs.resample(rmp_res).count()

        if rmp_res == 'm':
            assert min_cnt is None, 'For months, min_count must be None!'

            min_cnt_srs = tme_sts_srs.index.days_in_month.values.reshape(-1, 1)

        else:
            min_cnt_srs = min_cnt

        tme_tke_ixs = np.where((tme_sts_srs >= min_cnt_srs).values)[0]
        tme_tke_ixs_bgn, tme_tke_ixs_end = tme_tke_ixs[0], tme_tke_ixs[-1]

        otp_tme = tme_srs.resample(rmp_res).count().index[
            tme_tke_ixs_bgn:tme_tke_ixs_end + 1]
        #======================================================================

        for i, dim in enumerate(inp_xds.dimensions):
            otp_ncf_hdl.createDimension(dim, inp_xds.shape[i])

        for i, dim in enumerate(inp_yds.dimensions):

            if dim in otp_ncf_hdl.dimensions: continue

            otp_ncf_hdl.createDimension(dim, inp_yds.shape[i])

        assert inp_tds.ndim == 1, inp_tds.ndim

        otp_ncf_hdl.createDimension(inp_tds.dimensions[0], None)
        #======================================================================

        otp_xds = otp_ncf_hdl.createVariable(
            inp_xlb,
            inp_xds.dtype,
            dimensions=inp_xds.dimensions)

        otp_xds[:] = inp_xds[:].data

        otp_yds = otp_ncf_hdl.createVariable(
            inp_ylb,
            inp_yds.dtype,
            dimensions=inp_yds.dimensions)

        otp_yds[:] = inp_yds[:].data
        #======================================================================

        otp_tds = otp_ncf_hdl.createVariable(
            inp_tlb,
            inp_tds.dtype,
            dimensions=inp_tds.dimensions)

        otp_tds.calendar = inp_tcr
        otp_tds.units = otp_tus

        otp_tcs = nc.date2num(
            otp_tme.to_pydatetime(),
            units=otp_tus,
            calendar=inp_tcr).astype(np.int64)

        otp_tds[:] = otp_tcs
        #======================================================================

        otp_ncf_hdl.createVariable(
            inp_vlb,
            otp_dte,
            dimensions=inp_vds.dimensions,
            complevel=otp_cmp_lvl,
            chunksizes=(1, inp_vds.shape[1], inp_vds.shape[2]))
        #======================================================================
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
