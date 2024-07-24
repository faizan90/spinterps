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
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc

DEBUG_FLAG = True


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\hydmod\hmg3d')
    os.chdir(main_dir)

    inp_ncf_pth = Path(r'tem_hourly_427_2km.nc')

    rmp_res = 'D'
    min_cnt = 24
    rmp_typ = 'mean'  # 'sum'  #

    otp_ncf_pth = Path(rf'{inp_ncf_pth.stem}_RR{rmp_res}_RT{rmp_typ}.nc')

    inp_vlb = 'EDK'
    inp_xlb = 'X'
    inp_ylb = 'Y'
    inp_tlb = 'time'

    otp_tus = 'days since 1900-01-01'
    otp_cmp_lvl = 1

    # Applied to shift the entire time series by this offset.
    tim_dla = pd.Timedelta(-1, unit='h')
    #==========================================================================

    print('Reading input data...')
    with nc.Dataset(inp_ncf_pth) as ncf_hdl:

        inp_dta = ncf_hdl[inp_vlb][:].data.astype(dtype=np.float32, order='f')

        inp_tds = ncf_hdl[inp_tlb]
        inp_tcs = inp_tds[:].data

        inp_tus = inp_tds.units
        inp_tcr = inp_tds.calendar

        inp_tds = None

    ncf_hdl = None
    #==========================================================================

    print('Converting netCDF to DataFrame...')

    inp_tix = pd.DatetimeIndex(nc.num2date(
        inp_tcs,
        inp_tus,
        inp_tcr,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True))

    if tim_dla:
        inp_tix += tim_dla

    inp_tus = None

    inp_dfe = pd.DataFrame(
        index=inp_tix,
        columns=[
            str((i, j))
            for i in range(inp_dta.shape[1])
            for j in range(inp_dta.shape[2])],
        dtype=np.float32)

    for ij in inp_dfe.columns:
        i, j = eval(ij)
        inp_dfe.loc[:, ij] = inp_dta[:, i, j]

    inp_dta = None
    #==========================================================================

    print('Counting timesteps...')

    cts_dfe = inp_dfe.resample(rmp_res).count().astype(np.float32)

    if rmp_res == 'm':
        assert min_cnt is None, 'For months, min_count must be None!'

        min_cnt_ser = cts_dfe.index.days_in_month.values.reshape(-1, 1)

    else:
        min_cnt_ser = min_cnt

    cts_dfe[cts_dfe < min_cnt_ser] = np.nan
    cts_dfe[cts_dfe >= min_cnt_ser] = 1.0

    assert cts_dfe.max().max() <= 1.0, cts_dfe.max().max()

    min_cnt_ser = None
    #==========================================================================

    print('Rsampling...')

    rmp_dfe = getattr(inp_dfe.resample(rmp_res), rmp_typ)()

    inp_dfe = None

    rmp_dfe *= cts_dfe

    rmp_dfe = rmp_dfe.astype(np.float32)

    cts_dfe = None
    #==========================================================================

    print('Preparing to write output...')

    with nc.Dataset(
        inp_ncf_pth) as inp_ncf_hdl, nc.Dataset(
            otp_ncf_pth, 'w') as otp_ncf_hdl:

        inp_vds = inp_ncf_hdl[inp_vlb]
        inp_xds = inp_ncf_hdl[inp_xlb]
        inp_yds = inp_ncf_hdl[inp_ylb]
        inp_tds = inp_ncf_hdl[inp_tlb]
        #======================================================================

        for i, dim in enumerate(inp_xds.dimensions):
            otp_ncf_hdl.createDimension(dim, inp_xds.shape[i])

        for i, dim in enumerate(inp_yds.dimensions):

            if dim in otp_ncf_hdl.dimensions: continue

            otp_ncf_hdl.createDimension(dim, inp_yds.shape[i])

        assert inp_tds.ndim == 1, inp_tds.ndim

        otp_ncf_hdl.createDimension(inp_tds.dimensions[0], rmp_dfe.shape[0])
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
            rmp_dfe.index.to_pydatetime(),
            units=otp_tus,
            calendar=inp_tcr).astype(inp_tcs.dtype)

        otp_tds[:] = otp_tcs
        #======================================================================

        print('Writing output data...')

        otp_vds = otp_ncf_hdl.createVariable(
            inp_vlb,
            rmp_dfe.values.dtype,
            dimensions=inp_vds.dimensions,
            complevel=otp_cmp_lvl,
            chunksizes=(1, inp_vds.shape[1], inp_vds.shape[2]))

        otp_dta = np.empty(
            (rmp_dfe.shape[0], inp_vds.shape[1], inp_vds.shape[2]),
            dtype=np.float32)

        for ij in rmp_dfe.columns:
            i, j = eval(ij)
            otp_dta[:, i, j] = rmp_dfe.loc[:, ij]

        otp_vds[:] = otp_dta

        otp_dta = None

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
