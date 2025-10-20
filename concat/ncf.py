# -*- coding: utf-8 -*-

'''
Created on 13.10.2025

@author: Faizan-TU Munich
'''

import numpy as np
import pandas as pd
import netCDF4 as nc


def cct_ncs(
        itp_phs,
        tme_lbl,
        xcs_lbl,
        ycs_lbl,
        vbe_lbs,
        tme_bgn,
        tme_end,
        otp_pth,
        tme_frq,
        tme_dta,
        tme_uts,
        tme_dvr,
        cmp_lvl):

    otp_int_flg = False

    with nc.Dataset(otp_pth, 'w') as otp_ncf_hdl:

        otp_ncf_hdl.set_auto_mask(False)

        fnd_fls_flg = False

        for itp_pth in itp_phs:

            fnd_fls_flg = True

            print(itp_pth.stem)

            with nc.Dataset(itp_pth) as itp_ncf_hdl:

                itp_tme = itp_ncf_hdl[tme_lbl]

                if not otp_int_flg:

                    if itp_ncf_hdl[xcs_lbl].ndim == 2:

                        otp_ncf_hdl.createDimension(
                            xcs_lbl, itp_ncf_hdl[xcs_lbl].shape[1])

                        otp_ncf_hdl.createDimension(
                            ycs_lbl, itp_ncf_hdl[ycs_lbl].shape[0])

                        otp_ncf_hdl.createDimension(tme_lbl, None)

                        otp_xcs = otp_ncf_hdl.createVariable(
                            xcs_lbl,
                            'd',
                            dimensions=(ycs_lbl, xcs_lbl))

                        otp_ycs = otp_ncf_hdl.createVariable(
                            ycs_lbl,
                            'd',
                            dimensions=(ycs_lbl, xcs_lbl))

                    elif itp_ncf_hdl[xcs_lbl].ndim == 1:

                        otp_ncf_hdl.createDimension(
                            xcs_lbl, itp_ncf_hdl[xcs_lbl].shape[0])

                        otp_ncf_hdl.createDimension(
                            ycs_lbl, itp_ncf_hdl[ycs_lbl].shape[0])

                        otp_ncf_hdl.createDimension(tme_lbl, None)

                        otp_xcs = otp_ncf_hdl.createVariable(
                            xcs_lbl,
                            'd',
                            dimensions=(xcs_lbl,))

                        otp_ycs = otp_ncf_hdl.createVariable(
                            ycs_lbl,
                            'd',
                            dimensions=(ycs_lbl,))

                    else:
                        raise NotImplementedError(itp_ncf_hdl[xcs_lbl].ndim)

                    otp_tme = otp_ncf_hdl.createVariable(
                        tme_lbl, 'f8', dimensions=tme_lbl)

                    otp_vbs = {}
                    for var_lab in vbe_lbs:
                        otp_vbe = otp_ncf_hdl.createVariable(
                            var_lab,
                            'f4',
                            dimensions=(tme_lbl, ycs_lbl, xcs_lbl),
                            compression='zlib',
                            complevel=cmp_lvl,
                            chunksizes=(1,
                                        itp_ncf_hdl[var_lab].shape[1],
                                        itp_ncf_hdl[var_lab].shape[2]))

                        otp_vbs[var_lab] = otp_vbe

                    otp_vbe = None

                    otp_xcs[:] = itp_ncf_hdl[xcs_lbl][:]
                    otp_ycs[:] = itp_ncf_hdl[ycs_lbl][:]

                    if tme_uts is not None:
                        otp_tme_uts = tme_uts

                    else:
                        otp_tme_uts = itp_tme.units

                    otp_tme_cdr = itp_tme.calendar

                    otp_tme.units = otp_tme_uts
                    otp_tme.calendar = otp_tme_cdr

                    if tme_frq is None:
                        if 'hours' in otp_tme_uts:
                            tme_frq = 'h'

                        elif 'days' in otp_tme_uts:
                            tme_frq = 'D'

                        elif 'minutes' in otp_tme_uts:
                            tme_frq = 'min'

                        else:
                            raise NotImplementedError(
                                f'Unknown time units in {otp_tme_uts}!')

                    otp_tme_ixs_pyn = pd.date_range(
                        tme_bgn,
                        tme_end,
                        freq=tme_frq).to_pydatetime()

                    if tme_dta is not None: otp_tme_ixs_pyn += tme_dta

                    otp_tme_ixs_all = nc.date2num(
                        otp_tme_ixs_pyn,
                        units=otp_tme_uts,
                        calendar=otp_tme_cdr)

                    otp_tme[:] = otp_tme_ixs_all

                    otp_tme_ixs_fgs = np.zeros(
                        otp_tme_ixs_all.size, dtype=bool)

                    otp_int_flg = True

                else:
                    assert otp_tme.units == otp_tme_uts
                    assert otp_tme.calendar == otp_tme_cdr

                    assert np.all(
                        np.isclose(otp_xcs[:], itp_ncf_hdl[xcs_lbl][:]))

                    assert np.all(
                        np.isclose(otp_ycs[:], itp_ncf_hdl[ycs_lbl][:]))

                if tme_dta is not None:

                    otp_tme_ixs_pyn = pd.DatetimeIndex(nc.num2date(
                        itp_tme[:].data,
                        units=itp_tme.units,
                        calendar=itp_tme.calendar,
                        only_use_cftime_datetimes=False))

                    if tme_dta is not None: otp_tme_ixs_pyn += tme_dta

                    otp_tme_ixs = nc.date2num(
                        otp_tme_ixs_pyn.to_pydatetime(),
                        units=otp_tme_uts,
                        calendar=otp_tme_cdr)  # .astype(np.int64)

                else:
                    if ((itp_tme.units != otp_tme_uts) or
                        (itp_tme.calendar != otp_tme_cdr)):

                        otp_tme_ixs = nc.num2date(
                            itp_tme[:].data,
                            units=itp_tme.units,
                            calendar=itp_tme.calendar)

                        otp_tme_ixs = nc.date2num(
                            otp_tme_ixs,
                            units=otp_tme_uts,
                            calendar=otp_tme_cdr)

                    else:
                        otp_tme_ixs = itp_tme[:].data  # .astype(np.int64)

                # Error when calendars don't match normally.
                assert np.in1d(otp_tme_ixs, otp_tme_ixs_all).all()

                if tme_dvr is not None:

                    tme_ist_ixs = otp_tme_ixs - otp_tme_ixs_all[0]

                    tme_ist_ixs /= tme_dvr

                    tme_ist_ixs = (np.round(tme_ist_ixs, 7)).astype(np.int64)

                else:
                    tme_ist_ixs = (
                        otp_tme_ixs - otp_tme_ixs_all[0]).astype(np.int64)

                assert np.unique(tme_ist_ixs).size == tme_ist_ixs.size
                #==============================================================

                assert not otp_tme_ixs_fgs[tme_ist_ixs].any()

                for var_lab in otp_vbs:
                    data = itp_ncf_hdl[var_lab][:].data.astype(np.float32)
                    otp_vbs[var_lab][tme_ist_ixs,:,:] = data

                    otp_tme_ixs_fgs[tme_ist_ixs] = True

        assert fnd_fls_flg, 'No files found!'

    return

