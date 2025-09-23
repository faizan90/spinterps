# -*- coding: utf-8 -*-

'''
Created on 02.09.2025

@author: Faizan-TU Munich
'''

import numpy as np
import netCDF4 as nc


def sth_ncs(ipt_ncs_phs_lbs, otp_ncf_pth, otp_dpe, cpn_lvl):

    ncs_vbs = []
    for ipt_ncf_pth_lbs in ipt_ncs_phs_lbs:
        ncf_vbs = get_ncf_vbs(*ipt_ncf_pth_lbs, otp_dpe)
        ncs_vbs.append(ncf_vbs)
    #==========================================================================

    kk = 0  # Reference ncs_vbs_idx.

    for ncf_vbs in ncs_vbs:
        assert ncs_vbs[kk].tsct == ncf_vbs.tsct
        assert ncs_vbs[kk].clwh == ncf_vbs.clwh
        assert ncs_vbs[kk].clht == ncf_vbs.clht

        assert np.isclose(ncs_vbs[kk].tcds, ncf_vbs.tcds).all()
        assert ncs_vbs[kk].tuts == ncf_vbs.tuts
        assert ncs_vbs[kk].tcdr == ncf_vbs.tcdr
    #==========================================================================

    # Stitching is column-wise.
    otp_ncf_rws = 0
    otp_ncf_cls = 0
    for ncf_vbs in ncs_vbs:
        otp_ncf_rws = max(ncf_vbs.rows, otp_ncf_rws)
        otp_ncf_cls += ncf_vbs.cols

    otp_ary = np.full(
        (ncs_vbs[kk].tsct, otp_ncf_rws, otp_ncf_cls),
        np.nan,
        dtype=ncs_vbs[kk].arry.dtype)

    ji = 0
    for ncf_vbs in ncs_vbs:
        ij = ncf_vbs.rows
        jj = ji + ncf_vbs.cols

        otp_ary[:,:ij, ji:jj] = ncf_vbs.arry

        ncf_vbs.arry = None

        ji = jj
    #==========================================================================

    out_ncf_hdl = nc.Dataset(otp_ncf_pth, 'w')

    assert out_ncf_hdl is not None
    #==========================================================================

    out_ncf_hdl.createDimension(ncs_vbs[kk].xlbl, otp_ary.shape[2])

    xcs_ncf_vbe = out_ncf_hdl.createVariable(
        ncs_vbs[kk].xlbl, ncs_vbs[kk].xcds.dtype, ncs_vbs[kk].xlbl)

    out_ncf_hdl.createDimension(ncs_vbs[kk].ylbl, otp_ary.shape[1])

    ycs_ncf_vbe = out_ncf_hdl.createVariable(
        ncs_vbs[kk].ylbl, ncs_vbs[kk].ycds.dtype, ncs_vbs[kk].ylbl)

    out_ncf_hdl.createDimension(ncs_vbs[kk].tlbl, otp_ary.shape[0])

    tcs_ncf_vbe = out_ncf_hdl.createVariable(
        ncs_vbs[kk].tlbl, ncs_vbs[kk].tcds.dtype, ncs_vbs[kk].tlbl)
    #==========================================================================

    otp_xcs = ncs_vbs[kk].xmin + (
        np.arange(0.0, otp_ary.shape[2]) * ncs_vbs[kk].clwh)

    otp_ycs = ncs_vbs[kk].ymax - (
        np.arange(0.0, otp_ary.shape[1]) * ncs_vbs[kk].clht)

    otp_tcs = ncs_vbs[kk].tcds
    tcs_ncf_vbe.units = ncs_vbs[kk].tuts
    tcs_ncf_vbe.calendar = ncs_vbs[kk].tcdr
    #==========================================================================

    xcs_ncf_vbe[:] = otp_xcs
    ycs_ncf_vbe[:] = otp_ycs
    tcs_ncf_vbe[:] = otp_tcs
    #==========================================================================

    vls_ncf_vbe = out_ncf_hdl.createVariable(
        ncs_vbs[kk].vlbl,
        otp_ary.dtype,
        (ncs_vbs[kk].tlbl, ncs_vbs[kk].ylbl, ncs_vbs[kk].xlbl),
        chunksizes=(1, otp_ary.shape[1], otp_ary.shape[2]),
        compression='zlib',
        complevel=cpn_lvl,)

    vls_ncf_vbe[:] = otp_ary
    #==========================================================================

    out_ncf_hdl.close()
    return


def get_ncf_vbs(
        inpt_xlbl,
        inpt_ylbl,
        inpt_tlbl,
        inpt_vlbl,
        inpt_path,
        otpt_dtpe,):

    assert inpt_path.exists()

    inpt_hndl = nc.Dataset(inpt_path, 'r')

    assert inpt_xlbl != inpt_ylbl
    assert inpt_xlbl != inpt_tlbl
    assert inpt_xlbl != inpt_vlbl

    assert inpt_ylbl != inpt_tlbl
    assert inpt_ylbl != inpt_vlbl

    assert inpt_tlbl != inpt_vlbl

    inpt_xcds = inpt_hndl[inpt_xlbl][:]
    assert inpt_xcds.ndim == 1
    assert np.isfinite(inpt_xcds).all()

    inpt_ycds = inpt_hndl[inpt_ylbl][:]
    assert inpt_ycds.ndim == 1
    assert np.isfinite(inpt_ycds).all()

    inpt_tcds = inpt_hndl[inpt_tlbl][:]
    assert inpt_tcds.ndim == 1
    assert np.isfinite(inpt_tcds).all()

    inpt_tuts = inpt_hndl[inpt_tlbl].units
    inpt_tcdr = inpt_hndl[inpt_tlbl].calendar

    inpt_arry = inpt_hndl[inpt_vlbl][:]

    assert inpt_arry.ndim == 3

    inpt_rows = inpt_arry.shape[1]
    inpt_cols = inpt_arry.shape[2]

    inpt_xmin = inpt_xcds[0]
    inpt_ymax = inpt_ycds[0]

    inpt_clwh = inpt_xcds[1] - inpt_xcds[0]
    inpt_clht = inpt_ycds[0] - inpt_ycds[1]

    inpt_xmax = inpt_xcds[-1]
    inpt_ymin = inpt_ycds[-1]

    inpt_tsct = inpt_arry.shape[0]

    inpt_hndl.close()
    #==========================================================================

    assert inpt_clwh > 0
    assert inpt_clht > 0

    assert inpt_xmax > inpt_xmin
    assert inpt_ymax > inpt_ymin

    assert np.isclose(inpt_clwh, inpt_xcds[+1:] - inpt_xcds[:-1]).all()
    assert np.isclose(inpt_clht, inpt_ycds[:-1] - inpt_ycds[+1:]).all()
    #==========================================================================

    otpt_tsct = inpt_tsct

    otpt_clwh = inpt_clwh
    otpt_clht = inpt_clht

    otpt_cols = inpt_cols
    otpt_rows = inpt_rows

    otpt_xmin = inpt_xmin
    otpt_xmax = inpt_xmax

    otpt_ymin = inpt_ymin
    otpt_ymax = inpt_ymax

    otpt_arry = inpt_arry.astype(otpt_dtpe)

    assert otpt_arry.shape == (
        inpt_tcds.shape[0], inpt_ycds.shape[0], inpt_xcds.shape[0])
    #==========================================================================

    otpt_vbls = NcfVbs()

    otpt_vbls.tsct = otpt_tsct

    otpt_vbls.xlbl = inpt_xlbl
    otpt_vbls.ylbl = inpt_ylbl
    otpt_vbls.tlbl = inpt_tlbl
    otpt_vbls.vlbl = inpt_vlbl

    otpt_vbls.clwh = otpt_clwh
    otpt_vbls.clht = otpt_clht

    otpt_vbls.rows = otpt_rows
    otpt_vbls.cols = otpt_cols

    otpt_vbls.xmin = otpt_xmin
    otpt_vbls.xmax = otpt_xmax
    otpt_vbls.ymin = otpt_ymin
    otpt_vbls.ymax = otpt_ymax

    otpt_vbls.arry = otpt_arry

    otpt_vbls.xcds = inpt_xcds
    otpt_vbls.ycds = inpt_ycds

    otpt_vbls.tcds = inpt_tcds
    otpt_vbls.tuts = inpt_tuts
    otpt_vbls.tcdr = inpt_tcdr
    #==========================================================================

    return otpt_vbls


class NcfVbs:

    def __init__(self):
        return
