# -*- coding: utf-8 -*-

'''
Created on 02.09.2025

@author: Faizan-TU Munich
'''

import numpy as np
from osgeo import gdal


def sth_rts(ipt_rts_phs, otp_rtr_pth, otp_dpe):

    rts_vbs = []
    for ipt_rtr_pth in ipt_rts_phs:
        rtr_vbs = get_rtr_vbs(ipt_rtr_pth, otp_dpe)
        rts_vbs.append(rtr_vbs)
    #==========================================================================

    kk = 0  # Reference rts_vbs_idx.

    for rtr_vbs in rts_vbs:
        assert rts_vbs[kk].bdct == rtr_vbs.bdct
        assert rts_vbs[kk].clwh == rtr_vbs.clwh
        assert rts_vbs[kk].clht == rtr_vbs.clht
    #==========================================================================

    # Stitching is column-wise.
    otp_rtr_rws = 0
    otp_rtr_cls = 0
    for rtr_vbs in rts_vbs:
        otp_rtr_rws = max(rtr_vbs.rows, otp_rtr_rws)
        otp_rtr_cls += rtr_vbs.cols

    otp_ary = np.full(
        (rts_vbs[kk].bdct, otp_rtr_rws, otp_rtr_cls),
        np.nan,
        dtype=rts_vbs[kk].arry.dtype)

    ji = 0
    for rtr_vbs in rts_vbs:
        ij = rtr_vbs.rows
        jj = ji + rtr_vbs.cols

        otp_ary[:,:ij, ji:jj] = rtr_vbs.arry

        rtr_vbs.arry = None

        ji = jj
    #==========================================================================

    # Due to possibly different NaNs.
    otp_gtp = {
        np.float32: gdal.GDT_Float32,
        np.float64: gdal.GDT_Float64,
        # np.int32: gdal.GDT_Int32,
        # np.uint32: gdal.GDT_UInt32,
        # np.int16: gdal.GDT_Int16,
        # np.uint16: gdal.GDT_UInt16,
        # np.int8: gdal.GDT_Byte,
        }[otp_ary.dtype.type]

    out_dvr = rts_vbs[kk].drvr

    out_ras_hdl = out_dvr.Create(
        str(otp_rtr_pth.absolute()),
        otp_ary.shape[2],  # Columns.
        otp_ary.shape[1],  # Rows.
        otp_ary.shape[0],  # Bands.
        otp_gtp,  # dtype.
        options=['COMPRESS=LZW', 'BIGTIFF=YES'])

    assert out_ras_hdl is not None
    #==========================================================================

    for i in range(otp_ary.shape[0]):
        out_bnd = out_ras_hdl.GetRasterBand(i + 1)
        out_bnd.WriteArray(otp_ary[i,:,:])
        out_bnd.SetNoDataValue(np.nan)
    #==========================================================================

    out_ras_hdl.SetGeoTransform(
        [rts_vbs[kk].xmin,
         +rts_vbs[kk].clwh,
         0,
         rts_vbs[kk].ymax,
         0,
         -rts_vbs[kk].clht])

    out_ras_hdl.SetProjection(rts_vbs[kk].proj)

    out_ras_hdl = None
    return


def get_rtr_vbs(inpt_path, otpt_dtpe):

    assert inpt_path.exists()

    inpt_hndl = gdal.Open(str(inpt_path.absolute()))

    assert inpt_hndl is not None

    inpt_rows = inpt_hndl.RasterYSize
    inpt_cols = inpt_hndl.RasterXSize

    inpt_gtfm = inpt_hndl.GetGeoTransform()

    inpt_xmin = +inpt_gtfm[0]
    inpt_ymax = +inpt_gtfm[3]

    inpt_clwh = +inpt_gtfm[1]
    inpt_clht = -inpt_gtfm[5]

    inpt_xmax = inpt_xmin + (inpt_cols * inpt_clwh)
    inpt_ymin = inpt_ymax - (inpt_rows * inpt_clht)

    assert inpt_xmax > inpt_xmin
    assert inpt_ymax > inpt_ymin

    inpt_bdct = inpt_hndl.RasterCount

    inpt_arry = np.full(
        (inpt_bdct, inpt_rows, inpt_cols),
        np.nan,
        dtype=np.float64)

    for bnd_idx in range(1, inpt_bdct + 1):

        inpt_nodv_band = inpt_hndl.GetRasterBand(
            bnd_idx).GetNoDataValue()

        inpt_arry_band = np.array(
            inpt_hndl.GetRasterBand(bnd_idx).ReadAsArray(),
            dtype=np.float64)

        if ((inpt_nodv_band is not None) and
            (not np.isnan(inpt_nodv_band))):

            inpt_arry_band[
                np.isclose(inpt_arry_band, inpt_nodv_band)] = np.nan

        inpt_arry[bnd_idx - 1,:,:] = inpt_arry_band

    inpt_proj = inpt_hndl.GetProjection()
    inpt_drvr = inpt_hndl.GetDriver()

    # inpt_mask = np.isfinite(inpt_arry).any(axis=0)

    inpt_hndl = None
    #==========================================================================

    # inpt_rows_fint_idxs = np.where(inpt_mask.any(axis=1))[0]
    #
    # otpt_uppr_fint_indx = inpt_rows_fint_idxs[+0] + 0
    # otpt_botm_fint_indx = inpt_rows_fint_idxs[-1] + 1
    #
    # assert otpt_uppr_fint_indx < otpt_botm_fint_indx
    #
    # inpt_cols_fint_idxs = np.where(inpt_mask.any(axis=0))[0]
    #
    # otpt_left_fint_indx = inpt_cols_fint_idxs[+0] + 0
    # otpt_rght_fint_indx = inpt_cols_fint_idxs[-1] + 1
    #
    # assert otpt_left_fint_indx < otpt_rght_fint_indx
    #==========================================================================

    otpt_bdct = inpt_bdct

    otpt_clwh = inpt_clwh
    otpt_clht = inpt_clht

    otpt_cols = inpt_cols
    otpt_rows = inpt_rows

    otpt_xmin = inpt_xmin
    otpt_xmax = inpt_xmax

    otpt_ymin = inpt_ymin
    otpt_ymax = inpt_ymax

    otpt_arry = inpt_arry.astype(otpt_dtpe)

    otpt_proj = inpt_proj
    otpt_drvr = inpt_drvr

    # otpt_cols = otpt_rght_fint_indx - otpt_left_fint_indx
    # otpt_rows = otpt_botm_fint_indx - otpt_uppr_fint_indx

    # otpt_xmin = inpt_xmin + (otpt_left_fint_indx * inpt_clwh)
    # otpt_xmax = inpt_xmin + (otpt_rght_fint_indx * inpt_clwh)
    #
    # otpt_ymin = inpt_ymax - (otpt_uppr_fint_indx * inpt_clht)
    # otpt_ymax = inpt_ymax - (otpt_botm_fint_indx * inpt_clht)

    # otpt_arry = (inpt_arry[:,
    #     otpt_uppr_fint_indx:otpt_botm_fint_indx,
    #     otpt_left_fint_indx:otpt_rght_fint_indx]).astype(otpt_dtpe)

    assert otpt_arry.shape == (
        otpt_bdct, otpt_rows, otpt_cols)

    # otpt_mask = inpt_mask[
    #     otpt_uppr_fint_indx:otpt_botm_fint_indx,
    #     otpt_left_fint_indx:otpt_rght_fint_indx].copy()
    #==========================================================================

    otpt_vbls = RtrVbs()

    otpt_vbls.bdct = otpt_bdct

    otpt_vbls.clwh = otpt_clwh
    otpt_vbls.clht = otpt_clht

    otpt_vbls.rows = otpt_rows
    otpt_vbls.cols = otpt_cols

    # otpt_vbls.urix = otpt_uppr_fint_indx
    # otpt_vbls.bmix = otpt_botm_fint_indx
    # otpt_vbls.ltix = otpt_left_fint_indx
    # otpt_vbls.rtix = otpt_rght_fint_indx

    otpt_vbls.xmin = otpt_xmin
    otpt_vbls.xmax = otpt_xmax
    otpt_vbls.ymin = otpt_ymin
    otpt_vbls.ymax = otpt_ymax

    otpt_vbls.arry = otpt_arry
    # otpt_vbls.mask = otpt_mask

    otpt_vbls.proj = otpt_proj
    otpt_vbls.drvr = otpt_drvr
    #==========================================================================

    return otpt_vbls


class RtrVbs:

    def __init__(self):
        return
