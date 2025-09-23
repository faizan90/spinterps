# -*- coding: utf-8 -*-

'''
Created on 02.09.2025

@author: Faizan-TU Munich
'''

import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()


def sth_pts(ipt_rts_pts_phs, otp_pts_pth):

    pts_vbs = []
    for ipt_rtr_pts_phs in ipt_rts_pts_phs:
        pnt_vbs = get_pts_vbs(*ipt_rtr_pts_phs)
        pts_vbs.append(pnt_vbs)
    #==========================================================================

    kk = 0  # Reference pts_vbs_idx.

    for pnt_vbs in pts_vbs:
        assert pts_vbs[kk].bdct == pnt_vbs.bdct
        assert pts_vbs[kk].clwh == pnt_vbs.clwh
        assert pts_vbs[kk].clht == pnt_vbs.clht
    #==========================================================================

    # Stitching is column-wise.
    otp_rtr_rws = 0
    otp_rtr_cls = 0
    for pnt_vbs in pts_vbs:
        otp_rtr_rws = max(pnt_vbs.rows, otp_rtr_rws)

        pnt_vbs.dafe['col'] += otp_rtr_cls

        otp_rtr_cls += pnt_vbs.cols
    #==========================================================================

    for pnt_vbs in pts_vbs:
        pnt_vbs.dafe['X'] = (
            pts_vbs[kk].xmin + (pts_vbs[kk].clwh * pnt_vbs.dafe['col'].values))

        pnt_vbs.dafe['Y'] = (
            pts_vbs[kk].ymax - (pts_vbs[kk].clht * pnt_vbs.dafe['row'].values))
    #==========================================================================

    otp_pts_dfe = pd.concat([pnt_vbs.dafe for pnt_vbs in pts_vbs], axis=0)

    otp_pts_dfe.to_csv(otp_pts_pth.with_suffix('.csv'), sep=';')
    otp_pts_dfe.to_pickle(otp_pts_pth.with_suffix('.pkl'))
    return


def get_pts_vbs(inpt_rstr_path, inpt_pnts_path, inpt_xlbl, inpt_ylbl):

    assert inpt_rstr_path.exists()
    assert inpt_pnts_path.exists()

    inpt_hndl = gdal.Open(str(inpt_rstr_path.absolute()))

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

    inpt_bdct = inpt_hndl.RasterCount

    inpt_hndl = None
    #==========================================================================

    pnts_dafe = pd.read_csv(inpt_pnts_path, sep=';', index_col=0)

    pnts_dafe['row'] = np.nan
    pnts_dafe['col'] = np.nan

    assert np.isfinite(pnts_dafe[inpt_xlbl].values).all()
    assert np.isfinite(pnts_dafe[inpt_ylbl].values).all()

    for pont_labl in pnts_dafe.index:
        xcdt, ycdt = pnts_dafe.loc[pont_labl, [inpt_xlbl, inpt_ylbl]]

        # Don't convert to integers.
        pont_roww = (inpt_ymax - ycdt) / inpt_clht
        pont_coln = (xcdt - inpt_xmin) / inpt_clwh

        assert 0 <= pont_roww < inpt_rows
        assert 0 <= pont_coln < inpt_cols

        pnts_dafe.loc[pont_labl, 'row'] = pont_roww
        pnts_dafe.loc[pont_labl, 'col'] = pont_coln
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

    otpt_dafe = pnts_dafe.copy()
    #==========================================================================

    otpt_vbls = PntVbs()

    otpt_vbls.bdct = otpt_bdct

    otpt_vbls.clwh = otpt_clwh
    otpt_vbls.clht = otpt_clht

    otpt_vbls.rows = otpt_rows
    otpt_vbls.cols = otpt_cols

    otpt_vbls.xmin = otpt_xmin
    otpt_vbls.xmax = otpt_xmax
    otpt_vbls.ymin = otpt_ymin
    otpt_vbls.ymax = otpt_ymax

    otpt_vbls.dafe = otpt_dafe
    #==========================================================================

    return otpt_vbls


class PntVbs:

    def __init__(self):
        return
