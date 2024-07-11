# -*- coding: utf-8 -*-

'''
Created on 09.07.2024

@author: Faizan-TU Munich
'''

from multiprocessing import Manager, Pool as MPPool

import numpy as np
from osgeo import ogr, gdal
from pyproj import Transformer

from ..mpg import DummyLock

from ..misc import print_sl, print_el

ogr.UseExceptions()
gdal.UseExceptions()

from .aa_ras_to_ras import ResampleRasToRas


class ResampleRasToRasClss(ResampleRasToRas):

    def __init__(self, verbose=True):

        ResampleRasToRas.__init__(self, verbose)

        self._get_wtd_arr_ras = None
        return

    def resample(self):

        assert self._inp_set_flg
        assert self._otp_set_flg

        src_crd_obj = self._get_crd_obj_ras(self._src_pth)
        dst_crd_obj = self._get_crd_obj_ras(self._dst_pth)

        src_xcs = src_crd_obj.get_x_coordinates()
        src_ycs = src_crd_obj.get_y_coordinates()

        src_xcs, src_ycs = self._get_crd_msh(src_xcs, src_ycs)

        dst_xcs = dst_crd_obj.get_x_coordinates()
        dst_ycs = dst_crd_obj.get_y_coordinates()

        dst_xcs, dst_ycs = self._get_crd_msh(dst_xcs, dst_ycs)
        #======================================================================

        if self._vb:
            print_sl()

        if self._mpg_ncs > 1:
            if self._vb: print('Initiating multiprocessing pool...')

            self._mpg_pol = MPPool(self._mpg_ncs)
            self._mpg_mmr = Manager()
            self._mpg_lck = self._mpg_mmr.Lock()
            self._mpg_dct = self._mpg_mmr.dict()

        else:
            self._mpg_lck = DummyLock()
        #======================================================================

        src_dst_tfm = Transformer.from_crs(
            self._get_ras_crs(self._src_pth),
            self._get_ras_crs(self._dst_pth),
            always_xy=True)

        if self._vb: print('Transforming source coordinates\' mesh...')

        self._tfm_msh(src_xcs, src_ycs, src_dst_tfm)

        if self._vb: print(f'Source original mesh shape: {src_xcs.shape}...')
        if self._vb: print(f'Desti. original mesh shape: {dst_xcs.shape}...')
        #======================================================================

        (src_xcs,
         src_ycs,
         src_wts_dct,
         (src_beg_row,
          src_end_row,
          src_beg_col,
          src_end_col),
         (dst_beg_row,
          dst_end_row,
          dst_beg_col,
          dst_end_col)) = self._get_wts_vrs(src_xcs, src_ycs, dst_xcs, dst_ycs)
        #======================================================================

        if self._vb: print('Reading input array...')

        dst_arr, = self._get_ras_vrs(
            self._dst_pth,
            dst_beg_row,
            dst_end_row,
            dst_beg_col,
            dst_end_col,
            1)

        src_arr, = self._get_ras_vrs(
            self._src_pth,
            src_beg_row,
            src_end_row,
            src_beg_col,
            src_end_col,
            None)
        #======================================================================

        if self._vb: print('Computing resampled array...')

        src_dst_arr = self._get_wtd_arr_ras_clss(
            src_arr, dst_arr, dst_xcs, src_wts_dct)

        if self._vb: print(
            f'Source transformed array shape: {src_dst_arr.shape}...')
        #======================================================================

        if self._vb: print('Saving resampled raster...')

        self._cvt_ras_fle(
            self._dst_pth,
            self._out_pth,
            src_dst_arr,
            dst_xcs[dst_beg_row, dst_beg_col],
            dst_ycs[dst_beg_row, dst_beg_col])

        if self._mpg_ncs > 1:
            if self._vb: print('Terminating multiprocessing pool...')

            self._mpg_pol.close()
            self._mpg_pol.join()
            self._mpg_pol = None

            self._mpg_mmr = None
            self._mpg_lck = None
            self._mpg_dct = None

        else:
            self._mpg_lck = None
        #======================================================================

        if self._vb:
            print_el()
        return

    def _get_wtd_arr_ras_clss(self, src_arr, dst_arr, dst_xcs, src_wts_dct):

        assert src_arr.shape[0] == 1, 'Only one band is allowed!'

        src_uqs = np.unique(src_arr)
        src_unq_dct = {src_uqs[i]: i for i in range(src_uqs.size)}

        src_dst_arr = np.full(
            (src_uqs.shape[0], dst_xcs.shape[0] - 1, dst_xcs.shape[1] - 1),
            np.nan,
            dtype=np.float32)

        assert dst_arr.shape[1:] == src_dst_arr.shape[1:], (
                dst_arr.shape, src_dst_arr.shape)

        for (i, j), wts_dct in src_wts_dct.items():

            if np.isnan(dst_arr[0, i, j]): continue

            src_dst_arr [:, i, j] = 0.0

            for (k, l), wht in wts_dct.items():
                if np.isnan(src_arr[0, k, l]): continue

                src_dst_arr[src_unq_dct[src_arr[0, k, l]], i, j] += wht

        return src_dst_arr
