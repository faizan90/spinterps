# -*- coding: utf-8 -*-

'''
Created on 08.07.2024

@author: Faizan-TU Munich
'''

import pickle
from pathlib import Path
from math import ceil, floor
from timeit import default_timer
from multiprocessing import Manager, Pool as MPPool

import numpy as np
from osgeo import ogr, gdal
from pyproj import Transformer
import matplotlib.pyplot as plt

from ..extract import ExtractGTiffCoords

from ..mpg import SHMARY2 as SHMARY
from ..mpg import DummyLock
# from ..mpg import SHMARGS
# from ..mpg import init_shm_arrs, fill_shm_arrs, get_shm_arr, free_shm_arrs

from ..misc import print_sl, print_el, ret_mp_idxs, get_n_cpus

ogr.UseExceptions()
gdal.UseExceptions()


class ResampleRasToRas:

    def __init__(self, verbose=True):

        assert isinstance(verbose, bool), type(verbose)

        self._vb = verbose

        # SRC is resampled to become DST.
        self._src_pth = None
        self._dst_pth = None

        self._mpg_ncs = None  # Number of processes.
        self._mpg_pol = None  # MP Pool.
        self._mpg_mmr = None  # Manager.
        self._mpg_lck = None  # Lock.
        self._mpg_dct = None  # Dictionary.

        self._src_wts_dct_sav_flg = False

        self._inp_set_flg = False
        self._otp_set_flg = False
        return

    def set_inputs(self, path_to_src, path_to_dst, n_cpus):

        if self._vb:
            print_sl()

            print('Setting resampling inputs...')

        # Source.
        assert isinstance(path_to_src, Path), type(path_to_src)
        assert path_to_src.exists(), path_to_src

        self._src_pth = path_to_src.absolute()

        # Desti.
        assert isinstance(path_to_dst, Path), type(path_to_dst)
        assert path_to_dst.exists(), path_to_dst

        self._dst_pth = path_to_dst.absolute()

        # MPG.
        if isinstance(n_cpus, str):
            assert n_cpus == 'auto', 'Invalid n_cpus!'

            n_cpus = get_n_cpus()

        else:
            assert isinstance(n_cpus, int), 'n_cpus is not an integer!'

            assert n_cpus > 0, 'Invalid n_cpus!'

        self._mpg_ncs = n_cpus
        #======================================================================

        if self._vb:
            print(f'INFO: Set the following parameters for the inputs:')

            print(f'Source path: {self._src_pth}')
            print(f'Desti. path: {self._dst_pth}')
            print(f'No. of Procs: {self._mpg_ncs}')

            print_el()

        self._inp_set_flg = True
        return

    def set_outputs(self, path_to_out):

        if self._vb:
            print_sl()

            print('Setting resampling outputs...')

        assert isinstance(path_to_out, Path), type(path_to_out)

        path_to_out = path_to_out.absolute()

        assert path_to_out.parents[0].exists(), path_to_out.parents[0]

        self._out_pth = path_to_out

        if self._vb:
            print(f'INFO: Set the following parameters for the outputs:')

            print(f'Output path: {self._out_pth}')

            print_el()

        self._otp_set_flg = True
        return

    def resample(self):

        assert self._inp_set_flg
        assert self._otp_set_flg

        src_crd_obj = self._get_crd_obj_ras(self._src_pth)

        src_xcs = src_crd_obj.get_x_coordinates()
        src_ycs = src_crd_obj.get_y_coordinates()

        src_crd_obj = None

        src_xcs, src_ycs = self._get_crd_msh(src_xcs, src_ycs)

        dst_crd_obj = self._get_crd_obj_ras(self._dst_pth)

        dst_xcs = dst_crd_obj.get_x_coordinates()
        dst_ycs = dst_crd_obj.get_y_coordinates()

        dst_xcs, dst_ycs = self._get_crd_msh(dst_xcs, dst_ycs)

        dst_crd_obj = None
        #======================================================================

        if self._vb: print_sl()

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
            always_xy=True,
            only_best=True)

        dst_src_tfm = Transformer.from_crs(
            self._get_ras_crs(self._dst_pth),
            self._get_ras_crs(self._src_pth),
            always_xy=True,
            only_best=True)

        if self._vb: print('Transforming source coordinates\' mesh...')

        src_xcs_ogl, src_ycs_ogl = src_xcs.copy(), src_ycs.copy()

        self._tfm_msh(src_xcs, src_ycs, src_dst_tfm)

        if self._vb: print(f'Source original mesh shape: {src_xcs.shape}...')
        if self._vb: print(f'Desti. original mesh shape: {dst_xcs.shape}...')
        #======================================================================

        (xcs_ply_src_ful,
         ycs_ply_src_ful) = self._get_msh_ply_cns_cds(src_xcs, src_ycs)

        if self._vb:
            print('Calculating intersection indices of both rasters...')

        ((src_beg_row,
          src_end_row,
          src_beg_col,
          src_end_col),
         (dst_beg_row,
          dst_end_row,
          dst_beg_col,
          dst_end_col)) = self._get_itt_ixs(src_xcs, src_ycs, dst_xcs, dst_ycs)
        #======================================================================

        if self._vb: print('Snipping coordinates\' mesh...')

        src_xcs, src_ycs = (
            src_xcs[src_beg_row:src_end_row, src_beg_col:src_end_col],
            src_ycs[src_beg_row:src_end_row, src_beg_col:src_end_col])

        src_xcs_ogl, src_ycs_ogl = (
            src_xcs_ogl[src_beg_row:src_end_row, src_beg_col:src_end_col],
            src_ycs_ogl[src_beg_row:src_end_row, src_beg_col:src_end_col])

        src_xmn = src_xcs.min()
        src_xmx = src_xcs.max()

        src_ymn = src_ycs.min()
        src_ymx = src_ycs.max()

        if self._vb: print(f'Source snipped mesh shape: {src_xcs.shape}...')
        #======================================================================

        (xcs_ply_dst_ful,
         ycs_ply_dst_ful) = self._get_msh_ply_cns_cds(dst_xcs, dst_ycs)

        if self._vb: print(f'Adjusting desti. bounds...')

        (dst_beg_row,
         dst_end_row,
         dst_beg_col,
         dst_end_col) = self._get_adj_dst_crd(
            src_xmn,
            src_xmx,
            src_ymn,
            src_ymx,
            dst_xcs,
            dst_ycs,
            dst_beg_row,
            dst_end_row,
            dst_beg_col,
            dst_end_col)
        #======================================================================

        dst_xcs, dst_ycs = (
            dst_xcs[dst_beg_row:dst_end_row, dst_beg_col:dst_end_col],
            dst_ycs[dst_beg_row:dst_end_row, dst_beg_col:dst_end_col])

        dst_xmn = dst_xcs.min()
        dst_xmx = dst_xcs.max()

        dst_ymn = dst_ycs.min()
        dst_ymx = dst_ycs.max()

        if self._vb: print(f'Desti. snipped mesh shape: {dst_xcs.shape}...')

        assert dst_xmn >= src_xmn, (dst_xmn, src_xmn)
        assert dst_xmx <= src_xmx, (dst_xmx, src_xmx)

        assert dst_ymn >= src_ymn, (dst_ymn, src_ymn)
        assert dst_ymx <= src_ymx, (dst_ymx, src_ymx)
        #======================================================================

        if True:  # Diagnostics.

            if self._vb:
                print('Plotting full and intersection source and desti. '
                      'polygons...')

            self._plt_src_dst_eps(
                src_xcs,
                src_ycs,
                dst_xcs,
                dst_ycs,
                xcs_ply_src_ful,
                ycs_ply_src_ful,
                xcs_ply_dst_ful,
                ycs_ply_dst_ful)
        #======================================================================

        if self._vb: print('Reading input arrays...')

        dst_arr = self._get_ras_vrs(
            self._dst_pth,
            dst_beg_row,
            dst_end_row,
            dst_beg_col,
            dst_end_col,
            1)[0]

        dst_arr_shp = dst_arr.shape

        src_arr, = self._get_ras_vrs(
            self._src_pth,
            src_beg_row,
            src_end_row,
            src_beg_col,
            src_end_col,
            None)
        #======================================================================

        if self._vb: print('Initiating resampled array...')

        src_dst_arr = np.full(
            (src_arr.shape[0], dst_xcs.shape[0] - 1, dst_xcs.shape[1] - 1),
            np.nan,
            dtype=np.float32)

        assert dst_arr_shp[1:] == src_dst_arr.shape[1:], (
                dst_arr_shp, src_dst_arr.shape)

        if self._vb: print(
            f'Source transformed array shape: {src_dst_arr.shape}...')
        #======================================================================

        if self._vb: print('Getting source adjustment variables...')

        (src_bfr_rws,
         src_bfr_cns,
         src_xmn_min,
         src_ymx_max,
         src_cel_hgt,
         src_cel_wdh) = self._get_src_adj_vrs(
            dst_xcs, dst_ycs, src_xcs_ogl, src_ycs_ogl)

        _ = src_bfr_rws, src_bfr_cns, src_xmn_min, src_ymx_max, src_cel_wdh
        #======================================================================

        if self._vb: print('Computing resampled array...')

        ags = RTRARGS()

        # Constants and other small objects go here. The rest in _cpt_wts_vrs.
        ags.rsm_flg = True
        ags.vbe_flg = self._vb
        ags.src_cel_hgt = src_cel_hgt
        ags.dst_src_tfm = dst_src_tfm
        ags.src_wts_dct_sav_flg = self._src_wts_dct_sav_flg

        self._cpt_wts_vrs(
            ags,
            src_xcs,
            src_ycs,
            dst_xcs,
            dst_ycs,
            src_arr,
            dst_arr,
            src_xcs_ogl,
            src_ycs_ogl,
            src_dst_arr)

        ags = None
        src_arr = None
        src_xcs = src_xcs_ogl = None
        src_ycs = src_ycs_ogl = None
        #======================================================================

        if self._vb: print('Saving resampled raster...')

        self._cvt_ras_fle(
            self._dst_pth,
            self._out_pth,
            src_dst_arr,
            dst_xcs[0, 0],
            dst_ycs[0, 0])
        #======================================================================

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

        if self._vb: print_el()
        return

    def _get_src_adj_vrs(
            self, dst_xcs, dst_ycs, src_xcs_ogl, src_ycs_ogl):

        dst_ycs_dfs = dst_ycs[:-1,:] - dst_ycs[+1:,:]

        dst_cle_cdy = np.isclose(
            dst_ycs_dfs, dst_ycs[0, 0] - dst_ycs[1, 0]).all()

        dst_cel_hgt = np.abs(dst_ycs_dfs).max()

        dst_ycs_dfs = None

        dst_xcs_dfs = dst_xcs[:, 1:] - dst_xcs[:,:-1]

        dst_cle_cdx = np.isclose(
            dst_xcs_dfs, dst_xcs[0, 1] - dst_xcs[0, 0]).all()

        dst_cel_wdh = np.abs(dst_xcs_dfs).max()

        dst_xcs_dfs = None

        if dst_cle_cdy and dst_cle_cdx:
            if self._vb: print('Desti. cells are constant in size.')
        #======================================================================

        src_ycs_dfs = src_ycs_ogl[:-1,:] - src_ycs_ogl[+1:,:]

        src_cle_cdy = np.isclose(
            src_ycs_dfs, src_ycs_ogl[0, 0] - src_ycs_ogl[1, 0]).all()

        # src_cel_hgt = np.abs(src_ycs_dfs).max()
        src_cel_hgt = np.abs(src_ycs_dfs).min()

        assert src_cel_hgt > 0, src_cel_hgt

        src_ycs_dfs = None

        src_xcs_dfs = src_xcs_ogl[:, 1:] - src_xcs_ogl[:,:-1]

        src_cle_cdx = np.isclose(
            src_xcs_dfs, src_xcs_ogl[0, 1] - src_xcs_ogl[0, 0]).all()

        # src_cel_wdh = np.abs(src_xcs_dfs).max()
        src_cel_wdh = np.abs(src_xcs_dfs).min()

        assert src_cel_wdh > 0

        src_xcs_dfs = None

        if src_cle_cdy and src_cle_cdx:
            if self._vb: print('Source cells are constant in size.')
        #======================================================================

        src_rws_hgt_rel_crt = (src_cel_hgt / dst_cel_hgt)
        src_rws_rtn_rel_crt = (
            max(np.abs(src_ycs_ogl[+0, +0] - src_ycs_ogl[+0, -1]),
                np.abs(src_ycs_ogl[-1, +0] - src_ycs_ogl[-1, -1])) /
            src_cel_hgt)

        src_cns_wdh_rel_crt = (src_cel_wdh / dst_cel_wdh)
        src_cns_rtn_rel_crt = (
            max(np.abs(src_xcs_ogl[-1, +0] - src_xcs_ogl[+0, +0]),
                np.abs(src_xcs_ogl[-1, -1] - src_xcs_ogl[+0, -1])) /
            src_cel_wdh)

        # Buffer should be then taken as relative to the position i.e.,
        # divided by the number of rows and columns of xcs or ycs.
        src_bfr_rws = max(
            0, int(ceil(src_rws_hgt_rel_crt + src_rws_rtn_rel_crt)))

        src_bfr_cns = max(
            0, int(ceil(src_cns_wdh_rel_crt + src_cns_rtn_rel_crt)))

        # Cds may be 2D but have constant cell sizes.
        # Cell selection for intersection depends on this assumption.
        assert np.isclose(src_rws_rtn_rel_crt, 0), src_rws_rtn_rel_crt
        assert np.isclose(src_cns_rtn_rel_crt, 0), src_cns_rtn_rel_crt

        # To compensate for grid rotation temporarily.
        # This makes it take longer.
        # Ideally, gen_wts_dct_frm_crd_and_ply_sgl should take care of this.
        # src_bfr_rws *= 2
        # src_bfr_cns *= 2

        if False:
            src_xmn_min = src_xcs_ogl[int(src_xcs_ogl.shape[0] * 0.5), 0].min()
            src_ymx_max = src_ycs_ogl[0, int(src_xcs_ogl.shape[1] * 0.5)].max()

        else:
            src_xmn_min = src_xcs_ogl[0:, 0].min()
            src_ymx_max = src_ycs_ogl[0, 0:].max()

        return (
            src_bfr_rws,
            src_bfr_cns,
            src_xmn_min,
            src_ymx_max,
            src_cel_hgt,
            src_cel_wdh)

    def _get_adj_dst_crd(
            self,
            src_xmn,
            src_xmx,
            src_ymn,
            src_ymx,
            dst_xcs,
            dst_ycs,
            dst_beg_row,
            dst_end_row,
            dst_beg_col,
            dst_end_col):

        while dst_xcs[
            dst_beg_row:dst_end_row,
            dst_beg_col:dst_end_col].min() < src_xmn:

            dst_beg_col += 1

            if dst_beg_col >= dst_end_col: break

        dst_beg_col = min(dst_end_col, dst_beg_col)

        while dst_xcs[
            dst_beg_row:dst_end_row,
            dst_beg_col:dst_end_col].max() > src_xmx:

            dst_end_col -= 1

            if dst_end_col <= dst_beg_col: break

        dst_end_col = max(dst_end_col, dst_beg_col)

        while dst_ycs[
            dst_beg_row:dst_end_row,
            dst_beg_col:dst_end_col].min() < src_ymn:

            dst_end_row -= 1

            if dst_end_row <= dst_beg_row: break

        dst_end_row = max(dst_beg_row, dst_end_row)

        while dst_ycs[
            dst_beg_row:dst_end_row,
            dst_beg_col:dst_end_col].max() > src_ymx:

            dst_beg_row += 1

            if dst_beg_row >= dst_end_row: break

        dst_beg_row = min(dst_end_row, dst_beg_row)

        return (
            dst_beg_row,
            dst_end_row,
            dst_beg_col,
            dst_end_col)

    def _get_itt_ixs(self, src_xcs, src_ycs, dst_xcs, dst_ycs):

        src_ext_ply = self._get_ext_ply_frm_crd(src_xcs, src_ycs)
        dst_ext_ply = self._get_ext_ply_frm_crd(dst_xcs, dst_ycs)

        src_dst_ply_itt = src_ext_ply.Intersection(dst_ext_ply)
        assert src_dst_ply_itt.Area() > 0, src_dst_ply_itt.Area()

        if False:  # Diagnostics.

            dst_xmn, dst_xmx, dst_ymn, dst_ymx = dst_ext_ply.GetEnvelope()

            xcs_ply_dst = [dst_xmn, dst_xmn, dst_xmx, dst_xmx, dst_xmn]
            ycs_ply_dst = [dst_ymn, dst_ymx, dst_ymx, dst_ymn, dst_ymn]
            #==================================================================

            src_xmn, src_xmx, src_ymn, src_ymx = src_ext_ply.GetEnvelope()

            xcs_ply_src = [src_xmn, src_xmn, src_xmx, src_xmx, src_xmn]
            ycs_ply_src = [src_ymn, src_ymx, src_ymx, src_ymn, src_ymn]
            #==================================================================

            itt_xmn, itt_xmx, itt_ymn, itt_ymx = src_dst_ply_itt.GetEnvelope()

            xcs_ply_itt = [itt_xmn, itt_xmn, itt_xmx, itt_xmx, itt_xmn]
            ycs_ply_itt = [itt_ymn, itt_ymx, itt_ymx, itt_ymn, itt_ymn]
            #==================================================================

            fig = plt.figure()

            plt.gca().set_aspect('equal')

            # SRC.
            plt.plot(
                xcs_ply_src,
                ycs_ply_src,
                c='C0',
                lw=4.0,
                alpha=0.85,
                zorder=1,
                label='SRC')

            # DST.
            plt.plot(
                xcs_ply_dst,
                ycs_ply_dst,
                c='C1',
                lw=2.0,
                alpha=0.85,
                zorder=2,
                label='DST')

            # ITT.
            plt.plot(
                xcs_ply_itt,
                ycs_ply_itt,
                c='k',
                lw=1,
                alpha=0.85,
                zorder=5,
                label='ITT')

            plt.title('5')

            plt.legend()

            plt.show(block=True)

            plt.close(fig)
        #======================================================================

        (src_beg_row,
         src_end_row,
         src_beg_col,
         src_end_col) = self._get_ply_grd_ixs(
            src_dst_ply_itt, src_xcs, src_ycs)

        (dst_beg_row,
         dst_end_row,
         dst_beg_col,
         dst_end_col) = self._get_ply_grd_ixs(
            src_dst_ply_itt, dst_xcs, dst_ycs)

        return (
            (src_beg_row,
             src_end_row,
             src_beg_col,
             src_end_col),
            (dst_beg_row,
             dst_end_row,
             dst_beg_col,
             dst_end_col))

    def _cvt_ras_fle(
            self,
            ref_pth,
            out_pth,
            ras_arr,
            ref_xmn,
            ref_ymx,
            bnd_mda=None):

        ref_ras = gdal.Open(str(ref_pth), 0)
        assert ref_ras is not None

        assert ras_arr.ndim == 3, ras_arr.ndim

        ref_gtp = {
            np.float32: gdal.GDT_Float32,
            np.float64: gdal.GDT_Float64,
            np.int32: gdal.GDT_Int32,
            np.uint32: gdal.GDT_UInt32,
            np.int16: gdal.GDT_Int16,
            np.uint16: gdal.GDT_UInt16,
            np.int8: gdal.GDT_Byte,
            }[ras_arr.dtype.type]
        #======================================================================

        ref_gtm = ref_ras.GetGeoTransform()

        out_dvr = ref_ras.GetDriver()

        out_ras = out_dvr.Create(
            out_pth,
            ras_arr.shape[2],  # Columns.
            ras_arr.shape[1],  # Rows.
            ras_arr.shape[0],  # Bands.
            ref_gtp,  # dtype.
            options=['COMPRESS=LZW'])

        assert out_ras is not None
        #======================================================================

        for i in range(ras_arr.shape[0]):
            out_bnd = out_ras.GetRasterBand(i + 1)
            out_bnd.WriteArray(ras_arr[i,:,:])
            out_bnd.SetNoDataValue(np.nan)

            if bnd_mda is not None: out_bnd.SetMetadata(bnd_mda[i])
        #======================================================================

        out_ras.SetGeoTransform(
            [ref_xmn, ref_gtm[1], 0, ref_ymx, 0, ref_gtm[5]])

        out_ras.SetProjection(ref_ras.GetProjection())

        out_ras = ref_ras = None
        return

    def _get_ras_vrs(
            self, inp_pth, beg_row, end_row, beg_col, end_col, bnd_cnt):

        ras = gdal.Open(str(inp_pth))

        assert ras is not None

        if bnd_cnt is None:
            bnd_cnt = ras.RasterCount

        else:
            assert bnd_cnt >= ras.RasterCount
        #======================================================================

        arr = np.empty(
            (bnd_cnt,
             end_row - beg_row - 1,
             end_col - beg_col - 1),
            dtype=np.float32)

        for i in range(bnd_cnt):

            bnd_ndv = ras.GetRasterBand(i + 1).GetNoDataValue()

            bnd = ras.GetRasterBand(i + 1)

            bnd_arr = bnd.ReadAsArray(
                xoff=beg_col,
                yoff=beg_row,
                win_xsize=end_col - beg_col - 1,
                win_ysize=end_row - beg_row - 1,)

            bnd_arr = np.array(bnd_arr, dtype=np.float32)

            if not np.isnan(bnd_ndv):
                bnd_arr[np.isclose(bnd_arr, bnd_ndv)] = np.nan

            arr[i] = bnd_arr

        ras = None
        return (arr,)

    def _cpt_wts_vrs(
            self,
            ags,
            src_xcs,
            src_ycs,
            dst_xcs,
            dst_ycs,
            src_arr,
            dst_arr,
            src_xcs_ogl,
            src_ycs_ogl,
            src_dst_arr):

        '''
        Coordinates are 2D of cell corners.

        src and dst have same crs.

        This function is adapted to handle all sorts of transformation
        defined till now.
        '''

        # Range will be between 0 and 5.
        rsp_fgs = np.zeros(dst_arr.shape[1:], dtype=np.int8)

        # NaN flags for skipping cells that are all not finite.
        # Should make the computations speed up.
        nan_fgs = np.zeros(
            (src_ycs.shape[0] - 1, src_ycs.shape[1] - 1), dtype=np.int8)

        if not ags.ncf_flg:

            assert any([ags.rsm_flg, ags.css_flg])

            nan_fgs[(~np.isfinite(src_arr)).all(axis=0)] = 1

        else:
            assert all([not ags.rsm_flg, not ags.css_flg])

            # Due to np.int8 of nan_fgs.
            assert len(src_arr) < 127, len(src_arr)

            for key in src_arr:
                nan_fgs[(~np.isfinite(src_arr[key])).all(axis=0)] += 1

            nan_fgs[nan_fgs < len(src_arr)] = 0
            nan_fgs[nan_fgs == len(src_arr)] = 1

        # Large objects have to be assigned here.
        if self._mpg_pol is None:

            ags.mpg_flg = False

            ags.src_xcs = src_xcs
            ags.src_ycs = src_ycs
            ags.dst_xcs = dst_xcs
            ags.dst_ycs = dst_ycs

            ags.src_xcs_ogl = src_xcs_ogl
            ags.src_ycs_ogl = src_ycs_ogl

            ags.rsp_fgs = rsp_fgs
            ags.nan_fgs = nan_fgs

            if not ags.ncf_flg:

                assert any([ags.rsm_flg, ags.css_flg])

                ags.src_arr = src_arr
                ags.dst_arr = dst_arr
                ags.src_dst_arr = src_dst_arr

            else:
                assert all([not ags.rsm_flg, not ags.css_flg])

                setattr(ags, 'ncf_vrs', list(src_arr))

                setattr(ags, 'dst_arr', dst_arr)

                setattr(ags, 'nan_fgs', nan_fgs)

                for key in src_arr:
                    setattr(ags, f'src_arr__{key}', src_arr[key])
                    setattr(ags, f'src_dst_arr__{key}', src_dst_arr[key])

            btr = default_timer()

            ress = [gen_wts_dct_frm_crd_and_ply_sgl(
                (0, dst_xcs.shape[0] - 1, ags))]

            etr = default_timer()

        else:
            ags.mpg_flg = True

            # shm_ags = SHMARGS()
            # shm_arr_dct = {}
            #
            # shm_arr_dct['src_xcs'] = src_xcs
            # shm_arr_dct['src_ycs'] = src_ycs
            # shm_arr_dct['dst_xcs'] = dst_xcs
            # shm_arr_dct['dst_ycs'] = dst_ycs
            #
            # shm_arr_dct['src_xcs_ogl'] = src_xcs_ogl
            # shm_arr_dct['src_ycs_ogl'] = src_ycs_ogl
            #
            # shm_arr_dct['rsp_fgs'] = rsp_fgs
            # shm_arr_dct['nan_fgs'] = nan_fgs

            ags.src_xcs = SHMARY.frm_npy_ary(src_xcs)
            ags.src_ycs = SHMARY.frm_npy_ary(src_ycs)

            ags.dst_xcs = SHMARY.frm_npy_ary(dst_xcs)
            ags.dst_ycs = SHMARY.frm_npy_ary(dst_ycs)

            ags.src_xcs_ogl = SHMARY.frm_npy_ary(src_xcs_ogl)
            ags.src_ycs_ogl = SHMARY.frm_npy_ary(src_ycs_ogl)

            ags.rsp_fgs = SHMARY.frm_npy_ary(rsp_fgs)
            ags.nan_fgs = SHMARY.frm_npy_ary(nan_fgs)

            if not ags.ncf_flg:

                assert any([ags.rsm_flg, ags.css_flg])

                # shm_arr_dct['src_arr'] = src_arr
                # shm_arr_dct['dst_arr'] = dst_arr
                # shm_arr_dct['src_dst_arr'] = src_dst_arr

                ags.src_arr = SHMARY.frm_npy_ary(src_arr)
                ags.dst_arr = SHMARY.frm_npy_ary(dst_arr)
                ags.src_dst_arr = SHMARY.frm_npy_ary(src_dst_arr)

            else:
                assert all([not ags.rsm_flg, not ags.css_flg])

                setattr(ags, 'ncf_vrs', list(src_arr))
                # setattr(ags, 'dst_arr', dst_arr)

                ags.dst_arr = SHMARY.frm_npy_ary(dst_arr)

                # ncf_shm_nms = []
                for key in src_arr:
                    # shm_arr_dct[f'src_arr__{key}'] = src_arr[key]
                    # shm_arr_dct[f'src_dst_arr__{key}'] = src_dst_arr[key]

                    # ncf_shm_nms.append(f'src_dst_arr__{key}')

                    setattr(
                        ags,
                        f'src_arr__{key}',
                        SHMARY.frm_npy_ary(src_arr[key]))

                    setattr(
                        ags,
                        f'src_dst_arr__{key}',
                        SHMARY.frm_npy_ary(src_dst_arr[key]))
            #==================================================================

            # init_shm_arrs(ags, shm_ags, shm_arr_dct)

            mpg_ixs = ret_mp_idxs(
                dst_xcs.shape[0] - 1, self._mpg_pol._processes * 2)

            mpg_ags = (
                (i, j, ags) for i, j in zip(mpg_ixs[:-1], mpg_ixs[+1:]))

            btr = default_timer()

            ress = self._mpg_pol.map(
                gen_wts_dct_frm_crd_and_ply_sgl, mpg_ags, chunksize=1)

            ress = list(ress)

            etr = default_timer()
            #==================================================================

            # for key, val in shm_arr_dct.items():
            #
            #     if key == 'rsp_fgs':
            #         val[:] = get_shm_arr(ags, key)
            #         continue
            #
            #     if not ags.ncf_flg:
            #
            #         assert any([ags.rsm_flg, ags.css_flg])
            #
            #         if key not in ['src_dst_arr', ]: continue
            #
            #         if hasattr(ags, key): continue
            #
            #         val[:] = get_shm_arr(ags, key)
            #
            #     else:
            #         assert not any([ags.rsm_flg, ags.css_flg])
            #
            #         if key not in ncf_shm_nms: continue
            #
            #         ky1, ky2 = key.rsplit('__', 1)
            #
            #         locals()[ky1][ky2][:] = get_shm_arr(ags, key)
            #
            # free_shm_arrs(shm_ags)

            ags.src_xcs.close(); ags.src_xcs.unlink()
            ags.src_ycs.close(); ags.src_ycs.unlink()

            ags.dst_xcs.close(); ags.dst_xcs.unlink()
            ags.dst_ycs.close(); ags.dst_ycs.unlink()

            ags.src_xcs_ogl.close(); ags.src_xcs_ogl.unlink()
            ags.src_ycs_ogl.close(); ags.src_ycs_ogl.unlink()

            ags.rsp_fgs.close(); ags.rsp_fgs.unlink()
            ags.nan_fgs.close(); ags.nan_fgs.unlink()

            if not ags.ncf_flg:

                assert any([ags.rsm_flg, ags.css_flg])

                ags.src_arr.close(); ags.src_arr.unlink()
                ags.dst_arr.close(); ags.dst_arr.unlink()

                src_dst_arr[:] = ags.src_dst_arr
                ags.src_dst_arr.close(); ags.src_dst_arr.unlink()

            else:
                assert all([not ags.rsm_flg, not ags.css_flg])

                ags.dst_arr.close(); ags.dst_arr.unlink()

                for key in src_arr:
                    getattr(ags, f'src_arr__{key}').close()
                    getattr(ags, f'src_arr__{key}').unlink()

                    src_dst_arr[key][:] = getattr(ags, f'src_dst_arr__{key}')

                    getattr(ags, f'src_dst_arr__{key}').close()
                    getattr(ags, f'src_dst_arr__{key}').unlink()
            #==================================================================

        if self._vb: print(
            f'Weights resampling took: {etr - btr:0.1f} seconds.')

        # Hit and miss statistics.
        hit_ctr_ttl = 0
        hit_ctr_mis = 0
        for res in ress:
            hit_ctr_ttl += res[1]
            hit_ctr_mis += res[2]

        print(
            f'TTL: {hit_ctr_ttl}, '
            f'SCS_RATE: {(hit_ctr_ttl - hit_ctr_mis) / hit_ctr_ttl:0.1%}',
            f'MIS_RATE: {hit_ctr_mis / hit_ctr_ttl:0.1%}')

        if self._src_wts_dct_sav_flg:
            src_wts_dct = {}
            for res in ress: src_wts_dct.update(res[0])

            ress = None

        self._prc_rsp_fgs(rsp_fgs, dst_xcs, dst_ycs)
        #======================================================================

        if self._src_wts_dct_sav_flg:
            src_dct_pth = self._out_pth.with_name(
                f'{self._out_pth.stem}_src_wts_dct.pkl')

            with open(src_dct_pth, 'wb+') as pkl_hdl:
                pickle.dump(src_wts_dct, pkl_hdl)

        return

    def _prc_rsp_fgs(self, rsp_fgs, xcs, ycs):

        # Resampling flags. Make changes in _prc_rsp_fgs accordingly.
        # CODES: 0 = Unprocessed.
        #        1 = Processed.
        #        2 = Skipped because either intersecting rows or columns
        #            were incorrect.
        #        3 = Skipped because the intersecting area between source and
        #            destination cells was not equal.
        #        4 = Skipped because destination was NaN.
        #        5 = Source was NaN (at least one band/dataset).

        cds_dct = {
            0: 'Unprocessed',
            1: 'Processed correctly',
            2: 'Row/Column issue',
            3: 'Intersection area issue',
            4: 'Desti. was NaN',
            5: 'Source was NaN (at least one band/dataset)',
            }
        #======================================================================

        if self._vb:

            print('')
            print('Frequencies of resampling flags...')

            rsp_fgs_vls, rsp_fgs_fqs = np.unique(rsp_fgs, return_counts=True)

            ttl_cnt = float(rsp_fgs.size)
            for vle, fqy in zip(rsp_fgs_vls, rsp_fgs_fqs):

                try:
                    print(f'{cds_dct[vle]}: {fqy} ({fqy / ttl_cnt:.2%})')

                except KeyError:
                    print(f'UNKNOWN {vle}: {fqy} ({fqy / ttl_cnt:.2%})')

            print('')
        #======================================================================

        # ref_gtp = {
        #     np.float32: gdal.GDT_Float32,
        #     np.float64: gdal.GDT_Float64,
        #     np.int32: gdal.GDT_Int32,
        #     np.uint32: gdal.GDT_UInt32,
        #     np.int16: gdal.GDT_Int16,
        #     np.uint16: gdal.GDT_UInt16,
        #     np.int8: gdal.GDT_Byte,
        #     }[ras_arr.dtype.type]
        #======================================================================

        out_dvr = gdal.GetDriverByName('GTiff')

        otp_pth = self._out_pth.absolute()
        otp_pth = otp_pth.with_name(f'{self._out_pth.stem}__rsp_fgs.tif')

        out_ras = out_dvr.Create(
            str(otp_pth.absolute()),
            rsp_fgs.shape[1],  # Columns.
            rsp_fgs.shape[0],  # Rows.
            1,  # Bands.
            gdal.GDT_Float64,  # dtype.
            options=['COMPRESS=LZW'])

        assert out_ras is not None
        #======================================================================

        out_bnd = out_ras.GetRasterBand(1)
        out_bnd.WriteArray(rsp_fgs)
        # out_bnd.SetNoDataValue() # No value!
        #======================================================================

        ref_ras_hdl = gdal.Open(str(self._dst_pth))

        assert ref_ras_hdl is not None, self._dst_pth

        ref_ras_gfm = ref_ras_hdl.GetGeoTransform()

        ref_ras_gfm = list(ref_ras_gfm)

        ref_ras_gfm[0] = xcs[0, 0]
        ref_ras_gfm[3] = ycs[0, 0]

        ref_ras_crs = ref_ras_hdl.GetProjection()

        ref_ras_hdl = None
        #======================================================================

        out_ras.SetGeoTransform(ref_ras_gfm)
        out_ras.SetProjection(ref_ras_crs)

        out_ras = None
        return

    def _get_ply_grd_ixs(self, ply, xcs, ycs):

        assert xcs[0, +0] < xcs[0, -1], (xcs[0, +0], xcs[0, -1])
        assert ycs[+0, 0] > ycs[-1, 0], (ycs[+0, 0] > ycs[-1, 0])

        xcs_min, xcs_max, ycs_min, ycs_max = ply.GetEnvelope()
        #======================================================================

        # Cells not in ROI should be True.
        xcs_min_ixs = (xcs < xcs_min) | np.isclose(xcs, xcs_min)
        xcs_max_ixs = (xcs > xcs_max) | np.isclose(xcs, xcs_max)

        ycs_min_ixs = (ycs < ycs_min) | np.isclose(ycs, ycs_min)
        ycs_max_ixs = (ycs > ycs_max) | np.isclose(ycs, ycs_max)

        row_col_ixs = (
            (~(xcs_min_ixs | xcs_max_ixs)) & (~(ycs_max_ixs | ycs_min_ixs)))

        # Upper-left corner.
        xmn_ymx_row = np.where(row_col_ixs.any(axis=1))[0].min()
        xmn_ymx_col = np.where(row_col_ixs.any(axis=0))[0].min()

        # Lower-left corner.
        xmn_ymn_row = np.where(row_col_ixs.any(axis=1))[0].max()
        xmn_ymn_col = np.where(row_col_ixs.any(axis=0))[0].min()

        # Upper-right corner.
        xmx_ymx_row = np.where(row_col_ixs.any(axis=1))[0].min()
        xmx_ymx_col = np.where(row_col_ixs.any(axis=0))[0].max()

        # Lower-right corner.
        xmx_ymn_row = np.where(row_col_ixs.any(axis=1))[0].max()
        xmx_ymn_col = np.where(row_col_ixs.any(axis=0))[0].max()
        #======================================================================

        beg_row = min(
            [xmn_ymn_row,
             xmn_ymx_row,
             xmx_ymn_row,
             xmx_ymx_row])

        beg_row = max(0, beg_row)

        end_row = max(
            [xmn_ymn_row,
             xmn_ymx_row,
             xmx_ymn_row,
             xmx_ymx_row]) + 1

        beg_col = min(
            [xmn_ymn_col,
             xmn_ymx_col,
             xmx_ymn_col,
             xmx_ymx_col])

        beg_col = max(0, beg_col)

        end_col = max(
            [xmn_ymn_col,
             xmn_ymx_col,
             xmx_ymn_col,
             xmx_ymx_col]) + 1
        #======================================================================

        if False:  # Diagnostics.

            fig = plt.figure()

            plt.gca().set_aspect('equal')

            xcs_ply = [xcs_min, xcs_min, xcs_max, xcs_max, xcs_min]
            ycs_ply = [ycs_min, ycs_max, ycs_max, ycs_min, ycs_min]

            plt.plot(xcs_ply, ycs_ply, c='cyan', alpha=0.85, zorder=10)

            plt.pcolormesh(
                xcs[beg_row:end_row, beg_col:end_col],
                ycs[beg_row:end_row, beg_col:end_col],
                row_col_ixs[beg_row:end_row - 1, beg_col:end_col - 1],)

            plt.pcolormesh(xcs, ycs, row_col_ixs[:-1,:-1],)

            plt.show()

            plt.close(fig)
        #======================================================================

        # Take care of rotation and small adjustments.

        # Beg col.
        while xcs[beg_row, beg_col:end_col].min() > xcs_min:

            beg_col -= 1

            if beg_col <= 0: break

        beg_col = max(0, beg_col)

        while xcs[end_row - 1, beg_col:end_col].min() > xcs_min:

            beg_col -= 1

            if beg_col <= 0: break

        beg_col = max(0, beg_col)

        # End col.
        while xcs[beg_row, beg_col:end_col].max() <= xcs_max:

            end_col += 1

            if end_col >= xcs.shape[1]: break

        end_col = min(xcs.shape[1], end_col)

        while xcs[end_row - 1, beg_col:end_col].max() <= xcs_max:

            end_col += 1

            if end_col >= xcs.shape[1]: break

        end_col = min(xcs.shape[1], end_col)

        # End row.
        while ycs[beg_row:end_row, beg_col].min() >= ycs_min:

            end_row += 1

            if end_row >= xcs.shape[0]: break

        end_row = min(xcs.shape[0], end_row)

        while ycs[beg_row:end_row, end_col - 1].min() >= ycs_min:

            end_row += 1

            if end_row >= xcs.shape[0]: break

        end_row = min(xcs.shape[0], end_row)

        # Beg row.
        while ycs[beg_row:end_row, beg_col].max() < ycs_max:

            beg_row -= 1

            if beg_row <= 0: break

        beg_row = max(0, beg_row)

        while ycs[beg_row:end_row, end_col - 1].max() < ycs_max:

            beg_row -= 1

            if beg_row <= 0: break

        beg_row = max(0, beg_row)
        #======================================================================

        # Sanity check. Should always pass.
        assert xcs[beg_row:end_row, beg_col:end_col].size >= 2

        # X bounds checks.
        assert xcs[beg_row:end_row, beg_col:end_col].min() <= xcs_min, (
            xcs[beg_row:end_row, beg_col:end_col].min(), xcs_min)

        assert xcs[beg_row:end_row, beg_col:end_col].max() >= xcs_max, (
            xcs[beg_row:end_row, beg_col:end_col].max(), xcs_max)

        # Y bounds checks.
        assert ycs[beg_row:end_row, beg_col:end_col].min() <= ycs_min, (
            ycs[beg_row:end_row, beg_col:end_col].min(), ycs_min)

        assert ycs[beg_row:end_row, beg_col:end_col].max() >= ycs_max, (
            ycs[beg_row:end_row, beg_col:end_col].max(), ycs_max)

        # This works only if y goes low when rows go high.
        if False:
            # X limit bounds checks.
            assert not (xcs[beg_row:end_row, beg_col + 1:end_col].min() <=
                        xcs_min), (
                xcs[beg_row:end_row, beg_col:end_col].min(), xcs_min)

            assert not (xcs[beg_row:end_row, beg_col:end_col - 1].max() >=
                        xcs_max), (
                xcs[beg_row:end_row, beg_col:end_col].max(), xcs_max)

            # Y limit bounds checks.
            assert not (ycs[beg_row:end_row - 1, beg_col:end_col].min() <=
                        ycs_min), (
                ycs[beg_row:end_row, beg_col:end_col].min(), ycs_min)

            assert not (ycs[beg_row + 1:end_row, beg_col:end_col].max() >=
                        ycs_max), (
                ycs[beg_row:end_row, beg_col:end_col].max(), ycs_max)
        #======================================================================

        return beg_row, end_row, beg_col, end_col

    def _get_ext_ply_frm_crd(self, xcs, ycs):

        xcs_min, xcs_max, ycs_min, ycs_max = (
            xcs.min(), xcs.max(), ycs.min(), ycs.max())

        rng = ogr.Geometry(ogr.wkbLinearRing)

        rng.AddPoint(xcs_min, ycs_min)
        rng.AddPoint(xcs_max, ycs_min)
        rng.AddPoint(xcs_max, ycs_max)
        rng.AddPoint(xcs_min, ycs_max)
        rng.AddPoint(xcs_min, ycs_min)

        ply = ogr.Geometry(ogr.wkbPolygon)
        ply.AddGeometry(rng)

        return ply

    def _tfm_msh(self, xcs, ycs, src_dst_tfm):

        '''
        Inplace transform!
        '''

        assert xcs.ndim == 2, xcs.ndim
        assert ycs.ndim == 2, ycs.ndim

        assert xcs.shape == ycs.shape, (xcs.shape, ycs.shape)

        # Check if the transform is equal.
        i = j = 0
        xcs_tfm, ycs_tfm = src_dst_tfm.transform(xcs[i, j], ycs[i, j])

        if np.isclose(xcs_tfm, xcs[i, j]) and np.isclose(ycs[i, j], ycs_tfm):
            return

        ags = RTRARGS()
        ags.src_dst_tfm = src_dst_tfm

        if self._mpg_pol is None:
            ags.mpg_flg = False

            ags.xcs = xcs
            ags.ycs = ycs

            btr = default_timer()

            tfm_msh_sgl(((i for i in range(xcs.shape[0])), ags))

            etr = default_timer()

        else:
            ags.mpg_flg = True

            # shm_ags = SHMARGS()
            # shm_arr_dct = {}
            #
            # shm_arr_dct['xcs'] = xcs
            # shm_arr_dct['ycs'] = ycs

            ags.xcs = SHMARY.frm_npy_ary(xcs)
            ags.ycs = SHMARY.frm_npy_ary(ycs)
            #==================================================================

            # init_shm_arrs(ags, shm_ags, shm_arr_dct)

            mpg_ixs = ret_mp_idxs(xcs.shape[0], self._mpg_pol._processes)

            iis = np.arange(xcs.shape[0])

            # No shuffling needed here.
            mpg_ags = (
                (iis[i:j], ags) for i, j in zip(mpg_ixs[:-1], mpg_ixs[+1:]))

            btr = default_timer()

            list(self._mpg_pol.map(tfm_msh_sgl, mpg_ags, chunksize=1))

            etr = default_timer()
            #==================================================================

            # for key, val in shm_arr_dct.items():
            #
            #     if key not in ('xcs', 'ycs'): continue
            #
            #     if hasattr(ags, key): continue
            #
            #     val[:] = get_shm_arr(ags, key)
            #
            # free_shm_arrs(shm_ags)

            xcs[:] = ags.xcs
            ycs[:] = ags.ycs

            ags.xcs.close(); ags.xcs.unlink()
            ags.ycs.close(); ags.ycs.unlink()
            #==================================================================

        if self._vb: print(
            f'Coordinate transformation took: {etr - btr:0.1f} seconds.')
        return

    def _get_crd_msh(self, xcs, ycs):

        if xcs.ndim == 2:
            assert xcs.ndim == ycs.ndim, (xcs.ndim, ycs.ndim)
            return xcs, ycs

        assert xcs.ndim == 1, xcs.ndim
        assert ycs.ndim == 1, ycs.ndim

        assert xcs.size >= 2, xcs.size
        assert ycs.size >= 2, ycs.size

        xcs, ycs = np.meshgrid(xcs, ycs)

        return xcs, ycs

    def _get_ras_crs(self, inp_pth):

        ras = gdal.Open(str(inp_pth))

        assert ras is not None, inp_pth

        ras_crs = ras.GetProjection()

        assert ras_crs is not None

        ras = None

        return ras_crs

    def _get_crd_obj_ras(self, inp_pth):

        ras_crd_obj = ExtractGTiffCoords(verbose=False)  # self._vb

        ras_crd_obj.set_input(inp_pth)
        ras_crd_obj.extract_coordinates()

        return ras_crd_obj

    def _get_msh_ply_cns_cds(self, xcs, ycs):

        xcs_ulc = xcs[+0, +0]
        xcs_llc = xcs[-1, +0]

        xcs_urc = xcs[+0, -1]
        xcs_lrc = xcs[-1, -1]

        ycs_ulc = ycs[+0, +0]
        ycs_llc = ycs[-1, +0]

        ycs_urc = ycs[+0, -1]
        ycs_lrc = ycs[-1, -1]

        xcs_ply = [xcs_ulc, xcs_llc, xcs_lrc, xcs_urc, xcs_ulc]
        ycs_ply = [ycs_ulc, ycs_llc, ycs_lrc, ycs_urc, ycs_ulc]

        return xcs_ply, ycs_ply

    def _plt_src_dst_eps(
            self,
            src_xcs,
            src_ycs,
            dst_xcs,
            dst_ycs,
            xcs_ply_src_ful,
            ycs_ply_src_ful,
            xcs_ply_dst_ful,
            ycs_ply_dst_ful):

        (xcs_ply_src_itt,
         ycs_ply_src_itt) = self._get_msh_ply_cns_cds(src_xcs, src_ycs)

        (xcs_ply_dst_itt,
         ycs_ply_dst_itt) = self._get_msh_ply_cns_cds(dst_xcs, dst_ycs)
        #======================================================================

        fig = plt.figure()

        plt.gca().set_aspect('equal')

        # SRC_FUL.
        plt.plot(
            xcs_ply_src_ful,
            ycs_ply_src_ful,
            c='C0',
            lw=4.0,
            alpha=0.85,
            zorder=1,
            label='SRC_FUL')

        # SRC_ITT.
        plt.plot(
            xcs_ply_src_itt,
            ycs_ply_src_itt,
            c='C1',
            lw=3.0,
            alpha=0.85,
            zorder=4,
            label='SRC_ITT')

        # DST_FUL.
        plt.plot(
            xcs_ply_dst_ful,
            ycs_ply_dst_ful,
            c='C2',
            lw=2.0,
            alpha=0.85,
            zorder=2,
            label='DST_FUL')

        # DST_ITT.
        plt.plot(
            xcs_ply_dst_itt,
            ycs_ply_dst_itt,
            c='C3',
            lw=1.2,
            alpha=0.85,
            zorder=5,
            label='DST_ITT')

        plt.legend()

        plt.grid()
        plt.gca().set_axisbelow(True)

        # plt.show(block=True)

        plt.savefig(
            self._out_pth.with_name(f'{self._out_pth.stem}__itt_eps.png'),
            dpi=200)

        plt.close(fig)
        return


def gen_wts_dct_frm_crd_and_ply_sgl(ags):

    '''
    This function is supposed to handle all sorts of transformations.
    '''

    beg_rws_idx, end_rws_idx, ags = ags

    # Constants and other small objects go here.
    vbe_flg = ags.vbe_flg
    src_cel_hgt = ags.src_cel_hgt
    dst_src_tfm = ags.dst_src_tfm
    src_wts_dct_sav_flg = ags.src_wts_dct_sav_flg
    #==========================================================================

    # Verify all upcoming occurrences of these flags to be correct.
    if ags.rsm_flg:
        assert not any([ags.css_flg, ags.ncf_flg])

    elif ags.css_flg:
        assert not any([ags.rsm_flg, ags.ncf_flg])

        src_unq_dct = ags.src_unq_dct

    elif ags.ncf_flg:
        assert not any([ags.rsm_flg, ags.css_flg])

        ncf_vrs = ags.ncf_vrs

    else:
        raise NotImplementedError('Unknown flag for resampling!')
    #==========================================================================

    # Arrays and larger objects read afterwards.
    # if ags.mpg_flg: fill_shm_arrs(ags)

    src_xcs = ags.src_xcs
    src_ycs = ags.src_ycs
    dst_xcs = ags.dst_xcs
    dst_ycs = ags.dst_ycs
    dst_arr = ags.dst_arr

    src_xcs_ogl = ags.src_xcs_ogl
    src_ycs_ogl = ags.src_ycs_ogl

    # Resampling flags. Make changes in _prc_rsp_fgs accordingly.
    # CODES: 0 = Unprocessed.
    #        1 = Processed.
    #        2 = Skipped because either intersecting rows or columns
    #            were incorrect.
    #        3 = Skipped because the intersecting area between source and
    #            destination cells was not equal.
    #        4 = Skipped because destination was NaN.
    #        5 = Source was NaN (at least one band/dataset).
    rsp_fgs = ags.rsp_fgs

    # NaN flags for skipping cells that are all not finite.
    # Should make the computations speed up.
    nan_fgs = ags.nan_fgs

    if any([ags.rsm_flg, ags.css_flg]):

        assert not ags.ncf_flg

        src_arr = ags.src_arr
        src_dst_arr = ags.src_dst_arr

    elif ags.ncf_flg:

        for key in ncf_vrs:
            locals()[f'src_arr__{key}'] = getattr(
                ags, f'src_arr__{key}')

            locals()[f'src_dst_arr__{key}'] = getattr(
                ags, f'src_dst_arr__{key}')

    else:
        raise NotImplementedError('Unknown flag for resampling!')
    #==========================================================================

    assert src_xcs.ndim == 2, src_xcs.ndim
    assert src_xcs.size >= 2, src_xcs.shape
    assert src_xcs.shape == src_ycs.shape, (src_xcs.shape, src_ycs.shape)

    assert dst_xcs.ndim == 2, dst_xcs.ndim
    assert dst_xcs.size >= 2, dst_xcs.shape
    assert dst_xcs.shape == dst_ycs.shape, (dst_xcs.shape, dst_ycs.shape)
    #==========================================================================

    hit_ctr_ttl = 0
    hit_ctr_mis = 0

    src_wts_dct = {}
    src_bfr_ply_dct = {}
    src_bfr_ply_ixs_lst = []
    for i in range(beg_rws_idx, end_rws_idx):

        if i == 0: btr = etr = default_timer()

        if vbe_flg and (beg_rws_idx == 0) and (i == 1):
            print(f'One row took {etr - btr:0.2f} seconds.')

            print(
                f'Estimated time to go through all the '
                f'{dst_xcs.shape[0]} rows is '
                f'{2.05 * (etr - btr) * (end_rws_idx - beg_rws_idx):0.2f} '
                f'seconds.')

        # Leave here!
        assert np.unique(dst_ycs[i,:]).size == 1, dst_ycs[i,:]
        #======================================================================

        dst_ymn = dst_ycs[i + 1, 0]  # Same values in all columns.
        dst_ymx = dst_ycs[i + 0, 0]  # Same values in all columns.
        #======================================================================

        # Truncate polygons list.
        src_bfr_ply_lst_len = (
            src_ycs.shape[1] *
            int((dst_ycs[0, 0] - dst_ycs[1, 0]) / src_cel_hgt) + 1)

        for klt in src_bfr_ply_ixs_lst[:-src_bfr_ply_lst_len]:

            del src_bfr_ply_dct[klt]

        src_bfr_ply_ixs_lst = src_bfr_ply_ixs_lst[-src_bfr_ply_lst_len:]
        #======================================================================

        for j in range(dst_xcs.shape[1] - 1):

            # Mask or not? There should be an option for this.
            if np.isnan(dst_arr[0, i, j]):
                rsp_fgs[i, j] = 4
                continue

            if i == 0: assert np.unique(dst_xcs[:, j]).size == 1, dst_xcs[:, j]

            dst_xmn = dst_xcs[0, j + 0]  # Same values in all rows!
            dst_xmx = dst_xcs[0, j + 1]  # Same values in all rows!
            #==================================================================

            (src_beg_row_idx,
             src_end_row_idx,
             src_beg_col_idx,
             src_end_col_idx) = get_src_ogl_ixs(
                    dst_xmn,
                    dst_xmx,
                    dst_ymn,
                    dst_ymx,
                    src_xcs_ogl,
                    src_ycs_ogl,
                    dst_src_tfm)

            skp_flg = False

            if src_beg_row_idx >= src_end_row_idx: skp_flg = True
            if src_beg_col_idx >= src_end_col_idx: skp_flg = True

            if src_beg_row_idx < 0: skp_flg = True
            if src_end_row_idx < 0: skp_flg = True

            if src_beg_col_idx < 0: skp_flg = True
            if src_end_col_idx < 0: skp_flg = True

            if skp_flg:
                rsp_fgs[i, j] = 2
                continue
            #==================================================================

            assert src_beg_row_idx < src_end_row_idx, (
                src_beg_row_idx, src_end_row_idx)

            assert src_beg_col_idx < src_end_col_idx, (
                src_beg_col_idx, src_end_col_idx)
            #==================================================================

            if nan_fgs[src_beg_row_idx:src_end_row_idx,
                       src_beg_col_idx:src_end_col_idx].all():

                rsp_fgs[i, j] = 5
                continue
            #==================================================================

            rng = ogr.Geometry(ogr.wkbLinearRing)

            rng.AddPoint(dst_xmn, dst_ymn)
            rng.AddPoint(dst_xmx, dst_ymn)
            rng.AddPoint(dst_xmx, dst_ymx)
            rng.AddPoint(dst_xmn, dst_ymx)
            rng.AddPoint(dst_xmn, dst_ymn)

            dst_ply = ogr.Geometry(ogr.wkbPolygon)
            dst_ply.AddGeometry(rng)

            dst_ply_ara = dst_ply.Area()
            #==================================================================

            if False:  #  This one is not needed. Use the one below.
                # +1 is required to account for the extent of cell.
                src_xmn = src_xcs[
                    src_beg_row_idx:src_end_row_idx + 1,
                    src_beg_col_idx:src_end_col_idx + 1, ].min()

                src_xmx = src_xcs[
                    src_beg_row_idx:src_end_row_idx + 1,
                    src_beg_col_idx:src_end_col_idx + 1, ].max()

                src_ymn = src_ycs[
                    src_beg_row_idx:src_end_row_idx + 1,
                    src_beg_col_idx:src_end_col_idx + 1, ].min()

                src_ymx = src_ycs[
                    src_beg_row_idx:src_end_row_idx + 1,
                    src_beg_col_idx:src_end_col_idx + 1, ].max()

                cn1 = (dst_xmn >= src_xmn) or np.isclose(dst_xmn, src_xmn)
                cn2 = (dst_xmx <= src_xmx) or np.isclose(dst_xmx, src_xmx)
                cn3 = (dst_ymn >= src_ymn) or np.isclose(dst_ymn, src_ymn)
                cn4 = (dst_ymx <= src_ymx) or np.isclose(dst_ymx, src_ymx)

                if not all([cn1, cn2, cn3, cn4]):

                    # print(i, j, cn1, cn2, cn3, cn4)

                    src_xcs_ulc = src_xcs[src_beg_row_idx, src_beg_col_idx]
                    src_xcs_llc = src_xcs[src_end_row_idx, src_beg_col_idx]

                    src_xcs_urc = src_xcs[src_beg_row_idx, src_end_col_idx]
                    src_xcs_lrc = src_xcs[src_end_row_idx, src_end_col_idx]

                    src_ycs_ulc = src_ycs[src_beg_row_idx, src_beg_col_idx]
                    src_ycs_llc = src_ycs[src_end_row_idx, src_beg_col_idx]

                    src_ycs_urc = src_ycs[src_beg_row_idx, src_end_col_idx]
                    src_ycs_lrc = src_ycs[src_end_row_idx, src_end_col_idx]

                    rng = ogr.Geometry(ogr.wkbLinearRing)

                    rng.AddPoint(src_xcs_ulc, src_ycs_ulc)
                    rng.AddPoint(src_xcs_llc, src_ycs_llc)
                    rng.AddPoint(src_xcs_lrc, src_ycs_lrc)
                    rng.AddPoint(src_xcs_urc, src_ycs_urc)
                    rng.AddPoint(src_xcs_ulc, src_ycs_ulc)

                    src_ply_ful = ogr.Geometry(ogr.wkbPolygon)
                    src_ply_ful.AddGeometry(rng)

                    src_ply_ful_ara = src_ply_ful.Area()

                    assert src_ply_ful_ara > 0
                    assert dst_ply_ara <= src_ply_ful_ara

                    src_ful_dst_ara = src_ply_ful.Intersection(dst_ply).Area()

                    # DST polygon is outside!
                    # This happens for rotated grids sometimes.
                    if True and (not np.isclose(src_ful_dst_ara, dst_ply_ara)):
                        rsp_fgs[i, j] = 3
                        continue

                    fig = plt.figure()

                    plt.gca().set_aspect('equal')

                    xcs_ply = [dst_xmn, dst_xmn, dst_xmx, dst_xmx, dst_xmn]
                    ycs_ply = [dst_ymn, dst_ymx, dst_ymx, dst_ymn, dst_ymn]

                    plt.plot(
                        xcs_ply, ycs_ply, c='cyan', alpha=0.85, zorder=10)

                    xcs_ply = [
                        src_xcs_ulc,
                        src_xcs_llc,
                        src_xcs_lrc,
                        src_xcs_urc,
                        src_xcs_ulc]

                    ycs_ply = [
                        src_ycs_ulc,
                        src_ycs_llc,
                        src_ycs_lrc,
                        src_ycs_urc,
                        src_ycs_ulc]

                    plt.plot(
                        xcs_ply, ycs_ply, c='red', alpha=0.85, zorder=10)

                    plt.title('3')

                    plt.show(block=True)

                    plt.close(fig)
            #==================================================================

            if False:
                src_xcs_ulc = src_xcs[src_beg_row_idx, src_beg_col_idx]
                src_xcs_llc = src_xcs[src_end_row_idx, src_beg_col_idx]

                src_xcs_urc = src_xcs[src_beg_row_idx, src_end_col_idx]
                src_xcs_lrc = src_xcs[src_end_row_idx, src_end_col_idx]

                src_ycs_ulc = src_ycs[src_beg_row_idx, src_beg_col_idx]
                src_ycs_llc = src_ycs[src_end_row_idx, src_beg_col_idx]

                src_ycs_urc = src_ycs[src_beg_row_idx, src_end_col_idx]
                src_ycs_lrc = src_ycs[src_end_row_idx, src_end_col_idx]

                rng = ogr.Geometry(ogr.wkbLinearRing)

                rng.AddPoint(src_xcs_ulc, src_ycs_ulc)
                rng.AddPoint(src_xcs_llc, src_ycs_llc)
                rng.AddPoint(src_xcs_lrc, src_ycs_lrc)
                rng.AddPoint(src_xcs_urc, src_ycs_urc)
                rng.AddPoint(src_xcs_ulc, src_ycs_ulc)

                src_ply_ful = ogr.Geometry(ogr.wkbPolygon)
                src_ply_ful.AddGeometry(rng)

                src_ply_ful_ara = src_ply_ful.Area()

                assert src_ply_ful_ara > 0
                assert dst_ply_ara <= src_ply_ful_ara

                src_ful_dst_ara = src_ply_ful.Intersection(dst_ply).Area()

                # assert np.isclose(src_ful_dst_ara, dst_ply_ara), (
                #     i, j, src_ful_dst_ara, dst_ply_ara)

                # DST polygon is outside!
                # This happens for rotated grids sometimes.
                if True and (not np.isclose(src_ful_dst_ara, dst_ply_ara)):
                    rsp_fgs[i, j] = 3
                    continue

                fig = plt.figure()

                plt.gca().set_aspect('equal')

                xcs_ply = [dst_xmn, dst_xmn, dst_xmx, dst_xmx, dst_xmn]
                ycs_ply = [dst_ymn, dst_ymx, dst_ymx, dst_ymn, dst_ymn]

                plt.plot(xcs_ply, ycs_ply, c='cyan', alpha=0.85, zorder=10)

                xcs_ply = [
                    src_xcs_ulc,
                    src_xcs_llc,
                    src_xcs_lrc,
                    src_xcs_urc,
                    src_xcs_ulc]

                ycs_ply = [
                    src_ycs_ulc,
                    src_ycs_llc,
                    src_ycs_lrc,
                    src_ycs_urc,
                    src_ycs_ulc]

                plt.plot(xcs_ply, ycs_ply, c='red', alpha=0.85, zorder=10)

                plt.title('3')

                plt.show(block=True)

                plt.close(fig)
            #==================================================================

            ara_tot_ctr = 0
            ara_mis_ctr = 0

            wts_dct = {}
            src_ply_ara = 0.0
            for k in range(src_beg_row_idx, src_end_row_idx):
                for l in range(src_beg_col_idx, src_end_col_idx):

                    klt = (k, l)

                    # It's so strange, checking inside a list for an item
                    # takes a tremendous toll on speed.
                    if klt not in src_bfr_ply_dct:

                        src_bfr_ply_dct[klt] = get_ply_frm_crd_ary(
                            src_xcs, src_ycs, k, l)

                        src_bfr_ply_ixs_lst.append(klt)

                    ara_tot_ctr += 1

                    itt_ara = dst_ply.Intersection(src_bfr_ply_dct[klt]).Area()

                    if itt_ara == 0.0:
                        ara_mis_ctr += 1
                        continue

                    wts_dct[klt] = (itt_ara / dst_ply_ara)

                    src_ply_ara += itt_ara
            #==================================================================

            hit_ctr_ttl += ara_tot_ctr
            hit_ctr_mis += ara_mis_ctr

            if False:
                print(
                    i,
                    j,
                    ara_mis_ctr,
                    ara_tot_ctr,
                    f'{ara_mis_ctr / ara_tot_ctr:.2%}')

            if True and (not np.isclose(src_ply_ara, dst_ply_ara)):
                rsp_fgs[i, j] = 3
                continue

            else:
                assert np.isclose(dst_ply_ara, src_ply_ara), (
                    dst_ply_ara, src_ply_ara, f'{src_ply_ara/dst_ply_ara:.2%}')

            if src_wts_dct_sav_flg: src_wts_dct[(i, j)] = wts_dct

            if ags.rsm_flg:
                assert not any([ags.css_flg, ags.ncf_flg])

                src_dst_vls = np.zeros(
                    src_dst_arr.shape[0], dtype=src_dst_arr.dtype.type)

                for (k, l), wht in wts_dct.items():
                    src_dst_vls += wht * src_arr[:, k, l]

                src_dst_arr[:, i, j] = src_dst_vls

                if (~np.isfinite(src_dst_vls)).any():

                    rsp_fgs[i, j] = 5

                else:
                    rsp_fgs[i, j] = 1

            elif ags.css_flg:
                assert not any([ags.rsm_flg, ags.ncf_flg])

                src_dst_arr[:, i, j] = 0.0

                for (k, l), wht in wts_dct.items():
                    src_dst_arr[
                        src_unq_dct[str(src_arr[0, k, l])], i, j] += wht

                if (~np.isfinite(src_dst_arr[:, i, j])).any():

                    rsp_fgs[i, j] = 5

                else:
                    rsp_fgs[i, j] = 1

            elif ags.ncf_flg:
                assert not any([ags.rsm_flg, ags.css_flg])

                for key in ncf_vrs:
                    fil_wtd_cel(
                        i,
                        j,
                        wts_dct,
                        locals()[f'src_arr__{key}'],
                        locals()[f'src_dst_arr__{key}'],
                        rsp_fgs)

            else:
                raise NotImplementedError('Unknown flag for resampling!')

            #==================================================================

        if False:
            print(
                f'IDX: {i:04d}, '
                f'TTL: {hit_ctr_ttl}, '
                f'MIS: {hit_ctr_mis}, '
                f'SCS_RATE: {(hit_ctr_ttl - hit_ctr_mis) / hit_ctr_ttl:0.2%}')

        if i == 0: etr = default_timer()
        #======================================================================

    src_bfr_ply_ixs_lst = None
    src_bfr_ply_dct = None

    return src_wts_dct, hit_ctr_ttl, hit_ctr_mis


def get_src_ogl_ixs(
        dst_xmn,
        dst_xmx,
        dst_ymn,
        dst_ymx,
        src_xcs_ogl,
        src_ycs_ogl,
        dst_src_tfm):

    '''
    Assuming X increases left to right and Y increases from top to bottom.

    Assuming constant cell width and height in ogl cds.
    '''

    src_xmn = src_xcs_ogl[0, 0]
    src_ymx = src_ycs_ogl[0, 0]

    src_cel_hgt = src_xcs_ogl[0, 1] - src_xmn
    src_cel_wdh = src_ymx - src_ycs_ogl[1, 0]
    #==========================================================================

    ulc_xce, ulc_yce = dst_src_tfm.transform(dst_xmn, dst_ymx)
    ulc_cln = int(floor((ulc_xce - src_xmn) / src_cel_wdh))
    ulc_row = int(floor((src_ymx - ulc_yce) / src_cel_hgt))
    #==========================================================================

    urc_xce, urc_yce = dst_src_tfm.transform(dst_xmx, dst_ymx)
    urc_cln = int(ceil((urc_xce - src_xmn) / src_cel_wdh))
    urc_row = int(floor((src_ymx - urc_yce) / src_cel_hgt))
    #==========================================================================

    lrc_xce, lrc_yce = dst_src_tfm.transform(dst_xmx, dst_ymn)
    lrc_cln = int(ceil((lrc_xce - src_xmn) / src_cel_wdh))
    lrc_row = int(ceil((src_ymx - lrc_yce) / src_cel_hgt))
    #==========================================================================

    llc_xce, llc_yce = dst_src_tfm.transform(dst_xmn, dst_ymn)
    llc_cln = int(floor((llc_xce - src_xmn) / src_cel_wdh))
    llc_row = int(ceil((src_ymx - llc_yce) / src_cel_hgt))
    #==========================================================================

    bgn_row = min([ulc_row, urc_row, lrc_row, llc_row]) - 0
    end_row = max([ulc_row, urc_row, lrc_row, llc_row]) + 0

    bgn_cln = min([ulc_cln, urc_cln, lrc_cln, llc_cln]) - 0
    end_cln = max([ulc_cln, urc_cln, lrc_cln, llc_cln]) + 0
    #==========================================================================

    bgn_row = max(0, bgn_row)
    end_row = min(end_row, src_ycs_ogl.shape[0] - 1)
    end_row = max(1, end_row)

    bgn_cln = max(0, bgn_cln)
    end_cln = min(end_cln, src_xcs_ogl.shape[1] - 1)
    end_cln = max(1, end_cln)

    return bgn_row, end_row, bgn_cln, end_cln


def fil_wtd_cel(i, j, wts_dct, src_arr, src_dst_arr, rsp_fgs):

    src_dst_vls = np.zeros(src_dst_arr.shape[0], dtype=src_dst_arr.dtype.type)

    for (k, l), wht in wts_dct.items(): src_dst_vls += wht * src_arr[:, k, l]

    src_dst_arr[:, i, j] = src_dst_vls

    if (~np.isfinite(src_dst_vls)).all():

        rsp_fgs[i, j] = 5

    else:
        if rsp_fgs[i, j] == 0: rsp_fgs[i, j] = 1

    return


def get_ply_frm_crd_ary(xcs, ycs, i, j):

    rng = ogr.Geometry(ogr.wkbLinearRing)

    rng.AddPoint(xcs[i, j], ycs[i, j])
    rng.AddPoint(xcs[i, j + 1], ycs[i, j + 1])
    rng.AddPoint(xcs[i + 1, j + 1], ycs[i + 1, j + 1])
    rng.AddPoint(xcs[i + 1, j], ycs[i + 1, j])
    rng.AddPoint(xcs[i, j], ycs[i, j])

    ply = ogr.Geometry(ogr.wkbPolygon)
    ply.AddGeometry(rng)

    return ply


def tfm_msh_sgl(ags):

    '''
    Inplace transform!
    '''

    ii, ags = ags

    # if ags.mpg_flg: fill_shm_arrs(ags)

    # cls_cnt = ags.xcs.shape[1]

    for i in ii:

        (tfm_xcs,
         tfm_ycs) = ags.src_dst_tfm.transform(ags.xcs[i,:], ags.ycs[i,:])

        ags.xcs[i,:] = tfm_xcs
        ags.ycs[i,:] = tfm_ycs

        # This turned out to be too slow. I don't remember, why I didn't use
        # simple broadcasting. Perhaps, memory saving.
        # for j in range(cls_cnt):
        #
        #     (ags.xcs[i, j],
        #      ags.ycs[i, j]) = ags.src_dst_tfm.transform(
        #          ags.xcs[i, j],
        #          ags.ycs[i, j])

    return


class RTRARGS:

    def __init__(self):

        # gen_wts_dct_frm_crd_and_ply_sgl expects these flags to be defined.

        self.rsm_flg = False  # Raster to raster.
        self.css_flg = False  # Raster classes.
        self.ncf_flg = False  # NetCDF to raster.
        return
