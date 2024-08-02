# -*- coding: utf-8 -*-

'''
Created on 08.07.2024

@author: Faizan-TU Munich
'''

import pickle
from pathlib import Path
from timeit import default_timer
from multiprocessing import Manager, Pool as MPPool

import numpy as np
from osgeo import ogr, gdal
from pyproj import Transformer

from ..extract import ExtractGTiffCoords

from ..mpg import init_shm_arrs, fill_shm_arrs, get_shm_arr, free_shm_arrs
from ..mpg import SHMARGS, DummyLock

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
            always_xy=True)

        if self._vb: print('Transforming source coordinates\' mesh...')

        self._tfm_msh(src_xcs, src_ycs, src_dst_tfm)

        if self._vb: print(f'Source original mesh shape: {src_xcs.shape}...')
        if self._vb: print(f'Desti. original mesh shape: {dst_xcs.shape}...')
        #======================================================================

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

        src_xmn = src_xcs.min()
        src_xmx = src_xcs.max()

        src_ymn = src_ycs.min()
        src_ymx = src_ycs.max()

        if self._vb: print(f'Source snipped mesh shape: {src_xcs.shape}...')
        #======================================================================

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

        if self._vb: print('Reading input arrays...')

        dst_arr_shp = self._get_ras_vrs(
            self._dst_pth,
            dst_beg_row,
            dst_end_row,
            dst_beg_col,
            dst_end_col,
            1)[0].shape

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
             src_xcs, src_ycs, dst_xcs, dst_ycs)
        #======================================================================

        if self._vb: print('Computing resampled array...')

        ags = RTRARGS()

        # Constants and other small objects go here. The rest in _cpt_wts_vrs.
        ags.rsm_flg = True
        ags.vbe_flg = self._vb
        ags.src_bfr_rws = src_bfr_rws
        ags.src_bfr_cns = src_bfr_cns
        ags.src_xmn_min = src_xmn_min
        ags.src_ymx_max = src_ymx_max
        ags.src_cel_hgt = src_cel_hgt
        ags.src_cel_wdh = src_cel_wdh
        ags.src_wts_dct_sav_flg = self._src_wts_dct_sav_flg

        self._cpt_wts_vrs(
            ags,
            src_xcs,
            src_ycs,
            dst_xcs,
            dst_ycs,
            src_arr,
            src_dst_arr)

        ags = None
        src_arr = None
        src_xcs = None
        src_ycs = None
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

    def _get_src_adj_vrs(self, src_xcs, src_ycs, dst_xcs, dst_ycs):

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

        src_ycs_dfs = src_ycs[:-1,:] - src_ycs[+1:,:]

        src_cle_cdy = np.isclose(
            src_ycs_dfs, src_ycs[0, 0] - src_ycs[1, 0]).all()

        src_cel_hgt = np.abs(src_ycs_dfs).max()

        src_ycs_dfs = None

        src_xcs_dfs = src_xcs[:, 1:] - src_xcs[:,:-1]

        src_cle_cdx = np.isclose(
            src_xcs_dfs, src_xcs[0, 1] - src_xcs[0, 0]).all()

        src_cel_wdh = np.abs(src_xcs_dfs).max()

        src_xcs_dfs = None

        if src_cle_cdy and src_cle_cdx:
            if self._vb: print('Source cells are constant in size.')
        #======================================================================

        src_rws_hgt_rel_crt = int(src_cel_hgt / dst_cel_hgt)
        src_rws_rtn_rel_crt = np.ceil(
            max(np.abs(src_ycs[+0, +0] - src_ycs[+0, -1]),
                np.abs(src_ycs[-1, +0] - src_ycs[-1, -1])) / src_cel_hgt)

        src_cns_wdh_rel_crt = int(src_cel_wdh / dst_cel_wdh)
        src_cns_rtn_rel_crt = np.ceil(
            max(np.abs(src_xcs[-1, +0] - src_xcs[+0, +0]),
                np.abs(src_xcs[-1, -1] - src_xcs[+0, -1])) / src_cel_wdh)

        src_bfr_rws = int(max(0, src_rws_hgt_rel_crt + src_rws_rtn_rel_crt))
        src_bfr_cns = int(max(0, src_cns_wdh_rel_crt + src_cns_rtn_rel_crt))

        src_xmn_min = src_xcs[0:, 0].min()
        src_ymx_max = src_ycs[0, 0:].max()

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

            if bnd_mda is not None:
                out_bnd.SetMetadata(bnd_mda[i])
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
            src_dst_arr):

        '''
        Coordinates are 2D of cell corners.

        src and dst have same crs.

        This function is adapted to handle all sorts of transformation
         defined till now.
        '''

        btr = default_timer()

        # Large objects have to be assigned here.
        if self._mpg_pol is None:

            ags.mpg_flg = False

            ags.src_xcs = src_xcs
            ags.src_ycs = src_ycs
            ags.dst_xcs = dst_xcs
            ags.dst_ycs = dst_ycs

            if not ags.ncf_flg:

                assert any([ags.rsm_flg, ags.css_flg])

                ags.src_arr = src_arr
                ags.src_dst_arr = src_dst_arr

            else:
                assert all([not ags.rsm_flg, not ags.css_flg])

                setattr(ags, 'ncf_vrs', list(src_arr))

                ncf_shm_nms = []
                for key in src_arr:
                    setattr(ags, f'src_arr__{key}', src_arr[key])
                    setattr(ags, f'src_dst_arr__{key}', src_dst_arr[key])

                    ncf_shm_nms.append(f'src_dst_arr__{key}')

            src_wts_dct = gen_wts_dct_frm_crd_and_ply_sgl(
                (0, dst_xcs.shape[0] - 1, ags))

        else:
            ags.mpg_flg = True

            shm_ags = SHMARGS()
            shm_arr_dct = {}

            shm_arr_dct['src_xcs'] = src_xcs
            shm_arr_dct['src_ycs'] = src_ycs
            shm_arr_dct['dst_xcs'] = dst_xcs
            shm_arr_dct['dst_ycs'] = dst_ycs

            if not ags.ncf_flg:

                assert any([ags.rsm_flg, ags.css_flg])

                shm_arr_dct['src_arr'] = src_arr
                shm_arr_dct['src_dst_arr'] = src_dst_arr

            else:
                assert all([not ags.rsm_flg, not ags.css_flg])

                setattr(ags, 'ncf_vrs', list(src_arr))

                ncf_shm_nms = []
                for key in src_arr:
                    shm_arr_dct[f'src_arr__{key}'] = src_arr[key]
                    shm_arr_dct[f'src_dst_arr__{key}'] = src_dst_arr[key]

                    ncf_shm_nms.append(f'src_dst_arr__{key}')
            #==================================================================

            init_shm_arrs(ags, shm_ags, shm_arr_dct)

            mpg_ixs = ret_mp_idxs(
                dst_xcs.shape[0] - 1, self._mpg_pol._processes * 2)

            mpg_ags = (
                (i, j, ags)
                for i, j in zip(mpg_ixs[:-1], mpg_ixs[+1:]))

            ress = self._mpg_pol.map(
                gen_wts_dct_frm_crd_and_ply_sgl, mpg_ags, chunksize=1)

            ress = list(ress)
            #==================================================================

            for key, val in shm_arr_dct.items():

                if not ags.ncf_flg:

                    assert any([ags.rsm_flg, ags.css_flg])

                    if key not in ['src_dst_arr', ]: continue

                    if hasattr(ags, key): continue

                    val[:] = get_shm_arr(ags, key)

                else:
                    assert not any([ags.rsm_flg, ags.css_flg])

                    if key not in ncf_shm_nms: continue

                    ky1, ky2 = key.rsplit('__', 1)

                    locals()[ky1][ky2][:] = get_shm_arr(ags, key)

            free_shm_arrs(shm_ags)
            #==================================================================

            if self._src_wts_dct_sav_flg:
                src_wts_dct = {}
                for res in ress: src_wts_dct.update(res)
                ress = None
            #==================================================================

            etr = default_timer()

            if self._vb: print(
                f'Weights resampling took: {etr - btr:0.1f} seconds.')
            #==================================================================

        if self._src_wts_dct_sav_flg:
            src_dct_pth = self._out_pth.with_name(
                f'{self._out_pth.stem}_src_wts_dct.pkl')

            with open(src_dct_pth, 'wb+') as pkl_hdl:
                pickle.dump(src_wts_dct, pkl_hdl)

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

        # if False:  # Diagnostics.
        #     import matplotlib.pyplot as plt
        #     plt.gca().set_aspect('equal')
        #
        #     xcs_ply = [xcs_min, xcs_min, xcs_max, xcs_max, xcs_min]
        #     ycs_ply = [ycs_min, ycs_max, ycs_max, ycs_min, ycs_min]
        #
        #     plt.plot(xcs_ply, ycs_ply, c='cyan', alpha=0.85, zorder=10)
        #     plt.pcolormesh(
        #         xcs[beg_row:end_row, beg_col:end_col],
        #         ycs[beg_row:end_row, beg_col:end_col],
        #         row_col_ixs[beg_row:end_row - 1, beg_col:end_col - 1],)
        #     plt.pcolormesh(xcs, ycs, row_col_ixs[:-1,:-1],)
        #
        #     plt.close()
        #======================================================================

        # Take care of rotation and small adjustments.
        while xcs[beg_row:end_row, beg_col:end_col].min() > xcs_min:

            beg_col -= 1

            if beg_col <= 0: break

        beg_col = max(0, beg_col)

        while xcs[beg_row:end_row, beg_col:end_col].max() <= xcs_max:

            end_col += 1

            if end_col >= xcs.shape[1]: break

        end_col = min(xcs.shape[1], end_col)

        while ycs[beg_row:end_row, beg_col:end_col].min() >= ycs_min:

            end_row += 1

            if end_row >= xcs.shape[0]: break

        end_row = min(xcs.shape[0], end_row)

        while ycs[beg_row:end_row, beg_col:end_col].max() < ycs_max:

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
        if True:
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

            tfm_msh_sgl(((i for i in range(xcs.shape[0])), ags))

        else:
            ags.mpg_flg = True

            shm_ags = SHMARGS()
            shm_arr_dct = {}

            shm_arr_dct['xcs'] = xcs
            shm_arr_dct['ycs'] = ycs
            #==================================================================

            init_shm_arrs(ags, shm_ags, shm_arr_dct)

            mpg_ixs = ret_mp_idxs(xcs.shape[0], self._mpg_pol._processes)

            iis = np.arange(xcs.shape[0])

            # No shuffling needed here.
            mpg_ags = (
                (iis[i:j], ags) for i, j in zip(mpg_ixs[:-1], mpg_ixs[+1:]))

            list(self._mpg_pol.map(tfm_msh_sgl, mpg_ags, chunksize=1))
            #==================================================================

            for key, val in shm_arr_dct.items():

                if key not in ('xcs', 'ycs'): continue

                if hasattr(ags, key): continue

                val[:] = get_shm_arr(ags, key)

            free_shm_arrs(shm_ags)
            #==================================================================

        return

    def _get_crd_msh(self, xcs, ycs):

        if xcs.ndim == 2:
            assert xcs.ndim == ycs.ndim, (xcs.ndim, ycs.ndim)
            return

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


def gen_wts_dct_frm_crd_and_ply_sgl(ags):

    '''
    This function is supposed to handle all sorts of transformations.
    '''

    beg_rws_idx, end_rws_idx, ags = ags

    # Constants and other small objects go here.
    vbe_flg = ags.vbe_flg
    src_bfr_rws = ags.src_bfr_rws
    src_bfr_cns = ags.src_bfr_cns
    src_xmn_min = ags.src_xmn_min
    src_ymx_max = ags.src_ymx_max
    src_cel_hgt = ags.src_cel_hgt
    src_cel_wdh = ags.src_cel_wdh
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

    # Arrays and larger objects read here.
    if ags.mpg_flg: fill_shm_arrs(ags)

    src_xcs = ags.src_xcs
    src_ycs = ags.src_ycs
    dst_xcs = ags.dst_xcs
    dst_ycs = ags.dst_ycs

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

        assert np.unique(dst_ycs[i,:]).size == 1, dst_ycs[i,:]
        #======================================================================

        dst_ymn = dst_ycs[i + 1, 0]
        dst_ymx = dst_ycs[i + 0, 0]

        src_beg_row_idx = int((src_ymx_max - dst_ymx) / src_cel_hgt)
        src_end_row_idx = int(
            np.ceil((src_ymx_max - dst_ymn) / src_cel_hgt))

        if src_beg_row_idx > src_xcs.shape[0]: continue
        if src_end_row_idx < 0: continue

        # Buffer it roughly.
        src_beg_row_idx = max(0, src_beg_row_idx - src_bfr_rws)
        src_end_row_idx = min(
            src_ycs.shape[0] - 1, src_end_row_idx + src_bfr_rws)

        # Truncate.
        src_bfr_ply_lst_len = (
            src_ycs.shape[1] * (src_end_row_idx - src_beg_row_idx + 1))

        for klt in src_bfr_ply_ixs_lst[:-src_bfr_ply_lst_len]:

            del src_bfr_ply_dct[klt]

        src_bfr_ply_ixs_lst = src_bfr_ply_ixs_lst[-src_bfr_ply_lst_len:]
        #======================================================================

        for j in range(dst_xcs.shape[1] - 1):

            # Mask or not? There should be an option for this.
            # if np.isnan(dst_arr[0, i, j]): continue

            if i == 0: assert np.unique(dst_xcs[:, j]).size == 1, dst_xcs[:, j]

            dst_xmn = dst_xcs[0, j + 0]
            dst_xmx = dst_xcs[0, j + 1]

            src_beg_col_idx = int((dst_xmn - src_xmn_min) / src_cel_wdh)
            src_end_col_idx = int(
                np.ceil((dst_xmx - src_xmn_min) / src_cel_wdh))

            if src_beg_col_idx > src_xcs.shape[1]:
                print(i, j, '>', src_beg_col_idx, dst_xcs.shape[1])
                continue

            if src_end_col_idx < 0:
                print(i, j, '<', src_end_col_idx)
                continue

            # Buffer it roughly.
            src_beg_col_idx = max(0, src_beg_col_idx - src_bfr_cns)
            src_end_col_idx = min(
                src_xcs.shape[1] - 1, src_end_col_idx + src_bfr_cns)
            #==================================================================

            if True:
                # Incrementation/decrementation can be used to be within
                # bounds here.

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

                assert dst_xmn >= src_xmn, (dst_xmn, src_xmn)
                assert dst_xmx <= src_xmx, (dst_xmx, src_xmx)

                assert dst_ymn >= src_ymn, (dst_ymn, src_ymn)
                assert dst_ymx <= src_ymx, (dst_ymx, src_ymx)
            #==================================================================

            rng = ogr.Geometry(ogr.wkbLinearRing)

            rng.AddPoint(dst_xmn, dst_ymn)
            rng.AddPoint(dst_xmx, dst_ymn)
            rng.AddPoint(dst_xmx, dst_ymx)
            rng.AddPoint(dst_xmn, dst_ymx)
            rng.AddPoint(dst_xmn, dst_ymn)

            dst_ply = ogr.Geometry(ogr.wkbPolygon)
            dst_ply.AddGeometry(rng)
            #==================================================================

            dst_ply_ara = dst_ply.Area()
            src_ply_ara = 0.0

            ara_tot_ctr = 0
            ara_mis_ctr = 0

            wts_dct = {}
            for k in range(src_beg_row_idx, src_end_row_idx):
                for l in range(src_beg_col_idx, src_end_col_idx):

                    klt = (k, l)

                    # It's so strange, checking inside a list for an item
                    # takes a tremendous toll on speed.
                    if klt not in src_bfr_ply_dct:
                        src_bfr_ply_ixs_lst.append(klt)

                        src_bfr_ply_dct[klt] = get_ply_frm_crd_ary(
                            src_xcs, src_ycs, k, l)

                    ara_tot_ctr += 1

                    itt_ara = dst_ply.Intersection(src_bfr_ply_dct[klt]).Area()

                    if np.isclose(itt_ara, 0.0):
                        ara_mis_ctr += 1
                        continue

                    wts_dct[klt] = (itt_ara / dst_ply_ara)

                    src_ply_ara += itt_ara
            #==================================================================

            if False:
                print(
                    i,
                    j,
                    ara_mis_ctr,
                    ara_tot_ctr,
                    f'{ara_mis_ctr / ara_tot_ctr:.2%}')

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

            elif ags.css_flg:
                assert not any([ags.rsm_flg, ags.ncf_flg])

                src_dst_arr [:, i, j] = 0.0

                for (k, l), wht in wts_dct.items():
                    src_dst_arr[src_unq_dct[src_arr[0, k, l]], i, j] += wht

            elif ags.ncf_flg:
                assert not any([ags.rsm_flg, ags.css_flg])

                for key in ncf_vrs:
                    fil_wtd_cel(
                        i,
                        j,
                        wts_dct,
                        locals()[f'src_arr__{key}'],
                        locals()[f'src_dst_arr__{key}'],)

            else:
                raise NotImplementedError('Unknown flag for resampling!')
            #==================================================================

        if i == 0: etr = default_timer()
        #======================================================================

    src_bfr_ply_ixs_lst = None
    src_bfr_ply_dct = None
    return src_wts_dct


def fil_wtd_cel(i, j, wts_dct, src_arr, src_dst_arr):

    src_dst_vls = np.zeros(src_dst_arr.shape[0], dtype=src_dst_arr.dtype.type)

    for (k, l), wht in wts_dct.items(): src_dst_vls += wht * src_arr[:, k, l]

    src_dst_arr[:, i, j] = src_dst_vls
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

    if ags.mpg_flg:
        fill_shm_arrs(ags)

    for i in ii:
        for j in range(ags.xcs.shape[1]):

            (ags.xcs[i, j],
             ags.ycs[i, j]) = ags.src_dst_tfm.transform(
                 ags.xcs[i, j],
                 ags.ycs[i, j])

    return


class RTRARGS:

    def __init__(self):

        # gen_wts_dct_frm_crd_and_ply_sgl expects these flags to be defined.

        self.rsm_flg = False  # Raster to raster.
        self.css_flg = False  # Raster classes.
        self.ncf_flg = False  # NetCDF to raster.
        return

#==============================================================================
# Dead code afterwards.
#==============================================================================

# def get_ply_dct_frm_crd_sgl(ags):
#
#     '''
#     GDAL object cannot be pickled somehow. For now no MP.
#     '''
#
#     iis, ags = ags
#
#     if ags.mpg_flg:
#         fill_shm_arrs(ags)
#
#     if not ags.mpg_flg:
#         for i in iis:
#             for j in range(ags.xcs.shape[1] - 1):
#                 rng = ogr.Geometry(ogr.wkbLinearRing)
#
#                 rng.AddPoint(ags.xcs[i, j], ags.ycs[i, j])
#                 rng.AddPoint(ags.xcs[i, j + 1], ags.ycs[i, j + 1])
#                 rng.AddPoint(ags.xcs[i + 1, j + 1], ags.ycs[i + 1, j + 1])
#                 rng.AddPoint(ags.xcs[i + 1, j], ags.ycs[i + 1, j])
#                 rng.AddPoint(ags.xcs[i, j], ags.ycs[i, j])
#
#                 ply = ogr.Geometry(ogr.wkbPolygon)
#                 ply.AddGeometry(rng)
#
#                 ags.ply_dct[(i, j)] = ply
#
#     else:
#         raise Exception
#
#         # ply_dct = {}
#         # for i in iis:
#         #     for j in range(ags.xcs.shape[1] - 1):
#         #
#         #         # GDAL is around two times faster in creation.
#         #         ply = Polygon(
#         #             ((ags.xcs[i, j], ags.ycs[i, j]),
#         #              (ags.xcs[i, j + 1], ags.ycs[i, j + 1]),
#         #              (ags.xcs[i + 1, j + 1], ags.ycs[i + 1, j + 1]),
#         #              (ags.xcs[i + 1, j], ags.ycs[i + 1, j]),
#         #              (ags.xcs[i, j], ags.ycs[i, j]),)
#         #             )
#         #
#         #         ply_dct[(i, j)] = ply
#         #
#         # # with ags.mpg_lck:
#         # ags.ply_dct[i] = ply_dct
#         #
#         # del ply_dct
#
#     return

# def _get_wtd_arr_ras(self, src_arr, dst_arr, dst_xcs, src_wts_dct):
#
#     src_dst_arr = np.full(
#         (src_arr.shape[0], dst_xcs.shape[0] - 1, dst_xcs.shape[1] - 1),
#         np.nan,
#         dtype=np.float32)
#
#     assert dst_arr.shape[1:] == src_dst_arr.shape[1:], (
#             dst_arr.shape, src_dst_arr.shape)
#
#     if False:
#         for m in range(src_arr.shape[0]):
#             for (i, j), wts_dct in src_wts_dct.items():
#
#                 if np.isnan(dst_arr[0, i, j]): continue
#
#                 src_dst_val = 0.0
#                 for (k, l), wht in wts_dct.items():
#                     src_dst_val += wht * src_arr[m, k, l]
#
#                 src_dst_arr[m, i, j] = src_dst_val
#
#     else:
#         for (i, j), wts_dct in src_wts_dct.items():
#
#             if np.isnan(dst_arr[0, i, j]): continue
#
#             src_dst_vls = np.zeros(
#                 src_dst_arr.shape[0], dtype=src_dst_arr.dtype.type)
#
#             for (k, l), wht in wts_dct.items():
#                 src_dst_vls += wht * src_arr[:, k, l]
#
#             src_dst_arr[:, i, j] = src_dst_vls
#
#     return src_dst_arr

# def _gen_wts_dct_frm_crd_and_ply2(
#         self, src_xcs, src_ycs, dst_xcs, dst_ycs, src_ply_dct):
#
#     assert src_xcs.ndim == 2, src_xcs.ndim
#     assert src_xcs.size >= 2, src_xcs.shape
#     assert src_xcs.shape == src_ycs.shape, (src_xcs.shape, src_ycs.shape)
#
#     assert dst_xcs.ndim == 2, dst_xcs.ndim
#     assert dst_xcs.size >= 2, dst_xcs.shape
#     assert dst_xcs.shape == dst_ycs.shape, (dst_xcs.shape, dst_ycs.shape)
#
#     dst_ycs_dfs = dst_ycs[:-1,:] - dst_ycs[+1:,:]
#     dst_xcs_dfs = dst_xcs[:, 1:] - dst_xcs[:,:-1]
#
#     dst_cle_cdy = np.isclose(
#         dst_ycs_dfs, dst_ycs[0, 0] - dst_ycs[1, 0]).all()
#
#     dst_cle_cdx = np.isclose(
#         dst_xcs_dfs, dst_xcs[0, 1] - dst_xcs[0, 0]).all()
#
#     dst_cel_hgt = np.abs(dst_ycs_dfs).max()
#     dst_cel_wdh = np.abs(dst_xcs_dfs).max()
#
#     if dst_cle_cdy and dst_cle_cdx:
#         # dst_cle_cnd = True
#
#         if self._vb: print('Desti. cells are constant in size.')
#
#     else:
#         # dst_cle_cnd = False
#         pass
#
#     src_ycs_dfs = src_ycs[:-1,:] - src_ycs[+1:,:]
#     src_xcs_dfs = src_xcs[:, 1:] - src_xcs[:,:-1]
#
#     src_cle_cdy = np.isclose(
#         src_ycs_dfs, src_ycs[0, 0] - src_ycs[1, 0]).all()
#
#     src_cle_cdx = np.isclose(
#         src_xcs_dfs, src_xcs[0, 1] - src_xcs[0, 0]).all()
#
#     src_cel_hgt = np.abs(src_ycs_dfs).max()
#     src_cel_wdh = np.abs(src_xcs_dfs).max()
#
#     if src_cle_cdy and src_cle_cdx:
#         # src_cle_cnd = True
#
#         if self._vb: print('Source cells are constant in size.')
#
#     else:
#         # src_cle_cnd = False
#         pass
#
#     # src_rws_hgt_rel_crt = np.ceil(src_cel_hgt / dst_cel_hgt)
#     src_rws_hgt_rel_crt = int(src_cel_hgt / dst_cel_hgt)
#     src_rws_rtn_rel_crt = np.ceil(
#         max(np.abs(src_ycs[+0, +0] - src_ycs[+0, -1]),
#             np.abs(src_ycs[-1, +0] - src_ycs[-1, -1])) / src_cel_hgt)
#
#     # src_cns_wdh_rel_crt = np.ceil(src_cel_wdh / dst_cel_wdh)
#     src_cns_wdh_rel_crt = int(src_cel_wdh / dst_cel_wdh)
#     src_cns_rtn_rel_crt = np.ceil(
#         max(np.abs(src_xcs[-1, +0] - src_xcs[+0, +0]),
#             np.abs(src_xcs[-1, -1] - src_xcs[+0, -1])) / src_cel_wdh)
#
#     src_bfr_rws = int(max(0, src_rws_hgt_rel_crt + src_rws_rtn_rel_crt))
#     src_bfr_cns = int(max(0, src_cns_wdh_rel_crt + src_cns_rtn_rel_crt))
#
#     src_xmn_min = src_xcs[0:, 0].min()
#     src_ymx_max = src_ycs[0, 0:].max()
#
#     src_wts_dct = {}
#     for i in range(dst_xcs.shape[0] - 1):
#
#         if i == 0:
#             bt = et = default_timer()
#
#         if i == 1:
#             print(f'One row took {et - bt:0.2f} seconds.')
#
#             print(
#                 f'Estimated time to go through all the '
#                 f'{dst_xcs.shape[0]} rows is '
#                 f'{(et - bt) * (dst_xcs.shape[0] - 1):0.2f} seconds.')
#
#         assert np.unique(dst_ycs[i,:]).size == 1, dst_ycs[i,:]
#
#         dst_ymn = dst_ycs[i + 1, 0]
#         dst_ymx = dst_ycs[i + 0, 0]
#
#         src_beg_row_idx = int((src_ymx_max - dst_ymx) / src_cel_hgt)
#         src_end_row_idx = int(
#             np.ceil((src_ymx_max - dst_ymn) / src_cel_hgt))
#
#         if src_beg_row_idx > src_xcs.shape[0]: continue
#         if src_end_row_idx < 0: continue
#
#         # Buffer it roughly.
#         src_beg_row_idx = max(0, src_beg_row_idx - src_bfr_rws)
#         src_end_row_idx = min(
#             src_ycs.shape[0] - 1, src_end_row_idx + src_bfr_rws)
#
#         for j in range(dst_xcs.shape[1] - 1):
#
#             if i == 0:
#                 assert np.unique(dst_xcs[:, j]).size == 1, dst_xcs[:, j]
#
#             dst_xmn = dst_xcs[0, j + 0]
#             dst_xmx = dst_xcs[0, j + 1]
#
#             src_beg_col_idx = int((dst_xmn - src_xmn_min) / src_cel_wdh)
#             src_end_col_idx = int(
#                 np.ceil((dst_xmx - src_xmn_min) / src_cel_wdh))
#
#             if src_beg_col_idx > src_xcs.shape[1]:
#                 print(i, j, '>', src_beg_col_idx, dst_xcs.shape[1])
#                 continue
#
#             if src_end_col_idx < 0:
#                 print(i, j, '<', src_end_col_idx)
#                 continue
#
#             # Buffer it roughly.
#             src_beg_col_idx = max(0, src_beg_col_idx - src_bfr_cns)
#             src_end_col_idx = min(
#                 src_xcs.shape[1] - 1, src_end_col_idx + src_bfr_cns)
#             #==============================================================
#
#             if True:
#                 # Incrementation/decrementation can be used to be within
#                 # bounds here.
#
#                 # +1 is required to account for the extent of cell.
#                 src_xmn = src_xcs[
#                     src_beg_row_idx:src_end_row_idx + 1,
#                     src_beg_col_idx:src_end_col_idx + 1, ].min()
#
#                 src_xmx = src_xcs[
#                     src_beg_row_idx:src_end_row_idx + 1,
#                     src_beg_col_idx:src_end_col_idx + 1, ].max()
#
#                 src_ymn = src_ycs[
#                     src_beg_row_idx:src_end_row_idx + 1,
#                     src_beg_col_idx:src_end_col_idx + 1, ].min()
#
#                 src_ymx = src_ycs[
#                     src_beg_row_idx:src_end_row_idx + 1,
#                     src_beg_col_idx:src_end_col_idx + 1, ].max()
#
#                 assert dst_xmn >= src_xmn, (dst_xmn, src_xmn)
#                 assert dst_xmx <= src_xmx, (dst_xmx, src_xmx)
#
#                 assert dst_ymn >= src_ymn, (dst_ymn, src_ymn)
#                 assert dst_ymx <= src_ymx, (dst_ymx, src_ymx)
#             #==============================================================
#
#             rng = ogr.Geometry(ogr.wkbLinearRing)
#
#             rng.AddPoint(dst_xmn, dst_ymn)
#             rng.AddPoint(dst_xmx, dst_ymn)
#             rng.AddPoint(dst_xmx, dst_ymx)
#             rng.AddPoint(dst_xmn, dst_ymx)
#             rng.AddPoint(dst_xmn, dst_ymn)
#
#             ply = ogr.Geometry(ogr.wkbPolygon)
#             ply.AddGeometry(rng)
#
#             dst_ply_ara = ply.Area()
#             src_ply_ara = 0.0
#
#             ara_tot_ctr = 0
#             ara_mis_ctr = 0
#
#             wts_dct = {}
#             for k in range(src_beg_row_idx, src_end_row_idx):
#                 for l in range(src_beg_col_idx, src_end_col_idx):
#
#                     if (k, l) not in src_ply_dct: continue
#
#                     ara_tot_ctr += 1
#
#                     if False:
#                         src_ply_elp = src_ply_dct[(k, l)].GetEnvelope()
#
#                         src_xmn = src_ply_elp[0]
#                         src_xmx = src_ply_elp[1]
#
#                         src_ymn = src_ply_elp[2]
#                         src_ymx = src_ply_elp[3]
#
#                         assert dst_xmn >= src_xmn, (dst_xmn, src_xmn)
#                         assert dst_xmx <= src_xmx, (dst_xmx, src_xmx)
#
#                         assert dst_ymn >= src_ymn, (dst_ymn, src_ymn)
#                         assert dst_ymx <= src_ymx, (dst_ymx, src_ymx)
#
#                     itt_ara = ply.Intersection(src_ply_dct[(k, l)]).Area()
#
#                     if np.isclose(itt_ara, 0.0):
#                         ara_mis_ctr += 1
#                         continue
#
#                     wts_dct[(k, l)] = (itt_ara / dst_ply_ara)
#
#                     src_ply_ara += itt_ara
#
#             if False:
#                 print(
#                     i,
#                     j,
#                     ara_mis_ctr,
#                     ara_tot_ctr,
#                     f'{ara_mis_ctr / ara_tot_ctr:.2%}')
#
#             assert np.isclose(dst_ply_ara, src_ply_ara), (
#                 dst_ply_ara, src_ply_ara, f'{src_ply_ara/dst_ply_ara:.2%}')
#
#             src_wts_dct[(i, j)] = wts_dct
#
#         if i == 0:
#             et = default_timer()
#
#     return src_wts_dct
#
# def _gen_wts_dct_frm_crd_and_ply(
#         self, src_xcs, src_ycs, dst_xcs, dst_ycs, src_ply_dct):
#
#     assert src_xcs.ndim == 2, src_xcs.ndim
#     assert src_xcs.size >= 2, src_xcs.shape
#     assert src_xcs.shape == src_ycs.shape, (src_xcs.shape, src_ycs.shape)
#
#     assert dst_xcs.ndim == 2, dst_xcs.ndim
#     assert dst_xcs.size >= 2, dst_xcs.shape
#     assert dst_xcs.shape == dst_ycs.shape, (dst_xcs.shape, dst_ycs.shape)
#
#     src_wts_dct = {}
#     for i in range(dst_xcs.shape[0] - 1):
#
#         if i == 0:
#             bt = et = default_timer()
#
#         if i == 1:
#             print(f'One row took {et - bt:0.2f} seconds.')
#
#             print(
#                 f'Estimated time to go through all the '
#                 f'{dst_xcs.shape[0]} rows is '
#                 f'{(et - bt) * (dst_xcs.shape[0] - 1):0.2f} seconds.')
#
#         assert np.unique(dst_ycs[i,:]).size == 1, dst_ycs[i,:]
#
#         dst_ymn = dst_ycs[i + 1, 0]
#         dst_ymx = dst_ycs[i + 0, 0]
#
#         ymn_ixs = src_ycs <= dst_ymn
#         ymx_ixs = src_ycs >= dst_ymx
#
#         cnd_ymn = ymn_ixs.sum() > 0
#         cnd_ymx = ymx_ixs.sum() > 0
#
#         if not (cnd_ymn and cnd_ymx): continue
#
#         for j in range(dst_xcs.shape[1] - 1):
#
#             if i == 0:
#                 assert np.unique(dst_xcs[:, j]).size == 1, dst_xcs[:, j]
#
#             dst_xmn = dst_xcs[0, j + 0]
#             dst_xmx = dst_xcs[0, j + 1]
#
#             xmn_ixs = src_xcs <= dst_xmn
#             xmx_ixs = src_xcs >= dst_xmx
#
#             cnd_xmn = xmn_ixs.sum() > 0
#             cnd_xmx = xmx_ixs.sum() > 0
#
#             # Ys checked beforehand!
#             if not (cnd_xmn and cnd_xmx): continue
#
#             xmn_ymn_bls = xmn_ixs & ymn_ixs
#             xmn_ymx_bls = xmn_ixs & ymx_ixs
#             xmx_ymn_bls = xmx_ixs & ymn_ixs
#             xmx_ymx_bls = xmx_ixs & ymx_ixs
#
#             cnd_xmn_ymn = xmn_ymn_bls.sum() > 0
#             cnd_xmn_ymx = xmn_ymx_bls.sum() > 0
#             cnd_xmx_ymn = xmx_ymn_bls.sum() > 0
#             cnd_xmx_ymx = xmx_ymx_bls.sum() > 0
#
#             if not (
#                 cnd_xmn_ymn and
#                 cnd_xmn_ymx and
#                 cnd_xmx_ymn and
#                 cnd_xmx_ymx): continue
#
#             xmn_ymn_row = np.where(xmn_ymn_bls.sum(axis=1))[0][+0]
#             xmn_ymn_col = np.where(xmn_ymn_bls.sum(axis=0))[0][-1]
#
#             xmn_ymx_row = np.where(xmn_ymx_bls.sum(axis=1))[0][-1]
#             xmn_ymx_col = np.where(xmn_ymx_bls.sum(axis=0))[0][-1]
#
#             xmx_ymn_row = np.where(xmx_ymn_bls.sum(axis=1))[0][+0]
#             xmx_ymn_col = np.where(xmx_ymn_bls.sum(axis=0))[0][+0]
#
#             xmx_ymx_row = np.where(xmx_ymx_bls.sum(axis=1))[0][-1]
#             xmx_ymx_col = np.where(xmx_ymx_bls.sum(axis=0))[0][+0]
#
#             min_row = min(
#                 [
#                 xmn_ymn_row,
#                 xmn_ymx_row,
#                 xmx_ymn_row,
#                 xmx_ymx_row,
#                 ])
#
#             max_row = max(
#                 [
#                  xmn_ymn_row,
#                 xmn_ymx_row,
#                  xmx_ymn_row,
#                 xmx_ymx_row,
#                 ])
#
#             min_col = min(
#                 [xmn_ymn_col,
#                  xmn_ymx_col,
#                  xmx_ymn_col,
#                  xmx_ymx_col])
#
#             max_col = max(
#                 [xmn_ymn_col,
#                  xmn_ymx_col,
#                  xmx_ymn_col,
#                  xmx_ymx_col])
#
#             if False:
#
#                 if min_row > 0:
#                     min_row -= 1
#
#                 if min_col > 0:
#                     min_col -= 1
#
#                 if max_row < src_xcs.shape[0]:
#                     max_row += 1
#
#                 if max_col < src_xcs.shape[1]:
#                     max_col += 1
#
#             if False:
#                 print(i, j, max_row - min_row, max_col - min_col)
#
#             rng = ogr.Geometry(ogr.wkbLinearRing)
#
#             rng.AddPoint(dst_xmn, dst_ymn)
#             rng.AddPoint(dst_xmx, dst_ymn)
#             rng.AddPoint(dst_xmx, dst_ymx)
#             rng.AddPoint(dst_xmn, dst_ymx)
#             rng.AddPoint(dst_xmn, dst_ymn)
#
#             ply = ogr.Geometry(ogr.wkbPolygon)
#             ply.AddGeometry(rng)
#
#             dst_ply_ara = ply.Area()
#             src_ply_ara = 0.0
#
#             if False:
#                 # +1 is required to account for the extent of cell.
#                 src_xmn = src_xcs[
#                     min_row:max_row + 1, min_col:max_col + 1].min()
#
#                 src_xmx = src_xcs[
#                     min_row:max_row + 1, min_col:max_col + 1].max()
#
#                 src_ymn = src_ycs[
#                     min_row:max_row + 1, min_col:max_col + 1].min()
#
#                 src_ymx = src_ycs[
#                     min_row:max_row + 1, min_col:max_col + 1].max()
#
#                 assert dst_xmn >= src_xmn, (dst_xmn, src_xmn)
#                 assert dst_xmx <= src_xmx, (dst_xmx, src_xmx)
#
#                 assert dst_ymn >= src_ymn, (dst_ymn, src_ymn)
#
#                 assert dst_ymx <= src_ymx, (dst_ymx, src_ymx)
#
#             ara_tot_ctr = 0
#             ara_mis_ctr = 0
#
#             wts_dct = {}
#             for k in range(min_row, max_row):
#                 for l in range(min_col, max_col):
#
#                     ara_tot_ctr += 1
#
#                     itt_ara = ply.Intersection(src_ply_dct[(k, l)]).Area()
#
#                     if np.isclose(itt_ara, 0.0):
#                         ara_mis_ctr += 1
#                         continue
#
#                     wts_dct[(k, l)] = (itt_ara / dst_ply_ara)
#
#                     src_ply_ara += itt_ara
#
#             assert np.isclose(dst_ply_ara, src_ply_ara), (
#                 dst_ply_ara, src_ply_ara)
#
#             src_wts_dct[(i, j)] = wts_dct
#
#             if False:
#                 print(
#                     i,
#                     j,
#                     ara_mis_ctr,
#                     ara_tot_ctr,
#                     f'{ara_mis_ctr / ara_tot_ctr:.2%}')
#
#         if i == 0:
#             et = default_timer()
#
#     return src_wts_dct

# def _get_ply_dct_frm_crd(self, xcs, ycs):
#
#     assert xcs.ndim == 2, xcs.ndim
#     assert xcs.shape == ycs.shape, (xcs.shape, ycs.shape)
#
#     ply_dct = {}
#
#     ags = RTRARGS()
#
#     # MP does not work due to GDAL object being unpickleable.
#     if True:  # self._mpg_pol is None:
#         ags.mpg_flg = False
#
#         ags.xcs = xcs
#         ags.ycs = ycs
#
#         ags.ply_dct = ply_dct
#
#         _dtb = default_timer()
#
#         get_ply_dct_frm_crd_sgl(
#             ((i for i in range(xcs.shape[0] - 1)), ags))
#
#         _dte = default_timer()
#
#     else:
#         # Not worth it, for now!
#         ags.mpg_flg = True
#
#         ags.ply_dct = self._mpg_dct
#         ags.mpg_lck = self._mpg_lck
#
#         shm_ags = SHMARGS()
#         shm_arr_dct = {}
#
#         shm_arr_dct['xcs'] = xcs
#         shm_arr_dct['ycs'] = ycs
#         #==================================================================
#
#         init_shm_arrs(ags, shm_ags, shm_arr_dct)
#
#         mpg_ixs = ret_mp_idxs(xcs.shape[0] - 1, self._mpg_pol._processes)
#
#         iis = np.arange(xcs.shape[0] - 1)
#
#         # No shuffling needed here.
#         mpg_ags = (
#             (iis[i:j], ags) for i, j in zip(mpg_ixs[:-1], mpg_ixs[+1:]))
#
#         _dtb = default_timer()
#
#         list(self._mpg_pol.map(
#             get_ply_dct_frm_crd_sgl, mpg_ags, chunksize=1))
#
#         print('Re-reading...')
#         # This is too slow. I think the objects are recreated.
#         # Total waste of time. GDAL is faster but doies not allow pickling.
#         for key in self._mpg_dct: ply_dct.update(self._mpg_dct[key])
#
#             # del self._mpg_dct[key]
#
#         self._mpg_dct.clear()
#
#         _dte = default_timer()
#
#         print(f'get_ply_dct_frm_crd_sgl: {_dte - _dtb:0.2f} secs!')
#         #==================================================================
#
#         for key, val in shm_arr_dct.items():
#
#             if key not in ('xxx'): continue
#
#             if hasattr(ags, key): continue
#
#             val[:] = get_shm_arr(ags, key)
#
#         free_shm_arrs(shm_ags)
#         #==================================================================
#
#         raise Exception
#
#     return ply_dct

# def _get_ply_dct_frm_crd(self, xcs, ycs):
#
#     assert xcs.ndim == 2, xcs.ndim
#     assert xcs.shape == ycs.shape, (xcs.shape, ycs.shape)
#
#     rows, cols = xcs.shape
#
#     ply_dct = {}
#     for i in range(rows - 1):
#         for j in range(cols - 1):
#             rng = ogr.Geometry(ogr.wkbLinearRing)
#
#             rng.AddPoint(xcs[i, j], ycs[i, j])
#             rng.AddPoint(xcs[i, j + 1], ycs[i, j + 1])
#             rng.AddPoint(xcs[i + 1, j + 1], ycs[i + 1, j + 1])
#             rng.AddPoint(xcs[i + 1, j], ycs[i + 1, j])
#             rng.AddPoint(xcs[i, j], ycs[i, j])
#
#             ply = ogr.Geometry(ogr.wkbPolygon)
#             ply.AddGeometry(rng)
#
#             ply_dct[(i, j)] = ply
#
#     return ply_dct
