# -*- coding: utf-8 -*-

'''
Created on 08.07.2024

@author: Faizan-TU Munich
'''

from pathlib import Path

import numpy as np
from osgeo import ogr, gdal
from pyproj import Transformer

from ..extract import ExtractGTiffCoords
from ..misc import print_sl, print_el

ogr.UseExceptions()
gdal.UseExceptions()


class ResampleRasToRas:

    def __init__(self, verbose=True):

        assert isinstance(verbose, bool), type(verbose)

        self._vb = verbose

        # SRC is resampled to become DST.
        self._src_pth = None
        self._dst_pth = None

        self._inp_set_flg = False
        self._otp_set_flg = False
        return

    def set_inputs(self, path_to_src, path_to_dst):

        if self._vb:
            print_sl()

            print('Setting resampling inputs...')

        assert isinstance(path_to_src, Path), type(path_to_src)
        assert path_to_src.exists(), path_to_src

        self._src_pth = path_to_src.absolute()

        assert isinstance(path_to_dst, Path), type(path_to_dst)
        assert path_to_dst.exists(), path_to_dst

        self._dst_pth = path_to_dst.absolute()

        if self._vb:
            print(f'INFO: Set the following parameters for the inputs:')

            print(f'Source path: {self._src_pth}')
            print(f'Desti. path: {self._dst_pth}')

            print_el()

        self._inp_set_flg = True
        return

    def set_outputs(self, path_to_out):

        if self._vb:
            print_sl()

            print('Setting resampling outputs...')

        assert isinstance(path_to_out, Path), type(path_to_out)

        self._out_pth = path_to_out.absolute()

        if self._vb:
            print(f'INFO: Set the following parameters for the outputs:')

            print(f'Output path: {self._out_pth}')

            print_el()

        self._otp_set_flg = True
        return

    def resample(self):

        # TODO: Test with a single column of single row raster.

        assert self._inp_set_flg
        assert self._otp_set_flg

        src_crd_obj = self._get_crd_obj(self._src_pth)
        dst_crd_obj = self._get_crd_obj(self._dst_pth)

        src_xcs = src_crd_obj.get_x_coordinates()
        src_ycs = src_crd_obj.get_y_coordinates()

        src_xcs, src_ycs = self._get_crd_msh(src_xcs, src_ycs)

        dst_xcs = dst_crd_obj.get_x_coordinates()
        dst_ycs = dst_crd_obj.get_y_coordinates()

        dst_xcs, dst_ycs = self._get_crd_msh(dst_xcs, dst_ycs)
        #======================================================================

        if self._vb:
            print_sl()

        src_dst_tfm = Transformer.from_crs(
            self._get_ras_crs(self._src_pth),
            self._get_ras_crs(self._dst_pth),
            always_xy=True)

        if self._vb: print('Transforming source coordinates\' mesh...')

        self._tfm_msh(src_xcs, src_ycs, src_dst_tfm)

        if self._vb: print(f'Source original mesh shape: {src_xcs.shape}...')
        if self._vb: print(f'Desti. original mesh shape: {dst_xcs.shape}...')
        #======================================================================

        (src_wts_dct,
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

        src_dst_arr = self._get_wtd_arr(src_arr, dst_arr, dst_xcs, src_wts_dct)

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

        if self._vb:
            print_el()
        return

    def _cvt_ras_fle(self, ref_pth, out_pth, ras_arr, ref_xmn, ref_ymx):

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

        for i in range(ras_arr.shape[0]):
            out_bnd = out_ras.GetRasterBand(i + 1)
            out_bnd.WriteArray(ras_arr[i,:,:])
            out_bnd.SetNoDataValue(np.nan)

        out_ras.SetGeoTransform(
            [ref_xmn, ref_gtm[1], 0, ref_ymx, 0, ref_gtm[5]])

        out_ras.SetProjection(ref_ras.GetProjection())

        out_ras = ref_ras = None
        return

    def _get_wtd_arr(self, src_arr, dst_arr, dst_xcs, src_wts_dct):

        src_dst_arr = np.full(
            (src_arr.shape[0], dst_xcs.shape[0] - 1, dst_xcs.shape[1] - 1),
            np.nan,
            dtype=np.float32)

        assert dst_arr.shape[1:] == src_dst_arr.shape[1:], (
                dst_arr.shape, src_dst_arr.shape)

        for m in range(src_arr.shape[0]):
            for (i, j), wts_dct in src_wts_dct.items():

                if np.isnan(dst_arr[0, i, j]): continue

                src_dst_val = 0.0
                for (k, l), wt in wts_dct.items():
                    src_dst_val += wt * src_arr[m, k, l]

                src_dst_arr[m, i, j] = src_dst_val

        return src_dst_arr

    def _get_ras_vrs(
            self, inp_pth, beg_row, end_row, beg_col, end_col, bnd_cnt):

        ras = gdal.Open(str(inp_pth))

        assert ras is not None

        if bnd_cnt is None:
            bnd_cnt = ras.RasterCount

        else:
            assert bnd_cnt >= ras.RasterCount

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

    def _get_wts_vrs(self, src_xcs, src_ycs, dst_xcs, dst_ycs):

        '''
        Coordinates are 2D of cell corners.

        src and dst have same crs.
        '''

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
        #======================================================================

        src_xcs, src_ycs = (
            src_xcs[src_beg_row:src_end_row, src_beg_col:src_end_col],
            src_ycs[src_beg_row:src_end_row, src_beg_col:src_end_col])

        dst_xcs, dst_ycs = (
            dst_xcs[dst_beg_row:dst_end_row, dst_beg_col:dst_end_col],
            dst_ycs[dst_beg_row:dst_end_row, dst_beg_col:dst_end_col])

        if self._vb: print(f'Source snipped mesh shape: {src_xcs.shape}...')
        if self._vb: print(f'Desti. snipped mesh shape: {dst_xcs.shape}...')

        src_xmn = src_xcs.min()
        src_xmx = src_xcs.max()

        src_ymn = src_ycs.min()
        src_ymx = src_ycs.max()

        dst_xmn = dst_xcs.min()
        dst_xmx = dst_xcs.max()

        dst_ymn = dst_ycs.min()
        dst_ymx = dst_ycs.max()

        assert dst_xmn >= src_xmn, (dst_xmn, src_xmn)
        assert dst_xmx <= src_xmx, (dst_xmx, src_xmx)

        assert dst_ymn >= src_ymn, (dst_ymn, src_ymn)
        assert dst_ymx <= src_ymx, (dst_ymx, src_ymx)
        #======================================================================

        if self._vb: print('Creating source cell polygons...')

        src_ply_dct = self._get_ply_dct_frm_crd(src_xcs, src_ycs)

        if self._vb: print('Computing weights for resampling...')

        src_wts_dct = self._gen_wts_dct_frm_crd_and_ply(
            src_xcs, src_ycs, dst_xcs, dst_ycs, src_ply_dct)

        return (
            src_wts_dct,
            (src_beg_row,
             src_end_row,
             src_beg_col,
             src_end_col),
            (dst_beg_row,
             dst_end_row,
             dst_beg_col,
             dst_end_col))

    def _gen_wts_dct_frm_crd_and_ply(
            self, src_xcs, src_ycs, dst_xcs, dst_ycs, src_ply_dct):

        assert src_xcs.ndim == 2, src_xcs.ndim
        assert src_xcs.size >= 2, src_xcs.shape
        assert src_xcs.shape == src_ycs.shape, (src_xcs.shape, src_ycs.shape)

        # assert src_xcs.shape == dst_xcs.shape, (src_xcs.shape, dst_xcs.shape)
        # assert dst_xcs.shape == dst_ycs.shape, (dst_xcs.shape, dst_ycs.shape)

        assert dst_xcs.ndim == 2, dst_xcs.ndim
        assert dst_xcs.size >= 2, dst_xcs.shape
        assert dst_xcs.shape == dst_ycs.shape, (dst_xcs.shape, dst_ycs.shape)

        src_wts_dct = {}
        for i in range(dst_xcs.shape[0] - 1):

            assert np.unique(dst_ycs[i,:]).size == 1, dst_ycs[i,:]

            dst_ymn = dst_ycs[i + 1, 0]
            dst_ymx = dst_ycs[i + 0, 0]

            ymn_ixs = src_ycs <= dst_ymn
            ymx_ixs = src_ycs >= dst_ymx

            cnd_ymn = ymn_ixs.sum() > 0
            cnd_ymx = ymx_ixs.sum() > 0

            if not (cnd_ymn and cnd_ymx): continue

            for j in range(dst_xcs.shape[1] - 1):

                if i == 0:
                    assert np.unique(dst_xcs[:, j]).size == 1, dst_xcs[:, j]

                dst_xmn = dst_xcs[0, j + 0]
                dst_xmx = dst_xcs[0, j + 1]

                xmn_ixs = src_xcs <= dst_xmn
                xmx_ixs = src_xcs >= dst_xmx

                cnd_xmn = xmn_ixs.sum() > 0
                cnd_xmx = xmx_ixs.sum() > 0

                if not (cnd_ymn and cnd_ymx and cnd_xmn and cnd_xmx): continue

                xmn_ymn_bls = xmn_ixs & ymn_ixs
                xmn_ymx_bls = xmn_ixs & ymx_ixs
                xmx_ymn_bls = xmx_ixs & ymn_ixs
                xmx_ymx_bls = xmx_ixs & ymx_ixs

                cnd_xmn_ymn = xmn_ymn_bls.sum() > 0
                cnd_xmn_ymx = xmn_ymx_bls.sum() > 0
                cnd_xmx_ymn = xmx_ymn_bls.sum() > 0
                cnd_xmx_ymx = xmx_ymx_bls.sum() > 0

                if not (
                    cnd_xmn_ymn and
                    cnd_xmn_ymx and
                    cnd_xmx_ymn and
                    cnd_xmx_ymx): continue

                # min is max and max is min!
                xmn_ymn_row = np.where(xmn_ymn_bls.sum(axis=1))[0][+0]
                xmn_ymn_col = np.where(xmn_ymn_bls.sum(axis=0))[0][-1]

                xmn_ymx_row = np.where(xmn_ymx_bls.sum(axis=1))[0][-1]
                xmn_ymx_col = np.where(xmn_ymx_bls.sum(axis=0))[0][-1]

                xmx_ymn_row = np.where(xmx_ymn_bls.sum(axis=1))[0][+0]
                xmx_ymn_col = np.where(xmx_ymn_bls.sum(axis=0))[0][+0]

                xmx_ymx_row = np.where(xmx_ymx_bls.sum(axis=1))[0][-1]
                xmx_ymx_col = np.where(xmx_ymx_bls.sum(axis=0))[0][+0]

                min_row = min(
                    [
                     # xmn_ymn_row,
                    xmn_ymx_row,
                     # xmx_ymn_row,
                    xmx_ymx_row,
                    ])

                max_row = max(
                    [
                     xmn_ymn_row,
                     # xmn_ymx_row,
                     xmx_ymn_row,
                     # xmx_ymx_row,
                    ])

                min_col = min(
                    [xmn_ymn_col,
                     xmn_ymx_col,
                     xmx_ymn_col,
                     xmx_ymx_col])

                max_col = max(
                    [xmn_ymn_col,
                     xmn_ymx_col,
                     xmx_ymn_col,
                     xmx_ymx_col])

                if False:

                    if min_row > 0:
                        min_row -= 1

                    if min_col > 0:
                        min_col -= 1

                    if max_row < src_xcs.shape[0]:
                        max_row += 1

                    if max_col < src_xcs.shape[1]:
                        max_col += 1

                if False:
                    print(i, j, max_row - min_row, max_col - min_col)

                rng = ogr.Geometry(ogr.wkbLinearRing)

                rng.AddPoint(dst_xmn, dst_ymn)
                rng.AddPoint(dst_xmx, dst_ymn)
                rng.AddPoint(dst_xmx, dst_ymx)
                rng.AddPoint(dst_xmn, dst_ymx)
                rng.AddPoint(dst_xmn, dst_ymn)

                ply = ogr.Geometry(ogr.wkbPolygon)
                ply.AddGeometry(rng)

                dst_ply_ara = ply.Area()
                src_ply_ara = 0.0

                # +1 is required to account for the extent of cell.
                src_xmn = src_xcs[
                    min_row:max_row + 1, min_col:max_col + 1].min()

                src_xmx = src_xcs[
                    min_row:max_row + 1, min_col:max_col + 1].max()

                src_ymn = src_ycs[
                    min_row:max_row + 1, min_col:max_col + 1].min()

                src_ymx = src_ycs[
                    min_row:max_row + 1, min_col:max_col + 1].max()

                # How to account for the area of
                assert dst_xmn >= src_xmn, (dst_xmn, src_xmn)
                assert dst_xmx <= src_xmx, (dst_xmx, src_xmx)

                assert dst_ymn >= src_ymn, (
                    dst_ymn, src_ymn)

                assert dst_ymx <= src_ymx, (dst_ymx, src_ymx)

                wts_dct = {}
                for k in range(min_row, max_row):
                    for l in range(min_col, max_col):

                        itt_ara = ply.Intersection(src_ply_dct[(k, l)]).Area()

                        if np.isclose(itt_ara, 0.0): continue

                        wts_dct[(k, l)] = (itt_ara / dst_ply_ara)

                        src_ply_ara += itt_ara

                assert np.isclose(dst_ply_ara, src_ply_ara), (
                    dst_ply_ara, src_ply_ara)

                src_wts_dct[(i, j)] = wts_dct

        return src_wts_dct

    def _get_ply_dct_frm_crd(self, xcs, ycs):

        assert xcs.ndim == 2, xcs.ndim
        assert xcs.shape == ycs.shape, (xcs.shape, ycs.shape)

        rows, cols = xcs.shape

        ply_dct = {}
        for i in range(rows - 1):
            for j in range(cols - 1):
                rng = ogr.Geometry(ogr.wkbLinearRing)

                rng.AddPoint(xcs[i, j], ycs[i, j])
                rng.AddPoint(xcs[i, j + 1], ycs[i, j + 1])
                rng.AddPoint(xcs[i + 1, j + 1], ycs[i + 1, j + 1])
                rng.AddPoint(xcs[i + 1, j], ycs[i + 1, j])
                rng.AddPoint(xcs[i, j], ycs[i, j])

                ply = ogr.Geometry(ogr.wkbPolygon)
                ply.AddGeometry(rng)

                ply_dct[(i, j)] = ply

        return ply_dct

    def _get_ply_grd_ixs(self, ply, xcs, ycs):

        xcs_min, xcs_max, ycs_min, ycs_max = ply.GetEnvelope()

        xcs_min_ixs = xcs <= xcs_min
        xcs_max_ixs = xcs >= xcs_max

        ycs_min_ixs = ycs <= ycs_min
        ycs_max_ixs = ycs >= ycs_max
        #======================================================================

        # Upper-left corner.
        xmn_ymx_row = np.where(
            (xcs_min_ixs & ycs_max_ixs).any(axis=1))[0].max()

        xmn_ymx_col = np.where(
            (xcs_min_ixs & ycs_max_ixs).any(axis=0))[0].max()

        # Lower-left corner.
        xmn_ymn_row = np.where(
            (xcs_min_ixs & ycs_min_ixs).any(axis=1))[0].min()

        xmn_ymn_col = np.where(
            (xcs_min_ixs & ycs_min_ixs).any(axis=0))[0].max()

        # Upper-right corner.
        xmx_ymx_row = np.where(
            (xcs_max_ixs & ycs_max_ixs).any(axis=1))[0].max()

        xmx_ymx_col = np.where(
            (xcs_max_ixs & ycs_max_ixs).any(axis=0))[0].min()

        # Lower-right corner.
        xmx_ymn_row = np.where(
            (xcs_max_ixs & ycs_min_ixs).any(axis=1))[0].min()

        xmx_ymn_col = np.where(
            (xcs_max_ixs & ycs_min_ixs).any(axis=0))[0].min()
        #======================================================================

        # X bounds checks.
        assert xcs[xmn_ymx_row, xmn_ymx_col] <= xcs_min, (
            xcs[xmn_ymx_row, xmn_ymx_col], xcs_min)

        assert xcs[xmn_ymn_row, xmn_ymn_col] <= xcs_min, (
            xcs[xmn_ymn_row, xmn_ymn_col], xcs_min)

        assert xcs[xmx_ymx_row, xmx_ymx_col] >= xcs_max, (
            xcs[xmx_ymx_row, xmx_ymx_col], xcs_max)

        assert xcs[xmx_ymn_row, xmx_ymn_col] >= xcs_max, (
            xcs[xmx_ymn_row, xmx_ymn_col], xcs_max)

        # Y bounds checks.
        assert ycs[xmn_ymn_row, xmn_ymn_col] <= ycs_min, (
            ycs[xmn_ymx_row, xmn_ymx_col], ycs_min)

        assert ycs[xmx_ymn_row, xmx_ymn_col] <= ycs_min, (
            ycs[xmx_ymn_row, xmx_ymn_col], ycs_min)

        assert ycs[xmn_ymx_row, xmn_ymx_col] >= ycs_max, (
            ycs[xmn_ymx_row, xmn_ymx_col], ycs_max)

        assert ycs[xmx_ymx_row, xmx_ymx_col] >= ycs_max, (
            ycs[xmx_ymx_row, xmx_ymx_col], ycs_max)
        #======================================================================

        # Row bounds checks.
        assert xmn_ymx_row <= xmn_ymn_row, (xmn_ymx_row, xmn_ymn_row)
        assert xmx_ymx_row <= xmx_ymn_row, (xmx_ymx_row, xmx_ymn_row)
        assert xmn_ymx_row <= xmx_ymn_row, (xmn_ymx_row, xmx_ymn_row)

        # Column bounds checks.
        assert xmn_ymx_col <= xmx_ymx_col, (xmn_ymx_col, xmx_ymx_col)
        assert xmn_ymn_col <= xmx_ymn_col, (xmn_ymn_col, xmx_ymn_col)
        assert xmn_ymx_col <= xmn_ymn_col, (xmn_ymx_col, xmn_ymn_col)
        #======================================================================

        # Final row indices.
        beg_row = min(xmn_ymx_row, xmx_ymx_row)
        end_row = max(xmx_ymn_row, xmx_ymn_row) + 1

        # Final column indices.
        beg_col = min(xmn_ymn_col, xmn_ymx_col)
        end_col = min(xmx_ymn_col, xmx_ymn_col) + 1

        # Sanity check. Should always pass.
        assert xcs[beg_row:end_row, beg_col:end_col].size >= 2

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

        for i in range(xcs.shape[0]):
            for j in range(ycs.shape[0]):
                xcs[i, j], ycs[i, j] = src_dst_tfm.transform(
                    xcs[i, j], ycs[i, j])

        return

    def _get_crd_msh(self, xcs, ycs):

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

    def _get_crd_obj(self, inp_pth):

        ras_crd_obj = ExtractGTiffCoords(verbose=self._vb)

        ras_crd_obj.set_input(inp_pth)
        ras_crd_obj.extract_coordinates()

        return ras_crd_obj
