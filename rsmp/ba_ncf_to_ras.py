# -*- coding: utf-8 -*-

'''
Created on 09.07.2024

@author: Faizan-TU Munich
'''

from pathlib import Path
from multiprocessing import Manager, Pool as MPPool

import numpy as np
import netCDF4 as nc
from osgeo import ogr, gdal
from pyproj import Transformer
from pyproj.crs import CRS

from ..extract import ExtractNetCDFCoords

from ..mpg import DummyLock

from ..misc import print_sl, print_el

ogr.UseExceptions()
gdal.UseExceptions()

from .aa_ras_to_ras import ResampleRasToRas


class ResampleNCFToRas(ResampleRasToRas):

    def __init__(self, verbose=True):

        ResampleRasToRas.__init__(self, verbose)

        self._src_vrs = None
        self._src_xlb = None
        self._src_ylb = None
        self._src_tlb = None
        self._src_crs = None

        self._src_flp_flg = False

        self._ncf_dim_xlb_1dn = '_dimx_1D'
        self._ncf_dim_ylb_1dn = '_dimy_1D'

        self._ncf_dim_xlb_2dn = '_dimx_2D'
        self._ncf_dim_ylb_2dn = '_dimy_2D'

        self._ncf_dim_tlb = '_dimt'

        self._ncf_var_xlb_1dn = 'X1D'
        self._ncf_var_ylb_1dn = 'Y1D'
        self._ncf_var_xlb_2dn = 'X2D'
        self._ncf_var_ylb_2dn = 'Y2D'

        self._ncf_var_tlb = 'time'
        return

    def set_inputs(
            self,
            path_to_src,
            path_to_dst,
            n_cpus,
            src_vrs,
            src_xlb,
            src_ylb,
            src_tlb,
            src_crs):

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

            # No need advantage by using MP.
            n_cpus = 1  # get_n_cpus()

        else:
            assert isinstance(n_cpus, int), 'n_cpus is not an integer!'

            assert n_cpus > 0, 'Invalid n_cpus!'

        n_cpus = 1

        self._mpg_ncs = n_cpus

        # NCF Variables.
        assert isinstance(src_vrs, (list, tuple)), type(src_vrs)
        assert len(src_vrs)

        self._src_vrs = src_vrs

        # NCF X coordinates.
        assert isinstance(src_xlb, str), type(src_xlb)

        self._src_xlb = src_xlb

        # NCF Y coordinates.
        assert isinstance(src_ylb, str), type(src_ylb)

        self._src_ylb = src_ylb

        # NCF Time.
        assert isinstance(src_tlb, str), type(src_tlb)

        self._src_tlb = src_tlb

        # NCF coordinate system.
        assert isinstance(src_crs, CRS), type(src_crs)

        self._src_crs = src_crs
        #======================================================================

        if self._vb:
            print(f'INFO: Set the following parameters for the inputs:')

            print(f'Source path: {self._src_pth}')
            print(f'Desti. path: {self._dst_pth}')
            print(f'No. of Procs: {self._mpg_ncs}')
            print(f'Source variables: {self._src_vrs}')
            print(f'Source X label: {self._src_xlb}')
            print(f'Source Y label: {self._src_ylb}')
            print(f'Source time label: {self._src_tlb}')
            print(f'Source CRS: {self._src_crs}')

            print_el()

        self._inp_set_flg = True
        return

    def resample(self):

        assert self._inp_set_flg
        assert self._otp_set_flg

        # NOTE: Assuming coordinates are center when 1D and corner when 2D!
        src_crd_obj = self._get_crd_obj_ncf(self._src_pth)

        # GTiff are always corner.
        dst_crd_obj = self._get_crd_obj_ras(self._dst_pth)

        src_xcs = src_crd_obj.get_x_coordinates()
        src_ycs = src_crd_obj.get_y_coordinates()

        if src_xcs.ndim == 1:
            assert src_xcs[0] < src_xcs[1], src_xcs

            src_xcs = np.concatenate((
                (src_xcs - (0.5 * (src_xcs[1] - src_xcs[0]))),
                (src_xcs[-1] + (0.5 * (src_xcs[1] - src_xcs[0])),),
                ))

            # Code written while assuming y goes low when rows go high.
            if src_ycs[0] < src_ycs[1]:
                self._src_flp_flg = True

                src_ycs = src_ycs[::-1]

            src_ycs = np.concatenate((
                (src_ycs - (0.5 * (src_ycs[0] - src_ycs[1]))),
                (src_ycs[-1] + (0.5 * (src_ycs[1] - src_ycs[0])),),
                ))

        elif src_xcs.ndim == 2:

            raise NotImplementedError('Too much work!')

            assert src_xcs[0, 0] < src_xcs[0, 1], src_xcs

            if src_ycs[0, 0] < src_ycs[1, 0]:
                self._src_flp_flg = True

                src_ycs = src_ycs[::-1,:]

        else:
            raise NotImplementedError(src_xcs.ndim)

        self._vrf_ncf(
            self._src_pth, self._src_vrs, self._src_tlb, src_xcs, src_ycs)

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
            self._src_crs,
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

        src_tcs, src_tcs_unt, src_tcs_cal = self._get_ncf_tvs(
            self._src_pth, self._src_tlb)

        dst_xcs_1dn = dst_xcs[0,:-1] + (0.5 * (dst_xcs[0, 1] - dst_xcs[0, 0]))

        assert dst_xcs[+0, +0] < dst_xcs_1dn[+0] < dst_xcs[+0, +1]
        assert dst_xcs[-1, +0] < dst_xcs_1dn[+0] < dst_xcs[-1, +1]
        assert dst_xcs[+0, -2] < dst_xcs_1dn[-1] < dst_xcs[+0, -1]
        assert dst_xcs[-1, -2] < dst_xcs_1dn[-1] < dst_xcs[-1, -1]

        dst_ycs_1dn = dst_ycs[:-1, 0] + (0.5 * (dst_ycs[1, 0] - dst_ycs[0, 0]))

        if dst_ycs[0, 0] < dst_ycs[1, 0]:
            assert dst_ycs[+0, +0] < dst_ycs_1dn[+0] < dst_ycs[+1, +0]
            assert dst_ycs[-2, +0] < dst_ycs_1dn[-1] < dst_ycs[-1, +0]
            assert dst_ycs[+0, -1] < dst_ycs_1dn[+0] < dst_ycs[+1, -1]
            assert dst_ycs[-2, -1] < dst_ycs_1dn[-1] < dst_ycs[-1, -1]

        else:
            assert dst_ycs[+1, +0] < dst_ycs_1dn[+0] < dst_ycs[+0, +0]
            assert dst_ycs[-1, +0] < dst_ycs_1dn[-1] < dst_ycs[-2, +0]
            assert dst_ycs[+1, -1] < dst_ycs_1dn[+0] < dst_ycs[+0, -1]
            assert dst_ycs[-1, -1] < dst_ycs_1dn[-1] < dst_ycs[-2, -1]

        self._int_ncf(
            dst_xcs_1dn,
            dst_ycs_1dn,
            dst_xcs,
            dst_ycs,
            src_tcs,
            src_tcs_unt,
            src_tcs_cal)

        if self._src_flp_flg:

            if self._vb: print('Verifying Y coordinates\' flip...')

            self._vrf_flp(
                self._src_xlb,
                self._src_ylb,
                self._src_pth,
                src_beg_row,
                src_end_row,
                src_beg_col,
                src_end_col,
                src_xcs,
                src_ycs,
                src_dst_tfm)

        for var in self._src_vrs:

            if self._vb:
                print('\n')
                print(f'NCF Variable: {var}')

            src_arr, = self._get_ncf_vrs(
                self._src_pth,
                src_beg_row,
                src_end_row,
                src_beg_col,
                src_end_col,
                var,
                self._src_flp_flg)
            #==================================================================

            if self._vb: print('Computing resampled array...')

            src_dst_arr = self._get_wtd_arr_ncf(
                src_arr, dst_arr, dst_xcs, src_wts_dct)

            src_arr = None

            if self._vb:
                print(
                    f'Source ({var}) transformed array shape: '
                    f'{src_dst_arr.shape}...')
            #==================================================================

            if self._vb: print('Saving resampled netCDF...')

            self._cvt_ncf_fle(var, src_dst_arr)

            src_dst_arr = None

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

    def _cvt_ncf_fle(self, var, arr):

        cmp_lvl = 1

        ncf_hdl = nc.Dataset(self._out_pth, 'r+')

        ncf_hdl.set_auto_mask(False)

        ncf_var = ncf_hdl.createVariable(
            var,
            'd',
            dimensions=(self._ncf_dim_tlb,
                        self._ncf_dim_ylb_1dn,
                        self._ncf_dim_xlb_1dn,),
            fill_value=False,
            compression='zlib',
            complevel=cmp_lvl,
            chunksizes=(1, arr.shape[1], arr.shape[2]))

        ncf_var[:] = arr

        ncf_hdl.close()
        return

    def _get_wtd_arr_ncf(self, src_arr, dst_arr, dst_xcs, src_wts_dct):

        src_dst_arr = self._get_wtd_arr_ras(
            src_arr, dst_arr, dst_xcs, src_wts_dct)

        return src_dst_arr

    def _int_ncf(
            self, xcs_1dn, ycs_1dn, xcs_2dn, ycs_2dn, tcs, tcs_unt, tcs_cal):

        ncf_hdl = nc.Dataset(self._out_pth, 'w')

        ncf_hdl.set_auto_mask(False)

        ncf_hdl.createDimension(self._ncf_dim_xlb_1dn, xcs_1dn.shape[0])
        ncf_hdl.createDimension(self._ncf_dim_ylb_1dn, ycs_1dn.shape[0])

        ncf_hdl.createDimension(self._ncf_dim_xlb_2dn, xcs_2dn.shape[1])
        ncf_hdl.createDimension(self._ncf_dim_ylb_2dn, xcs_2dn.shape[0])

        ncf_hdl.createDimension(self._ncf_dim_tlb, tcs.shape[0])

        ncf_xcs_1dn = ncf_hdl.createVariable(
            self._ncf_var_xlb_1dn,
            'd',
            dimensions=(self._ncf_dim_xlb_1dn,))

        ncf_ycs_1dn = ncf_hdl.createVariable(
            self._ncf_var_ylb_1dn,
            'd',
            dimensions=(self._ncf_dim_ylb_1dn,))

        ncf_xcs_2dn = ncf_hdl.createVariable(
            self._ncf_var_xlb_2dn,
            'd',
            dimensions=(self._ncf_dim_ylb_2dn, self._ncf_dim_xlb_2dn,))

        ncf_ycs_2dn = ncf_hdl.createVariable(
            self._ncf_var_ylb_2dn,
            'd',
            dimensions=(self._ncf_dim_ylb_2dn, self._ncf_dim_xlb_2dn,))

        ncf_tcs = ncf_hdl.createVariable(
            self._ncf_var_tlb,
            'i8',
            dimensions=(self._ncf_dim_tlb,))

        ncf_xcs_1dn[:] = xcs_1dn
        ncf_ycs_1dn[:] = ycs_1dn
        ncf_xcs_2dn[:] = xcs_2dn
        ncf_ycs_2dn[:] = ycs_2dn
        ncf_tcs[:] = tcs

        ncf_tcs.units = tcs_unt
        ncf_tcs.calendar = tcs_cal

        ncf_hdl.close()
        return

    def _vrf_flp(
            self,
            xlb,
            ylb,
            inp_pth,
            beg_row,
            end_row,
            beg_col,
            end_col,
            ref_xcs,
            ref_ycs,
            crs_tfm):

        '''
        Enter when flip is True for Y.
        '''

        beg_row, end_row = -end_row - 0, -beg_row - 1

        with nc.Dataset(inp_pth) as ncf_hdl:

            if ncf_hdl[xlb].ndim == 1:
                tst_xcs = ncf_hdl[xlb][beg_col:end_col - 1].data
                tst_ycs = ncf_hdl[ylb][beg_row:end_row].data

                tst_ycs = tst_ycs[::-1]

                tst_xcs = np.concatenate((
                    (tst_xcs - (0.5 * (tst_xcs[1] - tst_xcs[0]))),
                    (tst_xcs[-1] + (0.5 * (tst_xcs[1] - tst_xcs[0])),),
                    ))

                tst_ycs = np.concatenate((
                    (tst_ycs + (0.5 * (tst_ycs[0] - tst_ycs[1]))),
                    (tst_ycs[-1] + (0.5 * (tst_ycs[1] - tst_ycs[0])),),
                    ))

                tst_xcs, tst_ycs = self._get_crd_msh(tst_xcs, tst_ycs)

            elif ncf_hdl[xlb].ndim == 2:
                tst_xcs = ncf_hdl[xlb][
                    beg_row:end_row, beg_col:end_col - 1].data

                tst_ycs = ncf_hdl[ylb][
                    beg_row:end_row, beg_col:end_col - 1].data

                tst_ycs = tst_ycs[::-1,:]

            else:
                raise NotImplementedError(ncf_hdl[xlb].ndim)

        assert ref_xcs.shape == tst_xcs.shape, (ref_xcs.shape, tst_xcs.shape)
        assert ref_ycs.shape == tst_ycs.shape, (ref_ycs.shape, tst_ycs.shape)

        self._tfm_msh(tst_xcs, tst_ycs, crs_tfm)

        # These can fail if there is significant roation of the grid.
        # More likely to happen whe source grid is fine.
        assert np.isclose(ref_xcs.min(), tst_xcs.min()), (
            ref_xcs.min(), tst_xcs.min())

        assert np.isclose(ref_xcs.max(), tst_xcs.max()), (
            ref_xcs.max(), tst_xcs.max())

        assert np.isclose(ref_ycs.min(), tst_ycs.min()), (
            ref_ycs.min(), tst_ycs.min())

        assert np.isclose(ref_ycs.max(), tst_ycs.max()), (
            ref_ycs.max(), tst_ycs.max())

        return

    def _get_ncf_tvs(self, inp_pth, tlb):

        with nc.Dataset(inp_pth) as ncf_hdl:

            ncf_tcs = ncf_hdl[tlb]

            tcs = ncf_tcs[:].data

            tcs_unt = ncf_tcs.units
            tcs_cal = ncf_tcs.calendar

        return (tcs, tcs_unt, tcs_cal)

    def _get_ncf_vrs(
            self,
            inp_pth,
            beg_row,
            end_row,
            beg_col,
            end_col,
            var,
            flp_flg):

        with nc.Dataset(inp_pth) as ncf_hdl:

            if flp_flg:
                beg_row, end_row = -end_row - 1, -beg_row - 1

            arr = ncf_hdl[var][:, beg_row:end_row, beg_col:end_col].data

            if flp_flg:
                for i in range(arr.shape[0]):
                    arr[i,:,:] = arr[i,::-1,:]

        return (arr,)

    def _vrf_ncf(self, pth, vrs, tlb, xcs, ycs):

        with nc.Dataset(pth) as ncf_hdl:

            assert tlb in ncf_hdl.variables, (tlb, ncf_hdl.variables)

            for var in vrs:

                assert isinstance(var, str), type(var)

                assert var in ncf_hdl.variables, (var, ncf_hdl.variables)

                assert ncf_hdl[var].ndim == 3, ncf_hdl[var].ndim

                if xcs.ndim == 1:
                    assert (xcs.size - 1) == ncf_hdl[var].shape[2], (
                        xcs.shape, ncf_hdl[var].shape)

                    assert (ycs.size - 1) == ncf_hdl[var].shape[1], (
                        ycs.shape, ncf_hdl[var].shape)

                elif xcs.ndim == 2:

                    assert (xcs.shape[0] - 1, xcs.shape[1] - 1) == (
                        ncf_hdl[var].shape[1:]), (
                            xcs.shape, ncf_hdl[var].shape)

                    assert (ycs.shape[0] - 1, ycs.shape[1] - 1) == (
                        ncf_hdl[var].shape[1:]), (
                            ycs.shape, ncf_hdl[var].shape)

                else:
                    raise NotImplementedError(xcs.ndim)

        return

    def _get_crd_obj_ncf(self, inp_pth):

        ras_crd_obj = ExtractNetCDFCoords(verbose=self._vb)

        ras_crd_obj.set_input(inp_pth, self._src_xlb, self._src_ylb)
        ras_crd_obj.extract_coordinates()

        return ras_crd_obj
