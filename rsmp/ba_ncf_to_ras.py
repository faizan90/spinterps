# -*- coding: utf-8 -*-

'''
Created on 09.07.2024

@author: Faizan-TU Munich
'''

from math import ceil
from pathlib import Path
from multiprocessing import Manager, Pool as MPPool

import numpy as np
import psutil as ps
import netCDF4 as nc
from pyproj.crs import CRS
from osgeo import ogr, gdal
from pyproj import Transformer
import matplotlib.pyplot as plt

from ..extract import ExtractNetCDFCoords

from ..mpg import DummyLock

from ..misc import print_sl, print_el, get_n_cpus

ogr.UseExceptions()
gdal.UseExceptions()

from .aa_ras_to_ras import ResampleRasToRas, RTRARGS


class ResampleNCFToRas(ResampleRasToRas):

    def __init__(self, verbose=True):

        ResampleRasToRas.__init__(self, verbose)

        self._src_vrs = None
        self._src_xlb = None
        self._src_ylb = None
        self._src_tlb = None
        self._src_crs = None

        # Numerical precision.
        self._nml_prn = None

        # Compression level.
        self._cpn_lvl = None

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
            src_crs,
            nml_prn,
            cpn_lvl,):

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
            n_cpus = get_n_cpus()

        else:
            assert isinstance(n_cpus, int), 'n_cpus is not an integer!'

            assert n_cpus > 0, 'Invalid n_cpus!'

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

        # Destination NCF numerical precision in terms of digits after decimal.
        assert isinstance(nml_prn, int), (
            f'nml_prn not an integer ({type(nml_prn)})!')

        assert nml_prn >= 0, 'nml_prn must be greater than or equal to zero!'

        self._nml_prn = nml_prn

        # Destination NCF compression level.
        assert isinstance(cpn_lvl, int), (
            f'cpn_lvl not and integer ({type(cpn_lvl)}!')

        assert 0 <= cpn_lvl <= 9, (
            f'cpn_lvl ({cpn_lvl}) not between 0 and 9!')

        self._cpn_lvl = cpn_lvl
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
            print(f'Desti. precision: {self._nml_prn}')
            print(f'Desti. compression level: {self._cpn_lvl}')

            print_el()

        self._inp_set_flg = True
        return

    def resample(self):

        assert self._inp_set_flg
        assert self._otp_set_flg

        # NOTE: Assuming coordinates are center when 1D and corner when 2D!
        src_crd_obj = self._get_crd_obj_ncf(self._src_pth)

        src_xcs = src_crd_obj.get_x_coordinates()
        src_ycs = src_crd_obj.get_y_coordinates()

        src_crd_obj = None

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
                (src_ycs + (0.5 * (src_ycs[0] - src_ycs[1]))),
                (src_ycs[-1] + (0.5 * (src_ycs[1] - src_ycs[0])),),
                ))

        elif src_xcs.ndim == 2:

            # raise NotImplementedError('Too much work!')

            assert src_xcs[0, 0] < src_xcs[0, 1], src_xcs

            if src_ycs[0, 0] < src_ycs[1, 0]:
                self._src_flp_flg = True

                src_ycs = src_ycs[::-1,:]
                src_xcs = src_xcs[::-1,:]

        else:
            raise NotImplementedError(src_xcs.ndim)

        self._vrf_ncf(
            self._src_pth, self._src_vrs, self._src_tlb, src_xcs, src_ycs)

        src_xcs, src_ycs = self._get_crd_msh(src_xcs, src_ycs)

        # GTiff are always corners.
        dst_crd_obj = self._get_crd_obj_ras(self._dst_pth)

        dst_xcs = dst_crd_obj.get_x_coordinates()
        dst_ycs = dst_crd_obj.get_y_coordinates()

        dst_crd_obj = None

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
            always_xy=True,
            only_best=True)

        dst_src_tfm = Transformer.from_crs(
            self._get_ras_crs(self._dst_pth),
            self._src_crs,
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

        if self._vb: print('Reading desti. array...')

        dst_arr = self._get_ras_vrs(
            self._dst_pth,
            dst_beg_row,
            dst_end_row,
            dst_beg_col,
            dst_end_col,
            1)[0]

        dst_arr_shp = dst_arr.shape

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
        #======================================================================

        if self._vb: print('Initializing netCDF file...')

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
        #======================================================================

        if self._vb: print('Reading source array(s)...')

        src_ays = {}
        for var in self._src_vrs:

            src_ays[var] = self._get_ncf_vrs(
                self._src_pth,
                src_beg_row,
                src_end_row - 1,
                src_beg_col,
                src_end_col - 1,
                var,
                self._src_flp_flg)[0]

        if self._vb: print('Initiating resampled array(s)...')

        src_dst_ays = {}
        for var in self._src_vrs:

            src_dst_ays[var] = np.full(
                (src_ays[var].shape[0],
                 dst_xcs.shape[0] - 1,
                 dst_xcs.shape[1] - 1),
                np.nan,
                dtype=np.float32)

            assert dst_arr_shp[1:] == src_dst_ays[var].shape[1:], (
                    var, dst_arr_shp, src_dst_ays[var].shape)

        if self._vb: print(
            f'Source transformed array(s) shape: {src_dst_ays[var].shape}...')
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
        ags.ncf_flg = True
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
            src_ays,
            dst_arr,
            src_xcs_ogl,
            src_ycs_ogl,
            src_dst_ays)

        ags = None
        src_ays = None
        src_xcs = src_xcs_ogl = None
        src_ycs = src_ycs_ogl = None
        #======================================================================

        if self._vb: print('Saving resampled raster...')

        for var in self._src_vrs:
            self._cvt_ncf_fle(var, src_dst_ays[var])

            src_dst_ays[var] = None
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

    def _cvt_ncf_fle(self, var, arr):

        ncf_hdl = nc.Dataset(self._out_pth, 'r+')

        ncf_hdl.set_auto_mask(False)

        ncf_var = ncf_hdl.createVariable(
            var,
            arr.dtype,
            dimensions=(self._ncf_dim_tlb,
                        self._ncf_dim_ylb_1dn,
                        self._ncf_dim_xlb_1dn,),
            fill_value=False,
            compression='zlib',
            complevel=self._cpn_lvl,
            shuffle=True,
            chunksizes=(1, arr.shape[1], arr.shape[2]))

        ncf_var[:] = arr

        ncf_hdl.close()
        return

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
                tst_xcs = ncf_hdl[xlb][beg_row:end_row + 1, beg_col:end_col].data
                tst_ycs = ncf_hdl[ylb][beg_row:end_row + 1, beg_col:end_col].data

                tst_xcs = tst_ycs[::-1,:]
                tst_ycs = tst_ycs[::-1,:]

            else:
                raise NotImplementedError(ncf_hdl[xlb].ndim)

        assert ref_xcs.shape == tst_xcs.shape, (ref_xcs.shape, tst_xcs.shape)
        assert ref_ycs.shape == tst_ycs.shape, (ref_ycs.shape, tst_ycs.shape)

        self._tfm_msh(tst_xcs, tst_ycs, crs_tfm)

        if False:
            # These can fail if there is significant rotation of the grid.
            # More likely to happen when source grid is fine.
            assert np.isclose(ref_xcs.min(), tst_xcs.min()), (
                ref_xcs.min(), tst_xcs.min())

            assert np.isclose(ref_xcs.max(), tst_xcs.max()), (
                ref_xcs.max(), tst_xcs.max())

            assert np.isclose(ref_ycs.min(), tst_ycs.min()), (
                ref_ycs.min(), tst_ycs.min())

            assert np.isclose(ref_ycs.max(), tst_ycs.max()), (
                ref_ycs.max(), tst_ycs.max())

        if False:  # Diagnostics.

            ref_xmn, ref_xmx = ref_xcs.min(), ref_xcs.max()
            ref_ymn, ref_ymx = ref_ycs.min(), ref_ycs.max()

            tst_xmn, tst_xmx = tst_xcs.min(), tst_xcs.max()
            tst_ymn, tst_ymx = tst_ycs.min(), tst_ycs.max()

            plt.gca().set_aspect('equal')

            xcs_ply = [ref_xmn, ref_xmn, ref_xmx, ref_xmx, ref_xmn]
            ycs_ply = [ref_ymn, ref_ymx, ref_ymx, ref_ymn, ref_ymn]

            plt.plot(
                xcs_ply, ycs_ply, c='cyan', alpha=0.85, zorder=10, label='REF')

            xcs_ply = [tst_xmn, tst_xmn, tst_xmx, tst_xmx, tst_xmn]
            ycs_ply = [tst_ymn, tst_ymx, tst_ymx, tst_ymn, tst_ymn]

            plt.plot(
                xcs_ply, ycs_ply, c='r', alpha=0.85, zorder=10, label='TST')

            plt.title(f'ref_ply and tst_ply')

            plt.show()

            plt.close()

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

        inp_cnk_sze = 1024 ** 3

        with nc.Dataset(inp_pth) as ncf_hdl:

            # A quick fix for now. Suboptimal.
            if not flp_flg:
                if (end_row - beg_row - 1) == ncf_hdl[var].shape[1]:
                    end_row_fnl = end_row - 1

                else:
                    end_row_fnl = end_row

                if (end_col - beg_col - 1) == ncf_hdl[var].shape[2]:
                    end_col_fnl = end_col - 1

                else:
                    end_col_fnl = end_col

                ary = np.empty(
                    (ncf_hdl[var].shape[0],
                     end_row_fnl - beg_row,
                     end_col_fnl - beg_col),
                    dtype=np.float32)

            else:
                ary = np.empty(
                    (ncf_hdl[var].shape[0],
                     end_row - beg_row,
                     end_col - beg_col),
                    dtype=np.float32)

            assert ary.shape[1] <= ncf_hdl[var].shape[1], (
                ary.shape[1], ncf_hdl[var].shape[1])

            assert ary.shape[2] <= ncf_hdl[var].shape[2], (
                ary.shape[2], ncf_hdl[var].shape[2])

            # Memory management.
            inp_vle_cts = np.prod(ncf_hdl[var].shape, dtype=np.uint64)

            tot_mmy = min(
                inp_cnk_sze, int(ps.virtual_memory().available * 0.5))

            mmy_cns_cnt = ceil(
                (inp_vle_cts / tot_mmy) * ncf_hdl[var].dtype.itemsize)

            assert mmy_cns_cnt >= 1, mmy_cns_cnt

            if mmy_cns_cnt > 1:
                if self._vb: print('Memory not enough to read in one go!')

                mmy_ixs = np.linspace(
                    0,
                    ncf_hdl[var].shape[0],
                    (mmy_cns_cnt + 1),
                    endpoint=True,
                    dtype=np.int64)

                assert mmy_ixs[+0] == 0, mmy_ixs
                assert mmy_ixs[-1] == ncf_hdl[var].shape[0], mmy_ixs

                assert np.unique(mmy_ixs).size == mmy_ixs.size, (
                    np.unique(mmy_ixs).size, mmy_ixs.size)

            else:
                mmy_ixs = np.array([0, ncf_hdl[var].shape[0]])

            if flp_flg:
                beg_row, end_row = -end_row - 1, -beg_row - 1

            for mmy_idx_bgn, mmy_idx_end in zip(mmy_ixs[:-1], mmy_ixs[1:]):

                ary[mmy_idx_bgn:mmy_idx_end,:,:] = ncf_hdl[var][
                    mmy_idx_bgn:mmy_idx_end,
                    beg_row:end_row,
                    beg_col:end_col].astype(np.float32)

            if flp_flg:
                for i in range(ary.shape[0]): ary[i,:,:] = ary[i,::-1,:]

        return (ary,)

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

                    assert (xcs.shape[0], xcs.shape[1]) == (
                        ncf_hdl[var].shape[1] + 1,
                        ncf_hdl[var].shape[2] + 1), (
                            xcs.shape, ncf_hdl[var].shape)

                    assert (ycs.shape[0], ycs.shape[1]) == (
                        ncf_hdl[var].shape[1] + 1,
                        ncf_hdl[var].shape[2] + 1), (
                            ycs.shape, ncf_hdl[var].shape)

                else:
                    raise NotImplementedError(xcs.ndim)

        return

    def _get_crd_obj_ncf(self, inp_pth):

        ras_crd_obj = ExtractNetCDFCoords(verbose=False)  # self._vb

        ras_crd_obj.set_input(inp_pth, self._src_xlb, self._src_ylb)
        ras_crd_obj.extract_coordinates()

        return ras_crd_obj
