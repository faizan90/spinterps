'''
Created on Nov 25, 2018

@author: Faizan
'''
from math import ceil, floor
from timeit import default_timer

import numpy as np
import pandas as pd
import netCDF4 as nc
from osgeo import ogr
import shapefile as shp

from .drift import KrigingDrift as KDT
from .vgclus import VariogramCluster as VC
from .bdpolys import SpInterpBoundaryPolygons as SIBD
from ..misc import (
    get_aligned_shp_bds_and_cell_size,
    get_ras_props,
    print_sl,
    print_el,
    chk_pt_cntmnt_in_polys_mp)


class SpInterpPrepare(SIBD, KDT):

    def __init__(self):

        SIBD.__init__(self)
        KDT.__init__(self)

        self._plot_polys = None
        self._cntn_idxs = None

        # In case poly_shp is not set but if drift raster is set, then the
        # bounds should stay within that of the drift raster.
        self._bds_buff_ncells = 0

        self._prpd_flag = False
        return

    def _cmpt_aligned_coordinates(self):

        '''
        Given the alignment raster compute the cell size and bounds of the
        interpolation grid such that they align completely with the
        alignment raster.
        '''

        assert self._algn_ras_set_flag, 'Call set_alignment_raster first!'

        beg_time = default_timer()

        ((fin_x_min,
          fin_x_max,
          fin_y_min,
          fin_y_max),
         cell_size) = get_aligned_shp_bds_and_cell_size(
             str(self._poly_shp), self._algn_ras, self._cell_bdist)

        fin_x_min -= self._bds_buff_ncells * cell_size
        fin_x_max += self._bds_buff_ncells * cell_size
        fin_y_min -= self._bds_buff_ncells * cell_size
        fin_y_max += self._bds_buff_ncells * cell_size

        self._x_min = fin_x_min
        self._x_max = fin_x_max
        self._y_min = fin_y_min
        self._y_max = fin_y_max
        self._cell_size = cell_size

        if self._vb:
            print_sl()

            print('Computed the following aligned coordinates:')
            print('Minimum X:', self._x_min)
            print('Maximum X:', self._x_max)
            print('Minimum Y:', self._y_min)
            print('Maximum Y:', self._y_max)
            print('Cell size:', self._cell_size)

            end_time = default_timer()
            print(f'Took {end_time - beg_time:0.2f} seconds.')

            print_el()

        return

    def _cmpt_corner_coordinates(self):

        '''
        If alignment raster is unspecified then take the minima and maxima
        of the station coordinates as the bounds for the interpolation
        grids.

        If external drift kriging is turned on then use the cell size of
        the first drift raster.

        Else the  cell size should have been specified manually in
        the set_misc_settings function.

        Error is raised if cell size is not set.
        '''

        if self._cell_sel_prms_set and (not self._algn_ras_set_flag):
            in_ds = ogr.Open(str(self._poly_shp))
            assert in_ds, f'Could not open {self._poly_shp}!'

            lyr_count = in_ds.GetLayerCount()

            assert lyr_count, f'No layers in {self._poly_shp}!'
            assert lyr_count == 1, f'More than one layer in {self._poly_shp}!'

            in_lyr = in_ds.GetLayer(0)
            envelope = in_lyr.GetExtent()

            assert envelope, f'No envelope for {self._poly_shp}!'
            in_ds.Destroy()

            self._x_min, self._x_max, self._y_min, self._y_max = envelope

        else:
            self._x_min = self._crds_df['X'].min()
            self._x_max = self._crds_df['X'].max()

            self._y_min = self._crds_df['Y'].min()
            self._y_max = self._crds_df['Y'].max()

        self._x_min -= self._cell_bdist
        self._x_max += self._cell_bdist

        self._y_min -= self._cell_bdist
        self._y_max += self._cell_bdist

        if self._vb:
            print_sl()
            print('Computed the following corner coordinates:')
            print('Minimum X:', self._x_min)
            print('Maximum X:', self._x_max)
            print('Minimum Y:', self._y_min)
            print('Maximum X:', self._x_max)
            print_el()
        return

    def _prepare_crds(self):

        if self._edk_flag:
            assert self._x_min >= self._drft_x_min, (
                'Grid x_min outside of the drift rasters!')

            assert self._x_max <= self._drft_x_max, (
                'Grid x_max outside of drift rtasters!')

            assert self._y_min >= self._drft_y_min, (
                'Grid y_min outside of drift rasters!')

            assert self._y_max <= self._drft_y_max, (
                'Grid y_max outside of drift rasters!')

            self._min_col = int(
                floor((self._x_min - self._drft_x_min) / self._cell_size))

            self._max_col = int(
                ceil((self._x_max - self._drft_x_min) / self._cell_size)) - 1

            self._min_row = int(
                floor((self._drft_y_max - self._y_max) / self._cell_size))

            self._max_row = int(
                ceil((self._drft_y_max - self._y_min) / self._cell_size)) - 1

        else:
            self._min_col = 0

            self._max_col = int(
                ceil((self._x_max - self._x_min) / self._cell_size)) - 1

            self._min_row = 0

            self._max_row = int(
                ceil((self._y_max - self._y_min) / self._cell_size)) - 1

        assert 0 <= self._min_col <= self._max_col, (
            self._min_col, self._max_col)

        assert 0 <= self._min_row <= self._max_row, (
            self._min_row, self._max_row)
        #======================================================================

        strt_x_coord = self._x_min + (0.5 * self._cell_size)

        end_x_coord = strt_x_coord + (
            (self._max_col - self._min_col) * self._cell_size)

        strt_y_coord = self._y_max - (0.5 * self._cell_size)

        end_y_coord = strt_y_coord - (
            (self._max_row - self._min_row) * self._cell_size)

        interp_x_coords = np.linspace(
            strt_x_coord, end_x_coord, (self._max_col - self._min_col + 1))

        interp_y_coords = np.linspace(
            strt_y_coord, end_y_coord, (self._max_row - self._min_row + 1))

        interp_x_coords_mesh, interp_y_coords_mesh = np.meshgrid(
            interp_x_coords, interp_y_coords)

        # Must not move.
        self._interp_crds_orig_shape = interp_x_coords_mesh.shape

        self._interp_x_crds_plt_msh, self._interp_y_crds_plt_msh = None, None

        if self._plot_figs_flag:
            interp_x_coords_plt = np.linspace(
                self._x_min, end_x_coord + (0.5 * self._cell_size),
                (self._max_col - self._min_col + 2))

            interp_y_coords_plt = np.linspace(
                self._y_max, end_y_coord - (0.5 * self._cell_size),
                (self._max_row - self._min_row + 2))

            self._interp_x_crds_plt_msh, self._interp_y_crds_plt_msh = (
                np.meshgrid(interp_x_coords_plt, interp_y_coords_plt))
        #======================================================================

        self._nc_x_crds = interp_x_coords
        self._nc_y_crds = interp_y_coords

        self._interp_x_crds_msh = interp_x_coords_mesh.ravel()
        self._interp_y_crds_msh = interp_y_coords_mesh.ravel()

        assert (
            self._interp_x_crds_msh.ndim ==
            self._interp_y_crds_msh.ndim ==
            1)

        assert (self._interp_x_crds_msh.size == self._interp_y_crds_msh.size)
        return

    def _select_nearby_cells(self):

        '''
        If interp_around_polys_flag is True then interpolate only those
        cells that are near or inside the polygons.
        '''

        beg_time = default_timer()

        if self._vb:
            print_sl()
            print(self._interp_x_crds_msh.shape[0],
                  'cells to interpolate per step before intersection!')

        crds_df = pd.DataFrame(
            data={'X': self._interp_x_crds_msh, 'Y': self._interp_y_crds_msh},
            dtype=float)

        fin_cntn_idxs_set = chk_pt_cntmnt_in_polys_mp(
            self._geom_buff_cells, crds_df, self._n_cpus)

        fin_cntn_idxs = np.zeros(self._interp_x_crds_msh.shape[0], dtype=bool)
        for idx in fin_cntn_idxs_set:
            fin_cntn_idxs[idx] = True

        n_fin_cntn_idxs = fin_cntn_idxs.sum()

        if self._vb:
            print(
                n_fin_cntn_idxs,
                'cells to interpolate per step after intersection!')

            end_time = default_timer()
            print(f'Took {end_time - beg_time:0.2f} seconds.')

            print_el()

        assert n_fin_cntn_idxs, 'No cells selected for interpolation!'

        self._interp_x_crds_msh = self._interp_x_crds_msh[fin_cntn_idxs]
        self._interp_y_crds_msh = self._interp_y_crds_msh[fin_cntn_idxs]

        assert fin_cntn_idxs.dtype == np.bool, 'Expected a boolean array!'
        self._cntn_idxs = fin_cntn_idxs
        return

    def _initiate_nc(self):

        '''
        Create the output netCDF4 file. All interpolated grids, cell
        coordinates and time stamps will be saved in this file.
        '''

        nc_hdl = nc.Dataset(
            str(self._out_dir / (self._nc_out.split('.', 1)[0] + '.nc')),
            mode='w')

        nc_hdl.set_auto_mask(False)
        nc_hdl.createDimension(self._nc_xlab, self._nc_x_crds.shape[0])
        nc_hdl.createDimension(self._nc_ylab, self._nc_y_crds.shape[0])
        nc_hdl.createDimension(self._nc_tlab, self._time_rng.shape[0])

        x_coords_nc = nc_hdl.createVariable(
            self._nc_xlab, 'd', dimensions=self._nc_xlab)

        x_coords_nc[:] = self._nc_x_crds

        y_coords_nc = nc_hdl.createVariable(
            self._nc_ylab, 'd', dimensions=self._nc_ylab)

        y_coords_nc[:] = self._nc_y_crds

        time_nc = nc_hdl.createVariable(
            self._nc_tlab, 'i8', dimensions=self._nc_tlab)

        if self._index_type == 'date':
            time_nc[:] = nc.date2num(
                self._time_rng.to_pydatetime(),
                units=self._nc_tunits,
                calendar=self._nc_tcldr)

            time_nc.units = self._nc_tunits
            time_nc.calendar = self._nc_tcldr

        elif self._index_type == 'obj':
            time_nc[:] = np.arange(self._time_rng.shape[0], dtype=int)

        else:
            raise NotImplementedError(
                f'Unknown index_type: {self._index_type}!')

        for interp_arg in self._interp_args:
            ivar_name = interp_arg[2]

            nc_var = nc_hdl.createVariable(
                ivar_name,
                'd',
                dimensions=(self._nc_tlab, self._nc_ylab, self._nc_xlab),
                fill_value=False)

            nc_var.units = self._nc_vunits

            if interp_arg[0] == 'IDW':
                nc_var.standard_name = self._nc_vlab + (
                    f' ({ivar_name[:3]}_exp_{interp_arg[2]})')

            else:
                nc_var.standard_name = self._nc_vlab + f' ({ivar_name})'

        self._nc_file_path = nc_hdl.filepath()

        nc_hdl.Source = self._nc_file_path
        nc_hdl.close()
        return

    def _vrf_nebs(self):

        assert self._neb_sel_mthd in self._neb_sel_mthds

        n_avail_nebs = self._crds_df.shape[0]

        assert n_avail_nebs > 0

        if self._neb_sel_mthd == 'all':
            pass

        elif (self._neb_sel_mthd == 'nrst') or (self._neb_sel_mthd == 'pie'):
            if n_avail_nebs < self._n_nebs:

                print_sl()
                print(
                    f'WARNING: Setting the neighbor selection method to '
                    f'\'all\' because n_neighbors ({self._n_nebs}) '
                    f'is greater than the number of available stations '
                    f'({n_avail_nebs})!')
                print_el()

                self._neb_sel_mthd = 'all'
                self._n_nebs = None
                self._n_pies = None

        else:
            raise NotImplementedError

        return

    def _cluster_vgs(self):

        '''
        Should be called when final data_df and vg_ser are ready.
        This reorders the time series such that clustered variograms
        follow each other in time. This results in an efficient memory use,
        by having fewer variograms to compute for per thread.
        '''

        assert self._vgs_ser is not None

        vc_cls = VC(self._vgs_ser)

        clus_dict, self._vgs_ser = vc_cls.get_vgs_cluster()

        # Holds the indices that tell where to right to in the netcdf file.
        vgs_rord_tidxs_ser = pd.Series(
            index=self._vgs_ser.index, data=np.arange(self._vgs_ser.shape[0]))

        # Used to get the unique number of vgs per thread to compute memory
        # requirements.
        vgs_unq_ids = pd.Series(index=self._vgs_ser.index, dtype=np.int64)

        unq_id = 0
        vg_clus_tidxs = []
        for vg in clus_dict.keys():
            clus_idxs = clus_dict[vg]

            vg_clus_tidxs.extend(clus_idxs.tolist())

            vgs_unq_ids.loc[clus_idxs] = unq_id

            unq_id += 1

        assert len(vg_clus_tidxs) == self._vgs_ser.shape[0]

        vg_clus_tidxs = np.array(vg_clus_tidxs)

        self._data_df = self._data_df.reindex(index=vg_clus_tidxs)
        self._vgs_ser = self._vgs_ser.reindex(index=vg_clus_tidxs)
        self._vgs_unq_ids = vgs_unq_ids.reindex(index=vg_clus_tidxs)
        self._vgs_rord_tidxs_ser = vgs_rord_tidxs_ser.reindex(
            index=vg_clus_tidxs)
        return

    def _prepare(self):

        '''Main call for the preparation of required variables.'''

        assert any([
            self._ork_flag,
            self._spk_flag,
            self._edk_flag,
            self._idw_flag])

        if self._index_type == 'date':

            assert all([
                self._tbeg is not None,
                self._tend is not None,
                self._tfreq is not None]), (
                    'beg_time, end_time and time_freq are not set!')

            self._time_rng = pd.date_range(
                self._tbeg, self._tend, freq=self._tfreq)

        elif self._index_type == 'obj':
            self._time_rng = self._data_df.index

        else:
            raise NotImplementedError(
                f'Unknown index_type: {self._index_type}!')
        #======================================================================

        if self._edk_flag and (not self._algn_ras_set_flag):
            self._cell_size = get_ras_props(str(self._drft_rass[0]))[6]

        if self._algn_ras_set_flag:
            # This updates the cell size to that of the align raster.
            self._cmpt_aligned_coordinates()

        else:
            self._cmpt_corner_coordinates()

        print_sl()
        print('Final grid cell size:', self._cell_size)
        print_el()

        assert self._cell_size is not None, 'Cell size unspecified!'
        #======================================================================

        if self._cell_sel_prms_set:
            # cell_size should be set by now.
            self._select_nearest_stations()
        #======================================================================

        if self._cell_sel_prms_set and self._plot_figs_flag:
            sf = shp.Reader(str(self._poly_shp))

            self._plot_polys = [
                i.__geo_interface__ for i in sf.iterShapes()]
        #======================================================================

        if self._edk_flag:
            self._assemble_drift_data()
        #======================================================================

        self._prepare_crds()
        #======================================================================

        if self._cell_sel_prms_set and self._ipoly_flag:
            self._select_nearby_cells()
        #======================================================================

        self._vrf_nebs()
        #======================================================================

        if self._edk_flag:
            self._prepare_stns_drift()
        #======================================================================

        self._out_dir.mkdir(exist_ok=True)

        if self._plot_figs_flag:
            self._plots_dir = self._out_dir / 'interp_plots'
            self._plots_dir.mkdir(exist_ok=True)

            fig_dirs = {}

            if self._ork_flag:
                fig_dirs['OK'] = 'ord_krig_figs'

            if self._spk_flag:
                fig_dirs['SK'] = 'smp_krig_figs'

            if self._edk_flag:
                fig_dirs['EDK'] = 'ext_krig_figs'

            if self._idw_flag:
                for i, idw_exp in enumerate(self._idw_exps):
                    exp_str = ('%0.3f' % idw_exp).replace('.', '').rstrip('0')
                    fig_dirs[f'IDW_{i:03d}'] = 'idw_exp_%s_figs' % exp_str

            interp_plot_dirs = {}

            for fig_dir_lab in fig_dirs:
                fig_dir = self._plots_dir / fig_dirs[fig_dir_lab]

                fig_dir.mkdir(exist_ok=True)

                interp_plot_dirs[fig_dir_lab] = fig_dir
        #======================================================================

        self._interp_args = []
        if self._ork_flag:
            if self._plot_figs_flag:
                fig_dir = interp_plot_dirs['OK']

            else:
                fig_dir = None

            self._interp_args.append(('OK', fig_dir, 'OK'))

        if self._spk_flag:
            if self._plot_figs_flag:
                fig_dir = interp_plot_dirs['SK']

            else:
                fig_dir = None

            self._interp_args.append(('SK', fig_dir, 'SK'))

        if self._edk_flag:
            if self._plot_figs_flag:
                fig_dir = interp_plot_dirs['EDK']

            else:
                fig_dir = None

            self._interp_args.append(('EDK', fig_dir, 'EDK'))

        if self._idw_flag:
            for i, idw_exp in enumerate(self._idw_exps):
                idw_lab = f'IDW_{i:03d}'

                if self._plot_figs_flag:
                    fig_dir = interp_plot_dirs[idw_lab]

                else:
                    fig_dir = None

                self._interp_args.append(('IDW', fig_dir, idw_lab, idw_exp))
        #======================================================================

        self._initiate_nc()
        #======================================================================

        all_stns = self._data_df.columns.intersection(self._crds_df.index)

        assert all_stns.shape[0] > 1, (
            'Less than 2 common stations in data and station coordinates\' '
            'dataframes!')

        if self._edk_flag:
            all_stns = all_stns.intersection(self._stns_drft_df.index)

            assert all_stns.shape[0] > 1, (
                'Less than 2 common stations in data, station coordinates\' '
                'and station drifts\' dataframes!')

            self._stns_drft_df = self._stns_drft_df.loc[all_stns]
        #======================================================================

        self._data_df = self._data_df.loc[:, all_stns]
        self._crds_df = self._crds_df.loc[all_stns]

        self._data_df = self._data_df.reindex(self._time_rng)

        if self._vg_ser_set_flag:
            self._vgs_ser = self._vgs_ser.reindex(self._time_rng).astype(str)

            self._cluster_vgs()

        self._prpd_flag = True
        return

