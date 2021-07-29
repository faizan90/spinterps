'''
Created on Nov 25, 2018

@author: Faizan
'''
from timeit import default_timer

from ..misc import (
    print_sl, print_el, chk_pt_cntmnt_in_polys_mp, get_all_polys_in_shp)


class SpInterpBoundaryPolygons:

    def __init__(self):

        self._geom_buff_cells = None

        self._nrst_stns_slctd_flag = False
        return

    def _select_nearest_stations(self):

        '''
        Given the polygons_shapefile, select stations that have
        distances less than the station_select_buffer_distance to the
        nearest polygon.
        '''

        beg_time = default_timer()

        if self._vb:
            print_sl()
            print(
                'Selecting stations and cells within and around the '
                'shapefile polygons...')

        assert self._cell_sel_prms_set, (
            'Call set_cell_selection_parameters first!')
        #======================================================================

        geom_simplify_tol = self._poly_simplify_tol_ratio * self._cell_size

        assert geom_simplify_tol >= 0, geom_simplify_tol

        all_geoms = get_all_polys_in_shp(self._poly_shp, geom_simplify_tol)

        assert all_geoms.qsize(), 'Zero usable polygons in the shapefile!'
        #======================================================================

        stn_slct_polys = []
        for geom in all_geoms.queue:
            assert geom is not None, (
                'Something wrong with the geometries in the polygons in the '
                'bounds shapefile!')

            assert geom.GetGeometryCount() == 1, 'Geometry count not one!'

            geom_type = geom.GetGeometryType()
            geom_name = geom.GetGeometryName()

            assert geom_type == 3, (
                f'Unknown geometry type, name: {geom_type}, {geom_name}!')

            assert geom.Area() > 0, 'Geometry has no area!'

            stn_slct_polys.append(geom.Clone())

        # Do before modifying original polygons.
        if self._ipoly_flag:
            cell_slct_polys = [geom.Clone() for geom in stn_slct_polys]
        #======================================================================

        assert stn_slct_polys, (
            'Zero polygons selected in the polygons_shapefile!')

        if self._stn_bdist:
            stn_slct_polys = [
                geom.Buffer(self._stn_bdist) for geom in stn_slct_polys]

            if geom_simplify_tol:
                stn_slct_polys = [
                    geom.SimplifyPreserveTopology(geom_simplify_tol)
                    for geom in stn_slct_polys]

            assert all(
                [geom.GetGeometryCount() == 1 for geom in stn_slct_polys]), (
                'Geometry count changed after buffering!')
        #======================================================================

        if self._ipoly_flag:
            assert cell_slct_polys, 'Zero polygons in the polygons_shapefile!'

        if self._ipoly_flag and self._cell_bdist:
            cell_slct_polys = [
                geom.Buffer(self._cell_bdist) for geom in cell_slct_polys]

            if geom_simplify_tol:
                cell_slct_polys = [
                    geom.SimplifyPreserveTopology(geom_simplify_tol)
                    for geom in cell_slct_polys]

            assert all(
                [geom.GetGeometryCount() == 1 for geom in cell_slct_polys]), (
                'Geometry count changed after buffering!')
        #======================================================================

        if self._vb:
            print(len(stn_slct_polys), 'polygons in the polygons_shapefile.')

            print(
                self._data_df.shape[1],
                'stations in the coordinates dataframe.')

        fin_stns = chk_pt_cntmnt_in_polys_mp(
            stn_slct_polys, self._crds_df, self._n_cpus)

        assert fin_stns, (
            'Found zero stations that are close enough to the polygons!')
        #======================================================================

        if self._vb:
            print(
                f'{len(fin_stns)} stations out of {self._crds_df.shape[0]} '
                f'within buffer zone of polygons_shapefile.')

            end_time = default_timer()
            print(f'Took {end_time - beg_time:0.2f} seconds.')

            print_el()
        #======================================================================

        assert len(fin_stns) <= self._data_df.shape[1]
        assert len(fin_stns) <= self._crds_df.shape[0]

        self._data_df = self._data_df.loc[:, fin_stns]
        self._crds_df = self._crds_df.loc[fin_stns,:]

        if self._ipoly_flag:
            self._geom_buff_cells = cell_slct_polys

        self._nrst_stns_slctd_flag = True
        return
