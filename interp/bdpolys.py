'''
Created on Nov 25, 2018

@author: Faizan
'''

import numpy as np
from osgeo import ogr

from ..misc import cnvt_to_pt, chk_cntmt, print_sl, print_el


class SpInterpBoundaryPolygons:

    def __init__(self):

        self._geom_buff_cells = None

        self._nrst_stns_slctd_flag = False
        return

    def _select_nearest_stations(self):

        '''Given the polygons_shapefile, select stations that have
        distances less than the station_select_buffer_distance to the
        nearest polygon.
        '''

        if self._vb:
            print_sl()
            print(
                'Selecting stations within and around the shapefile '
                'polygons...')

        assert self._cell_sel_prms_set, (
            'Call set_cell_selection_parameters first!')

        bds_vec = ogr.Open(str(self._poly_shp))

        assert bds_vec is not None, (
            'Could not open the polygons_shapefile!')

        bds_lyr = bds_vec.GetLayer(0)

        all_stns = self._data_df.columns

        feat_buff_stns = []

        if self._ipoly_flag:
            feat_buff_cells = []

        temp_geoms = []
        for feat in bds_lyr:
            geom = feat.GetGeometryRef().Clone()

            assert geom is not None, (
                'Something wrong with the geometries in the '
                'polygons_shapefile!')

            geom_type = geom.GetGeometryType()

            if geom_type == 6:
                for sub_geom in geom:
                    temp_geoms.append(sub_geom.Clone())

            if geom_type == 3:
                temp_geoms.append(geom)

        assert temp_geoms, 'Zero usable polygons in the shapefile!'

        for geom in temp_geoms:
            assert geom is not None, (
                'Something wrong with the geometries in the '
                'polygons_shapefile!')

            geom_type = geom.GetGeometryType()
            geom_name = geom.GetGeometryName()

            assert geom_type == 3, (
                f'Unknown geometry type, name: {geom_type}, {geom_name}!')

            assert geom.Area() > 0, 'Geometry has no area!'

            feat_buff_stns.append(geom.Buffer(self._stn_bdist))

            assert (feat_buff_stns[-1].GetGeometryCount() == 1), (
                'Geometry count changed after buffering! '
                'Changing station_select_buffer_distance might help.')

            if self._ipoly_flag:

                feat_buff_cells.append(geom.Buffer(self._cell_bdist))

                assert (feat_buff_cells[-1].GetGeometryCount() == 1), (
                    'Geometry count changed after buffering! '
                    'Changing polygon_cell_buffer_distance might help.')

        bds_vec.Destroy()

        assert feat_buff_stns, (
            'Zero polygons selected in the polygons_shapefile!')

        if self._ipoly_flag:
            assert feat_buff_cells, (
                'Zero polygons in the polygons_shapefile!')

        if self._vb:
            print(len(feat_buff_stns), 'polygons in the polygons_shapefile.')
            print(len(all_stns), 'stations in the coordinates dataframe.')

        fin_stns = []
        for poly in feat_buff_stns:
            assert poly is not None, 'Corrupted polygon after buffering!'

            for stn in all_stns:
                if stn in fin_stns:
                    continue

                curr_pt = cnvt_to_pt(
                        *self._crds_df.loc[stn, ['X', 'Y']].values)

                if chk_cntmt(curr_pt, poly):
                    fin_stns.append(stn)

        assert fin_stns, (
            'Found zero stations that are close enough to the polygons!')

        fin_stns = np.unique(fin_stns)

        if self._vb:
            print(
                f'{len(fin_stns)} stations out of {self._crds_df.shape[0]} '
                f'within buffer zone of polygons_shapefile.')
            print_el()

        self._data_df = self._data_df.loc[:, fin_stns]
        self._crds_df = self._crds_df.loc[fin_stns, :]

        if self._ipoly_flag:
            self._geom_buff_cells = feat_buff_cells

        self._nrst_stns_slctd_flag = True
        return
