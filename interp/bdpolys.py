'''
Created on Nov 25, 2018

@author: Faizan
'''

import ogr
import numpy as np

from ..misc import cnvt_to_pt, chk_cntmt


class SpInterpBoundaryPolygons:

    def __init__(self):

        self._nrst_stns_slctd_flag=False
        return

    def _select_nearest_stations(self):

        '''Given the polygons_shapefile, select stations that have
        distances less than the station_select_buffer_distance to the
        nearest polygon.
        '''

        assert self._cell_sel_prms_set, (
            'Call set_cell_selection_parameters first!')

        bds_vec=ogr.Open(str(self._poly_shp))
        assert bds_vec is not None, (
            'Could not open the polygons_shapefile!')

        bds_lyr=bds_vec.GetLayer(0)

        all_stns=self._data_df.columns

        feat_buff_stns=[]

        if self._ipoly_flag:
            feat_buff_cells=[]

        for feat in bds_lyr:
            geom=feat.GetGeometryRef().Clone()
            assert geom is not None, (
                'Something wrong with the geometries in the '
                'polygons_shapefile!')

            assert geom.GetGeometryType()==3, 'Geometry not a polygon!'

            assert geom.Area()>0, 'Geometry has no area!'

            feat_buff_stns.append(geom.Buffer(self._stn_bdist))

            if self._ipoly_flag:
                feat_buff_cells.append(geom.Buffer(self._cell_bdist))

        bds_vec.Destroy()

        assert feat_buff_stns, 'Zero polygons in the polygons_shapefile!'

        if self._ipoly_flag:
            assert feat_buff_cells, (
                'Zero polygons in the polygons_shapefile!')

        if self._vb:
            print('\n', '#'*10, sep='')
            print(len(feat_buff_stns), 'polygons in the polygons_shapefile.')

        fin_stns=[]
        for poly in feat_buff_stns:
            assert poly is not None, 'Corrupt polygons after buffering!'

            for stn in all_stns:
                if stn in fin_stns:
                    continue

                curr_pt=cnvt_to_pt(
                    *self._crds_df.loc[stn, ['X', 'Y']].values)

                if chk_cntmt(curr_pt, poly):
                    fin_stns.append(stn)

        assert fin_stns, (
            'Found no stations that are close enough to the polygons!')

        if self._vb:
            print(
                f'{len(fin_stns)} stations out of {self._crds_df.shape[0]} '
                f'within buffer zone of polygons_shapefile.')
            print('#'*10)

        fin_stns=np.unique(fin_stns)

        self._data_df=self._data_df.loc[:, fin_stns]
        self._crds_df=self._crds_df.loc[fin_stns, :]

        if self._ipoly_flag:
            self._geom_buff_cells=feat_buff_cells

        self._nrst_stns_slctd_flag=True
        return
