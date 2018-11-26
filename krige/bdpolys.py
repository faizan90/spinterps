'''
Created on Nov 25, 2018

@author: Faizan
'''

import ogr
import numpy as np

from ..misc import cnvt_to_pt, chk_cntmt


class KrigingBoundaryPolygons:

    def __init__(self):

        self._nrst_stns_slctd_flag = False
        return

    def _select_nearest_stations(self):

        assert self._cell_sel_prms_set

        bds_vec = ogr.Open(str(self._poly_shp))
        assert bds_vec is not None

        bds_lyr = bds_vec.GetLayer(0)

        all_stns = self._data_df.columns

        feat_buff_stns = []

        if self._ipoly_flag:
            feat_buff_cells = []

        for feat in bds_lyr:  # just to get the names of the catchments
            geom = feat.GetGeometryRef().Clone()
            assert geom is not None

            assert isinstance(geom, ogr.Geometry)

            feat_buff_stns.append(geom.Buffer(self._stn_bdist))

            if self._ipoly_flag:
                feat_buff_cells.append(geom.Buffer(self._cell_bdist))

        bds_vec.Destroy()

        assert feat_buff_stns

        if self._ipoly_flag:
            assert feat_buff_cells

        if self._vb:
            print('\n', '#' * 10, sep='')
            print(len(feat_buff_stns), 'polygons in the polygons_shapefile.')

        fin_stns = []
        for poly in feat_buff_stns:
            for stn in all_stns:
                if stn in fin_stns:
                    continue

                curr_pt = cnvt_to_pt(
                    *self._crds_df.loc[stn, ['X', 'Y']].values)

                if chk_cntmt(curr_pt, poly):
                    fin_stns.append(stn)

        assert fin_stns

        if self._vb:
            print(
                f'{len(fin_stns)} stations out of {self._crds_df.shape[0]} '
                f'within buffer zone of polygons_shapefile.')
            print('#' * 10)

        fin_stns = np.unique(fin_stns)

        self._data_df = self._data_df.loc[:, fin_stns]
        self._crds_df = self._crds_df.loc[fin_stns, :]

        if self._ipoly_flag:
            self._geom_buff_cells = feat_buff_cells

        self._nrst_stns_slctd_flag = True
        return
