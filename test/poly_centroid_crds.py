# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Jan 11, 2024

10:49:55 AM

Keywords:

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import pandas as pd
from osgeo import ogr

DEBUG_FLAG = False


def main():

    main_dir = Path(r'D:\hydmod\neckar')
    os.chdir(main_dir)

    path_to_shp = Path(
        r'P:\Synchronize\TUM\lehre\WS\2023_24\HERBM\Exercises\final_assignment\solution\enz\Watershed\Shapes\dem__utm32n_3421_hydrsubbasins.shp')

    label_field = 'PolygonId'  # r'DN'

    label_pref = 'PET'

    ot_df_path = Path(r'D:\hydmod\neckar\daily_1961_2020_spinterp_2km_pet\3421_basin_centroids.csv')
    #==========================================================================

    cat_polys = get_shp_stuff(str(path_to_shp.absolute()), label_field)[0]

    crds_dict = {}
    for cat_id, cat_poly in cat_polys.items():
        centroid = cat_poly.Centroid()

        centroid_x, centroid_y = centroid.GetX(), centroid.GetY()

        crds_dict[f'{label_pref}{cat_id}'] = centroid_x, centroid_y

    crds_df = pd.DataFrame(data=crds_dict, index=['X', 'Y'])

    crds_df = crds_df.T

    crds_df.to_csv(ot_df_path, sep=';')
    return


def get_shp_stuff(in_shp, field_id):

    '''
    Get every geometry and the layer extents of a given shapefile

    Only one layer allowed

    Geometries are inside a dictionary whose keys are the values of field_id
    '''

    assert os.path.exists(in_shp), 'in_shp (%s) does not exist!' % in_shp

    in_ds = ogr.Open(in_shp)

    lyr_count = in_ds.GetLayerCount()

    assert lyr_count, 'No layers in %s!' % in_shp
    assert lyr_count == 1, 'More than one layer in %s' % in_shp

    geoms_dict = {}

    shp_spt_ref = None

    in_lyr = in_ds.GetLayer(0)

    envelope = in_lyr.GetExtent()
    assert envelope, 'No envelope!'

    feat_count = in_lyr.GetFeatureCount()
    assert feat_count, 'No features in %s!' % in_shp

    if shp_spt_ref is None:
        shp_spt_ref = in_lyr.GetSpatialRef()

    assert shp_spt_ref, '%s has no spatial reference!' % in_shp

    for j in range(feat_count):
        curr_feat = in_lyr.GetFeature(j)

        if curr_feat is not None:
            cat_no = curr_feat.GetFieldAsString(str(field_id))

        else:
            continue

        geom = curr_feat.GetGeometryRef().Clone()
        assert geom is not None

        geoms_dict[cat_no] = geom

    in_ds.Destroy()

    assert geoms_dict, 'No geometries found!'

    return geoms_dict, envelope


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
