'''
Created on May 27, 2019

@author: Faizan-Uni
'''

import ogr
import osr
import numpy as np
import netCDF4 as nc

ogr.UseExceptions()
osr.UseExceptions()


def get_polygon_nc_intersection_idxs(
        path_to_nc,
        path_to_poly_shp,
        poly_id,
        poly_id_field,
        nc_x_crds_lab,
        nc_y_crds_lab,
        min_area_thresh,
        max_cells_thresh):

    cat_vec = ogr.Open(path_to_poly_shp)
    lyr = cat_vec.GetLayer(0)

    feat_dict = {}
    feat_area_dict = {}
    cat_envel_dict = {}

    feat = lyr.GetNextFeature()

    while feat:
        geom = feat.GetGeometryRef()
        f_val = feat.GetFieldAsInteger(str(poly_id_field))
        if f_val is None:
            raise RuntimeError('Could not get f_val!')

        feat_area_dict[f_val] = geom.Area()  # do before transform

        feat_dict[f_val] = feat

        cat_envel_dict[f_val] = geom.GetEnvelope()  # do after transform
        feat = lyr.GetNextFeature()

    cat_vec.Destroy()

    in_nc = nc.Dataset(path_to_nc)
    lat_arr = in_nc.variables[nc_y_crds_lab][:]
    lon_arr = in_nc.variables[nc_x_crds_lab][:]
    in_nc.close()

    assert not np.all(np.isnan(lat_arr))
    assert not np.all(np.isnan(lon_arr))

    apply_cell_correc = True
    cell_size = abs(round(lon_arr[1] - lon_arr[0], 3))
    x_l_c = lon_arr[0]
    x_u_c = lon_arr[-1]
    y_l_c = lat_arr[0]
    y_u_c = lat_arr[-1]

    if x_l_c > x_u_c:
        x_l_c, x_u_c = x_u_c, x_l_c

    if y_l_c > y_u_c:
        y_l_c, y_u_c = y_u_c, y_l_c

    # raise Exception

    if apply_cell_correc:
        # because values are at the center of the cell
        # so I shift it back
        x_l_c -= (cell_size / 2.)
        x_u_c += (cell_size / 2.)
        y_l_c -= (cell_size / 2.)
        y_u_c += (cell_size / 2.)
    else:
        x_u_c += cell_size
        y_u_c += cell_size

    x_coords = np.arange(x_l_c, x_u_c * 1.00000001, cell_size)
    y_coords = np.arange(y_l_c, y_u_c * 1.00000001, cell_size)

#    print feat_dict.keys()

    geom = feat_dict[poly_id].GetGeometryRef()

    extents = cat_envel_dict[poly_id]
    cat_area = feat_area_dict[poly_id]

    inter_areas = []
    x_low, x_hi, y_low, y_hi = extents

    # adjustment to get all cells intersecting the polygon
    x_low = x_low - cell_size
    x_hi = x_hi + cell_size
    y_low = y_low - cell_size
    y_hi = y_hi + cell_size

    x_cors_idxs = np.where(np.logical_and(x_coords >= x_low,
                                          x_coords <= x_hi))[0]
    y_cors_idxs = np.where(np.logical_and(y_coords >= y_low,
                                          y_coords <= y_hi))[0]

    x_cors = x_coords[x_cors_idxs]
    y_cors = y_coords[y_cors_idxs]

    n_cells = x_cors.shape[0] * y_cors.shape[0]
    if n_cells > max_cells_thresh:
        use_cntnmt = True
        cell_area = cell_size ** 2
    else:
        use_cntnmt = False

    cat_x_idxs = []
    cat_y_idxs = []
    n_cells_cntned = 0

    for x_idx in range(x_cors.shape[0] - 1):
        for y_idx in range(y_cors.shape[0] - 1):
            if use_cntnmt:
                avg_x_cor = 0.5 * (x_cors[x_idx] + x_cors[x_idx + 1])
                avg_y_cor = 0.5 * (y_cors[y_idx] + y_cors[y_idx + 1])
                avg_cor = ogr.CreateGeometryFromWkt(("POINT (%f %f)" %
                                                     (avg_x_cor, avg_y_cor)))

                if geom.Contains(avg_cor):
                    inter_area = cell_area
                else:
                    inter_area = 0.0
            else:
                ring = ogr.Geometry(ogr.wkbLinearRing)

                ring.AddPoint(x_cors[x_idx], y_cors[y_idx])
                ring.AddPoint(x_cors[x_idx + 1], y_cors[y_idx])
                ring.AddPoint(x_cors[x_idx + 1], y_cors[y_idx + 1])
                ring.AddPoint(x_cors[x_idx], y_cors[y_idx + 1])
                ring.AddPoint(x_cors[x_idx], y_cors[y_idx])

                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                inter_poly = poly.Intersection(geom)

                # to get the area, I convert it to coordinate sys of
                # the shapefile that is hopefully in linear units
                inter_area = inter_poly.Area()

            if inter_area < min_area_thresh:
                continue

            n_cells_cntned += 1

            inter_areas.append(inter_area)
            cat_x_idxs.append((x_cors[x_idx] - x_l_c) / cell_size)
            cat_y_idxs.append((y_cors[y_idx] - y_l_c) / cell_size)

    cat_area_ratios_arr = np.divide(inter_areas, cat_area)
    cat_intersect_cols_arr = np.int64(np.round(cat_x_idxs, 6))
    cat_intersect_rows_arr = np.int64(np.round(cat_y_idxs, 6))

    normed_area = np.round(np.sum(cat_area_ratios_arr))
    print('Normalized area sum:', normed_area)

#    tot_time = np.round(timeit.default_timer() - strt_time, 3)
    print(poly_id, n_cells, n_cells_cntned, normed_area)

    return (poly_id,
            cat_intersect_rows_arr,
            cat_intersect_cols_arr,
            cat_area_ratios_arr)
