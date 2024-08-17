# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

Aug 10, 2024

6:10:35 PM

Description:

Keywords:

'''

import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
import netCDF4 as nc
from osgeo import ogr
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

ogr.UseExceptions()

DEBUG_FLAG = False


def main():

    raise NotImplementedError('Use ac_match_stat.py. It is the final one!')

    main_dir = Path(r'P:\Synchronize\IWS\Testings\spinterps\krig_sclg\spinterp_ppt5')
    os.chdir(main_dir)

    ncf_pth = Path(r'kriging.nc')

    vlb = 'OK'  # 'EST_VARS_OK'  #  'EDK'  # 'IDW_000'
    elb = 'EST_VARS_OK'
    # xlb = 'X'  # 'x_utm32n'  # 'longitude'  # 'lon'  #
    # ylb = 'Y'  # 'y_utm32n'  # 'latitude'  # 'lat'  #
    tlb = 'time'

    beg_tme, end_tme = pd.to_datetime(
        ['1990-01-01', '1990-12-31'], format='%Y-%m-%d')

    ply_pth = Path(r'../bayern_epsg32632.shp')
    ply_fld = 'ID_0'
    ply_bfr_dst = 10e3

    cds_pth = Path(r'../ppt_1D_gkd_dwd_crds.csv')
    tss_pth = Path(r'../ppt_1D_gkd_dwd_tss.pkl')

    nan_val = None
    #==========================================================================

    if cds_pth is not None:
        cds_df = pd.read_csv(cds_pth, sep=';', index_col=0)[['X', 'Y']]
        cds_df.index = cds_df.index.astype(str)

    with nc.Dataset(ncf_pth, 'r') as nc_hdl:
        # xcs = nc_hdl[xlb][...].data
        # ycs = nc_hdl[ylb][...].data

        tme_sps = nc.num2date(
            nc_hdl[tlb][:].data,
            units=nc_hdl[tlb].units,
            calendar=nc_hdl[tlb].calendar,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True)

        tme_sps = pd.DatetimeIndex(tme_sps)

        take_idxs_beg = tme_sps.get_loc(beg_tme)
        take_idxs_end = tme_sps.get_loc(end_tme) + 1

        tme_sps = tme_sps[take_idxs_beg:take_idxs_end]

        vbe_gds = nc_hdl[vlb][take_idxs_beg:take_idxs_end].data
        vbe_etn_vie_gds = nc_hdl[elb][take_idxs_beg:take_idxs_end].data ** 0.5

        vbe_gds_ctd = np.full(nc_hdl[vlb].shape, np.nan, dtype=vbe_gds.dtype)

        if nan_val is not None:
            vbe_gds[vbe_gds == nan_val] = np.nan
    #==========================================================================

    ply_hdl = fiona.open(str(ply_pth))

    tke_nms = get_ctd_cds_nms(
        cds_pth,
        ply_pth,
        ply_fld,
        0.0,
        ply_bfr_dst)

    tss_dfe = pd.read_pickle(tss_pth)

    cmn_nms = tke_nms.intersection(tss_dfe.columns)

    tss_dfe = tss_dfe.loc[:, cmn_nms]

    std_dns = np.full((tme_sps.size, 3), np.nan)
    for i, tme_stp in enumerate(tme_sps):

        # print(tme_stp)

        ref_std = tss_dfe.loc[tme_stp].std()
        tst_std = np.nanstd(vbe_gds[i,:,:])
        tst_mean = np.nanmean(vbe_gds[i,:,:])

        tst_grd_ctd, tst_std_ctd, tst_slr = get_slr_mlt(
            tst_mean,
            tst_std,
            ref_std,
            vbe_etn_vie_gds[i,:,:],
            vbe_gds[i,:,:])

        print(tme_stp, ref_std, tst_std, tst_std_ctd, tst_slr)

        vbe_gds_ctd[take_idxs_beg + i,:,:] = tst_grd_ctd

        std_dns[i,:] = ref_std, tst_std, tst_std_ctd

        # print('\n')
    #==========================================================================

    with nc.Dataset(ncf_pth, 'r+') as nc_hdl:

        vbe_ctd_lbl = f'{vlb}__CTD'

        if vbe_ctd_lbl in nc_hdl.variables:
            vbe_ctd_dst = nc_hdl[vbe_ctd_lbl]

        else:
            vbe_dst = nc_hdl[vlb]

            vbe_ctd_dst = nc_hdl.createVariable(
                vbe_ctd_lbl, vbe_gds_ctd.dtype, vbe_dst.dimensions)

            vbe_dst = None

        vbe_ctd_dst[:] = vbe_gds_ctd
    #==========================================================================

    plt.figure()

    min_lmt = np.nanmin(std_dns)
    max_lmt = np.nanmax(std_dns)

    min_lmt -= 0.025 * (max_lmt - min_lmt)
    max_lmt += 0.025 * (max_lmt - min_lmt)

    plt.plot([min_lmt, max_lmt], [min_lmt, max_lmt], alpha=0.75, ls='--', c='b')

    plt.scatter(
        std_dns[:, 0],
        std_dns[:, 1],
        c='k',
        alpha=0.8,
        edgecolors='none',
        label='OLD',
        s=10 ** 2)

    plt.scatter(
        std_dns[:, 0],
        std_dns[:, 2],
        c='b',
        alpha=0.8,
        edgecolors='none',
        label='NEW',
        s=7 ** 2)

    plt.grid()
    plt.gca().set_axisbelow(True)
    plt.gca().set_aspect('equal')

    plt.legend(loc='lower right')

    plt.xlabel('REF [-]')
    plt.ylabel('SIM [-]')

    plt.show()

    plt.close()
    ply_hdl.close()
    return


def get_slr_mlt(ref_men, ref_std, dst_std, tst_etn_vie_grd, tst_grd):

    '''
    Bisection algorithm.

    Multiplicative type of std correction.

    ref and tst are the same.
    '''

    # A scaling grid.
    dtn = np.full_like(tst_grd, np.nan, dtype=np.float32)
    dtn[np.isfinite(tst_grd)] = 1.0

    # etn_vie_lsr_ixs = np.where((tst_grd < (ref_men * 0.5)) & (tst_etn_vie_grd > 0.3))
    # etn_vie_gtr_ixs = np.where((tst_grd > (ref_men * 1.5)) & (tst_etn_vie_grd > 0.3))

    etn_vie_lsr_ixs = np.where((tst_grd < ref_men) & (tst_etn_vie_grd > 0))
    etn_vie_gtr_ixs = np.where((tst_grd > ref_men) & (tst_etn_vie_grd > 0))

    # etn_vie_max_lsr = np.nanmax(tst_etn_vie_grd[etn_vie_lsr_ixs])
    # etn_vie_max_gtr = np.nanmax(tst_etn_vie_grd[etn_vie_gtr_ixs])

    # NOTE: Scaling could be multiplicative or additive type.
    #       Here multiplicative.
    # dtn[etn_vie_lsr_ixs] = (
    #     0 + (tst_etn_vie_grd[etn_vie_lsr_ixs] / etn_vie_max_lsr))
    #
    # dtn[etn_vie_gtr_ixs] = (
    #     0 + (tst_etn_vie_grd[etn_vie_gtr_ixs] * etn_vie_max_gtr))

    # This reveals the station locations.
    # dtn_lsr = np.zeros_like(tst_etn_vie_grd)
    # dtn_lsr[etn_vie_lsr_ixs] = tst_etn_vie_grd[etn_vie_lsr_ixs]
    # dtn_lsr[np.isnan(tst_grd)] = np.nan
    #
    # dtn_gtr = np.zeros_like(tst_etn_vie_grd)
    # dtn_gtr[etn_vie_gtr_ixs] = tst_etn_vie_grd[etn_vie_gtr_ixs]
    # dtn_gtr[np.isnan(tst_grd)] = np.nan

    dtn_lsr = np.zeros_like(tst_etn_vie_grd)
    dtn_lsr[etn_vie_lsr_ixs] = 1.0
    dtn_lsr[np.isnan(tst_grd)] = np.nan

    dtn_gtr = np.zeros_like(tst_etn_vie_grd)
    dtn_gtr[etn_vie_gtr_ixs] = 1.0
    dtn_gtr[np.isnan(tst_grd)] = np.nan

    # dtn[etn_vie_lsr_ixs] = (
    #     (tst_etn_vie_grd[etn_vie_lsr_ixs]))
    #
    # dtn[etn_vie_gtr_ixs] = (
    #     1 + (tst_etn_vie_grd[etn_vie_gtr_ixs]))

    # dtn[np.isclose(tst_grd, ref_men)] = 0.0

    # dtn /= np.nanmean(dtn)

    if False:
        fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})

        axs = axs.ravel()

        # Make data.
        X = np.arange(dtn_lsr.shape[1])
        Y = np.arange(dtn_lsr.shape[0])[::-1]
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        Z = dtn_lsr
        surf = axs[0].plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Plot the surface.
        Z = dtn_gtr
        surf = axs[1].plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        axs[0].zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        axs[0].zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

        plt.close()

    min_tol = 1e-6

    tol = np.inf

    # if ref_std < dst_std:
    #     slr_lft = (dst_std / ref_std) * 0.5
    #     slr_rht = (ref_std / dst_std) * 1.5 ** 2
    #
    # elif ref_std > dst_std:
    #     slr_lft = 0.0
    #     slr_rht = (ref_std / dst_std) * 1.5
    #
    # else:
    #     return tst_grd, ref_std, 1.0

    slr_lft = 0.0
    slr_rht = 1.0  # max((ref_std / dst_std), (dst_std / ref_std)) * 2.0

    lft_std = np.nanstd(get_sld_grd(tst_grd, dtn_lsr, dtn_gtr, slr_lft, ref_men))
    rht_std = np.nanstd(get_sld_grd(tst_grd, dtn_lsr, dtn_gtr, slr_rht, ref_men))

    if False:
        fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})

        axs = axs.ravel()

        # Make data.
        X = np.arange(dtn_lsr.shape[1])
        Y = np.arange(dtn_lsr.shape[0])[::-1]
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        Z = get_sld_grd(tst_grd, dtn_lsr, dtn_gtr, slr_lft, ref_men)
        surf = axs[0].plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Plot the surface.
        Z = get_sld_grd(tst_grd, dtn_lsr, dtn_gtr, slr_rht, ref_men)
        surf = axs[1].plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        axs[0].zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        axs[0].zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

        plt.close()

    std_mde_pre = np.inf

    fal_flg = False

    itr_ctr = 0
    while tol > min_tol:

        slr_mde = 0.5 * (slr_lft + slr_rht)

        mde_std = np.nanstd(get_sld_grd(tst_grd, dtn_lsr, dtn_gtr, slr_mde, ref_men))

        if ((lft_std <= dst_std <= mde_std) or
            (lft_std >= dst_std >= mde_std)):

            slr_rht = slr_mde
            rht_std = mde_std

            # slr_lft = slr_mde
            # lft_std = mde_std

        elif ((mde_std <= dst_std <= rht_std) or
              ((mde_std >= dst_std >= rht_std))):

            slr_lft = slr_mde
            lft_std = mde_std

            # slr_rht = slr_mde
            # rht_std = mde_std

        else:
            print(f'Failed! ({lft_std:0.3f}, {mde_std:0.3f}, {rht_std:0.3f}, {dst_std:0.3f}, {itr_ctr})')
            fal_flg = True
            break

        tol = abs(std_mde_pre - mde_std)

        if tol < min_tol:
            print(f'Min. Tol. ({itr_ctr})')

        itr_ctr += 1

        std_mde_pre = mde_std

    tst_grd_fnl = get_sld_grd(tst_grd, dtn_lsr, dtn_gtr, slr_mde, ref_men)

    std_fnl = np.nanstd(tst_grd_fnl)

    tol = abs(mde_std - std_fnl)

    grd_fnl = tst_grd_fnl

    if False:
        fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})

        # axs = axs.ravel()

        # Make data.
        X = np.arange(dtn_lsr.shape[1])
        Y = np.arange(dtn_lsr.shape[0])[::-1]
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        Z = tst_grd
        axs[0, 0].plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Plot the surface.
        Z = tst_grd_fnl
        surf = axs[0, 1].plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Plot the surface.
        Z = dtn_lsr * (1 - slr_mde)
        axs[1, 0].plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Plot the surface.
        Z = dtn_gtr * (1 + slr_mde)
        surf = axs[1, 1].plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # axs[0].zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        # axs[0].zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

        plt.close()

    # if not fal_flg:
    #     tst_grd_fnl = get_sld_grd(tst_grd, dtn_lsr, dtn_gtr, slr_mde, ref_men)
    #
    #     std_fnl = np.nanstd(tst_grd_fnl)
    #
    #     tol = abs(mde_std - std_fnl)
    #
    #     grd_fnl = tst_grd_fnl
    #
    # else:
    #     std_fnl = ref_std
    #     slr_mde = 1.0
    #
    #     grd_fnl = tst_grd.copy()

    if False:
        fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})

        axs = axs.ravel()

        # Make data.
        X = np.arange(grd_fnl.shape[1])
        Y = np.arange(grd_fnl.shape[0])[::-1]
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        Z = tst_grd
        surf = axs[0].plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Plot the surface.
        Z = grd_fnl
        surf = axs[1].plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        axs[0].zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        axs[0].zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

        plt.close()

    return grd_fnl, std_fnl, slr_mde


def get_sld_grd(grd, lsr, gtr, slr, ref_mean):

    sld_grd = np.zeros_like(grd)
    sld_grd += lsr * (1 - slr)
    sld_grd += gtr * (1 + slr)
    sld_grd *= grd

    sld_grd = sld_grd + grd

    sld_grd[sld_grd < 0] = 0

    # Correction for the mean.
    sld_grd *= ref_mean / np.nanmean(sld_grd)

    return sld_grd


def get_ctd_cds_nms(
        in_crds_file,
        subset_shp_file,
        subset_shp_fld,
        simplyify_tol,
        shp_buff_dist):

    in_crds_df = pd.read_csv(
        in_crds_file,
        sep=';',
        index_col=0,
        engine='python',
        encoding='latin1')

    cat_poly = get_merged_poly(subset_shp_file, subset_shp_fld, simplyify_tol)
    cat_buff = cat_poly.Buffer(shp_buff_dist)
    assert cat_buff

    in_crds_x_fnt = np.isfinite(in_crds_df['X'].values)
    in_crds_y_fnt = np.isfinite(in_crds_df['Y'].values)

    in_crds_fnt = in_crds_x_fnt & in_crds_y_fnt

    in_crds_df = in_crds_df.loc[in_crds_fnt].copy()

    # Remove duplicate, the user can also implement proper selection
    # because a change station location means a new time series normally.
    keep_crds_stns_steps = ~in_crds_df.index.duplicated(keep='last')
    in_crds_df = in_crds_df.loc[keep_crds_stns_steps]

    in_crds_df.sort_index(inplace=True)

    contain_crds = get_stns_in_poly(in_crds_df, cat_buff)

    subset_crds_stns = in_crds_df.index.intersection(contain_crds)

    print(subset_crds_stns.size, 'stations selected in crds_df!')
    return subset_crds_stns


def get_stns_in_poly(crds_df, poly):

    contain_stns = []

    for i, stn in enumerate(crds_df.index):

        x, y = crds_df.iloc[i].loc[['X', 'Y']]

        pt = ogr.CreateGeometryFromWkt("POINT (%f %f)" % (float(x), float(y)))

        if pt is None:
            print(f'Station {stn} returned a Null point!')
            continue

        if poly.Contains(pt):
            contain_stns.append(stn)

    return contain_stns


def get_merged_poly(in_shp, field='DN', simplify_tol=0):

    '''Merge all polygons with the same ID in the 'field' (from TauDEM)

    Because sometimes there are some polygons from the same catchment,
    this is problem because there can only one cathcment with one ID,
    it is an artifact of gdal_polygonize.
    '''

    cat_ds = ogr.Open(str(in_shp))
    lyr = cat_ds.GetLayer(0)

    feat_dict = {}
    fid_to_field_dict = {}

    feat = lyr.GetNextFeature()
    while feat:
        fid = feat.GetFID()
        f_val = feat.GetFieldAsString(field)
        feat_dict[fid] = feat.Clone()
        fid_to_field_dict[fid] = f_val
        feat = lyr.GetNextFeature()

    fid_list = []
    for fid in list(fid_to_field_dict.keys()):
        fid_list.append(fid)

    if len(fid_list) > 1:
        cat_feat = feat_dict[fid_list[0]]
        merged_cat = cat_feat.GetGeometryRef().Buffer(0)
        for fid in fid_list[1:]:
            # The buffer with zero seems to fix invalid geoms somehow.
            curr_cat_feat = feat_dict[fid].Clone()
            curr_cat = curr_cat_feat.GetGeometryRef().Buffer(0)

            merged_cat = merged_cat.Union(curr_cat)
    else:
        cat = feat_dict[fid_list[0]].Clone()
        merged_cat = cat.GetGeometryRef().Buffer(0)

    if simplify_tol:
        merged_cat = merged_cat.Simplify(simplify_tol)

    cat_ds.Destroy()
    return merged_cat


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
