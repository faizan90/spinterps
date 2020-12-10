'''
@author: Faizan-Uni-Stuttgart

Dec 9, 2020

12:00:04 PM

'''
import os
import time
import timeit
from pathlib import Path
from math import factorial
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = True


def get_postve_dfnt_vg(vg_vals):

    n_vals = vg_vals.size

    pd_vg_vals = np.full(n_vals, np.nan)
    pd_vg_vals[0] = vg_vals[0]

    for i in range(1, n_vals):
        pval = pd_vg_vals[i - 1]
        cval = vg_vals[i]

        if cval < pval:
            pd_vg_vals[i] = pval

        else:
            pd_vg_vals[i] = cval

    assert np.all(np.isfinite(pd_vg_vals))
    assert np.all((pd_vg_vals[1:] - pd_vg_vals [:-1]) >= 0)

#     plt.title('Positive definite')
#     plt.plot(vg_vals, lw=2)
#     plt.plot(pd_vg_vals, lw=1)
#     plt.show()
#     plt.close()

    return pd_vg_vals


def get_simplified_vg(dists, vg_vals):

    vg_vals_diffs = vg_vals[1:] - vg_vals[:-1]

    take_idxs_raw = ~(np.isclose(vg_vals_diffs, 0))
    take_idxs = take_idxs_raw.copy()

    for i in range(1, take_idxs_raw.size):
        if take_idxs_raw[i]:
            take_idxs[i - 1] = True

    take_idxs[-1] = True

    sdists = np.full(take_idxs.sum() + 1, np.nan)
    svg_vals = sdists.copy()

    sdists[0] = dists[0]
    svg_vals[0] = vg_vals[0]

    sdists[1:] = dists[1:][take_idxs]
    svg_vals[1:] = vg_vals[1:][take_idxs]

    assert np.all(np.isfinite(sdists)) & np.all(np.isfinite(svg_vals))

#     plt.title('Simplified')
#     plt.plot(dists, vg_vals, lw=2)
#     plt.plot(sdists, svg_vals, lw=1)
#     plt.draw()
#     plt.show()
#     plt.close()

    return sdists, svg_vals


def get_smoothed_arr(in_arr, win_size, smooth_ftn_type):

    n_vals = in_arr.shape[0]

    smooth_ftn = getattr(np, smooth_ftn_type)

    smoothed_arr = np.zeros(n_vals - win_size + 1)
    for i in range(smoothed_arr.size):
        smoothed_arr[i] = smooth_ftn(in_arr[i:i + win_size])

    return smoothed_arr


def cmpt_comb_vg(args):

    (in_data_file,
     sep,
     classi_type,
     use_months,
     use_years,
     time_fmt,
     beg_time,
     end_time,
     ignore_zero_input_data_flag,
     in_crds_file,
     crds_cols,
     n_thresh_vals,
     ignore_zero_vg_flag,
     smoothing_ratio,
     fig_size,
     out_fig_path,
     dpi,
     y_lims,
     postve_dfnt_flag,
     simplify_flag,
    ) = args

    data_df = pd.read_csv(in_data_file, sep=sep, index_col=0)
    data_df.index = pd.to_datetime(data_df.index, format=time_fmt)

    data_df = data_df.loc[beg_time:end_time]

    if classi_type == 'months':
        month_idxs = np.zeros(data_df.shape[0], dtype=bool)
        for use_month in use_months:
            month_idxs |= data_df.index.month == use_month

        assert month_idxs.sum()

        data_df = data_df.loc[month_idxs]

    elif classi_type == 'years':
        year_idxs = np.zeros(data_df.shape[0], dtype=bool)
        for use_year in use_years:
            year_idxs |= data_df.index.year == use_year

        assert year_idxs.sum()

        data_df = data_df.loc[year_idxs]

    elif classi_type == 'none':
        pass

    else:
        raise ValueError(f'Unknown classi_type: {classi_type}!')

    if ignore_zero_input_data_flag:
        data_df.replace(0, np.nan, inplace=True)

    data_df.dropna(how='all', axis=1, inplace=True)

    crds_df = pd.read_csv(in_crds_file, sep=sep, index_col=0)

    crds_df = crds_df.loc[data_df.columns, crds_cols]

    n_stns = crds_df.shape[0]
    n_recs = data_df.shape[0]

    print('Total stations:', n_stns)
    print('Total records:', n_recs)

    assert all(data_df.shape)

    nnan_idxs_df = data_df.notna()
    n_combs = int(
        factorial(n_stns) / (factorial(2) * factorial(n_stns - 2)))

    print('No. of combinations:', n_combs)

    dists = np.full(n_combs, np.nan)
    vg_vals = np.full(n_combs, np.nan)

    comb_ctr = -1
    for i, stn_i in enumerate(crds_df.index):
        stn_i_vals = data_df[stn_i].values.copy()
        stn_i_nnan_idxs = nnan_idxs_df[stn_i].values

        crds_i = crds_df.loc[stn_i, crds_cols].values

        for j, stn_j in enumerate(crds_df.index):

            if i <= j:
                continue

            comb_ctr += 1

            stn_j_nnan_idxs = nnan_idxs_df[stn_j].values

            cmn_nnan_idxs = stn_i_nnan_idxs & stn_j_nnan_idxs
            cmn_n_idxs = cmn_nnan_idxs.sum()

            if cmn_n_idxs < n_thresh_vals:
                continue

            stn_j_vals = data_df[stn_j].values.copy()
            crds_j = crds_df.loc[stn_j, crds_cols].values

            comb_vg_vals = 0.5 * (
                (stn_i_vals[cmn_nnan_idxs] - stn_j_vals[cmn_nnan_idxs]) ** 2)

            if ignore_zero_vg_flag:
                zero_vg_idxs = comb_vg_vals > 0
                n_zero_vals = zero_vg_idxs.sum()

                if not n_zero_vals:
                    continue

                comb_vg_vals = comb_vg_vals[zero_vg_idxs]

            comb_vg_vals_stat = np.median(comb_vg_vals)

            dist = ((crds_i - crds_j) ** 2).sum() ** 0.5

            vg_vals[comb_ctr] = comb_vg_vals_stat
            dists[comb_ctr] = dist

    dists_nnan_idxs = np.isfinite(dists)
    print('dists_nnan_idxs:', dists_nnan_idxs.sum())
    assert dists_nnan_idxs.sum()

    dists = dists[dists_nnan_idxs]
    vg_vals = vg_vals[dists_nnan_idxs]

    assert np.all(np.isfinite(dists)) and np.all(np.isfinite(vg_vals))

#     if normalize_vg_flag:
#         vg_vals /= vg_vals.max()

    dists_sort_idxs = np.argsort(dists)
    dists_sorted = dists[dists_sort_idxs]
    vg_vals_sorted = vg_vals[dists_sort_idxs]

    if smoothing_ratio > 1:
        n_smooth_vals = min(smoothing_ratio, dists_sorted.size)

    elif smoothing_ratio == 0:
        n_smooth_vals = 1

    else:
        n_smooth_vals = int(smoothing_ratio * dists_sorted.size)

#     print('n_smooth_vals:', n_smooth_vals)
    assert n_smooth_vals

    dists_smoothed = get_smoothed_arr(dists_sorted, n_smooth_vals, 'mean')
    vg_vals_smoothed = get_smoothed_arr(
        vg_vals_sorted, n_smooth_vals, 'median')

    if postve_dfnt_flag:
        vg_vals_smoothed = get_postve_dfnt_vg(vg_vals_smoothed)

    if simplify_flag:
        dists_smoothed, vg_vals_smoothed = get_simplified_vg(
            dists_smoothed, vg_vals_smoothed)

    plt.figure(figsize=fig_size)

    plt.scatter(dists_sorted, vg_vals_sorted, alpha=0.3, c='blue')
    plt.plot(dists_smoothed, vg_vals_smoothed, alpha=0.75, c='red')

    plt.yscale('log')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Distance')
    plt.ylabel('Semi-variogram')

    plt.ylim(*y_lims)

    ttl = (
        f'ignore_zero_input_data_flag: {ignore_zero_input_data_flag}, '
        f'ignore_zero_vg_flag: {ignore_zero_vg_flag}\n'
        f'postve_dfnt_flag: {postve_dfnt_flag}, '
        f'simplify_flag: {simplify_flag}\n'
        f'smoothing_ratio: {smoothing_ratio}, '
        f'n_thresh_vals: {n_thresh_vals}, '
        f'crds_cols: {crds_cols}\n'
        f'Class Type: {classi_type}')

    if classi_type == 'months':
        ttl += f', Use Months: {use_months} '

    elif classi_type == 'years':
        ttl += f', Use Years: {use_years} '

    elif classi_type == 'none':
        pass

    else:
        raise ValueError(f'Unknown classi_type: {classi_type}!')

    plt.title(ttl, loc='left')

#     plt.show()

    plt.savefig(str(out_fig_path), dpi=dpi, bbox_inches='tight')
    plt.close()

    if classi_type == 'months':
        out_tup = (dists_smoothed, vg_vals_smoothed, use_months[0])

    elif classi_type == 'years':
        out_tup = (dists_smoothed, vg_vals_smoothed, use_years[0])

    elif classi_type == 'none':
        out_tup = (dists_smoothed, vg_vals_smoothed)

    else:
        raise ValueError(f'Unknown classi_type: {classi_type}!')

    return out_tup


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms\comb_vg')
    os.chdir(main_dir)

#     in_data_file = Path('../temperature_avg.csv')
#     in_crds_file = Path('../temperature_avg_coords.csv')
#     out_dir = Path('temp_1961_2015')

    in_data_file = Path('../precipitation.csv')
    in_crds_file = Path('../precipitation_coords.csv')
    out_dir = Path('ppt_no_zeros_1961_2015')

    sep = ';'
    time_fmt = '%Y-%m-%d'

    beg_time = '1961-01-01'
    end_time = '2015-12-31'

    y_lims = ((1e-2, 1e+2))

#     beg_time = '1990-01-01'
#     end_time = '1990-12-31'

#     classi_type = 'none'
    classi_type = 'months'
#     classi_type = 'years'

    use_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ]
    use_years = np.arange(1980, 1991, 1)

    crds_cols = ['X', 'Y']

    n_thresh_vals = 10
    smoothing_ratio = 500

    fig_size = (13, 7)
    dpi = 200

    dist_prec = 1
    vg_val_prec = 5

    n_cpus = 12

    ignore_zero_input_data_flag = True
    ignore_zero_vg_flag = True
    postve_dfnt_flag = True
    simplify_flag = True

    out_dir.mkdir(exist_ok=True)

    if classi_type == 'months':
        cb_lab = 'Month'
        cmap_vmin = 1.0
        cmap_vmax = 12.0
        cmap_xticks = np.linspace(0, 1, 12)
        cmap_xticklabels = np.linspace(1, 12, 12)
        cmpr_id = 'M'

        args_gen = (
            (in_data_file,
             sep,
             classi_type,
             [use_month],
             None,
             time_fmt,
             beg_time,
             end_time,
             ignore_zero_input_data_flag,
             in_crds_file,
             crds_cols,
             n_thresh_vals,
             ignore_zero_vg_flag,
             smoothing_ratio,
             fig_size,
             out_dir / f'{cmpr_id}{use_month}.png',
             dpi,
             y_lims,
             postve_dfnt_flag,
             simplify_flag,
        ) for use_month in use_months)

    elif classi_type == 'years':
        cb_lab = 'Year'
        cmap_vmin = min(use_years)
        cmap_vmax = max(use_years)

        cmap_xticks = np.linspace(
            0, 1, len(use_years))

        cmap_xticklabels = np.linspace(
            min(use_years), max(use_years), len(use_years))

        cmpr_id = 'Y'

        args_gen = (
            (in_data_file,
             sep,
             classi_type,
             None,
             [use_year],
             time_fmt,
             beg_time,
             end_time,
             ignore_zero_input_data_flag,
             in_crds_file,
             crds_cols,
             n_thresh_vals,
             ignore_zero_vg_flag,
             smoothing_ratio,
             fig_size,
             out_dir / f'{cmpr_id}{use_year}.png',
             dpi,
             y_lims,
             postve_dfnt_flag,
             simplify_flag,
        ) for use_year in use_years)

    elif classi_type == 'none':
        cmpr_id = 'N'

        args_gen = (
            (in_data_file,
             sep,
             classi_type,
             None,
             None,
             time_fmt,
             beg_time,
             end_time,
             ignore_zero_input_data_flag,
             in_crds_file,
             crds_cols,
             n_thresh_vals,
             ignore_zero_vg_flag,
             smoothing_ratio,
             fig_size,
             out_dir / f'full_ts.png',
             dpi,
             y_lims,
             postve_dfnt_flag,
             simplify_flag,
        ) for _ in range(1))

        n_cpus = 1

    else:
        raise ValueError(f'Unknown classi_type: {classi_type}!')

    if n_cpus == 1:
        ress = cmpt_comb_vg(next(args_gen))

    else:
        mp_pool = Pool(n_cpus)

        ress = list(mp_pool.imap_unordered(cmpt_comb_vg, args_gen))

        # mp_pool.join()

    with open(out_dir / f'{cmpr_id}_dists.csv', 'w') as dists_hdl:
        for res in ress:
            dists_hdl.write(f'{cmpr_id}{res[2]}{sep}')

            dists_hdl.write(
                f'{sep}'.join([f'{dist:0.{dist_prec}f}' for dist in res[0]]))

            dists_hdl.write('\n')

    with open(out_dir / f'{cmpr_id}_vg_vals.csv', 'w') as vg_vals_hdl:
        for res in ress:
            vg_vals_hdl.write(f'{cmpr_id}{res[2]}{sep}')

            vg_vals_hdl.write(
                f'{sep}'.join([f'{vg_val:0.{vg_val_prec}f}' for vg_val in res[1]]))

            vg_vals_hdl.write('\n')

    if classi_type in ('months', 'years'):
        cmap = plt.get_cmap('winter')
        cmap_mappable = plt.cm.ScalarMappable(cmap=cmap)
        cmap_mappable.set_array([])

        plt.figure(figsize=fig_size)

        for res in ress:
            plt.plot(
                res[0],
#                 res[1] / np.mean(res[1]),
                res[1],
                alpha=0.3,
                color=cmap((res[2] - cmap_vmin) / (cmap_vmax - cmap_vmin)))

        plt.yscale('log')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Distance')
        plt.ylabel('Semi-variogram')

        cb = plt.colorbar(
            cmap_mappable,
            ax=plt.gca(),
            label=cb_lab,
            fraction=0.05,
            aspect=20,
            orientation='vertical')

        cb.set_ticks(cmap_xticks)
        cb.set_ticklabels(cmap_xticklabels)

    #     plt.show()

        plt.savefig(
            str(out_dir / f'{cmpr_id}cmpr.png'),
            dpi=dpi,
            bbox_inches='tight')

        plt.close()

    elif classi_type == 'none':
        pass

    else:
        raise ValueError(f'Unknown classi_type: {classi_type}!')

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    freeze_support()

    if DEBUG_FLAG:
        try:
            main()

        except:
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
