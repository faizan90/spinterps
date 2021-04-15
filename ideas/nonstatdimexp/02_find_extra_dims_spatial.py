'''
@author: Faizan-Uni-Stuttgart

Jan 4, 2021

10:01:36 AM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, Bounds

from spinterps.misc import get_theo_vg_vals

plt.ioff()

DEBUG_FLAG = False


def get_l2_norm(crds_1, crds_2):

    return (((crds_1 - crds_2) ** 2)).sum() ** 0.5


def get_lags_evgs_tvgs(ts_vals, crds, fit_vg_str):

    lags, evgs, tvgs = [], [], []
    for i in range(0, ts_vals.shape[0]):
        val_1 = ts_vals[i]
        crds_1 = crds[i,:]
        for j in range(0, ts_vals.shape[0]):
            if j <= i:
                continue

            diff = (val_1 - ts_vals[j]) ** 2

            dist = get_l2_norm(crds_1, crds[j,:])

            lags.append(dist)
            evgs.append(diff)

    lags = np.array(lags)
    tvgs.append(get_theo_vg_vals(fit_vg_str, lags).tolist())

    return lags, evgs, tvgs


def obj_ftn(new_crds, crds, evgs, n_extra_dims, fit_vg_str, calls_ctr):

    fit_vg_str = f'{new_crds[0]:0.5f} {fit_vg_str}({new_crds[1]:0.1f})'

    old_crds = crds[:, -n_extra_dims:].copy()

    crds[:, -n_extra_dims:] = new_crds[2:].reshape(crds.shape[0], n_extra_dims)

#     print(new_crds.min(), new_crds.max())

#     tvgs = []
#     for i in range(1, n_lags + 1):
#         dists = get_l2_norm(crds[i:,:], crds[:-i,:])
#
#         tvgs.extend(get_theo_vg_vals(fit_vg_str, dists).tolist())

    lags = []
    for i in range(0, crds.shape[0]):
        crds_1 = crds[i,:]
        for j in range(0, crds.shape[0]):
            if j <= i:
                continue

            dist = get_l2_norm(crds_1, crds[j,:])
            lags.append(dist)

    lags = np.array(lags)

    tvgs = get_theo_vg_vals(fit_vg_str, lags)

    obj_val = ((evgs - tvgs) ** 2).sum()

    calls_ctr[0] += 1
    print(f'{obj_val:0.5E}', calls_ctr[0])

    crds[:, -n_extra_dims:] = old_crds

    return obj_val


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms\nonstat_dimexp')
    os.chdir(main_dir)

    in_data_file = Path('../temperature_avg.csv')
    in_crds_file = Path('../temperature_avg_coords.csv')

# 1961 temperature_stns
# Index(['T1197', 'T1254', 'T1468', 'T1875', 'T2074', 'T2080', 'T2088', 'T2095',
#        'T2349', 'T257', 'T2638', 'T2654', 'T2775', 'T2814', 'T2879', 'T2949',
#        'T3135', 'T3257', 'T330', 'T3313', 'T3402', 'T3425', 'T3432', 'T3486',
#        'T355', 'T3621', 'T3671', 'T3761', 'T3924', 'T4287', 'T4300', 'T4330',
#        'T4581', 'T4703', 'T4887', 'T4927', 'T4928', 'T4931', 'T4933', 'T5105',
#        'T5120', 'T5155', 'T5229', 'T5429', 'T5559', 'T5654', 'T5664', 'T5885',
#        'T5906', 'T5971', 'T727', 'T755', 'T772', 'T881'],
#       dtype='object')

    out_dir = Path(f'test_temp_1961_1961_spatial_01')

    fit_vg_str = 'Gau'

#     in_data_file = Path('../precipitation.csv')
#     in_crds_file = Path('../precipitation_coords.csv')
#     out_dir = Path('test_ts')

    sep = ';'
#     time_fmt = '%Y-%m-%d'

    time_step = '1995-01-01'

    min_crd, max_crd = -1e6, +1e6

    out_dir.mkdir(exist_ok=True)

    ts_vals = pd.read_csv(
        in_data_file, sep=sep, index_col=0).loc[time_step]

    ts_vals.dropna(inplace=True)
    crds_df = pd.read_csv(in_crds_file, sep=sep, index_col=0).loc[ts_vals.index]

    ts_vals = ts_vals.values
    crds = crds_df.loc[:, ['X', 'Y']].values

#     crds = np.arange(0, ts_vals.size, dtype=float).reshape(-1, 1)

    crds = np.concatenate((crds, np.zeros_like(crds)), axis=1)
    n_extra_dims = crds.shape[1] // 2

    assert np.all(np.isfinite(ts_vals))

    lags, evgs, _ = get_lags_evgs_tvgs(ts_vals, crds, f'600 {fit_vg_str}(200000)')

    evgs = np.array(evgs)

    bds = np.tile(
        [min_crd, max_crd], 2 + (crds.shape[0] * n_extra_dims)).reshape(-1, 2)

    bds[0, 0] = 10
    bds[0, 1] = 5000
    bds[1, 0] = 100
    bds[1, 1] = 20000000

    bds = Bounds(bds[:, 0], bds[:, 1])

    calls_ctr = np.array([0.0])

    old_crds = np.empty((crds.shape[0], n_extra_dims), order='f')
    tvgs = evgs.copy()
    dists = np.empty(crds.shape[0])

    print('Optimizing...')
    t_beg = timeit.default_timer()

    ress = differential_evolution(
        obj_ftn,
        bounds=bds,
        args=(crds, evgs, n_extra_dims, fit_vg_str, calls_ctr),
        maxiter=500,
        popsize=1,
        polish=False)

    print('Done.')

    print(f'{ress.fun:0.3E}', calls_ctr[0])

    print('Took', timeit.default_timer() - t_beg, 'secs')

    plt.scatter(lags, evgs, alpha=0.1, c='r')

    fit_vg_str = f'{ress.x[0]:0.5f} {fit_vg_str}({ress.x[1]:0.1f})'

    print(fit_vg_str)

    crds[:, -n_extra_dims:] = ress.x[2:].reshape(crds.shape[0], n_extra_dims)

    lags, evgs, tvgs = get_lags_evgs_tvgs(ts_vals, crds, fit_vg_str)

    plt.scatter(lags, evgs, alpha=0.1, c='k')
    plt.scatter(lags, tvgs, alpha=0.1, c='b')

    plt.show()

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
