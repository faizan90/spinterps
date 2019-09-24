'''
@author: Faizan-Uni-Stuttgart

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spinterps import Variogram as VG

plt.ioff()


def get_evg(vals, x_crds, y_crds):

    vg = VG(
        x=x_crds,
        y=y_crds,
        z=vals,
        mdr=1.0,
        nk=10,
        typ='var',
        perm_r_list=[1, 2],
        fil_nug_vg='Nug',
        ld=None,
        uh=None,
        h_itrs=100,
        opt_meth='L-BFGS-B',
        opt_iters=1000,
        fit_vgs=['Sph', 'Exp', 'Gau'],
        n_best=1,
        evg_name='classic',
        use_wts=False,
        ngp=5,
        fit_thresh=0.01)

    vg.fit()

    return vg


def plot_vg(vg, name, out_dir):

    evg = vg.vg_vg_arr
    h_arr = vg.vg_h_arr
    vg_fit = vg.vg_fit
    vg_names = vg.best_vg_names

    fit_vg_list = vg.vg_str_list
#     fit_vgs_no = len(fit_vg_list) - 1

    plt.figure(figsize=(15, 7))

    plt.plot(h_arr, evg, 'bo', alpha=0.3)

    for m in range(len(vg_names)):
        plt.plot(
            vg_fit[m][:, 0],
            vg_fit[m][:, 1],
            c=pd.np.random.rand(3,),
            linewidth=4,
            zorder=m,
            label=fit_vg_list[m],
            alpha=0.6)

    plt.grid()

    plt.xlabel('Distance')
    plt.ylabel('Variogram')

    plt.legend(loc=4, framealpha=0.7)

    plt.savefig(
        str(out_dir / f'{name}.png'),
        bbox_inches='tight')

#     plt.show()
    plt.close()
    return


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\variograms')
    os.chdir(main_dir)

    data_file = r'temperature_avg.csv'
    crds_file = r'temperature_avg_coords.csv'

    out_dir = Path(r'crds_rotation_effect')

    event_time = '1980-05-23'

    data_df = pd.read_csv(data_file, sep=';', index_col=0)
    crds_df = pd.read_csv(crds_file, sep=';', index_col=0)

    data_df.columns = data_df.columns.map(str)
    crds_df.index = crds_df.index.map(str)

    data_ser = data_df.loc[event_time, :].dropna()
    crds_ser = crds_df.loc[data_ser.index, ['X', 'Y']]

    assert data_ser.shape[0] > 10

    out_dir.mkdir(exist_ok=True)

    for j, pha in enumerate(np.linspace(-np.pi, np.pi, 10)):

        crds = np.full((crds_ser.shape[0], 2), np.nan)
        for i in range(crds_ser.shape[0]):
            tfm_arr = np.array(
                [[np.cos(pha), np.sin(pha)], [-np.sin(pha), np.cos(pha)]])

            crds[i] = np.matmul(tfm_arr, crds_ser[['X', 'Y']].values[i, :])

#         print(crds[i], crds_ser[['X', 'Y']].values[i, :])
#
#         print('#' * 70)

        vg = get_evg(data_ser.values, crds[:, 0], crds[:, 1])
        plot_vg(vg, j, out_dir)

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

    main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
