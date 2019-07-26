'''
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    in_df = pd.read_csv(
        r"P:\Synchronize\IWS\Dhiraj\max_tmp_stns.csv", sep=';', index_col=0)

    n_stns = in_df.shape[0]
    dist_mat = np.empty((n_stns, n_stns), dtype=float)
    min_dist_thresh = 1e1
    max_dist_thresh = 10000

    x_crds = in_df['X'].values
    y_crds = in_df['Y'].values

    for i in range(n_stns):
        for j in range(n_stns):
            dist_mat[i, j] = (
                ((x_crds[i] - x_crds[j]) ** 2) +
                ((y_crds[i] - y_crds[j]) ** 2)) ** 0.5

    dist_mat[dist_mat <= min_dist_thresh] = 0.0

    plt.imshow(dist_mat, vmin=0, vmax=max_dist_thresh)
    plt.colorbar()
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
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
