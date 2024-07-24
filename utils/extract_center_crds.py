'''
@author: Faizan-Uni-Stuttgart

Nov 2, 2020

12:43:01 PM

'''
import os
import time
import timeit
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(r'U:\TUM\projects\altoetting\chmg3d\data\G8_100m_1H_kostra_00060_100_uniform')
    os.chdir(main_dir)

    in_h5_file = Path(r'ppt.h5')

    sep = ';'

    float_fmt = '%0.1f'

    out_dir = main_dir

    with h5py.File(in_h5_file, mode='r', driver=None) as h5_hdl:
        keys = [int(key) for key in h5_hdl['x_cen_crds'].keys()]

        for key in keys:
            x_crds = h5_hdl[f'x_cen_crds/{key}'][:]
            y_crds = h5_hdl[f'y_cen_crds/{key}'][:]

            out_df = pd.DataFrame(
                index=np.arange(x_crds.size), columns=['X', 'Y'],
                dtype=float)

            out_df['X'] = x_crds
            out_df['Y'] = y_crds

            out_file_path = (
                out_dir / Path(f'{in_h5_file.stem}__crds_{key}.csv'))

            out_df.to_csv(
                out_file_path,
                sep=sep,
                float_format=float_fmt)

            print('Done with:', out_file_path)

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

