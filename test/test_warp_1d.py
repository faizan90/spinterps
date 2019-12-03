'''
@author: Faizan-Uni-Stuttgart

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from spinterps import WarpOneDim

plt.ioff()


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    src_x_data = np.linspace(0, 10, 11, dtype=np.float64)

    x_label = 'X'

    kernel_labels = np.array(['TRI', 'TRI'], dtype='<U16')

    kernel_prms = np.array(
        [[4.9, 7, 5, 1],
         [3, 5.1, 5, 1],
        ],
        dtype=np.float64)

    kernel_transs = np.array(
        [1, 1],
        dtype=np.float64)

#     kernel_labels = np.array(['TRI'], dtype='<U16')
#
#     kernel_prms = np.array(
#         [
#             [src_x_data.min(), src_x_data.max(), 5, 1],
#         ],
#         dtype=np.float64)
#
#     kernel_transs = np.array(
#         [-2],
#         dtype=np.float64)

#     kernel_labels = np.array(['TRI'], dtype='<U16')
#
#     kernel_prms = np.array(
#         [
#             [3, 5.1, 5, 1],
#         ],
#         dtype=np.float64)
#
#     kernel_transs = np.array(
#         [6],
#         dtype=np.float64)

    verbose = True

    warp_cls = WarpOneDim(verbose)

    warp_cls.set_src_one_dim_crds(src_x_data, x_label)

    warp_cls.set_kernel_one_dim_labels(kernel_labels)
    warp_cls.set_kernel_one_dim_prms(kernel_prms)
    warp_cls.set_kernel_one_dim_translations(kernel_transs)

    warp_cls.set_strength_flags(np.ones(src_x_data.size, dtype=np.int8))

    warp_cls._verify()

    warp_cls.compute_strength_one_dim_values()

    warp_cls.compute_warp_one_dim_values()

    dst_x_data = warp_cls.dst_crds

    strength_raw_values = warp_cls._strength_one_dim_raw_values

    for i in range(src_x_data.shape[0]):
        print(
            f'{src_x_data[i]:0.3f}',
            f'{dst_x_data[i]:0.3f}',
            strength_raw_values[:, i])

    plt.plot(
        [src_x_data[0], src_x_data[-1]],
        [src_x_data[0], src_x_data[-1]],
        alpha=0.8,
        lw=2)

    plt.scatter(src_x_data, dst_x_data, alpha=0.8, s=3, c='r')

    plt.grid()

    plt.show()

#     raise Exception

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

#     try:
#         main()
#
#     except:
#         import pdb
#         pdb.post_mortem()

    main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
