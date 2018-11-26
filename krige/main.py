'''
Created on Nov 25, 2018

@author: Faizan
'''

import os

from multiprocessing import Pool

from .steps import KrigingSteps as KS
from .data import KrigingData as KD
from .prepare import KrigingPrepare as KP
from ..misc import ret_mp_idxs

os.environ[str('MKL_NUM_THREADS')] = str(1)
os.environ[str('NUMEXPR_NUM_THREADS')] = str(1)
os.environ[str('OMP_NUM_THREADS')] = str(1)


class KrigingMain(KD, KP):

    def __init__(self, verbose=True):

        KD.__init__(self, verbose)
        KP.__init__(self)

        self._vgs_ser = None

        self._plot_polys = None
        self._cntn_idxs = None
        self._plots_dir = None
        self._interp_plot_dirs = None

        self._drft_rass = None
        self._n_drft_rass = None
        self._drft_arrs = None
        self._stns_drft_df = None
        self._drft_ndv = None

        self._cell_bdist = None

        self._idw_exps = None
        self._n_idw_exps = None

        self._interp_args = None

        self._n_cpus = 1
        self._n_cpus_scale = 1
        self._plot_figs_flag = False
        self._cell_size = None
        self._min_var_thr = None
        self._min_var_cut = None
        self._max_var_cut = None

        self._main_vrfd_flag = False
        return

    def verify(self):

        KD._KrigingData__verify(self)

        self._prepare()

        assert self._prpd_flag

        self._main_vrfd_flag = True
        return

    def krige(self):

        assert self._main_vrfd_flag

        if ((self._n_cpus > 1) and
            (self._data_df.shape[0] > (self._n_cpus + 1))):

            # TODO: should be a bit more intelligent
            # NaN steps will finish faster
            # TODO: divide steps based on available memory
            mp_idxs = ret_mp_idxs(
                self._data_df.shape[0],
                (self._n_cpus * self._n_cpus_scale))

            mp_pool = Pool(self._n_cpus)
            krg_map = mp_pool.map

        else:
            mp_idxs = [0, self._data_df.shape[0]]
            krg_map = map

        krg_steps_cls = KS(self)

        for interp_arg in self._interp_args:

            if interp_arg[0] == 'IDW':
                ivar_name = interp_arg[3]

            else:
                ivar_name = interp_arg[0]

            interp_gen = (
                (self._data_df.iloc[mp_idxs[i]: mp_idxs[i + 1]],
                 self._crds_df,
                 mp_idxs[i],
                 mp_idxs[i + 1],
                 interp_arg)

                for i in range(self._n_cpus * self._n_cpus_scale))

            krg_fld_ress = krg_map(krg_steps_cls.get_interp_flds, interp_gen)

            for krd_fld_res in krg_fld_ress:
                self._nc_hdl[ivar_name][
                    krd_fld_res[1]:krd_fld_res[2], :, :] = krd_fld_res[0]

            self._nc_hdl.sync()

            krd_fld_res[0] = None

        self._nc_hdl.Source = self._nc_hdl.filepath()
        self._nc_hdl.close()

        return
