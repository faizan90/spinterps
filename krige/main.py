'''
Created on Nov 25, 2018

@author: Faizan
'''

import os
from pathlib import Path

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

        self._drft_rass = None
        self._n_drft_rass = None

        self._idw_exps = None
        self._n_idw_exps = None

        self._ork_flag = False
        self._spk_flag = False
        self._edk_flag = False
        self._idw_flag = False

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
            ivar_name = interp_arg[3]

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

            del krd_fld_res, krg_fld_ress

        self._nc_hdl.Source = self._nc_hdl.filepath()
        self._nc_hdl.close()
        return

    def turn_ordinary_kriging_on(self):

        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set ordinary kriging flag to True.')

            print('#' * 10)

        self._ork_flag = True
        return

    def turn_ordinary_kriging_off(self):

        assert self._ork_flag
        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set ordinary kriging flag to False.')

            print('#' * 10)

        self._ork_flag = False
        return

    def turn_simple_kriging_on(self):

        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set simple kriging flag to True.')

            print('#' * 10)

        self._spk_flag = True
        return

    def turn_simple_kriging_off(self):

        assert self._spk_flag

        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set simple kriging flag to False.')

            print('#' * 10)

        self._spk_flag = False
        return

    def turn_external_drift_kriging_on(self, drift_rasters):

        '''
        Signal to do external drift kriging.

        Parameters
        ----------
        drift_rasters : iterable of paths
            An iterable (list or tuple) holding paths to the rasters that
            should be used as drifts. All rasters should have the same spatial
            properties and coordinate systems that are compatible with the
            station coordinates.
        '''

        assert hasattr(drift_rasters, '__iter__')

        self._drft_rass = []

        for drift_raster in drift_rasters:
            assert isinstance(drift_raster, (str, Path)), (
                'Supplied drift raster path is not a string or a '
                'pathlib.Path object!')

            drift_raster_path = Path(drift_raster).absolute()

            assert drift_raster_path.exists(), (
                'Supplied drift raster path does not point to a file!')

            assert drift_raster_path.is_file(), (
                'Supplied drift raster path does not point to a file!')

            self._drft_rass.append(drift_raster_path)

        self._drft_rass = tuple(self._drft_rass)

        self._n_drft_rass = len(self._drft_rass)

        assert self._n_drft_rass, 'Zero drift rasters were supplied!'

        if self._vb:
            print('\n', '#' * 10, sep='')

            print(
                'Set external drift kriging flag to True with the '
                'following drift rasters:')

            for drift_raster in self._drft_rass:
                print(str(drift_raster))

            print('#' * 10)

        self._edk_flag = True
        return

    def turn_external_drift_kriging_off(self):

        assert self._edk_flag

        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set external_drift kriging flag to False.')

            print('#' * 10)

        self._drft_rass = None
        self._n_drft_rass = None

        self._edk_flag = False
        return

    def turn_inverse_distance_weighting_on(self, idw_exps):

        '''
        Signal to do inverse distance weighting.

        Parameters
        ----------
        idw_exps : iterable of ints or floats
            The exponents to use in inverse distance weighting.
            Seperate interpolation grids are computed for each exponent.
        '''

        assert hasattr(idw_exps, '__iter__')

        self._idw_exps = []

        for idw_exp in idw_exps:
            assert isinstance(idw_exp, (int, float)), (
                'IDW exponent not a float or an int!')

            self._idw_exps.append(float(idw_exp))

        self._idw_exps = tuple(self._idw_exps)

        self._n_idw_exps = len(self._idw_exps)

        assert self._n_idw_exps, 'Zero IDW exponents given!'

        if self._vb:
            print('\n', '#' * 10, sep='')

            print(
                'Set inverse distance weighting flag to True with the '
                'following exponents:')

            print(self._idw_exps)

            print('#' * 10)

        self._idw_flag = True
        return

    def turn_inverse_distance_weighting_off(self):

        assert self._idw_flag

        if self._vb:
            print('\n', '#' * 10, sep='')

            print('Set inverse distance weighting flag to False.')

            print('#' * 10)

        self._idw_exps = None
        self._n_idw_exps = None

        self._idw_flag = False
        return
