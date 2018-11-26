'''
Created on Nov 25, 2018

@author: Faizan
'''

from pathlib import Path


class KrigingTypes:

    def __init__(self):

        self._ork_flag = False
        self._spk_flag = False
        self._edk_flag = False
        self._idw_flag = False
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

    def turn_external_drift_kriging_on(
            self,
            drift_rasters):

        assert hasattr(drift_rasters, '__iter__')

        self._drft_rass = []

        for drift_raster in drift_rasters:
            assert isinstance(drift_raster, (str, Path))

            drift_raster_path = Path(drift_raster).absolute()
            assert drift_raster_path.exists()
            assert drift_raster_path.is_file()

            self._drft_rass.append(drift_raster_path)

        self._drft_rass = tuple(self._drft_rass)

        self._n_drft_rass = len(self._drft_rass)

        assert self._n_drft_rass

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

            print(
                'Set external_drift kriging flag to False and deleted '
                'drift_rasters.')

            print('#' * 10)

        self._drft_rass = None
        self._n_drft_rass = None

        self._edk_flag = False
        return

    def turn_inverse_distance_weighting_on(
            self,
            idw_exps):

        assert hasattr(idw_exps, '__iter__')

        self._idw_exps = []

        for idw_exp in idw_exps:
            assert isinstance(idw_exp, (int, float))

            self._idw_exps.append(float(idw_exp))

        self._idw_exps = tuple(self._idw_exps)

        self._n_idw_exps = len(self._idw_exps)

        assert self._n_idw_exps

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
