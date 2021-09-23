'''
@author: Faizan3800X-Uni

Jul 5, 2021

7:43:30 AM
'''

from pathlib import Path

import pandas as pd

from ..misc import print_sl, print_el, get_n_cpus
from ..variograms.vgsinput import VariogramsData as VD


class VGCSettings(VD):

    '''
    The settings class for variogram clustering. It is supposed to be
    subclassed by another one.

    Here, all the required settings and inputs are specified that may be
    used by other classes that do the actual clustering. The data
    is set in the relevant methods of the VariogramsData class (whose child
    is this class).

    Settings and inputs are specified for empirical variogram computation,
    theoretical variogram fitting to the empirical. The empirical may be
    based on clustering vectors/timesteps.

    Finally, the fitted theoretical variograms, if more than one, may
    also be clustered to an even smaller number.

    Chosen variograms for each timestep/vector are saved as a text series.

    For more please read the full documentations of all the classes in this
    module.

    To see usage, take a look at the test_fitcvgs.py in the test
    directory of the spinterps module.

    Last updated: 2021-Sep-23
    '''

    _sett_clus_cevg_types = ('months', 'years', 'manual', 'none',)
    _sett_clus_cevg_evg_stats = ('mean', 'median', 'min', 'max',)
    _sett_clus_tvg_tvgs_all = ('Sph', 'Exp', 'Nug', 'Gau', 'Pow', 'Lin',)

    def __init__(self, verbose=True):

        VD.__init__(self, verbose), f'verbose not of the boolean datatype!'

        self._sett_clus_cevg_type = None
        self._sett_clus_cevg_ts = None
        self._sett_clus_cevg_ts_nan = None
        self._sett_clus_cevg_ignr_zeros_flag = False
        self._sett_clus_cevg_min_vlds = None
        self._sett_clus_cevg_smoothing = None
        self._sett_clus_cevg_evg_stat = None
        self._sett_clus_cevg_pstv_dfnt_flag = False
        self._sett_clus_cevg_simplify_flag = False
        self._sett_clus_cevg_norm_flag = False
        self._sett_clus_cevg_max_dist_thresh = None

        self._sett_clus_tvg_tvgs = None
        self._sett_clus_tvg_perm_sizes = None
        self._sett_clus_tvg_max_iters = None
        self._sett_clus_tvg_max_range = None
        self._sett_clus_tvg_apply_wts_flag = None

        self._sett_clus_ctvg_n_distances = None
        self._sett_clus_ctvg_ks_alpha = None
        self._sett_clus_ctvg_min_kg_abs_wt = None
        self._sett_clus_ctvg_max_nebs = None

        self._sett_clus_misc_outs_dir = None
        self._sett_clus_misc_n_cpus = None
        self._sett_clus_figs_flag = None

        self._sett_clus_cevg_set_flag = False
        self._sett_clus_tvg_set_flag = False
        self._sett_clus_ctvg_set_flag = False
        self._sett_clus_misc_set_flag = False
        self._sett_clus_verify_flag = False
        return

    def set_empirical_variogram_clustering_parameters(
            self,
            clus_type,
            ignore_zeros_flag,
            n_min_valid_values,
            smoothing,
            evg_stat,
            pstv_dfnt_flag,
            simplify_flag,
            norm_flag,
            max_dist_thresh,
            clus_ts=None,
            clus_ts_nan=None):

        f'''
        Specify settings for the computing empirical variogram(s).

        Parameters
        ----------
        clus_type : str
            Specify the method of clustering the input series of vectors.
            Each vector (row) represents a timestep or something similar.
            Each component of the vector represents a dimensions.
            So, each column can be seen as a timeseries of a station.
            clus_type determines how to take the vectors. e.g. if it
            is "months" then all the time series values that happen in
            a given month are seen as single timestep where the distance
            between the pairs are same but the difference between their
            values keeps changing. Same applies for "years".
            If "manual" then the argument "clus_ts" must be passed as a 1D
            vector whose length is equal to the number of vectors in the
            main data series. Then vectors are clustered based on the indices
            of unique values in "clus_ts" e.g. all the timesteps with two
            are taken as a single timestep to whom an empirical variogram
            is fitted. It is a generic form of having "months" or "years"
            as a clustering type. If "clus_ts_nan" is given then timesteps
            with the value clus_ts_nan are not used or fitted with anything.
            If clus_type is "none" then each vector is taken as is and
            a variogram fitted. This is equivalent to the classical
            variogram fitting. clus_type must be a string and only one of the
            values from {self._sett_clus_cevg_types}.
        ignore_zeros_flag : bool
            Whether to ignore the zero difference between any two components,
            while computing the variogram cloud. This may be important for
            variables such as precipitation where no rain values dominate
            the distribution and can result in low variance. Must be a
            boolean value.
        n_min_valid_values : int
            For each pair in each cluster, the possible number of values
        '''

        if self._vb:
            print_sl()

            print(
                'Setting parameters for clustering empirical variograms...\n')

        assert isinstance(clus_type, str)
        assert isinstance(ignore_zeros_flag, bool)
        assert isinstance(n_min_valid_values, int)
        assert isinstance(smoothing, (int, float))
        assert isinstance(evg_stat, str)
        assert isinstance(pstv_dfnt_flag, bool)
        assert isinstance(simplify_flag, bool)
        assert isinstance(norm_flag, bool)
        assert isinstance(max_dist_thresh, (int, float))
        assert isinstance(clus_ts, (pd.Series, type(None)))
        if clus_ts is not None:
            assert clus_ts.values.dtype == type(clus_ts_nan), (
                clus_ts.values.dtype, type(clus_ts_nan))

        assert clus_type in self._sett_clus_cevg_types
        assert n_min_valid_values > 0
        assert smoothing >= 0
        assert evg_stat in self._sett_clus_cevg_evg_stats
        assert max_dist_thresh > 0

        self._sett_clus_cevg_type = clus_type
        self._sett_clus_cevg_ignr_zeros_flag = ignore_zeros_flag
        self._sett_clus_cevg_min_vlds = n_min_valid_values
        self._sett_clus_cevg_smoothing = smoothing
        self._sett_clus_cevg_evg_stat = evg_stat
        self._sett_clus_cevg_pstv_dfnt_flag = pstv_dfnt_flag
        self._sett_clus_cevg_simplify_flag = simplify_flag
        self._sett_clus_cevg_norm_flag = norm_flag
        self._sett_clus_cevg_max_dist_thresh = max_dist_thresh
        self._sett_clus_cevg_ts = clus_ts
        self._sett_clus_cevg_ts_nan = clus_ts_nan

        if self._vb:

            print(
                f'Variogram clustering type: '
                f'{self._sett_clus_cevg_type}')

            print(
                f'Ignore zeros in variogram cloud: '
                f'{self._sett_clus_cevg_ignr_zeros_flag}')

            print(
                f'Minimum number of values required to make a '
                f'variogram cloud for each time series: '
                f'{self._sett_clus_cevg_min_vlds}')

            print(
                f'Smoothing: '
                f'{self._sett_clus_cevg_smoothing}')

            print(
                f'Empirical variogram computing statistic: '
                f'{self._sett_clus_cevg_evg_stat}')

            print(
                f'Make empirical variogram positive definite flag: '
                f'{self._sett_clus_cevg_pstv_dfnt_flag}')

            print(
                f'Simplify empirical variogram flag: '
                f'{self._sett_clus_cevg_simplify_flag}')

            print(
                f'Normalize empirical variogram flag: '
                f'{self._sett_clus_cevg_norm_flag}')

            print(
                f'Maximum distance threshold for the empirical variogram: '
                f'{self._sett_clus_cevg_max_dist_thresh}')

            if self._sett_clus_cevg_ts is not None:
                print(
                    f'Clustering time series\' shape: '
                    f'{self._sett_clus_cevg_ts.shape}')

                print(
                    f'Clustering time series NaN value: '
                    f'{self._sett_clus_cevg_ts_nan}')

            print_el()

        self._sett_clus_cevg_set_flag = True
        return

    def set_theoretical_variogram_parameters(
            self,
            theoretical_variograms,
            permutation_sizes,
            max_opt_iters,
            max_variogram_range,
            apply_wts_flag):

        if self._vb:
            print_sl()

            print(
                'Setting parameters for theoretical variogram fitting...\n')

        assert isinstance(theoretical_variograms, (tuple, list))

        theoretical_variograms = set(theoretical_variograms)

        assert len(theoretical_variograms)
        assert all([isinstance(tvg, str) for tvg in theoretical_variograms])
        assert all([tvg in self._sett_clus_tvg_tvgs_all
                    for tvg in theoretical_variograms])

        theoretical_variograms = tuple(theoretical_variograms)

        assert isinstance(permutation_sizes, (tuple, list))

        permutation_sizes = set(permutation_sizes)

        assert len(permutation_sizes)
        assert all([isinstance(ps, int) for ps in permutation_sizes])
        assert all([ps > 0 for ps in permutation_sizes])

        permutation_sizes = tuple(permutation_sizes)

        assert isinstance(max_opt_iters, int)
        assert max_opt_iters > 0

        assert isinstance(max_variogram_range, (int, float))
        assert max_variogram_range > 0

        assert isinstance(apply_wts_flag, bool)

        self._sett_clus_tvg_tvgs = theoretical_variograms
        self._sett_clus_tvg_perm_sizes = permutation_sizes
        self._sett_clus_tvg_max_iters = max_opt_iters
        self._sett_clus_tvg_max_range = float(max_variogram_range)
        self._sett_clus_tvg_apply_wts_flag = apply_wts_flag

        if self._vb:
            print(
                f'Theoretical variograms to use while fitting: '
                f'{self._sett_clus_tvg_tvgs}')

            print(
                f'Permutation sizes to try while fitting: '
                f'{self._sett_clus_tvg_perm_sizes}')

            print(
                f'Maximum optimization iterations: '
                f'{self._sett_clus_tvg_max_iters}')

            print(
                f'Maximum variogram range: '
                f'{self._sett_clus_tvg_max_range}')

            print(
                f'Apply weights based to distance from origin: '
                f'{self._sett_clus_tvg_apply_wts_flag}')

            print_el()

        self._sett_clus_tvg_set_flag = True
        return

    def set_theoretical_variogram_clustering_parameters(
            self,
            n_distance_intervals,
            ks_alpha,
            min_abs_kriging_weight,
            max_nrst_nebs):

        if self._vb:
            print_sl()

            print(
                'Setting parameters for theoretical variogram clustering...\n')

        assert isinstance(n_distance_intervals, int)
        assert isinstance(ks_alpha, float)
        assert isinstance(min_abs_kriging_weight, float)
        assert isinstance(max_nrst_nebs, int)

        assert n_distance_intervals > 1
        assert 0 < ks_alpha <= 1
        assert min_abs_kriging_weight >= 0
        assert max_nrst_nebs > 0

        self._sett_clus_ctvg_n_distances = n_distance_intervals
        self._sett_clus_ctvg_ks_alpha = ks_alpha
        self._sett_clus_ctvg_min_kg_abs_wt = min_abs_kriging_weight
        self._sett_clus_ctvg_max_nebs = max_nrst_nebs

        if self._vb:
            print(
                f'Number of discretizations for theoretical variograms: '
                f'{self._sett_clus_ctvg_n_distances}')

            print(
                f'Kolmogorov-Smirnov\'s Alpha: '
                f'{self._sett_clus_ctvg_ks_alpha}')

            print(
                f'Minimum absolute kriging weight to consider: '
                f'{self._sett_clus_ctvg_min_kg_abs_wt}')

            print(
                f'Maximum nearest neighbors for clustering: '
                f'{self._sett_clus_ctvg_max_nebs}')

        self._sett_clus_ctvg_set_flag = True
        return

    def set_misc_settings(self, outputs_dir, n_cpus, plot_figs_flag):

        '''
        Some more parameters

        Parameters
        ----------
        outputs_dir : str, Path-like
            Path to the directory where the outputs will be stored.
            Created if not there.
        n_cpus : string, integer
            Maximum number of processes to use to generate realizations.
            If the string 'auto' then the number of logical cores - 1
            processes are used. If an integer > 0 then that number of
            processes are used.
        '''

        if self._vb:
            print_sl()

            print(
                'Setting misc. settings for variogram clustering/fitting...\n')

        assert isinstance(outputs_dir, (str, Path))

        outputs_dir = Path(outputs_dir).absolute()

        assert outputs_dir.is_absolute()

        assert outputs_dir.parents[0].exists(), (
            'Parent directory of outputs dir does not exist!')

        if not outputs_dir.exists():
            outputs_dir.mkdir(exist_ok=True)

        if isinstance(n_cpus, str):
            assert n_cpus == 'auto', 'Invalid n_cpus!'

            n_cpus = get_n_cpus()

        else:
            assert isinstance(n_cpus, int), 'n_cpus is not an integer!'

            assert n_cpus > 0, 'Invalid n_cpus!'

        assert isinstance(plot_figs_flag, bool)

        self._sett_clus_misc_outs_dir = outputs_dir
        self._sett_clus_misc_n_cpus = n_cpus
        self._sett_clus_figs_flag = plot_figs_flag

        if self._vb:
            print(
                f'Outputs directory: '
                f'{self._sett_clus_misc_outs_dir}')

            print(
                f'Number of maximum process(es) to use: '
                f'{self._sett_clus_misc_n_cpus}')

            print(
                f'Plot figures flag: '
                f'{self._sett_clus_figs_flag}')

            print_el()

        self._sett_clus_misc_set_flag = True
        return

    def verify(self):

        assert self._data_set_flag
        assert self._sett_clus_cevg_set_flag
        assert self._sett_clus_misc_set_flag

        if self._sett_clus_ctvg_set_flag:
            assert self._sett_clus_tvg_set_flag

        self._sett_clus_verify_flag = True
        return

    __verify = verify

