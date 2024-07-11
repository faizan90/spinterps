
import os

# Numpy sneakily uses multiple threads sometimes. I don't want that.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPI_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Due to shitty tkinter errors.
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from .cyth import (
    fill_dists_one_pt,
    fill_wts_and_sum,
    get_mults_sum,
    sel_equidist_refs,
    fill_dists_2d_mat,
    fill_vg_var_arr,
    OrdinaryKriging,
    SimpleKriging,
    ExternalDriftKriging_MD,
    get_nd_dists)

from .variograms import FitVariograms, Variogram
from .interp import SpInterpMain

from .extract import (
    ExtractPolygons,
    ExtractPoints,
    ExtractNetCDFCoords,
    ExtractNetCDFValues,
    ExtractGTiffCoords,
    ExtractGTiffValues,
    GeomAndCrdsItsctIdxs,
    ReOrderIdxs,
    Extract)

from .vgclus import ClusteredVariograms

from .crds import CrdsReProjNC

from .rsmp import ResampleRasToRas, ResampleRasToRasClss, ResampleNCFToRas

from .misc import get_theo_vg_vals, show_formatted_elapsed_time, ret_mp_idxs
