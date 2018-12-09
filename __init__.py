import pyximport
pyximport.install()

from .idw_nebs import get_idw_arr

from .variograms import FitVariograms
from .interp import SpInterpMain
