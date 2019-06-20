'''
Created on Dec 10, 2018

@author: Faizan
'''
import pyximport
pyximport.install()

from .idw_nebs import get_idw_arr, sel_equidist_refs
from .krigings import (OrdinaryKriging, SimpleKriging, ExternalDriftKriging_MD)
