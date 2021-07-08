'''
@author: Faizan3800X-Uni

Jul 5, 2021

7:40:29 AM
'''

from .tvgs_clus import TVGSClus


class ClusteredVariograms(TVGSClus):

    def __init__(self, verbose=True):

        TVGSClus.__init__(self, verbose)
        return
