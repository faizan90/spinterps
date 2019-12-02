'''
Created on Nov 29, 2019

@author: Faizan-Uni
'''


class Base:

    def __init__(self, verbose=True):

        assert isinstance(verbose, bool), 'verbose can only be a boolean!'

        self._vb = bool(verbose)
        return

    def _verify(self):

        raise NotImplementedError

        return
