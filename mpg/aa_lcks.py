# -*- coding: utf-8 -*-

'''
Created on Jul 6, 2024

@author: Faizan-TU Munich
'''


class DummyLock:

    def __init__(self):
        return

    def __enter__(self):
        return

    def __exit__(self, exception_type, exception_value, exception_traceback):

        _ = exception_type, exception_value, exception_traceback
        return


class DummyManager:

    def __init__(self):
        return
