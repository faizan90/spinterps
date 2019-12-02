'''
Created on Nov 29, 2019

@author: Faizan-Uni
'''

import numpy as np


def kernel_triangular(*args):

    dst_crd, src_crd_beg, src_crd_cen, src_crd_end, exponent = args

    if ((dst_crd <= src_crd_beg) or (dst_crd >= src_crd_end)):
        weight = 0.0

    elif src_crd_beg < dst_crd < src_crd_cen:
        weight = (dst_crd - src_crd_beg) / (src_crd_cen - src_crd_beg)

    elif src_crd_cen < dst_crd < src_crd_end:
        weight = (src_crd_end - dst_crd) / (src_crd_end - src_crd_cen)

    else:
        weight = 1.0

    assert 0.0 <= weight <= 1.0

    return weight ** exponent


def kernel_triangular_test(*args):

    '''
    Exclamation marks are added later by another function to errors
    '''

    src_crd_beg, src_crd_cen, src_crd_end, exponent = args

    all_inputs = [src_crd_beg, src_crd_cen, src_crd_end, exponent]

    assert all([isinstance(x, float) for x in all_inputs]), (
            'At least one input to kernel_triangular is not a float')

    assert np.all(np.isfinite(all_inputs)), (
        'At least one input to kernel_triangular is invalid')

    assert src_crd_beg <= src_crd_cen <= src_crd_end, (
        'Source coordinates to kernel_triangular not ascending')

    assert 0 <= exponent < np.inf, (
        'Exponent of kernel_triangular is out ouf bounds')
    return
