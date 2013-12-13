#!/usr/bin/env python
# coding: utf-8

"""Various small statistical functions

"""

import numpy

def bin(x, y, bins):
    """Bin/bucket y according to values of x.

    Returns list of arrays, one element with values for bin.
    Digitising happens with numpy.digitize.

    :param x: Coordinate that y is binned along
    :type x: ndarray (1-D)
    :param y: Data to be binned.  First dimension must match length of x.
    :type y: ndarray
    :param bins: Bins according to which sort data.
    :type bins: nd-array (1-D)
    """
    if x.size == y.size == 0:
        return [y[()] for b in bins]
    digits = numpy.digitize(x, bins)
    binned = [y[digits == i] for i in range(len(bins))]
    return binned
