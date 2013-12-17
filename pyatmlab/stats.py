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
        All subsequent dimensions are left untouched.
    :type y: ndarray
    :param bins: Bins according to which sort data.
    :type bins: nd-array (1-D)
    :returns: List of arrays, one element per bin
    """
    if x.size == y.size == 0:
        return [y[()] for b in bins]
    digits = numpy.digitize(x, bins)
    binned = [y[digits == i, ...] for i in range(len(bins))]
    return binned

def bin_nd(binners, bins, data=None):
    """Bin/bucket data in arbitrary number of dimensions

    For example, one can bin geographical data according to lat/lon
    through:

    >>> binned = bin_nd([lats, lons], [lat_bins, lon_bins])

    The actually binned data are the indices for the arrays lats/lons,
    which hopefully corresponds to indices in your actual data.

    Data that does not fit in any bin, is not binned anywhere.

    Note: do NOT pass the 3rd argument, `data`.  This is used purely for
    the implementation using recursion.  Passing anything here explicitly
    is a recipe for disaster.

    :param binners: Axes that data is binned at.  This is akin to the
        x-coordinate in :function bin:.
    :type binners: list of 1-D ndarrays
    :param bins: Edges for the bins according to which bin data
    :type bins: list of 1-D ndarrays
    :returns: n-D ndarray, type 'object'
    """

    nd = len(binners)
    
    if nd == 0:
        return numpy.array([], dtype=numpy.uint64)

    if data is None:
        data = numpy.arange(binners[0].size, dtype=numpy.uint64)

    if nd > 1:
        innerbinned = bin(binners.pop(), data, bins.pop())
        v = []
        for (i, ib) in enumerate(innerbinned):
            v.append(bin_nd([x[ib] for x in binners], bins, ib))
        # go through some effort to make sure v[i, j, ...] is always
        # numpy.uint64, whereas v is numpy.object_
        return numpy.array([
            [numpy.array(e.tolist(), dtype=numpy.uint64)
                for e in l] for l in v])
        #return numpy.array(v, dtype=numpy.object_)
    else:
        return numpy.array(bin(binners[0], data, bins[0]),
                           dtype=numpy.object_)
