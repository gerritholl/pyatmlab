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

    if len(bins) != len(binners):
        raise ValueError("Length of bins must equal length of binners. "
            "Found {} bins, {} binners.".format(len(bins), len(binners)))
    
    for b in bins:
        if b.ndim != 1:
            raise ValueError("Bin-array must be 1-D. "
                "Found {}-D array.".format(b.ndim))

    if not all([b.size == binners[0].size for b in binners[1:]]):
        raise ValueError("All binners must have same length.")

    dims = numpy.array([b.size for b in bins])

    nd = len(binners)
    
    if nd == 0:
        return numpy.array([], dtype=numpy.uint64)

    indices = numpy.arange(binners[0].size, dtype=numpy.uint64)
    if data is None:
        data = indices

    if nd > 1:
        #innerbinned = bin(binners[-1], data, bins[-1])
        innerbinned = bin(binners[-1], indices, bins[-1])
        outerbinned = []
        for (i, ib) in enumerate(innerbinned):
            obinners = [x[ib] for x in binners[:-1]]
            ob = bin_nd(obinners, bins[:-1], data[ib])
            outerbinned.append(ob)

        # go through some effort to make sure v[i, j, ...] is always
        # numpy.uint64, whereas v is numpy.object_
        # do this in steps, see comment in the else:-block below for
        # reasoning
        #
        # We have outerbinned, which has length n_N, and contains ndarrays
        # of size n_1 * n_2 * ... * n_{N-1}.
        #
        # We want V to be n_1 * n_2 * ... * n_N, where N is the number of
        # dimensions we are binning.  
        #
        # The following could /probably/ be do with some sophisticated
        # list comprehension and permutation, but this is clearer.
        
        V = numpy.empty(shape=dims, dtype=numpy.object_)
        for i in range(len(outerbinned)):
            V[..., i] = outerbinned[i]

#        V.T[...] = [
#            [numpy.array(e.tolist(), dtype=numpy.uint64)
#                for e in l] for l in outerbinned]
        return V
        #return numpy.array(v, dtype=numpy.object_)
    else:
        # NB: I should not convert a list-of-ndarrays to an object-ndarray
        # directly.  If all nd-arrays have the some dimensions (such as
        # size x=0), the converted nd-array will have x as an additional
        # dimension, rather than having object arrays inside the
        # container.  To prevent this, explicitly initialise the ndarray.
        binned = bin(binners[0], data, bins[0])
        B = numpy.empty(shape=len(binned), dtype=numpy.object_)
        B[:] = binned
        return B

def bins_4D_to_2D(bins, ax1, ax2):
    """Small helper to concatenate bin contents.

    For example:

    >> bins_2D = cat_bins(bins_4D, 0, 1)

    See also iter_bins4D.

    A more generic bin concatenation is not currently implemented.
    """

    # Not sure how to implement this generically — will stick with only
    # 4D-to-2D for now

    merged = numpy.array(list(iter_bins4D(bins, ax1, ax2))).reshape(
        bins.shape[ax1], bins.shape[ax2])
    return merged
#    for i in range(binned_indices.shape[ax1]):
#        for j in range(binned_indices.shape[ax2]):
#            slc[ax1] = i
#            slc[ax2] = j

#    merged = numpy.array(
#        [numpy.concatenate(binned_indices[i, j, :, :].ravel())
#            for i in range(binned_indices.shape[0])
#            for j in range(binned_indices.shape[1])]).reshape(binned_indices.shape[:2])              
#
#    (i, j, Ellipsis)

def iter_bins4D(bins, ax1, ax2):
    """Helper for bins_4D_to_2D

    Taking a 4D array with arrays (such as returned by bin_nD), slicing it
    around axes ax1 and ax2.  For example, iter_bins4D(bins, 1, 2) where
    bins is [k, l, m, n] will result in a [k, n] array; i.e. [i, j] will
    contain [i, :, :, j].
    """
    slc = [slice(None)] * 4 
    for i in range(bins.shape[ax1]):
        for j in range(bins.shape[ax2]):
            slc[ax1] = i
            slc[ax2] = j
            yield numpy.concatenate(bins[slc].ravel())
