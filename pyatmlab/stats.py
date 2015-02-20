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

def bin_nd_sparse(binners, bins):
    """Similar to bin_nd, but for very sparse data

    When the number of entries to be binned is much smaller than the
    number of bins, it is very inefficient to go through all the bins as
    bin_nd does.  This function takes the same arguments but instead of
    returning an n-dimensional array, it will return an n*p array where n
    are the number of dimensions and p the number of datapoints, returning
    the coordinates that each point would be in.
    """

    M = numpy.empty(shape=(
        binners[0].shape[0] if binners[0].ndim>0 else 1, len(binners)),
            dtype="u2")
    for (i, (binner, bin_n)) in enumerate(zip(binners, bins)):
        if numpy.isnan(binner).any():
            raise ValueError("I found nans in bin {:d}.  I refuse to bin nans!".format(
                i))
        M[:, i] = numpy.digitize(binner, bin_n)
    return M
        

def binsnD_to_2d(bins, ax1, ax2):
    """Collapse n-D bucketing array to 2-D, merging buckets for other dims.

    Take a n-D bucketing array (n>=2), such as returned by bin_nD, and
    merge all buckets thus reducing it to 2D, retaining ax1 and ax2.
    For examble, binsnD_flatiter(bins, 1, 2) will return a bins-array of
    shape (bins.shape[1], bins.shape[2]), where each bucket [i, j] will
    contain all elements in bins[:, i, j, ...].

    :param bins: n-Dimensional array containing buckets.  As returned by
        bin_nD.
    :param ax1: First axis to keep.
    :param ax2: Second axis to keep.
    :returns: 2-Dimensional array of shape (bins.shape[ax1],
        bins.shape[ax2]) with all other dimensions merged.
    """
    slc = [slice(None)] * bins.ndim # NB: slice(None) corresponds to :
    Z = numpy.empty(shape=(bins.shape[ax1], bins.shape[ax2]),
                    dtype="object")
    for i in range(bins.shape[ax1]):
        for j in range(bins.shape[ax2]):
            slc[ax1] = i
            slc[ax2] = j
            Z[i, j] = numpy.concatenate(bins[slc].ravel())

    return Z
