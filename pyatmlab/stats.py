#!/usr/bin/env python
# coding: utf-8

"""Various statistical functions

"""

import numpy
import matplotlib.mlab

from typhon.math.stats import (bin, bin_nd)

def bin_nd_sparse(binners, bins):
    """Similar to bin_nd, but for very sparse data

    When the number of entries to be binned is much smaller than the
    number of bins, it is very inefficient to go through all the bins as
    bin_nd does.  This function takes the same arguments but instead of
    returning an n-dimensional array, it will return an n*p array where n
    are the number of dimensions and p the number of datapoints, returning
    the coordinates that each point would be in.
    """

    # FIXME: len(binners), or binners.shape[0]?
    M = numpy.zeros(shape=(len(bins), len(binners)), dtype="u2")
    for i in range(len(bins)):
        binner = binners[:, i]
        bin = bins[i]
        if numpy.isnan(binner).any():
            raise ValueError("I found nans in bin {:d}.  I refuse to bin nans!".format(
                i))
        M[i, :] = numpy.digitize(binner, bin)
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


class PCA(matplotlib.mlab.PCA):
    """Extend mlabs PCA class with a bit more functionality
    """

    def reconstruct(self, Y, truncate=0):
        """Reconstruct original vector from PC vector

        From a vector in PC axes, reconstruct the original vector.

        :param pcdp: L×n vector in PC space
        :param denormalise: 
        """

        truncate = truncate or self.Wt.shape[0]
        yscld = self.Wt[:truncate, :] @ Y[:, :truncate]
        return self.decenter(yscld)

    def decenter(self, yscld):
        """Undo self.center
        """
        return yscld * self.sigma + self.mu

    def estimate(self, y, n):
        """Estimate missing value

        Estimate the missing element for a vector.  For example, if twelve
        channels were used to construct a PCA but we have only 11, we can
        estimate what the 12th would have been, assuming we know which one
        is missing, of course.

        :param y: ndarray of dimensions L × n, where L can be any number
            and n should match the number of PCs.  The element that is to
            be predicted is thus contained in this array (probably masked)
            but not used by this method.
        :param n: index of which element is to be estimated.
        :returns: Full vector of same dimension as y, but with missing
            channel replace by a new estimate.
        """

        idx = numpy.delete(numpy.arange(y.shape[1]), n)

        yscld = self.center(y)

        est = self.decenter((self.Wt[:, idx] @ yscld[:, idx].T).T @ self.Wt)

        return est
