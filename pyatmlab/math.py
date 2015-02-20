#!/usr/bin/env python
# coding: utf-8

"""Various small mathematical functions

"""

import numpy
import numpy.linalg
import scipy

from . import tools
from .meta import expanddoc

inputs = """:param z: Height
    :type z: ndarray
    :param q: Quantity, dim 0 must be height.
    :type q: ndarray
    :param ignore_negative: Set negative values to 0
    :type ignore_negative: bool"""

@expanddoc
def layer2level(z, q, ignore_negative=False):
    """Converts layer to level. First dim. must be height.

    {inputs}
    :returns: Level-valued quantity.
    """
    dz = z[1:, ...] - z[:-1, ...]
    if ignore_negative:
        q[q<0]=0
    y_avg = (q[1:, ...] + q[:-1, ...])/2
    return (y_avg * numpy.atleast_2d(dz).T)

@expanddoc
def integrate_with_height(z, q, ignore_negative=False):
    """Calculate vertically integrated value

    {inputs}
    :returns: Vertically integrated value
    """

    return layer2level(z, q, ignore_negative).sum(0)

@expanddoc
def cum_integrate_with_height(z, q, ignore_negative=False):
    """Calculate cumulative integral with height

    {inputs}
    :returns: Cumulatively vertically integrated value
    """
    return layer2level(z, q, ignore_negative).cumsum(0)

#@tools.validator # comment out because fails for *args
def average_position_sphere(*args: (lambda a: len(a) in (1,2))):
    """Calculate the average position for a set of angles on a sphere

    This is quite imprecise, errors can be dozens of km.  For more
    advanced calculations, use proj4/pyproj.

    Input can be either:

    :param lat: Vector of latitudes
    :param lon: Vector of longitudes

    Or:

    :param locs: Nx2 ndarray with lats in column 0, lons in column 1
    """

    if len(args) == 1:
        locs = args[0]
        lat = locs[0, :]
        lons = locs[1, :]
    elif len(args) == 2:
        lat, lon = args

    X = numpy.cos(lat) * numpy.cos(lon)
    Y = numpy.cos(lat) * numpy.sin(lon)
    Z = numpy.sin(lat)

    xm = X.mean()
    ym = Y.mean()
    zm = Z.mean()

    lonm = numpy.arctan2(ym, xm)
    latm = numpy.arctan2(zm, numpy.sqrt(xm**2+ym**2))

    return (latm, lonm)

def linear_interpolation_matrix(x_old, x_new):
    """Get transformation matrix for linear interpolation.

    This is denoted by W in Calisesi, Soebijanta and Van Oss (2005).

    Does note extrapolate; values outside the range are equal to the
    outermost values.

    :param x_old: Original 1-D grid
    :param x_new: New 1-D grid for interpolation
    :returns ndarray W: Interpolation transformation matrix.
    """
    
    W = numpy.vstack(
        [numpy.interp(x_new, x_old, numpy.eye(x_old.size)[i, :],
            left=numpy.nan, right=numpy.nan)    
            for i in range(x_old.size)])
    #return W
    return W.T


#    return numpy.vstack(
#        [scipy.interpolate.InterpolatedUnivariateSpline(
#            x_old, eye(x_old.size)[i, :])(x_new) 
#                for i in range(x_old.size)])

def regrid_ak(A, z_old, z_new, cut=False):
    """Regrid averaging kernel matrix.

    Actual regridding done in apply_W_A, following Calisesi, Soebijanta
    and Van Oss (2005).

    :param A: Original averaging kernel
    :param z_old: Original z-grid
    :param z_new: New z-grid
    :param bool cut: Cut off, i.e. flag, when any z in the new grid is
        outside the old grid.
    :returns: (New averaging kernel, W)
    """

    if cut:
        # make sure we take care of flagged data
        valid = ~(A<-10).all(0)

        # Not on the new one!  We must output the same size for A every
        # time.
#        new_outside = ((z_new > numpy.nanmax(z_old)) | 
#                       (z_new < numpy.nanmin(z_old)))
        #               ~numpy.isfinite(z_old))
        z_old_valid = numpy.isfinite(z_old)
#        if z_old[z_old_valid].max() < z_new.max():
#            raise ValueError("z_new not a subset of z_old!")
        #W = linear_interpolation_matrix(z_old[z_old_valid], z_new[~new_outside])
        #W = linear_interpolation_matrix(z_old[z_old_valid], z_new)
        # Keep full W (unflagged) because I want to put them all in a
        # single ndarray later
        # TODO: retry with masked arrays after the bugfixes
        W = linear_interpolation_matrix(z_old, z_new)
        #z_new_ok = (z_new > z_old.min()) & (z_new < z_old.max())
        z_new_ok = numpy.isfinite(W).all(1)
        A_new = numpy.zeros(shape=(z_new.shape[0], z_new.shape[0]))
        A_new.fill(numpy.nan)
        # The first one seems to not work... the second one does
        #A_new[z_new_ok, :][:, z_new_ok] = apply_W_A(
        A_new[numpy.ix_(z_new_ok, z_new_ok)] = apply_W_A(
                W[:, z_old_valid][z_new_ok, :],
                A[z_old_valid, :][:, z_old_valid])
#        A_new[outside, outside] = numpy.nan
        return (A_new, W)
    else:
        W = linear_interpolation_matrix(z_old, z_new)
        return (apply_W_A(W, A), W)

def regrid_matrix(A, z_old, z_new):
    """Regrid single matrix between grids.

    Do not use for averaging kernels!

    :param A:
    :param z_old:
    :param nd-array z_new: 1-D array
    """

    if z_old.shape[1] != A.shape[1]:
        raise ValueError("Shapes dont match")
    scipy.interpolate.interp1d
    A_new = numpy.zeros(shape=(z_old.shape[0], z_new.shape[0]))
    for i in range(z_old.shape[0]):
        for x in range(A.shape[0]):
            ip = scipy.interpolate.interp1d(z_old[i, :], A[x, :],
                bounds_error=False)
            A_new[i, :] = ip(z_new)
    return A_new

def apply_W_A(W, A):
    """Regrid averaging kernel matrix using W

    If interpolation matrix W is already calculated, apply to averaging
    kernel matrix here.

    This follows the methodology outlined by Calisesi, Soebijanta and Van
    Oss (2005).
    """

    Wstar = numpy.linalg.pinv(W)
    return W.dot(A).dot(Wstar)

def convert_ak_ap2vmr(AKx, aprf):
    """Convert averaging kernel from SFIT4 units to vmr

    :param AKx: Averaging kernel from SFIT4
    :param aprf: A-priori
    :returns: Averaging kernel in VMR units
    """

    # Source: e-mail Stephanie 2014-06-17

    return numpy.diag(1/aprf).dot(AKx).dot(numpy.diag(aprf))

def smooth_profile(xh, ak, xa):
    """Calculated smoothed profile.

    Calculate a smoothed profile following Rodgers and Connor (2003).

    :param xh: High-resolution profile
    :param ak: Low-resolution averaging kernel [VMR]
    :param xa: Low-resolution a priori profile
    """

    OK = (numpy.isfinite(xa) &
          numpy.isfinite(numpy.diag(ak)) &
          numpy.isfinite(xh))
    xs = numpy.zeros_like(xa)
    xs.fill(numpy.nan)
    xs[OK] = xa[OK] + ak[numpy.ix_(OK,OK)].dot(xh[OK] - xa[OK])
    return xs

def mad(x):
    """Median absolute deviation
    """

    return numpy.median(numpy.abs(x - numpy.median(x)))

def get_transformation_matrix(f, n):
    """Obtain standard matrix for the linear transformaton

    For a given linear function taking a vector valued argument, obtain
    the standard matrix for the linear transformation.

    See Lay (2003), Linear Algebra and its Transformations, 3rd edition,
    Theorem 10 (page 83).

    :param callable f: Function for which to get the transformation
        matrix.  This might be a function or a functools.partial object,
        for example.  This function should take as input a vector of
        length n and return as output a vector of length m (m>0).
        Of course, this function should be linear.
    :param int n: Size of transformation matrix needed
    :returns: (m x n) transformation matrix.
    """

    I = numpy.eye(n)
    return numpy.hstack([f(I[:, i:(i+1)]) for i in range(n)])
