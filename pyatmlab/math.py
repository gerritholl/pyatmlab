#!/usr/bin/env python
# coding: utf-8

"""Various small mathematical functions

"""

import numpy
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
    return (y_avg * dz)

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

    :param x_old: Original 1-D grid
    :param x_new: New 1-D grid for interpolation
    :returns ndarray W: Interpolation transformation matrix.
    """
    
    return numpy.vstack(
        [numpy.interp(x_new, x_old, numpy.eye(x_old.size)[i, :])    
            for i in range(x_old.size)])


#    return numpy.vstack(
#        [scipy.interpolate.InterpolatedUnivariateSpline(
#            x_old, eye(x_old.size)[i, :])(x_new) 
#                for i in range(x_old.size)])

def regrid_ak(A, z_old, z_new):
    """Regrid averaging kernel matrix.

    This follows the methodology outlined by Calisesi, Soebijanta and Van
    Oss (2005).

    :param A: Original averaging kernel
    :param z_old: Original z-grid
    :param z_new: New z-grid
    :returns: New averaging kernel
    """

    W = linear_interpolation_matrix(z_old, z_new)
    Wstar = numpy.pinv(W)
    return W.dot(A).dot(Wstar)


