#!/usr/bin/env python
# coding: utf-8

"""Various small mathematical functions

"""

import numpy
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

#del inputs
