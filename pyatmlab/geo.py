#!/usr/bin/env python

# coding: utf-8

"""Various small geo functions

"""

def shift_longitudes(lon, rng):
    """Shift longitudes to [-180, 180] or [0, 360]

    No longitudes should exist outside [-180, 360] in the input data.

    :param lon: Longitudes to be shifted
    :param rng: Either (-180, 180) or (0, 360)
    """

    if lon.min() < -180 or lon.max() > 360:
        raise ValueError(("Invalid longitude range ({}-{}) in "
            "input").format(lon.min(), lon.max()))
    if rng == (-180, 180):
        lon[lon>180] -= 360
    elif rng == (0, 360):
        lon[lon<0] += 360
    else:
        raise ValueError("Unknown range: {}".format(rng))

    return lon
