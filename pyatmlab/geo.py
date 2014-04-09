#!/usr/bin/env python

# coding: utf-8

"""Various small geo functions

"""

def shift_longitudes(lon, rng):
    """Shift longitudes to [-180, 180] or [0, 360]

    No longitudes should exist outside [-180, 360] in the input data.

    :param lon: Longitudes to be shifted
    :param rng: Either (-180, 180) or (0, 360)
    :returns: New longitudes
    """

    if lon.ptp() >= 360:
        raise ValueError(("Invalid longitude range ({}-{}) in "
            "input").format(lon.min(), lon.max()))
    if rng == (-180, 180):
        lon[lon>180] -= (360 * (lon.max()//360 + 1))
        lon[lon<-180] += (360 * abs(lon.min()//360 + 1))
    elif rng == (0, 360):
        raise NotImplementedError("Not implemented")
        #lon[lon<0] += 360
    else:
        raise ValueError("Unknown range: {}".format(rng))

    return lon

def valid_geo(arr):
    """Check if geo-data are valid.

    Should be contains in arr["lat"], arr["lon"].

    Valid is -90 <= lat <= 90, -180 <= lon <= 180
    """

    return (arr["lat"].min() >= -90 and
            arr["lat"].max() <= 90 and
            arr["lon"].min() >= -180 and
            arr["lon"].min() <= 180)
