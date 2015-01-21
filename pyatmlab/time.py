"""Various time utilities
"""

import datetime

def mean_local_solar_time(utctime, lon):
    """Calculate mean local solar time.

    Calculates the mean local solar time for a specific longitude.
    This is not the true local solar time because it does not take into
    account the equation of time.  It is purely based on the hour angle of
    the Sun.  Do not use for astronomical calculations!

    :param time utctime: Time in UTC.  Should be a datetime.time object.
    :param float lon: Longitude in degrees.  Can be either in [-180, 180]
        or in [0, 360].
    :returns time: Time in mean local solar time.
    """

    hours_offset = lon/15
    dummy_datetime = datetime.datetime.combine(datetime.date.today(), utctime)
    new_dummy = dummy_datetime + datetime.timedelta(hours=hours_offset)
    return new_dummy.time()


def dt_to_doy_mlst(dt, lon):
    """From a datetime object, get day of year and MLST

    From a datetime object, calculate the day of year and the mean local
    solar time.

    :param datetime.datetime dt: Datetime object
    :param float lon: Longitude (needed for mean local solar time)
    :returns: (doy, mlst) where `doy` is a float between 0 and 366,
        and `mlst` is a float between 0 and 24.
    """

    utctime = dt.time()
    mlst = mean_local_solar_time(utctime, lon)
    frac_mlst = mlst.hour + mlst.minute/60 + mlst.second/3600

    dd = dt.date()
    doy = (dd - dd.replace(month=1, day=1)).days + 1
    return (doy, frac_mlst)
