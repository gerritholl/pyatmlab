"""Contains classes and functionally to calculate collocations
"""
# pylint: disable-msg=E1101

import math
import numpy


import pyproj

from . import dataset
from . import stats

class CollocatedDataset(dataset.HomemadeDataset):
    """Holds collocations.

    Attributes:

    primary
    secondary
    max_distance    Maximum distance in m
    max_interval    Maximum time interval in s.

    The following attributes may be changed at your own risk.  Changing
    should not affect results, but may affect performance.  Optimise based
    on application.

    bin_interval_time
    bin_interval_lat
    bin_interval_lon
    """

    primary = None
    secondary = None
    projection = "WGS84"
    ellipsoid = None

    max_distance = 0.0 # distance in m

    _max_interval = numpy.timedelta64(0, 's')
    @property
    def max_interval(self):
        """Maximum interval time.

        Can be set as a number, interpreted in seconds, or as a
        timedelta64 object.
        """
        return self._max_interval

    @max_interval.setter
    def max_interval(self, value):
        self._max_interval = numpy.timedelta64(value, 's')

    bin_interval_time = numpy.timedelta64(1, 'D')
    bin_interval_lat = 1.0 # degree
    bin_interval_lon = 1.0 # degree

    def __init__(self, primary, secondary, **kwargs):
        self.primary = primary
        self.secondary = secondary
        if "projection" in kwargs:
            self.projection = kwargs.pop("projection")
        self.ellipsoid = pyproj.Geod(ellps=self.projection)
        self.max_interval = 0

        super().__init__(**kwargs)

    def collocate_all(self, distance=0, interval=numpy.timedelta64(1, 's')):
        """Collocate all available data.
        """
        raise NotImplementedError("Not implemented yet")

    def collocate(self, arr1, arr2):
        """Collocate arrays in time, late, lon.

        Each of `arr1` and `arr2` must have ["time"] (datetime64),
        ["lat"] (float), and ["lon"] (float).

        Note that this is a low-level function, and you will likely want
        to call a higher level method such as collocate_all.
        """

        if self.max_interval == 0 or self.max_distance == 0:
            return None

        # all binning should be by truncation, not rounding; i.e.
        # 2010-01-01 23:00:00 is binned on 2010-01-01.

        # first bin both by time, which is a special case because
        # numpy.digitize, on which pyatmlab.stats.bin_nd relies, does not
        # support it; so we need to truncate both time series to a common
        # format, then use ints for the binning

        # truncate time series to resultion of self.bin_interval_time
        newtype = "<M8[{}]".format(self.bin_interval_time.dtype.str[-2])
        times_trunc = [arr["time"].astype(newtype) for arr in (arr1, arr2)]
        times_int = [time.astype(numpy.int64) for time in times_trunc]
        time_bins = numpy.arange(
            min(t.min() for t in times_trunc),
            max(t.max() for t in times_trunc),
            self.bin_interval_time)

        lats = [arr1["lat"], arr2["lat"]]
        lons = [arr1["lon"], arr2["lon"]]

        lat_bins = numpy.arange(
            numpy.floor(min(lat.min() for lat in lats)),
            numpy.ceil(max(lat.max() for lat in lats)+1),
            self.bin_interval_lat)

        # Note: this will be too large if longitudes cross (anti)meridian,
        # but that's no big deal
        lon_bins = numpy.arange(
            numpy.floor(min(lon.min() for lon in lons)),
            numpy.ceil(max(lon.max() for lon in lons)+1),
            self.bin_interval_lon)

        binned = [stats.bin_nd(
            [times_int[i], lats[i], lons[i]],
            [time_bins.astype(numpy.int64), lat_bins, lon_bins])
            for i in (0, 1)]

        bin_no = numpy.array([numpy.array([b.size for b in bb.flat]).reshape(bb.shape)
                for bb in binned])

        # number of neighbouring bins to look into
        binrange_time = math.ceil(self.max_interval/self.bin_interval_time)
        cell_height = 2 * math.pi * self.ellipsoid.b / 360
        cell_width = (2 * math.pi * numpy.cos(numpy.deg2rad(lat_bins)) *
                      self.ellipsoid.b / 360)

        binrange_lat = numpy.ceil(self.max_distance/
            (self.bin_interval_lat*cell_height))
        binrange_lon = numpy.ceil(self.max_distance/
            (self.bin_interval_lon*cell_width))

        all_p_met = []
        all_s_met = []

        for time_i in range(len(time_bins)):
            # range of secondary time bins
            t_s_min = max(0, time_i - binrange_time)
            t_s_max = min(time_bins.size-1, time_i + binrange_time + 1)

            # potentially skip lat & lon loops
            if (bin_no[0, time_i, :, :].max() == 0 or
                    bin_no[1, t_s_min:t_s_max, :, :].max() == 0):
                continue

            for lat_i in range(len(lat_bins)):
                # range of secondary lat bins
                lat_s_min = max(0, lat_i - binrange_lat)
                lat_s_max = min(lat_bins.size-1, lat_i + binrange_lat + 1)

                # potentially skip lon loop
                if (bin_no[0, time_i, lat_i, :].max() == 0 or
                        bin_no[1, t_s_min:t_s_max, lat_s_min:lat_s_max, :].max() == 0):
                    continue

                max_lon_range = max(binrange_lon[lat_s_min:lat_s_max])
                for lon_i in range(len(lon_bins)):
                    # range of secondary lon bins

                    # for width of lons consider polemost relevant
                    # latitude bin
                    lon_is = numpy.mod(numpy.arange(lon_i - max_lon_range,
                            lon_i+max_lon_range), lon_bins.size).astype('uint64')
                    #lon_s_min = max(0, lon_i - max_lon_range)
                    #lon_s_max = min(lon_bins.size-1, lon_i + max_lon_range + 1)

                    if (bin_no[0, time_i, lat_i, lon_i].max() == 0 or
                            bin_no[1, t_s_min:t_s_max,
                            lat_s_min:lat_s_max,
                            lon_is].sum() == 0):
                        continue

                    primary = arr1[binned[0][time_i, lat_i, lon_i]]
                    secondary = arr2[numpy.concatenate(binned[1][
                        t_s_min:t_s_max,
                        lat_s_min:lat_s_max,
                        lon_is].ravel().tolist())]

                    (p_met, s_met) = self._collocate_bucket(primary, secondary)

                    all_p_met.append(p_met)
                    all_s_met.append(s_met)

        return (numpy.concatenate(all_p_met), numpy.concatenate(all_s_met))

    def _collocate_bucket(self, primary, secondary):
        """Collocate a single bucket.  Internal function used by
        collocate.

        Expects two buckets containing measurements that will be
        brute-forced against each other.
        """

        if primary.size == 0 or secondary.size == 0:
            return (numpy.empty(shape=(0,), dtype=numpy.int64),
                    numpy.empty(shape=(0,), dtype=numpy.int64))

        # find pairs meeting time criterion
        intervals = (primary[:, numpy.newaxis]["time"] -
                     secondary[numpy.newaxis, :]["time"])
        time_met = abs(intervals) < self.max_interval
        (time_met_i1, time_met_i2) = time_met.nonzero()

        # find pairs meeting distance criterion
        p_time_met = primary[time_met_i1]
        s_time_met = secondary[time_met_i2]

        (_, _, dist) = self.ellipsoid.inv(
            p_time_met["lon"], p_time_met["lat"],
            s_time_met["lon"], s_time_met["lat"],
            radians=False)

        dist_met = dist < self.max_distance
        p_met = p_time_met[dist_met]
        s_met = s_time_met[dist_met]

        return p_met, s_met


