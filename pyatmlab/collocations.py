"""Contains classes and functionally to calculate collocations
"""
# pylint: disable-msg=E1101

import math
import numpy
import numpy.ma
import logging

import matplotlib.pyplot
import pyproj
import mpl_toolkits.basemap

from . import dataset
from . import stats
from . import geo
from . import tools
from . import graphics
from . import math as pamath

class CollocatedDataset(dataset.HomemadeDataset):
    """Holds collocations.

    Attributes:

    primary
    secondary
    max_distance    Maximum distance in m
    max_interval    Maximum time interval in s.
    projection      projection to use in calculations

    The following attributes may be changed at your own risk.  Changing
    should not affect results, but may affect performance.  Optimise based
    on application.  Subject to change.

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
        """Initialize with Dataset objects
        """
        self.primary = primary
        self.secondary = secondary
        if "projection" in kwargs:
            self.projection = kwargs.pop("projection")
        self.ellipsoid = pyproj.Geod(ellps=self.projection)
        self.max_interval = 0

        super().__init__(**kwargs)

    def find_granule_pairs(self, start_date=None, end_date=None):
        """Iterate through all (prim, sec) co-time granule pairs

        Can optionally pass in start_date and end_date, that will be passed
        on to the primary find_granules.
        """

        if start_date is None:
            start_date = max([self.primary.start_date,
                self.secondary.start_date])

        if end_date is None:
            end_date = min([self.primary.end_date,
                self.secondary.end_date])

        for gran_prim in self.primary.find_granules_sorted(start_date, end_date):
            for gran_sec in self.secondary.find_granules_sorted(
                *self.primary.get_times_for_granule(gran_prim)):
                yield (gran_prim, gran_sec)

    def read_aggregated_pairs(self, start_date=None, end_date=None):
        """Iterate and read aggregated co-time granule pairs

        Collect and read all secondary granules sharing the same primary.
        """
        old_prim = old_sec = None
        prim = None
        sec = []
        logging.info("Searching granule pairs")
        for (gran_prim, gran_sec) in self.find_granule_pairs(
            start_date, end_date):
#            logging.debug("Next pair: {} vs. {}".format(
#                gran_prim, gran_sec))
            if gran_prim != old_prim: # new primary
#                logging.debug("New primary: {!s}".format(gran_prim))
                if prim is not None: # if not first time, yield pair
#                    logging.debug("Yielding!")
                    yield (prim, numpy.concatenate(sec))
                prim = self.primary.read(gran_prim)
                old_prim = gran_prim

            if gran_sec != old_sec:
#                logging.debug("New secondary: {!s}".format(gran_sec))
                try:
                    sec.append(self.secondary.read(gran_sec))
                except ValueError as msg:
                    logging.error("Could not read {}: {}".format(
                        gran_sec, msg.args[0]))
                    continue
                old_sec = gran_sec

        yield (prim, numpy.concatenate(sec))

                

    def collocate_period(self, start_date=None, end_date=None):
        """Collocate period from start_date to end_date

        And return results
        """
        all_p_col = []
        all_s_col = []
#        all_p_ind = []

        # construct dtype
#        L = [[("i"+s, "i4"), ("lat"+s, "f8"), ("lon"+s, "f8"),
#            ("time"+s, "M8[s]")] for s in "12"]
        
#        dtp = L[0] + L[1]

        # FIXME: aggregate multiple granules before collocating, in
        # particular for cases with only one measurement per file
#        for (gran_prim, gran_sec) in self.find_granule_pairs(
#                start_date, end_date):
#            logging.info("Collocating {0!s} with {1!s}".format(
#                ran_prim, gran_sec))
#            prim = self.primary.read(gran_prim)
#            sec = self.secondary.read(gran_sec)
        logging.info(("Searching collocations {!s} vs. {!s}, distance {:.1f} km, "
                      "interval {!s}, period {!s} - {!s}").format(
                        self.primary.name, self.secondary.name,
                        self.max_distance/1e3, self.max_interval,
                        start_date, end_date))
        for (prim, sec) in self.read_aggregated_pairs(
                start_date, end_date):
            logging.info(("{} {:d} measurements spanning {!s} - {!s}, "
                          "{} {:d} measurements spanning {!s} - {!s}").format(
                    self.primary.name, prim.shape[0], min(prim["time"]),
                    max(prim["time"]),
                    self.secondary.name, sec.shape[0], min(sec["time"]),
                    max(sec["time"])))
            #(p_col, s_col) = self.collocate(prim, sec)
            logging.info("Collocating...")
            p_ind = self.collocate(prim, sec)
            logging.info("Found {0:d} collocations".format(p_ind.shape[0]))
            #all_p_ind.append(p_ind) # FIXME: get more useful than indices
            all_p_col.append(prim[p_ind[:, 0]])
            all_s_col.append(sec[p_ind[:, 1]])
            #all_s_col.append(s_col)
        #return numpy.concatenate(all_p_ind)
        return (numpy.concatenate(all_p_col), numpy.concatenate(all_s_col))

    def collocate_all(self, distance=0, interval=numpy.timedelta64(1, 's')):
        """Collocate all available data.
        """
        raise NotImplementedError("Not implemented yet")

    @tools.validator
    def collocate(self, arr1:geo.valid_geo, arr2:geo.valid_geo):
        """Collocate arrays in time, late, lon.

        Each of `arr1` and `arr2` must have ["time"] (datetime64),
        ["lat"] (float), and ["lon"] (float).

        Note that this is a low-level function, and you will likely want
        to call a higher level method such as collocate_all.

        :returns: N x 2 array with primary and secondary indices
        """

        # This algorithm can be optimised in a number of different ways:
        #
        # - Use quadtrees on a spherical grid instead of guessing grid
        # sizes
        # - Process only time that is in common
        # - For memory, loop through time and collocate bit by bit

        if self.max_interval == 0 or self.max_distance == 0:
            return numpy.empty(shape=(0, 2), dtype=numpy.uint64)

        # all binning should be by truncation, not rounding; i.e.
        # 2010-01-01 23:00:00 is binned on 2010-01-01.

        # first bin both by time, which is a special case because
        # numpy.digitize, on which pyatmlab.stats.bin_nd relies, does not
        # support it; so we need to truncate both time series to a common
        # format, then use ints for the binning

        # FIXME: this can be optimized by doing only further processing
        # for common time interval

        if (arr1["time"].max() + self.max_interval < arr2["time"].min() or
            arr2["time"].max() + self.max_interval < arr1["time"].min()):
            return numpy.empty(shape=(0, 2), dtype=numpy.uint64)

        # will finally want to return indices rather than copying actual
        # data, keep track of those
        ind1 = numpy.arange(arr1.shape[0])
        ind2 = numpy.arange(arr2.shape[0])

        # truncate time series to resolution of self.bin_interval_time
        newtype = "<M8[{}]".format(self.bin_interval_time.dtype.str[-2])
        times_trunc = [arr["time"].astype(newtype) for arr in (arr1, arr2)]
        times_int = [time.astype(numpy.int64) for time in times_trunc]
#        time_bins = numpy.arange(
#            min(t.min() for t in times_trunc),
#            max(t.max() for t in times_trunc),
#            self.bin_interval_time)

        lats = [arr1["lat"], arr2["lat"]]
        lons = [arr1["lon"], arr2["lon"]]

        bins = {}
        bin_intervals = dict(lat=self.bin_interval_lat,
            lon=self.bin_interval_lon, time=self.bin_interval_time)
        for k in ("lat", "lon"):
            kmax = max(v.max() for v in (arr1[k], arr2[k]))
            kmin = min(v.min() for v in (arr1[k], arr2[k]))
            bins[k] = numpy.linspace(
                kmin-1, kmax+1,
                max((((kmax+1)-(kmin-1))/bin_intervals[k], 2)))

        tmin = min(t.min() for t in times_trunc).astype(numpy.int64)
        tmax = max(t.max() for t in times_trunc).astype(numpy.int64)
        bins["time"] = numpy.linspace(
            tmin-1, tmax+1, ((tmax+1)-(tmin-1)) /
                bin_intervals["time"].astype(numpy.int64)).astype(newtype)
                    
#        lat_bins = numpy.linspace(
#            numpy.floor(min(lat.min() for lat in lats)-1),
#            numpy.ceil(max(lat.max() for lat in lats)+1),
#        lat_bins = numpy.arange(
#            numpy.floor(min(lat.min() for lat in lats)),
#            numpy.ceil(max(lat.max() for lat in lats)+1),
#            self.bin_interval_lat)

        # Note: this will be too large if longitudes cross (anti)meridian
#        lon_bins = numpy.arange(
#            numpy.floor(min(lon.min() for lon in lons)),
#            numpy.ceil(max(lon.max() for lon in lons)+1),
#            self.bin_interval_lon)

        # Perform the actual binning for primary, secondary.  This part
        # could be optimised a lot, ideally using quadtrees or at least by
        # guessing a more appropriate grid size.
        binned = [stats.bin_nd(
            [times_int[i], lats[i], lons[i]],
            [bins["time"].astype(numpy.int64), bins["lat"], bins["lon"]])
            for i in (0, 1)]

        # count the number of entries per bin
        bin_no = numpy.array([numpy.array([b.size for b in bb.flat]).reshape(bb.shape)
                for bb in binned])

        # some intermediate checking to verify all got binned
        if bin_no[0, ...].sum() != arr1.size or bin_no[1, ...].sum() != arr2.size:
            raise RuntimeError("Some data remained unbinned!")

        # number of neighbouring bins to look into
        binrange_time = math.ceil(self.max_interval/self.bin_interval_time)
        cell_height = 2 * math.pi * self.ellipsoid.b / 360
        cell_width = (2 * math.pi * numpy.cos(numpy.deg2rad(bins["lat"])) *
                      self.ellipsoid.b / 360)

        binrange_lat = numpy.ceil(self.max_distance/
            (self.bin_interval_lat*cell_height))
        binrange_lon = numpy.ceil(self.max_distance/
            (self.bin_interval_lon*cell_width))

        #all_p_met = []
        #all_s_met = []
        all_ind_met = []

        for time_i in range(len(bins["time"])):
            # range of secondary time bins
            t_s_min = max(0, time_i - binrange_time)
            t_s_max = min(bins["time"].size, time_i + binrange_time + 1)

            # potentially skip lat & lon loops
            if (bin_no[0, time_i, :, :].max() == 0 or
                    bin_no[1, t_s_min:t_s_max, :, :].max() == 0):
                continue

            for lat_i in range(len(bins["lat"])):
                # range of secondary lat bins
                lat_s_min = max(0, lat_i - binrange_lat)
                lat_s_max = min(bins["lat"].size, lat_i + binrange_lat + 1)

                # potentially skip lon loop
                if (bin_no[0, time_i, lat_i, :].max() == 0 or
                        bin_no[1, t_s_min:t_s_max, lat_s_min:lat_s_max, :].max() == 0):
                    continue

                max_lon_range = max(binrange_lon[lat_s_min:lat_s_max])
                for lon_i in range(len(bins["lon"])):
                    # range of secondary lon bins

                    # for width of lons consider polemost relevant
                    # latitude bin
                    lon_is = numpy.mod(numpy.arange(lon_i - max_lon_range,
                            lon_i+max_lon_range), bins["lon"].size).astype('uint64')
                    #lon_s_min = max(0, lon_i - max_lon_range)
                    #lon_s_max = min(lon_bins.size-1, lon_i + max_lon_range + 1)

                    if (bin_no[0, time_i, lat_i, lon_i].max() == 0 or
                            bin_no[1, t_s_min:t_s_max,
                            lat_s_min:lat_s_max,
                            lon_is].sum() == 0):
                        continue

                    selec1 = binned[0][time_i, lat_i, lon_i]
                    selec2 = numpy.ma.concatenate(binned[1][
                        t_s_min:t_s_max,
                        lat_s_min:lat_s_max,
                        lon_is].ravel().tolist())

                    primary = arr1[selec1]
                    secondary = arr2[selec2]

                    prim_ind = ind1[selec1]
                    sec_ind = ind2[selec2]

                    ind_met = self._collocate_bucket(
                        primary, secondary, prim_ind, sec_ind)

                    all_ind_met.append(ind_met)
                    #all_p_met.append(p_met)
                    #all_s_met.append(s_met)

        if len(all_ind_met) > 0:
            return numpy.concatenate(all_ind_met)
        else:
            return numpy.empty(shape=(0, 2), dtype=numpy.uint64)
            
#        return tuple((numpy.ma.concatenate 
#            if isinstance(x, numpy.ma.MaskedArray)
#            else numpy.concatenate)(x)
#                for x in (all_p_met, all_s_met))

    def _collocate_bucket(self, primary, secondary, prim_ind, sec_ind):
        """Collocate a single bucket.  Internal function used by
        collocate.

        Expects two buckets containing measurements that will be
        brute-forced against each other, as well as corresponding indices.

        Returns an N x 2 ndarray with indices selected from prim_ind,
        sec_ind.
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

        p_ind_time_met = prim_ind[time_met_i1]
        s_ind_time_met = sec_ind[time_met_i2]

        (_, _, dist) = self.ellipsoid.inv(
            p_time_met["lon"], p_time_met["lat"],
            s_time_met["lon"], s_time_met["lat"],
            radians=False)

        dist_met = dist < self.max_distance
        #p_met = p_time_met[dist_met]
        p_ind_met = p_ind_time_met[dist_met]
        #s_met = s_time_met[dist_met]
        s_ind_met = s_ind_time_met[dist_met]

        #return p_met, s_met
        return numpy.array([p_ind_met, s_ind_met], dtype=numpy.uint64).T



# visualisations

class CollocationDescriber:
    """Collects various functions to describe a set of collocations

    Initialise with a CollocatedDataset object as well as
    sets of measurements already collocated through
    CollocatedDataset.collocate
    """

    def __init__(self, cd, p_col, s_col):
        self.cd = cd
        self.p_col = p_col
        self.s_col = s_col

    def plot_scatter_dist_int(self, time_unit="h",
            plot_name=None):
        """Scatter plot of distance [km] and interval.

        Will write to plotdir.

        :param str time_unit:  Single argument time_unit, defaults to
            "h" = hour, can be any valid code for numpy.timedelta64.
        :param str plot_name:  Output filename for plot.  Defaults to
            colloc_scatter_dist_time_PRIMARY_SECONDARY.
        """

        (_, _, dist_m) = self.cd.ellipsoid.inv(
            self.p_col["lon"], self.p_col["lat"],
            self.s_col["lon"], self.s_col["lat"])
        interval = (self.p_col["time"] - self.s_col["time"]).astype(
            "m8[{}]".format(time_unit)).astype("i")
        max_i = self.cd.max_interval.astype(
            "m8[{}]".format(time_unit)).astype("i")

        f = matplotlib.pyplot.figure()
        ax = f.add_subplot(1, 1, 1)
        ax.plot(dist_m/1e3, interval, ".")
        ax.set_xlabel("Distance [km]")
        ax.set_ylabel("Interval [{}]".format(time_unit))
        ax.set_title("Collocations {} {}".format(
            self.cd.primary.name, self.cd.secondary.name))
        ax.set_xlim(0, self.cd.max_distance/1e3)
        ax.set_ylim(-max_i, +max_i)

        graphics.print_or_show(f, False,
            "colloc_scatter_dist_time_{}_{}.".format(
                self.cd.primary.__class__.__name__,
                self.cd.secondary.__class__.__name__))

    def map_collocs(self, **kwargs):
        """Display collocations on a map.

        Currently hardcoded to be around Eureka, Nunavut.
        """

        # check if either is stationary (like a ground station such as
        # Eureka)
        if ((self.p_col["lat"][1:] == self.p_col["lat"][:-1]).all() and
            (self.p_col["lon"][1:] == self.p_col["lon"][:-1]).all()):
            lon = self.p_col["lon"][0]
            lat = self.p_col["lat"][0]
            station = True
            other = self.s_col
            other_ds = self.cd.secondary
        elif ((self.s_col["lat"][1:] == self.s_col["lat"][:-1]).all() and
              (self.s_col["lon"][1:] == self.s_col["lon"][:-1]).all()):
            lon = self.s_col["lon"][0]
            lat = self.s_col["lat"][0]
            station = True
            other = self.p_col
            other_ds = self.cd.primary
        else:
            station = False
            other = None
            other_ds = None
            (lat, lon) = (80, -86) # arbitrary? ;-)

        f = matplotlib.pyplot.figure()
        ax = f.add_subplot(1, 1, 1)
        m = mpl_toolkits.basemap.Basemap(projection="laea",
            lat_0=lat, lon_0=lon, width=1500e3, height=1500e3,
            ax=ax, resolution="h")
        m.drawcoastlines(linewidth=0.3, color="0.5")
        m.etopo()
        (sb_lon, sb_lat) = m(m.urcrnrx, m.llcrnry, inverse=True)
        m.drawmapscale(sb_lon-7, sb_lat+2, sb_lon-7, sb_lat+2,
            length=500, units="km")
        m.drawparallels(numpy.arange(70., 89., 2.), zorder=2,
            linewidth=0.3, labels=[1, 0, 1, 0])
        m.drawmeridians(numpy.arange(-120., -41., 10.), latmax=88,
            linewidth=0.3, zorder=2, labels=[0, 1, 0, 1])

        m.plot(lon, lat, latlon=True, markersize=15,
            marker='o', markerfacecolor="white",
            markeredgecolor="red", markeredgewidth=2, zorder=4,
            label="Eureka", linestyle="None")
        if station:
            m.scatter(other["lon"], other["lat"], 50, marker='o',
                edgecolor="black", #edgewidth=4,
                facecolor="white", latlon=True, zorder=3,
                label=other_ds.name)
            ax.text(0.5, 1.08,
                "Collocations Eureka-{:s}".format(other_ds.name),
                 horizontalalignment='center',
                 fontsize=20,
                 transform = ax.transAxes)
            #ax.set_title("Collocations Eureka-{:s}".format(other_ds.name))

            ax.legend(loc="upper left", numpoints=1)
        else:
            m.scatter(self.p_col["lon"], self.p_col["lat"], 50, marker='o',
                edgecolor="black", facecolor="red", latlon=True, zorder=3,
                label=self.cd.primary.name)
            m.scatter(self.s_col["lon"], self.s_col["lat"], 50, marker='o',
                edgecolor="black", facecolor="blue", latlon=True, zorder=3,
                label=self.cd.secondary.name)
            for i in range(self.p_col.size):
                m.plot([self.p_col["lon"][i], self.s_col["lon"][i]],
                       [self.p_col["lat"][i], self.s_col["lat"][i]],
                       latlon=True,
                       marker=None, linestyle="--", linewidth=1,
                       color="black", zorder=2)
            ax.text(0.5, 1.08,
                "Collocations {:s}-{:s}".format(self.cd.primary.name,
                    self.cd.secondary.name, horizontalalignment="center",
                    fontsize=20, transform=ax.transAxes))
            ax.legend(loc="upper left", numpoints=1)
            
        graphics.print_or_show(f, False,
            "map_collocs_{}_{}.".format(
                self.cd.primary.__class__.__name__,
                self.cd.secondary.__class__.__name__))

    def write_statistics(self):
        """Write a bunch of collocation statistics to the screen
        """
        print("Found {:d} collocations".format(self.p_col.shape[0]))

        (_, _, dists) = self.cd.ellipsoid.inv(
            self.p_col["lon"], self.p_col["lat"],
            self.s_col["lon"], self.s_col["lat"])
        dists /= 1e3 # m -> km

        means = {}
        for c in ("p", "s"):
            means[c] = numpy.rad2deg(
                pamath.average_position_sphere(
                    numpy.deg2rad(getattr(self, c+"_col")["lat"]),
                    numpy.deg2rad(getattr(self, c+"_col")["lon"])))

        (mean_dir, _, mean_dist) = self.cd.ellipsoid.inv(
            means["p"][1], means["p"][0],
            means["s"][1], means["s"][0])
        mean_dist /= 1e3 # m -> km

        for (i, label) in ((0, "first"), (-1, "last")):
            print(("{} collocation: ({!s}, {!s}) at "
               "({:.2f}, {:.2f}), ({:.2f}, {:.2f}) ({:.2f} km)").format(
                   label.capitalize(),
                   self.p_col["time"][i], self.s_col["time"][i],
                   self.p_col["lat"][i], self.p_col["lon"][i],
                   self.s_col["lat"][i], self.s_col["lon"][i],
                   dists[i]))

        print(("Min/mean/median/max distance: " +
            " / ".join(("{:.2f} km",)*4)).format(
            dists.min(), dists.mean(), numpy.median(dists), dists.max()))

        print("Mean positions: ({:.2f}, {:.2f}), ({:.2f}, {:.2f})".format(
            means["p"][0], means["p"][1], means["s"][0], means["s"][1]))

        print(("Distance, direction between mean positions: "
               "{:.2f} km, {:.0f}°").format(mean_dist, mean_dir))
