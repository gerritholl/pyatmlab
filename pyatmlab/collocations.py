"""Contains classes and functionally to calculate collocations
"""
# pylint: disable-msg=E1101

import math
import numpy
import numpy.ma
import numpy.core.umath_tests
import logging
import datetime
import pathlib

import scipy
import scipy.stats
import scipy.interpolate

import matplotlib.pyplot
import matplotlib.backends.backend_agg
import matplotlib.figure
import pyproj
import mpl_toolkits.basemap

from . import dataset
from . import stats
from . import geo
from . import tools
from . import graphics
from . import math as pamath

from .constants import MB

class CollocatedDataset(dataset.HomemadeDataset):
    """Holds collocations.

    Attributes:

    primary
    secondary
    max_distance    Maximum distance in m
    max_interval    Maximum time interval in s.
    projection      projection to use in calculations
    stored_name     Filename to store collocs in.  String substition with
                    object dictionary (cd.__dict__), as well as start_date
                    and end_date.


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

    # NB: make sure basedir is unique for each collocated-dataset!
    subdir = ""
    stored_name = "collocations_{start_date!s}_{end_date!s}.npz"
    timefmt = "%Y%m%d%H%M%S" # used for start_date, end_date
    re = (r"collocations_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
                       r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})_"
                  r"(?P<year_end>\d{4})(?P<month_end>\d{2})(?P<day_end>\d{2})"
                  r"(?P<hour_end>\d{2})(?P<minute_end>\d{2})(?P<second_end>\d{2})"
                   "\.npz")

    max_distance = 0.0 # distance in m

    ## Properties

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


    ##

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

        if self.start_date is None:
            self.start_date = max(self.primary.start_date,
                self.secondary.start_date) - self.max_interval.astype(datetime.timedelta)

        if self.end_date is None:
            self.end_date = min(self.primary.end_date,
                self.secondary.end_date) + self.max_interval.astype(datetime.timedelta)


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
            (sec_start, sec_end) = self.primary.get_times_for_granule(gran_prim)
            for gran_sec in self.secondary.find_granules_sorted(
                sec_start-self.max_interval.astype(datetime.timedelta),
                sec_end+self.max_interval.astype(datetime.timedelta)):
                yield (gran_prim, gran_sec)

    def read_aggregated_pairs(self, start_date=None, end_date=None,
            maxsize=1e8):
        """Iterate and read aggregated co-time granule pairs

        Collect and read all secondary granules sharing the same primary.

        Will yield prematurely if size in bytes exceeds maxsize.
        """
        old_prim = old_sec = None
        prim = []
        sec = []
        logging.info("Searching granule pairs")
        primsize = secsize = size_ms = 0
        all_granule_pairs = list(self.find_granule_pairs(
            start_date, end_date))
        # since I'm reading many primaries at once, I may be returning
        # duplicate secondaries, which consequently results in duplicate
        # collocations.  Keep track of secondaries I've already read.
        secs_read = set()
        n_pairs = len(all_granule_pairs)
        logging.info("Found {:d} granule pairs".format(n_pairs))
        # FIXME: should concatenate both primary and secondary if both
        # have very short granules (such as one measurement per file).
        # Relevant for ACE vs Conway CH4+aux!
        for (i, (gran_prim, gran_sec)) in enumerate(all_granule_pairs):
#            logging.debug("Next pair: {} vs. {}".format(
#                gran_prim, gran_sec))
            if gran_prim != old_prim: # new primary
#                logging.debug("New primary: {!s}".format(gran_prim))
                try:
                    newprim = self.primary.read(gran_prim)
                except (ValueError, IOError) as msg:
                    logging.error("Could not read {}: {}".format(
                        gran_prim, msg.args[0]))
                    #continue
                else:
                    prim.append(newprim)
                    primsize += newprim.nbytes
                    old_prim = gran_prim
#                if prim is not None and sec != []: # if not first time, yield pair
#                    yield (prim, numpy.concatenate(sec))
#                    sec = []
#                    secsize = 0

            if gran_sec not in secs_read: # new secondary
#                logging.debug("New secondary: {!s}".format(gran_sec))
                try:
                    newsec = self.secondary.read(gran_sec)
                except (ValueError, IOError) as msg:
                    logging.error("Could not read {}: {}".format(
                        gran_sec, msg.args[0]))
                    continue
                else:
                    secs_read.add(gran_sec)
                    sec.append(newsec)
                    secsize += newsec.nbytes
                    old_sec = gran_sec

            if primsize + secsize > maxsize:
                # save memory, yield prematurely
                logging.info(("Read {:d}/{:d} granules.  "
                    "Yielding {:.2f} MB").format(
                        i, n_pairs, (primsize+secsize)/MB))
                logging.debug("Yielding due to size")
                yield (numpy.concatenate(prim), numpy.concatenate(sec))
                prim = []
                sec = []
                secs_read.clear()
                primsize = size_ms = secsize = 0

            totsize = primsize + secsize
            if totsize//1e7 > size_ms:
                logging.info(("Read {:d}/{:d} granules.  "
                    "Current size: {:.2f} MB").format(
                        i, n_pairs, (primsize+secsize)/MB))
                size_ms = totsize//1e7

        if len(prim) == 0 or len(sec) == 0:
            return

        yield (numpy.concatenate(prim), numpy.concatenate(sec))


    ########################################################
    ##
    ## Higher-level collocation routines
    ##
    ## Higher-level collocation routines take a time period, rather than
    ## arrays.  They take care of reading and storing.  Those are:
    ##
    ## Base layer:
    ##
    ##   - Read collocations stored on disk.
    ##   - Write collocations to disk (no return)
    ##   - Collocate period and return results (no reading/writing).
    ##     Best through generator to facilitate storing differently sized
    ##     chunks.
    ## 
    ## Above base layer:
    ##
    ##   - Get collocations for period, from disk if possible, store
    ##     results when appropriate
                

    ## Reading and storing resulting collocations

    def quicksave(self, prim, sec, f):
        """Store collocations to file

        To be implemented: more "robust" storing to netcdf.

        :param array prim: Primaries
        :param array sec: Secondaries
        :param str f: File to store to
        """
        if not isinstance(f, pathlib.Path):
            f = pathlib.Path(f)
        logging.info("Writing to {!s}".format(f))
        if not f.parent.exists():
            f.parent.mkdir(parents=True)
        with f.open('wb') as fp:
            numpy.savez_compressed(fp, prim=prim, sec=sec)

    def quickload(self, f):
        """Load collocations stored in quicksave format

        :param str f: File to load from
        :returns: (prim, sec)
        """

        logging.info("Reading from {!s}".format(f))
        D = numpy.load(str(f))
        return (D["prim"], D["sec"])


    def get_collocations(self, start_date=None, end_date=None,
            store_size_MB=100, fields=(None, None)):
        """Get collocations from start_date to end_date

        If available, read from disk.  If not, collocate on the fly and
        store results to disk.

        This is a generator.  It yields in chunks of 100 MB or so.  This
        is so subsequent processing doesn't hog memory needlessly.

        :param datetime start_date: First date to collocate
        :param datetime end_date: Last date to collocate
        :param store_size: Store results on disk after passing this size.
            Negative number means no storing.  Size in megabytes.
        :param (list, list) fields: If set, return only those fields.
            Should be a tuple of two lists.  If None, return all fields.
        """

#        all_p_col = []
#        all_s_col = []
#        all_p_ind = []

        # construct dtype
#        L = [[("i"+s, "i4"), ("lat"+s, "f8"), ("lon"+s, "f8"),
#            ("time"+s, "M8[s]")] for s in "12"]
        
#        dtp = L[0] + L[1]

#        for (gran_prim, gran_sec) in self.find_granule_pairs(
#                start_date, end_date):
#            logging.info("Collocating {0!s} with {1!s}".format(
#                ran_prim, gran_sec))
#            prim = self.primary.read(gran_prim)
#            sec = self.secondary.read(gran_sec)
        logging.info(("Getting collocations {!s} vs. {!s}, distance {:.1f} km, "
                      "interval {!s}, period {!s} - {!s}").format(
                        self.primary.name, self.secondary.name,
                        self.max_distance/1e3, self.max_interval,
                        start_date, end_date))

        # check whether we already have stored collocations.  Sort the
        # entire period in segments where we do or do not have collocation
        # information, to decide where we can read and where we need to
        # calculate collocations.  These are stored in two lists: segments
        # and collocs_sofar.
        segments = [] # time segments to still read
        collocs_stored_all = []
        # store 'grans' in list because I need to iterate over it twice
        # and index it later
        grans = list(self.find_granules_sorted(start_date, end_date))
        coltimes = [self.get_times_for_granule(gran) for gran in grans]
        if len(coltimes) == 0:
            segments.append((start_date, end_date))
            collocs_stored_all.append(None)
        else:
            # consider time before first segment found
            if coltimes[0][0] > start_date:
                segments.append((start_date, coltimes[0][0]))
                collocs_stored_all.append(None)
            # consider time between segments found
            #for i in range(len(coltimes)):
            for i in range(len(coltimes)):
                segments.append(coltimes[i])
                collocs_stored_all.append(self.quickload(grans[i]))
                if i<len(coltimes)-1 and coltimes[i][1] < coltimes[i+1][0]:
                    segments.append((coltimes[i][1], coltimes[i+1][0]))
                    collocs_stored_all.append(None)
            # consider time after last segment found
            if abs(coltimes[-1][1] - end_date) > datetime.timedelta(seconds=0.1):
                segments.append((coltimes[-1][1], end_date))
                collocs_stored_all.append(None)

        collocs_all = []
        for (segment, collocs_stored) in zip(segments, collocs_stored_all):
            if collocs_stored is None:
                begin, last = segment
                for collocs in self.collocate_period(*segment,
                        yield_size_MB=store_size_MB):
                    (prim, sec) = collocs
                    if prim.shape[0] != sec.shape[0]:
                        raise RuntimeError("Impossible output from"
                            "collocate_period")

                    # Make sure time intervals as stored in filenames
                    # cover full period.  Looking for the first and last
                    # collocation is insufficient as this covers only a
                    # subset of the time.
                    #
                    # Begin at either the start of the segment, or the
                    # last collocation from the previous batch.  End at
                    # the last collocation.  This leaves only the time
                    # from the last collocation until the end of the
                    # segment to be filled.


                    if prim.shape[0] > 0:
#                        first = min(prim["time"].min(), sec["time"].min()).astype(datetime.datetime)
                        last = max(prim["time"].max(), sec["time"].max()).astype(datetime.datetime)
                    else:
                        # store empty...
                        (first, last) = segment

                    f = self.find_granule_for_time(
                        start_date=begin.strftime(self.timefmt),
                        end_date=last.strftime(self.timefmt))
                    self.quicksave(prim, sec, f)
                    if fields == (None, None):
                        yield collocs
                    else:
                        yield (collocs[0][fields[0]], collocs[1][fields[1]])

                    begin = last 
                # end for all colloc batches
                if last < segment[1]: # store empty for remaining period
                    f = self.find_granule_for_time(
                        start_date=last.strftime(self.timefmt),
                        end_date=segment[1].strftime(self.timefmt))
                    self.quicksave(
                        numpy.empty(dtype=prim.dtype, shape=0),
                        numpy.empty(dtype=sec.dtype, shape=0),
                        f)
            else:
                if fields == (None, None):
                    yield collocs_stored
                else:
                    yield (collocs_stored[0][fields[0]],
                           collocs_stored[1][fields[1]])


    def collocate_period(self, start_date, end_date,
        yield_size_MB=100):
        """Collocate period and yield results.

        :param datetime start_date: Starting date
        :param datetime end_date: Ending date
        :param number yield_size: Size in MB after which to yield.
        """
        all_p_col = []
        all_s_col = []
        size_since_yield = 0
        yield_size = yield_size_MB*MB
        for (prim, sec) in self.read_aggregated_pairs(
                start_date, end_date):
            logging.info(("Collocating: "
                         "{:d} measurements for '{}', spanning {!s} - {!s}, "
                         "{:d} measurements for '{}', spanning {!s} - {!s}").format(
                    prim.shape[0], self.primary.name, min(prim["time"]),
                    max(prim["time"]),
                    sec.shape[0], self.secondary.name, min(sec["time"]),
                    max(sec["time"])))
            p_ind = self.collocate(prim, sec)
            logging.info("Found {0:d} collocations".format(p_ind.shape[0]))
            #all_p_ind.append(p_ind) # FIXME: get more useful than indices
            all_p_col.append(prim[p_ind[:, 0]])
            all_s_col.append(sec[p_ind[:, 1]])
            size = all_p_col[-1].nbytes + all_s_col[-1].nbytes
            size_since_yield += size
            logging.info(("So far {:d} collocations, total {:.0f} MB"
                ).format(sum(n.size for n in all_p_col), size_since_yield/MB))
            if size_since_yield > yield_size:
                yield (numpy.concatenate(all_p_col), numpy.concatenate(all_s_col))
                all_p_col = []
                all_s_col = []
                size_since_yield = 0
                
            #all_s_col.append(s_col)
        #return numpy.concatenate(all_p_ind)
        if len(all_p_col)>0:
            collocs = (numpy.concatenate(all_p_col), numpy.concatenate(all_s_col))
            yield collocs

    def collocate_all(self, distance=0, interval=numpy.timedelta64(1, 's')):
        """Collocate all available data.
        """
        raise NotImplementedError("Not implemented yet")


    ##################################################################
    ##
    ## Low-level collocation routines.  Likely not directly used.
    ##
    
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

        tmin = (min(t.min() for t in times_trunc)-self.bin_interval_time).astype(numpy.int64)
        tmax = (max(t.max() for t in times_trunc)+self.bin_interval_time).astype(numpy.int64)
        bins["time"] = numpy.linspace(
            tmin, tmax,
            (tmax-tmin) /
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
                    lon_is = numpy.unique(
                        numpy.mod(
                            numpy.arange(
                                lon_i - max_lon_range,
                                lon_i+max_lon_range),
                            bins["lon"].size).astype('uint64'))
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

    target      When smoothing profiles, smooth to this target.  It means
                we use the averaging kernel and z-grid therefrom.
    visualisation  Contains visualisation hints for drawing maps and so.

    """

    z_grid = None

    visualisation = dict(
        Eureka = dict(
            markersize=15,
            marker='o', markerfacecolor="white",
            markeredgecolor="red", markeredgewidth=2, zorder=4,
            label="Eureka", linestyle="None"),
        col0 = dict(
            marker='o',
            edgecolor="black",
            facecolor="white"),
        col1 = dict(
            marker='o',
            edgecolor="black",
            facecolor="red"),
        col2 = dict(
            marker='o',
            edgecolor="black",
            facecolor="blue"),
        col_line = dict(
            marker=None,
            linestyle="--",
            linewidth=1,
            color="black"))
    _target_vals = ["primary", "secondary"]

    _target = None
    @property
    def target(self):
        """See class docstring
        """
        #return self._target_vals[self._target] if self._target else None
        return self._target

    @target.setter
    def target(self, value):
        self._target = self._target_vals.index(value)

    def __init__(self, cd, p_col, s_col, **kwargs):
        self.cd = cd
        self.p_col = p_col
        self.s_col = s_col
        self.reset_filter()
        for (k, v) in kwargs.items():
            setattr(self, k, v)

    # limit etc.

    def reset_filter(self):
        """Completely reset the filter.  All collocations used.
        """

        self.mask = numpy.ones(shape=self.p_col.shape, dtype="bool")
        self.mask_label = "all"

    def filter(self, limit_str="UNDESCRIBED", dist=(0, numpy.inf),
            interval=(-numpy.inf, numpy.inf),
            prim_lims={}, sec_lims={}):
        """Set limits for further processing.

        This method sets a mask that characterises which collocations meet
        criteria and which ones do not.

        To be expanded.  Work in progress.

        :param str limit_str: Label for this filter.  Used in figures and
            so.
        :param dist: (min, max) distance [m]
        :param interval: (min, max) interval [s]
        :param dict prim_lims: Dictionary with keys corresponding to
            fields in the primary and values (min, max) thereof
        :param dict sec_lims: Like prim_lims but for secondary
        """
        mask = numpy.ones(shape=self.p_col.shape, dtype="bool")

        (_, _, dists) = self.cd.ellipsoid.inv(
            self.p_col["lon"], self.p_col["lat"],
            self.s_col["lon"], self.s_col["lat"])

        ints = self.s_col["time"] - self.p_col["time"]

        mask = mask & (dists > dist[0]) & (dists < dist[1])
        mask = mask & (ints > interval[0]) & (ints < interval[1])

        
        for (lims, db) in ((prim_lims, self.p_col),
                           (sec_lims, self.s_col)):
            for (field, (lo, hi)) in lims.items():
                mask = (mask    
                    & (db[field] > lo)
                    & (db[field] < hi))

        self.mask = mask
        self.mask_label = limit_str

        logging.debug("Filtering to {:d} elements".format(mask.sum()))

    # visualise

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
            self.p_col["lon"][self.mask], self.p_col["lat"][self.mask],
            self.s_col["lon"][self.mask], self.s_col["lat"][self.mask])
        interval = (self.p_col["time"][self.mask] -
                    self.s_col["time"][self.mask]).astype(
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
            "colloc_scatter_dist_time_{}_{}_{}.".format(
                self.cd.primary.__class__.__name__,
                self.cd.secondary.__class__.__name__,
                self.mask_label),
                data=numpy.vstack((dist_m/1e3, interval)).T)
        matplotlib.pyplot.close(f)

    def get_bare_collocation_map(self, lat, lon,
            sz=1500):
        """Get the bare map for plotting collocations.

        Don't actually plot any yet.

        Centered on (lat, lon).

        Returns tuple with (Figure, Axes, Basemap) objects.
        """

        f = matplotlib.pyplot.figure()
        ax = f.add_subplot(1, 1, 1)
        #sz = 1500e3 if station else 4000e3
        m = mpl_toolkits.basemap.Basemap(projection="laea",
            lat_0=lat, lon_0=lon, width=sz, height=sz,
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
        return (f, ax, m)

    def map_collocs(self):
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

        (f, ax, m) = self.get_bare_collocation_map(lat, lon,
            sz = 1500e3 if station else 4000e3)

#        f = matplotlib.pyplot.figure()
#        ax = f.add_subplot(1, 1, 1)
#        sz = 1500e3 if station else 4000e3
#        m = mpl_toolkits.basemap.Basemap(projection="laea",
#            lat_0=lat, lon_0=lon, width=sz, height=sz,
#            ax=ax, resolution="h")
#        m.drawcoastlines(linewidth=0.3, color="0.5")
#        m.etopo()
#        (sb_lon, sb_lat) = m(m.urcrnrx, m.llcrnry, inverse=True)
#        m.drawmapscale(sb_lon-7, sb_lat+2, sb_lon-7, sb_lat+2,
#            length=500, units="km")
#        m.drawparallels(numpy.arange(70., 89., 2.), zorder=2,
#            linewidth=0.3, labels=[1, 0, 1, 0])
#        m.drawmeridians(numpy.arange(-120., -41., 10.), latmax=88,
#            linewidth=0.3, zorder=2, labels=[0, 1, 0, 1])

        m.plot(lon, lat, latlon=True, 
            **self.visualisation["Eureka"])

        if station:
            m.scatter(other["lon"][self.mask], other["lat"][self.mask],
                50,
                latlon=True, zorder=3,
                label=other_ds.name,
                **self.visualisation["col0"])
            ax.text(0.5, 1.08,
                "Collocations Eureka-{:s}".format(other_ds.name),
                 horizontalalignment='center',
                 fontsize=20,
                 transform = ax.transAxes)
            #ax.set_title("Collocations Eureka-{:s}".format(other_ds.name))

            ax.legend(loc="upper left", numpoints=1)
        else:
            m.scatter(self.p_col["lon"][self.mask],
                self.p_col["lat"][self.mask], 50, 
                latlon=True, zorder=3,
                label=self.cd.primary.name,
                **self.visualisation["col1"])
            m.scatter(self.s_col["lon"][self.mask],
                self.s_col["lat"][self.mask], 50, 
                latlon=True, zorder=3,
                label=self.cd.secondary.name,
                **self.visualisation["col2"])
            for i in self.mask.nonzero()[0]:#range(self.p_col.size):
                m.plot([self.p_col["lon"][i], self.s_col["lon"][i]],
                       [self.p_col["lat"][i], self.s_col["lat"][i]],
                       latlon=True, zorder=2,
                       label="Collocated pair" if i==0 else None,
                       **self.visualisation["col_line"])
            ax.text(0.5, 1.08,
                "Collocations {:s}-{:s}".format(self.cd.primary.name,
                    self.cd.secondary.name), horizontalalignment="center",
                    fontsize=20, transform=ax.transAxes)
            ax.legend(loc="upper left", numpoints=1)
            
        graphics.print_or_show(f, False,
            "map_collocs_{}_{}_{}.".format(
                self.cd.primary.__class__.__name__,
                self.cd.secondary.__class__.__name__,
                self.mask_label))
        matplotlib.pyplot.close(f)

    def write_statistics(self):
        """Write a bunch of collocation statistics to the screen
        """
        print("Found {:d} collocations".format(self.p_col.shape[0]))

        print("With {:s} {:d}, {:s} {:d}".format(
            self.cd.primary.name,
            numpy.unique(self.p_col[["lat", "lon", "time"]]).shape[0],
            self.cd.secondary.name,
            numpy.unique(self.s_col[["lat", "lon", "time"]]).shape[0]))

        (_, _, dists) = self.cd.ellipsoid.inv(
            self.p_col["lon"][self.mask], self.p_col["lat"][self.mask],
            self.s_col["lon"][self.mask], self.s_col["lat"][self.mask])
        dists /= 1e3 # m -> km

        means = {}
        for c in ("p", "s"):
            means[c] = numpy.rad2deg(
                pamath.average_position_sphere(
                    numpy.deg2rad(getattr(self, c+"_col")["lat"][self.mask]),
                    numpy.deg2rad(getattr(self, c+"_col")["lon"][self.mask])))

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


    def interpolate_profiles(self, z_grid):
        """Interpolate profiles on a common z-grid

        Currently hardcoded for CH4, linear.

        :param z_grid: Altitude grid to compare on.  Both products will be
            interpolated onto this grid (may be a no-op for one).
        :returns: (p_ch4, s_ch4) interpolated profiles
        """

        # do not apply mask here, rather loop only through unmasked
        # elements further down.

        p_ch4 = self.p_col[self.cd.primary.aliases["CH4_profile"]]

        s_ch4 = self.s_col[self.cd.secondary.aliases["CH4_profile"]]

        # interpolate profiles onto a common grid
        val_ind = self.mask.nonzero()[0]
        p_ch4_int = numpy.zeros(shape=(val_ind.size, z_grid.size),
            dtype=p_ch4.dtype)
        s_ch4_int = numpy.zeros_like(p_ch4_int)

        k = 0
        for i in self.mask.nonzero()[0]:#range(self.p_col.size):
            p_z_i = self.cd.primary.get_z(self.p_col[i])
            s_z_i = self.cd.secondary.get_z(self.s_col[i])
            # workaround https://github.com/numpy/numpy/issues/2972
            p_valid = (p_ch4[i] > 0)
            s_valid = (s_ch4[i] > 0)
            p_ch4_i = (p_ch4[i].data 
                if isinstance(p_ch4, numpy.ma.MaskedArray)
                else p_ch4[i])[p_valid]
            s_ch4_i = (s_ch4[i].data
                if isinstance(s_ch4, numpy.ma.MaskedArray)
                else s_ch4[i])[s_valid]
            if p_valid.shape == p_z_i.shape:
                p_z_i = p_z_i[p_valid]
            if s_valid.shape == s_z_i.shape:
                s_z_i = s_z_i[s_valid]
            #
            if not p_valid.any() or not s_valid.any():
                p_ch4_int[k, :] = numpy.nan
                s_ch4_int[k, :] = numpy.nan
                k += 1
                continue

            p_interp = scipy.interpolate.interp1d(p_z_i, p_ch4_i,
                bounds_error=False)
            s_interp = scipy.interpolate.interp1d(s_z_i, s_ch4_i,
                bounds_error=False)

            p_ch4_int[k, :] = p_interp(z_grid)
            s_ch4_int[k, :] = s_interp(z_grid)
            k += 1

        return (p_ch4_int, s_ch4_int)
#                
#        self.s_col

    def extend_common_grid(self):
        """Extend both profiles to common grid.

        Sets self.z_grid accordingly.

        If target does not have unique z-grid, choose average and
        interpolate all onto that.
        """

        targ = (self.p_col, self.s_col)[self.target]
        targ_obj = (self.cd.primary, self.cd.secondary)[self.target]

        z_all = numpy.array([targ_obj.get_z(t) for t in targ])
#        if not (targ["z"] == targ["z"][0, :]).all():
#            raise ValueError("Inconsistent z-grid for target")
#        z = targ["z"][0, :]
        z = numpy.array(z_all, dtype="f8").mean(0)

        (p_ch4_int, s_ch4_int) = self.interpolate_profiles(z)
#        xa = targ["CH4_apriori"]
#
#        targ_ch4_int = (p_ch4_int, s_ch4_int)[self.target]
#        xh = (p_ch4_int, s_ch4_int)[1-self.target]
#        xh[numpy.isnan(xh)] = xa[numpy.isnan(xh)]

        self.z_grid = z
        return (p_ch4_int, s_ch4_int)


        # fill up 'nans' using a priori
#        for i in self.mask.nonzero()[0]:#range(self.p_col.size):
#            

    def smooth(self):
        """Smooth one profile with the others AK
        
        Normally the profile with the largest information content should
        be smoothed according to the AK of the other.  This is not
        determined automatically but determined by self.target which is
        set upon object creation.

        Source is equation 4 in:
        
        Rodgers and Connor (2003): Intercomparison of
        remote sounding instruments.  In: Journal of Geophysical Research,
        Vol 108, No. D3, 4116, doi:10.1029/2002JD002299

        Also calls self.extend_common_grid thus setting self.z_grid.

        Returns primary and secondary profiles, one smoothed, the other
        unchanged
        """

        targ = (self.p_col, self.s_col)[self.target][self.mask]
        #highres = (self.prim, self.sec)[1-self.target]
        (p_ch4_int, s_ch4_int) = self.extend_common_grid()

        xa = targ["CH4_apriori"]
        xh = (p_ch4_int, s_ch4_int)[1-self.target]

#        xh = s_int
#        xh[xnan(xh)] = xa[isnan(xh)]
        A = targ["CH4_ak"].swapaxes(1, 2)

        # correct A according to e-mail Stephanie 2014-06-17
        #
        # WARNING ERROR FIXME!  THIS IS HARDCODED FOR TARGET BEING EQUAL
        # TO PEARL!

        Avmr = numpy.rollaxis(
            numpy.dstack(
                [numpy.diag(1/targ["CH4_apriori"][i, :]).dot(
                    targ["CH4_ak"][i, :, :]).dot(
                    numpy.diag(targ["CH4_apriori"][i, :]))
                for i in range(targ.shape[0])]), 2, 0)

        #xs = xa + A.dot(xh-xa)
        xb = xh - xa
        xs = numpy.vstack(
            [xa[n, :] + A[n, :, :].dot(xb[n, :]) for n in range(Avmr.shape[0])])
#        xs = xa + numpy.core.umath_tests.matrix_multiply(A, xb[..., numpy.newaxis]).squeeze()

        if self.target == 0:
            return (p_ch4_int, xs)
        elif self.target == 1:
            return (xs, s_ch4_int)
        else:
            raise RuntimeError("Impossible!")


    def _compare_profiles(self, p_ch4_int, s_ch4_int,
            percs=(5, 25, 50, 75, 95)):
        """Helper for compare_profiles_{raw,smoothed}
        """

        D = {}
        D["diff"] = s_ch4_int - p_ch4_int
        D["rmsd"] = (D["diff"] ** 2)**(0.5)
        D["ratio"] = s_ch4_int / p_ch4_int
        D["prim"] = p_ch4_int
        D["sec"] = s_ch4_int

        #return numpy.array([
        return {k:
            numpy.array([scipy.stats.scoreatpercentile(D[k][:, i], percs)
                for i in range(D[k].shape[1])])
            for k in D.keys()}


    def compare_profiles_raw(self, z_grid,
            percs=(5, 25, 50, 75, 95)):
        """Return some statistics comparing profiles.

        Currently hardcoded for CH4.

        Arguments as for interpolate_profiles.

        Returns percentiles (5, 25, 50, 75, 95) for difference, 
        root mean square difference, ratio, and original values, as a
        dictionary.
        """

        (p_ch4_int, s_ch4_int) = self.interpolate_profiles(z_grid)

        return self._compare_profiles(p_ch4_int, s_ch4_int,
            percs=percs)

    def compare_profiles_smooth(self, _,
            percs=(5, 25, 50, 75, 95)):
        #
        # interpolate onto retrieval grid for dataset with less vertical
        # resolution
        #
        # use averaging kernel and a priori of dataset with less vertical
        # resolution

        (p_ch4_int, s_ch4_int) = self.smooth()

        return self._compare_profiles(p_ch4_int, s_ch4_int, percs=percs)


    def visualise_profile_comparison(self, z_grid, filters=None):
        """Visualise profile comparisons.

        Currently hardcoded for CH4.

        Arguments as for compare_profiles and interpolate_profiles, plus
        an additional argument `filters` that will be passed on to
        self.filter plus a color keyword arg, colour keyword arg must be a
        tuple for (raw, smooth).
        """

        if filters is None:
            filters = []

        p_locs = (5, 25, 50, 75, 95)
        p_styles = (':', '--', '-', '--', ':')
        p_widths = (0.5, 1, 2, 1, 0.5)

        colours = {}
        percs = {}
        lims = {}
        xlabels = dict(
            diff = "Delta CH4 [ppv]",
            rmsd = "RMSD CH4 [ppv]",
            ratio = "CH4 ratio [1]",
            prim = "Primary CH4 [ppv]",
            sec = "Secondary CH4 [ppv]")
        xlims = dict(
            diff = (-3e-7, 3e-7),
            rmsd = (0e-6, 5e-7),
            ratio = (0.8, 1.2),
            prim = (0, 2e-6),
            sec = (0, 2e-6))
        self.reset_filter()

        # see how smoothed or raw compare

        z_grids = {}
        percs[self.mask_label + "_smooth"] = self.compare_profiles_smooth(
                    z_grid, p_locs)
        z_grids["smooth"] = self.z_grid
        percs[self.mask_label + "_raw"] = self.compare_profiles_raw(
                    z_grid, p_locs)
        z_grids["raw"] = z_grid

        colours[self.mask_label + "_raw"] = "blue"
        colours[self.mask_label + "_smooth"] = "black"
        for fd in filters:
            lab = fd["limit_str"]
            (colours[lab + "_raw"], colours[lab + "_smooth"]) = fd.pop("color")
            self.filter(**fd)
            percs[fd["limit_str"] + "_raw"] = self.compare_profiles_raw(z_grid, p_locs)
            percs[fd["limit_str"] + "_smooth"] = self.compare_profiles_smooth(z_grid, p_locs)
#        percs = self.compare_profiles(z_grid, p_locs)
        #for (i, v) in enumerate("diff diff^2 ratio".split()):
        for quantity in percs[self.mask_label + "_raw"].keys():
            f = matplotlib.pyplot.figure()
            a = f.add_subplot(1, 1, 1)
            data = dict(smooth=[], raw=[])
            for filt in percs.keys():
                ff = "smooth" if "smooth" in filt else "raw"
                #if "smooth" in filt: z_grid = z_grids["smooth"]
                #elif "raw" in filt: z_grid = z_grids["raw"]
                #else: raise RuntimeError("Something wrong")
                for k in range(len(p_locs)):
                    a.plot(percs[filt][quantity][:, k], z_grids[ff],
                           color=colours[filt],
                           linestyle=p_styles[k], linewidth=p_widths[k],
                           label=(filt if k==2 else None))
                    data[ff].append(percs[filt][quantity][:, k])
            # end for percentiles
            # end for filters
#            a.plot(percs[1], z_grid, label="p diff^2", color="red")
#            a.plot(percs[2], z_grid, label="p ratio", color="black")
            a.legend()
            a.set_xlabel(xlabels[quantity])
            if quantity in xlims:
                a.set_xlim(xlims[quantity])
            a.set_ylabel("Elevation [m]")
            a.set_title("Percentiles 5/25/50/75/95 for" +
                "CH4 {}, {} vs. {}".format(quantity,
                    self.cd.primary.name, self.cd.secondary.name))
            a.grid(which="major")
            a.set_ylim([5e3, 60e3])
            ### FIXME: write data for both, will need to be in two
            ### files...
            graphics.print_or_show(f, False,
                "compare_profiles_ch4_{}_{}_{}_{}_{}_{}.".format(
                    self.cd.primary.__class__.__name__,
                    self.cd.primary.name.replace(" ", "_"),
                    self.cd.secondary.__class__.__name__,
                    self.cd.secondary.name.replace(" ", "_"),
                    quantity,
                    "multi"),
                    data=numpy.vstack((z_grids["raw"],)+tuple(data["raw"])).T)
            matplotlib.pyplot.close(f)
        # end for quantities
        self.reset_filter()

        # some specialised plots
##         iqr = {}
##         for quantity in ("prim", "sec", "diff"):
##             iqr[quantity] = (percs["all"][quantity][:, 3] -
##                              percs["all"][quantity][:, 1])
##         f = matplotlib.pyplot.figure()
##         a = f.add_subplot(1, 1, 1)
##         a.plot(iqr["prim"], z_grid, label=self.cd.primary.name)
##         a.plot(iqr["sec"], z_grid, label=self.cd.secondary.name)
##         a.plot(iqr["diff"], z_grid, label="difference")
##         a.set_xlabel("CH4 [ppv]")
##         a.set_ylabel("Altitude [m]")
##         a.set_title("CH4 IQR")
##         a.legend()
##         a.grid(which="major")
##         graphics.print_or_show(f, False,
##                 "iqr_{}_{}_{}_{}.".format(
##                     self.cd.primary.__class__.__name__,
##                     self.cd.primary.name.replace(" ", "_"),
##                     self.cd.secondary.__class__.__name__,
##                     self.cd.secondary.name.replace(" ", "_")),
##                 data=numpy.vstack(
##                     (z_grid, iqr["prim"], iqr["sec"], iqr["diff"])).T)
##         matplotlib.pyplot.close(f)

def find_collocation_duplicates(p_col, s_col):
    dt = numpy.dtype([("A", "f8"), ("B", "f8"), ("C", "M8[s]"), ("D",
    "f8"), ("E", "f8"), ("F", "f8")])


    merged = numpy.ascontiguousarray(
        numpy.vstack(
            (p_col["lat"], p_col["lon"], 
             p_col["time"].astype("M8[s]").astype("i8"),
             s_col["lat"], s_col["lon"], 
             s_col["time"].astype( "M8[s]").astype("i8"))
                 ).T).view(dt)[:, 0]
    uni = numpy.unique(merged)

    return uni
