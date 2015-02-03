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
import functools

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
from . import physics
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
                except (dataset.InvalidFileError) as msg:
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
                except (dataset.InvalidFileError, 
                        dataset.InvalidDataError) as msg:
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
                # If primary granules are much longer than secondary
                # granules, we might run over maxsize with a single
                # primary granule.  Make sure to reset it so it's reread.
                # (Setting to object() ensures == gives False)
                old_prim = object()
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

        :param datetime start_date: First date to collocate.  If not
            given, use last of (self.primary.start_date,
            self.secondary.start_date).
        :param datetime end_date: Last date to collocate.  If not given,
            use earliest of (self.primary.end_date,
            self.secondary.end_date).
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
        if start_date is None:
            start_date = max(self.primary.start_date,
                self.secondary.start_date)

        if end_date is None:
            end_date = min(self.primary.end_date,
                self.secondary.end_date)

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
                # select only those within the requested date-range
                inrange = ((collocs_stored[0]["time"] > start_date) &
                           (collocs_stored[1]["time"] > start_date) & 
                           (collocs_stored[0]["time"] < end_date) &
                           (collocs_stored[1]["time"] < end_date))
                collocs_stored = (
                    collocs_stored[0][inrange],
                    collocs_stored[1][inrange])
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

class CollocationDescriber:
    """Collects various functions to describe a set of collocations

    Initialise with a CollocatedDataset object as well as
    sets of measurements already collocated through
    CollocatedDataset.collocate

    target      When smoothing profiles, smooth to this target.  It means
                we use the averaging kernel and z-grid therefrom.  The
                'target' is thus the low resolution profile.  Smoothing in
                the sense of Rodgers and Connor (2003).
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

    figname_compare_profiles = ("compare_profiles_ch4"
        "_{self.cd.primary.__class__.__name__!s}"
        "_{self.cd.secondary.__class__.__name__}"
        "_targ{self.target}"
        "_{allmask}"
        "_{quantity}.")

    figname_compare_pc = ("compare_partial_columns_ch4"
        "_{self.cd.primary.__class__.__name__}"
        "_{self.cd.secondary.__class__.__name__}"
        "_targ{self.target}.")

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
            interval=
                (-numpy.timedelta64(numpy.iinfo('int64').max, 's'),
                 +numpy.timedelta64(numpy.iinfo('int64').max, 's')),
            prim_lims={}, sec_lims={},
            mask=None):
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
        :param ndarray mask: Alternately, set mask explicitly.  This may
            be useful for more complicated cases.  Note that the rest of
            the criteria are still added after.
        """
        if mask is None:
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

    ## Visualise actual collocations

    def plot_scatter_dist_int(self, time_unit="h",
            plot_name=None):
        """Scatter plot of distance [km] and interval.

        Write a figure to the plot directory and write coordinates to a
        file in the plotdata directory (for use with pgfplots, for
        example).

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

    ## Routines to write statistics on collocations

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

class ProfileCollocationDescriber(CollocationDescriber):
    """Routines specific for collocations between profile quantities

    This class contains profile-related methods.  Some may (still) be
    hardcoded to CH₄ but should be appropriate for any profile that can be
    treated similarly.
    """

    z_range = None

    ## Calculation methods

    def interpolate_profiles(self, z_grid,
            prim_fields=("CH4_profile",),
            sec_fields=("CH4_profile",),
            prim_filters=None,
            sec_filters=None):
        """Interpolate all profiles on a common z-grid

        (Currently hardcoded for CH4, linear.)

        :param z_grid: Altitude grid to compare on.  Both products will be
            interpolated onto this grid (may be a no-op for one).
        :param tuple prim_fields: Fields from primary to interpolate
        :param tuple sec_fields: Fields from secondary to interpolate
        :param dict prim_filters: Filters to apply to primary fields.
            The keys of the dict correspond to prim_fields.  The values
            are tuples with (forward, backward) which should be a pair of
            functions for the forward and backward transformation, for
            example, (log, exp)
        :param dict sec_filters: Filters to apply to secondary fields.
            Contents as for prim_filters.
        :returns: (p_ch4, s_ch4) interpolated profiles
        """

        # do not apply mask here, rather loop only through unmasked
        # elements further down.

        prims = [self.cd.primary.aliases.get(p, p) for p in prim_fields]
        secs = [self.cd.secondary.aliases.get(s, s) for s in sec_fields]

        p = {"orig": self.p_col}

        s = {"orig": self.s_col}

        # interpolate profiles onto a common grid
        val_ind = self.mask.nonzero()[0]
        p["int"] = numpy.zeros(shape=(val_ind.size,),
            dtype=[(x[0], x[1], z_grid.size) for x in p["orig"].dtype.descr
                        if x[0] in prims])
        p["W"] = numpy.zeros(shape=(val_ind.size,),
            dtype=[(x[0], "f4", (z_grid.size, x[2][0]))
                    for x in p["orig"].dtype.descr
                        if x[0] in prims])
        s["int"] = numpy.zeros_like(p["int"], 
            dtype=[(x[0], x[1], z_grid.size) for x in s["orig"].dtype.descr
                        if x[0] in secs])
        s["W"] = numpy.zeros(shape=(val_ind.size,),
            dtype=[(x[0], "f4", (z_grid.size, x[2][0]))
                    for x in s["orig"].dtype.descr
                        if x[0] in secs])

        p["filters"] = prim_filters or {}
        s["filters"] = sec_filters or {}
        p["obj"] = self.cd.primary
        s["obj"] = self.cd.secondary

        k = 0 # loops over output array
        for i in val_ind: # loops over input array
            p["z_i"] = self.cd.primary.get_z(self.p_col[i])
            s["z_i"] = self.cd.secondary.get_z(self.s_col[i])
            # masked arrays buggy https://github.com/numpy/numpy/issues/2972
            p["valid"] = numpy.isfinite(p["z_i"])
            s["valid"] = numpy.isfinite(s["z_i"])
            #p["z_i"] = p["z_i"][p["valid"]]
            #s["z_i"] = s["z_i"][s["valid"]]
            #
            if not p["valid"].any() or not s["valid"].any():
                p["int"][k].fill(numpy.nan)
                p["W"][k].fill(numpy.nan)
                s["int"][k].fill(numpy.nan)
                s["W"][k].fill(numpy.nan)
                k += 1
                continue

            for arr in (p, s):
                for f in arr["int"].dtype.names:
                    y = arr["orig"][f][i, :].copy()
                    # FIXME: to determine validity, don't simply say
                    # 'y>0', rather consider actual flagged values, should
                    # be documented... this will cause a bug when y can
                    # really be smaller than 0!
                    if arr["orig"][f].shape[1] == arr["valid"].size:
                        z = arr["z_i"].copy()
                        val = arr["valid"] & numpy.isfinite(y)
                        val[numpy.isfinite(y)] = (y[numpy.isfinite(y)]>=0)
                    else:
                        z = arr["obj"].get_z_for(arr["orig"][i], f).copy()
                        val = numpy.isfinite(z) & numpy.isfinite(y)
                        val[numpy.isfinite(y)] = (y[numpy.isfinite(y)]>=0)
                    #y = y[val]
                    y[~val] = numpy.nan
                    z[~val] = numpy.nan
                    if not val.any():
                        arr["int"][f][k, :] = numpy.nan
                        arr["W"][f][k, :, :] = W
                        continue
                    if f in arr["filters"]:
                        y[val] = arr["filters"][f][0](y[val])
                    # Need to store interpolation matrices for the purpose
                    # of error propagation calculations.
                    W = pamath.linear_interpolation_matrix(z, z_grid)
#                    interp = scipy.interpolate.interp1d(
#                        z[val], y[val], bounds_error=False)
#                    yp = interp(z_grid)
                    yp = W.dot(y) #  W@y
                    if f in arr["filters"]:
                        yp = arr["filters"][f][1](yp)
                    arr["int"][f][k, :] = yp
                    arr["W"][f][k, :, :] = W

            k += 1

        return (p["int"], s["int"], p["W"], s["W"])

    def regrid_profiles(self,
            *args, **kwargs):
        """Interpolate both profiles to grid of target.

        Take the z-grid from the target and interpolate all profiles from
        primary and secondary to this grid.  If each profile for the
        target has the same z-grid, the target interpolation will be a
        no-op.  If the z-grid for the target is not unique, take the mean
        and grid all profiles to this one.

        Set self.z_grid to the z_grid that is finally used, and return
        interpolated profiles (primary, secondary).

        This is just a thin shell around interpolate_profiles.
        """

        targ = (self.p_col, self.s_col)[self.target][self.mask]
        targ_obj = (self.cd.primary, self.cd.secondary)[self.target]

        z_all = numpy.array([targ_obj.get_z(t) for t in targ])
        z = numpy.nanmean(numpy.array(z_all, dtype="f8"), 0)

        self.z_grid = z
        return self.interpolate_profiles(z, *args, **kwargs)


    # This fails because it uses state information that changes between
    # calls, and this state information is not taken into account.
    #@functools.lru_cache(maxsize=10)
    _p_smooth = _s_smooth = None
    def smooth(self, field_specie="CH4_profile",
                     field_T="T", 
                     field_p="p",
                     field_S="S_CH4_profile",
                     reload=False):
        """Smooth one profile with the others AK
        
        Normally the profile with the largest information content should
        be smoothed according to the AK of the other.  This is not
        determined automatically but determined by self.target which
        should set upon object creation.

        Source is equation 4 in:
        
        Rodgers and Connor (2003): Intercomparison of
        remote sounding instruments.  In: Journal of Geophysical Research,
        Vol 108, No. D3, 4116, doi:10.1029/2002JD002299

        Also calls self.regrid_profiles thus setting self.z_grid.

        Returns primary and secondary profiles, one smoothed, the other
        unchanged.

        TODO:
            - Verify the regridding of the a-priori (xa).  It seems that
              z_xa is sometimes severely negative?  Is this still true?

            - Verify that the regridding of averaging kernels is now
              correct for profiles where lowest level of low-res
              AK is above the common z-grid to which we are trying
              to interpolate, effectively extrapolating, which is
              ill-conditioned.

            - For calculating partial columns, any regridding needs to be
              also applied to (at least) T and P!  Better set p_smooth,
              s_smooth as ndarrays containing several types...

            - Should turn around AK axes in reading routine, not here!
              And/or only if really needed!

            - Clean up the code... should first decide on a z-grid and
              then regrid everything (raw, smoothed, p, T, etc.) onto
              this?

            - Consider errors!  See Von Clarmann (2014), Vigouroux et al. (2007),
              Calisesi et al. (2005)
        """

        # Cache results "by hand"; it appears lru_cache doesn't work well
        # with methods depending on the state of a (mutable) object
        if not reload and (self._p_smooth is not None) and (self._s_smooth is not None):
            return (self._p_smooth, self._s_smooth)

        targ = (self.p_col, self.s_col)[self.target][self.mask]
        nontarg = (self.p_col, self.s_col)[1-self.target][self.mask]
        targobj = (self.cd.primary, self.cd.secondary)[self.target]
        nontargobj = (self.cd.primary, self.cd.secondary)[1-self.target]

        # Regrid primary and secondary CH4 profiles so that they share the
        # same z_grid, that will be set to self.z_grid.
        #
        # NB: if a profile's z_grid does not extend to the full range of
        # self.z_grid, any "extrapolated" values are set to numpy.nan by
        # regrid_profiles (through scipy.interpolate.interp1d)
        #
        # NB: Don't interpolate S here.  I need the error covariance
        # in the original as I need to combine the interpolation matrices
        # W_1 and W_2 (see Vigouroux et al. (2007) prior to applying them
        # to S.
        flds = (field_specie, field_T, field_p)
        filts = dict(p=(numpy.log, numpy.exp))
        (p_int, s_int, p_W, s_W) = self.regrid_profiles(
            prim_fields=flds,
            sec_fields=flds,
            prim_filters=filts,
            sec_filters=filts)
        p_ch4_int = p_int[self.cd.primary.aliases[field_specie]]
        s_ch4_int = s_int[self.cd.secondary.aliases[field_specie]]

        # Both p_ch4_int and s_ch4_int are
        # already regridded to both be on self.z_grid (see above)

        xh = (p_ch4_int, s_ch4_int)[1-self.target]
        z_xh = self.z_grid

        # Get "pure" a priori and averaging kernels.  May or may not be on
        # the same grid as target objects.
        (xa, z_xa, ak, z_ak) = self._get_xa_ak(targ, targobj)

        # FIXME: Should rather do this in reading routine...?
        # but: needs a priori that may be available only "elsewhere"
        # FIXME: Separate conversion and swapping
        if targobj.A_needs_converting:
           # correct ak according to e-mail Stephanie 2014-06-17
            ak = ak.swapaxes(1, 2)
            ak = numpy.rollaxis(
                numpy.dstack(
                    [pamath.convert_ak_ap2vmr(
                        ak[i, :, :], xa[i, :])
                    for i in range(xa.shape[0])]), 2, 0)

        ## Make sure everything is on the same grid:
        #
        # - Actual profiles
        #       -> regrid_profiles(), called above, guarantees those are
        #       on the same grid, set in self.z_grid.  This should be from
        #       the low-resolution profile (i.e. the target).
        #       Note that z_xh == self.z_grid
        # - Averaging kernels for low-resolution profile
        #       -> currently on z_ak
        # - A priori for low-resolution profile, on z_xa
        #       -> currently on z_xa
        #
        # It is useless if any of those extends above any of the others,
        # so we choose the lowest maximum z and cut off everything to
        # that.

        p_sa = s_sa = None # FIXME
        (xa, z_xa, ak, z_ak, xh, z_xh, p_int, s_int, p_sa, s_sa) = \
            self._limit_to_shared_range(xa, z_xa, ak, z_ak, xh, z_xh, 
                p_int, s_int, p_sa, s_sa)

        # regrid xa(z_xa) and ak(z_ak) onto the grid of z_xh
        (xa, z_xa, W_xa, ak, z_ak, W_ak) = self._regrid_xa_ak(xa, z_xa, ak, z_ak, z_xh)


        # OK :).  Now everything should be on the same z-grid!

        # where high-res profile is outside its z-range, set to a-priori
        # of low-res.  For example, PEARL may go down only to 19.5 km, but
        # the low-res xa might go down to 5 km.  Then set [5, 19.5] of
        # high-res equal to a priori of other.

        xh[numpy.isnan(xh)] = xa[numpy.isnan(xh)]

        # This is where the smoothing is actually performed!
        xs = numpy.vstack(
            [pamath.smooth_profile(xh[n, :], ak[n, :, :], xa[n, :])
                for n in range(ak.shape[0])])

        # Calculate variance
        # S_1 should be low-res, S_2 should be highres
        # `targ` is low-res

        W_1 = (p_W, s_W)[self.target][targobj.aliases[field_specie]]
        W_2 = (p_W, s_W)[1-self.target][nontargobj.aliases[field_specie]]
        S_1 = targ[targobj.aliases[field_S]]
        S_2 = nontarg[nontargobj.aliases[field_S]]
        # eliminate flagged values; should not be propagated in matrix
        # multiplication, so setting to 0 is appropriate
        S_1[S_1<-10]=0
        S_2[S_2<-10]=0
        S_d = [self._calc_error_propagation(S_1[i, :, :],
               W_1[i, :, :], ak[i, :, :], S_2[i, :, :], W_2[i, :, :])
                   for i in range(p_int.size)]
        S_d = numpy.rollaxis(numpy.dstack(S_d), 2, 0)

        # remove invalid data
        OK = numpy.isfinite(xs).any(1)
        # I don't really know what to do with still incomplete averaging
        # kernels (i.e. flagged levels), as I want all profiles on the
        # same grid.  Remove those for now.
        invalid = (ak<-10).any(1).any(1)
        OK = OK & (~invalid)

        xs = xs[OK, :]
        p_int = p_int[OK]
        s_int = s_int[OK]

        if not all(numpy.array_equal(self.z_grid, z) for z in
            (z_xa, z_ak, z_xh)):
            raise RuntimeError("z_grids should be equal by now!")

        self.z_smooth = z_xa
        logging.info("Z-grid after smoothing: {!s}".format(z_xa))
        (p_int, s_int)[1-self.target][nontargobj.aliases[field_specie]] = xs
#        if self.target == 0:
#            #p["CH4_profile"] = p_ch4_int
#            s["CH4_profile"] = xs
#            #self._p_smooth = p_ch4_int
#            #self._s_smooth = xs
#            #return (p_ch4_int, xs)
#        elif self.target == 1:
#            p["CH4_profile"] = xs
#            #s["CH4_profile"] = s_ch4_int
#            #self._p_smooth = xs
#            #self._s_smooth = s_ch4_int
#            #return (xs, s_ch4_int)
#        else:
#            raise RuntimeError("Impossible!")

        return (p_int, s_int, S_d)

    def partial_columns(self, smoothed=True, reload=False):
        """Calculate partial columns.

        """
        if ("parcol_CH4" in self.p_col.dtype.names and
            "parcol_CH4" in self.s_col.dtype.names and
            not reload):
            return (self.p_col["parcol_CH4"],
                    self.s_col["parcol_CH4"],
                    self.z_range)
        # FIXME: consider filtering here!

        shared_range = (max(self.cd.primary.range[0],
                            self.cd.secondary.range[0]),
                        min(self.cd.primary.range[1],
                            self.cd.secondary.range[1]))

        if smoothed:
            (p, s, S_d) = self.smooth()
            z = self.z_smooth
        else:
            raise NotImplementedError()

        # levels in 'z' within shared_range
        valid_range = (z > shared_range[0]) & (z < shared_range[1])

        z_valid = z[valid_range]
        p_valid_vmr = p[self.cd.primary.aliases["CH4_profile"]].T[valid_range, :]
        s_valid_vmr = s[self.cd.secondary.aliases["CH4_profile"]].T[valid_range, :]

        p_valid_nd = physics.vmr2nd(p_valid_vmr,
            p[self.cd.primary.aliases.get("T", "T")].T[valid_range, :],
            p[self.cd.primary.aliases.get("p", "p")].T[valid_range, :])
        s_valid_nd = physics.vmr2nd(p_valid_vmr,
            s[self.cd.secondary.aliases.get("T", "T")].T[valid_range, :],
            s[self.cd.secondary.aliases.get("p", "p")].T[valid_range, :])

        p_parcol = pamath.integrate_with_height(
            z_valid, p_valid_nd)
        s_parcol = pamath.integrate_with_height(
            z_valid, s_valid_nd)

        self.p_col = numpy.lib.recfunctions.append_fields(self.p_col,
            names=["parcol_CH4"],
            data=[p_parcol],
            usemask=False)
        self.s_col = numpy.lib.recfunctions.append_fields(self.s_col,
            names=["parcol_CH4"],
            data=[s_parcol],
            usemask=False)
        self.z_range = (z[valid_range].min(), z[valid_range].max())
        return (p_parcol, s_parcol, (z[valid_range].min(), z[valid_range].max()))

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
            numpy.array([scipy.stats.scoreatpercentile(
                D[k][numpy.isfinite(D[k][:, i]), i], percs)
                for i in range(D[k].shape[1])])
            for k in D.keys()}

    def compare_profiles_raw(self, z_grid,
            percs=(5, 25, 50, 75, 95)):
        """Return some statistics comparing profiles.

        Currently hardcoded for CH4.

        Returns percentiles (5, 25, 50, 75, 95) for difference, 
        root mean square difference, ratio, and original values, as a
        dictionary.
        """

        (p_int, s_int, p_W, s_W) = self.interpolate_profiles(z_grid,
            prim_fields=("CH4_profile",),
            sec_fields=("CH4_profile",))

        return self._compare_profiles(
            p_int[self.cd.primary.aliases["CH4_profile"]],
            s_int[self.cd.secondary.aliases["CH4_profile"]], percs=percs)

    def compare_profiles_smooth(self, _,
            percs=(5, 25, 50, 75, 95)):
        #
        # interpolate onto retrieval grid for dataset with less vertical
        # resolution
        #
        # use averaging kernel and a priori of dataset with less vertical
        # resolution

        (p_int, s_int, S_d) = self.smooth()

        return self._compare_profiles(
            p_int[self.cd.primary.aliases["CH4_profile"]],
            s_int[self.cd.secondary.aliases["CH4_profile"]], percs=percs)

    ## Helper methods for calculation methods

    def _get_xa_ak(self, targ, targobj):
        """Helper for smooth(...)
        """

        z_xa = z_ak = None

        ## Look for a priori and z-grid (will need to regrid later)
        if "ap" in targobj.aliases:
            xa = targ[targobj.aliases["ap"]]
            p_xa = targ[targobj.aliases.get("p", "p")]
            z_xa = targ[targobj.aliases.get("z", "z")]
        else:
            extra = targobj.get_additional_field(targ, "(smoothing)")
            xa = extra["ch4_ap"]
            p_xa = extra["p_ch4_ap"]
            if "z_ch4_ap" in extra.dtype.names:
                z_xa = extra["z_ch4_ap"]
            #xa = targobj.get_additional_field(targ, "CH4_apriori")

        ## Look for averaging kernel and z-grid (will need to regrid later)
        if "ak" in targobj.aliases:
            ak = targ[targobj.aliases["ak"]]
            p_ak = targ[targobj.aliases.get("p", "p")]
            z_ak = targ[targobj.aliases.get("z", "z")]
        elif extra is None:
            ak = targobj.get_additional_field(targ, "CH4_ak")
            raise RuntimeError("Found a priori but not AK outside ?!")
        else:
            ak = extra["ch4_ak"]
            p_ak = extra["p_ch4_ak"]
            if "z_ch4_ak" in extra.dtype.names:
                z_ak = extra["z_ch4_ap"]

        if z_xa is None: # convert from p_xa
            if ((p_xa.shape == targ[targobj.aliases.get("p", "p")].shape)
                and (p_xa - targ[targobj.aliases.get("p", "p")]).max()
                < 1e-3): # same grid
                z_xa = targ["z"]
            elif (p_xa.shape == targ["T"].shape == targ["h2o"].shape):
                # cannot be vectorised :(
                filler = numpy.empty(shape=(p_xa.shape[1],), dtype="f4")
                filler.fill(numpy.nan)
                z_xa = [(physics.p2z_hydrostatic(
                            p_xa[i, :],
                            targ[i]["T"],
                            targ[i]["h2o"],
                            targ[i]["p0"],
                            targ[i]["z0"],
                            targ[i]["lat"],
                            -1, extend=True)
                                if numpy.isfinite(xa[i]).any() else filler)
                            for i in range(targ.shape[0])]
                z_xa = numpy.vstack(z_xa)
            else:
                raise ValueError("Can't find z.  Should I try harder?")

        if z_ak is None: # convert from p_ak
            if ((p_ak.shape == targ[targobj.aliases.get("p", "p")].shape)
                and numpy.nanmax(p_ak - targ[targobj.aliases.get("p", "p")])
                < 1e-3):
                z_ak = targ["z"]
            elif (p_ak.shape == p_xa.shape) and numpy.nanmax(p_ak - p_xa) < 1e-3:
                z_ak = z_xa
            else:
                raise ValueError("Don't know how to get z.  Should I try harder?")
        return (xa, z_xa, ak, z_ak)

    def _get_shared_range(self, z_xa, z_ak, z_xh):
        """_get_shared_range(z_xa, z_ak, z_xh)

        Returns (highest_z_min, highest_z_min_index,
                 lowest_z_max, lowest_z_max_index),
        where index means 0 for z_xa, 1 for z_ak, 2 for z_xh.
        """

        each_z_max = numpy.array((
            (numpy.nanmin(numpy.nanmax(z_ak, 1)) if z_ak.ndim > 1
                     else numpy.nanmax(z_ak, 0)),
            (numpy.nanmin(numpy.nanmax(z_xa, 1)) if z_xa.ndim > 1
                     else numpy.nanmax(z_xa, 0)),
            (numpy.nanmin(numpy.nanmax(z_xh, 1)) if z_xh.ndim > 1
                     else numpy.nanmax(z_xh, 0))))

        each_z_min = numpy.array((
            (numpy.nanmax(numpy.nanmin(z_ak, 1)) if z_ak.ndim > 1
                     else numpy.nanmin(z_ak, 0)),
            (numpy.nanmax(numpy.nanmin(z_xa, 1)) if z_xa.ndim > 1
                     else numpy.nanmin(z_xa, 0)),
            (numpy.nanmax(numpy.nanmin(z_xh, 1)) if z_xh.ndim > 1
                     else numpy.nanmin(z_xh, 0))))

        lowest_z_max_i = each_z_max.argmin()
        highest_z_min_i = each_z_min.argmin()

        return (each_z_min[highest_z_min_i], highest_z_min_i,
                each_z_max[lowest_z_max_i], lowest_z_max_i)

    def _limit_to_shared_range(self, xa, z_xa, ak, z_ak, xh, z_xh,
        p_int, s_int, p_sa, s_sa):
        """Helper for smooth(...)
        """
        (highest_z_min, _, lowest_z_max, _) = self._get_shared_range(
            z_xa, z_ak, z_xh)

        if z_xh.max() > lowest_z_max:
            # cut off actual profiles
            toohigh = self.z_grid > lowest_z_max
            xh = xh[:, ~toohigh]
            z_xh = z_xh[~toohigh]
            self.z_grid = z_xh
            # NB: to smooth the high-res measurement `xh` with the
            # low-resolution averaging kernel `ak` and a priori `xa`, the
            # low--res measurement is not used.  However, it is still
            # returned, so should still be cut off so caller can
            # consistently process. 
            p_int_new = numpy.zeros(shape=p_int.shape, dtype=
                [(x[0], x[1], (~toohigh).sum())
                    for x in p_int.dtype.descr])
            s_int_new = numpy.zeros(shape=s_int.shape, dtype=
                [(x[0], x[1], (~toohigh).sum())
                    for x in s_int.dtype.descr])
            for f in p_int.dtype.names:
                p_int_new[f] = p_int[f][:, ~toohigh]
            for f in s_int.dtype.names:
                s_int_new[f] = s_int[f][:, ~toohigh]
            (p_int, s_int) = (p_int_new, s_int_new)
            #p_ch4_int = p_ch4_int[:, ~toohigh]
            #s_ch4_int = s_ch4_int[:, ~toohigh]

        ########################
        #
        # I'm going to have to regrid from z_ak to z_xh and from z_xa to
        # z_xh.  That means I want the largest values of z_xa and z_ak
        # to be LARGER # than the one for z_xh, not smaller!  Return now
        # and DO NOT remove levels from those!
        #
        ########################
        return (xa, z_xa, ak, z_ak, xh, z_xh, p_int, s_int, p_sa, s_sa)

    def _regrid_xa_ak(self, xa, z_xa, ak, z_ak, z_xh):
        """Helper for smooth(...)

        Regrids xa(z_xa) and ak(z_ak) onto the grid of z_xh

        Returns (xa, z_xa, W_xa, ak, z_ak, W_ak).

        TODO: also regrid error estimates.  This can be carried out using:
        Von Clarmann (2014), equation 8/13/14.  Actually no.  But it can
        be with: Vigouroux et al. (2007), which cites Calisesi et al.
        (2005).  Actually, just return W's and leave it to someone else?
        """
        # WARNING: What if z_ak.shape == z_xh.shape
        # but z_ak != z_xh?
        if z_ak.shape == z_xh.shape and not tools.array_equal_with_equal_nans(z_ak, z_xh):
            raise NotImplementedError("Improve z-axis checking!")
        if z_ak.shape != z_xh.shape:
            # regrid from z_xa to z_ak
            #z_ak_new = targ["z"]
            z_ak_new = z_xh
            z_ak_old = z_ak
            ak_old = ak
            with numpy.errstate(invalid="ignore"):
                (all_ak, all_W) = zip(*[pamath.regrid_ak(
                    ak_old[i, :, :], z_ak_old[i, :], z_ak_new, cut=True)
                        for i in range(z_ak.shape[0])])
                ak_new = numpy.dstack(all_ak)
                W_ak = numpy.dstack(all_W)
            ak = numpy.rollaxis(ak_new, 2, 0)
            W_ak = numpy.rollaxis(W_ak, 2, 0)
            z_ak = z_ak_new

        # WARNING: What if z_xa.shape == z_xh.shape
        # but z_xa != z_xh?
        if z_xa.shape == z_xh.shape and not tools.array_equal_with_equal_nans(z_xa, z_xh):
            raise NotImplementedError("Improve z-axis checking!")
        if z_xa.shape != z_xh.shape:
            #z_xa_new = targ["z"]
            z_xa_new = z_xh
            z_xa_old = z_xa
            xa_old = xa
            W = numpy.dstack([
                pamath.linear_interpolation_matrix(
                    z_xa_old[i, :], z_xa_new)
                        for i in range(z_xa.shape[0])])
            W = numpy.rollaxis(W, 2, 0)
            xa_new = numpy.vstack([
                W[i, :, :].dot(xa_old[i, :]) for i in range(W.shape[0])])
            W_xa = W
            xa = xa_new
            z_xa = z_xa_new

        return (xa, z_xa, W_xa, ak, z_ak, W_ak)

    def _calc_error_propagation(self, S_1, W_1, A_1, S_2, W_2):
        """Calculate random covariance matrix for comparison

        Upon regridding and smoothing, we need to calculate the error
        covariance matrix for the comparison ensemble.  For example, this
        is performed in Vigouroux et al. (2007) and in Calisesi et al.
        (2005).

        Note that this assumes the higher resolution instrument has a so
        much higher resolution that its averaging kernel can be
        approximated by I_n.

        FIXME — make sure averaging kernel is optimal with respect to the
        comparison ensemble, as noted in Rodgers and Connor (2003), page
        13-6, section 4.4, after equation 30.

        :param S_1:
        :param W_1:
        :param A_1:
        :param S_2:
        :param W_2:
        """

        W_12 = numpy.linalg.pinv(W_1).dot(W_2)
        S_12 = S_1 + A_1.dot(W_12).dot(S_2).dot(W_12.T).dot(A_1.T)
        return S_12

    ## Visualisation methods

    def plot_aks(self):
        """Visualise averaging kernels.

        Will average all averaging kernels and plot them as lines.
        """

        if "ak" in self.cd.primary.aliases:
            # make a copy because I'm going to set flagged values to 'nan'
            p_ak = self.p_col[self.cd.primary.aliases["ak"]].copy()
        else:
            p_ak = None

        if "ak" in self.cd.secondary.aliases:
            s_ak = self.s_col[self.cd.secondary.aliases["ak"]].copy()
        else:
            s_ak = None

        f = matplotlib.pyplot.figure()
        a1 = f.add_subplot(1, 2, 1)
        a2 = f.add_subplot(1, 2, 2)
        a_both = []

        mx = mn = 0
        data = []
        if "ak" in self.cd.primary.aliases:
            p_ak[p_ak<-100] = numpy.nan # presumed flagged
            mean_p_ak = numpy.nanmean(p_ak, 0)
            a1.plot(mean_p_ak.T, self.p_col["z"].mean(0))
            a1.set_title(self.cd.primary.name)
            a_both.append(a1)
            mx = numpy.nanmax(mean_p_ak)
            mn = numpy.nanmin(mean_p_ak)
            data.append(numpy.hstack(
                (self.p_col["z"].mean(0)[:, numpy.newaxis], mean_p_ak.T)))

        if "ak" in self.cd.secondary.aliases:
            s_ak[s_ak<-100] = numpy.nan # presumed flagged
            mean_s_ak = numpy.nanmean(s_ak, 0)
            a2.plot(mean_s_ak.T, self.s_col["z"].mean(0))
            a2.set_title(self.cd.secondary.name)
            a_both.append(a2)
            mx = numpy.max([mx, numpy.nanmax(mean_s_ak)])
            mn = numpy.min([mn, numpy.nanmin(mean_s_ak)])
            data.append(numpy.hstack(
                (self.s_col["z"].mean(0)[:, numpy.newaxis], mean_s_ak.T)))

        for a in a_both:
            a.set_xlabel("Mean averaging kernel []")
            a.set_ylabel("Elevation [m]")
            a.set_xlim([mn, mx])
            a.set_ylim([0, 40e3])

        graphics.print_or_show(f, False,
            "ak_{}_{}.".format(self.cd.primary.__class__.__name__,
                               self.cd.secondary.__class__.__name__),
            data=data)

        # And the matrices
        
        logging.info("Summarising sensitivities")
        if p_ak is not None:
            paks = physics.AKStats(p_ak, 
                name="{}_from_{}".format(
                    self.cd.primary.__class__.__name__,
                    self.cd.secondary.__class__.__name__))
            if not "parcol_CH4" in self.p_col.dtype.names:
                self.partial_columns()
            paks.summarise(data=self.p_col)

        if s_ak is not None:
            saks = physics.AKStats(s_ak, 
                name="{}_from_{}".format(
                    self.cd.secondary.__class__.__name__,
                    self.cd.primary.__class__.__name__))
            if not "parcol_CH4" in self.s_col.dtype.names:
                self.partial_columns()
            saks.summarise(data=self.s_col)


    def visualise_profile_comparison(self, z_grid, filters=None):
        """Visualise profile comparisons.

        Currently hardcoded for CH4.

        Arguments as for compare_profiles, plus
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
        profs = {}
        lims = {}
        xlabels = dict(
            diff = "Delta CH4 [ppv]",
            rmsd = "RMSD CH4 [ppv]",
            ratio = "CH4 ratio [1]",
            prim = "Primary CH4 [ppv]",
            sec = "Secondary CH4 [ppv]")
        xlims = dict(
            diff = (-1e-7, 1e-7),
            rmsd = (0, 2e-7),
            ratio = (0.9, 1.9),
            prim = (0, 2e-6),
            sec = (0, 2e-6))
        self.reset_filter()

        # see how smoothed or raw compare

        filter_modes = []

        filter_modes.append(self.mask_label)
        z_grids = {}
#        profs[self.mask_label + "_smooth"] = self.smooth()
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
            filter_modes.append(self.mask_label)
#        percs = self.compare_profiles(z_grid, p_locs)
        #for (i, v) in enumerate("diff diff^2 ratio".split()):
        # quantities such as diff, ratio, rmsd
        for quantity in percs[self.mask_label + "_raw"].keys():
            f = matplotlib.pyplot.figure()
            a = f.add_subplot(1, 1, 1)
            data = dict(smooth=[], raw=[])
            # filters such as all, nearby, small delta-SPV
            for filter_mode in filter_modes: # percs.keys():
                for ff in ("smooth", "raw"):
                    filt = "{}_{}".format(filter_mode, ff)
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
            a.set_ylim([5e3, 50e3])
            # Set y-lim according to ak's
            # (never mind for now, will set by hand in LaTeX code)
#            p_ak = self.p_col[self.cd.primary.aliases["ak"]]
#            s_ak = self.s_col[self.cd.secondary.aliases["ak"]]

            a.text(xlims[quantity][0]+0.1*xlims[quantity][1], 45e3, "{:d} profiles".format(self.mask.sum()))
            allmask = ','.join(filter_modes)
            graphics.print_or_show(f, False,
                self.figname_compare_profiles.format(**vars()),
                data=
                    (numpy.vstack((z_grids["raw"],)+tuple(data["raw"])).T,
                     numpy.vstack((z_grids["smooth"],)+tuple(data["smooth"])).T)
                    )
            matplotlib.pyplot.close(f)
        # end for quantities

        ## Plot all profiles
##        f = matplotlib.pyplot.figure()
##        a = f.add_subplot(1, 1, 1)
##        for (w, c) in [("raw", "black"), ("smooth", "red")]:
##            a.plot(profs[self.mask_label + "_" + w][self.target].T, z_grids[w],
##                  color=c, label=w + " prim", linewidth=0.3)
###            a.plot(profs[self.mask_label + "_" + w][1].T, z_grids[w],
###                  color=c, label=w + " sec", linewidth=0.3)
###        a.legend()
##        a.set_xlabel("CH4 [ppv]")
##        a.set_ylabel("Elevation [m]")
##        a.set_title("All profiles")
##        a.grid(which="major")
##        a.set_ylim([5e3, 60e3])
##        a.set_xlim([0, 2e-6])
##        graphics.print_or_show(f, False,
##            self.figname_compare_profiles.format(self=self,
##                quantity="spaghetti"))
##        matplotlib.pyplot.close(f)
        #

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

    def visualise_pc_comparison(self):
        """Visualise comparison for partial columns
        """
        (p_parcol, s_parcol, valid_range) = self.partial_columns(smoothed=True)

        (f, a) = matplotlib.pyplot.subplots()
        valid = numpy.isfinite(p_parcol) & numpy.isfinite(s_parcol)
        d_parcol = (s_parcol[valid] - p_parcol[valid])
        #a.plot(p_parcol[valid], s_parcol[valid], '.')
        a.plot(p_parcol[valid], d_parcol, '.')
        mx = max(p_parcol[valid].max(), s_parcol[valid].max())
        mn = min(p_parcol[valid].min(), s_parcol[valid].min())
        #a.plot([0, 2*mx], [0, 2*mx], linewidth=2, color="black")
        a.plot([0, 2*mx], [0, 0], linewidth=2, color="black")
        a.set_xlim(0.9*mn, 1.1*mx)
        #a.set_ylim(0.9*mn, 1.1*mx)
        a.set_xlabel("CH4 {:s} [molec./m^2]".format(self.cd.primary.name))
        a.set_ylabel("CH4 {:s} - {:s} [molec./m^2]".format(
            self.cd.primary.name, self.cd.secondary.name))
        a.set_title(("Partial columns {:.1f}--{:.1f}, "
                     "difference {:s} - {:s}").format(
                        valid_range[0]/1e3, valid_range[1]/1e3,
                        self.cd.primary.name, self.cd.secondary.name))
        graphics.print_or_show(f, None, self.figname_compare_pc.format(**vars()),
            data=numpy.vstack((p_parcol[valid],
                s_parcol[valid]-p_parcol[valid])).T)

        # also print some stats
        diff = {}
        diff["mean"] = d_parcol.mean()
        diff["std"] = d_parcol.std()
        diff["median"] = numpy.median(d_parcol)
        diff["sem"] = scipy.stats.sem(d_parcol)
        diff["mad"] = numpy.median(abs(d_parcol - numpy.median(d_parcol)))
        for (k, v) in diff.items():
            logging.info("Statistic: {:s}: {:.3e} molecules/m^2".format(k, v))

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
