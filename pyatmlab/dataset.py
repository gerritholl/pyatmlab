"""Module containing classes abstracting datasets
"""

import abc
import functools
import itertools
import logging
import pathlib
import re
import shelve
import string
import sys
import shutil
import tempfile

import datetime
import numpy
import numpy.lib.arraysetops
import numpy.lib.recfunctions

from . import tools

class InvalidFileError(Exception):
    """Raised when the requested information cannot be obtained from the file
    """

class InvalidDataError(Exception):
    """Raised when data is not how it should be.
    """

class Dataset(metaclass=abc.ABCMeta):
    """Represents a dataset.

    This is an abstract class.  More specific subclasses are
    SingleFileDataset and MultiFileDataset.  Do not subclass Dataset
    directly.

    Attributes defined here::

    - start_date::

        Starting date for dataset.  May be used to search through ALL
        granules.  WARNING!  If this is set at a time t_0 before the
        actual first measurement t_1, then the collocation algorith (see
        CollocatedDataset) will conclude that there are 0 collocations
        in [t_0, t_1], and will not realise if data in [t_0, t_1] are
        actually added later!

    - end_date::

        Similar to start_date, but for ending.

    - name::
        
        Name for the dataset.  May be used for logging purposes and so.

    - aliases::

        Aliases for field.  Dictionary can be useful if you want to
        programmatically loop through the same field for many different
        datasets, but they are named differently.  Aliases known to be
        used::

        - ch4_profile

    - unique_fields::
        
        Set of fields that make any individual measurement unique.

    """

    start_date = None
    end_date = None
    name = ""
    aliases = {}
    unique_fields = {"time", "lat", "lon"}

    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        if hasattr(self, k) or hasattr(type(self), k):
            object.__setattr__(self, k, v)
        else:
            raise AttributeError("Unknown attribute: {}. ".format(k))

    @abc.abstractmethod
    def find_granules(self, start=datetime.datetime.min,
                                  end=datetime.datetime.max):
        """Loop through all granules for indicated period.

        This is a generator that will loop through all granules from
        `start` to `end`, inclusive.

        See also: `find_granules_sorted`

        :param datetime start: Starting datetime, defaults to any
        :param datetime end: Ending datetime, defaults to any
        :yields: `pathlib.Path` objects for all granules
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def find_granules_sorted(self, start=None, end=None):
        """Yield all granules sorted by starting time then ending time.

        For details, see `find_granules`.
        """
        raise NotImplementedError()

    def read_period(self, start=None,
                          end=None,
                          onerror="skip",
                          fields="all"):
        """Read all granules between start and end, in bulk.

        :param datetime start: Starting time, None for any time
        :param datetime end: Ending time, None for any time
        :param str onerror: What to do with unreadable files.  Defaults to
            "skip", can be set to "raise".
        :param list fields: List of fields to read, or "all" (default).
        :returns: Masked array with all data in period.
        """

        start = start or self.start_date
        end = end or self.end_date

        contents = []
        for gran in self.find_granules(start, end):
            try:
                logging.info("Reading {!s}".format(gran))
                cont = self.read(str(gran), fields=fields)
            except (OSError, ValueError) as exc:
                if onerror == "skip":
                    print("Could not read file {}: {}".format(
                        gran, exc.args[0], file=sys.stderr))
                    continue
                else:
                    raise
            else:
                contents.append(cont)
        # retain type of first result, ordinary array of masked array
        arr = (numpy.ma.concatenate 
            if isinstance(contents[0], numpy.ma.MaskedArray)
            else numpy.concatenate)(contents)
        return arr if fields == "all" else arr[fields]
#        return numpy.ma.concatenate(list(
#            self.read(f) for f in self.find_granules(start, end)))

    def read_all(self, fields="all", onerror="skip"):
        """Read all data in one go.

        Warning: for some datasets, this may cause memory problems.

        :param fields: List of fields to read, or "all" (default)
        :param onerror: What to do in case of an error.  See
            `read_period`.
        """

        return self.read_period(fields=fields)
            
    @abc.abstractmethod
    def _read(self, f, fields="all"):
        """Read granule in file, low-level

        Shall return an ndarray with at least the fields lat, lon, time.

        :param str f: Path to file
        :param fields: List of fields to read, or "all" (default)
        """

        raise NotImplementedError()

    #@functools.lru_cache(maxsize=10)
    @tools.mutable_cache(maxsize=10)
    def read(self, f=None, fields="all"):
        """Read granule in file and do some other fixes

        Uses self._read.  Do not override, override _read instead.

        :param str f: Path to file
        :param list fields: Fields to read or "all" (default)
        """
        if isinstance(f, pathlib.PurePath):
            f = str(f)
        return self._read(f) if f is not None else self._read()

    def __str__(self):
        return "Dataset:" + self.name

    
    def combine(self, my_data, other_obj):
        """Combine with data from other dataset

        Combine a set of measurements from this dataset with another
        dataset, where each individual measurement correspond to exactly
        one from the other one, as identified by time/lat/lon, orbitid, or
        measurument id, or other characteristics.  The object attribute
        unique_fields determines how those are found.

        The other dataset may contain flags, DMPs, or different
        information altogether.

        :param ndarray my_data: Data as returned from self.read.
        :param Dataset other_obj: Other dataset from which to read
            corresponding data.
        """

        first = my_data["time"].min().astype(datetime.datetime)
        last = my_data["time"].max().astype(datetime.datetime)

        other_data = other_obj.read_period(first, last)
        other_ind = numpy.zeros(dtype="u4", shape=my_data.shape)
        found = numpy.zeros(dtype="bool", shape=my_data.shape)
        other_combi = numpy.zeros(dtype=other_data.dtype, shape=my_data.shape)
        
        # brute force algorithm for now.  Update if needed.
        for i in range(my_data.shape[0]):
            ident = [(my_data[i][f] == other_data[f]).nonzero()[0]
                        for f in self.unique_fields]
            # N arrays of numbers, find numbers occuring in each array.
            # Should be exactly one!
            secondaries = functools.reduce(numpy.lib.arraysetops.intersect1d, ident)
            if secondaries.shape[0] == 1: # all good
                other_ind[i] = secondaries[0]
                found[i] = True
            elif secondaries.shape[0] > 1: # don't know what to do!
                raise InvalidDataError(
                    "Expected 1 unique, got {:d}".format(
                        secondaries.shape[0]))
        # else, leave unfound
        other_combi[found] = other_data[other_ind[found]]
        # Meh.  even with usemask=False, append_fields internally uses
        # masked arrays.
#        return numpy.lib.recfunctions.append_fields(
#            my_data, other_combi.dtype.names,
#                [other_combi[s] for s in other_combi.dtype.names], usemask=False)
#       But what if fields have the same name?  Just return both...
#        dmp = numpy.zeros(shape=my_data.shape,
#            dtype=my_data.dtype.descr + other_combi.dtype.descr)
#        for f in my_data.dtype.names:
#            dmp[f] = my_data[f]
#        for f in other_combi.dtype.names:
#            dmp[f] = other_combi[f]
        return other_combi


class SingleFileDataset(Dataset):
    """Represents a dataset where all measurements are in one file.
    """

    start_date = datetime.datetime.min
    end_date = datetime.datetime.max
    srcfile = None

    def find_granules(self, start=datetime.datetime.min,
                            end=datetime.datetime.max):
        if start < self.end_date and end > self.start_date:
            yield self.srcfile

    def find_granules_sorted(self, start=datetime.datetime.min,
                                   end=datetime.datetime.max):
        yield from self.find_granules(start, end)

    def get_times_for_granule(self, gran=None):
        return (self.start_date, self.end_date)

class MultiFileDataset(Dataset):
    """Represents a dataset where measurements are spread over multiple
    files.

    If filenames contain timestamps, this information is used to determine
    the time for a granule or measurement.  If filenames do not contain
    timestamps, this information is obtained from the file contents.

    Contains the following attributes::

        basedir:
            Describes the directory under which all granules are located.
            Can be either a string or a pathlib.Path object.

        subdir::
            Describes the directory within basedir where granules are
            located.  May contain string formatting directives where
            particular fields are replaces, such as `year`, `month`, and
            `day`.  For example: `subdir = '{year}/{month}'.  Sorting
            cannot be more narrow than by day.

        re::
            Regular expression that should match valid granule files within
            `basedir` / `subdir`.  Should use symbolic group names to capture
            relevant information when possible, such as starting time, orbit
            number, etc.  For time identification, relevant fields are
            contained in MultiFileDataset.date_info, where each field also
            exists in a version with "_end" appended.
            MultiFileDataset.refields contains all recognised fields.

            If any *_end fields are found, the ending time is equal to the
            beginning time with any *_end fields replaced.  If no *_end
            fields are found, the `granule_duration` attribute is used to
            determine the ending time, or the file is read to get the
            ending time (hopefully the header is enough).

        granule_cache_file::
            If set, use this file to cache information related to
            granules.  This is used to cache granule times if those are
            not directly inferred from the filename.  Otherwise, this is
            not used.  The full path to this file shall be `basedir` /
            `granule_cache_file`.

        granule_duration::
            If the filename contains starting times but no ending times,
            granule_duration is used to determine the ending time.  This
            should be a datetime.timedelta object.
    """
    basedir = None
    subdir = ""
    re = None
    _re = None # compiled version
    granule_cache_file = None
    _granule_start_times = None
    granule_duration = None

    datefields = "year month day hour minute second".split()
    # likely extended later
    refields = ["".join(x)
        for x in itertools.product(datefields, ("", "_end"))]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for attr in ("basedir", "subdir", "granule_cache_file"):
            if (getattr(self, attr) is not None and
                not isinstance(getattr(self, attr), pathlib.PurePath)):
                setattr(self, attr, pathlib.Path(getattr(self, attr)))
        if self.granule_cache_file is not None:
            p = str(self.basedir / self.granule_cache_file)
            try:
                self._granule_start_times = shelve.open(p, protocol=4)
            except OSError:
                logging.error(("Unable to open granule file {} RW.  "
                               "Opening copy instead.").format(p))
                tf = tempfile.NamedTemporaryFile()
                shutil.copyfile(p, tf.name)
                #self._granule_start_times = shelve.open(p, flag='r')
                self._granule_start_times = shelve.open(tf.name)
        else:
            self._granule_start_times = {}
        if self.re is not None:
            self._re = re.compile(self.re)

    def find_dir_for_time(self, dt):
        """Find the directory containing granules/measurements at (date)time

        For a given datetime object, find the directory that contains
        granules/measurument files for this particular time.

        :param datetime dt: Timestamp for inquiry.  Any object with
            `year`, `month`, `day` attributes works.
        :returns: pathlib.Path object to relevant directory
        """
        return pathlib.Path(str(self.basedir / self.subdir).format(
            year=dt.year, month=dt.month, day=dt.day,
            doy=dt.timetuple().tm_yday))

    def get_subdir_resolution(self):
        """Return the resolution for the subdir precision.

        Returns "year", "month", "day", or None (if there is no subdir).
        """
        fm = string.Formatter()
        fields = {f[1] for f in fm.parse(str(self.subdir))}
        if "year" in fields:
            if "month" in fields:
                if "day" in fields:
                    return "day"
                return "month"
            elif "doy" in fields:
                return "day"
            return "year"

    def iterate_subdirs(self, d_start, d_end):
        """Iterate through all subdirs in dataset.

        Note that this does not check for existance of those directories.

        Yields a 2-element tuple where the first contains information on
        year(/month/day), and the second is the path.

        :param date d_start: Starting date
        :param date d_end: Ending date
        """

        # depending on resolution, iterate by year, month, or day.
        # Resolution is determined by provided fields in self.subdir.
        d = d_start
        res = self.get_subdir_resolution()

        pst = str(self.basedir / self.subdir)
        if res == "year":
            year = d.year
            while datetime.date(year, 1, 1) < d_end:
                yield (dict(year=year),
                    pathlib.Path(pst.format(year=year)))
                year += 1
        elif res == "month":
            year = d.year
            month = d.month
            while datetime.date(year, month, 1) < d_end:
                yield (dict(year=year, month=month),
                    pathlib.Path(pst.format(year=year, month=month)))
                if month == 12:
                    year += 1
                    month = 1
                else:
                    month += 1
        elif res == "day":
            #while d < d_end:
            if any(x[1] == "doy" for x in string.Formatter().parse(pst)):
                while d < d_end:
                    doy = d.timetuple().tm_yday
                    yield (dict(year=d.year, doy=doy),
                        pathlib.Path(pst.format(year=d.year, doy=doy)))
                    d += datetime.timedelta(days=1)
            else:
                while d < d_end:
                    yield (dict(year=d.year, month=d.month, day=d.day),
                        pathlib.Path(pst.format(
                            year=d.year, month=d.month, day=d.day)))
                    d = d + datetime.timedelta(days=1)
        else:
            yield ({}, self.basedir)
          
    def find_granules(self, dt_start=None, dt_end=None):
        """Yield all granules/measurementfiles in period
        """

        if dt_start is None:
            dt_start = self.start_date

        if dt_end is None:
            dt_end = self.end_date

        d_start = (dt_start.date()
                if isinstance(dt_start, datetime.datetime) 
                else dt_start)
        d_end = (dt_end.date()
                if isinstance(dt_end, datetime.datetime) 
                else dt_end)
        logging.debug(("Searching for {!s} granules between {!s} and {!s} "
                      ).format(self.name, dt_start, dt_end))
        for (timeinfo, subdir) in self.iterate_subdirs(d_start, d_end):
            if subdir.exists() and subdir.is_dir():
                logging.debug("Searching directory {!s}".format(subdir))
                for child in subdir.iterdir():
                    m = self._re.fullmatch(child.name)
                    if m is not None:
                        try:
                            (g_start, g_end) = self.get_times_for_granule(child,
                                **timeinfo)
                        except InvalidFileError as e:
                            logging.error(
                                "Skipping {!s}.  Problem: {}".format(
                                    child, e.args[0]))
                            continue
                        if g_end > dt_start and g_start < dt_end:
                            yield child
            
    def find_granules_sorted(self, dt_start=None, dt_end=None):
        """Yield all granules, sorted by times
        """

        allgran = list(self.find_granules(dt_start, dt_end))

        # I've been through all granules at least once, so all should be
        # cached now; no need for additional hints when granule timeinfo
        # obtainable only with hints from subdir, which is not included in
        # the re-matching method
        yield from sorted(allgran, key=self.get_times_for_granule)

    def _getyear(self, gd, s, alt):
        """Extract year info from group-dict

        Taking a group dict and a string, get an int for the year.
        The group dict should come from re.fullmatch().groupdict().  The
        second argument is typically "year" or "year_end".  If this is a
        4-digit string, it is taken as a year.  If it is a 2-digit string,
        it is taken as a 2-digit year, which is taken as 19xx for <= 68,
        and 20xx for >= 69, according to POSIX and ISO C standards.
        If there is no match, return alt.

        :param gd: Group-dict from regular expression
        :param s: String to match for
        :param alt: Alternative value
        """

        if s in gd:
            i = int(gd[s])
            if len(gd[s]) == 2:
                return 1900 + i if i> 68 else 2000 + i
            elif len(gd[s]) == 4:
                return i
            else:
                raise ValueError("Found {:d}-digit string for the year. "
                    "Expected 2 or 4 digits.  Giving up.".format(len(gd[s])))
        else:
            return alt

    def get_info_for_granule(self, p):
        """Return dict (re.fullmatch) for granule, based on re
        """

        if not isinstance(p, pathlib.Path):
            p = pathlib.Path(p)
        m = self._re.fullmatch(p.name)
        return m.groupdict()

    def get_times_for_granule(self, p,
            **kwargs):
        """For granule stored in `path`, get start and end times.

        May take hints for year, month, day, hour, minute, second, and
        their endings, according to self.date_fields
        """
        if not isinstance(p, pathlib.PurePath):
            p = pathlib.PurePath(p)
        if str(p) in self._granule_start_times.keys():
            (start, end) = self._granule_start_times[str(p)]
        else:
            gd = self.get_info_for_granule(p)
            if (any(f in gd.keys() for f in self.datefields) and
                (any(f in gd.keys() for f in {x+"_end" for x in self.datefields})
                        or self.granule_duration is not None)):
                st_date = [int(gd.get(p, kwargs.get(p, "0"))) for p in self.datefields]
                # month and day can't be 0...
                st_date[1] = st_date[1] or 1
                st_date[2] = st_date[2] or 1
                # maybe it's a two-year notation
                st_date[0] = self._getyear(gd, "year", kwargs.get("year", "0"))

                start = datetime.datetime(*st_date)
                if any(k.endswith("_end") for k in gd.keys()):
                    end_date = [int(gd.get(
                        p+"_end",
                        kwargs.get(p+"_end", None))) for p in self.datefields]
                    end_date[0] = self._getyear(gd, "year_end", kwargs.get("year_end", "0"))
                    end = datetime.datetime(*end_date)
                elif self.granule_duration is not None:
                    end = start + self.granule_duration
                else:
                    raise RuntimeError("This code should never execute")
            else:
                # implementation depends on dataset
                (start, end) = self.get_time_from_granule_contents(str(p))
                self._granule_start_times[str(p)] = (start, end)
        return (start, end)

    # not an abstract method because subclasses need to implement it /if
    # and only if starting/ending times cannot be determined from the filename
    def get_time_from_granule_contents(self, p):
        """Get datetime objects for beginning and end of granule

        If it returns None, then use same as start time.
        """
        raise ValueError(
            ("To determine starting and end-times for a {0} dataset, "
             "I need to read the file.  However, {0} has not implemented the "
             "get_time_from_granule_contents method.".format(
                type(self).__name__)))

class SingleMeasurementPerFileDataset(MultiFileDataset):
    """Represents datasets where each file contains one measurement.

    An example of this would be ACE, or some RO datasets.

    Special attributes::

        filename_fields::

            dict with {name, dtype} for fields that should be copied from
            the filename (as obtained with self.re) into the header
    """

    granule_duration = datetime.timedelta(0)
    filename_fields = {}


    @abc.abstractmethod
    def read_single(self, p, fields="all"):
        """Read a single measurement from a single file.

        Shall take one argument (a path object) and return a tuple with
        (header, measurement).  The header shall contain information like
        latitude, longitude, time.
        """

    _head_dtype = dict(CH4_total="<f4")
    def _read(self, p, fields="all"):
        """Reads a single measurement converted to ndarray
        """

        (head, body) = self.read_single(p, fields=fields)

        dt = [(s+body.shape if len(s)==2 else (s[0], s[1], s[2]+body.shape))
                for s in body.dtype.descr]
        dt.extend([("lat", "f8"), ("lon", "f8"), ("time", "M8[s]")])
        dt.extend([(s, self._head_dtype[s])
                for s in (head.keys() & self._head_dtype.keys())
                if s not in {"lat", "lon", "time"}])
        if self.filename_fields:
            info = self.get_info_for_granule(p)
            dt.extend(self.filename_fields.items())
        # This fails. https://github.com/numpy/numpy/issues/4583
        #D = numpy.ma.empty(1, dt)
        D = numpy.empty(1, dt)

        for nm in body.dtype.names:
            D[nm] = body[nm]

        for nm in {"lat", "lon", "time"}:
            D[nm] = head[nm]

        for nm in head.keys() & D.dtype.names:
            if nm not in {"lat", "lon", "time"}:
                D[nm] = head[nm]

        if self.filename_fields:
            for nm in self.filename_fields.keys():
                D[nm] = info[nm]

        return D

class HomemadeDataset(MultiFileDataset):
    """For any dataset created by pyatmlab.

    No content yet.
    """
    # dummy implementation for abstract methods, so that I can test other
    # things

    stored_name = ""

    def find_granule_for_time(self, **kwargs):
        """Find granule for specific time.

        May or may not exist.

        Arguments (kw only) are passed on to format directories stored in
        self.basedir / self.subdir / self.stored_name, along with
        self.__dict__.

        Returns path to granule.
        """

        d = self.basedir / self.subdir / self.stored_name
        subsdict = self.__dict__.copy()
        subsdict.update(**kwargs)
        nm = pathlib.Path(str(d).format(**subsdict))
        return nm

    def _read(self, f, fields="all"):
        raise NotImplementedError()

#    def find_granules(self, start, end):
#        raise StopIteration()

#    @abc.abstractmethod
#    def quicksave(self, f):
#        """Quick save to file
#
#        :param str f: File to save to
#        """


class ProfileDataset(Dataset):
    """Abstract superclass with stuff for anything having profiles
    """

    @abc.abstractmethod
    def get_z(self, dt):
        """Get z-profile in metre.

        Takes as argument a single measurement, with the dtype returned by
        its own reading routine.
        """

class StationaryDataset(Dataset):
    """Abstract superclass for any ground-station dataset
    """

    unique_fields = {"time"}
