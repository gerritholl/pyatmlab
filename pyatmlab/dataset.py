"""Module containing classes abstracting datasets
"""

import abc
import functools
import itertools
import pathlib
import re
import shelve
import string
import sys

import datetime
import numpy

class Dataset(metaclass=abc.ABCMeta):
    """Represents a dataset.

    This is an abstract class.  More specific subclasses are
    SingleFileDataset and MultiFileDataset.  Do not subclass Dataset
    directly.

    Attributes defined here::

    - start_date::

        Starting date for dataset.  May be used to search through ALL
        granules.

    - end_date::

        Similar to start_date, but for ending.

    - name::
        
        Name for the dataset.  May be used for logging purposes and so.

    """

    start_date = None
    end_date = None
    name = ""

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

    def read_period(self, start=datetime.datetime.min,
                          end=datetime.datetime.max,
                          onerror="skip"):
        """Read all granules between start and end, in bulk.

        :param datetime start: Starting time, None for any time
        :param datetime end: Ending time, None for any time
        :param str onerror: What to do with unreadable files.  Defaults to
            "skip", can be set to "raise".
        :returns: Masked array with all data in period.
        """

        contents = []
        for gran in self.find_granules(start, end):
            try:
                cont = self.read(str(gran))
            except ValueError as exc:
                if onerror == "skip":
                    print("Could not read file {}: {}".format(
                        gran, exc.args[0], file=sys.stderr))
                    continue
                else:
                    raise
            else:
                contents.append(cont)
        # retain type of first result, ordinary array of masked array
        return (numpy.ma.concatenate 
            if isinstance(contents[0], numpy.ma.MaskedArray)
            else numpy.concatenate)(contents)
#        return numpy.ma.concatenate(list(
#            self.read(f) for f in self.find_granules(start, end)))

    def read_all(self):
        """Read all data in one go.

        Warning: for some datasets, this may cause memory problems.
        """

        return self.read_period()
            
    @abc.abstractmethod
    def _read(self, f):
        """Read granule in file, low-level

        Shall return an ndarray with at least the fields lat, lon, time.

        :param str f: Path to file
        """

        raise NotImplementedError()

    @functools.lru_cache(maxsize=10)
    def read(self, f):
        """Read granule in file and do some other fixes

        Uses self._read.  Do not override, override _read instead.

        :param str f: Path to file
        """
        if isinstance(f, pathlib.PurePath):
            f = str(f)
        return self._read(f)


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
            MultiFileDataset.re_info contains all recognised fields.

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
            self._granule_start_times = shelve.open(str(self.basedir /
                self.granule_cache_file), protocol=4)
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
            year=dt.year, month=dt.month, day=dt.day))

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

        if res == "year":
            year = d.year
            while datetime.date(year, 1, 1) < d_end:
                yield (dict(year=year),
                    pathlib.Path(str(self.basedir / self.subdir).format(year=year)))
                year += 1
        elif res == "month":
            year = d.year
            month = d.month
            while datetime.date(year, month, 1) < d_end:
                yield (dict(year=year, month=month),
                    pathlib.Path(str(self.basedir / self.subdir).format(year=year, month=month)))
                if month == 12:
                    year += 1
                    month = 1
                else:
                    month += 1
        elif res == "day":
            while d < d_end:
                yield (dict(year=year, month=month, day=day),
                    pathlib.Path(str(self.basedir / self.subdir).format(
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
        for (timeinfo, subdir) in self.iterate_subdirs(d_start, d_end):
            if subdir.exists() and subdir.is_dir():
                for child in subdir.iterdir():
                    m = self._re.fullmatch(child.name)
                    if m is not None:
                        (g_start, g_end) = self.get_times_for_granule(child,
                            **timeinfo)
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

    def get_times_for_granule(self, p,
            **kwargs):
        """For granule stored in `path`, get start and end times.

        May take hints for year, month, day, hour, minute, second, and
        their endings, according to self.date_fields
        """
        if str(p) in self._granule_start_times.keys():
            (start, end) = self._granule_start_times[str(p)]
        else:
            m = self._re.fullmatch(p.name)
            gd = m.groupdict()
            if (any(f in gd.keys() for f in self.datefields) and
                (any(f in gd.keys() for f in {x+"end" for x in self.datefields})
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
                    end_date[0] = self._getyear(gd, "year_end", year)
                    end = datetime.datetime(**end_date)
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
        """
        raise ValueError(
            ("To determine starting and end-times for a {0} dataset, "
             "I need to read the file.  However, {0} has not implemented the "
             "get_time_from_granule_contents method.".format(
                type(self).__name__)))

class SingleMeasurementPerFileDataset(MultiFileDataset):
    """Represents datasets where each file contains one measurement.

    An example of this would be ACE, or some RO datasets.
    """

    @abc.abstractmethod
    def read_single(self, p):
        """Read a single measurement from a single file.

        Shall take one argument (a path object) and return a tuple with
        (header, measurement).  The header shall contain information like
        latitude, longitude, time.
        """

    def _read(self, p):
        """Reads a single measurement converted to ndarray
        """

        (head, body) = self.read_single(p)

        dt = [s+body.shape for s in body.dtype.descr]
        dt.extend([("lat", "f8"), ("lon", "f8"), ("time", "M8[s]")])
        # This fails. https://github.com/numpy/numpy/issues/4583
        #D = numpy.ma.empty(1, dt)
        D = numpy.empty(1, dt)

        for nm in body.dtype.names:
            D[nm] = body[nm]

        for nm in {"lat", "lon", "time"}:
            D[nm] = head[nm]

        return D

class HomemadeDataset(MultiFileDataset):
    """For any dataset created by pyatmlab.

    No content yet.
    """
    # dummy implementation for abstract methods, so that I can test other
    # things

    def _read(self, f):
        raise NotImplementedError()

    def find_granules(self, start, end):
        raise StopIteration()
