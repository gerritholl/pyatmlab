"""Module containing classes abstracting datasets
"""

import abc
import pathlib
import re
import shelve
import string

import datetime
import numpy

class Dataset(metaclass=abc.ABCMeta):
    """Represents a dataset.

    This is an abstract class.  More specific subclasses are
    SingleFileDataset and MultiFileDataset.  Do not subclass Dataset
    directly.
    """

    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        if hasattr(self, k) or hasattr(type(self), k):
            object.__setattr__(self, k, v)
        else:
            raise AttributeError("Unknown attribute: {}. ".format(k))

    @abc.abstractmethod
    def granules_for_period(self, start=datetime.datetime.min,
                                  end=datetime.datetime.max):
        """Loop through all granules for indicated period.

        This is a generator that will loop through all granules from
        `start` to `end`, inclusive.

        :param datetime start: Starting datetime, defaults to any
        :param datetime end: Ending datetime, defaults to any
        """
        raise NotImplementedError()


    def read_period(self, start=datetime.datetime.min,
                          end=datetime.datetime.max):
        """Read all granules between start and end, in bulk.

        :param datetime start: Starting time, None for any time
        :param datetime end: Ending time, None for any time
        :returns: Masked array with all data in period.
        """

        # retain type of first result, ordinary array of masked array
        contents = list(
            self.read(f) for f in self.granules_for_period(start, end))
        return (numpy.ma.concatenate 
            if isinstance(contents[0], numpy.ma.MaskedArray)
            else numpy.concatenate)(contents)
#        return numpy.ma.concatenate(list(
#            self.read(f) for f in self.granules_for_period(start, end)))

    def read_all(self):
        """Read all data in one go.

        Warning: for some datasets, this may cause memory problems.
        """

        return self.read_period()
            
    @abc.abstractmethod
    def read(self, f):
        """Read granule in file.

        Shall return an ndarray with at least the fields lat, lon, time.

        :param str f: Path to file
        """

        raise NotImplementedError()

class SingleFileDataset(Dataset):
    """Represents a dataset where all measurements are in one file.
    """

    start_date = datetime.datetime.min
    end_date = datetime.datetime.max
    srcfile = None

    def granules_for_period(self, start=datetime.datetime.min,
                            end=datetime.datetime.max):
        if start < self.end_date and end > self.start_date:
            yield self.srcfile

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
            number, etc.  For time identification, the following fields
            are relevant::

                - year
                - month
                - day
                - hour
                - minute
                - second
                - year_end
                - month_end
                - day_end
                - hour_end
                - minute_end
                - second_end

            If any *_end fields are found, the ending time is equal to the
            beginning time with any *_end fields replaced.  If no *_end
            fields are found, the `granule_duration` attribute is used to
            determine the ending time.

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
                yield ((year,), pathlib.Path(str(self.basedir / self.subdir).format(year=year)))
                year += 1
        elif res == "month":
            year = d.year
            month = d.month
            while datetime.date(year, month, 1) < d_end:
                yield ((year, month), pathlib.Path(str(self.basedir / self.subdir).format(year=year, month=month)))
                if month == 12:
                    year += 1
                    month = 1
                else:
                    month += 1
        elif res == "day":
            while d < d_end:
                yield (year, month, day), pathlib.Path(str(self.basedir / self.subdir.format(
                    year=d.year, month=d.month, day=d.day)))
                d = d + datetime.timedelta(days=1)
          
    def granules_for_period(self, dt_start, dt_end):
        """Yield all granules/measurementfiles in period
        """

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
                        (g_start, g_end) = self.get_times_for_granule(child, *timeinfo)
                        if g_end > dt_start and g_start < dt_end:
                            yield str(child)
            
    def get_times_for_granule(self, p,
            year=None, month=None, day=None):
        """For granule stored in `path`, get start and end times.

        May take hints for year, month, day.
        """
        m = self._re.fullmatch(p.name)
        gd = m.groupdict()
        if any(f in gd.keys() for f in {"year", "month", "day", "hour",
                    "minute", "second"}):
            start = datetime.datetime(
                gd.get("year", year),
                gd.get("month", month),
                gd.get("day", day),
                gd.get("hour", 0),
                gd.get("minute", 0),
                gd.get("second", 0))
            if any(k.endswith("_end") for k in gd.keys()):
                end = datetime.datetime(
                    gd.get("year_end", start.year),
                    gd.get("month_end", start.month),
                    gd.get("day_end", start.day),
                    gd.get("hour_end", start.hour),
                    gd.get("minute_end", start.minute),
                    gd.get("second_end", start.second))
            elif self.granule_duration is not None:
                end = start + self.granule_duration
            else:
                raise ValueError("Unable to determine ending time. "
                    "Please set self.duration or adapt re to get relevant "
                    "times from filename.  See MultiFileDataset class "
                    "docstring for details.")
        else: # need to read file or get from cache
            # implementation depends on dataset
            if p in self._granule_start_times.keys():
                (start, end) = self._granule_start_times[p]
            else:
                (start, end) = self.get_time_from_granule_contents(p)
                self._granule_start_times[p] = (start, end)
        return (start, end)

    # not an abstract method because subclasses need to implement it /if
    # and only if/ starting times cannot be determined from the filename
    def get_time_from_granule_contents(self, p):
        raise NotImplementedError("Subclass must implement me!")

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

    def read(self, p):
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

class HomemadeDataset(Dataset):
    """For any dataset created by pyatmlab.

    No content yet.
    """
    # dummy implementation for abstract methods, so that I can test other
    # things

    def read(self, f):
        raise NotImplementedError()

    def granules_for_period(self, start, end):
        raise StopIteration()
