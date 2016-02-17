
import datetime
import itertools
import collections
import logging

import numpy

from .. import dataset
from .. import constants


class ACEFTS(dataset.SingleMeasurementPerFileDataset,
             dataset.ProfileDataset):
    """SCISAT Atmospheric Chemistry Experiment FTS
    """
#    basedir = "/home/gerrit/sshfs/glacier/data/1/gholl/data/ACE"
    subdir = "{year:04d}-{month:02d}"
    re = r"(?P<type>s[sr])(?P<orbit>\d{5})v(?P<version>\d\.\d)\.asc"
    _time_format = "%Y-%m-%d %H:%M:%S"
    aliases = {"CH4_profile": "CH4",
        "delta_CH4_profile": "CH4_err",
        "p": "P_pa",
        "S_CH4_profile": "CH4_SA_fake"}
    filename_fields = {"orbit": "u4", "version": "U3", "type": "U2"}
    unique_fields = {"orbit", "type", "time"}
    n_prof = "z"
    range = (5e3, 150e3)

    @staticmethod
    def read_header(fp):
        """Read header from open file

        Should be opened at the beginning.  Will advance location from
        start of header to end of header.

        :param fp: File open at beginning of header
        :returns: Dictionary with header information
        """
        head = collections.OrderedDict()
        isempty = lambda line: not line.isspace()
        for line in itertools.takewhile(isempty, fp):
            k, v = line.split("|")
            head[k.strip()] = v.strip()
        if head == {}:
            raise dataset.InvalidFileError(
                "Unable to extract header from {0.name}.  Empty?".format(fp))
        return head

    def read_single(self, f, fields="all"):
        with open(f) as fp:
            head = self.read_header(fp)

            line = fp.readline()
            while line.isspace():
                line = fp.readline()

            names = line.replace("P (atm)", "P_atm").split()
            # numpy.ma.empty fails with datetime dtype
            # https://github.com/numpy/numpy/issues/4583
            #D = numpy.ma.empty((150,),
            D = numpy.empty((150,),
                list(zip(names, ["f4"]*len(names)))
                    + [("P_pa", "f4"), ("CH4_SA_fake", "f4", (150,))])

            for (n, line) in enumerate(fp):
                # why does this not work?
                # http://stackoverflow.com/q/22865877/974555
                #D[names][n] = tuple(float(f) for f in line.split())
                try:
                    vals = tuple(float(f) for f in line.split())
                except ValueError:
                    # raise InvalidFileError instead so I can catch more
                    # narrowly higher up in the stack
                    raise dataset.InvalidFileError("Unable to read content")
                for (i, name) in enumerate(names):
                    D[name][n] = vals[i]

        # km -> m
        D["z"] *= 1e3
        D["P_pa"] = D["P_atm"] * constants.atm

        # assume error covariance matrix to be diagonal
        # and convert std. error to variance.  Errors on flagged values
        # are 0.

        val = D["CH4_err"]>0
        D["CH4_SA_fake"] = numpy.diag(D["CH4_err"]**2)
        D["CH4_SA_fake"][:, ~val] = 0
        D["CH4_SA_fake"][~val, :] = 0

        head["lat"] = float(head["latitude"])
        head["lon"] = float(head["longitude"])
        # make sure lons are in (-180, 180)
        if head["lon"] < -180:
            head["lon"] += 360
        if head["lon"] > 180:
            head["lon"] -= 360

        # for time, strip off both incomplete timezone designation and
        # decimal part (truncating it to the nearest second)
        head["time"] = datetime.datetime.strptime(
            head["date"].split(".")[0].split("+")[0], self._time_format)

        return (head, D if fields=="all" else D[fields])
        
    def get_time_from_granule_contents(self, p):
        """Get time from granule contents.

        Takes str with path, returns two datetimes
        """
        with open(p) as f:
            head = self.read_header(f)
            # cut of "+00" part, datetime defaults to UTC and having only
            # hours is contrary to any standard, so strptime cannot handle
            # it
        return tuple(datetime.datetime.strptime(
            head[m + "_time"][:-3], self._time_format)
            for m in ("start", "end"))

    def get_z(self, meas):
        try:
            return super().get_z(meas)
        except IndexError:
            pass # parent failed, continue here
        m = meas["z"]
        if m[-1] < 150: # oops, still in km
            return m * 1e3
        else:
            return m

    def flag(self, arr):
        flagged = self.combine(arr, self.related["flags"])
        flnm = self.related["flags"].aliases["flag"]
        # See e-mail Patrick 2014-06-04
        with numpy.errstate(invalid="ignore"):
            badlev = flagged[flnm]>2
            badprof = ((flagged[flnm]>=4) & (flagged[flnm]<=6)).any(1)
        logging.info("Flagging {:d}/{:d} profiles and {:d}/{:d} levels".format(
            badprof.sum(), badprof.size, badlev.sum(), badlev.size))
        arr["CH4"][badlev] = numpy.nan
        arr["CH4"][badprof, :] = numpy.nan
        arr["CH4_err"][badlev] = numpy.nan
        arr["CH4_err"][badprof, :] = numpy.nan
        arr["CH4_SA_fake"][numpy.tile(badlev[:, :, numpy.newaxis], (1,1,150))] = numpy.nan
        arr["CH4_SA_fake"][badprof, :, :] = numpy.nan
        return arr
