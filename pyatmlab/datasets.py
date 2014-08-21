
import string
import datetime
import ast
import itertools
import calendar
import collections

import numpy
import scipy.interpolate

import h5py
from . import dataset
from . import physics
from . import math
from . import geo
from .constants import ppm as PPM, hecto as HECTO

class TansoFTSBase(dataset.ProfileDataset):
    """Applicable to both Tanso FTS versions
    """
    p_for_T_profile = numpy.array([1000, 975, 950, 925, 900, 850, 800,
        700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20,
        10])*HECTO # hPa -> Pa

    A_needs_converting = False

    def _read_common(self, h5f):
        """From open h5 file, read some fields common to both tanso versions
        """

        D = collections.OrderedDict()
        time_raw = h5f["scanAttribute"]["time"]
        # force UTC time
        D["time"] = numpy.array([numpy.datetime64(time_raw[i].decode('ascii')+'Z')
            for i in range(time_raw.size)], dtype="datetime64[us]")
        D["lat"] = h5f["Data"]["geolocation"]["latitude"]
        D["lon"] = h5f["Data"]["geolocation"]["longitude"]
        D["z0"] = h5f["Data"]["geolocation"]["height"]
        D["T"] = h5f["scanAttribute"]["referenceData"]["temperatureProfile"]
        D["h2o"] = h5f["scanAttribute"]["referenceData"]["waterVaporProfile"][...]*PPM
        D["p0"] = h5f["scanAttribute"]["referenceData"]["surfacePressure"][...] * HECTO
        D["id"] = h5f["scanAttribute"]["scanID"]

        return D

class TansoFTSv10x(dataset.MultiFileDataset, TansoFTSBase):
    """Tanso FTS v1.00, 1.01
    """

    re = r"GOSATTFTS(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_02P02TV010[01]R\d{6}[0-9A-F]{5}\.h5"
    aliases = {"CH4_profile": "ch4_profile"}
    n_prof = "p"

    # NOTE: For Tanso FTS v1.0x, there are THREE pressure profiles:
    #
    # - CH4 has 23 levels: 1165.9, 857.7, 735.6, 631.0, 541.2, 464.2,
    #   398.1, 341.4, 287.3, 237.1, 195.7, 161.6, 133.4, 100.0, 75.0, 51.1,
    #   34.8, 23.7, 16.2, 10.0, 5.6, 1.0, 0.1 hPa (GOSAT Product Description)
    #
    # - For each retrieval, the average pressure in each of 22 layers is
    #   stored (this one varies)
    #
    # - For temperature and water vapour, there are 21 vertical levels:
    #   1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250,
    #   200, 150, 100, 70, 50, 30, 20, 10 hPa.
    #
    # To get CH4 on z-grid rather than p-grid, need to convert p-grid to
    # z-grid.  This needs temperature and water vapour profiles.  Either
    # interpolate *and* extrapolate z-grid onto retrieval grid, or
    # interpolate retrieval *and* averaging kernel onto T/Q-grid.  Prefer
    # the former because ak-interpolation I found is linear which is OK
    # for z but not for p?
    #

    # implementation of abstract methods
    def _read(self, path, fields="all"):
        D = collections.OrderedDict()
        with h5py.File(path, 'r') as h5f:
            D.update(self._read_common(h5f))
            D["ch4_profile"] = (h5f["Data"]["originalProfile"]
                ["CH4Profile"][:]*PPM)
            D["ch4_profile_e"] = (h5f["Data"]["originalProfile"]
                ["CH4ProfileError"][:]*PPM)
            D["ch4_ak"] = h5f["Data"]["originalProfile"]["CH4AveragingKernelMatrix"]
            D["p"] = h5f["Data"]["originalProfile"]["pressure"][...]*HECTO
            D["z0"] = h5f["Data"]["geolocation"]["height"]
            A = numpy.empty(shape=D["time"].size,
                dtype=[(k, D[k].dtype, D[k].shape[1:]) for k in D.keys()])
            for k in D.keys():
                A[k] = D[k][:]
        return A if fields=="all" else A[fields]

    def get_z(self, obj):
        try:
            return super().get_z(obj)
        except IndexError:
            pass # parent failed, continue here
        # See comment near top of this class; different p-grids, different
        # z-grids.  Want z-grid corresponding to CH4-profile.
        p_for_T = self.p_for_T_profile # down up to 1 kPa
        p_for_CH4 = obj["p"] # down to ~50 Pa
        # inter- and extra-polate T and h2o onto the p_for_CH4 grid
        # This introduces an extrapolation error but between 1 kPa and 50
        # Pa I don't really care, should be insignificant.

        # highest level is lowest pressure i.e. lower bound, and vice
        # versa.  Therefore reserve p, f(p) so p is going up (increasing
        # pressure)
        xb = min(p_for_CH4.min(), p_for_T.min())
        # including obj["p0"] here guarantees reference pressure/altitude
        # always included in profile, satisfying Patricks algorithm
        # (p2z_hydrostatic)
        xe = max(p_for_CH4.max(), p_for_T.max(), obj["p0"])
        tck_T = scipy.interpolate.splrep(p_for_T[::-1], obj["T"][::-1],
            xb=xb,
            xe=xe,
            k=3)
        tck_h2o = scipy.interpolate.splrep(p_for_T[::-1], obj["h2o"][::-1],
            xb=xb,
            xe=xe,
            k=3)

        # extrapolate, add extra pressure points.  Make sure that
        # p0 >= p_grid[0], also include 100 kPa as this was in the
        # original grid.  For completeness, merge both pressure grids,
        # then "unpack" later.  Sort by index so I can later extract the
        # parts that were belonging to p_for_CH4.
        #
        # FIXME: Do I really need to "unpack" it again?  Or just stick
        # with the merged grid?!
        p_full = []
        p_full.append(obj["p0"])
        p_full.extend(p_for_T)
        p_for_ch4_i0 = len(p_full)
        p_full.extend(p_for_CH4)
        p_for_ch4_i1 = len(p_full)
        p_full = numpy.array(p_full, dtype="f4")
        p_inds = numpy.argsort(p_full)[::-1] # high to low pressure

        p_full_sorted = p_full[p_inds]
        p_full_sorted = numpy.ma.masked_less(p_full_sorted, 0)
        T_retgrid = numpy.ma.masked_array(
            scipy.interpolate.splev(p_full_sorted, tck_T),
            p_full_sorted.mask)
        q_retgrid = numpy.ma.masked_array(
            scipy.interpolate.splev(p_full_sorted, tck_h2o),
            p_full_sorted.mask)

        # numpy warns even though I masked the invalid values... please
        # don't :(
        ed = numpy.seterr(invalid="ignore", divide="ignore")
        z_extp = physics.p2z_hydrostatic(
            p_full_sorted, T_retgrid, q_retgrid,
            obj["p0"], obj["z0"], obj["lat"], -1)
        numpy.seterr(**ed)
        
        # return only elements belonging to p_for_CH4
        # this yields the "inverse" sorting indices to get back the
        # original order, but only for those that were part of p_for_CH4
        p_i_was_CH4 = ((p_inds>=p_for_ch4_i0) & (p_inds<p_for_ch4_i1)).nonzero()[0]

        return z_extp[p_i_was_CH4].data
#        inv_inds[-len(p_for_CH4):]]

#        z_for_T = physics.p2z_hydrostatic(
#            self.p_for_T_profile, obj["T"], obj["h2o"], 
#            obj["p0"], obj["z0"], obj["lat"], -1)

    def get_time_from_granule_contents(self, p):
        M = self.read(p, fields={"time"})
        return (M["time"].min(), M["time"].max())
            

class TansoFTSv001(dataset.SingleFileDataset, TansoFTSBase):
    # Retained for historical reasons

#    start_date = datetime.datetime(2010, 3, 23, 2, 24, 54, 210)
#    end_date = datetime.datetime(2010, 10, 31, 20, 34, 50, 814)
#    srcfile = ("/home/gerrit/sshfs/glacier/data/1/gholl/data/1403050001/"
#               "GOSATTFTS20100316_02P02TV0001R14030500010.h5")
#    name = "GOSAT Tanso FTS"

    p_for_interp_profile = numpy.array([1000, 700, 500, 300, 100, 50,
        10])*HECTO
    aliases = {"CH4_profile": "ch4_profile_raw"}

    @classmethod
    def vmr_to_column_density(cls, data):
        """Convert VMR to column density

        Currently hardcoded to CH4 profiles and the dtype by this classes'
        read method.

        Returns number column density and its error

        Note that a small error may be introduced if profiles do not go
        down to the reference provided by (z0, p0), because the p2z
        function needs a reference inside the p-profile.  This is
        mitigated by adding a dummy level with a dummy temperature, then
        removing this after the z-profile is calculated.  This has a minor
        (<0.1%) effect on the remaining levels.
        """
        # temperature interpolated on each pressure grid
        ncd = numpy.empty(shape=data.shape, dtype="f8")
        ncd_e = numpy.empty_like(ncd)

        for n in range(data.size):
            if data["T"].mask.size == 1:
                T = data["T"][n, ::-1].data
            else:
                T = data["T"][n, ::-1].data[~data["T"][n, ::-1].mask]

            if data["h2o"].mask.size == 1:
                h2o = data["h2o"][n, ::-1].data
            else:
                h2o = data["h2o"][n, ::-1].data[~data["h2o"][n, ::-1].mask]

            T_interpolator = scipy.interpolate.interp1d(
                cls.p_for_T_profile[::-1], T)# should be increasing

            h2o_interpolator = scipy.interpolate.interp1d(
                cls.p_for_T_profile[::-1], h2o)
            
            p = data["p_raw"][n, :].data[~data["p_raw"][n, :].mask]*HECTO

            T_interp = T_interpolator(p)
            h2o_interp = h2o_interpolator(p)

            nd = physics.vmr2nd(
                data["ch4_profile_raw"][n, :].data[
                    ~data["ch4_profile_raw"][n, :].mask],#*PPM,
                T_interp, p)

            nd_e = physics.vmr2nd(
                data["ch4_profile_raw_e"][n, :].data[
                    ~data["ch4_profile_raw_e"][n, :].mask],#*PPM,
                T_interp, p)

            #z = physics.p2z_oversimplified(p)
            if data["p0"][n] > p[0]:
                # need to add dummy p[0] < p to satisfy p2z
                #
                # this is inaccurate, but has little effect on the
                # *relative* location of other levels (<1 m), so it is an
                # acceptable source of error
                dummy = True
                pp = numpy.r_[data["p0"][n], p]
                Tp = numpy.r_[287, T_interp]
                # although putting h2o-surface to 0 is an EXTREMELY
                # bad approximation, it /still/ doesn't matter for the
                # relative thicknesses higher up
                h2op = numpy.r_[h2o_interp[0], h2o_interp]
            else:
                dummy = False
                (pp, Tp, h2op) = (p, T_interp, h2o_interp)
            #z = physics.p2z_hydrostatic(p, T_interp, h2o_interp,
            z = physics.p2z_hydrostatic(pp, Tp, h2op*PPM,
                p0=data["p0"][n],
                z0=data["z0"][n],
                lat=data["lat"][n])[dummy:]

            ncd[n] = math.integrate_with_height(z, nd)
            ncd_e[n] = math.integrate_with_height(z, nd_e)

        return (ncd, ncd_e)


    # implementation of abstract methods

    def _read(self, path=None, fields="all"):
        """Read Tanso FTS granule.  Currently hardcoded for CH4 raw&interp.
        
        """
        if path is None:
            path = self.srcfile

        with h5py.File(path, 'r') as h5f:
            D = collections.OrderedDict()
            D.update(self._read_common(h5f))
            p = h5f["Data"]["interpolatedProfile"]["pressure"][:]
            p *= HECTO # Pa -> hPa
            if self.p_for_interp_profile is not None:
                if not (self.p_for_interp_profile == p).all():
                    raise ValueError("Found inconsistent pressure"
                            " profiles!")
            else:
                self.p_for_interp_profile = p
            #D["p"] = h5f["Data"]["originalProfile"]["pressure"]
            D["ch4_profile_interp"] = (h5f["Data"]["interpolatedProfile"]
                ["CH4Profile"][:]*PPM)
            D["ch4_profile_raw"] = (h5f["Data"]["originalProfile"]
                ["CH4Profile"][:]*PPM)
            D["ch4_profile_raw_e"] = (h5f["Data"]["originalProfile"]
                ["CH4ProfileError"][:]*PPM)
            D["p_raw"] = h5f["Data"]["originalProfile"]["pressure"]

            A = numpy.empty(shape=time_raw.size,
                dtype=[(k, D[k].dtype, D[k].shape[1:]) for k in D.keys()])
            for k in D.keys():
                A[k] = D[k][:]
            # Unfortunately, masked arrays are buggy
#            A = A.view(numpy.ma.MaskedArray)
#            for k in {"ch4_profile_interp", "ch4_profile_raw",
#                      "ch4_profile_raw_e", "p_raw", "T"}:
#                A.mask[k][A.data[k]<0] = True

        return A if fields=="all" else A[fields]

    def get_z(self, meas): 
        try:
            return super().get_z(meas)
        except IndexError:
            pass # parent failed, continue here
#            if data["T"].mask.size == 1:
#                T = data["T"][n, ::-1].data
#            else:
#                T = data["T"][n, ::-1].data[~data["T"][n, ::-1].mask]
#
#            if data["h2o"].mask.size == 1:
#                h2o = data["h2o"][n, ::-1].data
#            else:
#                h2o = data["h2o"][n, ::-1].data[~data["h2o"][n, ::-1].mask]


        # Can't use masked arrays due to bug:
        # https://github.com/numpy/numpy/issues/2972
        # Use workaround instead.
        if isinstance(meas, numpy.ma.core.MaskedArray):
            meas = meas.data

        valid = meas["p_raw"] > 0

        T_interpolator = scipy.interpolate.interp1d(
            self.p_for_T_profile[::-1],
            meas["T"][::-1])# should be increasing

        h2o_interpolator = scipy.interpolate.interp1d(
            self.p_for_T_profile[::-1],
            meas["h2o"][::-1])
        
#            p = data["p_raw"][n, :].data[~data["p_raw"][n, :].mask]*HECTO
        p = meas["p_raw"] * HECTO
        p = p[valid]

        T_interp = T_interpolator(p)
        h2o_interp = h2o_interpolator(p)

#            nd = physics.vmr2nd(
#                data["ch4_profile_raw"][n, :].data[
#                    ~data["ch4_profile_raw"][n, :].mask]*PPM,
#                T_interp, p)
#
#            nd_e = physics.vmr2nd(
#                data["ch4_profile_raw_e"][n, :].data[
#                    ~data["ch4_profile_raw_e"][n, :].mask]*PPM,
#                T_interp, p)

        #z = physics.p2z_oversimplified(p)
        if meas["p0"] > p[0]:
            # need to add dummy p[0] < p to satisfy p2z
            #
            # this is inaccurate, but has little effect on the
            # *relative* location of other levels (<1 m), so it is an
            # acceptable source of error
            dummy = True
            pp = numpy.hstack((meas["p0"], p))
            Tp = numpy.hstack((287, T_interp))
            # although putting h2o-surface to a low value is an EXTREMELY
            # bad approximation, it /still/ doesn't matter for the
            # relative thicknesses higher up
            h2op = numpy.hstack((h2o_interp[0], h2o_interp))
        else:
            dummy = False
            (pp, Tp, h2op) = (p, T_interp, h2o_interp)
        #z = physics.p2z_hydrostatic(p, T_interp, h2o_interp,
        z = physics.p2z_hydrostatic(pp, Tp, h2op*PPM,
            p0=meas["p0"],
            z0=meas["z0"],
            lat=meas["lat"])[dummy:]
        return z


class NDACCAmes(dataset.MultiFileDataset):
    """NDACC Ames-style file

    Documented at http://www.ndsc.ncep.noaa.gov/data/formats/
    """

    re = r"eutc(?P<year>\d{2})(?P<month>\d{2})\.sgf"
    name = "PEARL Bruker IFS"

    type_core = [(spec + "_" + tp, numpy.uint16 if tp=="n" else numpy.float32)
        for spec in ("O3", "HCL", "HF", "HNO3", "CLONO2", "N2O", "CO", "CH4")
        for tp in ("total", "total_e", "ss", "ss_e") +
            (("ts", "ts_e") if spec in {"O3", "N2O", "CO", "CH4"} else ()) +
            ("n",)]
    dtype = (
        [("time_yearfrac", numpy.float32)] +
        [("doy", numpy.uint16), ("year", numpy.uint16), 
         ("month", numpy.uint8), ("day", numpy.uint8),
         ("lat", numpy.float32), ("lon", numpy.float32),
         ("elev", numpy.float32)] +
        type_core)

    def get_time_from_granule_contents(self, path):
        # Read the entire granule, as the end information is sometimes
        # incorrect
        M = self.read(path)
        return (min(M["time"]).item(), max(M["time"]).item())
#        with open(path, 'rt', encoding="ascii") as fp:
#            for _ in range(7):
#                fp.readline()
#            y1, m1, d1, y2, m2, d2 = tuple(
#                int(d) for d in fp.readline().split())
#        return (datetime.datetime(y1, m1, d1, 0, 0, 0),
#                datetime.datetime(y2, m2, d2, 23, 59, 59))
             
    def _read(self, path, fields="all"):
        """Read Bruker data in NDACC Gaines format

        Returns a masked array
        """

        with open(path, 'rt', encoding='ascii') as fp:
            # first 10 lines not relevant for now
            header = ''.join(fp.readline() for _ in range(10))

            # no. of measurements per record
            nv = int(fp.readline().strip())

            # factors for each record
            vscal = collect_values(fp, nv, numpy.float32)

            # fillers for each record
            vmiss = collect_values(fp, nv, numpy.float32)

            # next N=48 lines contain variable names
            varnames = [fp.readline().strip() for _ in range(nv)]

            # the same for aux variables
            nauxv = int(fp.readline().strip())
            ascal = collect_values(fp, nauxv, numpy.float32)
            amiss = collect_values(fp, nauxv, numpy.float32)
            varnames_aux = [fp.readline().strip() for _ in range(nauxv)]

            # special comments
            nscoml = int(fp.readline().strip())
            scom = ''.join(fp.readline() for _ in range(nscoml))

            # normal comments
            nncoml = int(fp.readline().strip())
            ncom = ''.join(fp.readline() for _ in range(nncoml))

            # and now the data!
            # ...which needs a prettier dtype
            L = []
            while True:
                try:
                    v = collect_values(fp, 1+nauxv+nv, self.dtype)
                except EOFError:
                    break
                else:
                    dt = numpy.datetime64(datetime.date(v["year"],
                        v["month"], v["day"]), 'D').astype("<M8[us]")
                    new = numpy.empty(dtype=[("time",
                        dt.dtype)]+v.dtype.descr, shape=())
                    new["time"] = dt
                    for field in v.dtype.names:
                        new[field] = v[field]
                    L.append(new if fields=="all" else new[fields])

        # now I have an array with fractional years as the time-axis.
        # I want to have datetime64
        A = numpy.array(L)
        #dts = [datetime.datetime(numpy.floor(d), 1, 1, 0, 0, 0) +
        #       datetime.timedelta(days=(365+calendar.isleap(2010))*(d-numpy.floor(d)))
        #       for d in A["time"]]
        #dtn = dict(A.dtype.fields)
        #dtn["time"] = (numpy.dtype("datetime64[us]"), 0)
        #AA = numpy.empty(dtype=dtn, shape=A.size)
        #AA["time"] = dts
        #for fld in set(A.dtype.names) - {'time'}:
        #    AA[fld] = A[fld]
        #return AA

        # apply masks and factors
        # No, rather not.  MaskedArrays are buggy.
#        M = A.view(numpy.ma.MaskedArray)
        M = A
        for (i, field) in enumerate(self.type_core):
#            M.mask[field[0]] = A[field[0]] == vmiss[i]
            A[field[0]] *= vscal[i]

        # lats are secretly wrongly typed
        M["lat"] /= 100
        M["lon"] /= 100

        # and lons want to be in -180, 180
        M["lon"] = geo.shift_longitudes(M["lon"], (-180, 180))
        return M


class ACEFTS(dataset.SingleMeasurementPerFileDataset,
             dataset.ProfileDataset):
    """SCISAT Atmospheric Chemistry Experiment FTS
    """
#    basedir = "/home/gerrit/sshfs/glacier/data/1/gholl/data/ACE"
    subdir = "{year:04d}-{month:02d}"
    re = r"(?P<type>s[sr])(?P<orbit>\d{5})v(?P<version>\d\.\d)\.asc"
    _time_format = "%Y-%m-%d %H:%M:%S"
    aliases = {"CH4_profile": "CH4"}
    filename_fields = {"orbit": "u4", "version": "S3"}
    unique_fields = {"orbit"}
    n_prof = "z"

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
                list(zip(names, ["f8"]*len(names))))

            for (n, line) in enumerate(fp):
                # why does this not work?
                # http://stackoverflow.com/q/22865877/974555
                #D[names][n] = tuple(float(f) for f in line.split())
                vals = tuple(float(f) for f in line.split())
                for (i, name) in enumerate(names):
                    D[name][n] = vals[i]

        # km -> m
        D["z"] *= 1e3

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
            

def collect_values(fp, N, dtp):
    """Collect N values from stream

    Must be contained in exact number of lines.
    This will advance the stream forward by the number of lines found to
    contain N numeric values, and return an ndarray of type tp containing
    those.

    :param file fp: Stream
    :param int N: Total no. expected values
    :param dtype tp: dtype for values
    :returns: ndarray of type dtype with values found in file
    """ 
    L = []
    while len(L) < N:
        line = fp.readline()
        if line == "":
            raise EOFError("File ended prematurely")
        L.extend(ast.literal_eval(f) for f in line.strip().split())
    if len(L) != N:
        raise ValueError("Unexpected number of values.  Expected:"
            "{N:d}.  Got: {L}".format(N=N, L=len(L)))
    if numpy.dtype(dtp).isbuiltin == 0:
        flat_dtp = numpy.dtype(list(zip(
            (''.join(s) for s in itertools.product(string.ascii_letters, repeat=2)),
            (item for sublist in 
            [[x[1]]*(numpy.product(x[2]) if len(x)>2 else 1) for x in numpy.dtype(dtp).descr]
                    for item in sublist))))
        return numpy.array(tuple(L), dtype=flat_dtp).view(dtp)
    else:
        return numpy.array(L, dtype=dtp)


