"""Ground-based remote sensing datasets
"""

import string
import datetime
import ast
import itertools
import logging

import numpy

import pytz
try:
    import pyhdf.SD
except ImportError:
    logging.error("Failed to import pyhdf, some readers will fail")

from .. import dataset
from .. import geo
from .. import io

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
            vscal = io.collect_values(fp, nv, numpy.float32)

            # fillers for each record
            vmiss = io.collect_values(fp, nv, numpy.float32)

            # next N=48 lines contain variable names
            varnames = [fp.readline().strip() for _ in range(nv)]

            # the same for aux variables
            nauxv = int(fp.readline().strip())
            ascal = io.collect_values(fp, nauxv, numpy.float32)
            amiss = io.collect_values(fp, nauxv, numpy.float32)
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
                    v = io.collect_values(fp, 1+nauxv+nv, self.dtype)
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


class Eureka_PRL_CH4_HDF(dataset.MultiFileDataset, dataset.ProfileDataset):
    # NOTE: there is a bug in older versions of Python-hdf4 that causes it to
    # crash on some HDF files.  The Eureka Bruker CH4 HDF files happen to
    # have a crash on
    # CH4.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_UNCERTAINTY.SYSTEMATIC.COVARIANCE
    # The bug was fixed 2014-11-26

    # For PEARL collocating with TANSO, 80% of profiles have >50%
    # sensitivity between 3.9 and 29.9 km, and 50% have >50% sensitivity
    # between 3.9 km and 31.5 km
    # For PEARL collocation with ACE, 80% of profiles have >50%
    # sensitivity between 3.9 km and 26.9 km, and 50% have >50%
    # sensitivity between 0.8 km and 29.9 km

    A_needs_converting = False
    A_needs_swapping = True

    range = (3.5e3, 30e3)

    # It does appear timezones are now fixed
    #timezone = "Etc/GMT-5"
    timezone = "UTC"

    # specific field for Eureka PEARL HDF
    altitude_boundaries = None

    aliases = {"CH4_profile": "CH4_VMR",
        "S_CH4_profile": "CH4_SA_random",
        "ap": "CH4_ap",
        "ak": "CH4_ak"}

    _nlev = 47
    _dtp = [("time", "datetime64[s]"),
            ("lat", "f4"),
            ("lon", "f4"),
            ("p0", "f4"),
            ("z0", "f4"),
            ("T0", "f4"),
            ("z", "f4", _nlev),
            ("p", "f4", _nlev),
            ("T", "f4", _nlev),
            ("CH4_VMR", "f4", _nlev),
            ("CH4_ak", "f4", (_nlev, _nlev)),
            ("CH4_ap", "f4", _nlev),
            ("CH4_SA_random", "f4", (_nlev, _nlev)),
            ("CH4_SA_system", "f4", (_nlev, _nlev)),
            ("CH4_pc", "f4", _nlev),
            ("CH4_ap_pc", "f4", _nlev),
            ("CH4_tc", "f4"),
            ("CH4_ap_tc", "f4"),
            ("CH4_ak_tc", "f4", _nlev),
            ("delta_CH4_tc_random", "f4"),
            ("delta_CH4_tc_system", "f4"),
            ("sza", "f4"),
            ("saa", "f4"),
            ("H2O_VMR", "f4", _nlev)
            ]

    _trans = {"DATETIME": "time",
        "LATITUDE.INSTRUMENT": "lat",
        "LONGITUDE.INSTRUMENT": "lon",
        "ALTITUDE.INSTRUMENT": "z0",
        "SURFACE.PRESSURE_INDEPENDENT": "p0",
        "SURFACE.TEMPERATURE_INDEPENDENT": "T0",
        "ALTITUDE": "z",
        "PRESSURE_INDEPENDENT": "p",
        "TEMPERATURE_INDEPENDENT": "T",
        "CH4.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR": "CH4_VMR",
        "CH4.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_APRIORI": "CH4_ap",
        "CH4.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_AVK": "CH4_ak",
        "CH4.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_UNCERTAINTY.RANDOM.COVARIANCE":
            "CH4_SA_random",
        "CH4.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR_UNCERTAINTY.SYSTEMATIC.COVARIANCE":
           "CH4_SA_system",
        "CH4.COLUMN.PARTIAL_ABSORPTION.SOLAR": "CH4_pc",
        "CH4.COLUMN.PARTIAL_ABSORPTION.SOLAR_APRIORI": "CH4_ap_pc",
        "CH4.COLUMN_ABSORPTION.SOLAR": "CH4_tc",
        "CH4.COLUMN_ABSORPTION.SOLAR_APRIORI": "CH4_ap_tc",
        "CH4.COLUMN_ABSORPTION.SOLAR_AVK": "CH4_ak_tc",
        "CH4.COLUMN_ABSORPTION.SOLAR_UNCERTAINTY.RANDOM.STANDARD":
            "delta_CH4_tc_random",
        "CH4.COLUMN_ABSORPTION.SOLAR_UNCERTAINTY.SYSTEMATIC.STANDARD":
            "delta_CH4_tc_system",
        "ANGLE.SOLAR_ZENITH.ASTRONOMICAL": "sza",
        "ANGLE.SOLAR_AZIMUTH": "saa",
        "H2O.MIXING.RATIO.VOLUME_ABSORPTION.SOLAR": "H2O_VMR"}
    #_invtrans = {v: k for k, v in _trans.items()}
    _fld_copy_scal = {"lat", "lon", "p0", "T0", "z0", "CH4_tc",
        "CH4_ap_tc", "sza", "saa",
        "delta_CH4_tc_random", "delta_CH4_tc_system"}
    _fld_copy_onevec = {"z"}
    _fld_copy_vec = {"p", "T", "CH4_VMR", "CH4_pc", "CH4_ap_pc",
        "CH4_ak_tc", "H2O_VMR", "CH4_ap"}
    _fld_copy_mat = {"CH4_ak", "CH4_SA_system", "CH4_SA_random"}
    _fld_copy_not = {"time"} # specially treated
    def _read(self, path=None, fields="all"):
        """Read granule"""
        if path is None:
            path = self.srcfile

        sd = pyhdf.SD.SD(path)
        (n_ds, n_attr) = sd.info()

        n_elem = sd.select(0).info()[2]
        M = numpy.empty(shape=(n_elem,), dtype=self._dtp)
        dtm_mjd2k = sd.select(sd.nametoindex("DATETIME")).get()
        dtm_mjd2k_s = dtm_mjd2k * 86400
        M["time"] = (numpy.datetime64(datetime.datetime(2000, 1,
                                                         1, 0, 0, 0)) +
                         dtm_mjd2k_s.astype("timedelta64[s]"))
        # It appears timez are still in UTC-0500.  Correct accordingly.
        tz = pytz.timezone(self.timezone)
        M["time"] = [t + numpy.timedelta64(tz.utcoffset(t)) for t in M["time"]]
        # check direction as I may need to turnaround the data
        z = sd.select(sd.nametoindex("ALTITUDE")).get()
        direc = int(numpy.sign(z[-1]-z[0]))
        # simple copy
        for (full, short) in self._trans.items():
            sds = sd.select(sd.nametoindex(full))
            if short in self._fld_copy_scal:
                M[short] = sds.get()
            elif short in self._fld_copy_onevec:
                M[short] = sds.get()[::direc]
            elif short in self._fld_copy_vec:
                M[short] = sds.get()[:, ::direc]
            elif short in self._fld_copy_mat:
                M[short] = sds.get()[:, :, ::direc][:, ::direc, :]
            elif short in self._fld_copy_not:
                pass
            else:
                logging.error("Don't know where to put {} ({})!".format(full, short))

            (offset, factor, unit) = sds.attributes()["VAR_SI_CONVERSION"].split(';')
            factor = ast.literal_eval(factor)
            if not unit in {"rad", "mol m-2", "s"}:
                M[short] *= factor

        self.altitude_boundaries = sd.select(
            sd.nametoindex("ALTITUDE.BOUNDARIES")).get()

        # Now done above
        #M["p0"] *= HECTO
        #M["p"] *= HECTO
        #M["z"] *= KILO
        #M["z0"] *= KILO
        #M["CH4_VMR"] *= PPM

#        for i in range(n_ds):
#            (nm, rank, dims, tp, n_attr) = sd.select(i).info()
#            if (rank==1 and dims==n_elem):
#                dtp.append((nm, "<f4"))
#            elif (rank>1 and dims[0]==n_elem):
#                dtp.append((nm, "<f4", dims[1:]))

        return M
