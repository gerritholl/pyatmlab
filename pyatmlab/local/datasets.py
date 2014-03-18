
import string
import datetime
import ast
import itertools
import calendar

import numpy

import h5py
from .. import dataset

class TansoFTS(dataset.SingleFileDataset):

    start_date = datetime.datetime(2010, 3, 23, 2, 24, 54, 210)
    end_date = datetime.datetime(2010, 10, 31, 20, 34, 50, 814)
    srcfile = ("/home/gerrit/sshfs/glacier/data/1/gholl/data/1403050001/"
               "GOSATTFTS20100316_02P02TV0001R14030500010.h5")
    name = "GOSAT Tanso FTS"

    p_for_T_profile = numpy.array([1000, 975, 950, 925, 900, 850, 800,
        700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20,
        10])*100 # hPa -> Pa
    p_for_interp_profile = None

    # implementation of abstract methods

    def read(self, path=None):
        """Read Tanso FTP granule.  Currently hardcoded for CH4 raw profiles.
        
        """
        if path is None:
            path = self.srcfile

        with h5py.File(path, 'r') as h5f:
            D = {}
            time_raw = h5f["scanAttribute"]["time"]
            # force UTC time
            D["time"] = numpy.array([numpy.datetime64(time_raw[i].decode('ascii')+'Z')
                for i in range(time_raw.size)], dtype="datetime64[us]")
            D["lat"] = h5f["Data"]["geolocation"]["latitude"]
            D["lon"] = h5f["Data"]["geolocation"]["longitude"]
            p = h5f["Data"]["interpolatedProfile"]["pressure"][:]
            p *= 100 # Pa -> hPa
            if self.p_for_interp_profile is not None:
                if not (self.p_for_interp_profile == p).all():
                    raise ValueError("Found inconsistent pressure"
                            " profiles!")
            else:
                self.p_for_interp_profile = p
            #D["p"] = h5f["Data"]["originalProfile"]["pressure"]
            D["ch4_profile"] = h5f["Data"]["interpolatedProfile"]["CH4Profile"]
            D["T"] = h5f["scanAttribute"]["referenceData"]["temperatureProfile"]


            A = numpy.empty(shape=time_raw.size,
                dtype=[(k, D[k].dtype, D[k].shape[1:]) for k in D.keys()])
            for k in D.keys():
                A[k] = D[k][:]
            A = A.view(numpy.ma.MaskedArray)
            A.mask["ch4_profile"][A.data["ch4_profile"]<0] = True
        return A

class NDACCGainesBruker(dataset.SingleFileDataset):
    """NDACC Gaines Bruker-style file

    Documented at http://www.ndsc.ncep.noaa.gov/data/formats/
    """

    start_date = datetime.datetime(2010, 2, 24, 0, 0, 0)
    end_date = datetime.datetime(2010, 10, 16, 0, 0, 0)
    srcfile = ("/home/gerrit/sshfs/glacier/data/1/gholl/data"
               "/BrukerIFS/eutc1001.sgf")
    name = "PEARL Bruker IFS"

#    dtype = [("time", "datetime64[us]", 1),
#             ("aux", numpy.uint32, 7),
#             ("var", numpy.uint32, 48)]))
#
    
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
             
    def read(self, path=None):
        """Read Bruker data in NDACC Gaines format

        Returns a masked array
        """

        if path is None:
            path = self.srcfile

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
                    L.append(new)

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
        M = A.view(numpy.ma.MaskedArray)
        for (i, field) in enumerate(self.type_core):
            M.mask[field[0]] = A[field[0]] == vmiss[i]
            A[field[0]] *= vscal[i]

        # lats are secretly wrongly typed
        M["lat"] /= 100
        M["lon"] /= 100
        return M



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


