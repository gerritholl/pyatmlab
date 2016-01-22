"""Datasets for TOVS/ATOVS
"""

import io
import tempfile
import subprocess
import datetime
import logging
import gzip
import abc

import numpy

import netCDF4
import dateutil

try:
    import coda
except ImportError:
    logging.warn("Unable to import coda, won't read IASI EPS L1C")
    

from .. import dataset
from .. import tools
from .. import constants
from .. import physics
from .. import math as pamath
from .. import ureg
from .. import config

from . import _tovs_defs

class Radiometer:
    srf_dir = ""
    srf_backend_response = ""
    srf_backend_f = ""

class HIRS(dataset.MultiFileDataset, Radiometer):
    """High-resolution Infra-Red Sounder.

    This class can read HIRS l1b as published in the NOAA CLASS archive.

    Work in progress.

    TODO/FIXME:

    - What is the correct way to use the odd bit parity?  Information in
      NOAA KLM User's Guide pages 3-31 and 8-154, but I'm not sure how to
      apply it.
    """

    name = "hirs"
    format_definition_file = ""
    n_channels = 20
    n_calibchannels = 19
    n_minorframes = 64
    n_perline = 56
    count_start = 2
    count_end = 22
    

    def _read(self, path, fields="all", return_header=False,
                    apply_scale_factors=True, calibrate=True,
                    apply_flags=True):
        if path.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open
        with opener(str(path), 'rb') as f:
            self.seekhead(f)
            header_bytes = f.read(self.header_dtype.itemsize)
            header = numpy.frombuffer(header_bytes, self.header_dtype)
            scanlines_bytes = f.read()
            scanlines = numpy.frombuffer(scanlines_bytes, self.line_dtype)
        n_lines = header["hrs_h_scnlin"][0]
        if fields != "all":
            scanlines = scanlines[fields]
        if apply_scale_factors:
            (header, scanlines) = self._apply_scale_factors(header, scanlines)
        if calibrate:
            if not apply_scale_factors:
                raise ValueError("Can't calibrate if not also applying"
                                 " scale factors!")
            lat = scanlines["hrs_pos"][:, ::2]
            lon = scanlines["hrs_pos"][:, 1::2]

            cc = scanlines["hrs_calcof"].reshape(n_lines, self.n_channels, 
                    self.line_dtype["hrs_calcof"].shape[0]//self.n_channels)
            cc = self.get_cc(scanlines)
            cc = cc[:, numpy.argsort(self.channel_order), ...]
            elem = scanlines["hrs_elem"].reshape(n_lines,
                        self.n_minorframes, self.n_wordperframe)
            # x & ~(1<<12)   ==   x - 1<<12     ==    x - 4096    if this
            # bit is set
            counts = elem[:, :self.n_perline, self.count_start:self.count_end]
            counts = counts - self.counts_offset
            counts = counts[:, :, numpy.argsort(self.channel_order)]
            rad_wn = self.calibrate(cc, counts)
            # Convert radiance to BT
            #(wn, c1, c2) = header["hrs_h_tempradcnv"].reshape(self.n_calibchannels, 3).T
            (wn, c1, c2) = self.get_wn_c1_c2(header)
            # convert wn to SI units
            wn /= constants.centi
            bt = self.rad2bt(rad_wn[:, :, :self.n_calibchannels], wn, c1, c2)
            # Copy over all fields... should be able to use
            # numpy.lib.recfunctions.append_fields but incredibly slow!
            scanlines_new = numpy.empty(shape=scanlines.shape,
                dtype=(scanlines.dtype.descr +
                    [("radiance", "f4", (self.n_perline, self.n_channels,)),
                     ("counts", "i2", (self.n_perline, self.n_channels,)),
                     ("bt", "f4", (self.n_perline, self.n_calibchannels,)),
                     ("lat", "f8", (self.n_perline,)),
                     ("lon", "f8", (self.n_perline,))]))
            for f in scanlines.dtype.names:
                scanlines_new[f] = scanlines[f]
            scanlines_new["radiance"] = physics.specrad_wavenumber2frequency(rad_wn)
            scanlines_new["counts"] = counts
            scanlines_new["bt"] = bt
            scanlines_new["lat"] = lat
            scanlines_new["lon"] = lon
            scanlines = scanlines_new
            if apply_flags:
                # initially, nothing is masked
                scanlines = numpy.ma.masked_array(scanlines)
                scanlines = self.get_mask_from_flags(scanlines)
        elif apply_flags:
            raise ValueError("I refuse to apply flags when not calibrating ☹")

        return (header, scanlines) if return_header else scanlines
       
    def check_parity(self, counts):
        """Verify parity for counts
        
        NOAA KLM Users Guide – April 2014 Revision, Section 3.2.2.4,
        Page 3-31, Table 3.2.2.4-1:

        > Minor Word Parity Check is the last bit of each minor Frame
        > or data element and is inserted to make the total number of
        > “ones” in that data element odd. This permits checking for
        > loss of data integrity between transmission from the instrument
        > and reconstruction on the ground.

        """

    def rad2bt(self, rad_wn, wn, c1, c2):
        """Apply the standard radiance-to-BT conversion from NOAA KLM User's Guide.

        Applies the standard radiance-to-BT conversion as documented by
        the NOAA KLM User's Guide.  This is based on a linearisation of a
        radiance-to-BT mapping for the entire channel.  A more accurate
        method is available in pyatmlab.physics.SRF.channel_radiance2bt,
        which requires explicit consideration of the SRF.  Such
        consideration is implicit here.  That means that this method
        is only valid assuming the nominal SRF!

        This method relies on values reported in the header of each
        granule.  See NOAA KLM User's Guide, Table 8.3.1.5.2.1-1., page
        8-108.  Please convert to SI units first.

        NOAA KLM User's Guide, Section 7.2.

        :param rad_wn: Spectral radiance per wanenumber
            [W·sr^{-1}·m^{-2}·{m^{-1}}^{-1}]
        :param wn: Central wavenumber [m^{-1}].
            Note that unprefixed SI units are used.
        :param c1: c1 as contained in hrs_h_tempradcnv
        :param c2: c2 as contained in hrs_h_tempradcnv
        """

        rad_f = physics.specrad_wavenumber2frequency(rad_wn)
        # standard inverse Planck function
        T_uncorr = physics.specrad_frequency_to_planck_bt(rad_f,
            physics.wavenumber2frequency(wn))

        T_corr = (T_uncorr - c1)/c2

        return T_corr

    # translation from HIRS.l1b format documentation to dtypes
    _trans_tovs2dtype = {"C": "|S",
                         "I1": ">i1",
                         "I2": ">i2",
                         "I4": ">i4"}
    _cmd = ("pdftotext", "-f", "{first}", "-l", "{last}", "-layout",
            "{pdf}", "{txt}")
    @classmethod
    def get_definition_from_PDF(cls, path_to_pdf):
        """Get HIRS definition from NWPSAF PDF.

        This method needs the external program pdftotext.  Put the result
        in header_dtype manually, but there are some corrections (see
        comments in source code).

        :param str path_to_pdf: Path to document
            NWPSAF-MF-UD-003_Formats.pdf
        :returns: (head_dtype, head_format, line_dtype, line_format)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = tmpdir + "/def"
            subprocess.check_call([a.format(
                first=cls.pdf_definition_pages[0],
                last=cls.pdf_definition_pages[1], pdf=path_to_pdf,
                txt=tmpfile) for a in cls._cmd])
#            head_fmt.seek(0, io.SEEK_END)
#            line_fmt.seek(0, io.SEEK_END)
            head_dtype = []
            line_dtype = []
            with open(tmpfile, encoding="utf-8") as tf:
                for line in tf:
                    if not line.strip().startswith("hrs"):
                        continue
                    (name, type, ws, nw, *descr) = line.strip().split()
                    dtp = head_dtype if name.startswith("hrs_h") else line_dtype
                    dtp.append(
                        (name,
                         cls._trans_tovs2dtype[type] + 
                                (ws if type=="C" else ""),
                         tools.safe_eval(nw)))
        return (head_dtype, line_dtype)

    def _apply_scale_factors(self, header, scanlines):
        new_head_dtype = self.header_dtype.descr.copy()
        new_line_dtype = self.line_dtype.descr.copy()
        for (i, dt) in enumerate(self.header_dtype.descr):
            if dt[0] in _tovs_defs.HIRS_scale_factors[self.version]:
                new_head_dtype[i] = (dt[0], ">f8") + dt[2:]
        for (i, dt) in enumerate(self.line_dtype.descr):
            if dt[0] in _tovs_defs.HIRS_scale_factors[self.version]:
                new_line_dtype[i] = (dt[0], ">f8") + dt[2:]
        new_head = numpy.empty(shape=header.shape, dtype=new_head_dtype)
        new_line = numpy.empty(shape=scanlines.shape, dtype=new_line_dtype)
        for (targ, src) in [(new_head, header), (new_line, scanlines)]:
            for f in targ.dtype.names:
                # NB: I can't simply say targ[f] = src[f] / 10**0, because
                # this will turn it into a float and refuse to cast it
                # into an int dtype
                if f in _tovs_defs.HIRS_scale_factors[self.version]:
                    # FIXME: does this work for many scanlines?
                    targ[f] = src[f] / numpy.power(
                            _tovs_defs.HIRS_scale_bases[self.version],
                            _tovs_defs.HIRS_scale_factors[self.version][f])
                else:
                    targ[f] = src[f]
        return (new_head, new_line)

    @abc.abstractmethod
    def get_wn_c1_c2(self, header):
        ...

    @abc.abstractmethod
    def seekhead(self, f):
        ...

    @abc.abstractmethod
    def calibrate(self, cc, counts):
        ...
            
    @abc.abstractmethod
    def get_mask_from_flags(self, lines):
        ...

    @abc.abstractmethod
    def get_cc(self, scanlines):
        ...
 
class HIRSPOD(HIRS):
    n_wordperframe = 22
    counts_offset = 0

    def id2no(self, satid):
        """Translate satellite id to satelline number.

        Follows POD guide, Table 2.0.4-3.

        WARNING: Does not support NOAA-6 or TIROS-N!
        """

        return _tovs_defs.HIRS_ids[self.version][satid]

    def seekhead(self, f):
        f.seek(0, io.SEEK_SET)

    def calibrate(self, cc, counts):
        """Apply the standard calibration from NOAA POD Guide

        POD Guide, section 4.5
        """

        # Equation 4.5-1
        # should normally have no effect as channels should be linear,
        # according to POD Guide, page 4-26
        # order is 0th, 1st, 2nd order term
        nc = cc[:, numpy.newaxis, :, 2, :]
        counts = nc[..., 0] + nc[..., 1] * counts + nc[..., 2] * counts**2

        # Equation 4.5.1-1
        # Use auto-coefficient.  There's also manual coefficient.
        # order is 2nd, 1st, 0th order term
        ac = cc[:, numpy.newaxis, :, 1, :]
        rad = ac[..., 2] + ac[..., 1] * counts + ac[..., 0] * counts**2

        if not (cc[:, :, 0, :]==0).all():
            raise ValueError("Found non-zero values for manual coefficient!")

        return rad

    def get_wn_c1_c2(self, header):
        h =  _tovs_defs.HIRS_coeffs[self.version][self.id2no(header["hrs_h_satid"][0])]
        return numpy.vstack([h[i] for i in range(1, 20)]).T

    def get_mask_from_flags(self, lines):
        bad = (lines["hrs_qualind"].data & ((1<<32)-(1<<8))) != 0
        lines["bt"].mask[bad] = True
        return lines

    def process_elem(self):
        #encoder_position = ascontiguousarray(el0[:, 0].view("<u2")) & ((1<<7)-1)
        #el_cal_level = (ascontiguousarray(el0[:, 0:2]).view("<u4")[:,0] & ((1<<13)-(1<<8))) >> 8
        #chpm = (ascontiguousarray(el0[:, 0:2]).view("<u4")[:,0] & ((1<<19)-(1<<13))) >> 13
        #tiptop = ascontiguousarray(el0[:, 0:2]).view(">u4").squeeze()
        #encoder_position = (ascontiguousarray(el0[:, 0:2]).view(">u4")[:,0] & ((1<<32)-(1<<24)))>>24

        out_of_sync =        (tiptop &  (1<< 6)) >> 6 == 0
        element_number =     (tiptop & ((1<<13) - (1<<7)))  >>  7
        ch1_period_monitor = (tiptop & ((1<<19) - (1<<13))) >> 13
        el_cal_level =       (tiptop & ((1<<24) - (1<<19))) >> 19
        encoder_position =   (tiptop & ((1<<32) - (1<<24))) >> 24
        
    def get_cc(self, scanlines):
        cc = scanlines["hrs_calcof"].reshape(scanlines.shape[0], 3,
                self.n_channels, 3)
        cc = numpy.swapaxes(cc, 2, 1)
        return cc
        

class HIRS2(HIRSPOD):
    #satellites = {"tirosn", "noaa06", "noaa07", "noaa08", "noaa09", "noaa10",
    # NOAA-6 and TIROS-N currently not supported due to duplicate ids.  To
    # fix this, would need to improve HIRSPOD.id2no.
    satellites = {"noaa07", "noaa08", "noaa09", "noaa10",
                  "noaa11", "noaa12", "noaa13", "noaa14"}
    version = 2

    header_dtype = _tovs_defs.HIRS_header_dtypes[2]
    line_dtype = _tovs_defs.HIRS_line_dtypes[2]
    channel_order = numpy.asarray(_tovs_defs.HIRS_channel_order[2])
    
class HIRSKLM(HIRS):
    counts_offset = 4096
    n_wordperframe = 24
    def seekhead(self, f):
        f.seek(0, io.SEEK_SET)
        if f.peek(3) in {b"NSS", b"CMS", b"DSS", b"UKM"}:
            f.seek(0, io.SEEK_SET)
        else: # assuming additional header
            f.seek(512, io.SEEK_SET)
    def calibrate(self, cc, counts):
        """Apply the standard calibration from NOAA KLM User's Guide.

        NOAA KLM User's Guide, section 7.2, equation (7.2-3), page 7-12,
        PDF page 286:

        r = a₀ + a₁C + a₂²C

        where C are counts and a₀, a₁, a₂ contained in hrs_calcof as
        documented in the NOAA KLM User's Guide:
            - Section 8.3.1.5.3.1, Table 8.3.1.5.3.1-1. and
            - Section 8.3.1.5.3.2, Table 8.3.1.5.3.2-1.,
        """
        rad = (cc[:, numpy.newaxis, :, 2]
             + cc[:, numpy.newaxis, :, 1] * counts 
             + cc[:, numpy.newaxis, :, 0]**2 * counts)
        # This is apparently calibrated in units of mW/m2-sr-cm-1.
        # Convert to SI units.
        rad *= constants.milli
        rad *= constants.centi # * not /, because it's 1/(cm^{-1}) = cm^1
        return rad

    def get_wn_c1_c2(self, header):
        return header["hrs_h_tempradcnv"].reshape(self.n_calibchannels, 3).T

    def get_mask_from_flags(self, lines):
        # These four entries are contained in each data frame and consider
        # the quality of the entire frame.  See Table 8.3.1.5.3.1-1. and
        # Table 8.3.1.5.3.2-1., 
        badline = (lines["hrs_qualind"] | lines["hrs_linqualflgs"]) != 0
        badchan = lines["hrs_chqualflg"] != 0
        badmnrframe = lines["hrs_mnfrqual"] != 0
        # Some lines are marked as space view or black body view
        badline |= (lines["hrs_scntyp"] != 0)
        # NOAA KLM User's Guide, page 8-154: Table 8.3.1.5.3.1-1.
        # consider flag for “valid”
        elem = lines["hrs_elem"].reshape(lines.shape[0], 64, 24)
        cnt_flags = elem[:, :, 22]
        valid = (cnt_flags & (1<<15)) != 0
        badmnrframe |= (~valid)
        # should consider parity flag... but how?
        #return (badline, badchannel, badmnrframe)

        # Where a channel is bad, mask the entire scanline
        lines["bt"].mask |= badchan.mask[:, numpy.newaxis, :19]

        # Where a minor frame is bad, mask all channels
        lines["bt"].mask |= badmnrframe[:, :56, numpy.newaxis]

        # Where an entire line is bad, mask all channels at entire
        # scanline
        lines["bt"].mask |= badline[:, numpy.newaxis, numpy.newaxis]

        # Where radiances are negative, mask individual values as masked
        lines["bt"].mask |= (lines["radiance"][:, :, :19] < 0)

        # Where counts==0, mask individual values
        lines["bt"].mask |= (elem[:, :56, 2:21]==0)

        return lines

    def get_cc(self, scanlines):
        cc = scanlines["hrs_calcof"].reshape(scanlines.shape[0], self.n_channels, 
                self.line_dtype["hrs_calcof"].shape[0]//self.n_channels)
        return cc

class HIRS3(HIRSKLM):
    pdf_definition_pages = (26, 37)
    version = 3

    satellites = {"noaa15", "noaa16", "noaa17"}

    header_dtype = _tovs_defs.HIRS_header_dtypes[3]
    line_dtype = _tovs_defs.HIRS_line_dtypes[3]

    channel_order = numpy.asarray(_tovs_defs.HIRS_channel_order[3])

class HIRS4(HIRSKLM):
    satellites = {"noaa18", "noaa19", "metopa", "metopb"}
    pdf_definition_pages = (38, 54)
    version = 4

    header_dtype = _tovs_defs.HIRS_header_dtypes[4]
    line_dtype = _tovs_defs.HIRS_line_dtypes[4]
    
    channel_order = numpy.asarray(_tovs_defs.HIRS_channel_order[4])

class IASINC(dataset.MultiFileDataset, dataset.HyperSpectral):
    """Read IASI from NetCDF
    """
    _dtype = numpy.dtype([
        ("time", "M8[s]"),
        ("lat", "f4"),
        ("lon", "f4"),
        ("satellite_zenith_angle", "f4"),
        ("satellite_azimuth_angle", "f4"),
        ("solar_zenith_angle", "f4"),
        ("solar_azimuth_angle", "f4"),
        ("spectral_radiance", "f4", 8700)])
    name = "iasinc"
    start_date = datetime.datetime(2003, 1, 1, 0, 0, 0)
    end_date = datetime.datetime(2013, 12, 31, 23, 59, 59)
    granule_duration = datetime.timedelta(seconds=1200)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        freqfile = self.basedir / "frequency.txt"
        if freqfile.exists():
            self.frequency = numpy.loadtxt(str(freqfile))
            
    def _read(self, path, fields="all", return_header=False):
        if fields == "all":
            fields = self._dtype.names
        logging.debug("Reading {!s}".format(path))
        with netCDF4.Dataset(str(path), 'r', clobber=False) as ds:
            scale = ds["scale_factor"][:]
            scale_valid = numpy.isfinite(scale) & (scale > 0)
            wavenumber = ds["wavenumber"][:] * ureg.parse_expression(ds["wavenumber"].units.replace("m-1", "m^-1"))
            wavenumber_valid = numpy.isfinite(wavenumber) & (wavenumber.m > 0)
            if not numpy.array_equal(scale_valid, wavenumber_valid):
                raise ValueError("Scale and wavenumber inconsistently valid")
            if self.wavenumber is None:
                self.wavenumber = wavenumber[wavenumber_valid]
            elif (abs(self.wavenumber - wavenumber[wavenumber_valid]).max()
                    > (0.05 * (1/ureg.centimetre))):
                raise ValueError("Inconsistent wavenumbers!")

            dtp = [x for x in self._dtype.descr if x[0] in fields]
            if dtp[-1][0] == "spectral_radiance":
                dtp[-1] = (dtp[-1][0], dtp[-1][1], wavenumber_valid.sum())

            M = numpy.zeros(
                dtype=dtp,
                shape=(len(ds.dimensions["along_track"]),
                       len(ds.dimensions["across_track"])))
            time_ref = numpy.datetime64(datetime.datetime.strptime(
                        ds["time"].gsics_reference_time,
                        "%Y-%m-%dT%H:%M:%S+00:00"), "s")
            dlt = numpy.array(ds["time"][:], dtype="m8[s]")
            M["time"] = (time_ref + dlt)[:, numpy.newaxis]
            for var in set(M.dtype.names) - {"time", "spectral_radiance"}:
                M[var] = ds[var][...]
            if "spectral_radiance" in M.dtype.names:
                M["spectral_radiance"][:, :, :] = (
                        ds["spectral_radiance"][:, :, scale_valid] /
                        scale[scale_valid][numpy.newaxis, numpy.newaxis, :])

        return M

class IASIEPS(dataset.MultiFileDataset, dataset.HyperSpectral):
    """Read IASI from EUMETSAT EPS L1C
    """

    name = "iasi"
    start_date = datetime.datetime(2007, 5,  29, 5, 8, 56)
    end_date = datetime.datetime(2015, 11, 17, 16, 38, 59)
    granule_duration = datetime.timedelta(seconds=6200)
    _dtype = numpy.dtype([
        ("time", "M8[ms]"),
        ("lat", "f4", (4,)),
        ("lon", "f4", (4,)),
        ("satellite_zenith_angle", "f4", (4,)),
        ("satellite_azimuth_angle", "f4", (4,)),
        ("solar_zenith_angle", "f4", (4,)),
        ("solar_azimuth_angle", "f4", (4,)),
        ("spectral_radiance", "f4", (4, 8700))])

    # Minimum temporary space for unpacking
    minspace = 2.5e9

    @staticmethod
    def __obtain_from_mdr(c, field):
        fieldall = numpy.concatenate([getattr(x.MDR, field)[:, :, :,
            numpy.newaxis] for x in c.MDR if hasattr(x, 'MDR')], 3)
        fieldall = numpy.transpose(fieldall, [3, 0, 1, 2])
        return fieldall

    def _read(self, path, fields="all", return_header=False):
        tmpdira = config.conf["main"]["tmpdir"]
        tmpdirb = config.conf["main"]["tmpdirb"]
        tmpdir = (tmpdira 
            if shutil.disk_usage(tmpdira).free < self.minspace
            else tmpdirb)
            
        with tempfile.NamedTemporaryFile(mode="wb", dir=tmpdir, delete=True) as tmpfile:
            with gzip.open(str(path), "rb") as gzfile:
                logging.debug("Decompressing {!s}".format(path))
                gzcont = gzfile.read()
                logging.debug("Writing decompressed file to {!s}".format(tmpfile.name))
                tmpfile.write(gzcont)
                del gzcont

            # All the hard work is in coda
            logging.debug("Reading {!s}".format(tmpfile.name))
            cfp = coda.open(tmpfile.name)
            c = coda.fetch(cfp)
            logging.debug("Sorting info...")
            n_scanlines = c.MPHR.TOTAL_MDR
            start = datetime.datetime(*coda.time_double_to_parts_utc(c.MPHR.SENSING_START))
            has_mdr = numpy.array([hasattr(m, 'MDR') for m in c.MDR],
                        dtype=numpy.bool)
            bad = [(m.MDR.DEGRADED_PROC_MDR|m.MDR.DEGRADED_INST_MDR)
                if hasattr(m, 'MDR') else True
                    for m in c.MDR]
            dlt = numpy.concatenate(
                [m.MDR.OnboardUTC[:, numpy.newaxis]
                    for m in c.MDR
                    if hasattr(m, 'MDR')], 1) - c.MPHR.SENSING_START
            M = numpy.ma.zeros(
                dtype=self._dtype,
                shape=(n_scanlines, 30))
            M["time"][has_mdr] = numpy.datetime64(start, "ms") + numpy.array(dlt*1e3, "m8[ms]").T
            specall = self.__obtain_from_mdr(c, "GS1cSpect")
            M["spectral_radiance"][has_mdr] = specall
            locall = self.__obtain_from_mdr(c, "GGeoSondLoc")
            M["lon"][has_mdr] = locall[:, :, :, 0]
            M["lat"][has_mdr] = locall[:, :, :, 1]
            satangall = self.__obtain_from_mdr(c, "GGeoSondAnglesMETOP")
            M["satellite_zenith_angle"][has_mdr] = satangall[:, :, :, 0]
            M["satellite_azimuth_angle"][has_mdr] = satangall[:, :, :, 1]
            solangall = self.__obtain_from_mdr(c, "GGeoSondAnglesSUN")
            M["solar_zenith_angle"][has_mdr] = solangall[:, :, :, 0]
            M["solar_azimuth_angle"][has_mdr] = solangall[:, :, :, 1]
            for fld in M.dtype.names:
                M.mask[fld][~has_mdr, ...] = True
                M.mask[fld][bad, ...] = True
            m = c.MDR[0].MDR
            wavenumber = (m.IDefSpectDWn1b * numpy.arange(m.IDefNsfirst1b, m.IDefNslast1b+0.1) * (1/ureg.metre))
            if self.wavenumber is None:
                self.wavenumber = wavenumber
            elif abs(self.wavenumber - wavenumber).max() > (0.05 * 1/(ureg.centimetre)):
                raise ValueError("Inconsistent wavenumbers")
            return M

class IASISub(dataset.HomemadeDataset, dataset.HyperSpectral):
    name = "iasisub"
    pass
