"""Datasets for TOVS/ATOVS
"""

import io
import tempfile
import subprocess
import datetime
import logging

import numpy

import netCDF4
import dateutil

from .. import dataset
from .. import tools

class Radiometer:
    srf_dir = ""
    srf_backend_response = ""
    srf_backend_f = ""

class HIRS(dataset.MultiFileDataset, Radiometer):
    """High-resolution Infra-Red Sounder.

    This class can read HIRS l1b as published in the NOAA CLASS archive.

    Work in progress.
    """

    name = "hirs"
    format_definition_file = ""

    def _read(self, path, fields="all", return_header=False):
        if path.suffix == ".gz":
            opener = gzip.open
        else:
            opener = open
        with opener(str(path), 'rb') as f:
            if f.read(3) in {b"NSS", b"CMS", b"DSS", b"UKM"}:
                f.seek(0, io.SEEK_SET)
            else: # assuming additional header
                f.seek(512, io.SEEK_SET)

            header_bytes = f.read(4608)
            header = numpy.frombuffer(header_bytes, self.header_dtype)
            scanlines_bytes = f.read()
            scanlines = numpy.frombuffer(scanlines_bytes, self.scanline_dtype)
        if fields != "all":
            scanlines = scanlines[fields]
        return (header, scanlines) if return_header else scanlines
            

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

class HIRS2(HIRS):
    satellites = {"tirosn", "noaa06", "noaa07", "noaa08", "noaa09", "noaa10",
                  "noaa11", "noaa12", "noaa13", "noaa14"}
    pass # to be defined
    
class HIRS3(HIRS):
    pdf_definition_pages = (26, 37)

    satellites = {"noaa15", "noaa16", "noaa17"}

    # Obtained using get_definition_from_PDF.  Please note correction!
    header_dtype =  numpy.dtype([('hrs_h_siteid', '|S3', 1),
                      ('hrs_h_blank', '|S1', 1),
                      ('hrs_h_l1bversnb', '>u2', 1),
                      ('hrs_h_l1bversyr', '>u2', 1),
                      ('hrs_h_l1bversdy', '>u2', 1),
                      ('hrs_h_reclg', '>u2', 1),
                      ('hrs_h_blksz', '>u2', 1),
                      ('hrs_h_hdrcnt', '>u2', 1),
                      ('hrs_h_filler0', '>u2', 3),
                      ('hrs_h_dataname', '|S42', 1),
                      ('hrs_h_prblkid', '|S8', 1),
                      ('hrs_h_satid', '>u2', 1),
                      ('hrs_h_instid', '>u2', 1),
                      ('hrs_h_datatyp', '>u2', 1),
                      ('hrs_h_tipsrc', '>u2', 1),
                      ('hrs_h_startdatajd', '>u4', 1),
                      ('hrs_h_startdatayr', '>u2', 1),
                      ('hrs_h_startdatady', '>u2', 1),
                      ('hrs_h_startdatatime', '>u4', 1),
                      ('hrs_h_enddatajd', '>u4', 1),
                      ('hrs_h_enddatayr', '>u2', 1),
                      ('hrs_h_enddatady', '>u2', 1),
                      ('hrs_h_enddatatime', '>u4', 1),
                      ('hrs_h_cpidsyr', '>u2', 1),
                      ('hrs_h_cpidsdy', '>u2', 1),
                      ('hrs_h_filler1', '>u2', 4),
                      ('hrs_h_inststat1', '>u4', 1),
                      ('hrs_h_filler2', '>u2', 1),
                      ('hrs_h_statchrecnb', '>u2', 1),
                      ('hrs_h_inststat2', '>u4', 1),
                      ('hrs_h_scnlin', '>u2', 1),
                      ('hrs_h_callocsclin', '>u2', 1),
                      ('hrs_h_misscnlin', '>u2', 1),
                      ('hrs_h_datagaps', '>u2', 1),
                      ('hrs_h_okdatafr', '>u2', 1),
                      ('hrs_h_pacsparityerr', '>u2', 1),
                      ('hrs_h_auxsyncerrsum', '>u2', 1),
                      ('hrs_h_timeseqerr', '>u2', 1),
                      ('hrs_h_timeseqerrcode', '>u2', 1),
                      ('hrs_h_socclockupind', '>u2', 1),
                      ('hrs_h_locerrind', '>u2', 1),
                      ('hrs_h_locerrcode', '>u2', 1),
                      ('hrs_h_pacsstatfield', '>u2', 1),
                      ('hrs_h_pacsdatasrc', '>u2', 1),
                      ('hrs_h_filler3', '>u4', 1),
                      ('hrs_h_spare1', '|S8', 1),
                      ('hrs_h_spare2', '|S8', 1),
                      ('hrs_h_filler4', '>u2', 5),
                      ('hrs_h_autocalind', '>u2', 1),
                      ('hrs_h_solarcalyr', '>u2', 1),
                      ('hrs_h_solarcaldy', '>u2', 1),
                      ('hrs_h_calinf', '>u4', 80),
                      ('hrs_h_filler5', '>u4', 2),
                      ('hrs_h_tempradcnv', '>u4', 57),
                      ('hrs_h_20solfiltirrad', '>u2', 1),
                      # CORRECTION! NWPSAF guide says there is 1 field
                      # here, but in reality it is 2 (see NOAA KLM User's
                      # Guide, page 8-109, PDF page 420)
                      #('hrs_h_20equifiltwidth', '>u2', 1), 
                      ('hrs_h_20equifiltwidth', '>u2', 2), # CORRECTION!
                      ('hrs_h_filler6', '>u4', 1),
                      ('hrs_h_modelid', '|S8', 1),
                      ('hrs_h_nadloctol', '>u2', 1),
                      ('hrs_h_locbit', '>u2', 1),
                      ('hrs_h_filler7', '>u2', 1),
                      ('hrs_h_rollerr', '>u2', 1),
                      ('hrs_h_pitcherr', '>u2', 1),
                      ('hrs_h_yawerr', '>u2', 1),
                      ('hrs_h_epoyr', '>u2', 1),
                      ('hrs_h_epody', '>u2', 1),
                      ('hrs_h_epotime', '>u4', 1),
                      ('hrs_h_smaxis', '>u4', 1),
                      ('hrs_h_eccen', '>u4', 1),
                      ('hrs_h_incli', '>u4', 1),
                      ('hrs_h_argper', '>u4', 1),
                      ('hrs_h_rascnod', '>u4', 1),
                      ('hrs_h_manom', '>u4', 1),
                      ('hrs_h_xpos', '>u4', 1),
                      ('hrs_h_ypos', '>u4', 1),
                      ('hrs_h_zpos', '>u4', 1),
                      ('hrs_h_xvel', '>u4', 1),
                      ('hrs_h_yvel', '>u4', 1),
                      ('hrs_h_zvel', '>u4', 1),
                      ('hrs_h_earthsun', '>u4', 1),
                      ('hrs_h_filler8', '>u4', 4),
                      ('hrs_h_rdtemp', '>u2', 6),
                      ('hrs_h_bptemp', '>u2', 6),
                      ('hrs_h_eltemp', '>u2', 6),
                      ('hrs_h_pchtemp', '>u2', 6),
                      ('hrs_h_fhcc', '>u2', 6),
                      ('hrs_h_scnmtemp', '>u2', 6),
                      ('hrs_h_fwmtemp', '>u2', 6),
                      ('hrs_h_p5v', '>u2', 6),
                      ('hrs_h_p10v', '>u2', 6),
                      ('hrs_h_p75v', '>u2', 6),
                      ('hrs_h_m75v', '>u2', 6),
                      ('hrs_h_p15v', '>u2', 6),
                      ('hrs_h_m15v', '>u2', 6),
                      ('hrs_h_fwmcur', '>u2', 6),
                      ('hrs_h_scmcur', '>u2', 6),
                      ('hrs_h_pchcpow', '>u2', 6),
                      ('hrs_h_filler9', '>u4', 890)])

    line_dtype = numpy.dtype([('hrs_scnlin', '>i2', 1),
                      ('hrs_scnlinyr', '>i2', 1),
                      ('hrs_scnlindy', '>i2', 1),
                      ('hrs_clockdrift', '>i2', 1),
                      ('hrs_scnlintime', '>i4', 1),
                      ('hrs_scnlinf', '>i2', 1),
                      ('hrs_mjfrcnt', '>i2', 1),
                      ('hrs_scnpos', '>i2', 1),
                      ('hrs_scntyp', '>i2', 1),
                      ('hrs_filler1', '>i4', 2),
                      ('hrs_qualind', '>i4', 1),
                      ('hrs_linqualflgs', '>i4', 1),
                      ('hrs_chqualflg', '>i2', 20),
                      ('hrs_mnfrqual', '>i1', 64),
                      ('hrs_filler2', '>i4', 4),
                      ('hrs_calcof', '>i4', 60),
                      ('hrs_scalcof', '>i4', 60),
                      ('hrs_filler3', '>i4', 3),
                      ('hrs_navstat', '>i4', 1),
                      ('hrs_attangtime', '>i4', 1),
                      ('hrs_rollang', '>i2', 1),
                      ('hrs_pitchang', '>i2', 1),
                      ('hrs_yawang', '>i2', 1),
                      ('hrs_scalti', '>i2', 1),
                      ('hrs_ang', '>i2', 168),
                      ('hrs_pos', '>i4', 112),
                      ('hrs_filler4', '>i4', 2),
                      ('hrs_elem', '>i2', 1536),
                      ('hrs_filler5', '>i4', 3),
                      ('hrs_digbinvwbf', '>i2', 1),
                      ('hrs_digitbwrd', '>i2', 1),
                      ('hrs_aninvwbf', '>i4', 1),
                      ('hrs_anwrd', '>i1', 16),
                      ('hrs_filler6', '>i4', 11)])

class HIRS4(HIRS):
    satellites = {"noaa18", "noaa19", "metopa", "metopb"}
    pdf_definition_pages = (38, 54)

    # Obtained using get_definition_from_PDF.  Please note correction!
    head_dtype = numpy.dtype([('hrs_h_siteid', '|S3', 1),
                      ('hrs_h_blank', '|S1', 1),
                      ('hrs_h_l1bversnb', '>i2', 1),
                      ('hrs_h_l1bversyr', '>i2', 1),
                      ('hrs_h_l1bversdy', '>i2', 1),
                      ('hrs_h_reclg', '>i2', 1),
                      ('hrs_h_blksz', '>i2', 1),
                      ('hrs_h_hdrcnt', '>i2', 1),
                      ('hrs_h_filler0', '>i2', 3),
                      ('hrs_h_dataname', '|S42', 1),
                      ('hrs_h_prblkid', '|S8', 1),
                      ('hrs_h_satid', '>i2', 1),
                      ('hrs_h_instid', '>i2', 1),
                      ('hrs_h_datatyp', '>i2', 1),
                      ('hrs_h_tipsrc', '>i2', 1),
                      ('hrs_h_startdatajd', '>i4', 1),
                      ('hrs_h_startdatayr', '>i2', 1),
                      ('hrs_h_startdatady', '>i2', 1),
                      ('hrs_h_startdatatime', '>i4', 1),
                      ('hrs_h_enddatajd', '>i4', 1),
                      ('hrs_h_enddatayr', '>i2', 1),
                      ('hrs_h_enddatady', '>i2', 1),
                      ('hrs_h_enddatatime', '>i4', 1),
                      ('hrs_h_cpidsyr', '>i2', 1),
                      ('hrs_h_cpidsdy', '>i2', 1),
                      ('hrs_h_fov1offset', '>i2', 4),
                      ('hrs_h_instrtype', '|S6', 1),
                      ('hrs_h_inststat1', '>i4', 1),
                      ('hrs_h_filler1', '>i2', 1),
                      ('hrs_h_statchrecnb', '>i2', 1),
                      ('hrs_h_inststat2', '>i4', 1),
                      ('hrs_h_scnlin', '>i2', 1),
                      ('hrs_h_callocsclin', '>i2', 1),
                      ('hrs_h_misscnlin', '>i2', 1),
                      ('hrs_h_datagaps', '>i2', 1),
                      ('hrs_h_okdatafr', '>i2', 1),
                      ('hrs_h_pacsparityerr', '>i2', 1),
                      ('hrs_h_auxsyncerrsum', '>i2', 1),
                      ('hrs_h_timeseqerr', '>i2', 1),
                      ('hrs_h_timeseqerrcode', '>i2', 1),
                      ('hrs_h_socclockupind', '>i2', 1),
                      ('hrs_h_locerrind', '>i2', 1),
                      ('hrs_h_locerrcode', '>i2', 1),
                      ('hrs_h_pacsstatfield', '>i2', 1),
                      ('hrs_h_pacsdatasrc', '>i2', 1),
                      ('hrs_h_filler2', '>i4', 1),
                      ('hrs_h_spare1', '|S8', 1),
                      ('hrs_h_spare2', '|S8', 1),
                      ('hrs_h_filler3', '>i2', 5),
                      ('hrs_h_autocalind', '>i2', 1),
                      ('hrs_h_solarcalyr', '>i2', 1),
                      ('hrs_h_solarcaldy', '>i2', 1),
                      ('hrs_h_calinf', '>i4', 80),
                      # CORRECTION! NWPSAF calls his hrs_h_filler5, which
                      # already occurs a few lines down.
                      ('hrs_h_filler4', '>i4', 2),
                      ('hrs_h_tempradcnv', '>i4', 57),
                      ('hrs_h_20solfiltirrad', '>i2', 1),
                      ('hrs_h_20equifiltwidth', '>i2', 1),
                      ('hrs_h_filler5', '>i4', 1),
                      ('hrs_h_modelid', '|S8', 1),
                      ('hrs_h_nadloctol', '>i2', 1),
                      ('hrs_h_locbit', '>i2', 1),
                      ('hrs_h_filler6', '>i2', 1),
                      ('hrs_h_rollerr', '>i2', 1),
                      ('hrs_h_pitcherr', '>i2', 1),
                      ('hrs_h_yawerr', '>i2', 1),
                      ('hrs_h_epoyr', '>i2', 1),
                      ('hrs_h_epody', '>i2', 1),
                      ('hrs_h_epotime', '>i4', 1),
                      ('hrs_h_smaxis', '>i4', 1),
                      ('hrs_h_eccen', '>i4', 1),
                      ('hrs_h_incli', '>i4', 1),
                      ('hrs_h_argper', '>i4', 1),
                      ('hrs_h_rascnod', '>i4', 1),
                      ('hrs_h_manom', '>i4', 1),
                      ('hrs_h_xpos', '>i4', 1),
                      ('hrs_h_ypos', '>i4', 1),
                      ('hrs_h_zpos', '>i4', 1),
                      ('hrs_h_xvel', '>i4', 1),
                      ('hrs_h_yvel', '>i4', 1),
                      ('hrs_h_zvel', '>i4', 1),
                      ('hrs_h_earthsun', '>i4', 1),
                      ('hrs_h_filler7', '>i4', 4),
                      ('hrs_h_rdtemp', '>i4', 6),
                      ('hrs_h_bptemp', '>i4', 6),
                      ('hrs_h_eltemp', '>i4', 6),
                      ('hrs_h_pchtemp', '>i4', 6),
                      ('hrs_h_fhcc', '>i4', 6),
                      ('hrs_h_scnmtemp', '>i4', 6),
                      ('hrs_h_fwmtemp', '>i4', 6),
                      ('hrs_h_p5v', '>i4', 6),
                      ('hrs_h_p10v', '>i4', 6),
                      ('hrs_h_p75v', '>i4', 6),
                      ('hrs_h_m75v', '>i4', 6),
                      ('hrs_h_p15v', '>i4', 6),
                      ('hrs_h_m15v', '>i4', 6),
                      ('hrs_h_fwmcur', '>i4', 6),
                      ('hrs_h_scmcur', '>i4', 6),
                      ('hrs_h_pchcpow', '>i4', 6),
                      ('hrs_h_iwtcnttmp', '>i4', 30),
                      ('hrs_h_ictcnttmp', '>i4', 24),
                      ('hrs_h_tttcnttmp', '>i4', 6),
                      ('hrs_h_fwcnttmp', '>i4', 24),
                      ('hrs_h_patchexpcnttmp', '>i4', 6),
                      ('hrs_h_fsradcnttmp', '>i4', 6),
                      ('hrs_h_scmircnttmp', '>i4', 6),
                      ('hrs_h_pttcnttmp', '>i4', 6),
                      ('hrs_h_sttcnttmp', '>i4', 6),
                      ('hrs_h_bpcnttmp', '>i4', 6),
                      ('hrs_h_electcnttmp', '>i4', 6),
                      ('hrs_h_patchfcnttmp', '>i4', 6),
                      ('hrs_h_scmotcnttmp', '>i4', 6),
                      ('hrs_h_fwmcnttmp', '>i4', 6),
                      ('hrs_h_chsgcnttmp', '>i4', 6),
                      ('hrs_h_conversions', '>i4', 11),
                      ('hrs_h_moonscnlin', '>i2', 1),
                      ('hrs_h_moonthresh', '>i2', 1),
                      ('hrs_h_avspcounts', '>i4', 20),
                      ('hrs_h_startmanyr', '>i2', 1),
                      ('hrs_h_startmandy', '>i2', 1),
                      ('hrs_h_startmantime', '>i4', 1),
                      ('hrs_h_endmanyr', '>i2', 1),
                      ('hrs_h_endmandy', '>i2', 1),
                      ('hrs_h_endmantime', '>i4', 1),
                      ('hrs_h_deltav', '>i4', 3),
                      ('hrs_h_mass', '>i4', 2),
                      ('hrs_h_filler8', '>i2', 1302)])

    line_dtype =  numpy.dtype([('hrs_scnlin', '>i2', 1),
                      ('hrs_scnlinyr', '>i2', 1),
                      ('hrs_scnlindy', '>i2', 1),
                      ('hrs_clockdrift', '>i2', 1),
                      ('hrs_scnlintime', '>i4', 1),
                      ('hrs_scnlinf', '>i2', 1),
                      ('hrs_mjfrcnt', '>i2', 1),
                      ('hrs_scnpos', '>i2', 1),
                      ('hrs_scntyp', '>i2', 1),
                      ('hrs_filler1', '>i4', 2),
                      ('hrs_qualind', '>i4', 1),
                      ('hrs_linqualflgs', '>i4', 1),
                      ('hrs_chqualflg', '>i2', 20),
                      ('hrs_mnfrqual', '|S1', 64),
                      ('hrs_filler2', '>i4', 4),
                      ('hrs_calcof', '>i4', 60),
                      ('hrs_scalcof', '>i4', 60),
                      ('hrs_yawsteering', '>i2', 3),
                      ('hrs_totattcorr', '>i2', 3),
                      ('hrs_navstat', '>i4', 1),
                      ('hrs_attangtime', '>i4', 1),
                      ('hrs_rollang', '>i2', 1),
                      ('hrs_pitchang', '>i2', 1),
                      ('hrs_yawang', '>i2', 1),
                      ('hrs_scalti', '>i2', 1),
                      ('hrs_ang', '>i2', 168),
                      ('hrs_pos', '>i4', 112),
                      ('hrs_moonang', '>i2', 1),
                      # CORRECTION: NWPSAF formatting guide calls this
                      # filler4.  Should be filler3.
                      ('hrs_filler3', '>i2', 3),
                      ('hrs_elem', '>i2', 1536),
                      # CORRECTION: NWPSAF formatting guide calls this
                      # filler5.  Should be filler4.
                      ('hrs_filler4', '>i4', 3),
                      ('hrs_digitbupdatefg', '>i2', 1),
                      ('hrs_digitbwrd', '>i2', 1),
                      ('hrs_analogupdatefg', '>i4', 1),
                      ('hrs_anwrd', '|S1', 16),
                      ('hrs_filler5', '>i4', 11)])


    

class IASI(dataset.MultiFileDataset, dataset.HyperSpectral):
    _dtype = numpy.dtype([
        ("time", "M8[s]"),
        ("lat", "f4"),
        ("lon", "f4"),
        ("satellite_zenith_angle", "f4"),
        ("satellite_azimuth_angle", "f4"),
        ("solar_zenith_angle", "f4"),
        ("solar_azimuth_angle", "f4"),
        ("spectral_radiance", "f4", 8700)])
    name = "iasi"
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
            wavenumber = ds["wavenumber"][:]
            wavenumber_valid = numpy.isfinite(wavenumber) & (wavenumber > 0)
            if not numpy.array_equal(scale_valid, wavenumber_valid):
                raise ValueError("Scale and wavenumber inconsistently valid")
            if self.wavenumber is None:
                self.wavenumber = wavenumber[wavenumber_valid]
            elif abs(self.wavenumber - wavenumber[wavenumber_valid]).max() > 0.05:
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
