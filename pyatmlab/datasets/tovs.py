"""Datasets for TOVS/ATOVS
"""

import io
import tempfile
import subprocess
import datetime
import logging
import gzip
import shutil
import abc
import pathlib
import dbm

import numpy
import scipy.interpolate

import netCDF4
import dateutil
import progressbar

try:
    import coda
except ImportError:
    logging.warn("Unable to import coda, won't read IASI EPS L1C")
    
import typhon.datasets.dataset
import typhon.utils.metaclass
import typhon.physics.units
from typhon.datasets.tovs import (Radiometer, HIRS, HIRSPOD, HIRS2,
    HIRSKLM, HIRS3, HIRS4)

from .. import dataset
from .. import tools
from .. import constants
from .. import physics
from .. import math as pamath
from ..units import ureg
from .. import config
from .. import units

from . import _tovs_defs

    
class HIRS2I(HIRS2):
    # identical fileformat, I believe
    satellites = {"noaa11", "noaa14"}

# HIRSFCDR and friends now in FCDR_HIRS.fcdr

class IASINC(dataset.MultiFileDataset, typhon.datasets.dataset.HyperSpectral):
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
            wavenumber = ureg.Quantity(ds["wavenumber"][:],
                ureg.parse_expression(ds["wavenumber"].units.replace("m-1", "m^-1")))
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

class IASIEPS(dataset.MultiFileDataset, typhon.datasets.dataset.HyperSpectral):
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
    # Warning: race conditions needs to be addressed.
    # As a workaround, choose very large minspace.
    minspace = 1e10

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
            if shutil.disk_usage(tmpdira).free > self.minspace
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
            bad = numpy.array([
                (m.MDR.DEGRADED_PROC_MDR|m.MDR.DEGRADED_INST_MDR)
                        if hasattr(m, 'MDR') else True
                        for m in c.MDR],
                            dtype=numpy.bool)
            dlt = numpy.concatenate(
                [m.MDR.OnboardUTC[:, numpy.newaxis]
                    for m in c.MDR
                    if hasattr(m, 'MDR')], 1) - c.MPHR.SENSING_START
            M = numpy.ma.zeros(
                dtype=self._dtype,
                shape=(n_scanlines, 30))
            M["time"][has_mdr] = numpy.datetime64(start, "ms") + numpy.array(dlt*1e3, "m8[ms]").T
            specall = self.__obtain_from_mdr(c, "GS1cSpect").astype("f8")
            # apply scale factors
            first = c.MDR[0].MDR.IDefNsfirst1b
            last = c.MDR[0].MDR.IDefNslast1b
            for (slc_st, slc_fi, fact) in zip(
                    filter(None, c.GIADR_ScaleFactors.IDefScaleSondNsfirst),
                    c.GIADR_ScaleFactors.IDefScaleSondNslast,
                    c.GIADR_ScaleFactors.IDefScaleSondScaleFactor):
                # Documented intervals are closed [a, b]; Python uses
                # half-open [a, b).
                specall[..., (slc_st-first):(slc_fi-first+1)] *= pow(10.0, -fact)
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

class IASISub(dataset.HomemadeDataset, typhon.datasets.dataset.HyperSpectral):
    name = "iasisub"
    subdir = "{month}"
    stored_name = "IASI_1C_selection_{year}_{month}_{day}.npz"
    re = r"IASI_1C_selection_(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<day>\d{1,2}).npz"
    start_date = datetime.datetime(2011, 1, 1, 0, 0, 0)
    end_date = datetime.datetime(2011, 12, 31, 23, 59, 59)

    
    def _read(self, *args, **kwargs):
        if self.frequency is None:
            self.frequency = numpy.loadtxt(self.freqfile)
        return super()._read(*args, **kwargs)

    def get_times_for_granule(self, p, **kwargs):
        gd = self.get_info_for_granule(p)
        (year, month, day) = (int(gd[m]) for m in "year month day".split())
        # FIXME: this isn't accurate, it usually starts slightly later...
        start = datetime.datetime(year, month, day, 0, 0, 0)
        # FIXME: this isn't accurate, there may be some in the next day...
        end = datetime.datetime(year, month, day, 23, 59, 59)
        return (start, end)



def which_hirs_fcdr(satname):
    """Given a satellite, return right HIRS object
    """
    for h in {HIRS2FCDR, HIRS3FCDR, HIRS4FCDR}:
        if satname in h.satellites:
            return h()
            break
    else:
        raise ValueError("Unknown HIRS satellite: {:s}".format(satname))
