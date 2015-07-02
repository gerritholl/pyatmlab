
import collections

import numpy
import scipy.interpolate

import h5py

from .. import dataset
from .. import physics
from .. import math as pamath
from ..constants import ppm as PPM, hecto as HECTO

class TansoFTSBase(dataset.ProfileDataset):
    """Applicable to both Tanso FTS versions
    """
    p_for_T_profile = numpy.array([1000, 975, 950, 925, 900, 850, 800,
        700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20,
        10])*HECTO # hPa -> Pa

    A_needs_converting = False
    A_needs_swapping = False

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
    aliases = {"CH4_profile": "ch4_profile",
               "ak": "ch4_ak",
               "delta_CH4_profile": "ch4_profile_e"}
    n_prof = "p"
    # For TANSO collocating with PEARL, 50% of profiles have > 40%
    # sensitivity between 5.3 km and 9.7 km.
    #
    # For TANSO collocating with ACE, 20% of profiles have > 30%
    # sensitivity between 5.2 km and 9.5 km.
    range = (5e3, 10e3)

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

    def get_z(self, obj, field=None):
        if field is None:
            try:
                return super().get_z(obj)
            except IndexError:
                pass # parent failed, continue here
        elif field != "T":
            raise ValueError("No special case for {}".format(field))
        # See comment near top of this class; different p-grids, different
        # z-grids.  Want z-grid corresponding to CH4-profile.
        p_for_T = self.p_for_T_profile # down up to 1 kPa
        p_for_CH4 = obj["p"] # down to ~50 Pa
        p_CH4_valid = numpy.isfinite(p_for_CH4)
        p_CH4_valid[p_CH4_valid] = p_for_CH4[p_CH4_valid] > 0
        p_for_CH4 = p_for_CH4[p_CH4_valid]
        p_T_valid = numpy.isfinite(p_for_T)
        p_T_valid[p_T_valid] = p_for_T[p_T_valid] > 0
        p_for_T = p_for_T[p_T_valid]
        p_valid = p_T_valid if field == "T" else p_CH4_valid
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
        # The purpose of this interpolation is to find temperature and
        # water vapour mixing ratio in order to convert pressure to
        # elevation.  Even setting water vapour mixing ratio completely
        # to 0 leads to sub-metre differences in elevation calculations.
        # This means that a "1-D spline" (i.e. linear interpolation in T
        # and log(vmr_h2o), k=1) is acceptable.
        xe = max(p_for_CH4.max(), p_for_T.max(), obj["p0"])
        tck_T = scipy.interpolate.splrep(p_for_T[::-1], obj["T"][::-1],
            xb=xb,
            xe=xe,
            k=1)
        tck_logh2o = scipy.interpolate.splrep(p_for_T[::-1], numpy.log(obj["h2o"][::-1]),
            xb=xb,
            xe=xe,
            k=1)

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
        p_for_T_i0 = len(p_full)
        p_full.extend(p_for_T)
        p_for_T_i1 = len(p_full)
        p_for_ch4_i0 = len(p_full)
        p_full.extend(p_for_CH4)
        p_for_ch4_i1 = len(p_full)
        p_full = numpy.array(p_full, dtype="f4")
        p_inds = numpy.argsort(p_full)[::-1] # high to low pressure

        # masked arrays go wrong within the p2z algorithm...
        # rather do masking "by hand" :(
        p_full_sorted = p_full[p_inds]
        ok = (numpy.isfinite(p_full_sorted) &
              (p_full_sorted > 0))
#        p_full_sorted = numpy.ma.masked_less(p_full_sorted, 0)
#        T_retgrid = numpy.ma.masked_array(
        T_retgrid = numpy.zeros_like(p_full_sorted)
        T_retgrid.fill(numpy.nan)
        T_retgrid[ok] = scipy.interpolate.splev(
            p_full_sorted[ok], tck_T, ext=1)#,
#            p_full_sorted.mask)
#        q_retgrid = numpy.ma.masked_array(
        logq_retgrid = numpy.zeros_like(p_full_sorted)
        logq_retgrid.fill(numpy.nan)
        logq_retgrid[ok] = scipy.interpolate.splev(
            p_full_sorted[ok], tck_logh2o, ext=1)#,
        q_retgrid = numpy.zeros_like(p_full_sorted)
        q_retgrid.fill(numpy.nan)
        q_retgrid[ok] = numpy.exp(logq_retgrid[ok])
#            p_full_sorted.mask)

        z_extp = numpy.zeros(shape=p_full_sorted.shape, dtype="f4")
        z_extp.fill(numpy.nan)
        ok = (ok &
              numpy.isfinite(T_retgrid) &
              numpy.isfinite(q_retgrid))
        ok[ok] = (ok[ok] & 
              (T_retgrid[ok] > 0) &
              (q_retgrid[ok] > 0))
        z_extp[ok] = physics.p2z_hydrostatic(
            p_full_sorted[ok], T_retgrid[ok], q_retgrid[ok],
            obj["p0"], obj["z0"], obj["lat"], -1)
        
        # return only elements belonging to p_for_CH4
        # this yields the "inverse" sorting indices to get back the
        # original order, but only for those that were part of p_for_CH4
        p_i_was_CH4 = ((p_inds>=p_for_ch4_i0) & (p_inds<p_for_ch4_i1)).nonzero()[0]
        p_i_was_T   = ((p_inds>=p_for_T_i0)   & (p_inds<p_for_T_i1)).nonzero()[0]

        z_ret = numpy.zeros(dtype="f4", 
            shape=(p_T_valid if field=="T" else p_CH4_valid).shape)
        z_ret[p_valid] = z_extp[p_i_was_T if field=="T" else p_i_was_CH4]
        z_ret[~p_valid] = numpy.nan
        return z_ret
        #return z_extp[p_i_was_CH4]#.data # no masked array...
#        inv_inds[-len(p_for_CH4):]]

#        z_for_T = physics.p2z_hydrostatic(
#            self.p_for_T_profile, obj["T"], obj["h2o"], 
#            obj["p0"], obj["z0"], obj["lat"], -1)
    
    def get_z_for(self, obj, field):
        return self.get_z(obj, field)

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

            ncd[n] = pamath.integrate_with_height(z, nd)
            ncd_e[n] = pamath.integrate_with_height(z, nd_e)

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

