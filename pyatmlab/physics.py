#!/usr/bin/env python

# coding: utf-8

"""Various small physics functions

Mostly obtained from PyARTS
"""

import logging
import numbers
import datetime
import calendar
import itertools

import numpy
import matplotlib
import matplotlib.dates

import pyproj


#from .constants import (h, k, R_d, R_v, c)
from . import constants as c
from . import math as pamath
from . import time as pytime
from . import tools
from . import graphics
from . import stats
from . import io as pyio

class AKStats:

    filename = "sensitivity_{mode}_matrix_{name}."
    #cmap = "afmhot_r"
    #cmap = trollimage.colormap.spectral
    cmap = "Spectral_r"

    def __init__(self, aks, name="UNDEFINED"):
        self.aks = aks.copy()
        with numpy.errstate(invalid="ignore"):
            self.aks[self.aks<=-999] = numpy.nan
        self.name = name

    def summarise(self, data):
        """Look at various statistics.
        """

        with numpy.errstate(invalid="warn"):
            self.plot_sensitivity_range(z=data["z"])
            self.plot_sensitivity_density(z=data["z"])
        self.plot_histogram()
        self.summarise_dof_stats(data)

    def dofs(self):
        """Calculate degrees of freedom.

        According to Rodgers (2000), equation (2.80), page 37.

        This is the trace of the averaging kernels.
        """

        # Circumvent https://github.com/numpy/numpy/issues/5560
        #return self.aks.trace(axis1=1, axis2=2)
        return numpy.trace(self.aks, axis1=1, axis2=2)

    def sensitivities(self):
        """Calculate sensitivities.

        According to:
          R. L. Batchelor et al.: Ground-based FTS comparisons ad ACE
          validation at Eureka during IPY.  Page 57.
        Sum of each row of the averaging kernel matrix defines sensitivity
        to measurument.  Note that I - A is the sensitivity to the a priori,
        soand the sum of the rows of (I - A) + the sum of the rows of A
        equals 1; so these two could be interpreted as percentages.
        """

        return self.aks.sum(1)

    def sensitivity_density_matrix(self, sens_fractions=numpy.linspace(0, 1, 11)):
        """What fraction of profiles have sensitivity >x at level y?

        :param ndarray sens_fractions: Fractions x to consider
        :returns: (sens_fractions, sens_mat),
            where sens_mat is a matrix with fraction at each level with at
            least sensitivity x.
        """

        sensitivities = self.sensitivities()
        with numpy.errstate(invalid="ignore"):
            sensmat = numpy.vstack([(sensitivities>x).sum(0) / sensitivities.shape[0]
                for x in sens_fractions])
        return (sens_fractions, sensmat)

    def sensitivity_range_matrix(self,
        sens_fractions = numpy.linspace(0, 1, 11),
        sens_counters = None):
        """
        For each profile, how many layers have sensitivity of at least x?
        
        Creates a "sensitivity score" matrix.  How many profiles have at
        least y layers of sensitivity >= x?

        :param ndarray sens_fractions:
            Array of shape (N,).
            Sensitivities to consider.  Defaults to 0, 0.1, ..., 1.0.
            Will be used to count no. of profiles with sensitivity larger
            than this fraction.
        :param ndarray sens_counters:
            Integer array of shape (p,),
            indicating the count as to how many profiles
            have at least this many layers with sensitivity larger than x.
            Defaults to arange(self.aks.shape[1]+1)
        :returns:
            Tuple (sens_fractions, sens_counters, sensmat), where sensmat
            is an N x p matrix, N being the number of sensitivities to
            consider, and p the counters.  In each coordinate define by
            the ararys sens_fractions and sens_counters, it contains the
            fraction of profiles that have at least p levels above
            sensitivity x.
        """
        
        if sens_counters is None:
            sens_counters = numpy.arange(self.aks.shape[1]+1)

        sensitivities = self.sensitivities()
        with numpy.errstate(invalid="ignore"):
            sensmat = numpy.vstack([((sensitivities >= x).sum(1)>=y).sum() 
                for x in sens_fractions
                for y in sens_counters]).reshape(
                    sens_fractions.shape[0], sens_counters.shape[0])
        return (sens_fractions, sens_counters, sensmat / self.aks.shape[0])

    def sensitivity_range_matrix_z(self, z,
            arr_dz = numpy.linspace(0, 30e3, 11),
            arr_sens = numpy.linspace(0, 1, 11)):
        """Like sensitivity_range_matrix but with elevation units.

        Returns a matrix with the fraction of elements where sensitivity
        exceeds 'x' for a range of elevations 'dz'.

        :param z: Matrix containing elevations.  Shape must match
            self.aks.shape[1:].
        :param arr_dz: Array of delta-z to consider.
        :param arr_sens: Array of sensitivities to consider.
        :returns: (arr_dz, arr_sens, mat)
        """

        mat = numpy.zeros(shape=(arr_dz.size, arr_sens.size))
        sensitivities = self.sensitivities()
        with numpy.errstate(invalid="ignore"):
            allmsk = [sensitivities > x for x in arr_sens]
        # find highest and lowest z for each column, corresponding to msk
        #
        # Note: in some cases there are 'gaps', i.e. sensitivity mask
        # looks like [True, True, False, False, False, True, True, True,
        # True, True, True, True, True, False, False, ...].  In this case
        # we take the 'False' along for now.  This may also yield more
        # than one 'last'.

        for (msk_i, msk) in enumerate(allmsk):
            # first True in each column
            mskno = msk.nonzero()
            if not msk.any(): # all mat 0
                continue
            (_, ii) = numpy.unique(mskno[0], return_index=True) 
            makes_sense = msk.any(1)
            firsts = numpy.zeros(shape=sensitivities.shape[0])
            firsts[makes_sense] = mskno[1][ii] # NB: goes wrong if 0 True values
            firsts[~makes_sense] = -1

            # last True in each column is element before!
            lasts = numpy.zeros(shape=sensitivities.shape[0])
            lasts[makes_sense] = numpy.hstack((mskno[1][ii[1:]-1], mskno[1][-1]))
            lasts[~makes_sense] = 0

            lower = numpy.array([z[i, firsts[i]] for i in range(firsts.shape[0])])
            upper = numpy.array([z[i, lasts[i]] for i in range(lasts.shape[0])])
            dz = upper - lower
            # by handling > as false, nans are counted as not in range.
            # There may be nans in z
            with numpy.errstate(invalid="ignore"):
                for (dz_k, dz_lim) in enumerate(arr_dz):
                    mat[dz_k, msk_i] = (dz >= dz_lim).sum() / sensitivities.shape[0]

        return (arr_dz, arr_sens, mat)


    def plot_sensitivity_density(self,
            nstep=11,
            z=None):
        """Visualise where sensors are typically sensitive
        """

        (sens_frac, sensmat) = self.sensitivity_density_matrix()

        # regrid sensitivity matrix for z
        if z.ndim > 1:
            if (z.min(0) == z.max(0)).all():
                z = z[0, :]
            else:
                newz = numpy.nanmean(z, 0)
                logging.info("Regridding sensitivity matrices")
                A_new = pamath.regrid_matrix(sensmat, z, newz)
                logging.info("Done")
                z = newz

        # write some diagnostics
        for (i, f) in enumerate(sens_frac):
            for p in (0.2, 0.5, 0.8):
                makes_sense = z[sensmat[i, :]>p]
                if makes_sense.any():
                    logging.info(("Altitude range with at least "
                        "{:.0%} >{:.0%} sensitive: {:.1f}--{:.1f} km").format(
                            p, f, makes_sense.min()/1e3,
                            makes_sense.max()/1e3))
                else:
                    logging.info(("Never more than {:.0%} with "
                        "sensitivity {:.0%} :(").format(
                        sensmat[i, :].max(), f))
            
        #f = matplotlib.pyplot.figure()
        (f, a) = matplotlib.pyplot.subplots() # = f.add_subplot(1, 1, 1)
        cs = a.contourf(sens_frac, z, sensmat.T,
            numpy.linspace(0, 1, nstep),
            cmap=self.cmap)
        #cs.clabel(colors="blue")
        cb = f.colorbar(cs)
        a.set_xlabel("Sensitivity")
        a.set_ylabel("Elevation [m]")
        cb.set_label("Fraction")
        a.set_title("Elevation sensitivity density {}".format(self.name))
        a.grid(which="major", color="white")
        
        graphics.print_or_show(
            f, False, self.filename.format(mode="density_z", name=self.name),
            data=numpy.vstack([(sens_frac[i], z[j], sensmat[i,j])
                               for i in range(sens_frac.size)
                               for j in range(z.size)]).reshape(
                                    sens_frac.size, z.size, 3))

    def plot_sensitivity_range(self,
            nstep=11,
            z=None):
        """Visualise vertical range of sensitivities
        """

        # Degrees of freedom according to:
        #   Rodgers (2000)
        #   Equation (2.80), Page 37

        dofs = self.dofs()

        sensitivities = self.sensitivities()

        max_sensitivities = sensitivities.max(1)

        (sens_fractions, sens_counters, sensmat) = self.sensitivity_range_matrix()

        f = matplotlib.pyplot.figure()
        a = f.add_subplot(1, 1, 1)
        cs = a.contourf(sens_fractions, sens_counters, sensmat.T,
            numpy.linspace(0, 1, nstep),
            cmap=self.cmap)
        #cs.clabel(colors="blue")
        cb = f.colorbar(cs)
        a.set_xlabel("Sensitivity")
        a.set_ylabel("No. of levels")
        cb.set_label("Fraction")
        a.set_title(("Fraction of profiles with at least N layers "
                     "sensitivity > x"))
        graphics.print_or_show(
            f, False, self.filename.format(mode="range_n", name=self.name))

        (arr_dz, arr_sens, mat_z) = self.sensitivity_range_matrix_z(z=z)

        f = matplotlib.pyplot.figure()
        a = f.add_subplot(1, 1, 1)
        cs = a.contourf(arr_sens, arr_dz, mat_z,
            numpy.linspace(0, 1, nstep),
            cmap=self.cmap)
        #cs.clabel(colors="blue")
        cb = f.colorbar(cs)
        a.set_xlabel("Sensitivity")
        a.set_ylabel("Delta z [m]")
        cb.set_label("Fraction")
        a.set_title(("Fraction of profiles with sensitivity > x "
                     "throughout a certain vertical range"))
        graphics.print_or_show(
            f, False, self.filename.format(mode="range_z", name=self.name))

    def plot_histogram(self):
        dofs = self.dofs()
        (f, a) = matplotlib.pyplot.subplots()
        (N, x, p) = a.hist(dofs[numpy.isfinite(dofs)], 20)
        a.set_xlabel("DOFs")
        a.set_ylabel("count")
        a.set_title("histogram DOF collocated {}".format(self.name))
        graphics.print_or_show(f, False,
            "hist_dof_{}.".format(self.name),
                data = dofs[numpy.isfinite(dofs)]) # let pgfplots do the hist

    _dof_binners = dict(
        doy = dict(
            label = "Day of year",
            bins = numpy.linspace(0, 366, 24),
            invert = False,
            timeax = False),
        mlst = dict(
            label = "Mean local solar time",
            bins = numpy.linspace(0, 24, 24),
            invert = False,
            timeax = False),
        time = dict(
            label = "Date",
            invert = False,
            timeax = True),
        lat = dict(
            label = "Latitude",
            invert = False,
            timeax = False),
        lon = dict(
            label = "Longitude",
            timeax = False,
            invert = True),
        parcol = dict(
            label = "Par. col. CH4",
            timeax = False,
            invert = False),
#        dof = dict(
#            label = "DOF",
#            timeax = False,
#            invert = False),
        )

    def summarise_dof_stats(self, data):
        """Make and plot some DOF summaries
        """

        dofs = self.dofs()

        # get day of year and mean local solar time
        
        # (NB: outer zip effectively unzips)
        (doy, mlst) = zip(*(pytime.dt_to_doy_mlst(
                                dt.astype(datetime.datetime),
                                lon)
                for (dt, lon) in zip(data["time"], data["lon"])))

        # Prepare matplotlib date axes
        ml = matplotlib.dates.DayLocator(bymonthday=[1, 15])
        datefmt = matplotlib.dates.DateFormatter("%m/%d")

        D = dict(doy={}, mlst={}, lat={}, lon={}, parcol={})
        D["doy"]["data"] = numpy.array(doy)
        D["mlst"]["data"] = numpy.array(mlst)
        D["lat"]["data"] = data["lat"]
        D["lon"]["data"] = data["lon"]
        D["parcol"]["data"] = data["parcol_CH4"]
#        D["dof"]["data"] = dofs

        for k in D.keys():
            if k in self._dof_binners and "bins" in self._dof_binners[k]:
                D[k]["bins"] = self._dof_binners[k]["bins"]
            else:
                D[k]["bins"] = numpy.linspace(
                    D[k]["data"].min()*0.99, D[k]["data"].max()*1.01, 10)

        binners = sorted(D.keys())
        binned_indices = stats.bin_nd(
                            [D[k]["data"] for k in binners],
                            [D[k]["bins"] for k in binners])
        # make "fake" date range where I will use only month and day,
        # so I can use date-based plotting
        D["time"] = dict(data=numpy.array([(datetime.date(2015, 1, 1)
                        + datetime.timedelta(days=int(d))).toordinal()
                        for d in D["doy"]["data"]]))
        D["time"]["bins"] = numpy.array([(datetime.date(2015, 1, 1)
                        + datetime.timedelta(days=int(d))).toordinal()
                        for d in D["doy"]["bins"]])
        # replace "doy" by "time"
        binners[binners.index("doy")] = "time"

        combis = sorted([tuple(x) for x in 
                    {frozenset(x)
                        for x in itertools.product(
                            range(binned_indices.ndim),
                            range(binned_indices.ndim))} if len(x)>1])
        # combis = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), ...]
        for (i1, i2) in combis:
            names = dict(x=binners[i1], y=binners[i2])

            merged = stats.binsnD_to_2d(binned_indices, i1, i2)
        
            stat = {}
            for (nm, func) in [
                    ("median", numpy.median),
                    ("mad", pamath.mad)]:
                stat[nm] = numpy.array(
                    [(func(dofs[merged.flat[k]])
                        if merged.flat[k].size>0
                        else numpy.nan)
                            for k in range(merged.size)]
                    ).reshape(merged.shape)
            stat["count"] = numpy.array(
                [merged.flat[k].size for k in range(merged.size)]
                    ).reshape(merged.shape)
            stat[""] = None # special value for scatter

            if (stat["count"]>1).sum() < 2:
                continue # all in one bin, don't plot
            #

            for (statname, val) in stat.items():

#                for mode in ("pcolor", "scatter"):
                (f, a) = matplotlib.pyplot.subplots()
#                    if mode == "pcolor":
                if statname == "":
                    pc = a.scatter(D[names["x"]]["data"],
                                   D[names["y"]]["data"], 
                                   c=dofs,
                                   s=50,
                                   cmap=self.cmap)
                else:
                    val_masked = numpy.ma.masked_array(val,
                        numpy.isnan(val))
                    pc = a.pcolor(D[names["x"]]["bins"],
                                  D[names["y"]]["bins"],
                                  val_masked.T,
                                  cmap=self.cmap)
#                    elif mode == "scatter":
                cb = f.colorbar(pc)

                #for (axname, axis, fmt) in [
                for axlt in "xy":
                    axname = names[axlt]
                    axis = getattr(a, "{}axis".format(axlt))
                    if self._dof_binners[axname]["timeax"]:
                        axis.set_major_locator(ml)
                        axis.set_major_formatter(datefmt)
                        getattr(f, "autofmt_{}date".format(axlt))()
                    if self._dof_binners[axname]["invert"]:
                        getattr(a, "invert_{}axis".format(axlt))()
                    getattr(a, "set_{}label".format(axlt))(
                        self._dof_binners[axname]["label"])

                cb.set_label("DOF " + statname)
                a.set_title("DOF " + self.name)
                graphics.print_or_show(f, False,
                    "DOF_{}_{}_{}_{}.".format(
                    self.name, names["x"], names["y"], statname),
                        data=(numpy.vstack((D[names["x"]]["data"],
                            D[names["y"]]["data"], dofs)).T
                                if (statname=="" and names["x"]=="time") else None))
                # write "by hand" because of datetime
                if names["x"] == "time" and statname=="":
                    with open("{:s}/{:s}".format(pyio.plotdatadir(),
                            "DOF_time_{:s}_{:s}.dat".format(
                                self.name, names['y'])), 'w') as f:
                        for i in range(dofs.shape[0]):
                            f.write("{:%Y-%m-%d} {:.3f} {:.3f}\n".format(
                                datetime.date.fromordinal(D["time"]["data"][i]),
                                D[names['y']]["data"][i], dofs[i]))

                    
                    
                    
        
def mixingratio2density(mixingratio, p, T):
    """Converts mixing ratio (e.g. kg/kg) to density (kg/m^3) for dry air.

    Uses the ideal gas law and the effective molar mass of Earth air.

    :param mixingratio: Mixing ratio [1]
    :param p: Pressure [Pa]
    :param T: Temperature [K]
    :returns: Density [kg/m^3]
    """

    # from ideal gas law, mass of 1 m³ of air:
    #
    # ρ = p/(R*T)

    m_air = p/(c.R_d*T)
    return mixingratio * m_air

def mixingratio2rh(w, p, T):
    """For water on Earth, convert mixing-ratio to relative humidity

    :param w: water vapour mixing ratio [1]
    :param p: pressure [Pa]
    :param T: temperature [K]
    :returns: relative humidity [1]
    """

    eps = c.R_d/c.R_v # Wallace and Hobbs, 3.14
    e = w/(w+eps)*p # Wallace and Hobbs, 3.59
    e_s = vapour_P(T)
    return e/e_s # Wallace and Hobbs, 3.64

def rh2mixingratio(rh, p, T):
    """Convert relative humidity to water vapour mixing ratio

    Based on atmlabs h2o/thermodynomics/relhum_to_vmr.m.

    :param rh: Relative humidity [1]
    :param p: Pressure [Pa]
    :param T: Temperature [K]
    :returns: Water vapour mixing ratio [1]
    """

    return rh * vapour_P(T) / p

def specific2mixingratio(q):
    """Convert specific humidity [kg/kg] to volume mixing ratio
    """

    # Source: extract_arts_1.f90

    eps = c.R_d/c.R_v
    return q / ( q + eps*(1-q) )


def vapour_P(T):
    """Calculate saturation vapour pressure.

    Calculates the saturated vapour pressure (Pa)
    of water using the Hyland-Wexler eqns (ASHRAE Handbook).

    (Originally in PyARTS)
    
    :param T: Temperature [K]
    :returns: Vapour pressure [Pa]
    """
    
    A = -5.8002206e3
    B = 1.3914993
    C = -4.8640239e-2
    D = 4.1764768e-5
    E = -1.4452093e-8
    F = 6.5459673
    
    Pvs = numpy.exp(A/T + B + C*T + D*T**2 + E*T**3 + F*numpy.log(T))
    return Pvs

def specific2iwv(z, q):
    """Calculate integrated water vapour [kg/m^2] from z, q

    :param z: Height profile [m]
    :param q: specific humidity profile [kg/kg]
    :returns: Integrated water vapour [kg/m^2]
    """

    mixing_ratio = specific2mixingratio(q)
    return pamath.integrate_with_height(z, mixing_ratio)

def rh2iwv(z, rh, p, T):
    """Calculate integrated water vapour [kg/m^2] from z, rh

    :param z: Height profile [m]
    :param rh: Relative humidity profile [1]
    :param p: Pressure profile [Pa]
    :param T: Temperature profile [T]
    :returns: Integrated water vapour [kg/m^2]
    """
    mixing_ratio = rh2mixingratio(rh, p, T)
    return pamath.integrate_with_height(z, mixingratio2density(mixing_ratio, p, T))

def mixingratio2iwv(z, r, p, T):
    """Calculate integrated water vapour [kg/m^2] from z, r

    :param z: Height profile [m]
    :param r: mixing ratio profile [kg/kg]
    :param p: Pressure profile [Pa]
    :param T: Temperature profile [T]
    :returns: Integrated water vapour [kg/m^2]
    """

    return pamath.integrate_with_height(z, mixingratio2density(r, p, T))

def wavelength2frequency(wavelength):
    """Converts wavelength (in meters) to frequency (in Hertz)

    :param wavelength: Wavelength [m]
    :returns: Frequency [Hz]
    """

    return c.c/wavelength

def wavenumber2frequency(wavenumber):
    """Converts wavenumber (in m^-1) to frequency (in Hz)

    :param wavenumber: Wave number [m^-1]
    :returns: Frequency [Hz]
    """

    return c.c*wavenumber

def frequency2wavelength(frequency):
    """Converts frequency [Hz] to wave length [m]

    :param frequency: Frequency [Hz]
    :returns: Wave length [m]
    """

    return c.c/frequency

def frequency2wavenumber(frequency):
    """Converts frequency [Hz] to wave number [m^-1]

    :param frequency: Frequency [Hz]
    :returns: Wave number [m^-1]
    """
    return frequency/c.c

def vmr2nd(vmr, T, p):
    """Convert volume mixing ratio [] to number density

    :param vmr: Volume mixing ratio or volume fraction.  For example,
        taking methane density in ppmv, first multiply by `constants.ppm`,
        then pass here.
    :param T: Temperature [K]
    :param p: Pressure [Pa]
    :returns: Number density in molecules per m^3
    """

    # ideal gas law: p = n_0 * k * T
    return  vmr * p / (c.k * T)

def p2z_oversimplified(p):
    """Convert pressure to altitude with oversimplified assumptions.

    Neglects the virtual temperature correction, assumes isothermal
    atmosphere with pressure dropping factor 10 for each 16 km.  Use a
    better function...

    :param p: Pressure [Pa]
    :returns: Altitude [m]
    """

    return 16e3 * (5 - numpy.log10(p) )

@tools.validator
def p2z_hydrostatic(p:numpy.ndarray,
    T:numpy.ndarray,
    h2o,
    p0:(numpy.number, numbers.Number, numpy.ndarray),
    z0:(numpy.number, numbers.Number, numpy.ndarray),
    lat:(numpy.number, numbers.Number, numpy.ndarray)=45,
    z_acc:(numpy.number, numbers.Number, numpy.ndarray)=-1,
    ellps="WGS84",
    extend=False):
    """Calculate hydrostatic elevation

    Translated from
    https://www.sat.ltu.se/trac/rt/browser/atmlab/trunk/geophysics/pt2z.m

    WARNING: seems to get siginificant errors.  Testing with an ACE
    profile between 8.5 and 150 km, I get errors from 10 up to +100 metre
    between 10 and 50 km, increasing to +300 metre at 100 km, after which
    the bias changes sign, crosses 0 at 113 km and finally reaches -4000
    metre at 150 km.  This is not due to humidity.  Atmlabs pt2z version
    differs only 30 metre from mine.  In %, this error is below 0.3% up to
    100 km, then changes sign and reaching -3% at 150 km.  For many
    purposes this is good enough, though, and certainly better than
    p2z_oversimplified.

    :param array p: Pressure [Pa]
    :param array T: Temperature [K].  Must match the size of p.
    :param h2o: Water vapour [vmr].  If negligible, set to 0.  Must be
        either scalar, or match the size of p and T.
    :param p0:
    :param z0:
    :param lat: Latitude [degrees].  This has some effect on the vertical
        distribution of gravitational acceleration, leading to difference
        of some 500 metre at 150 km.  Defaults to 45°.
    :param z_acc: Up to what precision to iteratively calculate the
        z-profile.  If -1, run two iterations, which should be accurate,
        according to the comment below.
    :param str ellps: Ellipsoid to use.  The function relies on
        pyproj.Geod, which is an interface to the proj library.  For a
        full table of ellipsoids, run 'proj -le'.
    :param bool extend: If p0, z0 outside of p, z range, extend
        artificially.  WARNING: This will assume CONSTANT T, h2o!
    :returns array z: Array of altitudes [m].  Same size as p and T.

    Original description:
% PT2Z   Hydrostatic altitudes
%
%    Calculates altitudes fulfilling hydrostatic equilibrium, based on
%    vertical profiles of pressure, temperature and water vapour. Pressure
%    and altitude of a reference point must be specified.
%
%    Molecular weights and gravitational constants are hard coded and
%    function is only valid for the Earth.
%
%    As the gravitation changes with altitude, an iterative process is
%    needed. The accuracy can be controlled by *z_acc*. The calculations
%    are repeated until the max change of the altitudes is below *z_acc*. If
%    z_acc<0, the calculations are run twice, which should give an accuracy
%    better than 1 m.
%
% FORMAT   z = pt2z( p, t, h2o, p0, z0 [,lat,z_acc,refell] )
%       
% OUT   z         Altitudes [m].
% IN    p         Column vector of pressures [Pa].
%       t         Column vector of temperatures [K].
%       h2o       Water vapour [VMR]. Vector or a scalar, e.g. 0.
%       p0        Pressure of reference point [Pa].
%       z0        Altitude of reference point [m].
%       lat       Latitude. Default is 45.
%       z_acc     Accuracy for z. Default is -1.
%       ellipsoid Reference ellipsoid data, see *ellipsoidmodels*.
%                 Default is data matching WGS84.

% 2005-05-11   Created by Patrick Eriksson.
"""

#32  function z = pt2z(p,t,h2o,p0,z0,varargin)
#33  %
#34  [lat,z_acc,ellipsoid] = optargs( varargin, { 45, -1, NaN } );
#35  %
    ellipsoid = pyproj.Geod(ellps=ellps)
#36  if isnan(ellipsoid)
#37    ellipsoid = ellipsoidmodels('wgs84');
#38  end
#39                                                                              %&%
#40  rqre_nargin( 5, nargin );                                                   %&%
#41  rqre_datatype( p, @istensor1 );                                             %&%
#42  rqre_datatype( t, @istensor1 );                                             %&%
#43  rqre_datatype( h2o, @istensor1 );                                           %&%
#44  rqre_datatype( p0, @istensor0 );                                            %&%
#45  rqre_datatype( z0, @istensor0 );                                            %&%
#46  rqre_datatype( lat, @istensor0 );                                           %&%

    if not p.size == T.size:
        raise ValueError("p and T must have same length")

    if p.min() < 0:
        raise ValueError("Found negative pressures")

    if T.min() < 0:
        raise ValueError("Found negative temperatures")

#47  np = length( p );
#48  if length(t) ~= np                                                          %&%
#49    error('The length of *p* and *t* must be identical.');                    %&%
#50  end                                                                         %&%

    if not (isinstance(h2o, numbers.Real) or h2o.size in (p.size, 1)):
        raise ValueError("h2o must have length of p or be scalar")

#51  if ~( length(h2o) == np  |  length(h2o) == 1 )                              %&%
#52    error('The length of *h2o* must be 1 or match *p*.');                     %&%
#53  end                                                                         %&%


# FIXME IS THIS NEEDED?  Yes — See e-mail Patrick 2014-08-11
    if p0 > p[0] or p0 < p[-1]:
        if extend:
            if p0 > p[0]: # p[0] is largest pressure, p0 even larger
                extend = "below"
                p = numpy.hstack([p0, p])
                T = numpy.hstack([T[0], T])
                h2o = numpy.hstack([h2o[0], h2o])
            elif p0 < p[-1]:
                extend = "above" # p[-1] is smallest pressure, p0 even smaller
                p = numpy.hstack([p, p0])
                T = numpy.hstack([T, T[-1]])
                h2o = numpy.hstack([h2o, h2o[-1]])
        else:
            raise ValueError(("reference pressure ({:.2f}) must be "
           "in total pressure range ({:.2f} -- {:.2f})").format(
               p0, p[0], p[-1]))
# END FIXME

#54  if p0 > p(1)  |  p0 < p(np)                                                 %&%
#55    error('Reference point (p0) can not be outside range of *p*.');           %&%
#56  end                                                                         %&%
#57  
#58  
#59  %= Expand *h2o* if necessary
#60  %
#61  if  length(h2o) == 1
#62    h2o = repmat( h2o, np, 1 );
#63  end
    if isinstance(h2o, numbers.Real) or h2o.size == 1:
        h2o = h2o * numpy.ones_like(p)

    if h2o.max() > 1:
        raise ValueError("Found h2o vmr values up to {:.2f}.  Expected < 1.".format(h2o.max()))
##64  
#65  
#66  %= Make rough estimate of *z*
#67  %
#68  z = p2z_simple( p );
    z = p2z_oversimplified(p)
#69  z = shift2refpoint( p, z, p0, z0 );
    z = _shift2refpoint(p, z, p0, z0)
#70  
#71  
#72  %= Set Earth radius and g at z=0
#73  %
#74  re = ellipsoidradii( ellipsoid, lat );
    # APPROXIMATION!  Approximate radius at latitude by linear
    # interpolation in cos(lat) between semi-major-axis and
    # semi-minor-axis
    #
    # Get radius at latitude
    re = (ellipsoid.a * numpy.cos(numpy.deg2rad(lat))
        + ellipsoid.b * (1-numpy.cos(numpy.deg2rad(lat))))
#75  g0 = lat2g0( lat );
    g0 = lat2g0(lat)
#76  
#77  
#78  %= Gas constant and molecular weight of dry air and water vapour
#79  %
#80  r  = constants( 'GAS_CONST' );
#81  md = 28.966;
#82  mw = 18.016;
#83  %
#84  k  = 1-mw/md;        % 1 - eps         
    k = 1 - c.M_w/c.M_d
#85  rd = 1e3 * r / md;   % Gas constant for 1 kg dry air
    rd = 1e3 * c.R / c.M_d  # gas constant for 1 kg dry air
#86  
#87  
#88  %= How to end iterations
#89  %
#90  if z_acc < 0
#91    niter = 2;
#92  else
#93    niter = 99;
#94  end 
    niter = 2 if z_acc < 0 else 99
#95  
#96  for iter = 1:niter
    for i in range(niter):
#97  
#98    zold = z;
#99   
        zold = z
#100   g = z2g( re, g0, z );
        g = z2g(re, g0, z)
#101 
#102   for i = 1 : (np-1)
        for i in range(p.size-1):
#103      
#104         gp  = ( g(i) + g(i+1) ) / 2;
            gp = (g[i] + g[i+1]) / 2
#105  
#106         %-- Calculate average water VMR (= average e/p)
#107         hm  = (h2o(i)+h2o(i+1)) / 2;
            hm = (h2o[i] + h2o[i+1]) / 2
#108  
#109         %--  The virtual temperature (no liquid water)
#110         tv = (t(i)+t(i+1)) / ( 2 * (1-hm*k) );   % E.g. 3.16 in Wallace&Hobbs
#111  
            tv = (T[i] + T[i+1]) / (2 * (1 - hm*k))

#112         %-- The change in vertical altitude from i to i+1
#113         dz = rd * (tv/gp) * log( p(i)/p(i+1) );
            dz = rd * (tv/gp) * numpy.log(p[i]/p[i+1])
#114         z(i+1) = z(i) + dz;
            z[i+1] = z[i] + dz
#115      
#116   end
#117  
#118   %-- Match the altitude of the reference point
#119   z = shift2refpoint( p, z, p0, z0 );
        z = _shift2refpoint(p, z, p0, z0)
#120 
#121   if z_acc >= 0 & max(abs(z-zold)) < z_acc
#122     break;
#123   end
        if z_acc >= 0 and max(abs(z-zold)) < z_acc:
            break
#124  
#125 end
#126 
#127 return
    # correct for extending
    if extend == "below": # lowest pressure extra
        return z[1:]
    elif extend == "above": # highest pressure extra
        return z[:-1]
    else:
        return z

#128 %----------------------------------------------------------------------------
#129 
#130 function z = shift2refpoint( p, z, p0, z0 )
#131   %
#132   z = z - ( interpp( p, z, p0 ) - z0 );
#133   %
#134 return

def _shift2refpoint(p, z, p0, z0):
    """Given z(p), shift this to include (p0, z0)

    Taken from atmlabs equivalent function
    https://www.sat.ltu.se/trac/rt/browser/atmlab/trunk/geophysics/pt2z.m
    """
    #return z - (pamath.interpp(p, z, p0) - z0)
    # revert p, z because for numpy.interp x-coor must be increasing
    return z - (numpy.interp(numpy.log(p0), numpy.log(p[::-1]), z[::-1]) - z0)

def z2g(r_geoid, g0, z):
    """Calculate gravitational acceleration at elevation

    Derived from atmlabs equivalent function
    https://www.sat.ltu.se/trac/rt/browser/atmlab/trunk/geophysics/pt2z.m

    :param r: surface radius at point [m]
    :param g0: surface gravitational acceleration at point [m/s^2]
    :param z: elevation [m]
    :returns: gravitational acceleration at point [m/s^2]
    """
#137 function g = z2g(r_geoid,g0,z)
#138   %
#139   g = g0 * (r_geoid./(r_geoid+z)).^2;
    return g0 * (r_geoid/(r_geoid+z))**2;
#140   %
#141 return
#142 

def lat2g0(lat):
    """Calculate surface gravitational acceleration for latitude

    This function is stolen from atmlab:
    https://www.sat.ltu.se/trac/rt/browser/atmlab/trunk/geophysics/pt2z.m

    From the original description:

    Expression below taken from Wikipedia page "Gravity of Earth", that is stated
    to be: International Gravity Formula 1967, the 1967 Geodetic Reference System
    Formula, Helmert's equation or Clairault's formula.

    :param lat: Latitude [degrees]
    :returns: gravitational acceleration [m/s]
    """

    x  = numpy.abs( lat );
    # see docstring for source of parametrisation
    return 9.780327 * ( 1 + 5.3024e-3*numpy.sin(numpy.deg2rad(x))**2 
                          + 5.8e-6*numpy.sin(numpy.deg2rad(2*x)**2 ))

