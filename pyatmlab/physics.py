#!/usr/bin/env python

# coding: utf-8

"""Various small physics functions

Mostly obtained from PyARTS
"""

import numbers

import numpy

import pyproj

#from .constants import (h, k, R_d, R_v, c)
from . import constants as c
from . import math as pamath
from . import tools

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
    ellps="WGS84"):
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

