#!/usr/bin/env python

# coding: utf-8

"""Various small physics functions

Mostly obtained from PyARTS
"""

import numpy
from .constants import (h, k, R_d, R_v)
from . import math as pymath

def mixingratio2density(mixingratio, p, T):
    """Converts mixing ratio (e.g. kg/kg) to density (kg/m^3) for dry air

    :param mixingratio: Mixing ratio [1]
    :param p: Pressure [Pa]
    :param T: Temperature [K]
    :returns: Density [kg/m^3]
    """

    # from ideal gas law, mass of 1 m³ of air:
    #
    # ρ = p/(R*T)

    R_d = 287.0 # J K^-1 kg^-1
    m_air = p/(R_d*T)
    return mixingratio * m_air

def mixingratio2rh(w, p, T):
    """For water on Earth, convert mixing-ratio to relative humidity

    :param w: water vapour mixing ratio [1]
    :param p: pressure [Pa]
    :param T: temperature [K]
    :returns: relative humidity [1]
    """

    eps = R_d/R_v # Wallace and Hobbs, 3.14
    e = w/(w+eps)*p # Wallace and Hobbs, 3.59
    e_s = vapour_P(T)
    return e/e_s # Wallace and Hobbs, 3.64

def specific2mixingratio(q):
    """Convert specific humidity [kg/kg] to volume mixing ratio
    """

    # Source: extract_arts_1.f90

    eps = R_d/R_v
    return q / ( q + eps*(1-q) )


def vapour_P(T):
    """Calculate saturation vapour pressure.

    Calculates the saturated vapour pressure (Pa)
    of water using the Hyland-Wexler eqns (ASHRAE Handbook).
    
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

def calculate_iwv(z, q):
    """Calculate integrated water vapour [kg/m^2] from z, q

    :param z: Height profile [m]
    :param q: specific humidity profile [kg/kg]
    :returns: Integrated water vapour scalar [kg/m^2]
    """

    mixing_ratio = specific2mixingratio(q)
    return pymath.integrate_with_height(z, mixing_ratio)
