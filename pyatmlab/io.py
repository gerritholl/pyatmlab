#!/usr/bin/python
# coding: utf-8

"""Read and write particular datasets.

This module contains a collection of functions to locate, read, and
possibly write particular datasets, such as Chevallier data, NICAM data,
and others.
"""

# Some code in this module was originally obtained from the module
# PyARTS.io.

import os.path
import datetime

import numpy
from . import config

def get_chevallier_path(var, tp):
    """Get path to original Chevallier data.

    Requires that in ~/.pyatmlabrc, the configuration variable `cheval` in
    the section [main] is set to a directory where the Chevallier data are
    contained.

    :param var: What variable the Chevallier-data is maximised on.
    :param tp: Get atmosphere ('atm') or surface ('sfc')

    :returns: A string with the path to the Chevallier data file.
    """

    bd = config.get_config("cheval")
    return os.path.join(bd, "nwp_saf_{}_sampled.{}".format(var, tp))

def get_chevallier_arts_path(var, tp="all"):
    """Get path to ArtsXML Chevallier data extract.

    Require configuration variable `cheval_arts` set.
    See also :function get_chevallier_path:.

    :param var: What variable it is maximised on.
    :returns: A string with the path to the Chevallier data file.
    """

    bd = config.get_config("cheval_arts")
    return os.path.join(bd, "chevallierl91_{}_{}.xml.gz".format(tp, var))

# Obtained from FORTRAN-routine coming with Chevallier data.
#
# Atmospheric files:
#
#     temp(:),     &! 1) Temperature [K]                          (1-91)
#     hum(:),      &! 2) Humidity [kg/kg]                         (92-182)
#     ozo(:),      &! 3) Ozone [kg/kg]                            (183-273)
#     cc(:),       &! 4) Cloud Cover [0-1]                        (274-364)
#     clw(:),      &! 5) C Liquid W [kg/kg]                       (365-455)
#     ciw(:),      &! 6) C Ice W [kg/kg]                          (456-546)
#     rain(:),     &! 7) Rain [kg/(m2 *s)]                        (547-637)
#     snow(:),     &! 8) Snow [kg/(m2 *s)]                        (638-728)
#     w(:),        &! 9) Vertical Velocity [Pa/s]                 (729-819)
#     lnpsurf,     &!10) Ln of Surf Pressure in [Pa]              (820)
#     z0,          &!11) Surface geopotential [m2/s2]             (821) 
#     tsurf,       &!12) Surface Skin Temperature [K]             (822)
#     t2m,         &!13) 2m Temperature [K]                       (823)
#     td2m,        &!14) 2m Dew point temperature [K]             (824)
#     hum2m,       &!15) 2m Specific Humidity [kg/kg]             (825)
#     u10,         &!16) 10m wind speed U component [m/s]         (826)
#     v10,         &!17) 10m wind speed V component [m/s]         (827)
#     stratrsrf,   &!18) Stratiform rain at surface [kg/(m2 *s)]  (828)
#     convrsrf,    &!19) Convective rain at surface [kg/(m2 *s)]  (829)
#     snowsurf,    &!20) Snow at surface [kg/(m2 *s)]             (830)
#     lsm,         &!21) Land/sea Mask [0-1]                      (831)
#     lat,         &!22) Latitude [deg]                           (832)
#     long,        &!23) Longitude [deg]                          (833)
#     year,        &!24) Year                                     (834)
#     month,       &!25) Month                                    (835)
#     day,         &!26) Day                                      (836)
#     step,        &!27) Step                                     (837)
#     gpoint,      &!28) Grid point [1-843490]                    (838)
#     ind           !29) Index (rank-sorted)                      (839) 

chev_dtype_names = ("temp hum ozo cc clw ciw rain snow w lnpsurf z0 " 
                         "tsurf t2m td2m hum2m u10 v10 stratrsrf convrsrf "
                         "snowsurf lsm lat long year month day step gpoint "
                         "ind".split())
chev_dtype_sizes = [91] * 9 + [1] * 20
chev_dtype_types = [numpy.float64] * 29
chev_dtype_types[23] = numpy.uint16
for i in (24, 25, 26, 28):
    chev_dtype_types[i] = numpy.uint16
chev_dtype_types[27] = numpy.uint32
chev_dtype_atm = list(zip(chev_dtype_names, chev_dtype_types,
                      chev_dtype_sizes))

def read_chevallier(tp):
    """Read Chevallier data file.

    So far, reads only the atmospheric file.

    :param tp: variable that Chevallier is maximised upon
    :returns: An nd-array with a complicated dtype
    """
    f_atm = get_chevallier_path(tp, "atm")
    return numpy.loadtxt(f_atm, dtype=chev_dtype_atm)

def write_matrix_as_latex_table(x, y, M, outfile, xlabel=None, ylabel=None):
    """Write Matrix as LaTeX table.

    This is currently very inflexible, but may become more flexible in the
    future.  It's currently hardcoded for use with siunitx and booktabs.

    Needs packages booktabs, siunitx, multirow

    :param array x: Horizonal dimension
    :param array y: Vertical dimension
    :param 2d-array M: Matrix to write
    :param str outfile: Output file
    :param str xlabel: Label corresponding x-data
    :param str ylabel: Label corresponding to y-data
    """

    (rows, cols) = M.shape

    with open(outfile, 'w') as fp:
        fp.write(r'\footnotesize'"\n")
        fp.write(r"\setlength{\tabcolsep}{1mm}""\n")
        fp.write(r"\begin{tabular}{" + r"cc@{\hskip 5mm}" + "c"*cols + "}\n")
#        fp.write(r"\toprule" "\n")
#        fp.write(r"\midrule" "\n")
        if ylabel is not None:
            fp.write(r"\multirow{{{count:d}}}".format(count=rows) +
                      "{*}" +
                     r"{{\rotatebox{{90}}{{\large {ylabel}}}}}".format(ylabel=ylabel) +
                      "\n")
        for (i, row) in enumerate(M):
            fp.write(' & {:.5g} & '.format(y[i]) +
                     ' & '.join('{}'.format(c) for c in row) +
                     r"\\""\n")
#        fp.write(r"\bottomrule" "\n")
        fp.write(r"\addlinespace[5mm]""\n")
        fp.write(" & & " + ' & '.join('{:.5g}'.format(c) for c in x) + r"\\""\n")
        if xlabel is not None:
            fp.write(r" & \multicolumn{{{count:.5g}}}".format(count=cols+1) +
                      r"{{c}}{{\large {xlabel}}}\\".format(xlabel=xlabel)+"\n")
        fp.write(r"\end{tabular}" "\n")
        
def datadir():
    """Returns todays datadir.

    Configuration 'datadir' must be set.  Value is expanded with strftime.
    """

    return datetime.date.today().strftime(config.get_config('datadir'))


