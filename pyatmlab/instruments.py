#!/usr/bin/env python3.3

"""Classes to represent instruments in an abstract way.

Describe a channel by its centre frequency, width, shape, no. of
sidebands, etc., and write out its SRF.  The same for an entire
instrument, which is essentially a collection of channels.
"""

# $Id: instruments.py 8140 2013-01-24 15:47:30Z gerrit $

import datetime
now = datetime.datetime.now

import os.path

import numpy
import decimal
import numbers

from . import physics
from . import tools

class Channel(object):
    """Represents a (rectangular) radiometer channel.

    """

    def __init__(self, freq, noise, width, sideband=0):
        """Initialise radiometer channel.

        :param freq: Centre frequency [Hz].  This is converted to a
            decimal, and therefore best passed as a string.
        :param noise: Noise level [K]
        :param width: Width of each sideband [Hz].  This is converted to a
            decimal.
        :param sideband: Distance to sidebands (default 0).  This is
            converted to a decimal.
        """
        self.centre_frequency = decimal.Decimal(freq)
        self.noise = noise
        self.width = decimal.Decimal(width)
        self.sideband = decimal.Decimal(sideband)
        self.shape = "rectangular"

    def srf(self, step=99e6):
        """Returns SRF (x, y) with stepsize `step`

        :param step: Stepsize (resolution) [Hz]
        """
        if self.sideband == 0:
            centres = [self.centre_frequency]
        else:
            centres = [self.centre_frequency-self.sideband,
                       self.centre_frequency+self.sideband]
        if self.shape == "rectangular":
            edges = [(float(c-self.width), float(c+self.width)) 
                        for c in centres]
            pp = [numpy.arange(e[0], e[1]+step/2, step) for e in edges]
            f_grid = numpy.concatenate(pp)
            y = numpy.ones(f_grid.size)/f_grid.size
            return (f_grid, y)
        else:
            raise NotImplementedError("Unknown shape: {}".format(self.shape))

    @tools.validator
    def write_srf(self, to, quantity="frequency", factor: numbers.Number=1.0):
        """Write SRF to file.

        :param to: Output file.  Can be an open file, a string containing
            a filename, or a directory.  If it is a directory, the
            filename will be calculated intelligently.

        :param quantity: What unit the SRF is described in.  This should
            be a string, and can be 'frequency' (the defailt)',
            'wavelength', and 'wavenumber'.

        :param float factor: Multiplication factor for the quantity.
            Normally, everything is described in SI units.  However, if
            you want to describe the SRF in cm^-1 instead of m^-1, you
            should pass factor=0.01.
        """
        #tools.validate(Channel.write_srf, locals())
        try:
            to.write
        except AttributeError: # not an open file
            try:
                to = open(to, 'wb')
            except IsADirectoryError:
                to = open(os.path.join(to, "specresp_" +
                                           self.get_chanstr(full=True) +
                                           ".dat"),
                          'wb')

        (x, y) = self.srf()
        if quantity == "frequency":
            pass
        elif quantity == "wavelength":
            x = physics.frequency2wavelength(x)
        elif quantity == "wavenumber":
            x = physics.frequency2wavenumber(x)
        else:
            raise ValueError("Unknown quantity: {}".format(quantity))

        print(now(), "Writing to", to.name)
        numpy.savetxt(to, numpy.c_[x*factor, y])

            
    def get_chanstr(self, full=False):
        """Get the shortest string representation of frequency in GHz

        :param bool full: Be verbose in the channel description.  Defaults
            to False.
        """

        # if whole number, strip off decimal part
        # see also http://stackoverflow.com/q/11227620/974555
        chanstr = str(self.centre_frequency/decimal.Decimal("1e9"))
        if '.' in chanstr:
            chanstr = chanstr.rstrip('0').rstrip('.')

        if full:
            widthstr = "-" + str(self.width/decimal.Decimal("1e9"))
            if self.sideband == 0:
                sbstr = ""
            else:
                sbstr = "±" + str(self.sideband/decimal.Decimal("1e9"))
            return chanstr + sbstr + widthstr
        else:
            return chanstr

    def __repr__(self):
        return "<Channel:{} GHz>".format(self.get_chanstr())

class Radiometer(object):
    """Represents a radiometer.

    Initialise with list of Channel-objects.
    """

    name = None

    def __init__(self, channels, name="(noname)"):
        """Initialise Radiometer object

        :param channels: List of channels that the radiometer consists of.
        :type channels: List of Channel objects
        :param str name: Instrument name
        """
        self.channels = channels
        self.name = name

    def channel_string(self, pre="", full=False):
        """Get string with channel descriptions.

        Channel descriptions are compact string specifications of channel
        frequencies in GHz. Those are concatenated together separated by
        whitespace.

        :param str pre: String to prepend to each channel.
        :param bool full: Be verbose in the channel descriptions.
            Defaults to False.
        """

        return " ".join(pre+c.get_chanstr(full=full) for c in self.channels)

    def write_all_srf(self, d, *args, **kwargs):
        """Write all SRF to directory d.

        :param str d: Directory to which SRFs will be written.

        Remaining arguments passed on to Channel.write_srf
        """

        for chan in self.channels:
            chan.write_srf(d, *args, **kwargs)

    def __repr__(self):
        if self.name is not None:
            return "<Radiometer {} {} chans>".format(
                        self.name, len(self.channels))
        else:
            return "<Radiometer {} chans>".format(len(self.channels))

# FIXME: read from some external definition file

# icemusic all single band
icemusic = Radiometer([
    Channel("89E9", 0.1, "10E9"), # string to keep precision for Decimal
    Channel("150E9", 0.1, "10E9"),
    Channel("183.31E9", 0.1, "1.8E9"),
    Channel("186.31E9", 0.1, "1.8E9"),
    Channel("190.31E9", 0.1, "1.8E9"),
    Channel("205E9", 0.1, "10E9"),
    Channel("312.5E9", 0.1, "10E9"),
    Channel("321.5E9", 0.1, "3.2E9"),
    Channel("505E9", 0.1, "10E9"),
    Channel("530E9", 0.1, "10E9"),
    Channel("875E9", 0.1, "20E9"),
    Channel("1500E9", 0.1, "20E9")],
        name="icemusic")

# source: buehler07:_ciwsir_qjrms
ciwsir = Radiometer([
    Channel("183.31e9", 0.6, "1.4e9", "1.5e9"),
    Channel("183.31e9", 0.5, "2.0e9", "3.5e9"),
    Channel("183.31e9", 0.4, "3.0e9", "7.0e9"),
    Channel("243.2e9", 0.5, "3.0e9", "2.5e9"),
    Channel("325.15e9", 1.0, "1.6e9", "1.5e9"),
    Channel("325.15e9", 0.8, "2.4e9", "3.5e9"),
    Channel("325.15e9", 0.7, "3.0e9", "9.5e9"),
    Channel("448.0e9", 1.9, "1.2e9", "1.4e9"),
    Channel("448.0e9", 1.4, "2.0e9", "3.0e9"),
    Channel("448.0e9", 1.2, "3.0e9", "7.2e9"),
    Channel("664.0e9", 1.5, "5.0e9", "4.2e9")],
        name="ciwsir")

# source: EPS-SG End User Requirements Document
ici = Radiometer([
    Channel("183.31e9", 0.7, "2.000e9", "7.0e9"),
    Channel("183.31e9", 0.7, "1.500e9", "3.4e9"),
    Channel("183.31e9", 0.7, "1.500e9", "2.0e9"),
    Channel("243.2e9", 0.6, "3.000e9", "2.5e9"),
    Channel("325.15e9", 1.1, "3.000e9", "9.5e9"),
    Channel("325.15e9", 1.2, "2.400e9", "3.5e9"),
    Channel("325.15e9", 1.4, "1.600e9", "1.5e9"),
    Channel("448e9", 1.3, "3.000e9", "7.2e9"),
    Channel("448e9", 1.5, "2.000e9", "3.0e9"),
    Channel("448e9", 1.9, "1.200e9", "1.4e9"),
    Channel("664e9", 1.5, "5.000e9", "4.2e9"),
    ],
        name="ici")

del now
