#!/usr/bin/env python
# coding: utf-8

"""Interact with matplotlib and other plotters

"""

import os.path
import datetime
now = datetime.datetime.now
import logging

import numpy
import matplotlib
import matplotlib.pyplot
from . import config
from . import io

def plotdir():
    """Returns todays plotdir.

    Configuration 'plotdir' must be set.  Value is expanded with strftime.
    """
    return datetime.date.today().strftime(config.get_config('plotdir'))

def print_or_show(fig, show, outfile, in_plotdir=True, tikz=None, data=None):
    """Either print or save figure, or both, depending on arguments.

    Taking a figure, show and/or save figure in the default directory,
    obtained with :func:plotdir.  Creates plot directory if needed.

    :param fig: Figure to store.  
    :type fig: matplotlib.Figure object
    :param show: Show figure or not
    :type show: boolean
    :param outfile: File to write figure to, or list of files.  If the
        string ends in a '.', write to x.png and x.pdf.
    :type outfile: string or list of strings
    :param in_plotdir: If true, write to default plot directory.  If
        false, write to currect directory or use absolute path.
    :type in_plotdir: boolean
    :param tikz: Try to write tikz code with matplotlib2tikz.  Requires
        that the latter is installed.
    :type tikz: boolean
    :param data: Store associated data in .dat file (useful for pgfplots).
        May be a list of ndarrays, which results in multiple numbered datafiles.
    :type data: ndarray or list thereof
    """

    if outfile is not None:
        outfiles = [outfile] if isinstance(outfile, str) else outfile
        if isinstance(outfile, str):
            if outfile.endswith("."):
                outfiles = [outfile+ext for ext in ("png", "pdf")]
            else:
                outfiles = [outfile]

        # interpret as sequence
        for outf in outfiles:
            if in_plotdir and not '/' in outf:
                outf = os.path.join(plotdir(), outf)
            logging.info("Writing to file: {}".format(outf))
            if not os.path.exists(os.path.dirname(outf)):
                os.makedirs(os.path.dirname(outf))
            fig.canvas.print_figure(outf)
    if show:
        matplotlib.pyplot.show()
    if tikz is not None:
        import matplotlib2tikz
        print(now(), "Writing also to:", os.path.join(plotdir(), tikz))
        matplotlib2tikz.save(os.path.join(plotdir(), tikz))
    if data is not None:
        if not os.path.exists(io.plotdatadir()):
            os.makedirs(io.plotdatadir())
        if isinstance(data, numpy.ndarray):
            data = (data,)
        # now take it as a loop
        for (i, dat) in enumerate(data):
            outf = os.path.join(io.plotdatadir(),
                "{:s}{:d}.dat".format(
                    os.path.splitext(outfiles[0])[0], i))
            fmt = ("%d" if issubclass(dat.dtype.type, numpy.integer) else
                    '%.18e')
            if len(dat.shape) < 3:
                numpy.savetxt(outf, dat, fmt=fmt)
            elif len(dat.shape) == 3:
                io.savetxt_3d(outf, dat, fmt=fmt)
            else:
                raise ValueError("Cannot write {:d}-dim ndarray to textfile".format(
                    len(dat.shape)))

