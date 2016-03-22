#!/usr/bin/env python
# coding: utf-8

"""Interact with matplotlib and other plotters

"""

import os.path
import datetime
now = datetime.datetime.now
import logging
import subprocess
import sys
import pickle

import numpy
import matplotlib
import matplotlib.pyplot
from . import config
from . import io
from . import meta
from . import tools

def plotdir():
    """Returns todays plotdir.

    Configuration 'plotdir' must be set.  Value is expanded with strftime.
    """
    return datetime.date.today().strftime(config.get_config('plotdir'))

def print_or_show(fig, show, outfile, in_plotdir=True, tikz=None,
                  data=None, store_meta="", close=True):
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
    :param store_meta: Also store other info.  This is a string that will
        be written to a file.  If not set or set to None, it will just
        write the pyatmlab version.  The file will use the same basename
        as the outfile, but replacing the extention by "info".  However,
        this only works if outfile is a string and not a list thereof.
        To write nothing, pass an empty string.
    :type store_meta: str.
    :param close: If true, close figure.  Defaults to true.
    :type close: bool.
    """

    if outfile is not None:
        outfiles = [outfile] if isinstance(outfile, str) else outfile
        if isinstance(outfile, str):
            if outfile.endswith("."):
                outfiles = [outfile+ext for ext in ("png", "pdf")]
                infofile = outfile + "info"
                figfile = outfile + "pkl"
            else:
                outfiles = [outfile]
                infofile = None
                figfile = None

        pr = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE) 
        info = " ".join(sys.argv) + "\n" + pr.stdout.decode("utf-8") + "\n"
        info += tools.get_verbose_stack_description()
        if infofile is not None and info:
            if in_plotdir and not "/" in infofile:
                infofile = os.path.join(plotdir(), infofile)
            logging.info("Writing info to {:s}".format(infofile))
            with open(infofile, "w", encoding="utf-8") as fp:
                fp.write(info)
        if figfile is not None:
            if in_plotdir and not "/" in figfile:
                figfile = os.path.join(plotdir(), figfile)
            logging.info("Writing figure object to {:s}".format(figfile))
            with open(figfile, "wb") as fp:
                pickle.dump(fig, fp, protocol=4)
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

    if close:
        matplotlib.pyplot.close(fig)

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

