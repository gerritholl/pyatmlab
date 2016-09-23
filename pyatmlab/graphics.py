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
import lzma
import pathlib

import numpy
import matplotlib
import matplotlib.pyplot
import mpl_toolkits.basemap
from . import config
from . import io
from . import meta
from . import tools

def pcolor_on_map(m, lon, lat, C, **kwargs):
    """Wrapper around pcolor on a map, in case we cross the IDL

    preventing spurious lines
    """

    # Need to investigate why and how to solve:
    # -175 - minor polar problems (5° missing)
    # -178 - polar problems (2° missing)
    # -179 - some problems (1° missing)
    # -179.9 - many problems
    # perhaps I need to mask neighbours or so?
    C1 = numpy.ma.masked_where(lon<-175, C, copy=True)
    p1 = m.pcolor(lon, lat, C1, latlon=True, **kwargs)
#    C2 = numpy.ma.masked_where(lon<0, C.data)
#    p2 = m.pcolor(lon, lat, C2, latlon=True, **kwargs)
#    mixed = lon.ptp(1)>300
#    homog = ~mixed
#    homognum = homog.nonzero()[0]
#
#    breaks = numpy.diff(homognum) > 5
#    breakedge = numpy.r_[-1, breaks.nonzero()[0], homog.sum()-1]
#
#    for (l, e) in zip(breakedge, breakedge[1:]):
#        p1 = m.pcolor(lon[homognum[(l+1):(e+1)], :],
#                      lat[homognum[(l+1):(e+1)], :],
#                      C[homognum[(l+1):(e+1)], :],
#                      latlon=True, **kwargs)
#    lon.mask = lat.mask = C.mask = west
#    p2 = m.pcolor(lon, lat, C, latlon=True, **kwargs)
    # remaining lines manually
#    for mix in mixed.nonzero()[0]:
#        west = lon[mix, :] <= 0
#        east = lon[mix, :] > 0
#        for h in (west, east):
#            m.pcolor(lon[mix, h],
#                     lat[mix, h],
#                     C[mix, h],
#                     latlon=True, **kwargs)
            # For some reason, if I don't include ascontiguousarray here,
            # I run into a SystemError in proj4.  I haven't been able to
            # find a minimum example that reproduces the bug :(
            #
            # And another bug: I can't pcolor a single line when using
            # latlon=True, as shiftdata will fail...
            #
            # But even when I can, it still goes wrong because pcolor
            # doesn't show the single line... :( why is masking not
            # working?

#            (x, y) = m(numpy.ascontiguousarray(lon[mix:(mix+1), h]),
#                       numpy.ascontiguousarray(lat[mix:(mix+1), h]))
#            m.pcolor(x, y, C[mix:(mix+1), h], latlon=False, **kwargs)
    return p1

def map_orbit_double_with_stats(lon, lat, C, U, lab1, lab2,  title, filename):
    """Map orbit with uncertainty and histograms
    """

    (f, a_all) = matplotlib.pyplot.subplots(2, 4,
                gridspec_kw = {'width_ratios':[12, 1, 3, 8],
                               "hspace": 0.3},
                figsize=(15, 8))

    # workaround for easy way of creating extra space...
    for a in a_all[:, 2]:
        a.set_visible(False)

    m_all = []
    for a in a_all[:, 0]:
        m = mpl_toolkits.basemap.Basemap(projection="moll",
                resolution="c", ax=a, lon_0=0)
        m.drawcoastlines()
        m.drawmeridians(numpy.arange(-180, 180, 30))
        m.drawparallels(numpy.arange(-90, 90, 30))
        m_all.append(m)

    pcr = pcolor_on_map(
        m_all[0], lon, lat,
        C, cmap="viridis")

    pcu = pcolor_on_map(
        m_all[1], lon, lat,
        U, cmap="inferno_r")

    cb1 = f.colorbar(pcr, cax=a_all[0, 1])
    cb1.set_label(lab1)#"Counts")

    cb2 = f.colorbar(pcu, cax=a_all[1, 1])
    cb2.set_label(lab2)#"Random uncertainty [counts]")

    a_all[0, 3].hist(C.ravel(), 50)
    a_all[0, 3].set_xlabel(lab1)#"Counts")
    a_all[1, 3].hist(U.ravel(), 50)
    a_all[1, 3].set_xlabel(lab2)#r"$\Delta$ Counts")
    for a in a_all[:, 3]:
        a.grid("on")
        a.set_ylabel("Number")

    f.suptitle(title)

    #f.subplots_adjust(wspace=0.2, hspace=0.2)

    print_or_show(f, False, filename)

def plotdir():
    """Returns todays plotdir.

    Configuration 'plotdir' must be set.  Value is expanded with strftime.
    """
    return datetime.date.today().strftime(config.get_config('plotdir'))

def print_or_show(fig, show, outfile, in_plotdir=True, tikz=None,
                  data=None, store_meta="", close=True,
                  dump_pickle=True):
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
        
        bs = pathlib.Path(plotdir())
        if isinstance(outfile, str):
            if outfile.endswith("."):
                outfiles = [bs / pathlib.Path(outfile+ext) for ext in ("png", "pdf")]
                infofile = bs / pathlib.Path(outfile + "info")
                figfile = bs / pathlib.Path(outfile + "pkl.xz")
            else:
                outfiles = [bs / pathlib.Path(outfile)]
                infofile = None
                figfile = None

        if infofile is not None:
            infofile.parent.mkdir(parents=True, exist_ok=True)

            logging.debug("Obtaining verbose stack info")
            pr = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE) 
            info = " ".join(sys.argv) + "\n" + pr.stdout.decode("utf-8") + "\n"
            info += tools.get_verbose_stack_description()

#        if infofile is not None and info:
            logging.info("Writing info to {!s}".format(infofile))
            with infofile.open("w", encoding="utf-8") as fp:
                fp.write(info)
        if dump_pickle and figfile is not None:
            logging.info("Writing figure object to {!s}".format(figfile))
            with lzma.open(str(figfile), "wb", preset=lzma.PRESET_DEFAULT) as fp:
                pickle.dump(fig, fp, protocol=4)
        # interpret as sequence
        for outf in outfiles:
            logging.info("Writing to file: {!s}".format(outf))
            outf.parent.mkdir(parents=True, exist_ok=True)
            fig.canvas.print_figure(str(outf))
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
