#!/usr/bin/env python

"""Describe batches of data: atmospheric db, scattering db, etc.

This module contains a class with functionality to read atmospheric data
from a variety of formats and write it to a variety of formats.

Mostly obtained from PyARTS.
"""

import sys
import abc
import copy
import random
import itertools
import pickle
import pathlib
import logging
import lzma

import numpy
import datetime
now = datetime.datetime.now

import numpy.lib.recfunctions
#import numpy.ma
import scipy.io

import matplotlib.mlab # contains PCA class
#import matplotlib.pyplot
#import matplotlib.cm

import progressbar

from . import tools
from . import stats
from . import config

class AtmosphericDatabase:
    """Represents an atmospheric database

    Apart from the default constructor, those constructors may be useful:

    - :func:`AtmosphericDatabase.from_evans07`
    - :func:`AtmosphericDatabase.from_evans12`

    Attributes:

    - ``data``
    - ``instrument``
    - ``name``
    """

    def __init__(self, **kwargs):
        """Constructor for atmospheric database

        Takes only keyword arguments, where each keyword ends up as an
        instance attribute, i.e.
        AtmosphericDatabase(instrument=pyatmlab.instruments.ici).
        """

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_evans07(cls, dbfile, instrument=None):
        """Read from db in style for Evans study in 2006-2007

        :param dbfile: Path to database-file
        :param instrument: Optional argument, pass instrument.
        """

        with open(dbfile, 'r') as fp:
            
            ncases = int(fp.readline().strip().split()[0])
            fp.readline() # ncloudpar ntemplay nrhlay nchan nztab nmutab ntinfo nuinfo
            fp.readline() # RH layers tops / bottoms
            chan_names = fp.readline().split("=")[0].strip().split()
            fp.readline() # ztab
            fp.readline() # mutab
            fp.readline() # header line

            # pre-allocate
            n_bt = len(chan_names)
            Z = numpy.zeros(ncases, dtype=cls.get_dtype(n_bt))

            i = 0
            while i<ncases:
                line = fp.readline()
                (iwp, dme, zmed, *rest) = (float(x) for x in line.strip().split())
                fp.readline() # empty line
                bt = numpy.array([float(x) for x in fp.readline().strip().split()])
                Z["BT"][i] = bt
                Z["Dme"][i] = dme
                Z["Zmed"][i] = zmed
                Z["IWP"][i] = iwp
                i += 1
                if i%(ncases//5) == 0:
                    print(now(), "done", "{}/{}".format(i, ncases))

        return cls(data=Z, instrument=instrument, name="colorado2007")

    @classmethod
    def from_evans12(cls, dbfile, instrument=None, profile=False,
                   radar=False, maxlen=None, ext=False):
        """Read from db in style for Evans study in 2012

        :param dbfile: Path to database-file
        :param instrument: Optional argument, pass instrument.
        :param profile: Include profile information.  Defaults to False.
        :param radar: Include radar information.  Defaults to False.
        :param int maxlen: Optional maximum length of db to read.
        :param bool ext: Read extended version, with manual additions by
            Gerrit 2014-01-16 (i.e. also outputting non-retrieved
            quantities used in the RTM)
        """

        print(now(), "Reading", dbfile)
        fp = open(dbfile, 'r')

        # skip and interpret header
        fp.readline() # header line
        ncases = int(fp.readline().strip().split()[0])
        maxlen = maxlen or ncases
        fp.readline() # no. of dimensions of independent Gaussian prior space
        fp.readline() # no. of elements in observation vector
        nchans = int(fp.readline().strip().split()[0])
        chan_names = fp.readline().split()
#            print(now(), "compare:", self.instrument.channel_string(), chan_names)

        # no. values per channel
        chan_lengths = numpy.array([int(c)
                                    for c in fp.readline().strip().split()])
        fp.readline() # no. of viewing angles
        fp.readline() # cos for each viewing angle
        fp.readline() # random seed
        fp.readline() # no. retrieved quantities
        fp.readline() # no. integrated cloud params
        n_layer_rh = int(fp.readline().strip().split()[0]) # no. retrieved humidity levels
        fp.readline() # height per humidity layer
        n_layer_ice = int(fp.readline().strip().split()[0]) # no. retrieved ice cloud layers
        (ice_bottom, ice_top) = (float(c) for c in
                                 fp.readline().split('!')[0].strip().split()) # bottom and top height ice cloud region
        ice_axis = numpy.linspace(ice_bottom, ice_top, n_layer_ice)
        fp.readline() # header line
        if ext:
            n_aux = int(fp.readline().strip().split()[0]) # no. aux

        chan_is_radar = numpy.array(["radar" in x.lower()
                                     for x in chan_names])

        # construct dtype
        #
        # Will have two dtypes:
        # - one where all channels are named
        # - one with a single large dtype for all radiances

        dt = [("IWP", numpy.float32),
              ("Dme", numpy.float32),
              ("Zmed", numpy.float32)]

        dt.extend([("shape", 
                    [("plateagg", numpy.float32),
                     ("sphragg", numpy.float32),
                     ("snowagg", numpy.float32),
                     ("hail", numpy.float32)])])

        if profile:
            dt.append(("IWC", numpy.float32, n_layer_ice))
            dt.append(("Dme_prof", numpy.float32, n_layer_ice))
            dt.append(("RH", numpy.float32, n_layer_rh))

        if ext:
            n_per_aux = n_aux//4
            dt.append(("height", numpy.float32, n_per_aux))
            dt.append(("temp", numpy.float32, n_per_aux))
            dt.append(("pres", numpy.float32, n_per_aux))
            dt.append(("r", numpy.float32, n_per_aux))

        dt_alt = copy.copy(dt)

        dt.extend([(nm, numpy.float32, ln)
                   for (nm, ln) in zip(chan_names, chan_lengths)])

        dt_alt.append(("BT", numpy.float32,
                        numpy.count_nonzero(~chan_is_radar)))
        dt_alt.extend([(nm, numpy.float32, ln)
                       for (nm, ln, israd) in zip(chan_names,
                                                  chan_lengths,
                                                  chan_is_radar)
                       if israd])

#        if radar:
#            dt.extend([(nm, numpy.float32, chanlen)
#                       for (nm, chanlen, israd)
#                       in zip(chan_names, chan_lengths, chan_is_radar)
#                       if israd])

        # get index in measurement array for each measurement
        edges = numpy.r_[0, chan_lengths.cumsum()]
        locations = {chan_names[k]: slice(edges[k], edges[k+1])
                     for k in range(chan_lengths.size)}

#        passive = numpy.array([ln
#                               for ln in chan_lengths
#                               if not "radar" in nm.lower()]

        #n_bt = nchans - sum(["radar" in x.lower() for x in chan_names])

        # pre-allocate
        data = numpy.empty(maxlen, dtype=dt)
        data_alt = data.view(dtype=dt_alt)

        # use while, not for-loop, because I read several lines per
        # iteration
        i = 0
        while i<maxlen:
            line = fp.readline()
            if not line: # empty means EOF
                break
#            if i%nth != 0: # allow for reading less
#                continue
            (rvcheck, IWP, Dme, Zmed,
             plateagg, sphragg, snowagg, hail,
             meltLWP, cldLWP) = (float(x) for x in line.strip().split())
            line = fp.readline()
            RH = numpy.array([float(x) for x in line.strip().split()])
            line = fp.readline()
            IWC = numpy.array([float(x) for x in line.strip().split()])
            line = fp.readline()
            Dmeprof = numpy.array([float(x) for x in line.strip().split()])
            line = fp.readline()
            measurements = numpy.array([float(x)
                                        for x in line.strip().split()])
            if ext:
                line = fp.readline()
                aux = numpy.array([float(x) for x in
                                   line.strip().split()])
#            BT = numpy.array(measurements[~chan_is_radar])
#            radar = numpy.array(measurements[chan_is_radar])
    #        radar_integrated = numpy.array(measurements[9])
    #        radar_prof = numpy.array(measurements[10:])
            #print "measurements:", BT
            #print "IWP:", IWP
            #print "IWC:", IWC
            data["IWP"][i] = IWP
            data["Dme"][i] = Dme
            data["Zmed"][i] = Zmed
#            data["BT"][i] = BT
            data["shape"][i]["plateagg"] = plateagg
            data["shape"][i]["sphragg"] = sphragg
            data["shape"][i]["snowagg"] = snowagg
            data["shape"][i]["hail"] = hail

            for nm in chan_names:
                data[nm][i] = measurements[locations[nm]]

            if profile:
                data["IWC"][i] = IWC
                data["Dme_prof"][i] = Dmeprof
                data["RH"][i] = RH

            if ext:
                data["height"][i] = aux[0*n_per_aux:1*n_per_aux]
                data["temp"][i] = aux[1*n_per_aux:2*n_per_aux]
                data["pres"][i] = aux[2*n_per_aux:3*n_per_aux]
                data["r"][i] = aux[3*n_per_aux:4*n_per_aux]
#            if radar:
#                for radfield in chan_is_radar.nonzero()[0]:
#                    nm = chan_names[radfield]
#                    data[nm][i] = measurements
            i += 1
            if i%(maxlen//8) == 0:
                print(now(), "done", "{}/{}, {}%".format(i, maxlen,
                                                         (i/maxlen)*100))

        return cls(data=data_alt, instrument=instrument,
                   name="colorado2012", ice_axis=ice_axis)


    @tools.validator
    def __getitem__(self, key: str):
        """Get field from data.

        :param str key: Field name (from self.data["dtype"]).
        :returns: ndarray, view from self.data
        """
        return self.data[key]

    @tools.validator
    def __setitem__(self, key: str, val: numpy.ndarray):
        """Add field to data or set existing field to data.

        :param str key: Field name
        :param ndarray val: Field value.  Must match dimensions of
            self.data.  Note that this will be added with dtype([key,
            val.dtype]).
        """

        if key in self.data.dtype.names:
            self.data[key] = val
        else:
            prim = self.data
            sec = val.view(dtype=[(key, val.dtype)])
            self.data = numpy.lib.recfunctions.merge_arrays(
                (prim, sec)).view(dtype=(prim.dtype.descr + sec.dtype.descr))
            

    # calculating different statistics

#     def stats_IWP_Dme(self, func, minlen=20, *args, **kwargs):
#         """Calculate a statistic per logIWP/Dme
# 
#         :param func: Statistical function to apply for each bin
#         :param minlen: Mask statistic if number of values per bin are less
#             than this.  Defaults to 20.
# 
#         All remaining arguments are passed on to
#         :func:`tools.filter_array`.
# 
#         Returns (xbin, ybin, stats)
#         """
# 
#         xbin = numpy.linspace(-1, 4, 20)
#         ybin = numpy.linspace(0, 650, 20)
#         #pos = self.data["IWP"] > 0
#         minlen = 20
#         nchans = self.data["BT"].shape[1]
#         filler = numpy.empty(nchans)
#         filler.fill(numpy.nan)
#         filt = tools.filter_array(self.data, IWP=(1e-5, 1e10),
#                                   *args, **kwargs)
#         binned_data = tools.bin2d(self.data[filt], "IWP", xbin, "Dme", ybin,
#                                   filter1=numpy.log10)
#         binned_stats = (numpy.array([
#             [func(b["BT"]) if len(b["BT"])>minlen else filler for b in r]
#                 for r in binned_data]))
#         return (xbin, ybin, numpy.ma.masked_invalid(binned_stats))
# 
#     def stats3d(self, func, minlen=20):
#         """Calculate statistic for 3-D bins per IWP, Dme, Zmed
#         """
# 
#         bin_iwp = numpy.linspace(-1, 4, 14)
#         bin_dme = numpy.linspace(0, 650, 15)
#         bin_zmed = numpy.linspace(0, 20, 16)
# 
#         binned = tools.bin3d(self.data,
#            "IWP", bin_iwp,
#            "Dme", bin_dme,
#            "Zmed", bin_zmed,
#            filter1=numpy.log10)
# 
#         nchans = self.data["BT"].shape[1]
#         filler = numpy.empty(nchans)
#         filler.fill(numpy.nan)
# 
#         stats3d = numpy.array([[
#             [func(e["BT"]) if len(e["BT"])>minlen else filler for e in r]
#                 for r in c]
#                     for c in binned])
# 
#         return (bin_iwp, bin_dme, bin_zmed, stats3d)
#  
#     def shape_per_iwp(self):
#         """Get distribution of shapes as a function of IWP
#         """
# 
#         bins_iwp = numpy.linspace(-1, 4, 14)
#         filt = self.data["IWP"] > 0
#         binned = tools.bin(numpy.log10(self.data["IWP"][filt]),
#                            self.data[filt], bins_iwp)
#         shape_dist = numpy.empty_like(bins_iwp, dtype = self.data.dtype["shape"])
#         for shape in self.data.dtype["shape"].names:
#             shape_dist[shape] = [v["shape"][shape].mean() for v in binned]
#         return shape_dist
# 
#     # visualisations
# 
#     def plot_stat_IWP_Dme(self, statfunc, ch,
#                           minlen=20, statname=None, ax=None,
#                           *args, **kwargs):
#         """Visualise statistic IWP Dme in a hist2d.
# 
#         Requires instrument to be defined.
# 
#         Input:
#             - statistic (callable)
#             - channel
#             - minlen, statname, ax
#             - all other passed to tools.filter_array
# 
#         Optional input:
# 
#             - minlen
#             - statname
#             - axes object
# 
#         Returns:
# 
#             fig, ax, pcolormesh, colorbar
#         """
# 
#         if statname is None:
#             statname = statfunc.__name__
# 
#         if ax is None:
#            fig = matplotlib.pyplot.figure()
#            ax = fig.add_subplot(1, 1, 1,
#                xlabel="${}^{10}$log IWP [g/m^2]",
#                xscale="log",
#                ylabel="Dme [µm]",
#                title="{2} {0.name} {0.instrument.name} {1:s} GHz".format(
#                    self,
#                    self.instrument.channels[ch-1].get_chanstr(full=True),
#                    statname))
#        else:
#            fig = ax.figure
#
#        (xbin, ybin, stat) = self.stats_IWP_Dme(statfunc, minlen=minlen,
#                                *args, **kwargs)
#        cmap = matplotlib.cm.Spectral_r
#        cmap.set_bad(color="white", alpha=None)
#        #cmap.set_over(color="black")
#        #cmap.set_under(color="cyan")
#        pm = ax.pcolormesh(10**xbin, ybin, stat[..., ch-1].T, cmap=cmap)
#        #pm.set_clim([0, 15])
#        cb = fig.colorbar(pm)
#        #cb.set_label("interquantile range [K]")
#        
#        return (fig, ax, pm, cb)
#
#    def plot_lines_stat3d(self, statfunc):
#        (bins_iwp, bins_dme, bins_zmed, stat) = self.stats3d(statfunc)
#        isfin = numpy.isfinite(stat[..., 0])
#        fin_in_iwp = isfin.sum(2).sum(1)
#        fin_in_dme = isfin.sum(2).sum(0)
#        fin_in_zmed = isfin.sum(1).sum(0)
#        for (iwp_i, iwp) in enumerate(bins_iwp):
#            if fin_in_iwp[iwp_i] > 10:
#                # one zmed-line per dme for this iwp
#                plot(bins_dme, stat[iwp_i, ..., ch])
#                # one dme-line per zmed for this iwp
#                plot(bins_zmed, stat[iwp_i, ..., ch].T)
##            for (dme_i, dme) in bins_dme:
##                for (zmed_i, zmed) in bins_zmed:
##                    pass
#           

    # writing out the results in different ways

    def write_evans_obs(self, fn, sel=None):
        """Write as evans-obs file

        :param fn: File to write to
        :param sel: Selection to write to file, None (default) to write
            all
        """

        # from iceprofret.txt in the Evans distribution:
        #
        # Six header lines:
        #    Number of pixels
        #    Number of channels, total number of elements
        #    Channel IDs or names (matches channel IDs in retrieval database)
        #    number of elements in each channel
        #    additive uncertainties for each element
        #    multiplicative uncertainties for each element
        # For each line:
        #    Time(hours)  Cosine viewing zenith angle   Measurement for each element


        if sel is None:
            sel = numpy.arange(self.data.size)

        with open(fn, 'wt') as fp:
            # number of pixels
            fp.write("{}\n".format(sel.size))
            # no channels / no elements; both equal to size of BT
            fp.write("{} {}\n".format(self.data.dtype["BT"].shape[0],
                                      self.data.dtype["BT"].shape[0]))
            # names of channels
            fp.write("{}\n".format(self.instrument.channel_string(pre="im",
                                                                  width=True)))

            # no. element per channel
            fp.write(("1 " * len(self.instrument.channels)).strip())
            fp.write("\n")

            # additive uncertainties
            for chan in self.instrument.channels:
                fp.write(str(chan.noise) + " ")
            fp.write("\n")

            # multiplicative uncertainties (not implemented)
            for chan in self.instrument.channels:
                fp.write("0 ")
            fp.write("\n")

            for elem in sel:
                # time; n/a, so write fake
                fp.write("0.0 ")
                # cos(angle); only nadir implented
                fp.write("1.0 ")
                # write channel BT's
                for chan_bt in self.data[elem]["BT"]:
                    fp.write(str(chan_bt) + " ")
                fp.write("\n")


    def write_mat(self, fn, fields=None, sel=None):
        """Write to MAT file.

        Use :func:`scipy.io.savemat` to write the database to a
        MATLAB-style .mat file.
        
        :param fn: Filename to write to
        :param fields: List of fieldnames to write.  Must be subset of
            ``self.data.dtype.names``.  Defaults to ``None``, which means
            to write all fields.
        :param sel: Indices to write.  Defaults to ``None``, which means
            all scenes.
        """
        if fields is None:
            fields = list(self.data.dtype.names)
        if sel is None:
            sel = slice(None)
        print(now(), "Writing to {}".format(fn))
        scipy.io.savemat(fn, dict(data=self.data[fields][sel],
                                  ice_axis=self.ice_axis),
                         appendmat=False,
                         do_compression=True, oned_as="column")

class LookupTable(abc.ABC):
    """Use a lookup table to consider similar measurements

    This table is used to represent a large set of measurements by a small
    set.  It takes as input a particular measurement, and returns as
    output a canonical measurements.  A measurement is n-dimensional,
    consisting of lat, lon, time of day, day of year, partial column,
    degrees of freedom.

    A use case is when we have a large set of measurements, but only error
    estimates for a subset of those.

    Implemented using a lookup table based on stats.bin_nd.  Bins are
    based on training data.  If newly presented data does not look like
    any pre-trained data, an error is raised.

    Binning based on PCAs is currently being implemented.

    Attributes::

        axdata.  Dictionary with keys corresponding to the axes to be
            considered in the lookup table; names should correspond to
            fields in data.  Each value is itself a dictionary with keys::

                nsteps: Number of steps in linspace

        bins
        db

    To create an instance, use either .fromData or .fromFile (if
    available).

    """

    #_loaded = False
    axdata = bins = db = None

    use_pca = False


    def compact_summary(self):
        """Return string with compact summary

        Suitable in filename
        """

        if self.use_pca:
            s = "PCA_{:s}_{:d}_{:.1f}".format(
                ",".join(self.axdata["PCA"]["fields"]),
                self.axdata["PCA"]["npc"],
                self.axdata["PCA"]["scale"])
        else:
            s = "-".join(
                    ["{:s}_{:d}".format(k,v["nsteps"])
                        for (k, v) in sorted(self.axdata.items())])
        return s

    def __repr__(self):
        return "<{}:{}>".format(self.__class__.__name__,
            self.compact_summary())


    @property
    def fields(self):
        if self.use_pca:
            return self.axdata["PCA"]["fields"]
        else:
            return self.axdata["fields"]

    def get_index_tuple(self, dat, full=False):
        """Get a tuple of indices for use in the lookup table

        Returns either the full tuple, or only the elements
        considering to the number of PCs considered originally.
        """
        if self.use_pca:
            fields = self.axdata["PCA"]["fields"]
            t = tuple(
                stats.bin_nd_sparse(self.pca.project(numpy.vstack(
                                        [dat[ax] for ax in fields]).T),
                                    self.bins).squeeze().tolist())
            if not full:
                t = t[:self.axdata["PCA"]["npc"]]

            return t
        else:
            fields = list(self.axdata.keys())
            return tuple(stats.bin_nd_sparse(
                    numpy.atleast_2d([dat[ax]
                    for ax in fields]), self.bins).squeeze().tolist())

    def get_index_tuples(self, data):
        """Yield tuples of indices for use in the lookup table
        """
        # FIXME: can be faster when `dat` is large
        for dat in data:
            yield (self.get_index_tuple(dat), dat)

       
    def lookup_all(self, data):
        # FIXME: can be faster when dat is large
        logging.info("Looking up {:d} radiances".format(data.size))
        bar = progressbar.ProgressBar(maxval=data.size,
            widgets=[progressbar.Bar("=", "[", "]"), " ",
                     progressbar.Percentage()])
        bar.start()
        for (i, dat) in enumerate(data):
            try:
                yield self.lookup(dat)
            except KeyError:
                logging.error("Not found for no. {:d}. :( "
                    "Should implement lookaround, enlarge LUT, or make it denser!".format(i))
                continue
                #yield None
        #yield from (self.lookup(dat) for dat in data)
            bar.update(i)
        bar.finish()

    def lookaround(self, dat):
        """Yield all neighbouring datapoints

        Look at all neighbouring datapoints.  NB those may be 2^N where N is
        the length of tup!  Very slow!
        """
        tup = self.get_index_tuple(dat)
        manytup = itertools.product(*[range(i-1,i+2) for i in tup])
        yield from (t for t in manytup if t in self.db)

    def _get_bins(self, data, axdata, pca=False):
        bins = []
        if pca:
            # This means axdata has a single key "PCA" with fields
            # “scale”.  It also means that `data` is in PCA space, i.e.
            # pca.Y.
#            rmin = numpy.nanmin(data, 0)
#            rmin -= 0.001*abs(rmin)
#            rmax = numpy.nanmax(data, 0)
#            rmax += 0.001*abs(rmax)
            # number of bins per PC
            nbins = numpy.ceil(self.pca.fracs*100*axdata["PCA"]["scale"])
            bins = [numpy.linspace(data[:, i].min(),
                                   data[:, i].max(),
                                   max(p, 2))
                        for (i, p) in enumerate(nbins)]
            return bins[:axdata["PCA"]["npc"]]
#            b = [self._get_bins_from_range(rmi, rma, axdata, "PCA")
#                    for (rmi, rma) in zip(rmin, rmax)]
#            raise NotImplementedError("Not implemented yet!")
        else:
            for ax in axdata.keys():
                if "range" in axdata[ax].keys():
                    (rmin, rmax) = axdata[ax]["range"]
                else:
                    rmin = numpy.nanmin(data[ax])
                    rmin -= 0.001*abs(rmin)
                    rmax = numpy.nanmin(data[ax])
                    rmax += 0.001*abs(rmax)

                b = self._get_bins_from_range(rmin, rmax, axdata, ax)
#                for case in tools.switch(axdata[ax].get("mode", "linear")):
#                    if case("linear"):
#                        b = numpy.linspace(rmin, rmax, axdata[ax]["nsteps"])
#                        break
#                    if case("optimal"):
#                        inrange = (data[ax] >= rmin) & (data[ax] <= rmax)
#                        b = scipy.stats.scoreatpercentile(data[ax][inrange],
#                                numpy.linspace(0, 100, axdata[ax]["nsteps"]))
#                        break
#                    if case():
#                        raise ValueError("ax {!s} unknown mode {!s}, I know "
#                            "'linear' and 'optimal'".format(axdata[ax], axdata[ax]["mode"]))
                bins.append(b)
            return bins
            # end for
        # end if

    @staticmethod
    def _get_bins_from_range(rmin, rmax, axdata, ax):
        """Small helper for _get_bins.

        From extrema and `axdata` description, get either linearly or
        logarithmically spaced bins.
        """
        for case in tools.switch(axdata[ax].get("mode", "linear")):
            if case("linear"):
                b = numpy.linspace(rmin, rmax, axdata[ax]["nsteps"])
                break
            if case("optimal"):
                inrange = (data[ax] >= rmin) & (data[ax] <= rmax)
                b = scipy.stats.scoreatpercentile(data[ax][inrange],
                        numpy.linspace(0, 100, axdata[ax]["nsteps"]))
                break
            if case():
                raise ValueError("ax {!s} unknown mode {!s}, I know "
                    "'linear' and 'optimal'".format(ax, axdata[ax]["mode"]))
        return b

    @staticmethod
    def _make_pca(data, axdata):
        
        fields = axdata["PCA"]["fields"]
        valid_range = axdata["PCA"]["valid_range"]
        if not all([issubclass(data[x].dtype.type, numpy.floating)
                        for x in fields]):
            logging.warning("Casting all data to float64 for PCA")

        data_mat = numpy.vstack([data[x] for x in fields]).T
        valid = numpy.all((data_mat > valid_range[0]) &
                          (data_mat < valid_range[1]), 1)
        return matplotlib.mlab.PCA(data_mat[valid, :])

    def lookup(self, dat):
        tup = self.get_index_tuple(dat)
        return self[tup]

    @staticmethod
    @abc.abstractmethod
    def _choose():
        ...

    @classmethod
    @abc.abstractmethod
    def fromData(cls):
        ...

    @abc.abstractmethod
    def __getitem__(self):
        ...

    @abc.abstractmethod
    def __setitem__(self):
        ...

class SmallLookupTable(LookupTable):
    """Lookup table small enough to be in memory
    """
    @classmethod
    def fromFile(cls, file):
        with open(file, 'rb') as fp:
            (axdata, bins, db) = pickle.load(fp)
        self = cls()
        self.axdata = axdata
        self.bins = bins
        self.db = db
        #self._loaded = True
        return self

    def propose_filename(self):
        return "similarity_db_{}".format(self.compact_summary())

    def toFile(self, file):
        """Store lookup table to a file
        """
        with open(file, 'wb') as fp:
            pickle.dump((self.axdata, self.bins, self.db), fp,
                    protocol=4)

    @classmethod
    def fromData(cls, data, axdata):
        """Build lookup table from data

        ``data`` should be a structured ndarray with dtype fields

        ``axdata`` should be a ``collections.OrderedDict`` where the keys
        refer to fields from `data` to use, and the values are
        dictionaries with the keys.  If regular binning is used (i.e. no
        PCA), those keys are:

            nsteps (mandatory)
                number of steps in binning data
            mode
                string that can be either "linear" (use linspace between
                extremes) or "optimal" (choose bins based on percentiles so
                1-D binning would create equal content in each).
                Defaults to linear.
            range
                tuple with (min, max) of range within which to bin data.
                Defaults to extremes of data.

        
        It is also possible to bin based on PCA.  In this case, ``axdata''
        should have a single key "PCA".  When binning based on PCA, the
        number of bins per PC are proportional the the proportion of
        variance along each PC axis (pca.fracs).  By default, the number
        of bins is the % of variability associated with the axis, i.e. if
        the first PC explains 67% of variability and the second 25%, there
        will be 67 and 25 bins, respectively.  This can be scaled by
        setting the key `scale` to something other than one.

            fields
                Sequence of strings: what fields to use in PCA-based
                analysis.

            npc
                Integer, how many PCs to consider

            scale
                Float, defaults to 1.0, for scaling the number of bins.

            valid_range
                Before performing PCA, require that ALL input vectors are
                within this range, otherwise discard.

        """
        self = cls()
        self.use_pca = list(axdata.keys()) == ["PCA"]
        self.pca = None
        if self.use_pca:
            # _make_pca considers axdata["PCA"]["fields"]
            self.pca = self._make_pca(data, axdata)
            # _get_bins considers axdata["PCA"]["npc"]
            #                 and axdata["PCA"]["scale"]
            bins = self._get_bins(self.pca.Y, axdata, pca=True)
            binned_indices = stats.bin_nd(
                [self.pca.Y[:, i] for i in range(self.pca.Y.shape[1])])
        else:
            fields = axdata.keys()
            bins = self._get_bins(data, axdata, pca=False)
            binned_indices = stats.bin_nd(
                [data[ax] for ax in fields], bins)
        db = {}
        # Do something for every bin.  `_choose` is implemented in another
        # class, it might do a count, it might choose one, or it might
        # choose all.
        for ii in itertools.product(*(range(i) for i in
                binned_indices.shape)):
            # ii should be a tuple that can be passed directly
            if binned_indices[ii].size > 0:
                db[ii] = data[self._choose(binned_indices[ii])]

        self.axdata = axdata
        self.bins = bins
        self.db = db
        #self._loaded = True
        return self

    def __getitem__(self, tup):
        return self.db[tup]

    def __setitem__(self, tup, val):
        self.db[tup] = val

    def keys(self):
        yield from self.db.keys()
 
class LargeLookupTable(LookupTable):
    """Lookup table too large in memory, mapped to directory

    """
    basename = "bucket/{coor:s}/contents.npy.xz"
    _db = {}
    _maxcache = 1e9 # 100 MB
    _N = 0

    def propose_dirname(self):
        return "large_similarity_db_{}".format(self.compact_summary())

    @classmethod
    def fromData(cls, data, axdata, use_pca=False):
        # docstring copied from SmallLookupTable

        self = cls()
        self.use_pca = use_pca
        if use_pca:
            self.pca = self._make_pca(data, axdata)
            bins = self._get_bins(self.pca.Y, axdata, pca=True)
        else:
            bins = self._get_bins(data, axdata, pca=False)
        self.axdata = axdata
        self.bins = bins
        if not self.bucket_dir().is_dir():
            self.bucket_dir().mkdir(parents=True)
        self.storemeta()
        self.addData(data)
        return self
    fromData.__doc__ = SmallLookupTable.__doc__

    def addData(self, data):
        """Add a lot of data
        """
        k = set()
        bar = progressbar.ProgressBar(maxval=len(data),
            widgets=[progressbar.Bar("=", "[", "]"), " ",
                    progressbar.Percentage()])
        bar.start()
        for (i, (t, contents)) in enumerate(self.get_index_tuples(data)):
            if t in k:
                cur = self[t]
#                contents = contents[numpy.array([contents[i] not in cur
#                        for i in range(contents.size)])]
#                if contents.size > 0:
                if contents not in cur:
                    self[t] = numpy.hstack((self[t], numpy.atleast_1d(contents)))
            else:
                self[t] = numpy.atleast_1d(contents)
            k.add(t)
            bar.update(i+1)
        bar.finish()
        self.dumpcache() # store and clear
        self.storemeta()

    @classmethod
    def fromDir(cls, arg):
        """Initialise from directory

        `arg` can be either a directory (str or pathlib.Path), or a
        dictionary of axdata describing such (see fromData docstring).
        """
        self = cls()

        if isinstance(arg, dict):
            self.axdata = arg
            self.use_pca = "PCA" in arg
            dir = self.bucket_dir()
        else:
            dir = arg

        with (pathlib.Path(dir) / "info.npy").open(mode="rb") as fp:
            logging.info("Reading into {!s}".format(pathlib.Path(dir)))
            (self.axdata, self.bins) = pickle.load(fp)
        if "PCA" in self.axdata:
            self.use_pca = True
            with (pathlib.Path(dir) / "pca.npy").open(mode="rb") as fp:
                self.pca = pickle.load(fp)
        return self

    def keys(self):
        """Yield keys one by one.  Reads from directory, no caching!
        """
        for p in self.bucket_dir().iterdir():
            if p.name.startswith("bucket_") and p.name.endswith(".npy.xz"):
                yield tuple(int(s) for s in p.name[7:-7].split("-"))

    def __setitem__(self, tup, data):
        self._db[tup] = data
        if len(self._db) > self._N: # recalculate size
            totsize = sum(v.nbytes for v in self._db.values())
            if totsize > self._maxcache:
                logging.debug("Size {:,} exceeds max cache {:,}, "
                              "dumping {:d} keys".format(totsize,
                              self._maxcache, len(self._db)))
                self.dumpcache()
            else:
                self._N += 10 # i.e. after every 10 new entries, check size

    def __getitem__(self, tup):
        if tup in self._db: # cached
            return self._db[tup]
        else:
            path = self.bucket_name(tup)
            if not path.exists():
                raise KeyError("No entry for {!s}".format(tup))
            with lzma.open(str(path), mode="rb") as fp:
                try:
                    v = numpy.load(fp)
                except Exception as e:
                    raise type(e)(str(e) + " while reading {!s}".format(
                        path)).with_traceback(sys.exc_info()[2])
            self._db[tup] = v
            return v

    def dumpcache(self):
        sizes = [v.size for v in self._db.values()]
        logging.info("Dumping cache for {:,} profiles in {:d} buckets to {!s}".format(
                    sum(sizes), len(self._db), self.bucket_dir()))
        bar = progressbar.ProgressBar(maxval=len(self._db), 
                    widgets=[progressbar.Bar('=', '[', ']'), ' ',
                    progressbar.Percentage()])
        bar.start()
        newdirs = 0
        counts = numpy.zeros(dtype=numpy.uint32, shape=(max(sizes)+1),)
        for (i, (k, v)) in enumerate(self._db.items()):
            path = self.bucket_name(k)
            if not path.parent.is_dir():
                #logging.info("Creating directory {!s}".format(path.parent))
                path.parent.mkdir(parents=True)
                newdirs += 1
#            if v.nbytes > 1e6:
#                logging.debug("Storing {:d}/{:d}, {:,} bytes to {!s}".format(
#                    i, len(self._db), v.nbytes, path))
            counts[v.size] += 1
            with lzma.open(str(path), mode="wb") as fp:
                numpy.save(fp, v)
            bar.update(i+1)
        bar.finish()
        logging.info("Stored cache.  Created {:d} new directories. "
                     "Profiles per bucket histogram: {!s}".format(newdirs, counts))

        self.clearcache()

    def storemeta(self):
        """Store metadata for database
        """
        d = self.bucket_dir()
        with (d / "info.npy").open(mode="wb") as fp:
            pickle.dump((self.axdata, self.bins), fp, protocol=4)
        if self.use_pca:
            with (d / "pca.npy").open(mode="wb") as fp:
                pickle.dump(self.pca, fp, protocol=4)

    def loadmeta(self):
        """Load metadata for database
        """
        d = self.bucket_dir()
        with (d / "info.npy").open(mode="rb") as fp:
            (self.axdata, self.bins) = pickle.load(fp)

    def clearcache(self):
        self._db.clear()
        self._N = 0

    def bucket_dir(self):
        return (pathlib.Path(config.conf["main"]["lookup_table_dir"]) /
                self.propose_dirname())

    def bucket_name(self, coor):
        """Return full path to bucket at coor
        """
        return (self.bucket_dir() /
                self.basename.format(coor="/".join("{:02d}".format(x) for x in coor)))

class SimilarityLookupTable(LookupTable):
    def propose_filename(self):
        return "tanso_similarity_db_{}".format(self.compact_summary())

    @staticmethod
    def _choose(data):
        """Choose one of the data to use for building the db
        """

        return random.choice(data)


class FullLookupTable(LookupTable):
    """Like a similarity lookup table, but keeps all entries

    """

    @staticmethod
    def _choose(data):
        return data


class CountingLookupTable(LookupTable):
    """Provide counting only, effectively creating a histogram.
    """

    @staticmethod
    def _choose(data):
        return data.size

class SmallSimilarityLookupTable(SmallLookupTable, SimilarityLookupTable):
    pass

class LargeSimilarityLookupTable(LargeLookupTable, SimilarityLookupTable):
    pass

class SmallFullLookupTable(SmallLookupTable, FullLookupTable):
    pass

class LargeFullLookupTable(LargeLookupTable, FullLookupTable):
    pass

class SmallCountingLookupTable(SmallLookupTable, CountingLookupTable):
    pass
