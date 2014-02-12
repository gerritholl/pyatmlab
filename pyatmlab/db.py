#!/usr/bin/env python

"""Describe batches of data: atmospheric db, scattering db, etc.

This module contains a class with functionality to read atmospheric data
from a variety of formats and write it to a variety of formats.

Mostly obtained from PyARTS.
"""

import copy

import numpy
import datetime
now = datetime.datetime.now

import numpy.lib.recfunctions
#import numpy.ma
import scipy.io

#import matplotlib.pyplot
#import matplotlib.cm

#from . import arts_math
from . import tools

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
