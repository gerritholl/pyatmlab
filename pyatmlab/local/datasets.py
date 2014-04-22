"""Specific datasets
"""

from ..datasets import (TansoFTS, NDACCAmes, ACEFTS)
import pathlib
import datetime

datadir = pathlib.Path("/home/gerrit/sshfs/glacier/data/1/gholl/data")

tanso = TansoFTS(
    start_date = datetime.datetime(2010, 3, 23, 2, 24, 54, 210),
    end_date = datetime.datetime(2010, 10, 31, 20, 34, 50, 814),
    srcfile = (datadir / "1403050001" /
               "GOSATTFTS20100316_02P02TV0001R14030500010.h5"),
    name = "Tanso FTS")

ndacc_ames_eureka = NDACCAmes(basedir=datadir / "BrukerIFS",
    start_date = datetime.datetime(2006, 8, 1, 0, 0, 0),
    end_date = datetime.datetime(2010, 10, 1, 0, 0, 0),
    granule_cache_file = "times.dat",
    name = "NDACC Eureka")

acefts = ACEFTS(basedir=datadir / "ACE",
    start_date = datetime.datetime(2006, 1, 1, 0, 0, 0),
    end_date = datetime.datetime(2012, 10, 1, 0, 0, 0),
    granule_cache_file = "times.dat",
    name = "ACE FTS")
