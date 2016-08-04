"""Module containing classes abstracting datasets
"""

import abc
import functools
import itertools
import logging
import pathlib
import re
import shelve
import string
import sys
import shutil
import tempfile
import copy

import datetime
import numpy
import numpy.lib.arraysetops
import numpy.lib.recfunctions

import pytz

import typhon.datasets.dataset

from . import tools
from . import time as atmtime
from . import physics
from . import config

class DataFileError(Exception):
    """Superclass for any datafile issues
    """

class InvalidFileError(DataFileError):
    """Raised when the requested information cannot be obtained from the file
    """

class InvalidDataError(DataFileError):
    """Raised when data is not how it should be.
    """

class Dataset(typhon.datasets.dataset.Dataset):
    timezone = pytz.UTC

    def __init__(self, *, memory=None, **kwargs):
        # set memorisation on methods.  See note on "caching methods" at
        # https://pythonhosted.org/joblib/memory.html
        tools.setmem(self, memory)
        super().__init__(**kwargs)

    @tools.mark_for_disk_cache(
        process=dict(
            my_data=lambda x: x.view(dtype="i1")))
    def combine(self, my_data, other_obj):
        return super().combine(my_data, other_obj)

    @staticmethod
    def extend_with_doy_localtime(M):
        """Calculate DOY and mean local solar time

        :param M: ndarray with dtype as returned by self.read 
        :returns M: ndarray with dtype extended with doy and mlst
        """

        (all_doy, all_mlst) = (numpy.array(x) for x in zip(
            *[atmtime.dt_to_doy_mlst(M[i]["time"].astype(datetime.datetime),
                                     M[i]["lon"])
                for i in range(M.shape[0])]))
        return numpy.lib.recfunctions.append_fields(M, 
            names=["doy", "mlst"],
            data=[all_doy, all_mlst],
            dtypes=["u2", "f4"])
    
    @staticmethod
    def extend_with_dofs(M):
        """Calculate DOFs

        :param M: ndarray with dtype as for self.read
        :returns M: M with added field dof
        """

        return numpy.lib.recfunctions.append_fields(M,
            names=["dof"],
            data=[physics.AKStats(M["ch4_ak"],
                    name="DUMMY").dofs()],
            dtypes=["f4"])

    def flag(self, arr):
        """Must be implemented by child
        """
        return arr

class SingleFileDataset(Dataset, typhon.datasets.dataset.SingleFileDataset):
    pass

class MultiFileDataset(Dataset, typhon.datasets.dataset.MultiFileDataset):
    pass

class SingleMeasurementPerFileDataset(MultiFileDataset, typhon.datasets.dataset.SingleMeasurementPerFileDataset):

    _head_dtype = dict(CH4_total="<f4")
    def _read(self, p, fields="all"):
        super()._read(p, fields)

class HomemadeDataset(MultiFileDataset, typhon.datasets.dataset.HomemadeDataset):
    pass

class ProfileDataset(Dataset):
    """Abstract superclass with stuff for anything having profiles

    Additional attributes compared to its parent::

    - range::

        If applicable, a tuple with (lo, hi) vertical limits of
        sensitivity range in metre.

    """

    # source of profile sizes.  Can be a number (fixed size) or a string
    # (field name to get size from)
    n_prof = "p"

    # does the A priori need converting?  It does for PEARL, does not for
    # others
    A_needs_converting = tools.NotTrueNorFalse
    A_needs_swapping = tools.NotTrueNorFalse

    range = None

    #@tools.mutable_cache(maxsize=10)
    def read(self, f=None, fields="all"):
        M = super().read(f, fields)
        if isinstance(self.n_prof, str):
            n_prof = M.dtype[self.n_prof].shape
        else:
            n_prof = self.n_prof
        if not "z" in M.dtype.names:
            
            newM = numpy.empty(shape=M.shape,
                dtype=M.dtype.descr + [("z", "f4", n_prof)])
            for nm in M.dtype.names:
                newM[nm] = M[nm]
            logging.debug("Adding {:d} z-profiles".format(M.shape[0]))
            for i in range(M.shape[0]):
                newM["z"][i, :] = self.get_z(M[i])
            M = newM
        return M

    def get_z(self, dt):
        """Get z-profile in metre.

        Takes as argument a single measurement, with the dtype returned by
        its own reading routine.
        """
        return dt["z"]

    def get_z_for(self, dt, field):
        """Get z-profile for particular field.

        Takes as argument a single measurement and a field name.  This is
        different from get_z because sometimes, some fields are on
        different grids (such as T for Tanso).
        """
        return self.get_z(dt)

class StationaryDataset(Dataset):
    """Abstract superclass for any ground-station dataset
    """

    unique_fields = {"time"}
