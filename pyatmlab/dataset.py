"""Module containing classes abstracting datasets
"""

import abc
import datetime
import numpy

class Dataset(metaclass=abc.ABCMeta):
    """Represents a dataset.

    """

    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, k, v):
        if hasattr(self, k):
            object.__setattr__(self, k, v)
        else:
            raise AttributeError("Unknown attribute: {}. ".format(k))

    @abc.abstractmethod
    def granules_for_period(self, start=datetime.datetime.min,
                                  end=datetime.datetime.max):
        """Loop through all granules for indicated period.

        This is a generator that will loop through all granules from
        `start` to `end`, inclusive.

        :param datetime start: Starting datetime, defaults to any
        :param datetime end: Ending datetime, defaults to any
        """
        raise NotImplementedError()


    def read_period(self, start=datetime.datetime.min,
                          end=datetime.datetime.max):
        """Read all granules between start and end, in bulk.

        :param datetime start: Starting time, None for any time
        :param datetime end: Ending time, None for any time
        :returns: Masked array with all data in period.
        """

        return numpy.ma.concatenate(list(
            self.read(f) for f in self.granules_for_period(start, end)))

    def read_all(self):
        """Read all data in one go.

        Warning: for some datasets, this may cause memory problems.
        """

        return self.read_period()
            
    @abc.abstractmethod
    def read(self, f):
        """Read granule in file.

        :param str f: Path to file
        """

        raise NotImplementedError()

class SingleFileDataset(Dataset):
    start_date = datetime.datetime.min
    end_date = datetime.datetime.max
    srcfile = None

    def granules_for_period(self, start=datetime.datetime.min,
                            end=datetime.datetime.max):
        if start < self.end_date and end > self.start_date:
            yield self.srcfile


class HomemadeDataset(Dataset):
    """For any dataset created by pyatmlab.

    No content yet.
    """
    pass
