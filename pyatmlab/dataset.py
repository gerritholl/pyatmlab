"""Module containing classes abstracting datasets
"""

import abc

@abc.ABCMeta
class Dataset:
    """Represents a dataset.

    """

    wanted = set()
    wanted_all = set()

    def __init__(self):
        super().__init__()
        for obj in self.__class__.mro():
            if hasattr(obj, "wanted"):
                self.wanted_all &= self.wanted

    def __setattr__(self, k, v):
        if k in self.wanted_all:
            object.__setattr__(self, k, v)
        else:
            raise AttributeError(("Unknown attribute: {}. " +
                "Known attributes: {}.").format(k, ", ".join(self.wanted_all)))

    @abstractmethod
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
        """

        return numpy.concatenate(list(
            self.read_granule(f) for f in self.granules_for_period(start, end)))
            
    @abstractmethod
    def read_granule(self, f):
        """Read granule in file.

        :param str f: Path to file
        """

        raise NotImplementedError()

class SingleFileDataset(Dataset):
    wanted = {"start_date", "end_date", "srcfile"}

    def granules_for_period(self, start=datetime.datetime.min,
                            end=datetime.datetime.max):
        if start < self.end_date and end > self.start_date:
            yield self.srcfile
