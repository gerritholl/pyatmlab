#!/usr/bin/python
# coding: utf-8

"""Interact with the Atmospheric Radiative Transfer Simulator (ARTS)

A lot of the code here is based on PyARTS.
"""

class ArtsType:
    """Represent any type in ARTS (Workspace Groups)
    """

    @classmethod
    def read(self, f):
        """Read Arts-type from path

        :param f: Path to file to read from
        :returns: ArtsType object
        """

        with open(f, 'r') as fp:
            return self.read_from_openfile(f)

    @classmethod
    def read_from_openfile(self, fp):
        """Read ARTS-type from file object

        :param fp: Open file to read from
        :returns: ArtsType object
        """
