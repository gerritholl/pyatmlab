#!/usr/bin/python
# coding: utf-8

"""Functions related to pyatmlab itself, or for internal use
"""

import sys

def get_version(path_to_changelog):
    """Obtain version number from latest ChangeLog entry
    
    :param path_to_changelog: Path to ChangeLog file
    :type path_to_changelog: String-like
    :returns: String with version number major.minor.micro
    """

    with open(path_to_changelog, 'r') as f:
        # should be on third line
        f.readline()
        f.readline()
        line = f.readline()
        return line[line.find("pyatmlab")+9:].strip().replace('-', '.')

def expanddoc(f):
    """Function decorator applying str.format-replacement on function docstring

    :param f: function to alter docstring at
    :returns: function
    """
    f.__doc__ = f.__doc__.format(**vars(sys.modules[f.__module__]))
    return f
