#!/usr/bin/python
# coding: utf-8

"""Functions related to pyatmlab itself
"""

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
