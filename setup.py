import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def get_version():
    """Obtain version number from latest ChangeLog entry
    """

    with open(os.path.join(os.path.dirname(__file__), "ChangeLog"), 'r') as f:
        # should be on third line
        f.readline()
        f.readline()
        line = f.readline()
        return line[line.find("pyatmlab")+9:].strip().replace('-', '.')

setup(
    name = "pyatmlab",
    version = get_version(),
    author = "Gerrit Holl",
    author_email = "gerrit.holl@gmail.com",
    description = ("Diverse collection of tools for working with "
                   "atmospheric data from remote sensing, modelling, "
                   "and related applications."),
    license = "Modified BSD License",
    keywords = "atmosphere, satellite, remote sensing, radiative transfer",
    url = "http://packages.python.org/pyatmlab",
    packages=['pyatmlab', 'tests'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.3",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
)
