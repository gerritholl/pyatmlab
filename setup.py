import subprocess
import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def get_version():
    """Obtain version number
    """
    #from pyatmlab.meta import get_version as get_v
    from pyatmlab import __version__ as v
    if v.endswith("+"):
        #cp = subprocess.run(["git", "log", "--format=%H", "-n", "1"], stdout=subprocess.PIPE, check=True)
        cp = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            stdout=subprocess.PIPE, check=True)
        v += cp.stdout.decode("ascii").strip()
    return v
    #return get_v(os.path.join(os.path.dirname(__file__), "ChangeLog"))

setup(
    name = "pyatmlab",
    version = get_version(),
    author = "Gerrit Holl",
    author_email = "g.holl@reading.ac.uk",
    description = ("Diverse collection of tools for working with "
                   "atmospheric data from remote sensing, modelling, "
                   "and related applications."),
    license = "Modified BSD License",
    keywords = "atmosphere, satellite, remote sensing, radiative transfer",
    url = "http://packages.python.org/pyatmlab",
    packages=['pyatmlab', 'tests', 'pyatmlab.datasets', 'pyatmlab.arts'],
    install_requires=["numpy>=1.10", "scipy>=0.16", "pyproj>=1.9",
                      "statsmodels>=0.6", "pytz>=2015.4",
                      "progressbar2>=3.3",
                      "matplotlib>=1.4", "pint>=0.6",
                      "numexpr>=2.4",
                      "h5py>=2.5",
                      "netCDF4>=1.1",
                      "scikit-learn>=0.17",
                      "typhon>=0.2.7"],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
)
