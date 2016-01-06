#!/usr/bin/env python

from . import meta

__version__ = "0.1.0+"

__doc__ = """This is pyatmlab
"""

from pint import UnitRegistry
ureg = UnitRegistry()
ureg.define("micro- = 1e-6 = µ-")
