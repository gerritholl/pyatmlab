#!/usr/bin/env python

from . import meta

__version__ = "0.1.0+"

__doc__ = """This is pyatmlab
"""

from pint import UnitRegistry
ureg = UnitRegistry()
ureg.define("micro- = 1e-6 = µ-")

# aid conversion between different radiance units
sp2 = ureg.add_context("radiance")
sp2.add_transformation(
    "[length] * [mass] / [time] ** 3",
    "[mass] / [time] ** 2",
    lambda ureg, x: x / ureg.speed_of_light)
sp2.add_transformation(
    "[mass] / [time] ** 2",
    "[length] * [mass] / [time] ** 3",
    lambda ureg, x: x * ureg.speed_of_light)
ureg.add_context(sp2)
