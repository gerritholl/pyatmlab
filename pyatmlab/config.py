#!/usr/bin/python
# coding: utf-8

"""Handle specific configuration settings, such as data locations.

Configuration is handled with a `.pyatmlabrc` file with a
:mod:`configparser` syntax.
"""

import os.path
import configparser

class _Configurator(object):
    config = None

    def init(self):
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        config.read(os.path.expanduser('~/.pyatmlabrc'))
        self.config = config
    
    def __call__(self, arg):
        if self.config is None:
            self.init()
        return self.config.get("main", arg)
config = _Configurator()
config.init()
conf = config.config

def get_config(arg):
    """Get value for configuration variable.

    :param arg: Name of configuration variable
    :type arg: String-like
    :returns: Value for configuration variable
    """
    confer = _Configurator()

    return confer(arg)
