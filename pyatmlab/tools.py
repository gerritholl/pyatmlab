#!/usr/bin/env python

"""Contains functionality that doesn't fit elsewhere
"""

class switch(object):
    """Simulate a switch-case statement.

    http://code.activestate.com/recipes/410692/
    """

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:
            self.fall = True
            return True
        else:
            return False
