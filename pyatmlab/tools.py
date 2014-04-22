#!/usr/bin/env python

"""Contains functionality that doesn't fit elsewhere
"""

import collections.abc
import inspect
import functools

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

# Following inspired by http://stackoverflow.com/a/7811344/974555
def validate(func, locals):
    """Validate function with arguments.

    Inside an annotated function (see PEP-3107), do type checking on the
    arguments.  An annotation may be either a type or a callable.  Use
    like this:

    def f(x: str, b: int):
        validate(f, locals())
        ... # proceed

    or use the validator annotation:
    @validator
    def f(x: str, b: int):
        ... # proceed
    """
    for var, test in func.__annotations__.items():
        value = locals[var]
        _validate_one(var, test, value)

def _validate_one(var, test, value):
    """Verify that var=value passes test

    Internal function for validate
    """
    if isinstance(test, type): # check for subclass
        if not isinstance(value, test):
            raise TypeError(("Wrong type for argument '{}'.  "
                   "Expected: {}.  Got: {}.").format(
                        var, test, type(value)))
    elif callable(test):
        if not test(value):
            raise TypeError(("Failed test for argument '{0}'.  "
                             "Value: {1}.  Test {2.__name__} "
                             "failed.").format(
                var, value if len(repr(value)) < 10000 else "(too long)", test))
    elif isinstance(test, collections.abc.Sequence): # at least one should be true
        passed = False
        for t in test:
            try:
                _validate_one(var, t, value)
            except TypeError:
                pass
            else:
                passed = True
        if not passed:
            raise TypeError(("All tests failed for argument '{0}'. "
                             "Value: {1}.  Tests: {2!s}").format(
                var, value, test))
    else:
        raise RuntimeError("I don't know how to validate test {}!".format(test))

def validator(func):
    """Decorator to automagically validate a function with arguments.

    Uses functionality in 'validate', required types/values determined
    from decorations.  Example::

        @validator
        def f(x: numbers.Number, y: numbers.Number, mode: str):
            return x+y

    Does not currently work for *args and **kwargs style arguments.
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        fsig = inspect.signature(func)
        # initialise with defaults; update with positional arguments; then
        # update with keyword arguments
        lcls = dict([(x.name, x.default) for x in fsig.parameters.values()])
        lcls.update(**dict(zip([x.name for x in fsig.parameters.values()], args)))
        lcls.update(**kwargs)
        validate(func, lcls)
        return func(*args, **kwargs)

    return inner

def cat(*args):
    """Concatenate either ndarray or ma.MaskedArray

    Arguments as for numpy.concatenate or numpy.ma.concatenate.
    First argument determines type.
    """

    if isinstance(args[0][0], numpy.ma.MaskedArray):
        return numpy.ma.concatenate(*args)
    else:
        return numpy.concatenate(*args)
