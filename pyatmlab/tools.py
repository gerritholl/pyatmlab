#!/usr/bin/env python

"""Contains functionality that doesn't fit elsewhere
"""

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

    """
    for var, test in func.__annotations__.items():
        value = locals[var]
        if isinstance(test, type): # check for subclass
            if not isinstance(value, test):
                raise TypeError(("Wrong type for argument '{}'.  "
                       "Expected: {}.  Got: {}.").format(
                            var, test, type(value)))
        elif callable(test):
            if not test(value):
                raise TypeError(("Failed test for argument '{0}'.  "
                                 "Value: {1}.  Test {2.__name__}"
                                 "failed.").format(var, value, test))
        else:
            raise RuntimeError("I don't know how to validate test {}!".format(test))

def validator(func):
    """Decorator to automagically validate a function with arguments.

    Uses functionality in 'validate', required types/values determined
    from decorations.  Example::

        @validator
        def f(x: numbers.Number, y: numbers.Number, mode: str):
            return x+y
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
