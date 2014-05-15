#!/usr/bin/env python

"""Contains functionality that doesn't fit elsewhere
"""

import collections.abc
import inspect
import functools
import shelve
import struct
import logging
import pickle

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
        _validate_one(var, test, value, func)

def _validate_one(var, test, value, func):
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
                _validate_one(var, t, value, func)
            except TypeError:
                pass
            else:
                passed = True
        if not passed:
            raise TypeError(("All tests failed for function {0}, argument '{1}'. "
                             "Value: {2}, type {3}.  Tests: {4!s}").format(
                func.__qualname__, var, value, type(value), test))
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

def disk_lru_cache(path):
    """Like functools.lru_cache, but stored on disk

    Returns a decorator.

    :param str path: File to use for caching.
    :returns function: Decorator
    """

    sentinel = object()
    make_key = functools._make_key
    def decorating_function(user_function):
        cache = shelve.open(path, protocol=4, writeback=True)
        cache_get = cache.get

        def wrapper(*args, **kwds):
            key = str(make_key(args, kwds, False, kwd_mark=(42,)))
            result = cache_get(key, sentinel)
            if result is not sentinel:
                logging.debug(("Getting result from cache "
                    "{!s}, (key {!s}").format(path, key))
                return result
            logging.debug("No result in cache")
            result = user_function(*args, **kwds)
            logging.debug("Storing result in cache")
            cache[key] = result
            cache.sync()
            return result

        return functools.update_wrapper(wrapper, user_function)

    return decorating_function


def mutable_cache(maxsize=10):

    sentinel = object()
    make_key = functools._make_key
    def decorating_function(user_function):
        cache = {}
        cache_get = cache.get
        keylist = [] # don't make it too long

        def wrapper(*args, **kwds):
            # Problem with pickle: dataset objects (commonly passed as
            # 'self') contain a cache which is a shelve object which
            # cannot be pickled.  Would need to create a proper pickle
            # protocol for dataset objects... maybe later
            #key = pickle.dumps(args, 1) + pickle.dumps(kwds, 1)
            key = str(args) + str(kwds)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                logging.debug(("Getting result from cache "
                    " (key {!s}").format(key))
                return result
#            logging.debug("No result in cache")
            result = user_function(*args, **kwds)
#            logging.debug("Storing result in cache")
            cache[key] = result
            keylist.append(key)
            if len(keylist) > maxsize:
                del cache[keylist[0]]
                del keylist[0]
            return result

        return functools.update_wrapper(wrapper, user_function)

    return decorating_function


    def __init__(self, fn):
        self.fn = fn
        self.memo = {}
        self.keys = [] # don't store too many

    def __call__(self, *args, **kwds):
        str = pickle.dumps(args, 1)+pickle.dumps(kwds, 1)
        if not str in self.memo:
            self.memo[str] = self.fn(*args, **kwds)
            self.keys.append(str)
            if len(self.keys) > maxsize:
                del self.memo[self.keys[0]]
                del self.keys[0]

        return self.memo[str]
