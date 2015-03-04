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
import copy

import numpy

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

# Use a metaclass to inherit docstrings
#
# http://stackoverflow.com/a/8101118/974555
class DocStringInheritor(type):
    '''A variation on
    http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
    by Paul McGuire
    '''
    def __new__(meta, name, bases, clsdict):
        if not('__doc__' in clsdict and clsdict['__doc__']):
            for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()):
                doc=mro_cls.__doc__
                if doc:
                    clsdict['__doc__']=doc
                    break
        for attr, attribute in clsdict.items():
            if not attribute.__doc__:
                for mro_cls in (mro_cls for base in bases for mro_cls in base.mro()
                                if hasattr(mro_cls, attr)):
                    doc=getattr(getattr(mro_cls,attr),'__doc__')
                    if doc:
                        attribute.__doc__=doc
                        # added by Gerrit
                        attribute.__doc__ += "\n\nDocstring inherited from parent"
                        break
        return type.__new__(meta, name, bases, clsdict)

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

class NotTrueNorFalseType:
    """Not true, nor false.

    A singleton class whose instance can be used in place of True or
    False, to represent a value which has no true-value nor false-value,
    e.g. a boolean flag the value of which is unknown or undefined.

    By Stack Overflow user shx2, http://stackoverflow.com/a/25330344/974555
    """
    def __new__(cls, *args, **kwargs):
        # singleton
        try:
            obj = cls._obj
        except AttributeError:
            obj = object.__new__(cls, *args, **kwargs)
            cls._obj = obj
        return obj

    def __bool__(self):
        raise TypeError('%s: Value is neither True nor False' % self)

    def __repr__(self):
        return 'NotTrueNorFalse'

NotTrueNorFalse = NotTrueNorFalseType()

def rec_concatenate(seqs, ax=0):
    """Concatenate record arrays even if name order differs.

    Takes the first record array and appends data from the rest, by name,
    even if the name order or the specific dtype differs.
    """

    try:
        return numpy.concatenate(seqs)
    except TypeError as e:
        if e.args[0] != "invalid type promotion":
            raise
    # this part is only reached if we do have the TypeError with "invalid
    # type promotion"

    if ax != 0 or any(s.ndim>1 for s in seqs):
        raise ValueError("Liberal concatenations must be 1-d")
    if len(seqs) < 2:
        raise ValueError("Must concatenate at least 2")
    M = numpy.empty(shape=sum(s.shape[0] for s in seqs),
                    dtype=seqs[0].dtype)
    for nm in M.dtype.names:
        if M.dtype[nm].names is not None:
            M[nm] = rec_concatenate([s[nm] for s in seqs])
        else:
            M[nm] = numpy.concatenate([s[nm] for s in seqs])
    return M

def array_equal_with_equal_nans(A, B):
    """Like array_equal, but nans compare equal
    """

    return numpy.all((A == B) | (numpy.isnan(A) & numpy.isnan(B)))

def mark_for_disk_cache(**kwargs):
    """Mark method for later caching
    """
    def mark(meth, d):
        meth.disk_cache = True
        meth.disk_cache_args = d
        return meth
    return functools.partial(mark, d=kwargs)

def setmem(obj, memory):
    """Process marks set by mark_for_disk_cache on a fresh instance

    Meant to be called from __init__ as `setmem(self, memory)
    """

    if memory is not None:
        for k in dir(obj):
            meth = getattr(obj, k)
            if hasattr(meth, "disk_cache") and meth.disk_cache:
#                args = copy.deepcopy(meth.disk_cache_args)
#                if "process" in args:
#                    args["process"] = {k:(v if callable(v) else getattr(obj, v))
#                        for (k, v) in args["process"].items()}
                setattr(obj, k, memory.cache(meth, **meth.disk_cache_args))
