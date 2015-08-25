"""Low level XML handlers

This module contains low level XML handlers.  Each handler is a function
that takes as an argument an element (pyatmlab.xml.ArtsElement) and
returns as a result an appropriate type.  The name of the function matches
the tag of the element, for example, for Numeric, _handlers.Numeric will
be called.  (Note that this violates standard naming principles.)
"""

import numpy

def arts(elem):
    return elem[0].value() # FIXME: always one child?

def Array(elem):
    return [t.value() for t in elem]

def String(elem):
    return elem.text
SpeciesTag = String

def Index(elem):
    return int(elem.text)

def Numeric(elem):
    return float(elem.text)

def Vector(elem):
    # sep=" " seems to work even when separated by newlines, see
    # http://stackoverflow.com/q/31882167/974555
    arr = numpy.fromstring(elem.text, sep=" ")
    if arr.size != int(elem.attrib["nelem"]):
        raise ValueError("Expected {:s} elements, found {:d}"
                         " elements!".format(arr.attrib["nelem"],
                         arr.size))
    return arr

# Source: ARTS developer guide, section 3.4
dim_names = ["ncols", "nrows", "npages", "nbooks", "nshelves",
             "nvitrines", "nlibraries"]

def Matrix(elem):
    flatarr = numpy.fromstring(elem.text, sep=" ")
    # turn dims around: in ARTS, [10 x 1 x 1] means 10 pages, 1 row, 1 col
    dims = [dim for dim in dim_names if dim in elem.attrib.keys()][::-1]
    return flatarr.reshape([int(elem.attrib[dim]) for dim in dims])

Tensor3 = Tensor4 = Tensor5 = Tensor6 = Tensor7 = Matrix

# rest not yet implemented
