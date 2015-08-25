"""Read and write Arts XML types

This module contains classes for reading and writing Arts XML types.
It is built on top of xml.etree.ElementTree, available since Python 2.5,
but this module relies on the version 3.3 or newer.  Most users will only
need the 'parse' function.

FIXME:
- Does <arts> tag always contain exactly one child?
"""

import xml.etree.ElementTree

from . import _handlers


class ArtsElement(xml.etree.ElementTree.Element):
    """Element with value interpretation
    """
    def value(self):
        try:
            return getattr(_handlers, self.tag)(self)
        except AttributeError:
            raise ValueError("Don't know how to handle <{}>!".format(self.tag))

def parse(source):
    """Parse ArtsXML file from source.

    This function is very similar to xml.etree.ElementTree.parse, except
    that elements will be of type ArtsElement rather than
    xml.etree.Element.  See documentation for ArtsElement for how this
    helps you.
    """
    return xml.etree.ElementTree.parse(source,
            parser=xml.etree.ElementTree.XMLParser(
                target=xml.etree.ElementTree.TreeBuilder(
                    element_factory=ArtsElement)))

