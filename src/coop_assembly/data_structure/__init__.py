"""
********************************************************************************
data_structure
********************************************************************************

.. currentmodule:: coop_assembly.data_structure

OverallStructure
----------------

OverallStructure is used in the geometry generation process, where we need to keep
track of three-bar-group and large node group information.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    OverallStructure

BarStructure
----------------

todo

.. autosummary::
    :toctree: generated/
    :nosignatures:

    BarStructure

"""

from __future__ import print_function

from .overall_structure import *
from .bar_structure import *

__all__ = ['OverallStructure', 'BarStructure']
