"""
********************************************************************************
planning
********************************************************************************

.. currentmodule:: coop_assembly.planning

This module contains all the planning related components.

Representation
--------------------

We will extract minimal graph information out of the BarStructure to simplify the implementation.
We will use the following three types:
    1. elements: list, indices of BarStructure vertices for bars
    2. axis_pts_from_element: dict, bar_v -> two axis end points
    3. connector_from_element: dict, bar_v -> connector lines, each line contains two points
    4. element_bodies: dicr, bar_v -> pybullet body

robot setup
--------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_picknplace_robot_data
    ROBOT_URDF
    ROBOT_SRDF
    WS_URDF
    WS_SRDF
    IK_MODULE

visualization
--------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    color_structure

"""

from __future__ import print_function

from .robot_setup import *
# from .stream import *
from .utils import *
from .visualization import *
