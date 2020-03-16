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
    3. element_bodies: dicr, bar_v -> pybullet body

    4. connector_from_element: dict, bar_v -> connector lines, each line contains two points

Connectors are represented by a list, each entry is a fronzenset of connector line end points.

# TODO: connectors and elements are both vertices in the assembly graph
if connector is `physical`, connector's existence **does not** depend on its connected element's existence
if connector is `virtual`, connector's existence depends on its connected element's existence
In the case of double-tangent bar system, the connector is virtual.

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

loading pybullet env
-----------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    load_world

visualization
--------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    color_structure
    draw_ordered

"""

from __future__ import print_function

from .robot_setup import *
# from .stream import *
# from .utils import *
# from .visualization import *
