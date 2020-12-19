import os
import pytest
import numpy as np
from collections import defaultdict
from numpy.linalg import norm
import json
from termcolor import cprint
from itertools import islice
from collections import namedtuple

from pybullet_planning import connect, has_gui, LockRenderer, remove_handles, add_line, \
    draw_pose, EndEffector, unit_pose, link_from_name, end_effector_from_body, get_link_pose, \
    dump_world, set_pose, WorldSaver, reset_simulation, disconnect, get_pose, get_date, RED, GREEN, refine_path, joints_from_names, \
    set_joint_positions, create_attachment, wait_if_gui, apply_alpha, set_color, create_shape, get_mesh_geometry, create_flying_body, \
    SE3, YELLOW, get_movable_joints, get_relative_pose, multiply, plan_joint_motion, Pose, Euler, invert

from coop_assembly.data_structure import BarStructure, OverallStructure, MotionTrajectory
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions import contact_to_ground
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE

from coop_assembly.planning import get_picknplace_robot_data, TOOL_LINK_NAME, EE_LINK_NAME
from coop_assembly.planning.utils import load_world
from coop_assembly.planning.visualization import color_structure, draw_ordered, draw_element, label_elements, label_connector, set_camera, draw_partial_ordered
from coop_assembly.planning.visualization import display_trajectories
from coop_assembly.planning.utils import get_element_neighbors, get_connector_from_elements, check_connected, get_connected_structures, \
    flatten_commands

from coop_assembly.planning.stream import get_bar_grasp_gen_fn, get_place_gen_fn, get_pregrasp_gen_fn, command_collision, \
    get_element_body_in_goal_pose, se3_conf_from_pose
from coop_assembly.planning.regression import regression
from coop_assembly.planning.parsing import load_structure, RESULTS_DIRECTORY, unpack_structure
from coop_assembly.planning.validator import validate_trajectories, validate_pddl_plan
from coop_assembly.planning.utils import recover_sequence, Command
from coop_assembly.planning.robot_setup import get_gripper_mesh_path, get_disabled_collisions, ROBOT_NAME

@pytest.mark.color_structure
def test_color_structure(viewer, file_spec):
    bar_struct, _ = load_structure(file_spec, viewer)
    element_bodies = bar_struct.get_element_bodies()
    printed = set([0,1,2,3])
    color_structure(element_bodies, printed, 4)
    wait_if_gui()


@pytest.mark.draw
def test_draw_ordered(viewer, file_spec):
    bar_struct, _ = load_structure(file_spec, viewer)
    endpts_from_element = bar_struct.get_axis_pts_from_element()
    h = draw_ordered(list(bar_struct.vertices()), endpts_from_element)
    wait_if_gui()
    remove_handles(h)

    elements_from_layer = defaultdict(set)
    for v in bar_struct.vertices():
        elements_from_layer[bar_struct.vertex[v]['layer']].add(v)
    draw_partial_ordered(elements_from_layer, endpts_from_element)
    wait_if_gui()


@pytest.mark.connector
def test_connector(viewer):
    # visual test
    file_spec = '12_bars'
    bar_struct, _ = load_structure(file_spec, viewer)
    element_bodies = bar_struct.get_element_bodies(apply_alpha(RED, 0.3))
    handles = []
    handles.extend(label_elements(element_bodies))
    wait_if_gui()
    remove_handles(handles)

    elements = list(element_bodies.keys())
    contact_from_connectors = bar_struct.get_connectors(scale=1e-3)
    connectors = list(contact_from_connectors.keys())

    # * connectors from bar
    connector_from_elements = get_connector_from_elements(connectors, elements)
    for bar in bar_struct.vertices():
        handles = []
        bar_connectors = connector_from_elements[bar]
        for c in list(bar_connectors):
            handles.append(add_line(*contact_from_connectors[c], color=(1,0,0,1), width=2))
        color_structure(element_bodies, set(), next_element=bar, built_alpha=0.6)
        remove_handles(handles)

    # * neighbor elements from elements
    element_neighbors = get_element_neighbors(connectors, elements)
    for element, connected_bars in element_neighbors.items():
        color_structure(element_bodies, connected_bars, element, built_alpha=0.6)
        wait_if_gui()

    grounded_elements = bar_struct.get_grounded_bar_keys()

    printed_elements = set([2])
    assert check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([0,1])
    assert check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([9,10,11])
    assert not check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([1,9,10,11])
    assert not check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([2,7,9,11,10])
    grounded_elements = bar_struct.get_grounded_bar_keys()
    assert check_connected(connectors, grounded_elements, printed_elements)

@pytest.mark.connector_db
def test_connector_debug(viewer, file_spec):
    # visual test
    bar_struct, _ = load_structure(file_spec, viewer)
    element_bodies = bar_struct.get_element_bodies(color=(1,0,0,0.3))
    handles = []
    handles.extend(label_elements(element_bodies))
    wait_if_gui()
    remove_handles(handles)

    # elements = list(element_bodies.keys())
    contact_from_connectors = bar_struct.get_connectors(scale=1e-3)
    connectors = list(contact_from_connectors.keys())

    grounded_elements = bar_struct.get_grounded_bar_keys()

    printed_elements = set([0])
    assert check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([0,1])
    assert check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([0,3])
    assert check_connected(connectors, grounded_elements, printed_elements)

