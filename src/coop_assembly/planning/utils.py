import numpy as np
from collections import defaultdict, deque

from pybullet_planning import HideOutput, load_pybullet, set_static, set_joint_positions, joints_from_names, \
    create_plane, set_point, Point
from coop_assembly.help_functions.shared_const import METER_SCALE
from coop_assembly.planning.robot_setup import get_picknplace_robot_data, get_robot_init_conf, BUILT_PLATE_Z
from pddlstream.utils import get_connected_components

def load_world(use_floor=True, built_plate_z=BUILT_PLATE_Z):
    robot_data, ws_data = get_picknplace_robot_data()
    robot_urdf, _, _, _, joint_names, _ = robot_data

    print(robot_urdf)
    obstacles = []
    with HideOutput():
        robot = load_pybullet(robot_urdf, fixed_base=True)
        # 1/velocity = weight
        # print([get_max_velocity(robot, joint) for joint in get_movable_joints(robot)])
        set_static(robot)
        set_joint_positions(robot, joints_from_names(robot, joint_names), get_robot_init_conf())
        if use_floor:
            floor = create_plane()
            obstacles.append(floor)
            set_point(floor, Point(x=1.2, z=built_plate_z))
        else:
            floor = None
    return obstacles, robot

##################################################

def get_connector_from_elements(connectors, elements):
    connector_from_elements = defaultdict(set)
    for e in elements:
        for c in connectors:
            if e in c:
                connector_from_elements[e].add(c)
    return connector_from_elements

def get_element_neighbors(connectors, elements):
    connector_from_elements = get_connector_from_elements(connectors, elements)
    element_neighbors = defaultdict(set)
    for e in elements:
        for c in connector_from_elements[e]:
            element_neighbors[e].update(c)
        element_neighbors[e].remove(e)
    return element_neighbors

##################################################

def check_connected(connectors, grounded_elements, printed_elements):
    """check if a given partial structure is connected to the ground

    Parameters
    ----------
    connectors : list of 2-int tuples
        each entry are the indices into the element set,
    grounded_elements : set
        grounded element ids
    printed_elements : set
        printed element ids

    Returns
    -------
    [type]
        [description]
    """
    # TODO: for stability might need to check 2-connected
    if not printed_elements:
        return True
    printed_grounded_elements = set(grounded_elements) & printed_elements
    if not printed_grounded_elements:
        return False
    element_neighbors = get_element_neighbors(connectors, printed_elements)
    queue = deque(printed_grounded_elements)
    visited_elements = set()
    while queue:
        n_element = queue.popleft()
        for element in element_neighbors[n_element]:
            if element in printed_elements and element not in visited_elements:
                visited_elements.add(element)
                queue.append(element)
    return printed_elements <= visited_elements

def get_connected_structures(connectors, elements):
    edges = {(e1, e2) for e1, neighbors in get_element_neighbors(connectors, elements).items()
             for e2 in neighbors}
    return get_connected_components(elements, edges)

##################################################
