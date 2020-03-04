import numpy as np
from collections import defaultdict

from pybullet_planning import HideOutput, load_pybullet, set_static, set_joint_positions, joints_from_names, \
    create_plane, set_point, Point
from coop_assembly.help_functions.shared_const import METER_SCALE
from coop_assembly.planning.robot_setup import get_picknplace_robot_data, get_robot_init_conf, BUILT_PLATE_Z

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

def get_connector_neighbors(connector_from_element, elements):
    """find connected elements for each connector

    Parameters
    ----------
    connector_from_element : dict
        bar vkey -> connector ids
    elements : list of int
        bar keys

    Returns
    -------
    dict
        connector id -> set of connected bar vkey
    """
    connector_neighbors = defaultdict(set)
    for bar in elements:
        for connector in connector_from_element[bar]:
            connector_neighbors[connector].add(bar)
    return connector_neighbors


def get_element_neighbors(connector_from_element, elements):
    """find neighbor bars for each bar

    Parameters
    ----------
    connector_from_element : dict
        bar vkey -> connector ids
    elements : list of int
        bar vkey list

    Returns
    -------
    dict
        bar key -> set of neighbor bar keys
    """
    # get neighbor via the connector's neighbor
    connector_neighbors = get_connector_neighbors(connector_from_element, elements)
    element_neighbors = defaultdict(set)
    for bar in elements:
        for c in connector_from_element[bar]:
            element_neighbors[bar].update(connector_neighbors[c])
        element_neighbors[bar].remove(bar)
    return element_neighbors

##################################################

def check_connected(grounded_connectors, printed_elements):
    if not printed_elements:
        return True
    node_neighbors = get_node_neighbors(printed_elements)
    queue = deque(ground_nodes)
    visited_nodes = set(ground_nodes)
    visited_elements = set()
    while queue:
        node1 = queue.popleft()
        for element in node_neighbors[node1]:
            visited_elements.add(element)
            node2 = get_other_node(node1, element)
            if node2 not in visited_nodes:
                queue.append(node2)
                visited_nodes.add(node2)
    return printed_elements <= visited_elements

def get_connected_structures(elements):
    edges = {(e1, e2) for e1, neighbors in get_element_neighbors(elements).items()
             for e2 in neighbors}
    return get_connected_components(elements, edges)

##################################################
