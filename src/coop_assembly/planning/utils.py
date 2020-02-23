import numpy as np
from collections import defaultdict

from pybullet_planning import HideOutput, load_pybullet, set_static, set_joint_positions, joints_from_names, \
    create_plane, set_point, Point, set_camera_pose
from coop_assembly.help_functions.shared_const import METER_SCALE
from coop_assembly.planning.robot_setup import get_picknplace_robot_data, get_robot_init_conf

def load_world(use_floor=True, built_plate_z=-0.025):
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

###########################################

def set_camera(node_points):
    centroid = np.average(node_points, axis=0) * METER_SCALE
    camera_offset = 0.25 * np.array([1, 1, 1])
    set_camera_pose(camera_point=centroid + camera_offset, target_point=centroid)

##################################################

def get_connector_neighbors(b_struct, elements):
    # TODO
    node_neighbors = defaultdict(set)
    for e in elements:
        n1, n2 = e
        node_neighbors[n1].add(e)
        node_neighbors[n2].add(e)
    return node_neighbors

def nodes_from_elements(elements):
    return {n for e in elements for n in e}

def get_element_neighbors(elements):
    node_neighbors = get_node_neighbors(elements)
    element_neighbors = defaultdict(set)
    for e in elements:
        n1, n2 = e
        element_neighbors[e].update(node_neighbors[n1])
        element_neighbors[e].update(node_neighbors[n2])
        element_neighbors[e].remove(e)
    return element_neighbors

##################################################
