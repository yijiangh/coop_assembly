import json
from collections import namedtuple
import numpy as np
from numpy.linalg import norm
from compas.datastructures import Network
from termcolor import cprint

from pybullet_planning import set_camera_pose, connect, create_box, wait_if_gui, set_pose, create_plane, \
    draw_pose, unit_pose, set_camera_pose2, Pose, Point, Euler, RED, BLUE, GREEN, CLIENT, HideOutput, create_obj, apply_alpha, \
    create_flying_body, create_shape, get_mesh_geometry, get_movable_joints, get_configuration, set_configuration, get_links, \
    has_gui, set_color, reset_simulation, disconnect, get_date, WorldSaver, LockRenderer, YELLOW, add_line, draw_circle, pairwise_collision, \
    body_collision_info, get_distance, draw_collision_diagnosis, get_aabb, BodySaver, BLACK

from coop_assembly.planning.parsing import get_assembly_path
from coop_assembly.data_structure import WorldPose, MotionTrajectory
from coop_assembly.geometry_generation.utils import get_element_neighbors
from coop_assembly.planning.visualization import set_camera, label_points

from .stream import pose_from_xz_values

# millemeter
GROUND_BUFFER = np.array([1000,0,100])

###########################################

# Element = namedtuple('Element', ['index', 'axis_endpoints', 'radius', 'body', 'initial_pose', 'goal_pose',
#                                  'grasps', 'goal_supports', 'layer'])

Element2D = namedtuple('Element2D', ['index',
                                     'axis_endpoints',
                                     'wlh',
                                     'body', # 'element_robot',
                                     'initial_pose', 'goal_pose',
                                    #  'grasps', 'layer'
                                     ])

###########################################

def parse_2D_truss(problem, scale=1e-3, debug=False):
    problem_path = get_assembly_path(problem)
    with open(problem_path) as json_file:
        data = json.load(json_file)
        cprint('Parsed from : {}'.format(problem_path), 'green')

    if 'data' in data:
        data = data['data']
    net = Network.from_data(data)

    # TODO waiting for compas update to use ordered dict for nodes
    # node_points, edges = net.to_nodes_and_edges()
    node_points = [np.array([net.node[v]['x'], 0, net.node[v]['z']]) + GROUND_BUFFER for v in range(net.number_of_nodes())]
    ground_nodes = [v for v, attr in net.nodes(True) if attr['fixed'] == True]

    initial_pose = WorldPose('init', pose_from_xz_values([2.0,0,2.0]))
    length = 0.01 # out-of-plane thickness
    element_from_index = {}
    grounded_elements = []
    with LockRenderer(True):
        for e, e_attr in net.edges(True):
            height = e_attr['radius'] * scale
            shrink = e_attr['shrink'] * scale
            width = norm(node_points[e[0]] - node_points[e[1]]) * scale
            wlh = [width - 2*shrink, length, height]

            mid_pt = (node_points[e[0]] + node_points[e[1]]) / 2 * scale
            # assert abs(mid_pt[1]) < 1e-9

            diff = (node_points[e[1]] - node_points[e[0]])
            pitch = np.math.atan2(diff[0], diff[2])
            e_pose = pose_from_xz_values([mid_pt[0],mid_pt[2],pitch+np.pi/2])
            e2d = Element2D(e, (node_points[e[0]]*scale, node_points[e[1]]*scale),
                            wlh, create_box(*wlh),
                            initial_pose, WorldPose(e, e_pose))
            element_from_index[e] = e2d
            set_pose(e2d.body, e2d.goal_pose.value)

            if e_attr['fixed']:
                grounded_elements.append(e)

    connectors = {}
    element_neighbors = get_element_neighbors(element_from_index)
    for e, ens in element_neighbors.items():
        for en in ens:
            connectors[(e, en)] = None

    # * collision check for beams at goal poses
    collided_pairs = set()
    # `p_tol` is based on some manual experiement,
    # might need to be changed accordingly for specific scales and input models
    p_tol = 1e-3
    for i in element_from_index:
        for j in element_from_index:
            if i == j:
                continue
            if (i, j) not in collided_pairs and (j, i) not in collided_pairs:
                if pairwise_collision(element_from_index[i].body, element_from_index[j].body):
                    cr = body_collision_info(element_from_index[i].body, element_from_index[j].body)
                    penetration_depth = get_distance(cr[0][5], cr[0][6])
                    if penetration_depth > p_tol:
                        cprint('({}-{}) colliding : penetrating depth {:.4E}'.format(i,j, penetration_depth), 'red')
                        collided_pairs.add((i,j))
                        if debug:
                            draw_collision_diagnosis(cr, focus_camera=False)
    assert len(collided_pairs) == 0, 'model has mutual collision between elements!'
    cprint('No mutual collisions among elements in the model | penetration threshold: {}'.format(p_tol), 'green')

    set_camera(node_points, camera_dir=np.array([0,-1,0]), camera_dist=1.0)

    # draw the ideal truss that we want to achieve
    label_points([pt*1e-3 for pt in node_points])
    for e in net.edges():
        p1 = node_points[e[0]] * 1e-3
        p2 = node_points[e[1]] * 1e-3
        add_line(p1, p2, color=apply_alpha(BLACK, 0.1), width=0.5)
    for v in ground_nodes:
        draw_circle(node_points[v]*1e-3, 0.01)

    return element_from_index, connectors, grounded_elements

