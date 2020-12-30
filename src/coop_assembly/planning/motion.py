import time
import numpy as np
from termcolor import cprint
from itertools import product
import random, colorsys
from scipy.spatial.qhull import QhullError

from pybullet_planning import get_movable_joints, link_from_name, set_pose, \
    multiply, invert, inverse_kinematics, plan_direct_joint_motion, Attachment, set_joint_positions, plan_joint_motion, \
    get_configuration, wait_for_user, point_from_pose, HideOutput, load_pybullet, draw_pose, unit_quat, create_obj, \
    add_body_name, get_pose, pose_from_tform, connect, WorldSaver, get_sample_fn, \
    wait_for_duration, enable_gravity, enable_real_time, trajectory_controller, simulate_controller, \
    add_fixed_constraint, remove_fixed_constraint, Pose, Euler, get_collision_fn, LockRenderer, user_input, GREEN, BLUE, set_color, \
    joints_from_names, INF, wait_for_user, check_initial_end, BASE_LINK, get_aabb, aabb_union, aabb_overlap, BodySaver, draw_aabb, \
    step_simulation, SE3, get_links, remove_all_debug, apply_affine, vertices_from_link, get_aabb_vertices, AABB, convex_hull, \
    create_mesh, apply_alpha, get_sample_fn, get_distance_fn, get_extend_fn, pairwise_collision, remove_body, birrt, RED, elapsed_time
from coop_assembly.data_structure import Element
from coop_assembly.data_structure.utils import MotionTrajectory
from .utils import get_index_from_bodies
from .robot_setup import CONTROL_JOINT_NAMES, get_disabled_collisions, IK_MODULE, get_custom_limits, JOINT_WEIGHTS, EE_LINK_NAME, \
    BUILD_PLATE_CENTER, ROBOT_NAME, JOINT_RESOLUTIONS, ROBOT_NAME
from .stream import ENABLE_SELF_COLLISIONS, get_element_body_in_goal_pose, POS_STEP_SIZE, ORI_STEP_SIZE, MAX_DISTANCE

DYNMAIC_RES_RATIO = 0.3
# DYNMAIC_RES_RATIO = 0.1
CONVEX_BUFFER = 0.3

##################################################

EE_INITIAL_POINTs = {
    'kuka' : np.array(BUILD_PLATE_CENTER) + np.array([0.4, 0, 0.6]),
    'abb_track' : np.array(BUILD_PLATE_CENTER) + np.array([0.4, 0, 1.6]),
}
EE_INITIAL_POINT = EE_INITIAL_POINTs[ROBOT_NAME]
EE_INITIAL_EULER = np.array([0, np.pi/2, 0])
EE_INITIAL_CONF = np.concatenate([EE_INITIAL_POINT, EE_INITIAL_EULER])

# TODO: derived from bounding box
EE_CUSTOM_LIMITS = {
    # 'x': (0.25, 1.0),
    # 'y': (-1.0, 1.0),
    # 'z': (-0.3, 0.4),
    'x': (0., 2.0),
    'y': (-2.0, 2.0),
    'z': (-0.3, 1.5),
}
# EE_RESOLUTION = np.array([0.003]*3 + [np.pi/60]*3)
EE_RESOLUTION = np.array([0.05]*3 + [np.pi/10]*3)
# EE_RESOLUTION = np.array([0.01]*3 + [np.pi/6]*3)

##################################################

def get_pairs(iterator):
    try:
        last = next(iterator)
    except StopIteration:
        return
    for current in iterator:
        yield last, current
        last = current

# https://github.com/caelan/pb-construction/blob/24b05b62b6a1febec38b44d2457e2b8e14de1021/extrusion/motion.py#L30
def create_bounding_mesh(bodies=None, node_points=None, buffer=0.):
    """[summary]

    Parameters
    ----------
    bodies : a list of int, optional
        bodies, by default None
    node_points : [type], optional
        [description], by default None
    buffer : float, optional
        safety buffer distance on the boundary of the convex hull, by default 0.

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    e
        [description]
    """
    # TODO: use bounding boxes instead of points
    # TODO: connected components
    assert bodies or node_points
    printed_points = []
    if node_points is not None:
        printed_points.extend(node_points)
    if bodies is not None:
        for body in bodies:
            printed_points.extend(apply_affine(get_pose(body), vertices_from_link(body, BASE_LINK)))

    if buffer != 0.:
        half_extents = buffer*np.ones(3)
        for point in list(printed_points):
            printed_points.extend(np.array(point) + np.array(corner)
                                  for corner in get_aabb_vertices(AABB(-half_extents, half_extents)))

    rgb = colorsys.hsv_to_rgb(h=random.random(), s=1, v=1)
    #rgb = RED
    try:
        mesh = convex_hull(printed_points)
        # handles = draw_mesh(mesh)
        return create_mesh(mesh, under=True, color=apply_alpha(rgb, 0.2))
    except QhullError as e:
        raise e

###############################################

def compute_motion(robot, fixed_obstacles, element_from_index,
                   printed_elements, start_conf, end_conf, attachments=[],
                   collisions=True, bar_only=False, max_time=INF,
                   buffer=CONVEX_BUFFER, max_distance=MAX_DISTANCE, smooth=100, debug=False): #, **kwargs):
    # TODO: can also just plan to initial conf and then shortcut
    if not bar_only:
        joints = joints_from_names(robot, CONTROL_JOINT_NAMES)
        weights = JOINT_WEIGHTS
        resolutions = JOINT_RESOLUTIONS
        # if ROBOT_NAME == 'kuka':
        #     resolutions = np.divide(RESOLUTION * np.ones(weights.shape), weights)
        # elif ROBOT_NAME == 'abb_track':
        #     # resolutions = np.divide(RESOLUTION * np.ones(weights.shape), weights)
        #     resolutions = np.hstack([GANTRY_RESOLUTION, RESOLUTION*np.ones(6)])
        disabled_collisions = get_disabled_collisions(robot)
        custom_limits = get_custom_limits(robot)
    else:
        joints = get_movable_joints(robot)
        weights = np.ones(len(joints))
        # resolutions = np.divide(RESOLUTION * np.ones(weights.shape), weights)
        # resolutions = np.array([POS_STEP_SIZE]*3 + [ORI_STEP_SIZE]*3)
        resolutions = EE_RESOLUTION
        disabled_collisions = {}
        custom_limits = {}
        joint_from_group = dict(zip(SE3, joints))
        custom_limits = {joint_from_group[j]: l for j, l in EE_CUSTOM_LIMITS.items()}

    assert len(joints) == len(end_conf)

    element_obstacles = get_element_body_in_goal_pose(element_from_index, printed_elements)
    hulls, obstacles = {}, []
    if collisions:
        obstacles = set(fixed_obstacles) | element_obstacles

    set_joint_positions(robot, joints, start_conf)
    extra_disabled_collisions = set()
    for attach in attachments:
        attach.assign()
        # prune the link that's adjacent to the attach link to disable end effector / bar collision checks
        ee_link = link_from_name(robot, EE_LINK_NAME) if not bar_only else get_links(robot)[-1]
        extra_disabled_collisions.add(((robot, ee_link), (attach.child, BASE_LINK)))

    # construct a bounding box around the built elements
    bounding = None
    if printed_elements:
        node_points = []
        for e in printed_elements:
            node_points.extend(element_from_index[e].axis_endpoints)
        bounding = create_bounding_mesh(bodies=None, node_points=node_points,
                                        buffer=buffer)

    sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(robot, joints, weights=weights)
    extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(robot, joints, obstacles=obstacles, attachments=attachments, self_collisions=ENABLE_SELF_COLLISIONS,
                                    disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance)
    fine_extend_fn = get_extend_fn(robot, joints, resolutions=DYNMAIC_RES_RATIO*resolutions) #, norm=INF)

    def test_bounding(q):
        set_joint_positions(robot, joints, q)
        for attach in attachments:
            attach.assign()
            # set_color(attach.child, RED)
        # attachment_collision =
        # if len(attachments)>0:
        #     wait_for_user('attach collision: {}'.format(attachment_collision))
        collision = (bounding is not None) and (pairwise_collision(robot, bounding, max_distance=max_distance) or \
            any([pairwise_collision(attach.child, bounding, max_distance=max_distance) for attach in attachments]))
        return q, collision

    def dynamic_extend_fn(q_start, q_end):
        # TODO: retime trajectories to be move more slowly around the structure
        for (q1, c1), (q2, c2) in get_pairs(map(test_bounding, extend_fn(q_start, q_end))):
            # print(c1, c2, len(list(fine_extend_fn(q1, q2))))
            # set_joint_positions(robot, joints, q2)
            # wait_for_user()
            if c1 and c2:
                for q in fine_extend_fn(q1, q2):
                    # set_joint_positions(robot, joints, q)
                    # wait_for_user()
                    yield q
            else:
                yield q2

    def element_collision_fn(q):
        if collision_fn(q):
            return True
        #for body in get_bodies_in_region(get_aabb(robot)): # Perform per link?
        #    if (element_from_body.get(body, None) in printed_elements) and pairwise_collision(robot, body):
        #        return True
        for hull, bodies in hulls.items():
            if pairwise_collision(robot, hull) and any(pairwise_collision(robot, body) for body in bodies):
                return True
        return False

    path = None
    if check_initial_end(start_conf, end_conf, collision_fn, diagnosis=debug):
        path = birrt(start_conf, end_conf, distance_fn, sample_fn, dynamic_extend_fn, element_collision_fn,
                     restarts=50, iterations=100, smooth=smooth, max_time=max_time)

    if bounding is not None:
        remove_body(bounding)
    for hull in hulls:
        remove_body(hull)

    # path = plan_joint_motion(robot, joints, end_conf, obstacles=obstacles, attachments=attachments,
    #                          self_collisions=ENABLE_SELF_COLLISIONS, disabled_collisions=disabled_collisions,
    #                          extra_disabled_collisions=extra_disabled_collisions,
    #                          weights=weights, resolutions=resolutions, custom_limits=custom_limits,
    #                          diagnosis=DIAGNOSIS, **kwargs)
    if path is None:
        cprint('Failed to find a motion plan!', 'red')
        return None

    element=None
    if len(attachments) > 0:
        index_from_body = get_index_from_bodies(element_from_index)
        element = index_from_body[attachments[0].child]

    return MotionTrajectory(robot, joints, path, attachments=attachments, element=element, tag='transit')

def compute_motions(robot, fixed_obstacles, element_from_index, initial_conf, print_trajectories, **kwargs):
    # TODO: reoptimize for the sequence that have the smallest movements given this
    # TODO: sample trajectories
    # TODO: more appropriate distance based on displacement/volume
    cprint('Transfer/Transition planning.', 'green')
    if print_trajectories is None:
        return None
    #if any(isinstance(print_traj, MotionTrajectory) for print_traj in print_trajectories):
    #    return print_trajectories
    start_time = time.time()
    printed_elements = []
    all_trajectories = []
    # current_conf = initial_conf
    for i, print_traj in enumerate(print_trajectories):
        # if not np.allclose(current_conf, print_traj.start_conf, rtol=0, atol=1e-8):
        if 'place_retreat' == print_traj.tag:
            printed_elements.append(print_traj.element)

        if print_traj.tag == 'place_approach':
            attachments = print_traj.attachments
            tag = 'transfer'
            motion_traj = compute_motion(robot, fixed_obstacles, element_from_index,
                                         printed_elements, initial_conf, print_traj.start_conf,
                                         attachments=attachments, **kwargs)
        elif print_traj.tag == 'place_retreat':
            attachments = []
            tag = 'transit'
            motion_traj = compute_motion(robot, fixed_obstacles, element_from_index,
                                         printed_elements, print_traj.end_conf, initial_conf,
                                         attachments=attachments, **kwargs)
        else:
            raise ValueError(print_traj.tag)
        motion_traj.tag = tag
        if motion_traj is None:
            return None
        print('{}) {} | Time: {:.3f}'.format(i, motion_traj, elapsed_time(start_time)))

        if print_traj.tag == 'place_approach':
            all_trajectories.append(motion_traj)
            all_trajectories.append(print_traj)
        elif print_traj.tag == 'place_retreat':
            all_trajectories.append(print_traj)
            all_trajectories.append(motion_traj)

    motion_traj = compute_motion(robot, fixed_obstacles, element_from_index,
                                 printed_elements, all_trajectories[-1].end_conf, initial_conf, **kwargs)
    if motion_traj is None:
        return None
    return all_trajectories + [motion_traj]
