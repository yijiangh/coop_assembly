import time
import numpy as np
from termcolor import cprint
from itertools import product

from pybullet_planning import get_movable_joints, link_from_name, set_pose, \
    multiply, invert, inverse_kinematics, plan_direct_joint_motion, Attachment, set_joint_positions, plan_joint_motion, \
    get_configuration, wait_for_user, point_from_pose, HideOutput, load_pybullet, draw_pose, unit_quat, create_obj, \
    add_body_name, get_pose, pose_from_tform, connect, WorldSaver, get_sample_fn, \
    wait_for_duration, enable_gravity, enable_real_time, trajectory_controller, simulate_controller, \
    add_fixed_constraint, remove_fixed_constraint, Pose, Euler, get_collision_fn, LockRenderer, user_input, GREEN, BLUE, set_color, \
    joints_from_names, INF, wait_for_user, check_initial_end, BASE_LINK, get_aabb, aabb_union, aabb_overlap, BodySaver, draw_aabb, \
    step_simulation, SE3, get_links, remove_all_debug

from coop_assembly.data_structure import Element
from coop_assembly.data_structure.utils import MotionTrajectory
from .utils import wait_if_gui, get_index_from_bodies
from .robot_setup import IK_JOINT_NAMES, get_disabled_collisions, IK_MODULE, get_custom_limits, RESOLUTION, JOINT_WEIGHTS, EE_LINK_NAME
from .stream import ENABLE_SELF_COLLISIONS, get_element_body_in_goal_pose, POS_STEP_SIZE, ORI_STEP_SIZE

DIAGNOSIS = False

##################################################

BAR_INITIAL_POINT = np.array([0.4, 0, 0.2])
BAR_INITIAL_EULER = np.array([0, np.pi/2, 0])
BAR_INITIAL_CONF = np.concatenate([BAR_INITIAL_POINT, BAR_INITIAL_EULER])

# TODO: derived from bounding box
BAR_CUSTOM_LIMITS = {
    'x': (0.25, 1.0),
    'y': (-1.0, 1.0),
    'z': (-0.3, 0.4),
}
# BAR_RESOLUTION = [0.003]*3 + [np.pi/60]*3
BAR_RESOLUTION = [0.1]*3 + [np.pi/6]*3

def compute_motion(robot, fixed_obstacles, element_from_index,
                   printed_elements, start_conf, end_conf, attachments=[],
                   collisions=True, bar_only=False, **kwargs):
    # TODO: can also just plan to initial conf and then shortcut
    if not bar_only:
        joints = joints_from_names(robot, IK_JOINT_NAMES)
        weights = JOINT_WEIGHTS
        resolutions = np.divide(RESOLUTION * np.ones(weights.shape), weights)
        disabled_collisions = get_disabled_collisions(robot)
        custom_limits = {}
    else:
        joints = get_movable_joints(robot)
        weights = np.ones(len(joints))
        # resolutions = np.divide(RESOLUTION * np.ones(weights.shape), weights)
        # resolutions = np.array([POS_STEP_SIZE]*3 + [ORI_STEP_SIZE]*3)
        resolutions = BAR_RESOLUTION
        disabled_collisions = {}
        custom_limits = {}
        joint_from_group = dict(zip(SE3, joints))
        custom_limits = {joint_from_group[j]: l for j, l in BAR_CUSTOM_LIMITS.items()}

    assert len(joints) == len(end_conf)

    element_obstacles = get_element_body_in_goal_pose(element_from_index, printed_elements)
    obstacles = set(fixed_obstacles) | element_obstacles
    if not collisions:
        obstacles = []

    set_joint_positions(robot, joints, start_conf)
    extra_disabled_collisions = set()
    for attach in attachments:
        attach.assign()
        # prune the link that's adjacent to the attach link to disable end effector / bar collision checks
        ee_link = link_from_name(robot, EE_LINK_NAME) if not bar_only else get_links(robot)[-1]
        extra_disabled_collisions.add(((robot, ee_link), (attach.child, BASE_LINK)))

    path = plan_joint_motion(robot, joints, end_conf, obstacles=obstacles, attachments=attachments,
                             self_collisions=ENABLE_SELF_COLLISIONS, disabled_collisions=disabled_collisions,
                             extra_disabled_collisions=extra_disabled_collisions,
                             weights=weights, resolutions=resolutions, custom_limits=custom_limits,
                             diagnosis=DIAGNOSIS, **kwargs)
    if path is None:
        cprint('Failed to find a motion plan!', 'red')
        return None

    element=None
    if len(attachments) > 0:
        index_from_body = get_index_from_bodies(element_from_index)
        element = index_from_body[attachments[0].child]

    return MotionTrajectory(robot, joints, path, attachments=attachments, element=element, tag='transit2place')

###################################

def display_trajectories(trajectories, time_step=0.02, video=False, animate=True):
    """[summary]

    Parameters
    ----------
    trajectories : [type]
        [description]
    time_step : float, optional
        [description], by default 0.02
    video : bool, optional
        [description], by default False
    animate : bool, optional
        if set to False, display sequence colormap only, skip trajectory animation, by default True
    """
    # node_points, ground_nodes,
    if trajectories is None:
        return
    # set_extrusion_camera(node_points)
    # planned_elements = recover_sequence(trajectories)
    # colors = sample_colors(len(planned_elements))
    # if not animate:
    #     draw_ordered(planned_elements, node_points)
    #     wait_for_user()
    #     disconnect()
    #     return

    video_saver = None
    if video:
        # handles = draw_model(planned_elements, node_points, ground_nodes) # Allows user to adjust the camera
        wait_if_gui()
        # remove_all_debug()
        # wait_for_duration(0.1)
        # video_saver = VideoSaver('video.mp4') # has_gui()
        # time_step = 0.001
    else:
        wait_if_gui('Ready to simulate trajectories.')

    remove_all_debug()
    #element_bodies = dict(zip(planned_elements, create_elements(node_points, planned_elements)))
    #for body in element_bodies.values():
    #    set_color(body, (1, 0, 0, 0))
    # connected_nodes = set(ground_nodes)
    # TODO: resolution depends on bar distance to convex hull of obstacles
    # TODO: fine resolution still results in collision?
    printed_elements = []
    print('Trajectories:', len(trajectories))
    for i, trajectory in enumerate(trajectories):
        #wait_for_user()
        #set_color(element_bodies[element], (1, 0, 0, 1))
        last_point = None
        handles = []

        if isinstance(trajectory, MotionTrajectory):
            for attach in trajectory.attachments:
                set_color(attach.child, GREEN)

        for conf in trajectory.iterate():
            # TODO: the robot body could be different
            # if isinstance(trajectory, PrintTrajectory):
            #     current_point = point_from_pose(trajectory.end_effector.get_tool_pose())
            #     if last_point is not None:
            #         # color = BLUE if is_ground(trajectory.element, ground_nodes) else RED
            #         color = colors[len(printed_elements)]
            #         handles.append(add_line(last_point, current_point, color=color, width=LINE_WIDTH))
            #     last_point = current_point

            if time_step is None:
                wait_for_user('{}'.format(conf))
            else:
                wait_for_duration(time_step)

        if isinstance(trajectory, MotionTrajectory):
            for attach in trajectory.attachments:
                set_color(attach.child, BLUE)
        #     if not trajectory.path:
        #         color = colors[len(printed_elements)]
        #         handles.append(draw_element(node_points, trajectory.element, color=color))
        #         #wait_for_user()
        #     is_connected = (trajectory.n1 in connected_nodes) # and (trajectory.n2 in connected_nodes)
            is_connected = True
            print('{}) {:9} | Connected: {} | Ground: {} | Length: {}'.format(
                i, str(trajectory), is_connected, True, len(trajectory.path)))
                # is_ground(trajectory.element, ground_nodes)
        #     if not is_connected:
        #         wait_for_user()
        #     connected_nodes.add(trajectory.n2)
        #     printed_elements.append(trajectory.element)

    # if video_saver is not None:
    #     video_saver.restore()
    wait_if_gui('Simulation finished.')
