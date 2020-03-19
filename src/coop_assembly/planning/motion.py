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
    step_simulation, SE3

from coop_assembly.data_structure import Element
from coop_assembly.data_structure.utils import MotionTrajectory
from .utils import wait_if_gui, get_index_from_bodies
from .robot_setup import IK_JOINT_NAMES, get_disabled_collisions, IK_MODULE, get_custom_limits, RESOLUTION, JOINT_WEIGHTS, EE_LINK_NAME
from .stream import ENABLE_SELF_COLLISIONS, get_element_body_in_goal_pose

##################################################

def compute_element_se3_motion(fixed_obstacles, element_from_index, printed_elements, initial_conf, final_conf, collision=True):
    # use bounding box
    group = SE3
    size = 4.
    custom_limits = {
        'x': (0.0, size),
        'y': (0.0, size),
        'z': (0.0, size),
    }

    element_obstacles = get_element_body_in_goal_pose(element_from_index, printed_elements)
    obstacles = set(fixed_obstacles) | element_obstacles
    if not collisions:
        obstacles = []

    client = get_client() # client is the new client for the body
    collision_id = clone_collision_shape(body, BASE_LINK, client)
    visual_id = clone_visual_shape(body, BASE_LINK, client)
    element_robot = create_flying_body(group, collision_id, visual_id)

    body_link = get_links(element_robot)[-1]
    joints = get_movable_joints(element_robot)
    joint_from_group = dict(zip(group, joints))
    print(joint_from_group)
    #print(get_aabb(element_robot, body_link))
    dump_body(element_robot, fixed=False)
    custom_limits = {joint_from_group[j]: l for j, l in custom_limits.items()}

    # initial_point = size*np.array([-1., -1., 0])
    # final_point = -initial_point
    # initial_euler = np.array([0,0,np.pi/2])
    # initial_conf = np.concatenate([initial_point, initial_euler])
    # final_conf = np.concatenate([final_point, initial_euler])

    set_joint_positions(element_robot, joints, initial_conf)
    path = plan_joint_motion(element_robot, joints, final_conf, obstacles=obstacles,
                             self_collisions=False, custom_limits=custom_limits)

    return path

##################################################

def compute_motion(robot, fixed_obstacles, element_from_index,
                   printed_elements, start_conf, end_conf, attachments=[],
                   collisions=True, max_time=INF, smooth=100):
    # TODO: can also just plan to initial conf and then shortcut
    joints = joints_from_names(robot, IK_JOINT_NAMES)
    assert len(joints) == len(end_conf)
    weights = JOINT_WEIGHTS
    resolutions = np.divide(RESOLUTION * np.ones(weights.shape), weights)
    disabled_collisions = get_disabled_collisions(robot)
    custom_limits = {}

    element_obstacles = get_element_body_in_goal_pose(element_from_index, printed_elements)
    obstacles = set(fixed_obstacles) | element_obstacles
    hulls = {}

    if not collisions:
        hulls = {}
        obstacles = []

    # sample_fn = get_sample_fn(robot, joints, custom_limits=custom_limits)
    # distance_fn = get_distance_fn(robot, joints, weights=weights)
    # extend_fn = get_extend_fn(robot, joints, resolutions=resolutions)
    # collision_fn = get_collision_fn(robot, joints, obstacles, attachments=attachments, self_collisions=ENABLE_SELF_COLLISION,
    #                                 disabled_collisions=disabled_collisions, custom_limits=custom_limits, max_distance=0.)
    # collision_fn = get_element_collision_fn(robot, obstacles)

    # def element_collision_fn(q):
    #     if collision_fn(q):
    #         return True
    #     #for body in get_bodies_in_region(get_aabb(robot)): # Perform per link?
    #     #    if (element_from_body.get(body, None) in printed_elements) and pairwise_collision(robot, body):
    #     #        return True
    #     for hull, bodies in hulls.items():
    #         if pairwise_collision(robot, hull) and any(pairwise_collision(robot, body) for body in bodies):
    #             return True
    #     return False

    # path = None

    # if check_initial_end(start_conf, end_conf, collision_fn):
    #     path = birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, element_collision_fn,
    #                  restarts=50, iterations=100, smooth=smooth, max_time=max_time)

    set_joint_positions(robot, joints, start_conf)
    extra_disabled_collisions = set()
    for attach in attachments:
        attach.assign()
        # prune the link that's adjacent to the attach link to disable end effector / bar collision checks
        extra_disabled_collisions.add(((robot, link_from_name(robot, EE_LINK_NAME)), (attach.child, BASE_LINK)))

    path = plan_joint_motion(robot, joints, end_conf, obstacles=obstacles, attachments=attachments,
                             self_collisions=ENABLE_SELF_COLLISIONS, disabled_collisions=disabled_collisions,
                             extra_disabled_collisions=extra_disabled_collisions,
                             weights=weights, resolutions=resolutions,
                             restarts=50, iterations=100, smooth=100, max_distance=0.0)

    element=None
    if len(attachments) > 0:
        index_from_body = get_index_from_bodies(element_from_index)
        element = index_from_body[attachments[0].child]

    # for hull in hulls:
    #     remove_body(hull)
    if path is None:
        cprint('Failed to find a motion plan!', 'red')
        return None

    return MotionTrajectory(robot, joints, path, attachments=attachments, element=element, tag='transit2place')

###################################

def display_trajectories(trajectories, animate=True, time_step=0.02, video=False):
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

    # video_saver = None
    # if video:
    #     handles = draw_model(planned_elements, node_points, ground_nodes) # Allows user to adjust the camera
    #     wait_for_user()
    #     remove_all_debug()
    #     wait_for_duration(0.1)
    #     video_saver = VideoSaver('video.mp4') # has_gui()
    #     time_step = 0.001
    # else:
    #     wait_for_user()

    #element_bodies = dict(zip(planned_elements, create_elements(node_points, planned_elements)))
    #for body in element_bodies.values():
    #    set_color(body, (1, 0, 0, 0))
    # connected_nodes = set(ground_nodes)
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

        for _ in trajectory.iterate():
            # TODO: the robot body could be different
            # if isinstance(trajectory, PrintTrajectory):
            #     current_point = point_from_pose(trajectory.end_effector.get_tool_pose())
            #     if last_point is not None:
            #         # color = BLUE if is_ground(trajectory.element, ground_nodes) else RED
            #         color = colors[len(printed_elements)]
            #         handles.append(add_line(last_point, current_point, color=color, width=LINE_WIDTH))
            #     last_point = current_point

            if time_step is None:
                wait_for_user()
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
    wait_if_gui()
