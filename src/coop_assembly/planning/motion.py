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
    step_simulation

from coop_assembly.data_structure.utils import MotionTrajectory
from .utils import wait_if_gui
from .robot_setup import IK_JOINT_NAMES, get_disabled_collisions, IK_MODULE, get_custom_limits, RESOLUTION, JOINT_WEIGHTS, EE_LINK_NAME
from .stream import ENABLE_SELF_COLLISIONS, get_element_body_in_goal_pose

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

    # from pybullet_planning import get_link_name, get_name
    # for ig in list(extra_disabled_collisions):
    #     b1 = ig[0][0]
    #     b2 = ig[1][0]
    #     l1 = ig[0][1]
    #     l2 = ig[1][1]
    #     b1_name = get_name(b1)
    #     b2_name = get_name(b2)
    #     l1_name = get_link_name(b1, l1)
    #     l2_name = get_link_name(b2, l2)
    #     print('disabled: (Body #{0}, Link #{1}) - (Body #{2} Link #{3})'.format(
    #         b1_name, l1_name, b2_name, l2_name))

    # collision_fn = get_collision_fn(robot, joints, obstacles, attachments=attachments, self_collisions=ENABLE_SELF_COLLISION,
    #                                 disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions,
    #                                 custom_limits=custom_limits, max_distance=0.)
    # print('Initial conf collision-free: ', check_initial_end(start_conf, end_conf, collision_fn, diagnosis=True))

    path = plan_joint_motion(robot, joints, end_conf, obstacles=obstacles, attachments=attachments,
                             self_collisions=ENABLE_SELF_COLLISIONS, disabled_collisions=disabled_collisions,
                             extra_disabled_collisions=extra_disabled_collisions,
                             weights=weights, resolutions=resolutions,
                             restarts=50, iterations=100, smooth=100, max_distance=0.0)

    # * sweeping volume check for attachment
    # attachment_aabbs = []
    # obstacle_aabbs = {ob : get_aabb(ob) for ob in list(obstacles)}
    # with BodySaver(robot):
    #     for conf in path:
    #         set_joint_positions(robot, joints, conf)
    #         for attach in attachments:
    #             attach.assign()
    #         # a dict for each timestep in the traj
    #         attachment_aabbs.append({attach.child : get_aabb(attach.child) for attach in attachments})

    #         # for at in attachments:
    #         #     draw_aabb(get_aabb(at.child))
    #         # wait_if_gui()
    # # for each attachment, union all the aabbs across all timesteps
    # swept_attachments = {attach.child : aabb_union(at_aabbs[attach.child] for at_aabbs in attachment_aabbs) for attach in attachments}
    # swept_overlap = [(obstacle_name, at_name) for obstacle_name, at_name in product(swept_attachments, obstacle_aabbs)
    #                  if aabb_overlap(obstacle_aabbs[obstacle_name], swept_attachments[at_name])]

    # for hull in hulls:
    #     remove_body(hull)
    if path is None:
        cprint('Failed to find a motion plan!', 'red')
        return None

    return MotionTrajectory(robot, joints, path, attachments=attachments)

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

        # if isinstance(trajectory, PrintTrajectory):
        #     if not trajectory.path:
        #         color = colors[len(printed_elements)]
        #         handles.append(draw_element(node_points, trajectory.element, color=color))
        #         #wait_for_user()
        #     is_connected = (trajectory.n1 in connected_nodes) # and (trajectory.n2 in connected_nodes)
        #     print('{}) {:9} | Connected: {} | Ground: {} | Length: {}'.format(
        #         i, str(trajectory), is_connected, is_ground(trajectory.element, ground_nodes), len(trajectory.path)))
        #     if not is_connected:
        #         wait_for_user()
        #     connected_nodes.add(trajectory.n2)
        #     printed_elements.append(trajectory.element)

    # if video_saver is not None:
    #     video_saver.restore()
    wait_if_gui()
