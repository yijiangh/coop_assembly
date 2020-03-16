import numpy as np
import random
import math
import time
from termcolor import cprint

from collections import namedtuple
from itertools import islice

# from extrusion.utils import get_disabled_collisions, get_custom_limits, MotionTrajectory

from pybullet_planning import link_from_name, set_pose, \
    multiply, invert, inverse_kinematics, plan_direct_joint_motion, Attachment, set_joint_positions, plan_joint_motion, \
    get_configuration, wait_for_interrupt, point_from_pose, HideOutput, load_pybullet, draw_pose, unit_quat, create_obj, \
    add_body_name, get_pose, pose_from_tform, connect, WorldSaver, get_sample_fn, \
    wait_for_duration, enable_gravity, enable_real_time, trajectory_controller, simulate_controller, \
    add_fixed_constraint, remove_fixed_constraint, Pose, Euler, get_collision_fn, LockRenderer, user_input, has_gui, \
    disconnect, unit_pose, Point, get_distance, sample_tool_ik, joints_from_names, interval_generator, get_floating_body_collision_fn, \
    interpolate_poses, create_attachment, plan_cartesian_motion, INF, GREEN, set_color, get_all_links, step_simulation, get_aabb, \
    get_bodies_in_region, pairwise_link_collision, BASE_LINK

from .robot_setup import EE_LINK_NAME, get_disabled_collisions, IK_MODULE, get_custom_limits, IK_JOINT_NAMES, BASE_LINK_NAME, TOOL_LINK_NAME
from .utils import wait_if_gui, Command
from coop_assembly.data_structure import Grasp, WorldPose, MotionTrajectory

# TODO: fix self collision
ENABLE_SELF_COLLISIONS = False
IK_MAX_ATTEMPTS = 1
PREGRASP_MAX_ATTEMPTS = 100
GRASP_MAX_ATTEMPTS = 100

# pregrasp delta sample
EPSILON = 0.01
ANGLE = np.pi/6

# pregrasp interpolation
POS_STEP_SIZE = 0.005
ORI_STEP_SIZE = np.pi/18

RETREAT_DISTANCE = 0.025

# collision checking safe margin
MAX_DISTANCE = 0.0

def get_goal_pose_gen_fn(element_from_index):
    def gen_fn(index):
        """return a world_from_goal_pose, the central point is invariant,
        just rotate around the bar's local z axis (for bars, the longitude axis)
        """
        body_pose = element_from_index[index].goal_pose.value
        # by default, the longitude axis is z
        # https://pybullet-planning.readthedocs.io/en/latest/reference/generated/pybullet_planning.interfaces.env_manager.create_cylinder.html#pybullet_planning.interfaces.env_manager.create_cylinder
        while True:
            theta = random.uniform(-np.pi, +np.pi)
            rotation = Pose(euler=Euler(yaw=theta))
            yield WorldPose(index, multiply(body_pose, rotation)),
    return gen_fn


def get_bar_grasp_gen_fn(element_from_index, tool_pose=unit_pose(), reverse_grasp=False, safety_margin_length=0.0):
    # converted from https://pybullet-planning.readthedocs.io/en/latest/reference/generated/pybullet_planning.primitives.grasp_gen.get_side_cylinder_grasps.html
    # to get rid of the rotation around the local z axis

    # rotate the cylinder's frame to make x axis align with the longitude axis
    longitude_x = Pose(euler=Euler(pitch=np.pi/2))
    def gen_fn(index):
        # can get from aabb as well
        bar_length = get_distance(*element_from_index[index].axis_endpoints)
        while True:
            # translation along the longitude axis
            slide_dist = random.uniform(-bar_length/2+safety_margin_length, bar_length/2-safety_margin_length)
            translate_along_x_axis = Pose(point=Point(slide_dist,0,0))

            for j in range(1 + reverse_grasp):
                # the base pi/2 is to make y align with the longitude axis, conforming to the convention (see image in the doc)
                # flip the gripper, gripper symmetry
                rotate_around_z = Pose(euler=[0, 0, math.pi/2 + j * math.pi])

                object_from_gripper = multiply(longitude_x, translate_along_x_axis, \
                    rotate_around_z, tool_pose)
                yield Grasp(index, None, None, invert(object_from_gripper), None),
    return gen_fn

######################################

def get_element_body_in_goal_pose(element_from_index, printed):
    for e in list(printed):
        set_pose(element_from_index[e].body, element_from_index[e].goal_pose.value)
    return {element_from_index[e].body for e in list(printed)}

def get_delta_pose_generator(epsilon=EPSILON, angle=ANGLE):
    """sample an infinitestimal pregrasp pose

    Parameters
    ----------
    epsilon : [type]
        [description]
    angle : [type], optional
        [description], by default np.pi/2

    Yields
    -------
    Pose
    """
    lower = [-epsilon]*3 + [-angle]*3
    upper = [epsilon]*3 + [angle]*3
    for [x, y, z, roll, pitch, yaw] in interval_generator(lower, upper): # halton?
        pose = Pose(point=[x,y,z], euler=Euler(roll=roll, pitch=pitch, yaw=yaw))
        yield pose

def get_pregrasp_gen_fn(element_from_index, fixed_obstacles, max_attempts=PREGRASP_MAX_ATTEMPTS, collision=True):
    pose_gen = get_delta_pose_generator()

    def gen_fn(index, pose, printed, diagnosis=False):
        body = element_from_index[index].body
        set_pose(body, pose.value)

        # element_obstacles = {element_from_index[e].body for e in list(printed)}
        element_obstacles = get_element_body_in_goal_pose(element_from_index, printed)
        obstacles = set(fixed_obstacles) | element_obstacles
        if not collision:
            obstacles = set()

        ee_collision_fn = get_floating_body_collision_fn(body, obstacles, max_distance=MAX_DISTANCE)

        for _ in range(max_attempts):
            delta_pose = next(pose_gen)
            offset_pose = multiply(pose.value, delta_pose)
            is_colliding = False
            offset_path = list(interpolate_poses(offset_pose, pose.value, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE))
            for p in offset_path[:-1]:
                # TODO: if colliding at the world_from_bar pose, use local velocity + normal check
                # TODO: normal can be derived from
                if ee_collision_fn(p):
                    is_colliding = True
                    break
            if not is_colliding:
                yield offset_path,
                break
        else:
            yield None,
    return gen_fn

######################################

# the initial pose is fixed, the goal poses can be generated by rotational symmetry
# so the total grasp posibility is generated by:
# rotational goal pose x grasp sliding
# the approach pose is independent of grasp and symmetry, can be generated independently

def get_pick_gen_fn(end_effector, element_from_index, fixed_obstacles, collision=True,
    max_attempts=IK_MAX_ATTEMPTS, max_grasp=GRASP_MAX_ATTEMPTS,
    allow_failure=False, verbose=False, **kwargs):

    robot = end_effector.robot
    ee_from_tool = invert(end_effector.tool_from_root)
    ik_joints = joints_from_names(robot, IK_JOINT_NAMES)
    robot_base_link = link_from_name(robot, BASE_LINK_NAME)
    if IK_MODULE:
        assert IK_MODULE.get_dof() == len(ik_joints)
        # free_dof safe_guard?
    else:
        # joint conf sample fn, used when ikfast is not used
        sample_fn = get_sample_fn(robot, ik_joints)
    tool_link = link_from_name(robot, TOOL_LINK_NAME)
    disabled_collisions = get_disabled_collisions(robot)

    goal_pose_gen_fn = get_goal_pose_gen_fn(element_from_index)
    grasp_gen = get_bar_grasp_gen_fn(element_from_index, reverse_grasp=True, safety_margin_length=0.005)
    pregrasp_gen_fn = get_pregrasp_gen_fn(element_from_index, fixed_obstacles, collision=collision) # max_attempts=max_attempts,

    retreat_distance = RETREAT_DISTANCE
    retreat_vector = retreat_distance*np.array([0, 0, -1])

    def gen_fn(index, printed=[], diagnosis=False):
        body = element_from_index[index].body
        element_obstacles = get_element_body_in_goal_pose(element_from_index, printed)
        obstacles = set(fixed_obstacles) | element_obstacles
        if not collision:
            obstacles = set()
        # attachment is assumed to be empty here, since pregrasp sampler guarantees that
        collision_fn = get_collision_fn(robot, ik_joints, obstacles=obstacles, attachments=[],
                                        self_collisions=ENABLE_SELF_COLLISIONS,
                                        disabled_collisions=disabled_collisions,
                                        custom_limits=get_custom_limits(robot),
                                        max_distance=MAX_DISTANCE)

        for attempt, (world_pose_t, grasp_t) in enumerate(zip(islice(goal_pose_gen_fn(index), max_grasp), islice(grasp_gen(index), max_grasp))):
            world_pose = world_pose_t[0]
            grasp = grasp_t[0]
            for _ in range(max_attempts):
                pregrasp_poses, = next(pregrasp_gen_fn(index, world_pose, printed))
                if not pregrasp_poses:
                    if verbose : print('pregrasp failure.')
                    continue

                pre_attach_poses = [multiply(bar_pose, invert(grasp.attach)) for bar_pose in pregrasp_poses]
                attach_pose = pre_attach_poses[-1]
                retreat_pose = pre_attach_poses[0]

                if IK_MODULE:
                    attach_conf = sample_tool_ik(IK_MODULE.get_ik, robot, ik_joints, attach_pose, robot_base_link, ik_tool_link_from_tcp=ee_from_tool)
                else:
                    set_joint_positions(robot, ik_joints, sample_fn())  # Random seed
                    attach_conf = inverse_kinematics(robot, tool_link, attach_pose)

                if (attach_conf is None):
                    if verbose : print('attach ik failure.')
                    continue
                if collision_fn(attach_conf, diagnosis):
                    if verbose : print('attach collision failure.')
                    continue

                set_joint_positions(robot, ik_joints, attach_conf)
                set_pose(body, pregrasp_poses[-1])
                attachment = create_attachment(robot, tool_link, body)
                # set_color(body, GREEN)
                # wait_if_gui()

                approach_conf = inverse_kinematics(robot, tool_link, retreat_pose)
                if (approach_conf is None):
                    if verbose : print('approach ik failure.')
                    continue
                if collision_fn(approach_conf, diagnosis):
                    if verbose : print('approach collision failure.')
                    continue

                # set_joint_positions(robot, ik_joints, approach_conf)
                # path = plan_direct_joint_motion(robot, ik_joints, attach_conf,
                #                                 obstacles=obstacles,
                #                                 self_collisions=ENABLE_SELF_COLLISION,
                #                                 disabled_collisions=disabled_collisions,
                #                                 attachments=[])
                # path = plan_cartesian_motion(robot, robot_base_link, tool_link, pregrasp_poses)
                # TODO: ladder graph-based Cartesian planning
                path = [approach_conf, attach_conf]
                if path is None:
                    if verbose : print('direct approach motion failure.')
                    continue
                approach_traj = MotionTrajectory(robot, ik_joints, path, attachments=[attachment], tag='place_approach', element=index)

                # * retreat motion
                # // retreat_traj = approach_traj.reverse()
                # // retreat_traj.attachments = []
                set_joint_positions(robot, ik_joints, attach_conf)
                retreat_pose = multiply(attach_pose, (retreat_vector, unit_quat()))
                retreat_conf = inverse_kinematics(robot, tool_link, retreat_pose)
                if (retreat_conf is None):
                    if verbose : print('retreat ik failure.')
                    continue
                if collision_fn(retreat_conf, diagnosis):
                    if verbose : print('retreat collision failure.')
                    continue
                path = [attach_conf, retreat_conf][1:]
                if path is None:
                    if verbose : print('direct retreat motion failure.')
                    continue
                retreat_traj = MotionTrajectory(robot, ik_joints, path, attachments=[], tag='place_retreat', element=index)

                if verbose:
                    cprint('E#{} | Attempts: {}'.format(index, attempt), 'green')
                yield Command([approach_traj, retreat_traj]),
                break
            else:
                # this will run if no break is called, prevent a StopIteraton error
                # https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops
                if allow_failure:
                    yield None,
        else:
            if verbose:
                cprint('E#{} | Attempts: {} | Max attempts exceeded!'.format(index, max_grasp), 'red')
            yield None,
            # return
    return gen_fn
