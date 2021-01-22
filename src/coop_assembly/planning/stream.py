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
    interpolate_poses, create_attachment, plan_cartesian_motion, INF, GREEN, BLUE, RED, set_color, get_all_links, step_simulation, get_aabb, \
    get_bodies_in_region, pairwise_link_collision, BASE_LINK, get_client, clone_collision_shape, clone_visual_shape, get_movable_joints, \
    create_flying_body, SE3, euler_from_quat, create_shape, get_cylinder_geometry, wait_if_gui, set_joint_positions, dump_body, get_links, \
    get_link_pose, get_joint_positions, intrinsic_euler_from_quat, implies, pairwise_collision, randomize, get_link_name, get_relative_pose, \
    remove_handles, apply_alpha, pairwise_link_collision_info, joint_from_name

from .robot_setup import EE_LINK_NAME, get_disabled_collisions, IK_MODULE, get_custom_limits, CONTROL_JOINT_NAMES, BASE_LINK_NAME, \
    TOOL_LINK_NAME, ROBOT_NAME, GANTRY_JOINT_LIMITS, IK_BASE_LINK_NAME, IK_JOINT_NAMES, JOINT_WEIGHTSs
from .utils import Command, prune_dominated, get_index_from_bodies
from coop_assembly.data_structure import Grasp, WorldPose, MotionTrajectory
from coop_assembly.help_functions.shared_const import METER_SCALE

# TODO: fix self collision
ENABLE_SELF_COLLISIONS = True
assert ENABLE_SELF_COLLISIONS
IK_MAX_ATTEMPTS = 1
PREGRASP_MAX_ATTEMPTS = 100
GRASP_MAX_ATTEMPTS = 100
GANTRY_MAX_ATTEMPTS = 30

ALLOWABLE_BAR_COLLISION_DEPTH = 1e-3

# pregrasp delta sample
EPSILON = 0.05
ANGLE = np.pi/3

# pregrasp interpolation
POS_STEP_SIZE = 0.01
ORI_STEP_SIZE = np.pi/18
# POS_STEP_SIZE = 0.005 # | 0.002
# ORI_STEP_SIZE = np.pi/36

RETREAT_DISTANCEs = {
    'kuka': 0.025,
    'abb_track': 0.07,
}
RETREAT_DISTANCE = RETREAT_DISTANCEs[ROBOT_NAME]

# collision checking safe margin
MAX_DISTANCE = 0.0

JOINT_JUMP_THRESHOLD = np.pi/3
# tolerance for joint values from pybullet's built-in IK
PB_IK_TOL = 1e-3

###########################################

def se3_conf_from_pose(p):
    return np.concatenate([p[0], intrinsic_euler_from_quat(p[1])])

###########################################

def get_bar_grasp_gen_fn(element_from_index, tool_pose=unit_pose(), reverse_grasp=False, safety_margin_length=0.0):
    """[summary]

    # converted from https://pybullet-planning.readthedocs.io/en/latest/reference/generated/pybullet_planning.primitives.grasp_gen.get_side_cylinder_grasps.html
    # to get rid of the rotation around the local z axis

    Parameters
    ----------
    element_from_index : [type]
        [description]
    tool_pose : [type], optional
        [description], by default unit_pose()
    reverse_grasp : bool, optional
        [description], by default False
    safety_margin_length : float, optional
        the length of the no-grasp region on the bar's two ends, by default 0.0

    Returns
    -------
    [type]
        [description]

    Yields
    -------
    [type]
        [description]
    """

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
    """sample generator for an infinitesimal \delta X \in SE(3)
    This is used as the pose difference between the pre-detach pose and the detach pose.

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

def get_pregrasp_gen_fn(element_from_index, fixed_obstacles, max_attempts=PREGRASP_MAX_ATTEMPTS, collision=True, teleops=False):
    """sample generator for a path \tao \subset SE(3) between the pre-detach pose and the goal pose of ab element.

    Parameters
    ----------
    element_from_index : [type]
        [description]
    fixed_obstacles : [type]
        [description]
    max_attempts : [type], optional
        the number of sampling trails, by default PREGRASP_MAX_ATTEMPTS
    collision : bool, optional
        [description], by default True
    teleops : bool, optional
        skip the interpolation between the key poses, by default False

    Returns
    -------
    [type]
        [description]

    Yields
    -------
    a list of Pose
        element body poses
    """
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
            if not teleops:
                offset_path = list(interpolate_poses(offset_pose, pose.value, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE))
            else:
                offset_path = [offset_pose, pose.value]
            for p in offset_path: # [:-1]
                # TODO: if colliding at the world_from_bar pose, use local velocity + normal check
                # TODO: normal can be derived from
                if ee_collision_fn(p, diagnosis=diagnosis):
                # if element_robot_collision_fnpose2conf(p)):
                    is_colliding = True
                    break
            if not is_colliding:
                yield offset_path,
                break
        else:
            yield None,
    return gen_fn

######################################

def command_collision(command, bodies, index_from_bodies=None, debug=False):
    """check if a command's trajectories collide with the given bodies.
       Return a list of [True/False] corresponding to the id used in ``bodies``
       Critical in pddlstream formulation.
    """
    # TODO: each new addition makes collision checking more expensive
    #offset = 4
    #for robot_conf in trajectory[offset:-offset]:
    collisions = [False for _ in range(len(bodies))]
    idx_from_body = dict(zip(bodies, range(len(bodies))))
    # TODO: use bounding cylinder for each element
    # TODO: separate into another method. Sort paths by tool poses first

    # * checking end effector and attachments only
    # this is done prior to the robot body
    # because this checking is less expensive and more indicative of success
    # than the full placement motion of the robot
    for trajectory in command.trajectories:
        robot = trajectory.robot
        # for conf in randomize(trajectory.path):
        for i, conf in enumerate(trajectory.path):
            # TODO: bisect or refine
            # TODO from joint name
            joints = get_movable_joints(robot)
            set_joint_positions(robot, joints, conf)
            for attach in trajectory.attachments:
                attach.assign()
                if debug:
                    set_color(attach.child, apply_alpha(GREEN,0.5))

            #for body, _ in get_bodies_in_region(tool_aabb):
            for i, body in enumerate(bodies):
                if body not in idx_from_body: # Robot
                    continue
                idx = idx_from_body[body]
                for attach in trajectory.attachments:
                    if not collisions[idx]:
                        # collisions[idx] |= pairwise_collision(attach.child, body)
                        # ignore if penetration depth is smaller than 5e-4
                        # TODO only allow for neighboring elements
                        msgs = pairwise_link_collision_info(attach.child, BASE_LINK, body, BASE_LINK)
                        for msg in msgs:
                            penetration_depth = get_distance(msg[5], msg[6])
                            collisions[idx] |= penetration_depth > ALLOWABLE_BAR_COLLISION_DEPTH
                    if debug:
                        if collisions[idx]:
                            body_name = index_from_bodies[body] if index_from_bodies else body
                            set_color(body, apply_alpha(BLUE,0.5))
                            print('Attach colliding E{} - E{}'.format(attach.child, body_name))
                            # wait_if_gui()

        # ! enable if end effector is a part of attachments
        # tool_link_name = TOOL_LINK_NAME if end_effector else get_link_name(robot, get_links(robot)[-1])
        # for tool_pose in randomize(trajectory.get_link_path(tool_link_name)): # TODO: bisect
        #     joints = get_movable_joints(robot)
        # if end_effector:
        #     end_effector.set_pose(tool_pose)
        # if not collisions[idx]:
        #     collisions[idx] |= pairwise_collision(end_effector.body, body)

    # * checking robot bodies
    for trajectory in command.trajectories:
        for conf in randomize(trajectory.path):
            set_joint_positions(trajectory.robot, trajectory.joints, conf)
            for i, body in enumerate(bodies):
                if not collisions[i]:
                    collisions[i] |= pairwise_collision(trajectory.robot, body)
                # if debug and collisions[idx]:
                #     body_name = index_from_bodies[body] if index_from_bodies else body
                    #   set_color(body, apply_alpha(BLUE,0.5))
                    #   print('colliding R{} - E{}'.format(trajectory.robot, body_name))
                    #   dump_body(trajectory.robot)
            # wait_if_gui("check command collision")

    #for element, unsafe in zip(elements, collisions):
    #    command.safe_per_element[element] = unsafe
    return collisions

######################################

def check_path(joints, path, collision_fn=None, jump_threshold=None, diagnosis=False):
    """return False if path is not valid
    """
    joint_jump_thresholds = jump_threshold or [JOINT_JUMP_THRESHOLD for jt in joints]
    for jt1, jt2 in zip(path[:-1], path[1:]):
        delta_j = np.abs(np.array(jt1) - np.array(jt2))
        if any(delta_j > np.array(joint_jump_thresholds)):
            return False
    if collision_fn is not None:
        for q in path:
            if collision_fn(q, diagnosis):
                return False
    return True

def compute_place_path(robot, tool_from_ee, pregrasp_poses, grasp, index, element_from_index, collision_fn,
    bar_only=False, verbose=False, diagnosis=False, retreat_vector=np.array([0, 0, -1]), teleops=False, gantry_sample_fn=None):
    """Give the grasp and EE workspace poses, compute cartesian planning for pre-detach ~ detach ~ post-detach process.
    """
    body = element_from_index[index].body
    pre_attach_poses = [multiply(bar_pose, invert(grasp.attach)) for bar_pose in pregrasp_poses]
    attach_pose = pre_attach_poses[-1]
    pre_attach_pose = pre_attach_poses[0]
    post_attach_pose = multiply(attach_pose, (retreat_vector, unit_quat()))
    post_attach_poses = list(interpolate_poses(attach_pose, post_attach_pose, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE))
    ee_from_tool = invert(tool_from_ee)

    if bar_only:
        # no inverse kinematics involved in the bar_only mode
        ee_joints = get_movable_joints(robot)
        ee_body_link = get_links(robot)[-1]
        # set ee: world_from_tool * tool_from_ee
        attach_conf = se3_conf_from_pose(multiply(attach_pose, tool_from_ee))
        set_joint_positions(robot, ee_joints, attach_conf)
        # set attached element
        set_pose(body, pregrasp_poses[-1])
        attachment = create_attachment(robot, ee_body_link, body)
        approach_path = [se3_conf_from_pose(multiply(p, tool_from_ee)) for p in pre_attach_poses]
        for conf in randomize(approach_path):
            if collision_fn(conf, diagnosis):
                return None
        retreat_path = [se3_conf_from_pose(multiply(p, tool_from_ee)) for p in post_attach_poses]
        for conf in randomize(retreat_path):
            if collision_fn(conf, diagnosis):
                return None
        approach_traj = MotionTrajectory(robot, ee_joints, approach_path, attachments=[attachment], tag='place_approach', element=index)
        retreat_traj = MotionTrajectory(robot, ee_joints, retreat_path, attachments=[], tag='place_retreat', element=index)
        command = Command([approach_traj, retreat_traj])
        return command

    tool_link = link_from_name(robot, TOOL_LINK_NAME)
    ik_base_link = link_from_name(robot, IK_BASE_LINK_NAME)
    ik_joints = joints_from_names(robot, IK_JOINT_NAMES)
    control_joints = joints_from_names(robot, CONTROL_JOINT_NAMES)
    custom_limits = get_custom_limits(robot)

    # tool_pose = get_link_pose(robot, tool_link)
    # draw_pose(tool_pose)

    # * attach IK
    if IK_MODULE:
        # free_dof safe_guard?
        if gantry_sample_fn and GANTRY_JOINT_LIMITS:
            gantry_joints = joints_from_names(robot, list(GANTRY_JOINT_LIMITS.keys()))
            for _ in range(GANTRY_MAX_ATTEMPTS):
                gantry_vals = gantry_sample_fn()
                set_joint_positions(robot, gantry_joints, gantry_vals)
                attach_conf = sample_tool_ik(IK_MODULE.get_ik, robot, ik_joints, attach_pose, ik_base_link, ik_tool_link_from_tcp=ee_from_tool)
                if attach_conf is not None:
                    attach_conf = np.hstack([gantry_vals, attach_conf])
                    break
        else:
            assert IK_MODULE.get_dof() == len(CONTROL_JOINT_NAMES), 'IKFast module dof {} | control joints dof {}'.format(
                    IK_MODULE.get_dof(), len(CONTROL_JOINT_NAMES))
            attach_conf = sample_tool_ik(IK_MODULE.get_ik, robot, ik_joints, attach_pose, ik_base_link, ik_tool_link_from_tcp=ee_from_tool)
        # assert attach_conf is not None
    else:
        # joint conf sample fn, used when ikfast is not used
        sample_fn = get_sample_fn(robot, control_joints)
        set_joint_positions(robot, control_joints, sample_fn())  # Random seed
        attach_conf = inverse_kinematics(robot, tool_link, attach_pose)

    if (attach_conf is None):
        if verbose : print('attach ik failure.')
        # handles = draw_pose(attach_pose)
        # wait_if_gui()
        # remove_handles(handles)
        return None

    if collision_fn(attach_conf, diagnosis):
        if verbose : print('attach collision failure.')
        return None

    set_joint_positions(robot, control_joints, attach_conf)
    set_pose(body, pregrasp_poses[-1])
    attachment = create_attachment(robot, tool_link, body)
    # set_color(body, GREEN)
    # wait_if_gui()

    # ! we might have the gantry move between the approach and attach
    approach_conf = inverse_kinematics(robot, tool_link, pre_attach_pose)
    if (approach_conf is None):
        if verbose : print('approach ik failure.')
        return None
    if collision_fn(approach_conf, diagnosis):
        if verbose : print('approach collision failure.')
        return None

    if teleops:
        approach_path = [approach_conf, attach_conf]
    else:
        set_joint_positions(robot, control_joints, approach_conf)
        # Cartesian planning only from the robot's base
        approach_path = plan_cartesian_motion(robot, ik_joints[0], tool_link, pre_attach_poses, custom_limits=custom_limits)
        if approach_path is not None:
            # ! pay attention to the jump_threshold
            if not check_path(control_joints, [approach_conf] + approach_path, collision_fn=collision_fn, jump_threshold=None, diagnosis=diagnosis):
                approach_path = None

    if approach_path is None:
        if verbose : print('direct approach motion failure.')
        return None
    if get_distance(approach_path[-1], attach_conf)>PB_IK_TOL:
        attach_conf = approach_path[-1]
    approach_traj = MotionTrajectory(robot, control_joints, approach_path, attachments=[attachment], tag='place_approach', element=index)

    # * retreat motion
    # ! we might have the gantry move between the approach and attach
    set_joint_positions(robot, control_joints, attach_conf)
    post_attach_conf = inverse_kinematics(robot, tool_link, post_attach_pose)
    if (post_attach_conf is None):
        if verbose : print('post-attach ik failure.')
        return None
    if collision_fn(post_attach_conf, diagnosis):
        if verbose : print('post-attach collision failure.')
        return None

    if teleops:
        retreat_path = [attach_conf, post_attach_conf]
    else:
        # detach to post-detach
        set_joint_positions(robot, control_joints, attach_conf)
        # Cartesian planning only from the robot's base
        retreat_path = plan_cartesian_motion(robot, ik_joints[0], tool_link, post_attach_poses, custom_limits=custom_limits)
        if retreat_path is not None:
            if not check_path(control_joints, [attach_conf] + retreat_path, collision_fn=collision_fn, jump_threshold=None, diagnosis=diagnosis):
                retreat_path = None
    if retreat_path is None:
        if verbose : print('direct retreat motion failure.')
        return None
    if get_distance(approach_path[-1], retreat_path[0])>PB_IK_TOL:
        if verbose: print('approach end and retreat start diff!')
        return None

    retreat_traj = MotionTrajectory(robot, control_joints, retreat_path, attachments=[], tag='place_retreat', element=index)
    return Command([approach_traj, retreat_traj])

######################################

# the initial pose is fixed, the goal poses can be generated by rotational symmetry
# so the total grasp posibility is generated by:
# rotational goal pose x grasp sliding
# the approach pose is independent of grasp and symmetry, can be generated independently

# choosing joint resolution:
# http://openrave.org/docs/0.6.6/openravepy/databases.linkstatistics/

def get_place_gen_fn(robot, tool_from_ee, element_from_index, fixed_obstacles, collisions=True,
    max_attempts=IK_MAX_ATTEMPTS, max_grasp=GRASP_MAX_ATTEMPTS, allow_failure=False, verbose=False, bar_only=False,
    precompute_collisions=False, teleops=False):
    if not collisions:
        precompute_collisions = False
    if not bar_only:
        control_joints = joints_from_names(robot, CONTROL_JOINT_NAMES)
        disabled_collisions = get_disabled_collisions(robot)
    else:
        # this is the bar robot
        control_joints = get_movable_joints(robot)
        disabled_collisions = {}
        # ee_body_link = get_links(end_effector)[-1]

    # conditioned sampler
    grasp_gen = get_bar_grasp_gen_fn(element_from_index, reverse_grasp=True, safety_margin_length=0.005)
    pregrasp_gen_fn = get_pregrasp_gen_fn(element_from_index, fixed_obstacles, collision=collisions, teleops=teleops) # max_attempts=max_attempts,

    retreat_distance = RETREAT_DISTANCE
    retreat_vector = retreat_distance*np.array([0, 0, -1])

    gantry_sample_fn = None
    if not bar_only and GANTRY_JOINT_LIMITS is not None:
        gantry_joints = joints_from_names(robot, list(GANTRY_JOINT_LIMITS.keys()))
        gantry_limits = {joint_from_name(robot, jn) : v for jn, v in GANTRY_JOINT_LIMITS.items()}
        gantry_sample_fn = get_sample_fn(robot, gantry_joints, custom_limits=gantry_limits)

    def gen_fn(element, printed=[], diagnosis=False):
        # assert implies(bar_only, element_from_index[element].element_robot is not None)
        # cprint('new stream fn - printed: {}'.format(printed), 'yellow')
        element_obstacles = get_element_body_in_goal_pose(element_from_index, printed)
        obstacles = set(fixed_obstacles) | element_obstacles
        if not collisions:
            obstacles = set()
        elements_order = [e for e in element_from_index if (e != element) and (element_from_index[e].body not in obstacles)]

        # attachment is assumed to be empty here, since pregrasp sampler guarantees that
        collision_fn = get_collision_fn(robot, control_joints, obstacles=obstacles, attachments=[],
                                        self_collisions=ENABLE_SELF_COLLISIONS,
                                        disabled_collisions=disabled_collisions,
                                        custom_limits={} if bar_only else get_custom_limits(robot),
                                        max_distance=MAX_DISTANCE)

        # keep track of sampled traj, prune newly sampled one with more collided element
        element_goal_pose = element_from_index[element].goal_pose
        trajectories = []
        for attempt, grasp_t in enumerate(islice(grasp_gen(element), max_grasp)):
            grasp = grasp_t[0]
            # * ik iterations, usually 1 is enough
            for _ in range(max_attempts):
                # ! when used in pddlstream (except incremental_sm), the pregrasp sampler assumes no elements assembled at all time
                pregrasp_poses, = next(pregrasp_gen_fn(element, element_goal_pose, printed, diagnosis=diagnosis))
                if not pregrasp_poses:
                    if verbose : print('pregrasp failure.')
                    continue

                command = compute_place_path(robot, tool_from_ee, pregrasp_poses, grasp, element, element_from_index, collision_fn,
                    bar_only=bar_only, verbose=verbose, diagnosis=diagnosis, retreat_vector=retreat_vector, teleops=teleops, \
                    gantry_sample_fn=gantry_sample_fn)
                if command is None:
                    continue

                # ? why update safe?
                # command.update_safe(printed)
                # TODO: not need this when running incremental + semantic attachment
                if precompute_collisions:
                    bodies_order = get_element_body_in_goal_pose(element_from_index, elements_order)
                    colliding = command_collision(command, bodies_order)
                    for element2, unsafe in zip(elements_order, colliding):
                        if unsafe:
                            command.set_unsafe(element2)
                        else:
                            command.set_safe(element2)

                # if not is_ground(element, ground_nodes) and (neighboring_elements <= command.colliding):
                #     continue # TODO If all neighbors collide

                trajectories.append(command)
                if precompute_collisions:
                    prune_dominated(trajectories)
                if command not in trajectories:
                    continue

                # if verbose:
                cprint('Place E#{} | Attempts: {} | Trajectories: {} | Colliding: {}'.format(element, attempt, len(trajectories), \
                        sorted([len(t.colliding) for t in trajectories])[0:3]), 'green')

                yield command,
                break
            # else:
            #     # this will run if no break is called, prevent a StopIteraton error
            #     # https://docs.python.org/3/tutorial/controlflow.html#break-and-continue-statements-and-else-clauses-on-loops
                # if allow_failure:
                    # yield None,
        else:
            if verbose:
                cprint('E#{} | Attempts: {} | Max attempts exceeded!'.format(element, max_grasp), 'red')

            if allow_failure:
                yield None,
            else:
                return
    return gen_fn
