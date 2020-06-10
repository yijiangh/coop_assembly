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
    get_link_pose, get_joint_positions, intrinsic_euler_from_quat, implies, pairwise_collision, randomize, get_link_name, base_values_from_pose

from coop_assembly.planning.utils import Command, prune_dominated
from coop_assembly.data_structure import Grasp, WorldPose, MotionTrajectory

# TODO: fix self collision
ENABLE_SELF_COLLISIONS = False
IK_MAX_ATTEMPTS = 1
PREGRASP_MAX_ATTEMPTS = 100
GRASP_MAX_ATTEMPTS = 100

# pregrasp delta sample
EPSILON = 0.05
ANGLE = np.pi/3

# pregrasp interpolation
POS_STEP_SIZE = 0.001
ORI_STEP_SIZE = np.pi/180

RETREAT_DISTANCE = 0.025

# collision checking safe margin
MAX_DISTANCE = 0.0

# def get_goal_pose_gen_fn(element_from_index):
#     def gen_fn(index):
#         """return a world_from_goal_pose, the central point is invariant,
#         just rotate around the bar's local z axis (for bars, the longitude axis)
#         """
#         body_pose = element_from_index[index].goal_pose.value
#         # by default, the longitude axis is z
#         # https://pybullet-planning.readthedocs.io/en/latest/reference/generated/pybullet_planning.interfaces.env_manager.create_cylinder.html#pybullet_planning.interfaces.env_manager.create_cylinder
#         while True:
#             theta = random.uniform(-np.pi, +np.pi)
#             rotation = Pose(euler=Euler(yaw=theta))
#             yield WorldPose(index, multiply(body_pose, rotation)),
#     return gen_fn


def get_2d_element_grasp_gen_fn(element_from_index, tool_pose=unit_pose(), reverse_grasp=False, safety_margin_length=0.0):
    # rotate the box's frame to make x axis align with the longitude axis
    longitude_x = Pose(euler=Euler(yaw=np.pi/2))
    def gen_fn(index):
        # can get from aabb as well
        bar_length = element_from_index[index].wlh[1]
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
    lower = [-epsilon]*2 + [-angle]
    upper = [epsilon]*2 + [angle]
    for [x, y, yaw] in interval_generator(lower, upper): # halton?
        pose = Pose(point=[x,y,0], euler=Euler(yaw=yaw))
        yield pose

def pose2conf(pose):
    return np.concatenate([np.array(pose[0]), intrinsic_euler_from_quat(np.array(pose[1]))])

def get_2d_pregrasp_gen_fn(element_from_index, fixed_obstacles, max_attempts=PREGRASP_MAX_ATTEMPTS, collision=True, teleops=False):
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
                if ee_collision_fn(p):
                # if element_robot_collision_fn(pose2conf(p)):
                    is_colliding = True
                    break
            if not is_colliding:
                yield offset_path,
                break
        else:
            yield None,
    return gen_fn

######################################

def command_collision(command, bodies):
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
        for conf in randomize(trajectory.path):
            # TODO: bisect or refine
            joints = get_movable_joints(robot)
            set_joint_positions(robot, joints, conf)
            for attach in trajectory.attachments:
                attach.assign()
                set_color(attach.child, (0,1,0,0.5))

            #for body, _ in get_bodies_in_region(tool_aabb):
            for i, body in enumerate(bodies):
                if body not in idx_from_body: # Robot
                    continue
                idx = idx_from_body[body]
                for attach in trajectory.attachments:
                    if not collisions[idx]:
                        collisions[idx] |= pairwise_collision(attach.child, body)

                    # if collisions[idx]:
                    #     print('colliding E{} - E{}'.format(attach.child, body))
                    #     wait_if_gui()

        # tool_link_name = TOOL_LINK_NAME if end_effector else get_link_name(robot, get_links(robot)[-1])
        # for tool_pose in randomize(trajectory.get_link_path(tool_link_name)): # TODO: bisect
        #     joints = get_movable_joints(robot)
        # if end_effector:
        #     end_effector.set_pose(tool_pose)
        # if not collisions[idx]:
        #     collisions[idx] |= pairwise_collision(end_effector.body, body)

    # * checking robot bodies
    for trajectory in command.trajectories:
        for robot_conf in randomize(trajectory.path):
            set_joint_positions(trajectory.robot, trajectory.joints, robot_conf)
            for i, body in enumerate(bodies):
                if not collisions[i]:
                    collisions[i] |= pairwise_collision(trajectory.robot, body)

                # if collisions[idx]:
                #     print('colliding R{} - E{}'.format(trajectory.robot, body))
                #     dump_body(trajectory.robot)
                #     wait_if_gui()

    #for element, unsafe in zip(elements, collisions):
    #    command.safe_per_element[element] = unsafe
    return collisions

######################################

def compute_2d_place_path(gripper_robot, pregrasp_poses, grasp, index, element_from_index, verbose=False, teleops=False):
    """ IK computation, attachment. EE path is given.
    """
    body = element_from_index[index].body
    # no inverse kinematics involved in the bar_only mode
    # element_robot = element_from_index[index].element_robot
    gripper_joints = get_movable_joints(gripper_robot)
    gripper_body_link = get_links(gripper_robot)[-1]

    # attach_conf = np.concatenate([pregrasp_poses[-1][0], intrinsic_euler_from_quat(pregrasp_poses[-1][1])])
    attach_conf = base_values_from_pose(pregrasp_poses[-1])
    set_joint_positions(gripper_robot, gripper_joints, attach_conf)
    set_pose(body, pregrasp_poses[-1])
    attachment = create_attachment(gripper_robot, gripper_body_link, body)
    command = Command([MotionTrajectory(gripper_robot, gripper_joints,
                      [base_values_from_pose(p) for p in pregrasp_poses], \
                      attachments=[attachment], tag='place_approach', element=index)])
    return command

######################################

# the initial pose is fixed, the goal poses can be generated by rotational symmetry
# so the total grasp posibility is generated by:
# rotational goal pose x grasp sliding
# the approach pose is independent of grasp and symmetry, can be generated independently

# choosing joint resolution:
# http://openrave.org/docs/0.6.6/openravepy/databases.linkstatistics/

def get_place_gen_fn(end_effector, element_from_index, fixed_obstacles, collisions=True,
    max_attempts=IK_MAX_ATTEMPTS, max_grasp=GRASP_MAX_ATTEMPTS, allow_failure=False, verbose=False, bar_only=False,
    precompute_collisions=False, teleops=False):
    if not collisions:
        precompute_collisions = False

    # goal_pose_gen_fn = get_goal_pose_gen_fn(element_from_index)
    grasp_gen = get_2d_element_grasp_gen_fn(element_from_index, reverse_grasp=True, safety_margin_length=0.005)
    pregrasp_gen_fn = get_2d_pregrasp_gen_fn(element_from_index, fixed_obstacles, collision=collisions, teleops=teleops) # max_attempts=max_attempts,

    retreat_distance = RETREAT_DISTANCE
    retreat_vector = retreat_distance*np.array([0, 0, -1])

    def gen_fn(element, printed=[], diagnosis=False):
        assert implies(bar_only, element_from_index[element].element_robot is not None)

        print('new stream fn - printed: {}'.format(printed))
        element_obstacles = get_element_body_in_goal_pose(element_from_index, printed)
        obstacles = set(fixed_obstacles) | element_obstacles
        if not collisions:
            obstacles = set()
        elements_order = [e for e in element_from_index if (e != element) and (element_from_index[e].body not in obstacles)]

        # attachment is assumed to be empty here, since pregrasp sampler guarantees that
        if not bar_only:
            collision_fn = get_collision_fn(robot, ik_joints, obstacles=obstacles, attachments=[],
                                            self_collisions=ENABLE_SELF_COLLISIONS,
                                            disabled_collisions=disabled_collisions,
                                            custom_limits=get_custom_limits(robot),
                                            max_distance=MAX_DISTANCE)

        # keep track of sampled traj, prune newly sampled one with more collided element
        trajectories = []
        for attempt, grasp_t in enumerate(islice(grasp_gen(element), max_grasp)):
            # world_pose = world_pose_t[0]
            world_pose = element_from_index[element].goal_pose
            grasp = grasp_t[0]
            # * ik iterations, usually 1 is enough
            for _ in range(max_attempts):
                # when used in pddlstream, the pregrasp sampler assumes no elements assembled at all time
                pregrasp_poses, = next(pregrasp_gen_fn(element, world_pose, printed))
                if not pregrasp_poses:
                    if verbose : print('pregrasp failure.')
                    continue

                command = compute_2d_place_path(end_effector, pregrasp_poses, grasp, element, element_from_index)
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
