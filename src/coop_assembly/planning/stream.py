import numpy as np
import random
import math

from collections import namedtuple

# from extrusion.utils import get_disabled_collisions, get_custom_limits, MotionTrajectory

from pybullet_planning import link_from_name, set_pose, \
    multiply, invert, inverse_kinematics, plan_direct_joint_motion, Attachment, set_joint_positions, plan_joint_motion, \
    get_configuration, wait_for_interrupt, point_from_pose, HideOutput, load_pybullet, draw_pose, unit_quat, create_obj, \
    add_body_name, get_pose, pose_from_tform, connect, WorldSaver, get_sample_fn, \
    wait_for_duration, enable_gravity, enable_real_time, trajectory_controller, simulate_controller, \
    add_fixed_constraint, remove_fixed_constraint, Pose, Euler, get_collision_fn, LockRenderer, user_input, has_gui, \
    disconnect, unit_pose, Point, get_distance, sample_tool_ik, joints_from_names, interval_generator, get_floating_body_collision_fn, \
    interpolate_poses, create_attachment, plan_cartesian_motion

from .robot_setup import EE_LINK_NAME, get_disabled_collisions, IK_MODULE, get_custom_limits, IK_JOINT_NAMES, BASE_LINK_NAME, TOOL_LINK_NAME
from coop_assembly.data_structure import Grasp, WorldPose, MotionTrajectory

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

def get_delta_pose_generator(epsilon=0.01, angle=np.pi/2, **kwargs):
    """sample an infinitestimal pregrasp pose

    Parameters
    ----------
    epsilon : [type]
        [description]
    angle : [type], optional
        [description], by default np.pi/2

    Yields
    -------
    [type]
        [description]
    """
    lower = [-epsilon]*3 + [-angle]*3
    upper = [epsilon]*3 + [angle]*3
    for [x, y, z, roll, pitch, yaw] in interval_generator(lower, upper):
        pose = Pose(point=[x,y,z], euler=Euler(roll=roll, pitch=pitch, yaw=yaw))
        yield pose

def get_pregrasp_gen_fn(element_from_index, fixed_obstacles, epsilon=10*1e-3, angle=np.pi/2, max_attempts=50, max_distance=0, \
    **kwargs):
    pose_gen = get_delta_pose_generator(epsilon, angle=angle, **kwargs)

    def gen_fn(index, pose, printed, diagnosis=False):
        body = element_from_index[index].body
        set_pose(body, pose.value)

        element_obstacles = {element_from_index[e].body for e in list(printed)}
        obstacles = set(fixed_obstacles) | element_obstacles

        ee_collision_fn = get_floating_body_collision_fn(body, obstacles, max_distance=max_distance)

        for _ in range(max_attempts):
            delta_pose = next(pose_gen)
            offset_pose = multiply(pose.value, delta_pose)
            is_colliding = False
            offset_path = list(interpolate_poses(offset_pose, pose.value, **kwargs))
            for p in offset_path[:-1]:
                # TODO: if colliding at the world_from_bar pose, use local velocity + normal check
                # TODO: normal can be derived from
                if ee_collision_fn(p):
                    is_colliding = True
                    break
            if not is_colliding:
                yield offset_path,
                break
        return None

    return gen_fn

######################################

# the initial pose is fixed, the goal poses can be generated by rotational symmetry
# so the total grasp posibility is generated by:
# rotational goal pose x grasp sliding
# the approach pose is independent of grasp and symmetry, can be generated independently

def get_ik_gen_fn(end_effector, element_from_index, fixed_obstacles, collision=True, max_attempts=25, \
    epsilon=0.01, angle=np.pi/2, max_distance=0, **kwargs):
    """return the ik generating function when placing
    # TODO: for now, we always assume the picking is collision-free

    Parameters
    ----------
    end_effector : EndEffector
        [description]
    element_from_index : [type]
        [description]
    obstacle_from_name : [type]
        [description]
    max_attempts : int, optional
        [description], by default 25

    Returns
    -------
    function handle
    """
    robot = end_effector.robot
    ee_from_tool = invert(end_effector.tool_from_root)
    # ik_joints = get_movable_joints(robot)
    ik_joints = joints_from_names(robot, IK_JOINT_NAMES)
    robot_base_link = link_from_name(robot, BASE_LINK_NAME)
    if IK_MODULE:
        assert IK_MODULE.get_dof() == len(ik_joints)
        # free_dof safe_guard?
    tool_link = link_from_name(robot, TOOL_LINK_NAME)
    disabled_collisions = get_disabled_collisions(robot)
    # joint conf sample fn, used when ikfast is not used
    sample_fn = get_sample_fn(robot, ik_joints)
    pregrasp_gen_fn = get_pregrasp_gen_fn(element_from_index, fixed_obstacles, epsilon=epsilon, angle=angle, max_attempts=max_attempts, max_distance=max_distance)

    # approach_distance = 0.1
    # approach_vector = approach_distance*np.array([0, 0, -1])

    def gen_fn(index, pose, grasp, printed=[], diagnosis=False):
        """[summary]

        Parameters
        ----------
        index : [type]
            index of the element
        pose : WorldPose
            world_from_object pose, can wire in a rotational sampler outside if rotational symmetry exists
        grasp : Grasp
            grasp instance

        Yields
        -------
        MotionTrajectory
            [description]
        """
        body = element_from_index[index].body
        set_pose(body, pose.value)

        element_obstacles = {element_from_index[e].body for e in list(printed)}
        obstacles = set(fixed_obstacles) | element_obstacles
        if not collision:
            obstacles = set()

        collision_fn = get_collision_fn(robot, ik_joints, obstacles=obstacles, attachments=[],
                                        self_collisions=True,
                                        disabled_collisions=disabled_collisions,
                                        custom_limits=get_custom_limits(robot))

        for _ in range(max_attempts):
            pregrasp_poses, = next(pregrasp_gen_fn(index, pose, printed))
            if not pregrasp_poses:
                continue

            pre_attach_poses = [multiply(bar_pose, invert(grasp.attach)) for bar_pose in pregrasp_poses]
            attach_pose = pre_attach_poses[-1]
            approach_pose = pre_attach_poses[0]
            # approach_pose = multiply(attach_pose, (approach_vector, unit_quat()))

            if IK_MODULE:
                attach_conf = sample_tool_ik(IK_MODULE.get_ik, robot, ik_joints, attach_pose, robot_base_link, ik_tool_link_from_tcp=ee_from_tool)
            else:
                set_joint_positions(robot, ik_joints, sample_fn())  # Random seed
                attach_conf = inverse_kinematics(robot, tool_link, attach_pose)

            if (attach_conf is None) or collision_fn(attach_conf, diagnosis):
                continue
            set_joint_positions(robot, ik_joints, attach_conf)
            attachment = create_attachment(robot, tool_link, body)

            #if USE_IKFAST:
            #    approach_conf = sample_tool_ik(robot, approach_pose, nearby_conf=attach_conf)
            #else:
            approach_conf = inverse_kinematics(robot, tool_link, approach_pose)
            if (approach_conf is None) or collision_fn(approach_conf, diagnosis):
                continue
            set_joint_positions(robot, ik_joints, approach_conf)

            path = plan_direct_joint_motion(robot, ik_joints, attach_conf,
                                            obstacles=obstacles,
                                            self_collisions=True,
                                            disabled_collisions=disabled_collisions,
                                            attachments=[attachment])
            # path = plan_cartesian_motion(robot, first_joint, target_link, waypoint_poses)
            if path is None: # TODO: retreat
                continue
            #path = [approach_conf, attach_conf]
            attachment = Attachment(robot, tool_link, grasp.attach, body)
            traj = MotionTrajectory(robot, ik_joints, path, attachments=[attachment])
            yield approach_conf, traj
            break
        # TODO: check if pddlstream is happy with returning None
        yield None
        return
    return gen_fn

