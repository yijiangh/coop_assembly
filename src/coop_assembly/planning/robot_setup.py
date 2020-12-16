"""this module contains all the robot related setup data.
Might consider exposed as arguments in the future.
"""

import os
# import pytest
import numpy as np
from termcolor import cprint

from compas.robots import RobotModel
from compas_fab.robots import Robot as RobotClass
from compas_fab.robots import RobotSemantics
from pybullet_planning import Pose, link_from_name, has_link, joint_from_name
import coop_assembly

# ! set robot here
# ROBOT_NAME  = 'kuka'
ROBOT_NAME  = 'abb_track'
BUILT_PLATE_Z = -0.025 # meter

ROBOT_URDFs = {
    'kuka' : 'kuka_kr6_r900/urdf/kuka_kr6_r900_gripper.urdf',
    'abb_track' : 'abb_irb4600_40_255/urdf/ECL_robot1_with_track.urdf',
    }
ROBOT_SRDFs = {
    'kuka' : 'kuka_kr6_r900/srdf/kuka_kr6_r900_mit_grasp.srdf',
    'abb_track' : 'abb_irb4600_40_255/srdf/ECL_robot1_with_track.srdf',
    }

EE_MESH_PATHs = {
    'kuka' : "models/kuka_kr6_r900/meshes/mit_arch_grasp_end_effector/collision/mit_arch_grasp_end_effector_collision.stl",
    'abb_track' : "models/abb_irb4600_40_255/meshes/collision/princeton_gripper_collision_m.stl",
    }
EE_MESH_PATH = EE_MESH_PATHs[ROBOT_NAME]

# WS_URDF = 'kuka_kr6_r900/urdf/mit_3-412_workspace.urdf'
# WS_SRDF = 'kuka_kr6_r900/srdf/mit_3-412_workspace.srdf'

try:
    import ikfast_kuka_kr6_r900
    import ikfast_abb_irb4600_40_255
    IK_MODULEs = {
        'kuka' : ikfast_kuka_kr6_r900,
        'abb_track' : ikfast_abb_irb4600_40_255,
    }
    IK_MODULE = IK_MODULEs[ROBOT_NAME]
except ImportError as e:
    IK_MODULE = None
    cprint('{}, Using pybullet ik fn instead'.format(e), 'red')

CUSTOM_LIMITS = {
    'kuka' : {
        'robot_joint_a1': (-np.pi/2, np.pi/2),
    },
    'abb' : {
    },
}

# joint resolution used in transit motions
# 0.003 captures pregrasp collision
# RESOLUTION = 0.003
RESOLUTION = 0.01

# INITIAL_CONF = [0.08, -1.57, 1.74, 0.08, 0.17, -0.08]
# INITIAL_CONF = [0, -np.pi/4, np.pi/4, 0, 0, 0]
INITIAL_CONF = np.radians([5., -90., 100, 5, 10, -5])

# TODO: compute joint weight as np.reciprocal(joint velocity bound) from URDF
JOINT_WEIGHTS = np.reciprocal([6.28318530718, 5.23598775598, 6.28318530718,
                               6.6497044501, 6.77187749774, 10.7337748998]) # sec / radian

#########################################

def get_picknplace_robot_data(robot_name=ROBOT_NAME):
    MODEL_DIR = coop_assembly.get_data('models')

    robot_urdf = os.path.join(MODEL_DIR, ROBOT_URDFs[robot_name])
    robot_srdf = os.path.join(MODEL_DIR, ROBOT_SRDFs[robot_name])

    move_group = 'manipulator_gripper'
    robot_model = RobotModel.from_urdf_file(robot_urdf)
    robot_semantics = RobotSemantics.from_srdf_file(robot_srdf, robot_model)
    robot = RobotClass(robot_model, semantics=robot_semantics)

    base_link_name = robot.get_base_link_name(group=move_group)
    ik_joint_names = robot.get_configurable_joint_names(group=move_group)
    disabled_self_collision_link_names = robot_semantics.disabled_collisions
    # disabled_self_collision_link_names = []
    # * the bare arm flange attach link
    ee_link_name = robot.get_end_effector_link_name(group='manipulator')
    # * the TCP link
    tool_link_name = robot.get_end_effector_link_name(group=move_group)
    # tool_link_name = None # set to None since end effector is not included in the robot URDF, but attached later

    # TODO
    workspace_urdf = None
    # workspace_urdf = os.path.join(MODEL_DIR, WS_URDF)
    # workspace_srdf = os.path.join(MODEL_DIR, WS_SRDF)
    # workspace_model = RobotModel.from_urdf_file(workspace_urdf)
    # workspace_semantics = RobotSemantics.from_srdf_file(workspace_srdf, workspace_model)
    # workspace_semantics = RobotSemantics.from_srdf_file(workspace_srdf, workspace_model)
    # workspace_robot_disabled_link_names = workspace_semantics.disabled_collisions
    workspace_robot_disabled_link_names = []

    return (robot_urdf, base_link_name, tool_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
           (workspace_urdf, workspace_robot_disabled_link_names)

def get_gripper_mesh_path():
    return coop_assembly.get_data(EE_MESH_PATH)

#################################

# TOOL_LINK: TCP link
# EE_LINK: attach link for EE geometry (i.e. the flange link)
(ROBOT_URDF, BASE_LINK_NAME, TOOL_LINK_NAME, EE_LINK_NAME, IK_JOINT_NAMES, DISABLED_SELF_COLLISION_LINK_NAMES), \
(WORKSPACE_URDF, WORKSPACE_ROBOT_DISABLED_LINK_NAMES) = get_picknplace_robot_data(ROBOT_NAME)

#################################

def get_disabled_collisions(robot):
    """get robot's link-link tuples disabled from collision checking

    Parameters
    ----------
    robot : [type]
        [description]

    Returns
    -------
    set of int-tuples
        int for link index in pybullet
    """
    return {tuple(link_from_name(robot, link)
                  for link in pair if has_link(robot, link))
                  for pair in DISABLED_SELF_COLLISION_LINK_NAMES}

def get_custom_limits(robot):
    """[summary]

    Parameters
    ----------
    robot : [type]
        [description]

    Returns
    -------
    [type]
        {joint index : (lower limit, upper limit)}
    """
    return {joint_from_name(robot, joint): limits
            for joint, limits in CUSTOM_LIMITS.items()}
