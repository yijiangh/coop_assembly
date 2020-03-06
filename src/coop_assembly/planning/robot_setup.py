"""this module contains all the robot related setup data.
Might consider exposed as arguments in the future.
"""

import os
import pytest
import numpy as np

from compas.robots import RobotModel
from compas_fab.robots import Robot as RobotClass
from compas_fab.robots import RobotSemantics
from pybullet_planning import Pose, link_from_name, has_link, joint_from_name
import coop_assembly

BUILT_PLATE_Z = -0.025 # meter

ROBOT_URDF = 'kuka_kr6_r900/urdf/kuka_kr6_r900_gripper.urdf'
ROBOT_SRDF = 'kuka_kr6_r900/srdf/kuka_kr6_r900_mit_grasp.srdf'

WS_URDF = 'kuka_kr6_r900/urdf/mit_3-412_workspace.urdf'
WS_SRDF = 'kuka_kr6_r900/srdf/mit_3-412_workspace.srdf'

import ikfast_kuka_kr6_r900
IK_MODULE = ikfast_kuka_kr6_r900

CUSTOM_LIMITS = {
    'robot_joint_a1': (-np.pi/2, np.pi/2),
}

# joint resolution used in transit motions
RESOLUTION = 0.1

# TODO: compute joint weight as np.reciprocal(joint velocity bound) from URDF
# JOINT_WEIGHTS = np.array([0.3078557810844393, 0.443600199302506, 0.23544367607317915,
#                           0.03637161028426032, 0.04644626184081511, 0.015054267683041092])

#########################################

def get_picknplace_robot_data():
    MODEL_DIR = coop_assembly.get_data('models')

    robot_urdf = os.path.join(MODEL_DIR, ROBOT_URDF)
    robot_srdf = os.path.join(MODEL_DIR, ROBOT_SRDF)

    workspace_urdf = os.path.join(MODEL_DIR, WS_URDF)
    workspace_srdf = os.path.join(MODEL_DIR, WS_SRDF)

    move_group = 'manipulator_gripper'
    robot_model = RobotModel.from_urdf_file(robot_urdf)
    robot_semantics = RobotSemantics.from_srdf_file(robot_srdf, robot_model)
    robot = RobotClass(robot_model, semantics=robot_semantics)

    base_link_name = robot.get_base_link_name(group=move_group)
    ee_link_name = robot.get_end_effector_link_name(group=move_group)
    # ee_link_name = None # set to None since end effector is not included in the robot URDF, but attached later
    ik_joint_names = robot.get_configurable_joint_names(group=move_group)
    disabled_self_collision_link_names = robot_semantics.get_disabled_collisions()
    tool_root_link_name = robot.get_end_effector_link_name(group='manipulator')

    workspace_model = RobotModel.from_urdf_file(workspace_urdf)
    workspace_semantics = RobotSemantics.from_srdf_file(workspace_srdf, workspace_model)
    workspace_robot_disabled_link_names = workspace_semantics.get_disabled_collisions()
    workspace_robot_disabled_link_names = []

    return (robot_urdf, base_link_name, tool_root_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
           (workspace_urdf, workspace_robot_disabled_link_names)

def get_picknplace_end_effector_urdf():
    return coop_assembly.get_data('models/kuka_kr6_r900/urdf/mit_arch_grasp_end_effector.urdf')

def get_picknplace_tcp_def():
    # TODO: should be derived from the end effector URDF
    # in meter
    return Pose(point=[-0.002851003, 0.001035, 0.188155183])

def get_robot_init_conf():
    # radius
    initial_conf = [0.08, -1.57, 1.74, 0.08, 0.17, -0.08]
    return initial_conf

#################################

(ROBOT_URDF, BASE_LINK_NAME, TOOL_ROOT_LINK_NAME, EE_LINK_NAME, IK_JOINT_NAMES, DISABLED_SELF_COLLISION_LINK_NAMES), \
(WORKSPACE_URDF, WORKSPACE_ROBOT_DISABLED_LINK_NAMES) = get_picknplace_robot_data()

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
