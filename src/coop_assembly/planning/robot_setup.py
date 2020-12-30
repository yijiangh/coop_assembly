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
INCLUDE_ENV_COLLISION_OBJS = 1

########################################

# BUILD_PLATE_CENTER = np.array([550, 0, -14.23])*1e-3
BUILD_PLATE_CENTERs = {
    'kuka' : np.array([500, 0, -14.23])*1e-3,
    # 'abb_track' : np.array([1.35, -2, 30.7*1e-3]),
    # * SP Arch
    'abb_track' : np.array([1.35, -1.5, 30.7*1e-3]),
    # * IT Arch
    # 'abb_track' : np.array([1.35, -1.5, 30.7*1e-3]),
    # * SP Column
    # 'abb_track' : np.array([1.35, -1.5, 30.7*1e-3]),
    # * IT Hydra
    # 'abb_track' : np.array([1.45, -2, 30.7*1e-3]),
    # * topopt tiny valut
    # 'abb_track' : np.array([1.5, -2, 30.7*1e-3]),
}
BUILD_PLATE_CENTER = BUILD_PLATE_CENTERs[ROBOT_NAME]

# [32, 33, 39, 40, 41, 24, 26, 31]

BOTTOM_BUFFER = 0.03
# * IT Hydra
# BOTTOM_BUFFER = 0.06

# * SP Arch
BASE_YAW = np.pi
# * IT Arch
# BASE_YAW = 0
# * SP Column
# BASE_YAW = 0
# * IT Hydra
# BASE_YAW = -np.pi-np.pi/10.0

BASE_ROLL = 0
BASE_PITCH = 0
# * IT Hydra
# BASE_PITCH = np.pi/10

####################
# SP_arch
# 'abb_track' : np.array([1.35, -1.5, 30.7*1e-3]),
# BASE_YAW = np.pi#-np.pi/18

########################################

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

import coop_assembly
def obj_files_from_dir(dir_path):
    cms = []
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith('.obj'):
            cms.append(os.path.join(dir_path, filename))
    return cms

WS_MESH_PATHs = {
    'kuka' : [],
    'abb_track' : obj_files_from_dir(coop_assembly.get_data("models/abb_irb4600_40_255/meshes/collision/ECL_env_collision")),
}
WS_MESH_PATH = WS_MESH_PATHs[ROBOT_NAME]

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

CUSTOM_LIMITSs = {
    'kuka' : {
        'robot_joint_a1': (-np.pi/2, np.pi/2),
    },
    'abb_track' : {
        # 'joint_2': (-0.62, 1.83),
    },
}
CUSTOM_LIMITS = CUSTOM_LIMITSs[ROBOT_NAME]

## Choosing joint resolutions:
## - http://lavalle.pl/planning/node217.html
## - http://openrave.org/docs/latest_stable/openravepy/databases.linkstatistics/
## joint resolution used in transit motions
# RESOLUTION = 0.005
# RESOLUTION = 0.01
# RESOLUTION_RATIO = 1
RESOLUTION_RATIO = 5

INITIAL_CONFs = {
    'kuka': np.radians([5., -90., 100, 5, 10, -5]),
    # 'abb_track' : np.hstack([1.0, np.radians([0,0,0,0,0,0])]),
    # * SP arch
    'abb_track' : np.hstack([0.0, np.radians([0,0,0,0,0,0])]),
    }
INITIAL_CONF = INITIAL_CONFs[ROBOT_NAME]

# TODO: compute joint weight as np.reciprocal(joint velocity bound) from URDF
JOINT_WEIGHTSs = {
    'kuka' : np.reciprocal([6.28318530718, 5.23598775598, 6.28318530718,
                            6.6497044501, 6.77187749774, 10.7337748998]), # sec / radian
    'abb_track' : np.reciprocal([0.1, 2.618, 2.618, 2.618,
                                 6.2832, 6.2832, 7.854]),
    # 'abb_track' : [0.01, 0.01079, 0.00725, 0.012249, 0.009173, 0.037541, 0.01313],
    }
JOINT_WEIGHTS = np.array(JOINT_WEIGHTSs[ROBOT_NAME])

JOINT_RESOLUTIONSs = {
    'kuka' : np.divide(np.ones(JOINT_WEIGHTS.shape), JOINT_WEIGHTS),
    'abb_track' : np.array([0.01, 0.01079, 0.00725, 0.012249, 0.009173, 0.037541, 0.01313]),
    }
JOINT_RESOLUTIONS = RESOLUTION_RATIO * np.array(JOINT_RESOLUTIONSs[ROBOT_NAME])

GANTRY_JOINT_LIMITSs = {
    'kuka' : None,
    'abb_track' : {
        'linear_axis_actuation_joint' : (0.0, 3.3), #3
        },
}
GANTRY_JOINT_LIMITS = GANTRY_JOINT_LIMITSs[ROBOT_NAME]

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
    control_joint_names = robot.get_configurable_joint_names(group=move_group)
    disabled_self_collision_link_names = robot_semantics.disabled_collisions

    ik_base_link_name = base_link_name
    ik_joint_names = control_joint_names
    if len(control_joint_names) > 6:
        ik_group = 'bare_arm'
        ik_base_link_name = robot.get_base_link_name(group=ik_group)
        ik_joint_names = robot.get_configurable_joint_names(group=ik_group)

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

    return (robot_urdf, base_link_name, tool_link_name, ee_link_name, control_joint_names, disabled_self_collision_link_names, ik_base_link_name, ik_joint_names), \
           (workspace_urdf, workspace_robot_disabled_link_names)

def get_gripper_mesh_path():
    return coop_assembly.get_data(EE_MESH_PATH)

#################################

# TOOL_LINK: TCP link
# EE_LINK: attach link for EE geometry (i.e. the flange link)
(ROBOT_URDF, BASE_LINK_NAME, TOOL_LINK_NAME, EE_LINK_NAME, CONTROL_JOINT_NAMES, DISABLED_SELF_COLLISION_LINK_NAMES, IK_BASE_LINK_NAME, IK_JOINT_NAMES), \
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
    limits = {joint_from_name(robot, joint): limits
              for joint, limits in CUSTOM_LIMITS.items()}
    if GANTRY_JOINT_LIMITS:
        gantry_limits = {joint_from_name(robot, jn) : v for jn, v in GANTRY_JOINT_LIMITS.items()}
        limits.update(gantry_limits)
    return limits
