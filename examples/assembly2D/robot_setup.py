import numpy as np
import os, sys
from termcolor import cprint
from pybullet_planning import set_camera_pose, connect, create_box, wait_if_gui, set_pose, create_plane, \
    draw_pose, unit_pose, set_camera_pose2, Pose, Point, Euler, RED, BLUE, GREEN, CLIENT, HideOutput, create_obj, apply_alpha, \
    create_flying_body, create_shape, get_mesh_geometry, get_movable_joints, get_configuration, set_configuration, get_links, \
    has_gui, set_color, reset_simulation, disconnect, get_date, WorldSaver, LockRenderer, YELLOW, add_line, draw_circle, pairwise_collision, \
    body_collision_info, get_distance, draw_collision_diagnosis, get_aabb, BodySaver, load_pybullet, get_collision_data, \
    link_from_name, clone_collision_shape, clone_visual_shape, get_relative_pose

from coop_assembly.planning.parsing import get_assembly_path

# robot geometry data files
HERE = os.path.dirname(__file__)
GRIPPER_URDF = os.path.join(HERE, 'data', 'Princeton_gripper.urdf')
# GRIPPER_OBJ_PATH = os.path.join(HERE, 'data', 'duck.obj')
# EE_FROM_TOOL = Pose(point=[0.0, 0.0, 0.3705])

EE_LINK_NAME = 'eef_base_link'
TOOL_LINK_NAME = 'eef_tcp_frame'

SE2_xz = ['x', 'z', 'pitch']
INITIAL_CONF = [-1.0, 0.0, 0.0]

# viz settings
GROUND_COLOR = 0.8*np.ones(3)
BACKGROUND_COLOR = [0.9, 0.9, 1.0] # 229, 229, 255
SHADOWS = True

##########################################

class Conf(object):
    """wrapper for robot (incl. the element robot) configurations in pddlstream formulation
    """
    def __init__(self, robot, positions=None, element=None):
        self.robot = robot
        self.joints = get_movable_joints(self.robot)
        if positions is None:
            positions = get_configuration(self.robot)
        self.positions = positions
        self.element = element
    def assign(self):
        set_configuration(self.robot, self.positions)
    def __repr__(self):
        return '{}(E{})'.format(self.__class__.__name__, self.element)

#############################################

def load_2d_world(viewer=False):
    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    with HideOutput():
       floor = create_plane(color=GROUND_COLOR)
       # duck_body = create_obj(DUCK_OBJ_PATH, scale=0.2 * 1e-3, color=apply_alpha(GREEN, 0.5))
       # treat end effector as a flying 2D robot
    #    collision_id, visual_id = create_shape(get_mesh_geometry(GRIPPER_OBJ_PATH, scale=1), #0.6 * 1e-3
        # collision=True, color=apply_alpha(YELLOW, 0.5))
       ee_body = load_pybullet(GRIPPER_URDF, fixed_base=True)
       tool_from_ee = get_relative_pose(ee_body, link_from_name(ee_body, EE_LINK_NAME), link_from_name(ee_body, TOOL_LINK_NAME))

       ee_body_link = link_from_name(ee_body, 'eef_base_link')
       collision_id = clone_collision_shape(ee_body, ee_body_link, CLIENT)
       visual_id = clone_visual_shape(ee_body, ee_body_link, CLIENT)
       end_effector = create_flying_body(SE2_xz, collision_id, visual_id)
    return end_effector, floor, tool_from_ee
