import os
import pytest

from pybullet_planning import wait_for_user, connect, has_gui, wait_for_user, LockRenderer, remove_handles, add_line, \
    draw_pose, get_side_cylinder_grasps, EndEffector, unit_pose, link_from_name, end_effector_from_body, get_link_pose

from coop_assembly.data_structure import BarStructure, OverallStructure
from coop_assembly.assembly_info_generation.offset_motion import get_offset_collision_test
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions import contact_to_ground
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE

from coop_assembly.planning import get_picknplace_robot_data, BUILT_PLATE_Z
from coop_assembly.planning import load_world, set_camera
from coop_assembly.planning import color_structure, draw_ordered, draw_element, label_elements, label_connector
from coop_assembly.planning import get_element_neighbors, get_connector_from_elements, check_connected, get_connected_structures

from coop_assembly.planning.stream import get_goal_pose_gen_fn, get_bar_grasp_gen_fn
from coop_assembly.planning import TOOL_LINK_NAME, EE_LINK_NAME

@pytest.fixture
def test_file_name():
    # @pytest.mark.parametrize('test_file_name', [('YJ_12_bars_point2triangle.json'),])
    return 'YJ_12_bars_point2triangle.json'

def wait_if_gui(enable=True):
    if has_gui() and enable:
        wait_for_user()

def load_structure(test_file_name, viewer, color=(1,0,0,0)):
    """connect pybullet env and load the bar system

    Parameters
    ----------
    test_file_name : [type]
        [description]
    viewer : [type]
        [description]
    color : tuple, optional
        [description], by default (1,0,0,0)

    Returns
    -------
    [type]
        [description]
    """
    connect(use_gui=viewer)
    with LockRenderer():
        load_world()

        here = os.path.dirname(os.path.abspath(__file__))
        test_data_dir = os.path.join(here, 'test_data')

        b_struct_data, o_struct_data, _ = parse_saved_structure_data(os.path.join(test_data_dir, test_file_name))
        o_struct = OverallStructure.from_data(o_struct_data)
        b_struct = BarStructure.from_data(b_struct_data)
        b_struct.create_pb_bodies(color=color)
        o_struct.struct_bar = b_struct # TODO: better way to do this
        set_camera([attr['point_xyz'] for v, attr in o_struct.vertices(True)])
    return b_struct, o_struct

def test_load_robot(viewer):
    robot_data, ws_data = get_picknplace_robot_data()
    robot_urdf, _, tool_link_name, ee_link_name, joint_names, _ = robot_data
    assert ee_link_name == 'eef_tcp_frame'
    assert tool_link_name == 'robot_tool0'
    connect(use_gui=viewer)
    load_world()
    wait_if_gui()

def test_rotate_goal_pose_gen(viewer, test_file_name):
    bar_struct, _ = load_structure(test_file_name, viewer)
    element_bodies = bar_struct.get_element_bodies()
    element_from_index = bar_struct.get_element_from_index()

    printed = set([0,1,2])
    chosen = 4
    goal_pose_gen_fn = get_goal_pose_gen_fn(element_from_index)
    handles = []
    for _ in range(5):
        goal_pose, = next(goal_pose_gen_fn(chosen))
        handles.extend(draw_pose(goal_pose, length=0.01))
        color_structure(element_bodies, printed, next_element=chosen, built_alpha=0.6)
        wait_if_gui(True)
        remove_handles(handles)

@pytest.mark.choreo_wip
def test_grasp_gen_fn(viewer, test_file_name):
    bar_struct, _ = load_structure(test_file_name, viewer)
    element_bodies = bar_struct.get_element_bodies()
    element_from_index = bar_struct.get_element_from_index()

    obstacles, robot = load_world()
    # draw_pose(get_link_pose(robot, link_from_name(robot, TOOL_LINK_NAME)))

    printed = set([0,1,2])
    chosen = 4

    # https://github.com/yijiangh/pybullet_planning/blob/dev/tests/test_grasp.py#L81
    color_structure(element_bodies, printed, next_element=chosen, built_alpha=0.6)
    n_attempts = 20
    # tool_pose = Pose(euler=Euler(yaw=np.pi/2))
    tool_pose = unit_pose()
    print('EE link : {} | tool link : {}'.format(EE_LINK_NAME, TOOL_LINK_NAME))
    end_effector = EndEffector(robot, ee_link=link_from_name(robot, EE_LINK_NAME),
                               tool_link=link_from_name(robot, TOOL_LINK_NAME),
                               visual=False, collision=True)

    goal_pose_gen_fn = get_goal_pose_gen_fn(element_from_index)
    grasp_gen = get_bar_grasp_gen_fn(element_from_index, tool_pose=tool_pose, \
        reverse_grasp=True, safety_margin_length=0.005)
    body_pose = element_from_index[chosen].goal_pose.value
    for _ in range(n_attempts):
        handles = []
        # couple rotations in goal pose' symmetry and translational grasp
        gripper_from_bar = next(grasp_gen(chosen))
        body_pose, = next(goal_pose_gen_fn(chosen))

        world_from_ee = end_effector_from_body(body_pose, gripper_from_bar)
        end_effector.set_pose(world_from_ee)

        handles.extend(draw_pose(world_from_ee, length=0.01))
        wait_if_gui()
        remove_handles(handles)


def test_color_structure(viewer, test_file_name):
    bar_struct, _ = load_structure(test_file_name, viewer)
    element_bodies = bar_struct.get_element_bodies()
    printed = set([0,1,2,3])
    color_structure(element_bodies, printed, 4)
    wait_if_gui()


def test_draw_ordered(viewer, test_file_name):
    bar_struct, _ = load_structure(test_file_name, viewer)
    endpts_from_element = bar_struct.get_axis_pts_from_element()
    draw_ordered(list(bar_struct.vertices()), endpts_from_element)
    wait_if_gui()

# @pytest.mark.choreo_wip
def test_connector(viewer, test_file_name):
    # visual test
    bar_struct, _ = load_structure(test_file_name, viewer, color=(1,0,0,0.3))
    element_bodies = bar_struct.get_element_bodies()
    handles = []
    handles.extend(label_elements(element_bodies))
    if has_gui():
        wait_for_user()
    remove_handles(handles)

    elements = list(element_bodies.keys())
    contact_from_connectors = bar_struct.get_connectors(scale=1e-3)
    connectors = list(contact_from_connectors.keys())

    # * connectors from bar
    connector_from_elements = get_connector_from_elements(connectors, elements)
    for bar in bar_struct.vertices():
        handles = []
        bar_connectors = connector_from_elements[bar]
        for c in list(bar_connectors):
            handles.append(add_line(*contact_from_connectors[c], color=(1,0,0,1), width=2))
        color_structure(element_bodies, set(), next_element=bar, built_alpha=0.6)
        # if has_gui():
        #     wait_for_user()
        remove_handles(handles)

    # * neighbor elements from elements
    element_neighbors = get_element_neighbors(connectors, elements)
    for element, connected_bars in element_neighbors.items():
        color_structure(element_bodies, connected_bars, element, built_alpha=0.6)
        if has_gui():
            wait_for_user()

    grounded_elements = bar_struct.get_grounded_bar_keys()

    printed_elements = set([9,10,11])
    assert not check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([1,9,10,11])
    assert not check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([2,9,11,10])
    grounded_elements = bar_struct.get_grounded_bar_keys()
    assert check_connected(connectors, grounded_elements, printed_elements)

def test_contact_to_ground(viewer, test_file_name):
    bar_struct, _ = load_structure(test_file_name, viewer, color=(1,0,0,0.3))
    element_bodies = bar_struct.get_element_bodies()
    handles = []
    handles.extend(label_elements(element_bodies))
    for bar_key in bar_struct.vertices():
        if bar_struct.vertex[bar_key]['grounded']:
            contact_pts = contact_to_ground(bar_struct.vertex[bar_key], built_plate_z=BUILT_PLATE_Z, scale=1)
            handles.append(add_line(*contact_pts, color=(1,0,0,0), width=2))
    wait_for_user()
    # and check connected test
