import os
import pytest
import numpy as np

from pybullet_planning import wait_for_user, connect, has_gui, wait_for_user, LockRenderer, remove_handles, add_line, \
    draw_pose, EndEffector, unit_pose, link_from_name, end_effector_from_body, get_link_pose, \
    dump_world, set_pose, WorldSaver, reset_simulation, disconnect, get_pose

from coop_assembly.data_structure import BarStructure, OverallStructure
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions import contact_to_ground
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE

from coop_assembly.planning import get_picknplace_robot_data, BUILT_PLATE_Z
from coop_assembly.planning import load_world, set_camera, wait_if_gui
from coop_assembly.planning import color_structure, draw_ordered, draw_element, label_elements, label_connector
from coop_assembly.planning import get_element_neighbors, get_connector_from_elements, check_connected, get_connected_structures

from coop_assembly.planning.stream import get_goal_pose_gen_fn, get_bar_grasp_gen_fn, get_ik_gen_fn, get_pregrasp_gen_fn
from coop_assembly.planning.regression import regression
from coop_assembly.planning import TOOL_LINK_NAME, EE_LINK_NAME
from coop_assembly.planning.motion import step_trajectory, display_trajectories

@pytest.fixture
def test_file_name():
    # return 'YJ_12_bars_point2triangle.json'
    return 'single_tet_point2triangle.json'

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
    assert ee_link_name == 'eef_base_link'
    assert tool_link_name == 'eef_tcp_frame'
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
        goal_pose = next(goal_pose_gen_fn(chosen))[0].value
        handles.extend(draw_pose(goal_pose, length=0.01))
        color_structure(element_bodies, printed, next_element=chosen, built_alpha=0.6)
        wait_if_gui(True)
        remove_handles(handles)

@pytest.mark.regression
def test_regression(viewer, test_file_name, collision, motion, stiffness, animate):
    bar_struct, o_struct = load_structure(test_file_name, viewer)
    fixed_obstacles, robot = load_world()

    with WorldSaver():
        plan, data = regression(robot, fixed_obstacles, bar_struct, collision=collision, motion=motion, stiffness=stiffness)
        print(data)

    dump_world()

    # reset_simulation()
    # disconnect()
    watch = True
    if watch and (plan is not None):
        # animate = not (args.disable or args.ee_only)
        # connect(use_gui=viewer)
        # set_camera([attr['point_xyz'] for v, attr in o_struct.vertices(True)])
        # _, robot = load_world()

        display_trajectories(plan, time_step=None, #video=True,
                             animate=animate)
        reset_simulation()
        disconnect()

@pytest.mark.stream
def test_grasp_gen_fn(viewer, test_file_name, collision):
    bar_struct, _ = load_structure(test_file_name, viewer)
    element_bodies = bar_struct.get_element_bodies()
    element_from_index = bar_struct.get_element_from_index()

    obstacles, robot = load_world()
    # draw_pose(get_link_pose(robot, link_from_name(robot, TOOL_LINK_NAME)))

    # printed = set([0,1,2,3])
    # chosen = 4
    printed = set([0,1,2,4,3])
    chosen = 5

    # https://github.com/yijiangh/pybullet_planning/blob/dev/tests/test_grasp.py#L81
    color_structure(element_bodies, printed, next_element=chosen, built_alpha=0.6)
    wait_if_gui(False)

    n_attempts = 10
    # tool_pose = Pose(euler=Euler(yaw=np.pi/2))
    tool_pose = unit_pose()
    end_effector = EndEffector(robot, ee_link=link_from_name(robot, EE_LINK_NAME),
                               tool_link=link_from_name(robot, TOOL_LINK_NAME),
                               visual=False, collision=True)

    goal_pose_gen_fn = get_goal_pose_gen_fn(element_from_index)
    grasp_gen = get_bar_grasp_gen_fn(element_from_index, tool_pose=tool_pose, \
        reverse_grasp=True, safety_margin_length=0.005)
    ik_gen = get_ik_gen_fn(end_effector, element_from_index, obstacles, max_attempts=n_attempts, collision=collision)

    # body_pose = element_from_index[chosen].goal_pose.value
    for _ in range(n_attempts):
        handles = []
        # couple rotations in goal pose' symmetry and translational grasp
        grasp, = next(grasp_gen(chosen))
        world_pose, = next(goal_pose_gen_fn(chosen))
        # pregrasp_poses, = next(pregrasp_gen_fn(chosen, world_pose, printed=printed))
        command, = next(ik_gen(chosen, world_pose, grasp, printed=printed, diagnosis=False))

        # visualize grasp
        p = world_pose.value
        gripper_from_bar = grasp.attach
        set_pose(element_from_index[chosen].body, p)
        world_from_ee = end_effector_from_body(p, gripper_from_bar)
        end_effector.set_pose(world_from_ee)
        handles.extend(draw_pose(world_from_ee, length=0.05))
        handles.extend(draw_pose(p, length=0.05))

        if not command:
            print('no command found')
            # gripper_from_bar = grasp.attach
            # for p in pregrasp_poses:
            #     set_pose(element_from_index[chosen].body, p)
            #     world_from_ee = end_effector_from_body(p, gripper_from_bar)
            #     end_effector.set_pose(world_from_ee)
            #     handles.extend(draw_pose(world_from_ee, length=0.01))
            #     wait_if_gui()
            print('-'*10)
        else:
            print('command found!')
            attach_traj = command.trajectories[0]
            time_step = None if has_gui() else 0.1
            # step_trajectory(attach_traj, attach_traj.attachments, time_step)
            display_trajectories([attach_traj], time_step=time_step, #video=True,
                                 animate=True)
            print('*'*10)

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
def test_connector(viewer):
    test_file_name = 'YJ_12_bars_point2triangle.json'
    # visual test
    bar_struct, _ = load_structure(test_file_name, viewer, color=(1,0,0,0.3))
    element_bodies = bar_struct.get_element_bodies()
    handles = []
    handles.extend(label_elements(element_bodies))
    wait_if_gui()
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
        remove_handles(handles)

    # * neighbor elements from elements
    element_neighbors = get_element_neighbors(connectors, elements)
    for element, connected_bars in element_neighbors.items():
        color_structure(element_bodies, connected_bars, element, built_alpha=0.6)
        wait_if_gui()

    grounded_elements = bar_struct.get_grounded_bar_keys()

    printed_elements = set([9,10,11])
    assert not check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([1,9,10,11])
    assert not check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([2,7,9,11,10])
    grounded_elements = bar_struct.get_grounded_bar_keys()
    assert check_connected(connectors, grounded_elements, printed_elements)

def test_contact_to_ground(viewer):
    test_file_name = 'YJ_12_bars_point2triangle.json'
    bar_struct, _ = load_structure(test_file_name, viewer, color=(1,0,0,0.3))
    element_bodies = bar_struct.get_element_bodies()
    handles = []
    handles.extend(label_elements(element_bodies))
    for bar_key in bar_struct.vertices():
        if bar_struct.vertex[bar_key]['grounded']:
            contact_pts = contact_to_ground(bar_struct.vertex[bar_key], built_plate_z=BUILT_PLATE_Z, scale=1)
            handles.append(add_line(*contact_pts, color=(1,0,0,0), width=2))
    wait_if_gui()
    # and check connected test
