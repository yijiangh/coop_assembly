import os
import pytest
import numpy as np
from collections import defaultdict
from numpy.linalg import norm
import json
from termcolor import cprint
from itertools import islice
from collections import namedtuple

from pybullet_planning import wait_for_user, connect, has_gui, wait_for_user, LockRenderer, remove_handles, add_line, \
    draw_pose, EndEffector, unit_pose, link_from_name, end_effector_from_body, get_link_pose, \
    dump_world, set_pose, WorldSaver, reset_simulation, disconnect, get_pose, get_date, RED, GREEN, refine_path, joints_from_names, \
    set_joint_positions, create_attachment, wait_if_gui, apply_alpha, set_color

from coop_assembly.data_structure import BarStructure, OverallStructure, MotionTrajectory
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions import contact_to_ground
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE

from coop_assembly.planning import get_picknplace_robot_data, BUILT_PLATE_Z, TOOL_LINK_NAME, EE_LINK_NAME, IK_JOINT_NAMES
from coop_assembly.planning.utils import load_world
from coop_assembly.planning.visualization import color_structure, draw_ordered, draw_element, label_elements, label_connector, set_camera, draw_partial_ordered
from coop_assembly.planning.utils import get_element_neighbors, get_connector_from_elements, check_connected, get_connected_structures, \
    flatten_commands

from coop_assembly.planning.stream import get_goal_pose_gen_fn, get_bar_grasp_gen_fn, get_place_gen_fn, get_pregrasp_gen_fn, command_collision, \
    get_element_body_in_goal_pose
from coop_assembly.planning.regression import regression
from coop_assembly.planning.motion import display_trajectories
from coop_assembly.planning.parsing import load_structure
from coop_assembly.planning.validator import validate_trajectories, validate_pddl_plan
from coop_assembly.planning.utils import recover_sequence, Command
from coop_assembly.planning.stripstream import get_pddlstream, solve_pddlstream
from coop_assembly.planning.run import run_pddlstream

@pytest.fixture
def results_dir():
    here = os.path.dirname(__file__)
    return os.path.join(here, 'results')

@pytest.mark.wip_pddl
def test_solve_pddlstream(viewer, file_spec, collision, bar_only, write, algorithm, watch, debug_mode):
    Arguments = namedtuple('Arguments', ['problem', 'algorithm', 'collisions', 'bar_only', 'partial_ordering', 'costs', 'teleops'])
    partial_ordering = True
    costs = False
    teleops = True
    step_sim = False

    args = Arguments(file_spec, algorithm, collision, bar_only, partial_ordering, costs, teleops)
    run_pddlstream(args, viewer=viewer, watch=watch, debug=debug_mode, step_sim=step_sim, write=write)

@pytest.mark.wip_pddlstream_parse
def test_parse_pddlstream(viewer, file_spec, collision, bar_only):
    bar_struct, o_struct = load_structure(file_spec, viewer, apply_alpha(RED, 0))
    fixed_obstacles, robot = load_world()

    robots = [robot]
    element_from_index = bar_struct.get_element_from_index()
    grounded_elements = bar_struct.get_grounded_bar_keys()
    contact_from_connectors = bar_struct.get_connectors(scale=METER_SCALE)
    connectors = list(contact_from_connectors.keys())

    pddlstream_problem = get_pddlstream(robots, fixed_obstacles, element_from_index, grounded_elements, connectors,
                                        collisions=collision)
    print('Init:', pddlstream_problem.init)
    # if not bar_only:
    #     assert ('Robot', 'r0') in pddlstream_problem.init
    assert all([('Grounded', i) in pddlstream_problem.init for i in grounded_elements])
    assert all([('Element', i) in pddlstream_problem.init for i in element_from_index])
    assert all([('Assembled', i) in pddlstream_problem.init for i in element_from_index])
    assert all([('Joined', e1, e2) in pddlstream_problem.init for e1, e2 in connectors])
    assert all([('Joined', e2, e1) in pddlstream_problem.init for e1, e2 in connectors])

    print('Goal:', pddlstream_problem.goal)
    assert 'and' == pddlstream_problem.goal[0]
    assert set([('Removed', i) for i in element_from_index]) <= set(pddlstream_problem.goal)

@pytest.mark.regression
def test_regression(viewer, file_spec, collision, motion, stiffness, watch, revisit, n_trails, write, bar_only):
    # TODO: retire this in the light of run.py
    bar_struct, o_struct = load_structure(file_spec, viewer, apply_alpha(RED, 0))
    fixed_obstacles, robot = load_world()
    # wait_if_gui()

    n_attempts = int(n_trails)
    success = 0
    splan = None
    for i in range(n_attempts):
        print('#'*10)
        with LockRenderer(True):
            plan, data = regression(robot, fixed_obstacles, bar_struct, collision=collision, motions=motion, stiffness=stiffness,
                revisit=revisit, verbose=False if n_attempts>1 else True, lazy=False, bar_only=bar_only)
            print(data)
        if plan is None:
            cprint('#{}: plan not found'.format(i), 'red')
        else:
            splan = plan
            success += 1
            cprint('#{}: plan found'.format(i), 'green')

    print('#'*10)
    print('revisit: {}'.format(revisit))
    print('motion: {}'.format(motion))
    print('collision: {}'.format(collision))
    print('{} : {} / {}'.format(file_spec, success, n_attempts))

    # watch = viewer
    if (splan is not None):
        if write:
            here = os.path.dirname(__file__)
            plan_path = '{}_regression_solution_{}.json'.format(file_spec, get_date())
            save_path = os.path.join(here, 'results', plan_path)
            with open(save_path, 'w') as f:
               json.dump({'problem' : file_spec,
                          'plan' : [p.to_data() for p in splan]}, f)
            cprint('Result saved to: {}'.format(save_path), 'green')
        if watch:
            # time_step = None if has_gui() else 0.01
            time_step = 0.01 # if bar_only else None
            display_trajectories(splan, time_step=time_step, #video=True,
                                 animate=False)
        if collision:
            valid = validate_pddl_plan(splan, bar_struct, fixed_obstacles, watch=False, allow_failure=has_gui(), \
                bar_only=bar_only, refine_num=1)
            assert valid
    reset_simulation()
    disconnect()

@pytest.mark.rotate_goal_pose
def test_rotate_goal_pose_gen(viewer, file_spec):
    bar_struct, _ = load_structure(file_spec, viewer)
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

@pytest.mark.stream
def test_stream(viewer, file_spec, collision):
    bar_struct, _ = load_structure(file_spec, viewer)
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
    # wait_if_gui()

    n_attempts = 5
    # tool_pose = Pose(euler=Euler(yaw=np.pi/2))
    tool_pose = unit_pose()
    end_effector = EndEffector(robot, ee_link=link_from_name(robot, EE_LINK_NAME),
                               tool_link=link_from_name(robot, TOOL_LINK_NAME),
                               visual=False, collision=True)

    goal_pose_gen_fn = get_goal_pose_gen_fn(element_from_index)
    grasp_gen = get_bar_grasp_gen_fn(element_from_index, tool_pose=tool_pose, \
        reverse_grasp=True, safety_margin_length=0.005)
    pick_gen = get_place_gen_fn(end_effector, element_from_index, obstacles, collisions=collision, verbose=True) #max_attempts=n_attempts,

    # body_pose = element_from_index[chosen].goal_pose.value
    for _ in range(n_attempts):
        handles = []
        # * sample goal pose and grasp
        # couple rotations in goal pose' symmetry and translational grasp
        grasp, = next(grasp_gen(chosen))
        world_pose, = next(goal_pose_gen_fn(chosen))
        # pregrasp_poses, = next(pregrasp_gen_fn(chosen, world_pose, printed=printed))

        # visualize grasp
        p = world_pose.value
        gripper_from_bar = grasp.attach
        set_pose(element_from_index[chosen].body, p)
        world_from_ee = end_effector_from_body(p, gripper_from_bar)

        # end_effector.set_pose(world_from_ee)
        # handles.extend(draw_pose(world_from_ee, length=0.05))
        # handles.extend(draw_pose(p, length=0.05))

        # * sample pick trajectory
        command, = next(pick_gen(chosen, printed=printed, diagnosis=False))

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
            trajs = flatten_commands([command])
            time_step = None if has_gui() else 0.1
            display_trajectories(trajs, time_step=time_step, #video=True,
                                 animate=True)
            print('*'*10)

        wait_if_gui()
        remove_handles(handles)


@pytest.mark.color_structure
def test_color_structure(viewer, file_spec):
    bar_struct, _ = load_structure(file_spec, viewer)
    element_bodies = bar_struct.get_element_bodies()
    printed = set([0,1,2,3])
    color_structure(element_bodies, printed, 4)
    wait_if_gui()


@pytest.mark.draw
def test_draw_ordered(viewer, file_spec):
    bar_struct, _ = load_structure(file_spec, viewer)
    endpts_from_element = bar_struct.get_axis_pts_from_element()
    h = draw_ordered(list(bar_struct.vertices()), endpts_from_element)
    wait_if_gui()
    remove_handles(h)

    elements_from_layer = defaultdict(set)
    for v in bar_struct.vertices():
        elements_from_layer[bar_struct.vertex[v]['layer']].add(v)
    draw_partial_ordered(elements_from_layer, endpts_from_element)
    wait_if_gui()


@pytest.mark.connector
def test_connector(viewer):
    # visual test
    file_spec = '12_bars'
    bar_struct, _ = load_structure(file_spec, viewer)
    element_bodies = bar_struct.get_element_bodies(apply_alpha(RED, 0.3))
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

    printed_elements = set([2])
    assert check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([0,1])
    assert check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([9,10,11])
    assert not check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([1,9,10,11])
    assert not check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([2,7,9,11,10])
    grounded_elements = bar_struct.get_grounded_bar_keys()
    assert check_connected(connectors, grounded_elements, printed_elements)

@pytest.mark.connector_db
def test_connector_debug(viewer, file_spec):
    # visual test
    bar_struct, _ = load_structure(file_spec, viewer)
    element_bodies = bar_struct.get_element_bodies(color=(1,0,0,0.3))
    handles = []
    handles.extend(label_elements(element_bodies))
    wait_if_gui()
    remove_handles(handles)

    # elements = list(element_bodies.keys())
    contact_from_connectors = bar_struct.get_connectors(scale=1e-3)
    connectors = list(contact_from_connectors.keys())

    grounded_elements = bar_struct.get_grounded_bar_keys()

    printed_elements = set([0])
    assert check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([0,1])
    assert check_connected(connectors, grounded_elements, printed_elements)

    printed_elements = set([0,3])
    assert check_connected(connectors, grounded_elements, printed_elements)

@pytest.mark.contact_ground
def test_contact_to_ground(viewer, file_spec):
    bar_struct, _ = load_structure(file_spec, viewer, color=(1,0,0,0.3))
    element_bodies = bar_struct.get_element_bodies()
    handles = []
    handles.extend(label_elements(element_bodies))
    for bar_key in bar_struct.vertices():
        if bar_struct.vertex[bar_key]['grounded']:
            contact_pts = contact_to_ground(bar_struct.vertex[bar_key], built_plate_z=BUILT_PLATE_Z, scale=1)
            handles.append(add_line(*contact_pts, color=(1,0,0,0), width=2))
    wait_if_gui()
    # and check connected test

# def test_load_robot(viewer):
#     robot_data, ws_data = get_picknplace_robot_data()
#     robot_urdf, _, tool_link_name, ee_link_name, joint_names, _ = robot_data
#     assert ee_link_name == 'eef_base_link'
#     assert tool_link_name == 'eef_tcp_frame'
#     connect(use_gui=viewer)
#     load_world()
#     wait_if_gui()
