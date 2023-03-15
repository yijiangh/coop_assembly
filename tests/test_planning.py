import os
import pytest
import numpy as np
from collections import defaultdict
from numpy.linalg import norm
import json
from termcolor import cprint, colored
from itertools import islice
from collections import namedtuple

import pybullet_planning as pp
from pybullet_planning import connect, has_gui, LockRenderer, remove_handles, add_line, \
    draw_pose, EndEffector, unit_pose, link_from_name, end_effector_from_body, get_link_pose, \
    dump_world, set_pose, WorldSaver, reset_simulation, disconnect, get_pose, get_date, RED, GREEN, refine_path, joints_from_names, \
    set_joint_positions, create_attachment, wait_if_gui, apply_alpha, set_color, create_shape, get_mesh_geometry, create_flying_body, \
    SE3, YELLOW, get_movable_joints, get_relative_pose, multiply, plan_joint_motion, Pose, Euler, invert

from coop_assembly.data_structure import BarStructure, OverallStructure, MotionTrajectory
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions import contact_to_ground
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE

from coop_assembly.planning import get_picknplace_robot_data, TOOL_LINK_NAME, EE_LINK_NAME
from coop_assembly.planning.utils import load_world
from coop_assembly.planning.visualization import color_structure, draw_ordered, draw_element, label_elements, label_connector, set_camera, draw_partial_ordered
from coop_assembly.planning.visualization import display_trajectories
from coop_assembly.planning.utils import get_element_neighbors, get_connector_from_elements, check_connected, \
    flatten_commands #get_connected_structures,

from coop_assembly.planning.stream import get_bar_grasp_gen_fn, get_place_gen_fn, get_pregrasp_gen_fn, command_collision, \
    get_element_body_in_goal_pose, se3_conf_from_pose
from coop_assembly.planning.regression import regression
from coop_assembly.planning.parsing import load_structure, RESULTS_DIRECTORY, unpack_structure
from coop_assembly.planning.validator import validate_trajectories, validate_pddl_plan
from coop_assembly.planning.utils import recover_sequence, Command
from coop_assembly.planning.robot_setup import get_gripper_mesh_path, get_disabled_collisions, ROBOT_NAME, INITIAL_CONF
from coop_assembly.planning.robot_setup import BUILD_PLATE_CENTER, BASE_YAW, BOTTOM_BUFFER

@pytest.fixture
def results_dir():
    here = os.path.dirname(__file__)
    return os.path.join(here, 'results')

# @pytest.mark.skip(reason='not ready to be auto tested...')
# @pytest.mark.pddlstream
# def test_solve_pddlstream(viewer, file_spec, collision, bar_only, write, algorithm, watch, debug_mode):
#     from coop_assembly.planning.run import run_pddlstream
#     Arguments = namedtuple('Arguments', ['problem', 'algorithm', 'collisions', 'bar_only', 'partial_ordering', 'costs', 'teleops'])
#     partial_ordering = True
#     costs = False
#     teleops = True
#     step_sim = False

#     args = Arguments(file_spec, algorithm, collision, bar_only, partial_ordering, costs, teleops)
#     run_pddlstream(args, viewer=viewer, watch=watch, debug=debug_mode, step_sim=step_sim, write=write)

@pytest.mark.skip(reason='not ready to be auto tested...')
@pytest.mark.pddlstream_parse
def test_parse_pddlstream(viewer, file_spec, collision, bar_only):
    from coop_assembly.planning.stripstream import get_pddlstream

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
    bar_struct, _ = load_structure(file_spec, viewer, apply_alpha(RED, 0))
    fixed_obstacles, robot = load_world()
    tool_from_ee = get_relative_pose(robot, link_from_name(robot, EE_LINK_NAME), link_from_name(robot, TOOL_LINK_NAME))
    # wait_if_gui()

    ee_mesh_path = get_gripper_mesh_path()
    collision_id, visual_id = create_shape(get_mesh_geometry(ee_mesh_path, scale=1e-3), collision=True, color=apply_alpha(YELLOW, 0.5))
    end_effector = create_flying_body(SE3, collision_id, visual_id)

    element_from_index = bar_struct.get_element_from_index()
    grounded_elements = bar_struct.get_grounded_bar_keys()
    contact_from_connectors = bar_struct.get_connectors(scale=METER_SCALE)
    connectors = list(contact_from_connectors.keys())

    n_attempts = int(n_trails)
    success = 0
    splan = None
    for i in range(n_attempts):
        print('#'*10)
        with LockRenderer(True):
            plan, data = regression(end_effector if bar_only else robot, tool_from_ee, fixed_obstacles, element_from_index, grounded_elements, connectors,
                collision=collision, motions=motion, stiffness=stiffness,
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
            valid = validate_pddl_plan(splan, fixed_obstacles, element_from_index, grounded_elements, watch=False, allow_failure=has_gui(), \
                bar_only=bar_only, refine_num=1)
            assert valid
    reset_simulation()
    disconnect()

import pybullet as p
from pybullet_planning import NULL_ID, STATIC_MASS, CARTESIAN_TYPES

def clone_body_fixed_joints(body):
    client = pp.get_client(client) # client is the new client for the body
    if links is None:
        links = pp.get_links(body)
    #movable_joints = [joint for joint in links if is_movable(body, joint)]
    new_from_original = {}
    base_link = get_link_parent(body, links[0]) if links else BASE_LINK
    new_from_original[base_link] = NULL_ID

    masses = []
    collision_shapes = []
    visual_shapes = []
    positions = [] # list of local link positions, with respect to parent
    orientations = [] # list of local link orientations, w.r.t. parent
    inertial_positions = [] # list of local inertial frame pos. in link frame
    inertial_orientations = [] # list of local inertial frame orn. in link frame
    parent_indices = []
    joint_types = []
    joint_axes = []
    for i, link in enumerate(links):
        new_from_original[link] = i
        joint_info = get_joint_info(body, link)
        dynamics_info = get_dynamics_info(body, link)
        masses.append(dynamics_info.mass)
        collision_shapes.append(clone_collision_shape(body, link, client) if collision else NULL_ID)
        visual_shapes.append(clone_visual_shape(body, link, client) if visual else NULL_ID)
        point, quat = get_local_link_pose(body, link)
        positions.append(point)
        orientations.append(quat)
        inertial_positions.append(dynamics_info.local_inertial_pos)
        inertial_orientations.append(dynamics_info.local_inertial_orn)
        parent_indices.append(new_from_original[joint_info.parentIndex] + 1) # TODO: need the increment to work
        joint_types.append(joint_info.jointType)
        joint_axes.append(joint_info.jointAxis)
    # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169

    base_dynamics_info = get_dynamics_info(body, base_link)
    base_point, base_quat = get_link_pose(body, base_link)
    new_body = p.createMultiBody(baseMass=base_dynamics_info.mass,
                                 baseCollisionShapeIndex=clone_collision_shape(body, base_link, client) if collision else NULL_ID,
                                 baseVisualShapeIndex=clone_visual_shape(body, base_link, client) if visual else NULL_ID,
                                 basePosition=base_point,
                                 baseOrientation=base_quat,
                                 baseInertialFramePosition=base_dynamics_info.local_inertial_pos,
                                 baseInertialFrameOrientation=base_dynamics_info.local_inertial_orn,
                                 linkMasses=masses,
                                 linkCollisionShapeIndices=collision_shapes,
                                 linkVisualShapeIndices=visual_shapes,
                                 linkPositions=positions,
                                 linkOrientations=orientations,
                                 linkInertialFramePositions=inertial_positions,
                                 linkInertialFrameOrientations=inertial_orientations,
                                 linkParentIndices=parent_indices,
                                 linkJointTypes=joint_types,
                                 linkJointAxis=joint_axes,
                                 physicsClientId=client)
    #set_configuration(new_body, get_joint_positions(body, movable_joints)) # Need to use correct client
    # for joint, value in zip(range(len(links)), get_joint_positions(body, links)):
    #     # TODO: check if movable?
    #     p.resetJointState(new_body, joint, value, targetVelocity=0, physicsClientId=client)

@pytest.mark.stream
def test_stream(viewer, file_spec, collision, bar_only, debug_mode):
    bar_struct, _ = load_structure(file_spec, viewer=True)

    # * transform model
    bar_radius = bar_struct.node[0]['radius']*METER_SCALE
    new_world_from_base = Pose(point=(BUILD_PLATE_CENTER + np.array([0,0,bar_radius+BOTTOM_BUFFER])))
    world_from_base = Pose(point=bar_struct.base_centroid(METER_SCALE))
    rotation = Pose(euler=Euler(yaw=BASE_YAW))
    tf = multiply(new_world_from_base, rotation, invert(world_from_base))
    bar_struct.transform(tf, scale=METER_SCALE)
    bar_struct.generate_grounded_connection()

    element_from_index = bar_struct.get_element_from_index(scale=METER_SCALE)
    element_bodies = bar_struct.get_element_bodies(scale=METER_SCALE, color=apply_alpha(RED,0.2))
    # element_from_index, grounded_elements, contact_from_connectors, connectors = \
    #     unpack_structure(bar_struct, scale=METER_SCALE, color=apply_alpha(RED,0.2)) #chosen_bars=chosen_bars,

    obstacles, robot = load_world(use_floor=True, built_plate_z=BUILD_PLATE_CENTER[2]-0.01)
    draw_pose(get_link_pose(robot, link_from_name(robot, TOOL_LINK_NAME)))
    draw_pose(get_link_pose(robot, link_from_name(robot, EE_LINK_NAME)))
    pp.set_joint_positions(robot, pp.get_movable_joints(robot), INITIAL_CONF)

    printed = set([0,1,2,3])
    chosen = 4
    #
    # printed = set()
    # # printed = set([0,1,2,4])
    # chosen = 0

    # https://github.com/yijiangh/pybullet_planning/blob/dev/tests/test_grasp.py#L81
    color_structure(element_bodies, printed, next_element=chosen, built_alpha=0.6)
    # wait_if_gui("Structure colored")

    n_attempts = 5
    tool_pose = unit_pose()
    tool_from_ee = get_relative_pose(robot, link_from_name(robot, EE_LINK_NAME), link_from_name(robot, TOOL_LINK_NAME))
    # if bar_only:
    # ee_mesh_path = get_gripper_mesh_path()
    # collision_id, visual_id = create_shape(get_mesh_geometry(ee_mesh_path, scale=1e-3), collision=True, color=apply_alpha(YELLOW, 0.5))
    # * create a dummy box as the robot to carry the end effector attachments
    pp.set_color(robot, apply_alpha(pp.RED, 0.1))

    collision_id, visual_id = pp.create_shape(pp.get_box_geometry(0.01,0.01,0.01), color=pp.RED)
    end_effector = create_flying_body(SE3, collision_id, visual_id)
    ee_joints = get_movable_joints(end_effector)
    set_joint_positions(end_effector, ee_joints, se3_conf_from_pose(pp.get_link_pose(robot, link_from_name(robot, EE_LINK_NAME))))
    # pp.set_pose(end_effector, pp.get_link_pose(robot, link_from_name(robot, EE_LINK_NAME)))

    selected_links = pp.get_link_subtree(robot, 6)[1:] # TODO: child_link_from_joint?
    sub_ee_robot = pp.clone_body(robot, links=selected_links, visual=True, collision=True) # TODO: joint limits
    flying_ee_attach = pp.create_attachment(end_effector, pp.get_links(end_effector)[-1], sub_ee_robot)
    ee_attachments = [flying_ee_attach]

    # pp.set_pose(jointed_gripper, pp.Pose(point=(0,1,0)))
    # pp.camera_focus_on_body(end_effector)

    grasp_gen = get_bar_grasp_gen_fn(element_from_index, tool_pose=tool_pose, reverse_grasp=True, safety_margin_length=0.005)
    # pregrasp_gen_fn = get_pregrasp_gen_fn(element_from_index, obstacles, collision=collision, teleops=False) # max_attempts=max_attempts,
    pick_gen = get_place_gen_fn(end_effector if bar_only else robot, tool_from_ee, element_from_index, obstacles,
        collisions=collision, verbose=True, bar_only=bar_only, attachments=ee_attachments, teleops=False) #max_attempts=n_attempts,

    element_goal_pose = element_from_index[chosen].goal_pose
    for _ in range(n_attempts):
        handles = []
        # * sample goal pose and grasp
        # couple rotations in goal pose' symmetry and translational grasp
        # sample a grasp separately to verify
        # grasp, = next(grasp_gen(chosen))
        # # pregrasp_poses, = next(pregrasp_gen_fn(chosen, element_goal_pose, printed=printed))

        # # visualize grasp
        # goal_pose = element_goal_pose.value
        # gripper_from_bar = grasp.gripper_from_object
        # set_pose(element_from_index[chosen].body, goal_pose)
        # world_from_tool = end_effector_from_body(goal_pose, gripper_from_bar)

        # # handles.extend(draw_pose(world_from_tool, length=0.05))
        # # handles.extend(draw_pose(goal_pose, length=0.05))

        # # if bar_only:
        # world_from_ee =pp.multiply(world_from_tool, tool_from_ee)
        # set_joint_positions(end_effector, ee_joints, se3_conf_from_pose(world_from_ee))
        # pp.camera_focus_on_body(end_effector)
        # for ea in ee_attachments:
        #     ea.assign()

        # * sample pick trajectory
        command, = next(pick_gen(chosen, printed=printed, diagnosis=debug_mode))

        if not command:
            cprint('no command found', 'red')
            # gripper_from_bar = grasp.attach
            # for p in pregrasp_poses:
            #     set_pose(element_from_index[chosen].body, p)
            #     world_from_ee = end_effector_from_body(p, gripper_from_bar)
            #     end_effector.set_pose(world_from_ee)
            #     handles.extend(draw_pose(world_from_ee, length=0.01))
            #     wait_if_gui()
            print('-'*10)
        else:
            cprint('command found!', 'green')
            trajs = flatten_commands([command])
            time_step = None if has_gui() else 0.1
            display_trajectories(trajs, time_step=time_step, #video=True,
                                 animate=True, element_from_index=element_from_index)
            print('*'*10)

        wait_if_gui()
        remove_handles(handles)

# https://github.com/yijiangh/assembly_instances/blob/master/tests/conftest.py#L25
@pytest.mark.load_robot
def test_load_robot(viewer, write):
    robot_data, ws_data = get_picknplace_robot_data()
    robot_urdf, _, tool_link_name, ee_link_name, joint_names, _, _, _ = robot_data
    assert ee_link_name == 'eef_base_link'
    assert tool_link_name == 'eef_tcp_frame'
    connect(use_gui=viewer)
    obstacles, robot = load_world(use_floor=False)
    disabled_collisions = get_disabled_collisions(robot)
    joints = joints_from_names(robot, joint_names)

    if ROBOT_NAME == 'abb_track':
        draw_pose(unit_pose(), length=1.0)
        start_conf = INITIAL_CONF
        end_conf = [3.6, 0.506, 0.471, -0.244, -1.239, 0.576, 0.000]

        set_joint_positions(robot, joints, start_conf)

        draw_pose(get_link_pose(robot, link_from_name(robot, TOOL_LINK_NAME)))
        draw_pose(get_link_pose(robot, link_from_name(robot, EE_LINK_NAME)))
        wait_if_gui()

        path = plan_joint_motion(robot, joints, end_conf, obstacles=obstacles, attachments=[],
                                 self_collisions=True, disabled_collisions=disabled_collisions,
                                 diagnosis=True,
                            )
                                #  extra_disabled_collisions=extra_disabled_collisions,
                                #  weights=weights, resolutions=resolutions, custom_limits=custom_limits,
                                #  diagnosis=DIAGNOSIS, **kwargs)
        for conf in path:
            set_joint_positions(robot, joints, conf)
            wait_if_gui()

        if write:
            save_path = os.path.join(RESULTS_DIRECTORY, '{}_test_traj.json'.format(ROBOT_NAME))
            with open(save_path, 'w') as f:
                data = {'robot':ROBOT_NAME, 'traj' : [list(conf) for conf in path]}
                json.dump(data, f)
                cprint('Result saved to: {}'.format(save_path), 'green')
    wait_if_gui(colored('Done.', 'green'))
