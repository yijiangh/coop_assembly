import os
import json
import random
import numpy as np
import pytest
from itertools import chain

from compas_fab.backends.pybullet import convert_mesh_to_pybullet_body
from compas_fab.assembly.datastructures import Assembly, UnitGeometry, Grasp

from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose, remove_handles
from pybullet_planning import multiply, unit_pose, get_relative_pose, invert
from pybullet_planning import interval_generator, sample_tool_ik
from pybullet_planning import Pose, Point, Euler, unit_pose, Attachment
from pybullet_planning import joints_from_names, link_from_name, has_link, get_collision_fn, get_disabled_collisions, \
    draw_pose, set_pose, set_joint_positions, dump_world, create_obj, get_body_body_disabled_collisions, get_link_pose, \
    create_attachment, set_pose, get_pose, link_from_name, BASE_LINK

from pychoreo.process_model.cartesian_process import CartesianProcess
from pychoreo.process_model.trajectory import Trajectory, MotionTrajectory
from pychoreo.utils.stream_utils import get_random_direction_generator, get_enumeration_pose_generator
from pychoreo.utils.parsing_utils import export_trajectory
from pychoreo.cartesian_planner.ladder_graph_interface import solve_ladder_graph_from_cartesian_process_list
from pychoreo.cartesian_planner.sparse_ladder_graph import SparseLadderGraph

from pychoreo_examples.picknplace.stream import build_picknplace_cartesian_process_seq
from pychoreo_examples.picknplace.trajectory import PicknPlaceBufferTrajectory
from pychoreo_examples.picknplace.visualization import display_picknplace_trajectories
from pychoreo_examples.picknplace.parsing import parse_saved_trajectory
from pychoreo_examples.picknplace.transition_planner import solve_transition_between_picknplace_processes

from coop_assembly.choreo.robot_setup import IK_MODULE
from coop_assembly.choreo.robot_setup import get_picknplace_robot_data, get_picknplace_end_effector_urdf, \
    get_picknplace_tcp_def, get_robot_init_conf

AVAIBLE_SOLVE_METHODS = ['ladder_graph', 'sparse_ladder_graph']

def sequenced_picknplace_plan(assembly_pkg_json_path, solve_method='sparse_ladder_graph', viewer=False, scale=1e-3,
    sample_time=5, sparse_time_out=2, jt_res=0.1, pos_step_size=0.01, ori_step_size=np.pi/16,
    viz_inspect=False, warning_pause=False, save_dir=None, **kwargs):
    """call pychoreo's planner to solve for picknplace Cartesian & transition trajectories.

    Robot setup is specified in coop_assembly.choreo_interface.robot_setup

    Parameters
    ----------
    assembly_pkg_json_path : [type]
        [description]
    solve_method : str, optional
        [description], by default 'sparse_ladder_graph'
    viewer : bool, optional
        [description], by default False
    scale : [type], optional
        [description], by default 1e-3
    sample_time : int, optional
        [description], by default 5
    sparse_time_out : int, optional
        [description], by default 2
    jt_res : float, optional
        joint resolution for transition planning, by default 0.1
    pos_step_size : float, optional
        interpolation step size for the end effector Cartesia movements, by default 0.01 (meter)
    ori_step_size : [type], optional
        [description], by default np.pi/16
    viz_inspect : bool, optional
        visualize planning process in a pybullet window (slow!), by default False
    warning_pause : bool, optional
        wait for user input if any of the planning process cannot find a solution, by default False
    save_dir : string, optional
        absolute directory path to save the planning result json file, by default None

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """

    # * load robot setup data
    (robot_urdf, base_link_name, tool_root_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
    (workspace_urdf, workspace_robot_disabled_link_names) = get_picknplace_robot_data()
    picknplace_end_effector_urdf = get_picknplace_end_effector_urdf()
    picknplace_tcp_def = get_picknplace_tcp_def()

    # * create robot and pb environment
    connect(use_gui=viewer)

    # * adjust camera pose (optional)
    if has_gui():
        camera_base_pt = (0,0,0)
        camera_pt = np.array(camera_base_pt) + np.array([1, 0, 0.5])
        set_camera_pose(tuple(camera_pt), camera_base_pt)

    with HideOutput():
        # * pybullet can handle ROS-package path URDF automatically now (ver 2.5.7)!
        robot = load_pybullet(robot_urdf, fixed_base=True)
        workspace = load_pybullet(workspace_urdf, fixed_base=True)
        # ee_body = create_obj(picknplace_end_effector_urdf)
        ee_body = load_pybullet(picknplace_end_effector_urdf)

    # * set robot idle configuration
    ik_joints = joints_from_names(robot, ik_joint_names)
    robot_start_conf = get_robot_init_conf()
    set_joint_positions(robot, ik_joints, robot_start_conf)

    # * create tool and tool TCP from flange (tool0) transformation
    root_link = link_from_name(robot, tool_root_link_name)
    # create end effector body
    ee_attach = Attachment(robot, root_link, unit_pose(), ee_body)
    # set up TCP transformation, just a renaming here
    root_from_tcp = picknplace_tcp_def
    if has_gui() :
        # draw_tcp pose
        ee_attach.assign()
        ee_link_pose = get_pose(ee_attach.child)
        draw_pose(multiply(ee_link_pose, root_from_tcp))

    # * specify ik fn wrapper
    ik_fn = IK_MODULE.get_ik
    def get_sample_ik_fn(robot, ik_fn, ik_joint_names, base_link_name, tool_from_root=None):
        def sample_ik_fn(world_from_tcp):
            if tool_from_root:
                world_from_tcp = multiply(world_from_tcp, tool_from_root)
            return sample_tool_ik(ik_fn, robot, ik_joint_names, base_link_name, world_from_tcp, get_all=True) #,sampled=[0])
        return sample_ik_fn
    # ik generation function stays the same for all cartesian processes
    sample_ik_fn = get_sample_ik_fn(robot, ik_fn, ik_joint_names, base_link_name, invert(root_from_tcp))

    # * load shape & collision data
    with open(assembly_pkg_json_path, 'r') as f:
        json_data = json.loads(f.read())
    assembly = Assembly.from_package(json_data)
    elements = assembly.elements
    for element in elements.values():
        for unit_geo in element.unit_geometries:
            unit_geo.rescale(scale)
    # TODO: scale derived from the assembly package unit
    # static_obstacles = []
    # for ug in assembly.static_obstacle_geometries.values():
    #     static_obstacles.extend(ug.mesh)

    # * load precomputed sequence / use assigned sequence
    # TODO: load this as function argument
    # element_seq = elements.keys()
    # TODO: sequence can come from contact analysis
    # TODO: make sure element key corresponds to bar_vkey
    # TODO: reformat picknplace instances in assembly_instances
    # which two elements are connecting
    # what's the contact normal:
    # E_i : 'E_j' : {n_i, p_i}
    element_seq = [0, 1, 2, 3, 4, 5]
    # element_seq = [3, 4, 5]
    print('sequence: ', element_seq)

    # visualize goal pose
    if has_gui():
        # viz_len = 0.003
        with WorldSaver():
            for e_id in element_seq:
                element = elements[e_id]
                with LockRenderer():
                    print('e_id #{} : {}'.format(e_id, element))
                    for unit_geo in element.unit_geometries:
                        for pb_geo in unit_geo.pybullet_bodies:
                            set_pose(pb_geo, random.choice(unit_geo.get_goal_frames(get_pb_pose=True)))
                # print('---------')
                # wait_for_user()
        # wait_for_user()

    # * construct ignored body-body links for collision checking
    # in this case, including self-collision between links of the robot
    disabled_self_collisions = get_disabled_collisions(robot, disabled_self_collision_link_names)
    # and links between the robot and the workspace (e.g. robot_base_link to base_plate)
    extra_disabled_collisions = get_body_body_disabled_collisions(robot, workspace, workspace_robot_disabled_link_names)
    # TODO: extra disabled collisions as function argument
    extra_disabled_collisions.update({
        ((robot, link_from_name(robot, 'robot_link_5')), (ee_body, link_from_name(ee_body, 'eef_base_link'))),
        ((robot, link_from_name(robot, 'robot_link_6')), (ee_body, link_from_name(ee_body, 'eef_base_link')))
        })

    # * create cartesian processes without a sequence being given, with random pose generators
    cart_process_seq = build_picknplace_cartesian_process_seq(
        element_seq, elements,
        robot, ik_joint_names, root_link, sample_ik_fn,
        ee_attachs=[ee_attach], self_collisions=True, disabled_collisions=disabled_self_collisions,
        obstacles=[workspace],extra_disabled_collisions=extra_disabled_collisions,
        tool_from_root=invert(root_from_tcp), viz_step=False, pick_from_same_rack=True,
        pos_step_size=pos_step_size, ori_step_size=ori_step_size)

    # specifically for UR5, because of its wide joint range, we need to apply joint value snapping
    for cp in cart_process_seq:
        cp.target_conf = robot_start_conf

    with LockRenderer(not viz_inspect):
        if solve_method == 'ladder_graph':
            print('\n'+'#' * 10)
            print('Solving with the vanilla ladder graph search algorithm.')
            cart_process_seq = solve_ladder_graph_from_cartesian_process_list(cart_process_seq,
                verbose=True, warning_pause=warning_pause, viz_inspect=viz_inspect, check_collision=False, start_conf=robot_start_conf)
        elif solve_method == 'sparse_ladder_graph':
            print('\n'+'#' * 10)
            print('Solving with the sparse ladder graph search algorithm.')
            sparse_graph = SparseLadderGraph(cart_process_seq)
            sparse_graph.find_sparse_path(verbose=True, vert_timeout=sample_time, sparse_sample_timeout=sparse_time_out)
            cart_process_seq = sparse_graph.extract_solution(verbose=True, start_conf=robot_start_conf)
        else:
            raise ValueError('Invalid solve method!')
        assert all(isinstance(cp, CartesianProcess) for cp in cart_process_seq)

        pnp_trajs = [[] for _ in range(len(cart_process_seq))]
        for cp_id, cp in enumerate(cart_process_seq):
            element_attachs = []
            for sp_id, sp in enumerate(cp.sub_process_list):
                assert sp.trajectory, '{}-{} does not have a Cartesian plan found!'.format(cp, sp)
                # ! reverse engineer the grasp pose
                if sp.trajectory.tag == 'pick_retreat':
                    unit_geo = elements[sp.trajectory.element_id].unit_geometries[0]
                    e_bodies = unit_geo.pybullet_bodies
                    for e_body in e_bodies:
                        set_pose(e_body, unit_geo.get_initial_frames(get_pb_pose=True)[0])
                        set_joint_positions(sp.trajectory.robot, sp.trajectory.joints, sp.trajectory.traj_path[0])
                        element_attachs.append(create_attachment(sp.trajectory.robot, root_link, e_body))

                if sp.trajectory.tag == 'pick_retreat' or sp.trajectory.tag == 'place_approach':
                    sp.trajectory.attachments= element_attachs
                pnp_trajs[cp_id].append(sp.trajectory)
        full_trajs = pnp_trajs

        # * transition motion planning between extrusions
        return2idle = True
        transition_traj = solve_transition_between_picknplace_processes(pnp_trajs, elements, robot_start_conf,
                                                                        disabled_collisions=disabled_self_collisions,
                                                                        extra_disabled_collisions=extra_disabled_collisions,
                                                                        obstacles=[workspace], return2idle=return2idle,
                                                                        resolutions=[jt_res]*len(ik_joints),
                                                                        **kwargs)

    # * weave the Cartesian and transition processses together
    for cp_id, print_trajs in enumerate(full_trajs):
        print_trajs.insert(0, transition_traj[cp_id][0])
        print_trajs.insert(3, transition_traj[cp_id][1])
    if return2idle:
        full_trajs[-1].append(transition_traj[-1][-1])

    if save_dir is None:
        here = os.path.dirname(__file__)
        save_dir = os.path.join(here, 'results')
    export_trajectory(save_dir, full_trajs, ee_link_name, indent=1, shape_file_path=assembly_pkg_json_path,
        include_robot_data=True, include_link_path=True)

    # * disconnect and close pybullet engine used for planning, visualizing trajectories will start a new one
    reset_simulation()
    disconnect()

    if viewer:
        cart_time_step = None
        tr_time_step = None
        display_picknplace_trajectories(robot_urdf, ik_joint_names,
                                        assembly_pkg_json_path, full_trajs, tool_root_link_name,
                                        ee_urdf=picknplace_end_effector_urdf, workspace_urdf=workspace_urdf, animate=True,
                                        cart_time_step=cart_time_step, tr_time_step=tr_time_step)

def visualize_picknplace_planning_results(pkg_json_path, save_file_path, viewer=False):
    # * create robot and pb environment
    (robot_urdf, base_link_name, tool_root_link_name, ee_link_name, ik_joint_names, disabled_self_collision_link_names), \
    (workspace_urdf, workspace_robot_disabled_link_names) = get_picknplace_robot_data()
    picknplace_end_effector_urdf = get_picknplace_end_effector_urdf()
    picknplace_tcp_def = get_picknplace_tcp_def()

    # parse with connect
    connect(use_gui=viewer)
    with HideOutput():
        _ = load_pybullet(robot_urdf, fixed_base=True)
    full_trajs = parse_saved_trajectory(save_file_path)
    disconnect()

    # visualize plan
    if viewer:
        cart_time_step = None
        tr_time_step = None
        display_picknplace_trajectories(robot_urdf, ik_joint_names,
                                        pkg_json_path, full_trajs, tool_root_link_name,
                                        ee_urdf=picknplace_end_effector_urdf, workspace_urdf=workspace_urdf, animate=True,
                                        cart_time_step=cart_time_step, tr_time_step=tr_time_step)
