from __future__ import print_function

import os
import argparse
import pytest
import numpy as np
import time
from numpy.linalg import norm
import json
from termcolor import cprint
from itertools import islice
from collections import defaultdict
import copy

from pybullet_planning import wait_for_user, connect, has_gui, wait_for_user, LockRenderer, remove_handles, add_line, \
    draw_pose, EndEffector, unit_pose, link_from_name, end_effector_from_body, get_link_pose, \
    dump_world, set_pose, WorldSaver, reset_simulation, disconnect, get_pose, RED, GREEN, refine_path, joints_from_names, \
    set_joint_positions, create_attachment, wait_if_gui, apply_alpha, set_color, get_relative_pose, create_shape, get_mesh_geometry, \
    create_flying_body, SE3, YELLOW, get_movable_joints, Attachment, Pose, invert, multiply, Euler, BLUE, elapsed_time

from coop_assembly.data_structure import BarStructure, OverallStructure, MotionTrajectory
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions import contact_to_ground
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE

from coop_assembly.planning import get_picknplace_robot_data, TOOL_LINK_NAME, EE_LINK_NAME, get_gripper_mesh_path
from coop_assembly.planning.visualization import color_structure, draw_ordered, draw_element, label_elements, label_connector, \
    display_trajectories, check_model, set_camera, visualize_stiffness, GROUND_COLOR
from coop_assembly.planning.utils import get_element_neighbors, get_connector_from_elements, check_connected, get_connected_structures, \
    flatten_commands

from coop_assembly.planning.stream import get_bar_grasp_gen_fn, get_place_gen_fn, get_pregrasp_gen_fn, command_collision, \
    get_element_body_in_goal_pose
from coop_assembly.planning.parsing import load_structure, PICKNPLACE_FILENAMES, save_plan, parse_plan, unpack_structure, Config, RESULTS_DIRECTORY
from coop_assembly.planning.validator import validate_trajectories, validate_pddl_plan, compute_plan_deformation
from coop_assembly.planning.utils import recover_sequence, Command, load_world, notify
from coop_assembly.planning.stripstream import get_pddlstream, solve_pddlstream, STRIPSTREAM_ALGORITHM, compute_orders
from coop_assembly.planning.regression import regression
from coop_assembly.planning.stiffness import create_stiffness_checker, evaluate_stiffness
from coop_assembly.planning.heuristics import HEURISTICS
from coop_assembly.planning.robot_setup import ROBOT_NAME, BUILD_PLATE_CENTER, BASE_YAW, BOTTOM_BUFFER, CONTROL_JOINT_NAMES, INITIAL_CONF
from coop_assembly.planning.motion import compute_motion

##################################################
def inspect_plan(args):
    if args.saved_plan is None:
        print('No saved plan file given.')
        return
    assert not args.bar_only, 'not implemented.'

    bar_struct, o_struct = load_structure(args.problem, args.viewer, apply_alpha(RED, 0))
    bar_radius = bar_struct.node[0]['radius']*METER_SCALE
    # transform model
    new_world_from_base = Pose(point=(BUILD_PLATE_CENTER + np.array([0,0,bar_radius+BOTTOM_BUFFER])))
    world_from_base = Pose(point=bar_struct.base_centroid(METER_SCALE))
    rotation = Pose(euler=Euler(yaw=BASE_YAW))
    tf = multiply(new_world_from_base, rotation, invert(world_from_base))
    bar_struct.transform(tf, scale=METER_SCALE)
    #
    bar_struct.generate_grounded_connection()

    fixed_obstacles, robot = load_world(use_floor=True, built_plate_z=BUILD_PLATE_CENTER[2])
    # robots = [end_effector] if args.bar_only else [robot]
    robots = [robot]
    joints = joints_from_names(robot, CONTROL_JOINT_NAMES)
    tool_from_ee = get_relative_pose(robot, link_from_name(robot, EE_LINK_NAME), link_from_name(robot, TOOL_LINK_NAME))

    # TODO chosen bars
    # chosen_bars = [int(b) for b in args.subset_bars] if args.subset_bars is not None else None
    chosen_bars = None
    element_from_index, grounded_elements, contact_from_connectors, connectors = \
        unpack_structure(bar_struct, chosen_bars=chosen_bars, scale=METER_SCALE, color=apply_alpha(RED,0.2))
    # color grounded elements
    for grounded_e in grounded_elements:
        try:
            set_color(element_from_index[grounded_e].body, apply_alpha(BLUE,0.2))
        except KeyError:
            pass
    # if args.subset_bars is not None:
    #     label_elements({k : element_from_index[k] for k in chosen_bars})

    parsed_data = parse_plan(args.saved_plan)
    e_trajs = parsed_data['plan']
    commands = []
    for bar_trajs in e_trajs:
        command_trajs = []
        if len(bar_trajs) == 5:
            del bar_trajs[0]
        for tdata in bar_trajs:
            attachments = []
            e_id = tdata['element']
            for at_data in tdata['attachments']:
                # attachment = jsonpickle.decode(at_data)
                attachment = Attachment.from_data(at_data,
                    parent=robot, child=element_from_index[e_id].body)
                attachments.append(attachment)
            traj = MotionTrajectory.from_data(tdata, robot, joints, attachments=attachments)
            command_trajs.append(traj)

        assert len(command_trajs) == 4, 'length {}'.format(len(command_trajs))
        commands.append(Command(command_trajs))

    old_trajectories = flatten_commands(commands)
    # sequence is kept anyways
    element_sequence = recover_sequence(old_trajectories, element_from_index)

    initial_conf = INITIAL_CONF
    start_time = time.time()
    printed_elements = []
    new_commands = []
    if args.replan_place or args.replan_motion:
        place_gen_fn = get_place_gen_fn(robot, tool_from_ee, element_from_index, fixed_obstacles, collisions=args.collisions, verbose=False,
            bar_only=args.bar_only, precompute_collisions=False, allow_failure=True, teleops=args.teleops)

        for i, command in enumerate(commands):
            element = element_sequence[i]
            attachments = []

            if args.replan_place:
                command, = next(place_gen_fn(element, printed=printed_elements))
                if command is None:
                    cprint('#{} Place planning failure.'.format(i), 'red')
                    # continue
                    return

            traj_from_tag = {traj.tag : traj for traj in command.trajectories}
            approach_attachments = traj_from_tag['place_approach'].attachments

            if args.replan_place or args.replan_motion:
                transfer_traj = compute_motion(robot, fixed_obstacles, element_from_index,
                                                printed_elements + [element], initial_conf, traj_from_tag['place_approach'].start_conf,
                                                attachments=approach_attachments,
                                                collisions=args.collisions, debug=args.debug)
                if transfer_traj is None:
                    cprint('#{} transfer planning failure.'.format(i), 'red')
                    return
                transfer_traj.tag = 'transfer'
                traj_from_tag['transfer'] = transfer_traj

                transit_traj = compute_motion(robot, fixed_obstacles, element_from_index,
                                             printed_elements, traj_from_tag['place_retreat'].end_conf, initial_conf,
                                             attachments=[],
                                             collisions=args.collisions, debug=args.debug)

                if transit_traj is None:
                    cprint('#{} transit planning failure.'.format(i), 'red')
                    return
                transit_traj.tag = 'transit'
                traj_from_tag['transit'] = transit_traj

            command.trajectories = [traj_from_tag['transfer'], traj_from_tag['place_approach'], traj_from_tag['place_retreat'], traj_from_tag['transit']]
            print('{}) {} | Time: {:.3f}'.format(i, command, elapsed_time(start_time)))

            printed_elements.append(element)
            new_commands.append(command)

    new_trajectories = flatten_commands(new_commands)

    if args.write:
        save_link_names = [TOOL_LINK_NAME]
        recomputed_plan_path = args.saved_plan.split('.json')[0] + \
            '_replan_rPlace{}_rTrans{}.json'.format(int(args.replan_place), int(args.replan_motion))
        save_path = os.path.join(RESULTS_DIRECTORY, recomputed_plan_path)
        recompute_data = parsed_data
        del recompute_data['plan']

        recompute_data['plan'] = []
        e_path = []
        e_id = new_trajectories[0].element
        for traj in new_trajectories:
            # print(traj)
            if traj.element is not None and e_id != traj.element:
                # break subprocess if there is a discontinuity in the element id
                recompute_data['plan'].append(copy.deepcopy(e_path))
                e_path = []
                e_id = traj.element
            tdata = traj.to_data()
            if save_link_names is not None:
                link_path_data = {link_name : traj.get_link_path(link_name) for link_name in save_link_names}
                tdata.update({'link_path' : link_path_data})
            e_path.append(tdata)
        else:
            recompute_data['plan'].append(e_path)

        with open(save_path, 'w') as f:
            json.dump(recompute_data, f)
            cprint('Recomputed result saved to: {}'.format(os.path.abspath(save_path)), 'green')

    if args.watch and has_gui():
        #label_nodes(node_points)
        elements = recover_sequence(new_trajectories, element_from_index)
        endpts_from_element = bar_struct.get_axis_pts_from_element()
        draw_ordered(elements, endpts_from_element)
        for e in element_from_index:
           set_color(element_from_index[e].body, (1, 0, 0, 0))
        if args.step_sim:
            time_step = None
        else:
            time_step = 0.01 if args.bar_only else 0.05
        display_trajectories(new_trajectories, time_step=time_step, element_from_index=element_from_index)
    # verify
    if args.collisions:
        valid = validate_pddl_plan(new_trajectories, fixed_obstacles, element_from_index, grounded_elements, watch=False, allow_failure=has_gui() or args.debug, \
            bar_only=args.bar_only, refine_num=1, debug=args.debug)
        cprint('Valid: {}'.format(valid), 'green' if valid else 'red')
        assert valid
    else:
        cprint('Collision disabled, no verfication performed.', 'yellow')
    reset_simulation()
    disconnect()

##################################################
# 1_exp_tets_regression-z_solution_20-12-22_14-21-01.json

def create_parser():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='single_tet',
                        help='The name of the problem to solve')
    parser.add_argument('-c', '--collisions', action='store_false',
                        help='Disable collision checking with obstacles')
    #
    parser.add_argument('--replan_place', action='store_true',
                        help='Recompute placing paths.')
    parser.add_argument('--replan_motion', action='store_true',
                        help='Recompute transit/transfer paths.')
    #
    parser.add_argument('-b', '--bar_only', action='store_true',
                        help='Only planning motion for floating bars, diable arm planning')
    parser.add_argument('-to', '--teleops', action='store_true',
                        help='Use teleop for trajectories (turn off in-between traj planning)')
    # parser.add_argument('--subset_bars', nargs='+', default=None,
    #                     help='Plan for only subset of bar indices.')

    parser.add_argument('--saved_plan', default=None, help='Parse a saved plan.')

    return parser

def main():
    parser = create_parser()
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    parser.add_argument('-w', '--watch', action='store_true', help='watch trajectories')
    parser.add_argument('-sm', '--step_sim', action='store_true', help='stepping simulation.')
    parser.add_argument('-wr', '--write', action='store_true', help='Export results')
    parser.add_argument('-db', '--debug', action='store_true', help='Debug verbose mode')
    # parser.add_argument('--check_model', action='store_true', help='Inspect model.')
    args = parser.parse_args()
    print('Arguments:', args)

    inspect_plan(args)

if __name__ == '__main__':
    main()
