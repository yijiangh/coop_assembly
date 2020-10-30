#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
import pytest
import numpy as np
from numpy.linalg import norm
import json
import jsonpickle
from termcolor import cprint
from itertools import islice
from collections import defaultdict

from pybullet_planning import wait_for_user, connect, has_gui, wait_for_user, LockRenderer, remove_handles, add_line, \
    draw_pose, EndEffector, unit_pose, link_from_name, end_effector_from_body, get_link_pose, \
    dump_world, set_pose, WorldSaver, reset_simulation, disconnect, get_pose, RED, GREEN, refine_path, joints_from_names, \
    set_joint_positions, create_attachment, wait_if_gui, apply_alpha, set_color, get_relative_pose, create_shape, get_mesh_geometry, \
    create_flying_body, SE3, YELLOW, get_movable_joints, Attachment

from coop_assembly.data_structure import BarStructure, OverallStructure, MotionTrajectory
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions import contact_to_ground
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE

from coop_assembly.planning import get_picknplace_robot_data, BUILT_PLATE_Z, TOOL_LINK_NAME, EE_LINK_NAME, IK_JOINT_NAMES, get_gripper_mesh_path
from coop_assembly.planning.visualization import color_structure, draw_ordered, draw_element, label_elements, label_connector, \
    display_trajectories, check_model, set_camera, visualize_stiffness
from coop_assembly.planning.utils import get_element_neighbors, get_connector_from_elements, check_connected, get_connected_structures, \
    flatten_commands

from coop_assembly.planning.stream import get_bar_grasp_gen_fn, get_place_gen_fn, get_pregrasp_gen_fn, command_collision, \
    get_element_body_in_goal_pose
from coop_assembly.planning.parsing import load_structure, PICKNPLACE_FILENAMES, save_plan, parse_plan, unpack_structure
from coop_assembly.planning.validator import validate_trajectories, validate_pddl_plan
from coop_assembly.planning.utils import recover_sequence, Command, load_world
from coop_assembly.planning.stripstream import get_pddlstream, solve_pddlstream, STRIPSTREAM_ALGORITHM, compute_orders
from coop_assembly.planning.regression import regression
from coop_assembly.planning.stiffness import create_stiffness_checker

ALGORITHMS = STRIPSTREAM_ALGORITHM + ['regression']

# BUILD_PLATE_CENTER = np.array([550, 0, -14.23])*1e-3
BUILD_PLATE_CENTER = np.array([650, 0, -14.23])*1e-3
# BOTTOM_BUFFER = 0.005
BOTTOM_BUFFER = 0.01
# BOTTOM_BUFFER = 0.1

########################################
# two_tets
# po, adaptive: 6 s
# po, focused: 24 s
# po, binding: 34 s

def run_planning(args, viewer=False, watch=False, debug=False, step_sim=False, write=False, saved_plan=None):
    bar_struct, o_struct = load_structure(args.problem, viewer, apply_alpha(RED, 0))
    # transform the bar struct to the center
    bar_radius = bar_struct.node[0]['radius']*METER_SCALE
    bar_struct.transform(BUILD_PLATE_CENTER + np.array([0,0,bar_radius+BOTTOM_BUFFER]), scale=METER_SCALE)
    bar_struct.generate_grounded_connection()

    fixed_obstacles, robot = load_world(built_plate_z=BUILD_PLATE_CENTER[2])
    tool_from_ee = get_relative_pose(robot, link_from_name(robot, EE_LINK_NAME), link_from_name(robot, TOOL_LINK_NAME))
    # end effector robot
    ee_mesh_path = get_gripper_mesh_path()
    # TODO offer option to turn off end effector collision by setting collision=False
    collision_id, visual_id = create_shape(get_mesh_geometry(ee_mesh_path, scale=1e-3), collision=True, color=apply_alpha(YELLOW, 0.5))
    end_effector = create_flying_body(SE3, collision_id, visual_id)

    # the arm itself
    robots = [end_effector] if args.bar_only else [robot]

    chosen_bars = [int(b) for b in args.subset_bars] if args.subset_bars is not None else None
    element_from_index, grounded_elements, contact_from_connectors, connectors = \
        unpack_structure(bar_struct, chosen_bars=chosen_bars, scale=METER_SCALE)

    bar_struct.set_body_color(RED, indices=chosen_bars)
    print('base: ', bar_struct.base_centroid(METER_SCALE))
    set_camera([bar_struct.base_centroid(METER_SCALE)], scale=1.)

    elements_from_layer = defaultdict(set)
    if args.partial_ordering:
        for v in bar_struct.nodes():
            elements_from_layer[bar_struct.node[v]['layer']].add(v)
        partial_orders = compute_orders(elements_from_layer)
    else:
        partial_orders = []
    print('Partial orders: ', partial_orders)

    if saved_plan is None:
        if args.check_model:
            check_model(bar_struct, chosen_bars, debug=args.debug)
        else:
            wait_if_gui('Please review structure\'s workspace position.')

        checker = None
        if args.stiffness and (checker is None):
            checker = create_stiffness_checker(bar_struct, verbose=False, debug=args.debug, save_model=args.save_cm_model)
            cprint('stiffness checker created.', 'green')

        # visualize_stiffness
        with WorldSaver():
            if args.algorithm in STRIPSTREAM_ALGORITHM:
                plan = solve_pddlstream(robots, tool_from_ee, fixed_obstacles, element_from_index, grounded_elements, connectors,
                    partial_orders=partial_orders,
                    collisions=args.collisions, bar_only=args.bar_only, algorithm=args.algorithm, costs=args.costs,
                    debug=debug, teleops=args.teleops)
            elif args.algorithm == 'regression':
                with LockRenderer(not args.debug):
                    plan, data = regression(end_effector if args.bar_only else robot, tool_from_ee, fixed_obstacles, element_from_index,
                        grounded_elements, connectors, collision=args.collisions,
                        motions=True, stiffness=True, revisit=False, verbose=debug,
                        lazy=False, bar_only=args.bar_only, partial_orders=partial_orders)
            else:
                raise NotImplementedError('Algorithm |{}| not in {}'.format(args.algorithm, ALGORITHMS))
        if plan is None:
            cprint('No plan found.', 'red')
            assert False, 'No plan found.'
        else:
            cprint('plan found.', 'green')
            if args.algorithm in STRIPSTREAM_ALGORITHM:
                commands = []
                place_actions = [action for action in reversed(plan) if action.name == 'place']
                start_conf_id = 1 if args.algorithm == 'incremental_sa' else 2
                for pc in place_actions:
                    print_command = pc.args[-1]
                    robot_name = pc.args[0]
                    for action in plan:
                        if action.name == 'move' and action.args[0] == robot_name and \
                            norm(action.args[start_conf_id].positions-print_command.start_conf)<1e-8:
                            commands.append(action.args[-1])
                            break
                    commands.append(print_command)
                trajectories = flatten_commands(commands)
            else:
                # regression plan is flattened already
                trajectories = plan
            if write:
                save_plan(args.problem, args.algorithm, trajectories, TCP_link_name=TOOL_LINK_NAME if not args.bar_only else None,
                    element_from_index=element_from_index, suffix='bar-only' if args.bar_only else None)
    else:
        # parse a saved plan
        robot = end_effector if "bar-only" in saved_plan else robot
        e_trajs = parse_plan(saved_plan)
        trajectories = []
        joints = get_movable_joints(robot)
        for trajs in e_trajs:
            for tdata in trajs:
                attachments = []
                e_id = tdata['element']
                for at_data in tdata['attachments']:
                    # attachment = jsonpickle.decode(at_data)
                    attachment = Attachment.from_data(at_data,
                        parent=robot, child=element_from_index[e_id].body)
                    attachments.append(attachment)
                traj = MotionTrajectory.from_data(tdata, robot, \
                    joints, attachments=attachments)
                trajectories.append(traj)

    if watch and has_gui():
        #label_nodes(node_points)
        elements = recover_sequence(trajectories, element_from_index)
        endpts_from_element = bar_struct.get_axis_pts_from_element()
        draw_ordered(elements, endpts_from_element)
        for e in element_from_index:
           set_color(element_from_index[e].body, (1, 0, 0, 0))
        if step_sim:
            time_step = None
        else:
            time_step = 0.01 if args.bar_only else 0.05
        display_trajectories(trajectories, time_step=time_step, element_from_index=element_from_index)
    # verify
    if args.collisions:
        valid = validate_pddl_plan(trajectories, fixed_obstacles, element_from_index, grounded_elements, watch=False, allow_failure=has_gui() or debug, \
            bar_only=args.bar_only, refine_num=1, debug=debug)
        cprint('Valid: {}'.format(valid), 'green' if valid else 'red')
        assert valid
    else:
        cprint('Collision disabled, no verfication performed.', 'yellow')
    reset_simulation()
    disconnect()


##################################################

def create_parser():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='single_tet', help='The name of the problem to solve')
    parser.add_argument('-a', '--algorithm', default='regression', choices=ALGORITHMS, help='Planning algorithms')
    parser.add_argument('-c', '--collisions', action='store_false', help='disable collision checking')
    parser.add_argument('-b', '--bar_only', action='store_true', help='only planning motion for floating bars')
    parser.add_argument('-po', '--partial_ordering', action='store_true', help='use partial ordering (if-any)')
    parser.add_argument('-co', '--costs', action='store_true', help='turn on cost_effective planning')
    parser.add_argument('-to', '--teleops', action='store_true', help='use teleop for trajectories (turn off in-between traj planning)')

    # parser.add_argument('-s', '--stiffness', action='store_false', help='disable stiffness')
    # parser.add_argument('-m', '--motion', action='store_true', help='enable transit motion')
    # parser.add_argument('--rfn', help='result file name')
    # parser.add_argument('--revisit', action='store_true', help='revisit in regression')

    parser.add_argument('--saved_plan', default=None, help='Parse a saved plan.')

    return parser

def main():
    parser = create_parser()
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    parser.add_argument('-n', '--n_trails', default=1, help='number of trails')
    parser.add_argument('-w', '--watch', action='store_true', help='watch trajectories')
    parser.add_argument('-sm', '--step_sim', action='store_true', help='stepping simulation.')
    parser.add_argument('-wr', '--write', action='store_true', help='Export results')
    parser.add_argument('-db', '--debug', action='store_true', help='Debug verbose mode')
    parser.add_argument('--stiffness', action='store_true', help='Turn on stiffness checking.')
    parser.add_argument('--check_model', action='store_true', help='Inspect model.')
    parser.add_argument('--save_cm_model', action='store_true', help='Export conmech model.')
    parser.add_argument('--subset_bars', nargs='+', default=None, help='Plan for only subset of bar indices.')
    args = parser.parse_args()
    print('Arguments:', args)

    success_cnt = 0
    if int(args.n_trails) == 1:
        # one-shot run, expose assertion errors
        run_planning(args, viewer=args.viewer, watch=args.watch, debug=args.debug, step_sim=args.step_sim, write=args.write, saved_plan=args.saved_plan)
    else:
        for i in range(int(args.n_trails)):
            try:
                run_planning(args, viewer=args.viewer, watch=args.watch, debug=args.debug, step_sim=args.step_sim, write=args.write, saved_plan=args.saved_plan)
            except:
                cprint('#{}: plan not found'.format(i), 'red')
                continue
            cprint('#{}: plan found'.format(i), 'green')
            success_cnt += 1
        print('#'*10)
        print('{} : {} / {}'.format(args.problem, success_cnt, args.n_trails))

if __name__ == '__main__':
    main()
