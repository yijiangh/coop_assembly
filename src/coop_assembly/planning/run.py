#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
import pytest
import numpy as np
from numpy.linalg import norm
import json
from termcolor import cprint
from itertools import islice
from collections import defaultdict

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
from coop_assembly.planning.visualization import color_structure, draw_ordered, draw_element, label_elements, label_connector, set_camera
from coop_assembly.planning.utils import get_element_neighbors, get_connector_from_elements, check_connected, get_connected_structures, \
    flatten_commands

from coop_assembly.planning.stream import get_goal_pose_gen_fn, get_bar_grasp_gen_fn, get_place_gen_fn, get_pregrasp_gen_fn, command_collision, \
    get_element_body_in_goal_pose
from coop_assembly.planning.motion import display_trajectories
from coop_assembly.planning.parsing import load_structure, PICKNPLACE_FILENAMES
from coop_assembly.planning.validator import validate_trajectories, validate_pddl_plan
from coop_assembly.planning.utils import recover_sequence, Command
from coop_assembly.planning.stripstream import get_pddlstream, solve_pddlstream, STRIPSTREAM_ALGORITHM, compute_orders
from coop_assembly.planning.regression import regression

ALGORITHMS = STRIPSTREAM_ALGORITHM + ['regression']

########################################
# two_tets
# po, adaptive: 6 s
# po, focused: 24 s
# po, binding: 34 s

def run_pddlstream(args, viewer=False, watch=False, debug=False, step_sim=False, write=False):
    bar_struct, o_struct = load_structure(args.problem, viewer, apply_alpha(RED, 0))
    fixed_obstacles, robot = load_world()

    robots = [robot]

    saver = WorldSaver()
    element_from_index = bar_struct.get_element_from_index()
    grounded_elements = bar_struct.get_grounded_bar_keys()
    contact_from_connectors = bar_struct.get_connectors(scale=METER_SCALE)
    connectors = list(contact_from_connectors.keys())

    elements_from_layer = defaultdict(set)
    if args.partial_ordering:
        for v in bar_struct.nodes():
            elements_from_layer[bar_struct.node[v]['layer']].add(v)
        partial_orders = compute_orders(elements_from_layer)
    else:
        partial_orders = []
    print('Partial orders: ', partial_orders)
    # input("Enter to proceed.")

    if args.algorithm in STRIPSTREAM_ALGORITHM:
        plan = solve_pddlstream(robots, fixed_obstacles, element_from_index, grounded_elements, connectors, partial_orders=partial_orders,
            collisions=args.collisions, bar_only=args.bar_only, algorithm=args.algorithm, costs=args.costs,
            debug=debug, teleops=args.teleops)
    elif args.algorithm == 'regression':
        with LockRenderer(True):
            plan, data = regression(robot, fixed_obstacles, element_from_index, grounded_elements, connectors, collision=args.collisions,
                motions=True, stiffness=True,
                revisit=False, verbose=True, lazy=False, bar_only=args.bar_only, partial_orders=partial_orders)
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
            here = os.path.dirname(__file__)
            plan_path = '{}_{}_solution_{}.json'.format(args.file_spec, args.algorithm, get_date())
            save_path = os.path.join(here, 'results', plan_path)
            with open(save_path, 'w') as f:
               json.dump({'problem' : args.file_spec,
                          'plan' : [p.to_data() for p in trajectories]}, f)
            cprint('Result saved to: {}'.format(save_path), 'green')
        if watch and has_gui():
            saver.restore()
            #label_nodes(node_points)
            elements = recover_sequence(trajectories, element_from_index)
            endpts_from_element = bar_struct.get_axis_pts_from_element()
            draw_ordered(elements, endpts_from_element)
            wait_if_gui('Ready to simulate trajectory.')
            for e in element_from_index:
               set_color(element_from_index[e].body, (1, 0, 0, 0))
            if step_sim:
                time_step = None
            else:
                time_step = 0.01 if args.bar_only else 0.05
            display_trajectories(trajectories, time_step=time_step)
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
    parser.add_argument('-a', '--algorithm', default='focused', choices=ALGORITHMS, help='Stripstream algorithms')
    parser.add_argument('-c', '--collisions', action='store_false', help='disable collision checking')
    parser.add_argument('-b', '--bar_only', action='store_true', help='only planning motion for floating bars')
    parser.add_argument('-po', '--partial_ordering', action='store_true', help='use partial ordering (if-any)')
    parser.add_argument('-co', '--costs', action='store_true', help='turn on cost_effective planning')
    parser.add_argument('-to', '--teleops', action='store_true', help='use teleop for trajectories (turn off in-between traj planning)')

    # parser.add_argument('-s', '--stiffness', action='store_false', help='disable stiffness')
    # parser.add_argument('-m', '--motion', action='store_true', help='enable transit motion')
    # parser.add_argument('--rfn', help='result file name')
    # parser.add_argument('--revisit', action='store_true', help='revisit in regression')

    return parser

def main():
    parser = create_parser()
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    parser.add_argument('-n', '--n_trails', default=1, help='number of trails')
    parser.add_argument('-w', '--watch', action='store_true', help='watch trajectories')
    parser.add_argument('-sm', '--step_sim', action='store_true', help='stepping simulation.')
    parser.add_argument('-wr', '--write', action='store_true', help='Export results')
    parser.add_argument('-db', '--debug', action='store_true', help='Debug verbose mode')
    args = parser.parse_args()
    print('Arguments:', args)

    success_cnt = 0
    if int(args.n_trails) == 1:
        # one-shot run, expose assertion errors
        run_pddlstream(args, viewer=args.viewer, watch=args.watch, debug=args.debug, step_sim=args.step_sim, write=args.write)
    else:
        for i in range(int(args.n_trails)):
            try:
                run_pddlstream(args, viewer=args.viewer, watch=args.watch, debug=args.debug, step_sim=args.step_sim, write=args.write)
            except:
                cprint('#{}: plan not found'.format(i), 'red')
                continue
            cprint('#{}: plan found'.format(i), 'green')
            success_cnt += 1
        print('#'*10)
        print('{} : {} / {}'.format(args.problem, success_cnt, args.n_trails))

if __name__ == '__main__':
    main()
