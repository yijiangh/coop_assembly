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
from coop_assembly.planning.regression import regression
from coop_assembly.planning.motion import display_trajectories
from coop_assembly.planning.parsing import load_structure, PICKNPLACE_FILENAMES
from coop_assembly.planning.validator import validate_trajectories, validate_pddl_plan
from coop_assembly.planning.utils import recover_sequence, Command
from coop_assembly.planning.stripstream import get_pddlstream, solve_pddlstream, STRIPSTREAM_ALGORITHM

########################################

def run_pddlstream(viewer, file_spec, collision, bar_only, write, algorithm, watch, debug=False, step_sim=False):
    bar_struct, o_struct = load_structure(file_spec, viewer, apply_alpha(RED, 0))
    fixed_obstacles, robot = load_world()

    robots = [robot]

    saver = WorldSaver()
    element_from_index = bar_struct.get_element_from_index()
    grounded_elements = bar_struct.get_grounded_bar_keys()
    contact_from_connectors = bar_struct.get_connectors(scale=METER_SCALE)
    connectors = list(contact_from_connectors.keys())

    plan = solve_pddlstream(robots, fixed_obstacles, element_from_index, grounded_elements, connectors, \
        collisions=collision, bar_only=bar_only, algorithm=algorithm, debug=debug)

    if plan is None:
        cprint('No plan found.', 'red')
        assert False, 'No plan found.'
    else:
        # TODO use acyclic graph to retract the correct order, the adaptive algorithm works well
        # TODO split into a function for reorganizing the actions
        commands = []
        place_actions = [action for action in reversed(plan) if action.name == 'place']
        for pc in place_actions:
            print_command = pc.args[-1]
            robot_name = pc.args[0]
            for action in plan:
                if action.name == 'move' and action.args[0] == robot_name and \
                    norm(action.args[1].positions-print_command.start_conf)<1e-8:
                    # norm(action.args[2].positions-print_command.start_conf)<1e-8:
                    commands.append(action.args[-1])
                    break
            commands.append(print_command)
        trajectories = flatten_commands(commands)

        # for t in trajectories:
        #     print(t.tag)
        #     print('st: {} \nend: {}'.format(t.start_conf, t.end_conf))
        #     print('====')

        if write:
            here = os.path.dirname(__file__)
            plan_path = '{}_pddl_solution_{}.json'.format(file_spec, get_date())
            save_path = os.path.join(here, 'results', plan_path)
            with open(save_path, 'w') as f:
               json.dump({'problem' : file_spec,
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
                time_step = 0.01 if bar_only else 0.05
            display_trajectories(trajectories, time_step=time_step)
        if collision:
            valid = validate_pddl_plan(trajectories, bar_struct, fixed_obstacles, watch=False, allow_failure=has_gui() or debug, \
                bar_only=bar_only, refine_num=1, debug=debug)
            cprint('Valid: {}'.format(valid), 'green' if valid else 'red')
            assert valid
    reset_simulation()
    disconnect()


##################################################

def create_parser():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', default='focused', choices=STRIPSTREAM_ALGORITHM, help='Stripstream algorithms')
    parser.add_argument('-c', '--collision', action='store_false', help='disable collision checking')
    parser.add_argument('-b', '--bar_only', action='store_true', help='only planning motion for floating bars')
    parser.add_argument('-w', '--watch', action='store_true', help='watch trajectories')
    parser.add_argument('-sm', '--step_sim', action='store_true', help='stepping simulation.')
    parser.add_argument('-wr', '--write', action='store_true', help='Export results')
    parser.add_argument('-db', '--debug', action='store_true', help='Debug verbose mode')
    parser.add_argument('-n', '--n_trails', default=1, help='number of trails')
    # parser.add_argument('-s', '--stiffness', action='store_false', help='disable stiffness')
    # parser.add_argument('-m', '--motion', action='store_true', help='enable transit motion')
    # parser.add_argument('--rfn', help='result file name')
    # parser.add_argument('--revisit', action='store_true', help='revisit in regression')

    return parser

def main():
    parser = create_parser()
    parser.add_argument('-p', '--problem', default='single_tet', choices=PICKNPLACE_FILENAMES, help='The name of the problem to solve')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    args = parser.parse_args()
    print('Arguments:', args)

    success_cnt = 0
    for i in range(int(args.n_trails)):
        try:
            run_pddlstream(args.viewer, args.problem, args.collision, args.bar_only, args.write, args.algorithm, args.watch, \
                debug=args.debug, step_sim=args.step_sim)
        except:
            cprint('#{}: plan not found'.format(i), 'red')
            continue
        cprint('#{}: plan found'.format(i), 'green')
        success_cnt += 1

    print('#'*10)
    print('{} : {} / {}'.format(args.problem, success_cnt, args.n_trails))

if __name__ == '__main__':
    main()
