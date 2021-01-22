#!/usr/bin/env python
from __future__ import print_function

import argparse
import os, sys
import json
from termcolor import cprint
import numpy as np
from numpy.linalg import norm
import math

HERE = os.path.dirname(os.path.abspath(__file__))

from coop_assembly.data_structure import WorldPose, MotionTrajectory
from coop_assembly.planning.utils import get_connector_from_elements, check_connected, \
    flatten_commands
from coop_assembly.planning.validator import validate_trajectories, validate_pddl_plan
from coop_assembly.planning.visualization import set_camera, label_points, display_trajectories

from pybullet_planning import set_camera_pose, connect, create_box, wait_if_gui, set_pose, create_plane, \
    draw_pose, unit_pose, set_camera_pose2, Pose, Point, Euler, RED, BLUE, GREEN, CLIENT, HideOutput, create_obj, apply_alpha, \
    create_flying_body, create_shape, get_mesh_geometry, get_movable_joints, get_configuration, set_configuration, get_links, \
    has_gui, set_color, reset_simulation, disconnect, get_date, WorldSaver, LockRenderer, YELLOW, add_line, draw_circle, pairwise_collision, \
    body_collision_info, get_distance, draw_collision_diagnosis, get_aabb, BodySaver, multiply, invert

from .stream import get_element_body_in_goal_pose, get_2d_place_gen_fn, pose_from_xz_values, xz_values_from_pose
from .robot_setup import load_2d_world, Conf, INITIAL_CONF
from .stripstream import STRIPSTREAM_ALGORITHM, solve_pddlstream
from .regression import regression
from .parsing import parse_2D_truss

ALGORITHMS = STRIPSTREAM_ALGORITHM + ['regression']

##################################################

def run_planning(args, viewer=False, watch=False, debug=False, step_sim=False, write=False):
    end_effector, floor, tool_from_ee = load_2d_world(viewer=args.viewer)
    # element_from_index, connectors, grounded_elements = get_assembly_problem()

    robots = [end_effector]
    fixed_obstacles = [floor]

    saver = WorldSaver()
    # elements_from_layer = defaultdict(set)
    # if args.partial_ordering:
    #     for v in bar_struct.nodes():
    #         elements_from_layer[bar_struct.node[v]['layer']].add(v)
    #     partial_orders = compute_orders(elements_from_layer)
    # else:
    partial_orders = []
    print('Partial orders: ', partial_orders)
    # input("Enter to proceed.")

    with LockRenderer(not debug):
        if args.algorithm in STRIPSTREAM_ALGORITHM:
            plan = solve_pddlstream(robots, tool_from_ee, fixed_obstacles, args.problem, partial_orders=partial_orders,
                collisions=args.collisions, algorithm=args.algorithm, costs=args.costs, debug=debug, teleops=args.teleops)
        elif args.algorithm == 'regression':
                plan, data = regression(robots[0], tool_from_ee, fixed_obstacles, args.problem, collision=args.collisions, motions=True, stiffness=True,
                    revisit=False, verbose=True, lazy=False, partial_orders=partial_orders)
                    # bar_only=args.bar_only,
                print(data)
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

        element_from_index, connectors, grounded_elements = parse_2D_truss(args.problem)
        if watch and has_gui():
            saver.restore()
            #label_nodes(node_points)
            # elements = recover_sequence(trajectories, element_from_index)
            # endpts_from_element = bar_struct.get_axis_pts_from_element()
            # draw_ordered(elements, endpts_from_element)
            for e in element_from_index:
               set_color(element_from_index[e].body, (1, 0, 0, 0))
            if step_sim:
                time_step = None
            else:
                time_step = 0.01
            display_trajectories(trajectories, time_step=time_step, element_from_index=element_from_index)
        # verify
        if args.collisions:
            valid = validate_pddl_plan(trajectories, fixed_obstacles, element_from_index, grounded_elements, watch=False,
                allow_failure=has_gui() or debug, refine_num=1, debug=debug, bar_only=True)
            cprint('Valid: {}'.format(valid), 'green' if valid else 'red')
            assert valid
        else:
            cprint('Collision disabled, no verfication performed.', 'yellow')
    reset_simulation()
    disconnect()

#####################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='2D_truss_0_skeleton.json', help='The name of the problem to solve')
    parser.add_argument('-a', '--algorithm', default='focused', choices=ALGORITHMS, help='Planning algorithms')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enable the pybullet viewer.')
    parser.add_argument('-c', '--collisions', action='store_false', help='Disable collision checking.')
    parser.add_argument('-co', '--costs', action='store_true', help='Uses unit costs')
    parser.add_argument('-to', '--teleops', action='store_true', help='use teleop for trajectories (turn off in-between traj planning)')
    parser.add_argument('-po', '--partial_ordering', action='store_true', help='use partial ordering (if-any)')
    parser.add_argument('-db', '--debug', action='store_true', help='debug mode')

    parser.add_argument('-w', '--watch', action='store_true', help='watch trajectories')
    parser.add_argument('-sm', '--step_sim', action='store_true', help='stepping simulation.')
    parser.add_argument('-wr', '--write', action='store_true', help='Export results')
    args = parser.parse_args()
    print('Arguments:', args)

    run_planning(args, viewer=args.viewer, watch=args.watch, debug=args.debug, step_sim=args.step_sim, write=args.write)

if __name__ == '__main__':
    main()
