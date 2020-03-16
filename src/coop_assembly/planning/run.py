#!/usr/bin/env python
from __future__ import print_function

import sys
import argparse
import cProfile
import pstats
import numpy as np
import time
import os
import json

from pybullet_planning import connect, disconnect, get_movable_joints, get_joint_positions, LockRenderer, \
    unit_pose, reset_simulation, draw_pose, apply_alpha, BLACK, Pose, Euler, has_gui, set_numpy_seed, \
    set_random_seed, INF, wait_for_user, link_from_name, get_link_pose, point_from_pose, WorldSaver, elapsed_time, \
    timeout, get_configuration, RED, wait_if_gui
from pddlstream.utils import Profiler

from .planning import regression
from .stripstream import plan_sequence

DEFAULT_MAX_TIME = 5 * 60

########################################

def solve_assembly(robot, obstacles, element_from_id, node_points, element_bodies, extrusion_path, ground_nodes, args,
                    partial_orders=[], precompute=False, **kwargs):
    # TODO: could treat ground as partial orders
    backtrack_limit = INF # 0 | INF
    seed = hash((time.time(), args.seed))
    set_numpy_seed(seed)
    set_random_seed(seed)

    # initial_position = point_from_pose(get_link_pose(robot, link_from_name(robot, TOOL_LINK)))
    # elements = sorted(element_bodies.keys())
    # sequence = plan_stiffness(extrusion_path, element_from_id, node_points, ground_nodes, elements,
    #                           initial_position=initial_position, max_time=INF, max_backtrack=INF)
    # if has_gui():
    #     draw_ordered(sequence, node_points)
    #     wait_for_user()
    # return

    if args.algorithm == 'stripstream':
        sampled_trajectories = []
        # if precompute:
        #     sampled_trajectories = sample_trajectories(robot, obstacles, node_points, element_bodies, ground_nodes)
        plan, data = plan_sequence(robot, obstacles, node_points, element_bodies, ground_nodes,
                                   trajectories=sampled_trajectories, collisions=not args.cfree,
                                   max_time=args.max_time, disable=args.disable, **kwargs)
    elif args.algorithm == 'regression':
        plan, data = regression(robot, obstacles, element_bodies, extrusion_path, partial_orders=partial_orders,
                                heuristic=args.bias, max_time=args.max_time,
                                backtrack_limit=backtrack_limit, collisions=not args.cfree,
                                disable=args.disable, stiffness=args.stiffness, motions=args.motions, **kwargs)
    else:
        raise ValueError(args.algorithm)
    # if args.motions:
    #     plan = compute_motions(robot, obstacles, element_bodies, initial_conf, plan, collisions=not args.cfree)
    return plan, data

def plan_assembly(args_list, viewer=False, verify=False, verbose=False, watch=False):
    results = []
    if not args_list:
        return results
    # TODO: setCollisionFilterGroupMask
    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    problems = {args.problem for args in args_list}
    assert len(problems) == 1
    [problem] = problems

    # TODO: change dir for pddlstream
    extrusion_path = get_extrusion_path(problem)
    # element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path, verbose=True)
    # elements = sorted(element_from_id.values())

    connect(use_gui=viewer) #, shadows=SHADOWS, color=BACKGROUND_COLOR)
    with LockRenderer(lock=True):
        #draw_pose(unit_pose(), length=1.)
        obstacles, robot = load_world()
        #draw_model(elements, node_points, ground_nodes)
        #wait_for_user()
        color = apply_alpha(BLACK, alpha=0) # 0, 1
        #color = None
        # element_bodies = dict(zip(elements, create_elements_bodies(node_points, elements, color=color)))
        # set_extrusion_camera(node_points)
        #if viewer:
        #    label_nodes(node_points)
        saver = WorldSaver()

    # checker = create_stiffness_checker(extrusion_path, verbose=False) # if stiffness else None
    #visualize_stiffness(extrusion_path)
    #debug_elements(robot, node_points, node_order, elements)

    for args in args_list:
        saver.restore()
        #initial_conf = get_joint_positions(robot, get_movable_joints(robot))
        with LockRenderer(lock=not viewer):
            start_time = time.time()
            plan, data = None, {}
            with timeout(args.max_time):
                with Profiler(num=10, cumulative=False):
                    plan, data = solve_assembly(robot, obstacles, element_from_id, node_points, element_bodies,
                                                 extrusion_path, ground_nodes, args, checker=checker)
            runtime = elapsed_time(start_time)

        # TODO: recover assembly order
        # sequence = recover_directed_sequence(plan)
        # if verify:
        #     max_translation, max_rotation = compute_plan_deformation(extrusion_path, recover_sequence(plan))
        #     valid = check_plan(extrusion_path, sequence)
        #     print('Valid:', valid)
        #     safe = validate_trajectories(element_bodies, obstacles, plan)
        #     print('Safe:', safe)
        #     data.update({
        #         'safe': safe,
        #         'valid': valid,
        #         'max_translation': max_translation,
        #         'max_rotation': max_rotation,
        #     })

        data.update({
            'runtime': runtime,
            'num_elements': len(elements),
            # 'ee_distance': compute_sequence_distance(node_points, sequence, start=initial_position, end=initial_position),
            # 'print_distance': get_print_distance(plan, teleport=True),
            # 'distance': get_print_distance(plan, teleport=False),
            'sequence': sequence,
            # 'parameters': get_global_parameters(),
            'problem':  args.problem,
            'algorithm': args.algorithm,
            'heuristic': args.bias,
            'plan_extrusions': not args.disable,
            'use_collisions': not args.cfree,
            'use_stiffness': args.stiffness,
        })
        print(data)
        #plan_path = '{}_solution.json'.format(args.problem)
        #with open(plan_path, 'w') as f:
        #    json.dump(plan_data, f, indent=2, sort_keys=True)
        results.append((args, data))

    if watch and (plan is not None):
        animate = not (args.disable or args.ee_only)
        display_trajectories(node_points, ground_nodes, plan, #time_step=None, video=True,
                             animate=animate)
        reset_simulation()
        disconnect()
    if not verbose:
        sys.stdout.close()
    return results

##################################################

def create_parser():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', default='regression',
                        help='Which algorithm to use')
    # parser.add_argument('-b', '--bias', default='plan-stiffness', choices=HEURISTICS,
    #                     help='Which heuristic to use')
    parser.add_argument('-c', '--cfree', action='store_true',
                        help='Disables collisions with obstacles')
    parser.add_argument('-d', '--disable', action='store_true',
                        help='Disables trajectory planning')
    parser.add_argument('-e', '--ee_only', action='store_true',
                        help='Disables arm planning')
    parser.add_argument('-m', '--motions', action='store_false',
                        help='Plans motions between each extrusion')
    parser.add_argument('-s', '--stiffness',  action='store_false',
                        help='Disables stiffness checking')
    parser.add_argument('-t', '--max_time', default=DEFAULT_MAX_TIME, type=int,
                        help='The max time')
    return parser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='choreo_brick_demo', help='The name of the problem to solve')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    args = parser.parse_args()
    print('Arguments:', args)

    plan_assembly([args], viewer=args.viewer, verbose=True, watch=True)

    # connect(use_gui=args.viewer)
    # robot, brick_from_index, obstacle_from_name = load_pick_and_place(args.problem)

    # np.set_printoptions(precision=2)
    # pr = cProfile.Profile()
    # pr.enable()
    # with WorldSaver():
    #     pddlstream_problem = get_pddlstream(robot, brick_from_index, obstacle_from_name,
    #                                         collisions=not args.cfree, teleport=args.teleport)
    #     with LockRenderer():
    #         solution = solve_focused(pddlstream_problem, planner='ff-wastar1', max_time=30)
    # pr.disable()
    # pstats.Stats(pr).sort_stats('cumtime').print_stats(10)
    # print_solution(solution)
    # plan, _, _ = solution
    # if plan is None:
    #     return

    # if not has_gui():
    #     connect(use_gui=True)
    #     _ = load_pick_and_place(args.problem)
    # step_plan(plan, time_step=(np.inf if args.teleport else 0.01))
    # #simulate_plan(plan)


if __name__ == '__main__':
    main()
