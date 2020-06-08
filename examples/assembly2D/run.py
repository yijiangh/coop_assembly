#!/usr/bin/env python
from __future__ import print_function

import argparse
import os, sys
from collections import namedtuple
from termcolor import cprint
import numpy as np
import math
import pybullet as pb

HERE = os.path.dirname(os.path.abspath(__file__))

# add your PDDLStream path here: https://github.com/caelan/pddlstream
sys.path.append(os.environ['PDDLSTREAM_PATH'])
from pddlstream.algorithms.downward import set_cost_scale
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.disabled import process_stream_plan
from pddlstream.language.constants import And, PDDLProblem, print_solution, DurativeAction, Equal
from pddlstream.language.generator import from_test, from_gen_fn, from_fn, empty_gen
from pddlstream.language.stream import StreamInfo, PartialInputs, WildOutput
from pddlstream.language.function import FunctionInfo
from pddlstream.utils import read, get_file_path, inclusive_range, INF
from pddlstream.language.temporal import compute_duration, get_end

from pybullet_planning import set_camera_pose, connect, pose_from_base_values, create_box, wait_if_gui, set_pose, create_plane, \
    draw_pose, unit_pose, set_camera_pose2, Pose, Point, Euler, RED, BLUE, GREEN, CLIENT

GROUND_COLOR = 0.8*np.ones(3)
BACKGROUND_COLOR = [0.9, 0.9, 1.0] # 229, 229, 255
SHADOWS = False

GRASP = np.array([0, 0])
# ELEMENT_TEMPLATE = 'E{}'
# CLAMP_TEMPLATE = 'C{}'
# GRIPPER_TEMPLATE = 'G{}'

ELEMENT_TEMPLATE = 'Element{}'
CLAMP_TEMPLATE = 'Clamp{}'
GRIPPER_TEMPLATE = 'Gripper{}'

Element = namedtuple('Element', ['index',
                                #  'axis_endpoints', 'radius',
                                 'body', # 'element_robot',
                                 'initial_pose', 'goal_pose',
                                #  'grasps', 'layer'
                                 ])

Gripper = namedtuple('Gripper', ['index',
                            'rest_pose',
                            ]
                            )

Clamp = namedtuple('Clamp', ['index',
                            'rest_pose',
                            ]
                            )

###################################################

def value_from_name(object_from_index, name):
    return object_from_index[int(name[1:])]

def is_close(p1, p2):
    return np.linalg.norm(p1-p2) < 1e-3

def get_dock_region_test(gripper_from_index, clamp_from_index):
    def test_fn(ee_name, p):
        # return interval_contains(regions[r], get_block_interval(b, p))
        if ee_name[0] == 'g':
            return is_close(p, value_from_name(gripper_from_index, ee_name).rest_pose)
        elif ee_name[1] == 'c':
            return is_close(p, value_from_name(clamp_from_index, ee_name).rest_pose)
        else:
            raise ValueError('not an end effector: {}'.format(ee_name))
    return test_fn

###################################################

def get_pddlstream_problem(element_from_index, grounded_elements, connectors, gripper_from_index, clamp_from_index):
    domain_pddl = read(os.path.join(HERE, 'domain.pddl'))
    stream_pddl = read(os.path.join(HERE, 'stream.pddl'))

    constant_map = {
        'gripper_restpose': gripper_from_index[0].rest_pose,
    }

    element_names = {e : ELEMENT_TEMPLATE.format(e) for e in element_from_index}
    gripper_names = {g : GRIPPER_TEMPLATE.format(g) for g in gripper_from_index}
    clamp_names = {c : CLAMP_TEMPLATE.format(c) for c in clamp_from_index}

    # robot
    initial_conf = np.array([0,-2])

    init = [
        ('CanMove',),
        ('Conf', initial_conf),
        ('AtConf', initial_conf),
        ('FlangeEmpty',),
        ('HandEmpty',),
        # Equal((TOTAL_COST,), 0)] + \
        ]

    for g, gp in gripper_from_index.items():
        init.extend([
            ('Gripper', gripper_names[g]),
            ('EndEffector', gripper_names[g]),
            ('Pose', gp.rest_pose),
            ('DockPose', gripper_names[g], gp.rest_pose),
            ('AtPose', gripper_names[g], gp.rest_pose),
        ])

    for c, cl in clamp_from_index.items():
        init.extend([
            ('Clamp', clamp_names[c]),
            ('EndEffector', clamp_names[c]),
            ('Pose', cl.rest_pose),
            ('DockPose', clamp_names[c], cl.rest_pose),
            ('AtPose', clamp_names[c], cl.rest_pose),
        ])

    init.extend(('Grounded', element_names[e]) for e in grounded_elements)
    for joined, join_pose  in connectors.items():
        init.append(('Joined', element_names[joined[0]], element_names[joined[1]]))
        init.append(('Joined', element_names[joined[1]], element_names[joined[0]]))
        init.append(('Pose', join_pose))
        init.append(('ConnectorPose', element_names[joined[0]], element_names[joined[1]], join_pose))
        init.append(('ConnectorPose', element_names[joined[1]], element_names[joined[0]], join_pose))

    # init.extend(('Order',) + tup for tup in partial_orders)

    for e, element in element_from_index.items():
        init.extend([
            ('Element', element_names[e]),
            ('Removed', e),
            ('Pose', element.initial_pose),
            ('Pose', element.goal_pose),
            ('AtPose', element_names[e], element.initial_pose),
            ('GoalPose', element_names[e], element.goal_pose),
        ])

    goal_literals = []
    # if return_home:
    #     goal_literals.extend(('AtConf', r, q) for r, q in initial_confs.items())
    goal_literals.extend(('Assembled', element_names[e]) for e in element_from_index)
    # goal_literals.extend(('AtPose', element_names[e], element.goal_pose) for e, element in element_from_index.items())
    goal = And(*goal_literals)

    cprint('Initial predicates:', 'blue')
    print(init)
    cprint('Goal predicates:', 'blue')
    print(goal)

    # TODO: convert to lower case
    stream_map = {
        # ? in this case of a floating gripper the IK is trivial
        'inverse-kinematics':  from_fn(lambda p: (p + GRASP,)),
        ##
        # ? if tested in collision, certify CollisionFree
        # 'test-cfree': from_test(lambda *args: not collision_test(*args)),
        #'sample-pose': from_gen_fn(lambda: ((np.array([x, 0]),) for x in range(len(poses), n_poses))),
        # ? simple enumerator
        # 'sample-pose': from_gen_fn(lambda: ((p,) for p in tamp_problem.poses)),
        # 't-dock_region': from_test(get_dock_region_test(gripper_from_index, clamp_from_index)),
        # # 'collision': collision_test,
        # 'distance': distance_fn,
    }

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

##################################################

def load_2d_world(viewer=False):
    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    floor = create_plane(color=GROUND_COLOR)
    # looking down from the top since it's 2D
    camera_target_point = [1,1,0]
    set_camera_pose(camera_target_point + np.array([1e-3,1e-3,1.5]), camera_target_point)
    draw_pose(unit_pose())

def get_assembly_problem():
    # creating beams
    width = 0.1
    h = 0.01 # this dimension doesn't matter
    shrink=0.15
    initial_pose = pose_from_base_values([3,-2,0])
    element_from_index = {0 : Element(0, create_box(width, 2*np.sqrt(2)-2*shrink, h), initial_pose, pose_from_base_values([0,1,np.pi/4])),
                          1 : Element(1, create_box(width, 2-2*shrink, h),          initial_pose, pose_from_base_values([1,1,0])),
                          2 : Element(2, create_box(width, 2*np.sqrt(2)-2*shrink, h), initial_pose, pose_from_base_values([2,1,-np.pi/4])),
                          3 : Element(3, create_box(width, 4, h, color=BLUE), initial_pose, pose_from_base_values([1,2,np.pi/2])),
                          }
    for ei, e in element_from_index.items():
        set_pose(e.body, e.goal_pose)

    connectors = {(0,3) : np.array([0,0]),
                  (1,3) : np.array([0,0]),
                  (2,3) : np.array([0,0]),
                  }
    grounded_elements = [0, 1, 2]
    return element_from_index, connectors, grounded_elements

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-p', '--problem', default='blocked', help='The name of the problem to solve')
    parser.add_argument('-a', '--algorithm', default='focused', help='Specifies the algorithm')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enable the pybullet viewer.')
    parser.add_argument('-uc', '--unit_cost', action='store_true', help='Uses unit costs')
    parser.add_argument('-db', '--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    print('Arguments:', args)

    load_2d_world(viewer=args.viewer)

    element_from_index, connectors, grounded_elements = get_assembly_problem()
    # wait_if_gui()

    gp_initial_conf = np.array([1,-2])
    gripper_from_index = {0 : Gripper(0, gp_initial_conf)}

    pddlstream_problem = get_pddlstream_problem(element_from_index, grounded_elements, connectors, gripper_from_index)

    success_cost = 0 if args.unit_cost else INF

    # discrete_planner = 'ff-ehc'
    # discrete_planner = 'ff-lazy-tiebreak' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
    # discrete_planner = 'max-astar'
    #
    # discrete_planner = 'ff-eager-tiebreak'  # Need to use a eager search, otherwise doesn't incorporate child cost
    # discrete_planner = 'ff-lazy'
    discrete_planner = 'ff-wastar3'

    if args.algorithm == 'focused':
        # ? Must-have: apply test-cfree implicitly, only needed in the optimistic algorithms
        stream_info = {
            # 'test-cfree': StreamInfo(negate=True),
        }

        #solution = solve_serialized(pddlstream_problem, planner='max-astar', unit_costs=args.unit, stream_info=stream_info)
        solution = solve_focused(pddlstream_problem, unit_costs=args.unit_cost, stream_info=stream_info, debug=False)
    elif args.algorithm == 'incremental':
        solution = solve_incremental(pddlstream_problem, planner=discrete_planner, max_time=600,
                                     success_cost=success_cost, unit_costs=not args.unit_cost,
                                     max_planner_time=300, debug=args.debug, verbose=True)
    else:
        raise ValueError(args.algorithm)

    print("="*20)
    if solution[0] is None:
        cprint('No solution found!', 'red')
        return
    else:
        cprint('Solution found!', 'green')
    print_solution(solution)
    plan, cost, facts = solution

    # if args.debug:
    #     cprint('certified facts: ', 'yellow')
    #     for fact in facts[0]:
    #         print(fact)
    #     if facts[1] is not None:
    #         # preimage facts: the facts that support the returned plan
    #         cprint('preimage facts: ', 'green')
    #         for fact in facts[1]:
    #             print(fact)

    # apply_plan(tamp_problem, plan)


if __name__ == '__main__':
    main()
