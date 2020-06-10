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

from coop_assembly.data_structure import WorldPose, MotionTrajectory

from pybullet_planning import set_camera_pose, connect, pose_from_base_values, create_box, wait_if_gui, set_pose, create_plane, \
    draw_pose, unit_pose, set_camera_pose2, Pose, Point, Euler, RED, BLUE, GREEN, CLIENT, HideOutput, create_obj, apply_alpha, \
    create_flying_body, create_shape, get_mesh_geometry, SE2, get_movable_joints, get_configuration, set_configuration, get_links

from .stream import get_element_body_in_goal_pose, get_2d_place_gen_fn

# viz settings
GROUND_COLOR = 0.8*np.ones(3)
BACKGROUND_COLOR = [0.9, 0.9, 1.0] # 229, 229, 255
SHADOWS = False

# robot geometry data files
HERE = os.path.dirname(__file__)
DUCK_OBJ_PATH = os.path.join(HERE, 'data', 'duck.obj')
INITIAL_CONF = [-1.0, 0.0, 0.0] # x, y, yaw

# representation used in pddlstream
ROBOT_TEMPLATE = 'r{}'
ELEMENT_ROBOT_TEMPLATE = 'e{}'

def index_from_name(robots, name):
    return robots[int(name[1:])]

class Conf(object):
    """wrapper for robot (incl. the element robot) configurations in pddlstream formulation
    """
    def __init__(self, robot, positions=None, element=None):
        self.robot = robot
        self.joints = get_movable_joints(self.robot)
        if positions is None:
            positions = get_configuration(self.robot)
        self.positions = positions
        self.element = element
    def assign(self):
        set_configuration(self.robot, self.positions)
    def __repr__(self):
        return '{}(E{})'.format(self.__class__.__name__, self.element)

###########################################
# convenient classes

Element2D = namedtuple('Element2D', ['index',
                                     'wlh',
                                     'body', # 'element_robot',
                                     'initial_pose', 'goal_pose',
                                    #  'grasps', 'layer'
                                     ])

###################################################

def get_pddlstream(robots, static_obstacles, element_from_index, grounded_elements, connectors,
                   partial_orders={}, printed=set(), removed=set(), collisions=True,
                   return_home=True, teleops=False, **kwargs): # checker=None, transit=False,
    # TODO update removed & printed
    assert not removed & printed, 'choose one of them!'
    remaining = set(element_from_index) - removed - printed
    element_obstacles = get_element_body_in_goal_pose(element_from_index, printed)
    obstacles = set(static_obstacles) | element_obstacles

    initial_confs = {ROBOT_TEMPLATE.format(i): Conf(robot, INITIAL_CONF) for i, robot in enumerate(robots)}

    domain_pddl = read(get_file_path(__file__, 'pddl/domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'pddl/stream.pddl'))
    constant_map = {}

    stream_map = {
        'sample-place': get_wild_place_gen_fn(robots, obstacles, element_from_index, grounded_elements,
                                              partial_orders=partial_orders, collisions=collisions, \
                                              initial_confs=initial_confs, teleops=teleops, **kwargs),
        'test-cfree': from_test(get_test_cfree()),
        # 'test-stiffness': from_test(test_stiffness),
    }

    #     stream_map.update({
    #         'sample-move': get_wild_transit_gen_fn(robots, obstacles, element_from_index, grounded_elements,
    #                                                partial_orders=partial_orders, collisions=collisions, bar_only=bar_only,
    #                                                initial_confs=initial_confs, teleops=teleops, **kwargs),
    #     })

    # * initial facts
    init = []
    # if transit:
    #     init.append(('Move',))
    for name, conf in initial_confs.items():
        init.extend([
            ('Robot', name),
            ('Conf', name, conf),
            ('AtConf', name, conf),
            ('CanMove', name),
        ])
    # static facts
    init.extend(('Grounded', e) for e in grounded_elements)
    init.extend(('Joined', e1, e2) for e1, e2 in connectors)
    init.extend(('Joined', e2, e1) for e1, e2 in connectors)
    init.extend(('Order',) + tup for tup in partial_orders)
    for e in remaining:
        init.extend([
            ('Element', e),
            ('Assembled', e),
        ])
    for rname in initial_confs:
            init.extend([('Assigned', rname, e) for e in remaining])

    # * goal facts
    goal_literals = []
    # if return_home:
    #     goal_literals.extend(('AtConf', r, q) for r, q in initial_confs.items())
    goal_literals.extend(('Removed', e) for e in remaining)
    goal = And(*goal_literals)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

##################################################

def get_wild_place_gen_fn(robots, obstacles, element_from_index, grounded_elements, partial_orders=[], collisions=True, \
        initial_confs={}, teleops=False, fluent_special=True, **kwargs):
    """ fluent_special : True if we are running incremental + semantic attachment
    """
    gen_fn_from_robot = {}
    for robot in robots:
        ee_link = get_links(robot)[-1]
        tool_link = get_links(robot)[-1]
        pick_gen_fn = get_2d_place_gen_fn(robot, element_from_index, obstacles, verbose=False, \
            collisions=collisions, teleops=teleops, **kwargs)
        gen_fn_from_robot[robot] = pick_gen_fn

    def wild_gen_fn(robot_name, element, fluents=[]):
        robot = index_from_name(robots, robot_name)
        printed = []
        # print('E{} - fluent {}'.format(element, fluents))
        for fact in fluents:
            if fact[0] == 'assembled':
                if fact[1] != element:
                    printed.append(fact[1])
            else:
                raise NotImplementedError(fact[0])
        for command, in gen_fn_from_robot[robot](element, printed=printed):
            if not fluent_special:
                q1 = Conf(robot, np.array(command.start_conf), element)
                q2 = Conf(robot, np.array(command.end_conf), element)
                outputs = [(q1, q2, command)]
            else:
                outputs = [(command,)]
            facts = []
            yield WildOutput(outputs, facts)
            # facts = [('Collision', command, e2) for e2 in command.colliding] if collisions else []
            # facts.append(('AtConf', robot_name, initial_confs[robot_name]))
            # cprint('print facts: {}'.format(command.colliding), 'yellow')
            # yield (q1, q2, command),

    return wild_gen_fn

def get_test_cfree():
    def test_fn(robot_name, traj, element):
        # return True if no collision detected
        return element not in traj.colliding
    return test_fn

##################################################

def load_2d_world(viewer=False):
    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    with HideOutput():
       floor = create_plane(color=GROUND_COLOR)
       # duck_body = create_obj(DUCK_OBJ_PATH, scale=0.2 * 1e-3, color=apply_alpha(GREEN, 0.5))
       # treat end effector as a flying 2D robot
       collision_id, visual_id = create_shape(get_mesh_geometry(DUCK_OBJ_PATH, scale=0.2 * 1e-3), collision=True, color=apply_alpha(GREEN, 0.5))
       end_effector = create_flying_body(SE2, collision_id, visual_id)

    # looking down from the top since it's 2D
    # TODO: view point a bit odd, can we change row value of the camera?
    camera_target_point = [0,0,0]
    set_camera_pose(camera_target_point + np.array([1e-3,1e-3,.5]), camera_target_point)
    draw_pose(unit_pose())
    return end_effector

def get_assembly_problem():
    # TODO: load 2D truss exported from GH
    # creating beams
    width = 0.01
    h = 0.01 # this dimension doesn't matter
    length = 0.2
    shrink = 0.015
    initial_pose = WorldPose('init', pose_from_base_values([0.3,-0.2,0]))
    element_dims = {0 : [width, length*np.sqrt(2)-2*shrink, h],
                    1 : [width, length-2*shrink, h],
                    2 : [width, length*np.sqrt(2)-2*shrink, h],
                    3 : [width, 2*length, h],
                    }
    element_from_index = {0 : Element2D(0, element_dims[0],
                                        create_box(*element_dims[0]),
                                        initial_pose, WorldPose(0, pose_from_base_values([0,length/2,np.pi/4]))),
                          1 : Element2D(1, element_dims[1],
                                        create_box(*element_dims[1]),
                                        initial_pose, WorldPose(1, pose_from_base_values([length/2,length/2,0]))),
                          2 : Element2D(2, element_dims[2],
                                        create_box(*element_dims[2]),
                                        initial_pose, WorldPose(2, pose_from_base_values([length,length/2,-np.pi/4]))),
                          3 : Element2D(3, element_dims[3],
                                        create_box(*element_dims[3], color=BLUE),
                                        initial_pose, WorldPose(3, pose_from_base_values([length/2,length,np.pi/2]))),
                          }
    for ei, e in element_from_index.items():
        set_pose(e.body, e.goal_pose.value)

    connectors = {(0,3) : np.array([0,0]),
                  (1,3) : np.array([0,0]),
                  (2,3) : np.array([0,0]),
                  }
    grounded_elements = [0, 1, 2]
    return element_from_index, connectors, grounded_elements

#####################################################

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-p', '--problem', default='blocked', help='The name of the problem to solve')
    parser.add_argument('-a', '--algorithm', default='focused', help='Specifies the algorithm')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enable the pybullet viewer.')
    parser.add_argument('-c', '--collisions', action='store_false', help='Disable collision checking.')
    parser.add_argument('-uc', '--unit_cost', action='store_true', help='Uses unit costs')
    parser.add_argument('-db', '--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    print('Arguments:', args)

    end_effector = load_2d_world(viewer=args.viewer)

    element_from_index, connectors, grounded_elements = get_assembly_problem()
    wait_if_gui()

    robots = [end_effector]
    static_obstacles = []
    pddlstream_problem = get_pddlstream(robots, static_obstacles, element_from_index, grounded_elements, connectors, collisions=args.collisions,
                   return_home=True, teleops=False)
                   # partial_orders={}, printed=set(), removed=set(),

    if args.debug:
        print('Init:', pddlstream_problem.init)
        print('Goal:', pddlstream_problem.goal)
    print('='*10)

    # success_cost = 0 if args.unit_cost else INF

    # # discrete_planner = 'ff-ehc'
    # # discrete_planner = 'ff-lazy-tiebreak' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
    # # discrete_planner = 'max-astar'
    # #
    # # discrete_planner = 'ff-eager-tiebreak'  # Need to use a eager search, otherwise doesn't incorporate child cost
    # # discrete_planner = 'ff-lazy'
    # discrete_planner = 'ff-wastar3'

    # if args.algorithm == 'focused':
    #     # ? Must-have: apply test-cfree implicitly, only needed in the optimistic algorithms
    #     stream_info = {
    #         # 'test-cfree': StreamInfo(negate=True),
    #     }

    #     #solution = solve_serialized(pddlstream_problem, planner='max-astar', unit_costs=args.unit, stream_info=stream_info)
    #     solution = solve_focused(pddlstream_problem, unit_costs=args.unit_cost, stream_info=stream_info, debug=False)
    # elif args.algorithm == 'incremental':
    #     solution = solve_incremental(pddlstream_problem, planner=discrete_planner, max_time=600,
    #                                  success_cost=success_cost, unit_costs=not args.unit_cost,
    #                                  max_planner_time=300, debug=args.debug, verbose=True)
    # else:
    #     raise ValueError(args.algorithm)

    # print("="*20)
    # if solution[0] is None:
    #     cprint('No solution found!', 'red')
    #     return
    # else:
    #     cprint('Solution found!', 'green')
    # print_solution(solution)
    # plan, cost, facts = solution

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
