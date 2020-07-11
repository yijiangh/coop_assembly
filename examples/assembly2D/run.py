#!/usr/bin/env python
from __future__ import print_function

import argparse
import os, sys
import cProfile
import json
import pstats
from collections import namedtuple
from termcolor import cprint
import numpy as np
from numpy.linalg import norm
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
from pddlstream.language.conversion import obj_from_pddl

from compas.datastructures import Network

from coop_assembly.data_structure import WorldPose, MotionTrajectory
from coop_assembly.planning.utils import get_element_neighbors, get_connector_from_elements, check_connected, get_connected_structures, \
    flatten_commands
from coop_assembly.planning.motion import display_trajectories
from coop_assembly.planning.validator import validate_trajectories, validate_pddl_plan
from coop_assembly.planning.parsing import get_assembly_path
from coop_assembly.planning.visualization import set_camera, label_points
from coop_assembly.geometry_generation.utils import get_element_neighbors

from pybullet_planning import set_camera_pose, connect, create_box, wait_if_gui, set_pose, create_plane, \
    draw_pose, unit_pose, set_camera_pose2, Pose, Point, Euler, RED, BLUE, GREEN, CLIENT, HideOutput, create_obj, apply_alpha, \
    create_flying_body, create_shape, get_mesh_geometry, get_movable_joints, get_configuration, set_configuration, get_links, \
    has_gui, set_color, reset_simulation, disconnect, get_date, WorldSaver, LockRenderer, YELLOW, add_line, draw_circle, pairwise_collision, \
    body_collision_info, get_distance, draw_collision_diagnosis, get_aabb, BodySaver

from .stream import get_element_body_in_goal_pose, get_2d_place_gen_fn, pose_from_xz_values, xz_values_from_pose

# pddlstream algorithm options
STRIPSTREAM_ALGORITHM = [
    'incremental',
    'incremental_sa', # semantic attachment
    'focused',
    'binding',
    'adaptive',
]
ALGORITHMS = STRIPSTREAM_ALGORITHM #+ ['regression']

SS_OPTIONS = {
    'focused' : {'max_skeletons':None, 'bind':False, 'search_sample_ratio':0,},
    'binding' : {'max_skeletons':None, 'bind':True, 'search_sample_ratio':0,},
    'adaptive': {'max_skeletons':INF, 'bind':True, 'search_sample_ratio':2,},
}

# viz settings
GROUND_COLOR = 0.8*np.ones(3)
BACKGROUND_COLOR = [0.9, 0.9, 1.0] # 229, 229, 255
SHADOWS = True

# robot geometry data files
HERE = os.path.dirname(__file__)
DUCK_OBJ_PATH = os.path.join(HERE, 'data', 'duck.obj')

SE2_xz = ['x', 'z', 'pitch']
INITIAL_CONF = [-1.0, 0.0, 0.0]

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
                   return_home=True, teleops=False, fluent_special=False, **kwargs): # checker=None, transit=False,
    # TODO update removed & printed
    assert not removed & printed, 'choose one of them!'
    remaining = set(element_from_index) - removed - printed
    element_obstacles = get_element_body_in_goal_pose(element_from_index, printed)
    obstacles = set(static_obstacles) | element_obstacles

    initial_confs = {ROBOT_TEMPLATE.format(i): Conf(robot, INITIAL_CONF) for i, robot in enumerate(robots)}

    if not fluent_special:
        domain_pddl = read(get_file_path(__file__, 'pddl/domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'pddl/stream.pddl'))
    else:
        domain_pddl = read(get_file_path(__file__, 'pddl/domain_fluent.pddl'))
        stream_pddl = read(get_file_path(__file__, 'pddl/stream_fluent.pddl'))

    constant_map = {}
    stream_map = {
        'sample-place': get_wild_2d_place_gen_fn(robots, obstacles, element_from_index, grounded_elements,
                                              partial_orders=partial_orders, collisions=collisions, \
                                              initial_confs=initial_confs, teleops=teleops, fluent_special=fluent_special, **kwargs),
        'test-cfree': from_test(get_test_cfree()),
        # 'test-stiffness': from_test(test_stiffness),
    }

    # if not fluent_special:
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

def get_wild_2d_place_gen_fn(robots, obstacles, element_from_index, grounded_elements, partial_orders=[], collisions=True, \
        initial_confs={}, teleops=False, fluent_special=False, **kwargs):
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
        # TODO: could check connectivity here
        #fluents = [] # For debugging
        robot = index_from_name(robots, robot_name)
        printed = []
        for fact in fluents:
            if fact[0] == 'assembled':
                if fact[1] != element:
                    printed.append(fact[1])
            else:
                raise NotImplementedError(fact[0])
        print('E{} - fluent printed {}'.format(element, printed))
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

def get_bias_fn(element_from_index):
    def bias_fn(state, goal, operators):
        assembled = {obj_from_pddl(atom.args[0]).value for atom in state.atoms if atom.predicate == 'assembled'}
        height = 0
        for index in assembled:
            element = element_from_index[index]
            with BodySaver(element.body):
                set_pose(element.body, element.goal_pose.value)
                lower, upper = get_aabb(element.body)
            height = max(height, upper[2])
        return height
    return bias_fn

def solve_pddlstream(robots, obstacles, element_from_index, grounded_elements, connectors, partial_orders={},
                     collisions=True, disable=False, max_time=60*4, algorithm='focused',
                     debug=False, costs=False, teleops=False, **kwargs):
    fluent_special = algorithm == 'incremental_sa'
    if fluent_special:
        cprint('Using incremental + semantic attachment.', 'yellow')

    pddlstream_problem = get_pddlstream(robots, obstacles, element_from_index, grounded_elements, connectors, collisions=collisions,
                   return_home=True, teleops=teleops, partial_orders=partial_orders, fluent_special=fluent_special)
                   # , printed=set(), removed=set(),

    if debug:
        print('Init:', pddlstream_problem.init)
        print('Goal:', pddlstream_problem.goal)
    print('='*10)

    # Reachability heuristics good for detecting dead-ends
    # Infeasibility from the start means disconnected or collision
    set_cost_scale(1)
    # planner = 'ff-ehc'
    # planner = 'ff-lazy-tiebreak' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
    # planner = 'max-astar'

    # planner = 'ff-eager-tiebreak'  # Need to use a eager search, otherwise doesn't incorporate child cost
    # planner = 'ff-lazy' | 'ff-wastar3'
    success_cost = 0 if costs else INF

    pr = cProfile.Profile()
    pr.enable()
    with LockRenderer(lock=False):
        if algorithm == 'incremental':
            discrete_planner = 'max-astar'
            solution = solve_incremental(pddlstream_problem, max_time=600, planner=discrete_planner,
                                        success_cost=success_cost, unit_costs=not costs,
                                        max_planner_time=300, debug=debug, verbose=True)
        elif algorithm == 'incremental_sa':
            discrete_planner = {
                'search': 'eager',
                'evaluator': 'greedy',
                #'heuristic': 'ff',
                'heuristic': ['ff', get_bias_fn(element_from_index)],
                'successors': 'all',
            }
            solution = solve_incremental(pddlstream_problem, max_time=600, planner=discrete_planner,
                                        success_cost=success_cost, unit_costs=not costs,
                                        max_planner_time=300, debug=debug, verbose=True)
        elif algorithm in SS_OPTIONS:
            # creates unique free variable for each output during the focused algorithm
            # (we have an additional search step that initially "shares" outputs, but it doesn't do anything in our domain)
            stream_info = {
                'sample-place': StreamInfo(PartialInputs(unique=True)),
                # 'sample-move': StreamInfo(PartialInputs(unique=True)),
                'test-cfree': StreamInfo(negate=True),
            }
            # TODO: effort_weight=0 will lead to noplan found
            effort_weight = 1e-3 if costs else None
            # effort_weight = 1e-3 if costs else 1 # | 0
            planner = 'ff-astar' if costs else 'ff-wastar3'

            # TODO what can we tell about a problem if it requires many iterations?
            # ? max_time: the maximum amount of time to apply streams
            # ? max_planner_time: maximal time allowed for the discrete search at each iter
            # ? effort_weight: a multiplier for stream effort compared to action costs
            # ? unit_efforts: use unit stream efforts rather than estimated numeric efforts
            # ? unit_costs: use unit action costs rather than numeric costs
            # ? reorder: if True, stream plans are reordered to minimize the expected sampling overhead
            # ? initial_complexity: the initial effort limit
            # ? max_failure only applies if max_skeletons=None
            # ? success_cost is the max cost of allowed solutions
            #   success_cost=0 runs the planner in a true anytime mode
            #   success_cost=INF terminates whenever any plan is found
            solution = solve_focused(pddlstream_problem, stream_info=stream_info,
                                     planner=planner, max_planner_time=60, max_time=max_time,
                                     max_skeletons=SS_OPTIONS[algorithm]['max_skeletons'],
                                     bind=SS_OPTIONS[algorithm]['bind'],
                                     search_sample_ratio=SS_OPTIONS[algorithm]['search_sample_ratio'],
                                     # ---
                                     unit_costs=not costs, success_cost=success_cost,
                                     unit_efforts=True, effort_weight=effort_weight,
                                     max_failures=0,  # 0 | INF
                                     reorder=False, initial_complexity=1,
                                     # ---
                                     debug=debug, verbose=True, visualize=False)
        else:
            raise NotImplementedError('Algorithm |{}| not in {}'.format(algorithm, STRIPSTREAM_ALGORITHM))
    pr.disable()
    if debug:
        pstats.Stats(pr).sort_stats('cumtime').print_stats(10)

    # print(solution)
    print_solution(solution)
    plan, _, facts = solution
    print('-'*10)
    # if debug:
    #     # cprint('certified facts: ', 'yellow')
    #     # for fact in facts[0]:
    #     #     print(fact)
    #     if facts[1] is not None:
    #         # preimage facts: the facts that support the returned plan
    #         cprint('preimage facts: ', 'green')
    #         for fact in facts[1]:
    #             print(fact)
    # TODO: post-process by calling planner again
    # TODO: could solve for trajectories conditioned on the sequence

    # is the irrelevant predicated discarded at the end?
    return plan


##################################################

def load_2d_world(viewer=False):
    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    with HideOutput():
       floor = create_plane(color=GROUND_COLOR)
       # duck_body = create_obj(DUCK_OBJ_PATH, scale=0.2 * 1e-3, color=apply_alpha(GREEN, 0.5))
       # treat end effector as a flying 2D robot
       collision_id, visual_id = create_shape(get_mesh_geometry(DUCK_OBJ_PATH, scale=0.2 * 1e-3), collision=True, color=apply_alpha(YELLOW, 0.5))
       end_effector = create_flying_body(SE2_xz, collision_id, visual_id)

    return end_effector, floor

def get_example_assembly_problem():
    # creating beams
    width = 0.01
    l = 0.01 # this dimension doesn't matter
    length = 0.2
    shrink = 0.015
    initial_pose = WorldPose('init', pose_from_xz_values([0.3,-0.2,0]))
    element_dims = {0 : [width, l, length*np.sqrt(2)-2*shrink],
                    1 : [width, l, length-2*shrink],
                    2 : [width, l, length*np.sqrt(2)-2*shrink],
                    3 : [width, l, 2*length],
                    }
    element_from_index = {0 : Element2D(0, element_dims[0],
                                        create_box(*element_dims[0]),
                                        initial_pose, WorldPose(0, pose_from_xz_values([0,length/2,-np.pi/4]))),
                          1 : Element2D(1, element_dims[1],
                                        create_box(*element_dims[1]),
                                        initial_pose, WorldPose(1, pose_from_xz_values([length/2,length/2,0]))),
                          2 : Element2D(2, element_dims[2],
                                        create_box(*element_dims[2]),
                                        initial_pose, WorldPose(2, pose_from_xz_values([length,length/2,np.pi/4]))),
                          3 : Element2D(3, element_dims[3],
                                        create_box(*element_dims[3], color=BLUE),
                                        initial_pose, WorldPose(3, pose_from_xz_values([length/2,length,np.pi/2]))),
                          }
    for ei, e in element_from_index.items():
        set_pose(e.body, e.goal_pose.value)

    # looking down from the top since it's 2D
    # TODO: use centroid of geometry here
    camera_target_point = [0.1,0,0.1]
    set_camera_pose(camera_target_point + np.array([0,-0.3,0]), camera_target_point)
    draw_pose(unit_pose())

    connectors = {(0,3) : None,
                  (1,3) : None,
                  (2,3) : None,
                  }
    grounded_elements = [0, 1, 2]
    return element_from_index, connectors, grounded_elements


def parse_2D_truss(problem, scale=1e-3, debug=False):
    problem_path = get_assembly_path(problem)
    with open(problem_path) as json_file:
        data = json.load(json_file)
        cprint('Parsed from : {}'.format(problem_path), 'green')

    net = Network.from_data(data)

    # TODO waiting for compas update to use ordered dict for nodes
    # node_points, edges = net.to_nodes_and_edges()
    node_points = [np.array([net.node[v]['x'], 0, net.node[v]['z']]) for v in range(net.number_of_nodes())]
    ground_nodes = [v for v, attr in net.nodes(True) if attr['fixed'] == True]

    initial_pose = WorldPose('init', pose_from_xz_values([2.0,0,2.0]))
    length = 0.01 # out-of-plane thickness
    element_from_index = {}
    grounded_elements = []
    for e, e_attr in net.edges(True):
        height = e_attr['radius'] * scale
        shrink = e_attr['shrink'] * scale
        width = norm(node_points[e[0]] - node_points[e[1]]) * scale
        wlh = [width - 2*shrink, length, height]

        mid_pt = (node_points[e[0]] + node_points[e[1]]) / 2 * scale
        # assert abs(mid_pt[1]) < 1e-9

        diff = (node_points[e[1]] - node_points[e[0]])
        pitch = np.math.atan2(diff[0], diff[2])
        e_pose = pose_from_xz_values([mid_pt[0],mid_pt[2],pitch+np.pi/2])
        e2d = Element2D(e, wlh,
                        create_box(*wlh),
                        initial_pose, WorldPose(e, e_pose))
        element_from_index[e] = e2d
        set_pose(e2d.body, e2d.goal_pose.value)

        if e_attr['fixed']:
            grounded_elements.append(e)

    connectors = {}
    element_neighbors = get_element_neighbors(element_from_index)
    for e, ens in element_neighbors.items():
        for en in ens:
            connectors[(e, en)] = None

    # * collision check for beams at goal poses
    collided_pairs = set()
    # `p_tol` is based on some manual experiement,
    # might need to be changed accordingly for specific scales and input models
    p_tol = 1e-3
    for i in element_from_index:
        for j in element_from_index:
            if i == j:
                continue
            if (i, j) not in collided_pairs and (j, i) not in collided_pairs:
                if pairwise_collision(element_from_index[i].body, element_from_index[j].body):
                    cr = body_collision_info(element_from_index[i].body, element_from_index[j].body)
                    penetration_depth = get_distance(cr[0][5], cr[0][6])
                    if penetration_depth > p_tol:
                        cprint('({}-{}) colliding : penetrating depth {:.4E}'.format(i,j, penetration_depth), 'red')
                        collided_pairs.add((i,j))
                        if debug:
                            draw_collision_diagnosis(cr, focus_camera=False)
    assert len(collided_pairs) == 0, 'model has mutual collision between elements!'
    cprint('No mutual collisions among elements in the model | penetration threshold: {}'.format(p_tol), 'green')

    set_camera(node_points, camera_dir=np.array([0,-1,0]), camera_dist=0.3)

    # draw the ideal truss that we want to achieve
    label_points([pt*1e-3 for pt in node_points])
    for e in net.edges():
        p1 = node_points[e[0]] * 1e-3
        p2 = node_points[e[1]] * 1e-3
        add_line(p1, p2, color=apply_alpha(BLUE, 0.3), width=0.5)
    for v in ground_nodes:
        draw_circle(node_points[v]*1e-3, 0.01)

    return element_from_index, connectors, grounded_elements

def run_pddlstream(args, viewer=False, watch=False, debug=False, step_sim=False, write=False):
    end_effector, floor = load_2d_world(viewer=args.viewer)
    # element_from_index, connectors, grounded_elements = get_assembly_problem()
    element_from_index, connectors, grounded_elements = parse_2D_truss(args.problem)

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

    if args.algorithm in STRIPSTREAM_ALGORITHM:
        plan = solve_pddlstream(robots, fixed_obstacles, element_from_index, grounded_elements, connectors, partial_orders=partial_orders,
            collisions=args.collisions, algorithm=args.algorithm, costs=args.costs, debug=debug, teleops=args.teleops)
    # elif args.algorithm == 'regression':
    #     with LockRenderer(True):
    #         plan, data = regression(robots[0], fixed_obstacles, bar_struct, collision=args.collisions, motions=True, stiffness=True,
    #             revisit=False, verbose=True, lazy=False, bar_only=args.bar_only, partial_orders=partial_orders)
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
            # elements = recover_sequence(trajectories, element_from_index)
            # endpts_from_element = bar_struct.get_axis_pts_from_element()
            # draw_ordered(elements, endpts_from_element)
            wait_if_gui('Ready to simulate trajectory.')
            for e in element_from_index:
               set_color(element_from_index[e].body, (1, 0, 0, 0))
            if step_sim:
                time_step = None
            else:
                time_step = 0.01
            display_trajectories(trajectories, time_step=time_step)
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
    parser.add_argument('-a', '--algorithm', default='focused', choices=ALGORITHMS, help='Stripstream algorithms')
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

    run_pddlstream(args, viewer=args.viewer, watch=args.watch, debug=args.debug, step_sim=args.step_sim, write=args.write)

if __name__ == '__main__':
    main()
