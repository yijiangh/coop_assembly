import os, sys
import cProfile
import pstats
from termcolor import cprint
import numpy as np

try:
    # prioritize local pddlstream first
    # add your PDDLStream path here: https://github.com/caelan/pddlstream
    sys.path.append(os.environ['PDDLSTREAM_PATH'])
except KeyError:
    cprint('No `PDDLSTREAM_PATH` found in the env variables, using pddlstream submodule', 'yellow')
    here = os.path.abspath(os.path.dirname(__file__))
    sys.path.extend([
        os.path.abspath(os.path.join(here, '..', '..', 'external', 'pddlstream')),
    ])

try:
    sys.path.append(os.environ['PYPLANNERS_PATH'])
except KeyError:
    cprint('No `PYPLANNERS_PATH` found in the env variables, using pyplanner submodule', 'yellow')
    here = os.path.abspath(os.path.dirname(__file__))
    pyplanner_path = os.path.abspath(os.path.join(here, '..', '..', 'external', 'pyplanners'))
    # Inside PDDLStream, it will look for 'PYPLANNERS_PATH'
    os.environ['PYPLANNERS_PATH'] = pyplanner_path

from pddlstream.algorithms.downward import set_cost_scale, parse_action
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

from pybullet_planning import set_camera_pose, connect, create_box, wait_if_gui, set_pose, create_plane, \
    draw_pose, unit_pose, set_camera_pose2, Pose, Point, Euler, RED, BLUE, GREEN, CLIENT, HideOutput, create_obj, apply_alpha, \
    create_flying_body, create_shape, get_mesh_geometry, get_movable_joints, get_configuration, set_configuration, get_links, \
    has_gui, set_color, reset_simulation, disconnect, get_date, WorldSaver, LockRenderer, YELLOW, add_line, draw_circle, pairwise_collision, \
    body_collision_info, get_distance, draw_collision_diagnosis, get_aabb, BodySaver

from coop_assembly.planning.utils import compute_z_distance

from .stream import get_element_body_in_goal_pose, get_2d_place_gen_fn, pose_from_xz_values, xz_values_from_pose
from .robot_setup import Conf, INITIAL_CONF
from .parsing import parse_2D_truss

SS_OPTIONS = {
    'focused' : {'max_skeletons':None, 'bind':False, 'search_sample_ratio':0,},
    'binding' : {'max_skeletons':None, 'bind':True, 'search_sample_ratio':0,},
    'adaptive': {'max_skeletons':INF, 'bind':True, 'search_sample_ratio':2,},
}
# pddlstream algorithm options
STRIPSTREAM_ALGORITHM = [
    'incremental',
    'incremental_sa', # semantic attachment
    'focused',
    'binding',
    'adaptive',
]

# representation used in pddlstream
ROBOT_TEMPLATE = 'r{}'
ELEMENT_ROBOT_TEMPLATE = 'e{}'

def index_from_name(robots, name):
    return robots[int(name[1:])]

##################################################

# def get_height(index):
#     with BodySaver(element.body):
#         set_pose(element.body, element.goal_pose.value)
#         _, upper = get_aabb(element.body)
#         z = upper[2]
#     return z

def get_bias_fn(element_from_index):
    def bias_fn(state, goal, operators):
        assembled = {obj_from_pddl(atom.args[0]).value for atom in state.atoms if atom.predicate == 'assembled'}
        height = 0
        for index in assembled:
            # element = element_from_index[index]
            height = max(height, compute_z_distance(element_from_index, index))
        return height
    return bias_fn

def get_order_fn(element_from_index):
    def order_fn(state, goal, operators):
        actions = [op for op in operators if op.__class__.__name__ == 'Action'] # TODO: repair ugliness
        height_from_action = {}
        for action in actions:
            name, args = parse_action(action.fd_action.name)
            height = 0
            if name == 'place':
                args = [obj_from_pddl(arg).value for arg in args]
                _, index, _ = args
                # element = element_from_index[index]
                # height = get_height(element)
                height = compute_z_distance(element_from_index, index)
            height_from_action[action] = height
        return sorted(actions, key=height_from_action.__getitem__, reverse=True)
    return order_fn

#############################################

def get_pddlstream(robots, tool_from_ee, static_obstacles, element_from_index, grounded_elements, connectors,
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
        'sample-place': get_wild_2d_place_gen_fn(robots, tool_from_ee, obstacles, element_from_index, grounded_elements,
                                              partial_orders=partial_orders, collisions=collisions, \
                                              initial_confs=initial_confs, teleops=teleops,
                                              fluent_special=fluent_special, **kwargs),
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

def get_wild_2d_place_gen_fn(robots, tool_from_ee, obstacles, element_from_index, grounded_elements,
        partial_orders=[], collisions=True, initial_confs={}, teleops=False, fluent_special=False, **kwargs):
    """ fluent_special : True if we are running incremental + semantic attachment
    """
    gen_fn_from_robot = {}
    for robot in robots:
        # ee_link = get_links(robot)[-1]
        # tool_link = get_links(robot)[-1]
        pick_gen_fn = get_2d_place_gen_fn(robot, tool_from_ee, element_from_index, obstacles, verbose=False, \
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

def solve_pddlstream(robots, tool_from_ee, obstacles, problem, partial_orders={},
                     collisions=True, disable=False, max_time=60*4, algorithm='focused',
                     debug=False, costs=False, teleops=False, **kwargs):
    element_from_index, connectors, grounded_elements = parse_2D_truss(problem)

    fluent_special = algorithm == 'incremental_sa'
    if fluent_special:
        cprint('Using incremental + semantic attachment.', 'yellow')

    pddlstream_problem = get_pddlstream(robots, tool_from_ee, obstacles, element_from_index, grounded_elements, connectors, collisions=collisions,
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
    with LockRenderer(lock=not debug):
        if algorithm == 'incremental':
            discrete_planner = 'max-astar'
            solution = solve_incremental(pddlstream_problem, max_time=600, planner=discrete_planner,
                                        success_cost=success_cost, unit_costs=not costs,
                                        max_planner_time=300, debug=debug, verbose=True)
        elif algorithm == 'incremental_sa':
            discrete_planner = {
                'search': 'eager',
                'evaluator': 'greedy',
                'heuristic': 'ff',
                #'heuristic': ['ff', get_bias_fn(element_from_index)],
                #'successors': 'all',
                'successors': get_order_fn(element_from_index),

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

