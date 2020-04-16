import numpy as np
from numpy.linalg import norm
import cProfile
import pstats
from termcolor import cprint

import os, sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.extend([
    os.path.join(here, 'pddlstream/'),
])

from pddlstream.algorithms.downward import set_cost_scale
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.disabled import process_stream_plan
from pddlstream.language.constants import And, PDDLProblem, print_solution, DurativeAction, Equal
from pddlstream.language.generator import from_test, from_gen_fn, from_fn
from pddlstream.language.stream import StreamInfo, PartialInputs, WildOutput
from pddlstream.language.function import FunctionInfo
from pddlstream.utils import read, get_file_path, inclusive_range
from pddlstream.language.temporal import compute_duration, get_end

from pybullet_planning import has_gui, get_movable_joints, get_configuration, set_configuration, WorldSaver, LockRenderer, \
    wait_if_gui, EndEffector, link_from_name, joints_from_names, intrinsic_euler_from_quat, get_links, create_attachment, \
    set_joint_positions, get_links, set_pose, INF
from .stream import get_element_body_in_goal_pose, get_place_gen_fn, ENABLE_SELF_COLLISIONS, get_pregrasp_gen_fn, command_collision
from .utils import flatten_commands, recover_sequence, Command
from .visualization import draw_ordered
from .motion import display_trajectories, compute_motion, BAR_INITIAL_CONF
from .robot_setup import EE_LINK_NAME, TOOL_LINK_NAME, IK_JOINT_NAMES, JOINT_WEIGHTS, RESOLUTION, get_disabled_collisions, INITIAL_CONF
from coop_assembly.data_structure.utils import MotionTrajectory
from coop_assembly.help_functions import METER_SCALE

STRIPSTREAM_ALGORITHM = [
    'incremental',
    'focused',
]

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

##################################################

def get_pddlstream(robots, static_obstacles, element_from_index, grounded_elements, connectors,
                   printed=set(), removed=set(), collisions=True,
                   temporal=False, transit=False, return_home=True, checker=None, bar_only=False, **kwargs):
    # TODO update removed & printed
    assert not removed & printed
    remaining = set(element_from_index) - removed - printed
    element_obstacles = get_element_body_in_goal_pose(element_from_index, printed)
    obstacles = set(static_obstacles) | element_obstacles

    partial_orders = set()
    if not bar_only:
        initial_confs = {ROBOT_TEMPLATE.format(i): Conf(robot, INITIAL_CONF) for i, robot in enumerate(robots)}
    else:
        robots = [element_from_index[e].element_robot for e in element_from_index]
        initial_confs = {ELEMENT_ROBOT_TEMPLATE.format(i): Conf(robot, BAR_INITIAL_CONF) for i, robot in enumerate(robots)}

    domain_pddl = read(get_file_path(__file__, 'pddl/temporal.pddl' if temporal else 'pddl/domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'pddl/stream.pddl'))
    constant_map = {}

    stream_map = {
        #'test-cfree': from_test(get_test_cfree(element_bodies)),
        'sample-move': get_wild_transit_gen_fn(robots, obstacles, element_from_index, grounded_elements,
                                               partial_orders=partial_orders, collisions=collisions, bar_only=bar_only,
                                               initial_confs=initial_confs,**kwargs),
        'sample-place': get_wild_place_gen_fn(robots, obstacles, element_from_index, grounded_elements,
                                              partial_orders=partial_orders, collisions=collisions, bar_only=bar_only, \
                                              initial_confs=initial_confs, **kwargs),
        # 'test-stiffness': from_test(test_stiffness),
    }

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
    init.extend(('Grounded', e) for e in grounded_elements)
    init.extend(('Joined', e1, e2) for e1, e2 in connectors)
    init.extend(('Joined', e2, e1) for e1, e2 in connectors)

    for e in remaining:
        init.extend([
            ('Element', e),
            ('Assembled', e),
        ])
    if not bar_only:
        for rname in initial_confs:
            init.extend([('Assigned', rname, e) for e in remaining])
    else:
        init.extend([('Assigned', rname, e) for e, rname in zip(remaining, initial_confs)])

    goal_literals = []
    # if return_home:
    #     goal_literals.extend(('AtConf', r, q) for r, q in initial_confs.items())
    goal_literals.extend(('Removed', e) for e in remaining)
    goal = And(*goal_literals)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

##################################################

def solve_pddlstream(robots, obstacles, element_from_index, grounded_elements, connectors,
                     collisions=True, disable=False, max_time=60, bar_only=False, algorithm='incremental', debug=False, **kwargs):
    pddlstream_problem = get_pddlstream(robots, obstacles, element_from_index, grounded_elements, connectors,
                                        collisions=collisions, bar_only=bar_only, **kwargs)
    if debug:
        print('Init:', pddlstream_problem.init)
        print('Goal:', pddlstream_problem.goal)
    print('='*10)

    # creates unique free variable for each output during the focused algorithm
    # (we have an additional search step that initially "shares" outputs, but it doesn't do anything in our domain)
    stream_info = {
        'sample-place': StreamInfo(PartialInputs(unique=True)),
        'sample-move': StreamInfo(PartialInputs(unique=True)),
    }

    # Reachability heuristics good for detecting dead-ends
    # Infeasibility from the start means disconnected or collision
    set_cost_scale(1)
    # planner = 'ff-ehc'
    # planner = 'ff-lazy-tiebreak' # Branching factor becomes large. Rely on preferred. Preferred should also be cheaper
    planner = 'ff-eager-tiebreak'  # Need to use a eager search, otherwise doesn't incorporate child cost
    # planner = 'max-astar'

    pr = cProfile.Profile()
    pr.enable()
    with LockRenderer(lock=True):
        if algorithm == 'incremental':
            solution = solve_incremental(pddlstream_problem, planner=planner, max_time=600,
                                        max_planner_time=300, debug=debug, verbose=True)
        elif algorithm == 'focused':
            # TODO what can we tell about a problem if it requires many iterations?
            # * max_skeletons=None, bind=False is focus
            # * max_skeletons=None, bind=True is binding
            # * max_skeletons!=None is adaptive
            # ? max_time: the maximum amount of time to apply streams
            # ? max_planner_time: maximal time allowed for the discrete search at each iter
            # ? effort_weight: a multiplier for stream effort compared to action costs
            # ? unit_efforts: use unit stream efforts rather than estimated numeric efforts
            # ? unit_costs: use unit action costs rather than numeric costs
            # ? reorder: if True, stream plans are reordered to minimize the expected sampling overhead
            # ? initial_complexity: the initial effort limit
            solution = solve_focused(pddlstream_problem, stream_info=stream_info,
                                     planner=planner, max_planner_time=60, max_time=max_time,
                                     max_skeletons=None, bind=False,
                                     # ---
                                     effort_weight=None, unit_efforts=True, unit_costs=False, # TODO: effort_weight=None vs 0
                                     max_failures=0,  # 0 | INF
                                     reorder=False, initial_complexity=1,
                                     # ---
                                     debug=debug, verbose=True, visualize=False)
        else:
            raise NotImplementedError('Algorithm |{}| not in {}'.format(algorithm, STRIPSTREAM_ALGORITHM))
    pr.disable()
    if debug:
        pstats.Stats(pr).sort_stats('cumtime').print_stats(10)

    if bar_only:
        # reset all element_robots to clear the visual scene
        for e in element_from_index:
            e_robot = element_from_index[e].element_robot
            set_joint_positions(e_robot, get_movable_joints(e_robot), np.zeros(6))

    # print(solution)
    print_solution(solution)
    plan, _, facts = solution
    print('-'*10)
    if debug:
        print('certified facts: ')
        for fact in facts[0]:
            print(fact)
        if facts[1] is not None:
            # preimage facts: the facts that support the returned plan
            print('preimage facts: ')
            for fact in facts[1]:
                print(fact)
    # TODO: post-process by calling planner again
    # TODO: could solve for trajectories conditioned on the sequence
    return plan

###############################################################

def get_wild_place_gen_fn(robots, obstacles, element_from_index, grounded_elements, partial_orders=[], collisions=True, \
    bar_only=False, initial_confs={}, **kwargs):
    gen_fn_from_robot = {}
    for robot in robots:
        ee_link = link_from_name(robot, EE_LINK_NAME) if not bar_only else get_links(robot)[-1]
        tool_link = link_from_name(robot, TOOL_LINK_NAME) if not bar_only else get_links(robot)[-1]
        # TODO end_effector is unused in bar_only setting
        end_effector = EndEffector(robot, ee_link=ee_link,
                                   tool_link=tool_link,
                                   visual=False, collision=True)
        pick_gen_fn = get_place_gen_fn(end_effector, element_from_index, obstacles, verbose=False, \
            precompute_collisions=True, collisions=collisions, bar_only=bar_only, **kwargs)
        gen_fn_from_robot[robot] = pick_gen_fn

    def wild_gen_fn(robot_name, element):
        robot = index_from_name(robots, robot_name)
        for command, in gen_fn_from_robot[robot](element):
            q1 = Conf(robot, np.array(command.start_conf), element)
            q2 = Conf(robot, np.array(command.end_conf), element)
            outputs = [(q1, q2, command)]
            # TODO Caelan said that we might not have to use wild facts to enforce collision-free
            facts = [('Collision', command, e2) for e2 in command.colliding] if collisions else []
            # facts.append(('AtConf', robot_name, initial_confs[robot_name]))
            cprint('print facts: {}'.format(command.colliding), 'yellow')
            yield WildOutput(outputs, facts)
    return wild_gen_fn

def get_wild_transit_gen_fn(robots, obstacles, element_from_index, grounded_elements, partial_orders=[], collisions=True, bar_only=False, \
    initial_confs={}, **kwargs):
    # TODO initial confs
    # https://github.com/caelan/pb-construction/blob/30b42e12c82de3ba4b117ffc380e58dd649c0ec5/extrusion/stripstream.py#L765

    def wild_gen_fn(robot_name, q2, current_command):
        # transit_start_conf = INITIAL_CONF if not bar_only else BAR_INITIAL_CONF
        # assert norm(q1.positions - transit_start_conf) < 1e-8
        init_q = initial_confs[robot_name]
        assert norm(q2.positions - current_command.start_conf) < 1e-8

        robot = index_from_name(robots, robot_name)
        attachments = current_command.trajectories[0].attachments
        traj = compute_motion(robot, obstacles, element_from_index, [],
                       init_q.positions, q2.positions, attachments=attachments,
                       collisions=collisions, bar_only=bar_only,
                       restarts=3, iterations=100, smooth=100, max_distance=0.0)
        if not traj:
            cprint('Transit sampling failed.', 'red')
            return

        command = Command([traj])
        elements_order = [e for e in element_from_index if (e != current_command.trajectories[0].element)]
            # and (element_from_index[e].body not in obstacles)]
        bodies_order = get_element_body_in_goal_pose(element_from_index, elements_order)
        colliding = command_collision(command, bodies_order)
        for element2, unsafe in zip(elements_order, colliding):
            if unsafe:
                command.set_unsafe(element2)
            else:
                command.set_safe(element2)
        facts = [('Collision', command, e2) for e2 in command.colliding] if collisions else []
        cprint('transit facts: {}'.format(command.colliding), 'blue')
        cprint('E#{} | Colliding: {}'.format(traj.element, len(command.colliding)), 'green')

        outputs = [(command,)]
        # facts = []
        yield WildOutput(outputs, facts)
    return wild_gen_fn

def test_stiffness(fluents=[]):
    assert all(fact[0] == 'printed' for fact in fluents)
    # TODO: to use the non-skeleton focused algorithm, need to remove the negative axiom upon success
    # elements = {fact[1] for fact in fluents}
    return True
