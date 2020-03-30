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
from pddlstream.algorithms.focused import solve_focused #, CURRENT_STREAM_PLAN
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
from .stream import get_element_body_in_goal_pose, get_place_gen_fn, ENABLE_SELF_COLLISIONS, get_pregrasp_gen_fn
from .utils import flatten_commands, recover_sequence, Command
from .visualization import draw_ordered
from .motion import display_trajectories, compute_motion, BAR_INITIAL_CONF
from .robot_setup import EE_LINK_NAME, TOOL_LINK_NAME, IK_JOINT_NAMES, JOINT_WEIGHTS, RESOLUTION, get_disabled_collisions, INITIAL_CONF
from coop_assembly.data_structure.utils import MotionTrajectory
from coop_assembly.help_functions import METER_SCALE

STRIPSTREAM_ALGORITHM = 'stripstream'
ROBOT_TEMPLATE = 'r{}'
ELEMENT_ROBOT_TEMPLATE = 'e{}'

def index_from_name(robots, name):
    return robots[int(name[1:])]

class Conf(object):
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
        return '{}(E#{})'.format(self.__class__.__name__, self.element)

##################################################

def get_pddlstream(robots, static_obstacles, element_from_index, grounded_elements, connectors,
                   printed=set(), removed=set(), collisions=True,
                   temporal=False, transit=False, return_home=True, checker=None, bar_only=False, **kwargs):
    assert not removed & printed
    remaining = set(element_from_index) - removed - printed
    element_obstacles = get_element_body_in_goal_pose(element_from_index, printed)
    obstacles = set(static_obstacles) | element_obstacles

    partial_orders = set()
    if not bar_only:
        initial_confs = {ROBOT_TEMPLATE.format(i): INITIAL_CONF for i, robot in enumerate(robots)}
    else:
        initial_confs = {ELEMENT_ROBOT_TEMPLATE.format(i): BAR_INITIAL_CONF for i in element_from_index}

    domain_pddl = read(get_file_path(__file__, 'pddl/temporal.pddl' if temporal else 'pddl/domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'pddl/stream.pddl'))
    constant_map = {}

    stream_map = {
        #'test-cfree': from_test(get_test_cfree(element_bodies)),
        'sample-move': get_wild_transit_gen_fn(robots, obstacles, element_from_index, grounded_elements,
                                            partial_orders=partial_orders, collisions=collisions, bar_only=bar_only, **kwargs),
        'sample-print': get_wild_place_gen_fn(robots, obstacles, element_from_index, grounded_elements,
                                              partial_orders=partial_orders, collisions=collisions, bar_only=bar_only, **kwargs),
        'test-stiffness': from_test(test_stiffness),
    }

    init = []
    # robot = robots[0]
    # init = [
    #     ('Robot', robot),
    #     ('Conf', robot, np.array(INITIAL_CONF)),
    #     ('AtConf', robot, np.array(INITIAL_CONF)),
    #     ('CanMove', robot),
    # ]
    # if transit:
    #     init.append(('Move',))
    for name, conf in initial_confs.items():
        robot = index_from_name(robots, name)
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
            ('Printed', e),
        ])

    goal_literals = []
    # if return_home:
    #     goal_literals.extend(('AtConf', r, q) for r, q in initial_confs.items())
    goal_literals.extend(('Removed', e) for e in remaining)
    goal = And(*goal_literals)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

##################################################

def solve_pddlstream(robots, obstacles, element_from_index, grounded_elements, connectors,
                     collisions=True, disable=False, max_time=30, bar_only=False, algorithm='incremental', debug=False, **kwargs):
    pddlstream_problem = get_pddlstream(robots, obstacles, element_from_index, grounded_elements, connectors,
                                        collisions=collisions, bar_only=bar_only, **kwargs)
    print('Init:', pddlstream_problem.init)
    print('Goal:', pddlstream_problem.goal)
    print('='*10)

    # creates unique free variable for each output during the focused algorithm
    # (we have an additional search step that initially "shares" outputs, but it doesn't do anything in our domain)
    stream_info = {
        'sample-print': StreamInfo(PartialInputs(unique=True)),
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
            solution = solve_incremental(pddlstream_problem, verbose=True, planner=planner, max_time=600,
                                        max_planner_time=300, debug=debug)
        elif algorithm == 'focused':
            solution = solve_focused(pddlstream_problem, max_time=max_time, stream_info=stream_info,
                                     effort_weight=None, unit_efforts=True, unit_costs=False, # TODO: effort_weight=None vs 0
                                     max_skeletons=None, bind=True, max_failures=0,  # 0 | INF
                                     planner=planner, max_planner_time=60, debug=debug, reorder=False, verbose=True,
                                     initial_complexity=1)
        else:
            raise NotImplementedError(algorithm)
    pr.disable()
    # pstats.Stats(pr).sort_stats('cumtime').print_stats(10)

    print_solution(solution)
    plan, _, certificate = solution
    print('-'*10)
    # print('certificate: ', certificate)
    # preimage facts: the facts that support the returned plan
    # TODO: post-process by calling planner again
    # TODO: could solve for trajectories conditioned on the sequence
    return plan

##################################################

def stripstream(robot, obstacles, bar_struct, **kwargs):
    robots = [robot]
    saver = WorldSaver()
    element_from_index = bar_struct.get_element_from_index()
    grounded_elements = bar_struct.get_grounded_bar_keys()
    element_from_index = bar_struct.get_element_from_index()

    plan = solve_pddlstream(robots, obstacles, element_from_index, grounded_elements, **kwargs)
    # plan = solve_serialized(robots, obstacles, node_points, element_bodies, ground_nodes, layer_from_n, **kwargs)

    data = {}
    if plan is None:
        return None, data

    if has_gui():
        saver.restore()
        #label_nodes(node_points)
        commands = [action.args[-1] for action in reversed(plan) if action.name == 'print']
        trajectories = flatten_commands(commands)

        elements = recover_sequence(trajectories, element_from_index)
        endpts_from_element = bar_struct.get_axis_pts_from_element()
        draw_ordered(elements, endpts_from_element)
        wait_if_gui('Ready to simulate trajectory.')

        display_trajectories(trajectories, time_step=0.02)

    return None, data

###############################################################

def get_wild_place_gen_fn(robots, obstacles, element_from_index, grounded_elements, partial_orders=[], collisions=True, **kwargs):
    robot = robots[0]
    end_effector = EndEffector(robot, ee_link=link_from_name(robot, EE_LINK_NAME),
                               tool_link=link_from_name(robot, TOOL_LINK_NAME),
                               visual=False, collision=True)
    pick_gen_fn = get_place_gen_fn(end_effector, element_from_index, obstacles, verbose=False, precompute_collisions=True, collisions=collisions, **kwargs)
    # TODO gen_fn dict

    def wild_gen_fn(_, element):
        for command, in pick_gen_fn(element):
            q1 = np.array(command.start_conf)
            q2 = np.array(command.end_conf)
            outputs = [(q1, q2, command)]
            facts = [('Collision', command, e2) for e2 in command.colliding] if collisions else []
            print('facts:', facts)
            yield WildOutput(outputs, facts)
    return wild_gen_fn

def get_wild_transit_gen_fn(robots, obstacles, element_from_index, grounded_elements, partial_orders=[], collisions=True, bar_only=False, **kwargs):
    # end_effector = EndEffector(robot, ee_link=link_from_name(robot, EE_LINK_NAME),
    #                            tool_link=link_from_name(robot, TOOL_LINK_NAME),
    #                            visual=False, collision=True)
    # pick_gen_fn = get_place_gen_fn(end_effector, element_from_index, obstacles, verbose=False, precompute_collisions=True, collisions=collisions, **kwargs)

    # def wild_gen_fn(_, start_conf, end_conf):
    def wild_gen_fn(robot_name, q, current_command):
        transit_start_conf = INITIAL_CONF if not bar_only else BAR_INITIAL_CONF
        assert norm(q - current_command.start_conf) < 1e-8, 'norm {}'.format(norm(q - current_command.start_conf))
        robot = index_from_name(robots, robot_name)
        traj = compute_motion(robot, obstacles, element_from_index,
                       [], transit_start_conf, current_command.start_conf, attachments=[],
                       collisions=collisions, max_time=INF, smooth=100, bar_only=bar_only, **kwargs)

        assert norm(q - np.array(traj.end_conf)) < 1e-8, 'norm {}'.format(norm(q - np.array(traj.end_conf)))

        # traj = compute_motion(robot, obstacles, element_from_index, [],
        #                       transit_start_conf, current_command.start_conf,
        #                       collisions=collision, attachments=current_command.trajectories[0].attachments,
        #                       max_time=max_time - elapsed_time(start_time), bar_only=bar_only)
        # for t, in pick_gen_fn(element):
        #     outputs = [(t,)]
        #     facts = [('Collision', t, e2) for e2 in t.colliding] if collisions else []
        #     yield WildOutput(outputs, facts)
        # traj = MotionTrajectory(robot, ik_joints, [start_conf, end_conf], tag='transit2place')
        command = Command([traj])
        outputs = [(command,)]
        facts = []
        yield WildOutput(outputs, facts)
    return wild_gen_fn

def test_stiffness(fluents=[]):
    assert all(fact[0] == 'printed' for fact in fluents)
    # TODO: to use the non-skeleton focused algorithm, need to remove the negative axiom upon success
    # elements = {fact[1] for fact in fluents}
    return True
