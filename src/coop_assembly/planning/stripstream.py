import numpy as np
import cProfile
import pstats

from pddlstream.algorithms.downward import set_cost_scale
from pddlstream.algorithms.focused import solve_focused #, CURRENT_STREAM_PLAN
from pddlstream.algorithms.disabled import process_stream_plan
from pddlstream.language.constants import And, PDDLProblem, print_solution, DurativeAction, Equal
from pddlstream.language.generator import from_test
from pddlstream.language.stream import StreamInfo, PartialInputs, WildOutput
from pddlstream.language.function import FunctionInfo
from pddlstream.utils import read, get_file_path, inclusive_range
from pddlstream.language.temporal import compute_duration, get_end

from pybullet_planning import has_gui, get_movable_joints, get_configuration, set_configuration, WorldSaver, LockRenderer, \
    wait_if_gui, EndEffector, link_from_name, joints_from_names, intrinsic_euler_from_quat, get_links, create_attachment, \
    set_joint_positions, get_links, set_pose
from .stream import get_element_body_in_goal_pose, get_pick_gen_fn, ENABLE_SELF_COLLISIONS, get_pregrasp_gen_fn
from .utils import flatten_commands, recover_sequence, Command
from .visualization import draw_ordered
from .motion import display_trajectories
from .robot_setup import EE_LINK_NAME, TOOL_LINK_NAME, IK_JOINT_NAMES, JOINT_WEIGHTS, RESOLUTION, get_disabled_collisions
from coop_assembly.data_structure.utils import MotionTrajectory
from coop_assembly.help_functions import METER_SCALE

STRIPSTREAM_ALGORITHM = 'stripstream'
ROBOT_TEMPLATE = 'r{}'

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
                   printed=set(), removed=set(),
                   temporal=False, transit=False, return_home=False, checker=None, **kwargs):
    assert not removed & printed
    remaining = set(element_from_index) - removed - printed
    element_obstacles = get_element_body_in_goal_pose(element_from_index, printed)
    obstacles = set(static_obstacles) | element_obstacles

    partial_orders = set()
    initial_confs = {ROBOT_TEMPLATE.format(i): Conf(robot) for i, robot in enumerate(robots)}

    domain_pddl = read(get_file_path(__file__, 'pddl/temporal.pddl' if temporal else 'pddl/domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'pddl/stream.pddl'))
    constant_map = {}

    stream_map = {
        #'test-cfree': from_test(get_test_cfree(element_bodies)),
        # 'sample-move': get_wild_move_gen_fn(robots, obstacles, element_from_index,
        #                                     partial_orders=partial_orders, **kwargs),
        'sample-print': get_wild_print_gen_fn(robots, obstacles, element_from_index, grounded_elements,
                                              partial_orders=partial_orders, **kwargs),
        #'test-stiffness': from_test(test_stiffness),
    }

    init = []
    # if transit:
    #     init.append(('Move',))
    # for name, conf in initial_confs.items():
    #     # robot = index_from_name(robots, name)
    #     # init_node = -robot
    #     # init_node = '{}-q0'.format(robot)
    #     init.extend([
    #         ('Robot', name),
    #         ('Conf', name, conf),
    #         ('AtConf', name, conf),
    #         ('Idle', name),
    #     ])
    init.extend(('Grounded', e) for e in grounded_elements)
    init.extend(('Joined', e1, e2) for e1, e2 in connectors)
    init.extend(('Joined', e2, e1) for e1, e2 in connectors)

    for e in remaining:
        init.extend([
            ('Element', e),
            ('Printed', e),
        ])

    goal_literals = []
    if return_home:
        goal_literals.extend(('AtConf', r, q) for r, q in initial_confs.items())
    goal_literals.extend(('Removed', e) for e in remaining)
    goal = And(*goal_literals)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

##################################################

def solve_pddlstream(robots, obstacles, element_from_index, grounded_elements, connectors,
                     collisions=True, disable=False, max_time=30, bar_only=False, **kwargs):
    pddlstream_problem = get_pddlstream(robots, obstacles, element_from_index, grounded_elements, connectors,
                                        collisions=collisions, disable=disable,
                                        precompute_collisions=True, **kwargs)
    print('Init:', pddlstream_problem.init)
    print('Goal:', pddlstream_problem.goal)

    stream_info = {
        'sample-print': StreamInfo(PartialInputs(unique=True)),
        # 'sample-move': StreamInfo(PartialInputs(unique=True)),
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
        # solution = solve_incremental(pddlstream_problem, planner='add-random-lazy', max_time=600,
        #                             max_planner_time=300, debug=True)
        solution = solve_focused(pddlstream_problem, stream_info=stream_info, max_time=max_time,
                                 effort_weight=None, unit_efforts=True, unit_costs=False, # TODO: effort_weight=None vs 0
                                 max_skeletons=None, bind=True, max_failures=0,  # 0 | INF
                                 planner=planner, max_planner_time=60, debug=True, reorder=False,
                                 initial_complexity=1)

    pr.disable()
    pstats.Stats(pr).sort_stats('cumtime').print_stats(10)

    print_solution(solution)
    plan, _, certificate = solution
    print('certificate all_facts:', certificate.all_facts)
    print('certificate preimage_facts:', certificate.preimage_facts)
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

def get_wild_move_gen_fn(robots, static_obstacles, element_bodies, partial_orders=set(), collisions=True, bar_only=False, **kwargs):
    # incoming_supporters, _ = neighbors_from_orders(partial_orders)

    def wild_gen_fn(name, conf1, conf2, *args):
        is_initial = (conf1.element is None) and (conf2.element is not None)
        is_goal = (conf1.element is not None) and (conf2.element is None)
        if is_initial:
            supporters = []
        elif is_goal:
            supporters = list(element_bodies)
        # else:
        #     supporters = [conf1.element]  # TODO: can also do according to levels
        #     retrace_supporters(conf1.element, incoming_supporters, supporters)
        element_obstacles = {element_bodies[e] for e in supporters}
        obstacles = set(static_obstacles) | element_obstacles
        if not collisions:
            obstacles = set()

        robot = index_from_name(robots, name)
        conf1.assign()
        joints = get_movable_joints(robot)
        joints = joints_from_names(robot, IK_JOINT_NAMES)
        # TODO: break into pieces at the furthest part from the structure

        weights = JOINT_WEIGHTS
        resolutions = np.divide(RESOLUTION * np.ones(weights.shape), weights)
        disabled_collisions = get_disabled_collisions(robot)
        custom_limits = {}

        path = [conf1, conf2]
        # path = plan_joint_motion(robot, joints, conf2.positions, obstacles=obstacles,
        #                          self_collisions=ENABLE_SELF_COLLISIONS, disabled_collisions=disabled_collisions,
        #                          weights=weights, resolutions=resolutions,
        #                          restarts=3, iterations=100, smooth=100)
        if not path:
            return
        path = [conf1.positions] + path[1:-1] + [conf2.positions]
        traj = MotionTrajectory(robot, joints, path)
        command = Command([traj])
        outputs = [(command,)]
        facts = []
        #facts = [('Collision', command, e2) for e2 in command.colliding] if collisions else []
        yield WildOutput(outputs, [('Dummy',)] + facts) # To force to be wild

    return wild_gen_fn

def get_wild_print_gen_fn(robots, static_obstacles, element_from_index, grounded_elements,
                          collisions=True, bar_only=False, **kwargs):
    # TODO: could reuse end-effector trajectories
    # gen_fn_from_robot = {}
    # for robot in robots:
    #     end_effector = EndEffector(robot, ee_link=link_from_name(robot, EE_LINK_NAME),
    #                                tool_link=link_from_name(robot, TOOL_LINK_NAME),
    #                                visual=False, collision=True)
    #     gen_fn_from_robot[robot] = get_pick_gen_fn(end_effector, element_from_index, static_obstacles, collision=collisions, \
    #                                verbose=False, bar_only=bar_only)

    pregrasp_gen_fn = get_pregrasp_gen_fn(element_from_index, static_obstacles, collision=collisions) # max_attempts=max_attempts,

    # def wild_gen_fn(name, element):
    #     # TODO: could cache this
    #     # sequence = [result.get_mapping()['?e'].value for result in CURRENT_STREAM_PLAN]
    #     # index = sequence.index(element)
    #     # printed = sequence[:index]
    #     # TODO: this might need to be recomputed per iteration
    #     robot = index_from_name(robots, name)
    #     for command, in gen_fn_from_robot[robot](element):
    #         q1 = Conf(robot, command.start_conf, element=element)
    #         q2 = Conf(robot, command.end_conf, element=element)
    #         outputs = [(q1, q2, command)]
    #         facts = [('Collision', command, e2) for e2 in command.colliding] if collisions else []
    #         yield WildOutput(outputs, [('Dummy',)] + facts)
    # return wild_gen_fn

    def dummpy_gen_fn(element):
        while True:
            # TODO element_robot
            # q1 = Conf(robots[0], positions=None, element=element)
            body = element_from_index[element].body
            world_pose = element_from_index[element].goal_pose
            pregrasp_poses, = next(pregrasp_gen_fn(element, world_pose, printed=[]))

            element_robot = element_from_index[element].element_robot
            element_joints = get_movable_joints(element_robot)
            element_body_link = get_links(element_robot)[-1]
            attach_conf = np.concatenate([pregrasp_poses[-1][0], intrinsic_euler_from_quat(pregrasp_poses[-1][1])])
            set_joint_positions(element_robot, element_joints, attach_conf)
            set_pose(body, pregrasp_poses[-1])
            attachment = create_attachment(element_robot, element_body_link, body)
            command = Command([MotionTrajectory(element_robot, element_joints,
                               [np.concatenate([p[0], intrinsic_euler_from_quat(p[1])]) for p in pregrasp_poses], \
                               attachments=[attachment], tag='place_approach', element=element)])
            outputs = [(command,)]
            facts = []
            yield WildOutput(outputs, [('Dummy',)] + facts)
    return dummpy_gen_fn
