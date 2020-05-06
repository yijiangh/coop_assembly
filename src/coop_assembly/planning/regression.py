import heapq
import random
import time
from collections import namedtuple
from termcolor import cprint
import numpy as np

import os, sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.extend([
    os.path.join(here, 'pddlstream/'),
])
from pddlstream.utils import outgoing_from_edges

from pybullet_planning import INF, get_movable_joints, get_joint_positions, randomize, has_gui, \
    remove_all_debug, wait_for_user, elapsed_time, implies, LockRenderer, EndEffector, link_from_name, \
    set_joint_positions

from coop_assembly.help_functions import METER_SCALE, create_bar_flying_body
from coop_assembly.data_structure.utils import MotionTrajectory
from .visualization import draw_element
from .stream import get_goal_pose_gen_fn, get_bar_grasp_gen_fn, get_place_gen_fn, get_pregrasp_gen_fn
from .utils import flatten_commands, Command, check_connected
from .motion import compute_motion, BAR_INITIAL_POINT, BAR_INITIAL_EULER
from .robot_setup import INITIAL_CONF, TOOL_LINK_NAME, EE_LINK_NAME

MAX_REVISIT = 5

Node = namedtuple('Node', ['action', 'state'])

def draw_action(node_points, printed, element):
    """printed elements are drawn green, current element drawn red
    """
    if not has_gui():
        return []
    with LockRenderer():
        remove_all_debug()
        handles = [draw_element(node_points, element, color=(1, 0, 0))]
        handles.extend(draw_element(node_points, e, color=(0, 1, 0)) for e in printed)
    wait_for_user()
    return handles

##################################################

def retrace_commands(visited, current_state, horizon=INF, reverse=False):
    command, prev_state = visited[current_state]
    if (prev_state is None): # or (horizon == 0): # TODO: why horizon
        return []
    prior_commands = retrace_commands(visited, prev_state, horizon=horizon-1, reverse=reverse)
    if reverse:
        return [command] + prior_commands
    return prior_commands + [command]

##################################################

def regression(robot, obstacles, bar_struct, partial_orders=[],
               max_time=INF, backtrack_limit=INF, revisit=False, bar_only=False,
               collision=True, stiffness=True, motions=True, lazy=True, checker=None, verbose=True, **kwargs):
    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = INITIAL_CONF if not bar_only else np.concatenate([BAR_INITIAL_POINT, BAR_INITIAL_EULER])

    # axis_pts_from_element = bar_struct.get_axis_pts_from_element()
    # element_bodies = bar_struct.get_element_bodies()
    element_from_index = bar_struct.get_element_from_index()
    all_elements = frozenset(element_from_index)
    grounded_elements = bar_struct.get_grounded_bar_keys()

    contact_from_connectors = bar_struct.get_connectors(scale=METER_SCALE)
    connectors = list(contact_from_connectors.keys())

    end_effector = EndEffector(robot, ee_link=link_from_name(robot, EE_LINK_NAME),
                               tool_link=link_from_name(robot, TOOL_LINK_NAME),
                               visual=False, collision=True)

    # if checker is None:
    #     checker = create_stiffness_checker(extrusion_path, verbose=False) # if stiffness else None
    # heuristic_fn = get_heuristic_fn(robot, extrusion_path, heuristic, checker=checker, forward=False)
    # TODO: partial ordering
    pick_gen_fn = get_place_gen_fn(end_effector, element_from_index, obstacles, collisions=collision, verbose=False, bar_only=bar_only,\
        precompute_collisions=False, allow_failure=True)

    # TODO: allow choice of config
    final_conf = initial_conf
    final_printed = all_elements
    queue = []
    visited = {final_printed: Node(None, None)}

    outgoing_from_element = outgoing_from_edges(partial_orders)
    def add_successors(printed, command):
        num_remaining = len(printed) - 1
        # assert 0 <= num_remaining
        for element in randomize(printed):
            # if not (outgoing_from_element[element] & printed) and implies(is_ground(element, ground_nodes), only_ground):
            if not (outgoing_from_element[element] & printed):
                visits = 0
                priority = (num_remaining, random.random())
                heapq.heappush(queue, (visits, priority, printed, element, command))

    if check_connected(connectors, grounded_elements, all_elements):
    # and (not stiffness or test_stiffness(extrusion_path, element_from_id, final_printed, checker=checker)):
        final_command = Command([MotionTrajectory(robot, joints, [final_conf])])
            #if not bar_only \
            # else Command([MotionTrajectory(None, None, [final_conf])])
        add_successors(final_printed, final_command)

    # if has_gui():
    #     sequence = sorted(final_printed, key=lambda e: heuristic_fn(final_printed, e, conf=None), reverse=True)
    #     remove_all_debug()
    #     draw_ordered(sequence, node_points)
    #     wait_for_user()

    plan = None
    min_remaining = len(all_elements)
    num_evaluated = max_backtrack = extrusion_failures = transit_failures = stiffness_failures = 0
    while queue and (elapsed_time(start_time) < max_time): #  and check_memory(): #max_memory):
        visits, priority, printed, element, current_command = heapq.heappop(queue)
        num_remaining = len(printed)
        backtrack = num_remaining - min_remaining
        max_backtrack = max(max_backtrack, backtrack)
        if backtrack_limit < backtrack:
            break # continue
        num_evaluated += 1

        if verbose:
            print('#'*10)
            print('Iteration: {} | Best: {} | Printed: {} | Element: {} | Time: {:.3f} | BT : {} | Visit: {}'.format(
                num_evaluated, min_remaining, len(printed), element, elapsed_time(start_time), backtrack, visits))
        next_printed = printed - {element}

        # * debug visualize
        # draw_action(axis_pts_from_element, next_printed, element)
        # if 3 < backtrack + 1:
        #    remove_all_debug()
        #    set_renderer(enable=True)
        #    draw_model(next_printed, node_points, ground_nodes)
        #    wait_for_user()

        if next_printed in visited:
            continue
        if not check_connected(connectors, grounded_elements, next_printed):
            if verbose:
                cprint('>'*5, 'red')
                cprint('Connectivity failure', 'red')
            continue

        if revisit and visits < MAX_REVISIT:
            heapq.heappush(queue, (visits + 1, priority, printed, current_command))

        command, = next(pick_gen_fn(element, printed=next_printed))
        if command is None:
            if verbose:
                cprint('<'*5, 'red')
                cprint('Pick planning failure.', 'red')
            extrusion_failures += 1
            continue

        # TODO: use pick conf
        # transit_start_conf = command.end_conf
        transit_start_conf = initial_conf
        if motions and not lazy:
            # transit to the start of last pick conf
            if not(bar_only and len(printed) == len(all_elements)):
                # skip to final conf transit motion if bar_only
                motion_traj = compute_motion(current_command.start_robot, obstacles, element_from_index, printed,
                                             transit_start_conf, current_command.start_conf,
                                             collisions=collision, attachments=current_command.trajectories[0].attachments,
                                             max_time=max_time - elapsed_time(start_time), bar_only=bar_only)
                if motion_traj is None:
                    transit_failures += 1
                    if verbose:
                        cprint('%'*5, 'red')
                        cprint('Transit planning failure.', 'yellow')
                    continue
                command.trajectories.append(motion_traj)

        if num_remaining < min_remaining:
            min_remaining = num_remaining
            if verbose : cprint('New best: {}'.format(num_remaining), 'green')

        visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs
        if not next_printed:
            min_remaining = 0
            commands = retrace_commands(visited, next_printed, reverse=True)
            plan = flatten_commands(commands)

            # * start config transit
            if motions and not lazy:
                motion_traj = compute_motion(plan[0].robot, obstacles, element_from_index, frozenset(),
                                             initial_conf, plan[0].start_conf,
                                             collisions=collision, attachments=command.trajectories[0].attachments,
                                             max_time=max_time - elapsed_time(start_time), bar_only=bar_only)
                if motion_traj is None:
                    plan = None
                    transit_failures += 1
                else:
                    plan.insert(0, motion_traj)
            # TODO: lazy
            # if motions and lazy:
            #     plan = compute_motions(robot, obstacles, element_bodies, initial_conf, plan,
            #                            collisions=collisions, max_time=max_time - elapsed_time(start_time))
            break
            # if plan is not None:
            #     break

        add_successors(next_printed, command)

    if bar_only:
        # reset all element_robots to clear the visual scene
        for e in element_from_index:
            e_robot = element_from_index[e].element_robot
            set_joint_positions(e_robot, get_movable_joints(e_robot), np.zeros(6))

    data = {
        #'memory': get_memory_in_kb(), # May need to update instead
        'num_evaluated': num_evaluated,
        'min_remaining': min_remaining,
        'max_backtrack': max_backtrack,
        'stiffness_failures': stiffness_failures,
        'extrusion_failures': extrusion_failures,
        'transit_failures': transit_failures,
    }
    return plan, data
