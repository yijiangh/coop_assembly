import heapq
import random
import time
from collections import namedtuple
from termcolor import cprint
import numpy as np

import os, sys
from pybullet_planning import INF, get_movable_joints, get_joint_positions, randomize, has_gui, \
    remove_all_debug, wait_for_user, elapsed_time, implies, LockRenderer, EndEffector, link_from_name, \
    set_joint_positions, get_relative_pose, WorldSaver, set_renderer, apply_alpha, RED, wait_for_duration, wait_if_gui

from coop_assembly.data_structure.utils import MotionTrajectory
from coop_assembly.geometry_generation.utils import outgoing_from_edges

from coop_assembly.planning.visualization import draw_element, color_structure, label_elements
# from coop_assembly.planning.stream import get_bar_grasp_gen_fn, get_place_gen_fn, get_pregrasp_gen_fn
from coop_assembly.planning.utils import flatten_commands, Command, check_connected, notify
# from coop_assembly.planning.motion import compute_motion, EE_INITIAL_POINT, EE_INITIAL_EULER, compute_motions
# from coop_assembly.planning.robot_setup import INITIAL_CONF # , TOOL_LINK_NAME, EE_LINK_NAME
# from coop_assembly.planning.stiffness import create_stiffness_checker, test_stiffness

from .parsing import parse_2D_truss
from .heuristics import get_heuristic_fn
from .stream import get_2d_place_gen_fn
from .robot_setup import INITIAL_CONF

PAUSE_UPON_BT = 0
MAX_REVISIT = 5

Node = namedtuple('Node', ['action', 'state'])

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

def regression(robot, tool_from_ee, obstacles, problem, partial_orders=[],
               heuristic='z', max_time=INF, backtrack_limit=INF, revisit=False, bar_only=False,
               collision=True, stiffness=True, motions=True, lazy=True, checker=None, fem_element_from_bar_id=None,
               verbose=False, chosen_bars=None, debug=False, teleops=False, **kwargs):
    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = INITIAL_CONF
    # if not bar_only else np.concatenate([EE_INITIAL_POINT, EE_INITIAL_EULER])

    # element_from_index, grounded_elements, _, connectors = \
    #     unpack_structure(bar_struct, chosen_bars=chosen_bars, scale=METER_SCALE, color=apply_alpha(RED,0.1))
    element_from_index, connectors, grounded_elements = parse_2D_truss(problem)
    # if stiffness and (checker is None or fem_element_from_bar_id is None):
    #     checker, fem_element_from_bar_id = create_stiffness_checker(bar_struct, verbose=False)

    heuristic_fn = get_heuristic_fn(robot, problem, heuristic) #, checker=checker, fem_element_from_bar_id=fem_element_from_bar_id, forward=False)

    # def get_2d_place_gen_fn(end_effector, element_from_index, fixed_obstacles, collisions=True,
    #     max_attempts=IK_MAX_ATTEMPTS, max_grasp=GRASP_MAX_ATTEMPTS, allow_failure=False, verbose=False, teleops=False):
    place_gen_fn = get_2d_place_gen_fn(robot, tool_from_ee, element_from_index, obstacles, collisions=collision,
        verbose=debug, allow_failure=True, teleops=teleops, precompute_collisions=False)

    # TODO: partial ordering

    # TODO: allow choice of config
    final_conf = initial_conf
    all_elements = frozenset(element_from_index)
    final_printed = all_elements
    queue = []
    visited = {final_printed: Node(None, None)}

    outgoing_from_element = outgoing_from_edges(partial_orders)
    print(outgoing_from_element)
    def add_successors(printed, command):
        num_remaining = len(printed) - 1
        # assert 0 <= num_remaining
        for element in randomize(printed):
            # if not (outgoing_from_element[element] & printed) and implies(is_ground(element, ground_nodes), only_ground):
            # print('E:{}, outgoing: {}, printed:{}, intersect: {}'.format(element, outgoing_from_element[element], printed, outgoing_from_element[element] & printed))
            # input()
            if not (outgoing_from_element[element] & printed):
                visits = 0
                bias = heuristic_fn(printed, element)
                priority = (num_remaining, bias, random.random())
                heapq.heappush(queue, (visits, priority, printed, element, command))

    # print('connectors: ', connectors)
    # print('grounded_elements: ', grounded_elements)
    # print('all_elements: ', all_elements)
    # * connectivity & stiffness constraint checking
    if check_connected(connectors, grounded_elements, all_elements):
        # if (not stiffness or test_stiffness(bar_struct, final_printed, checker=checker, fem_element_from_bar_id=fem_element_from_bar_id)):
        final_command = Command([MotionTrajectory(robot, joints, [final_conf])])
        add_successors(final_printed, final_command)
        # else:
        #     cprint('The completed state not stiff!', 'yellow')
    else:
        cprint('The completed state not connected to the ground!', 'yellow')
        if debug:
            print('connectors: ', connectors)
            print('grounded_elements: ', grounded_elements)
            print('all_elements: ', all_elements)

    # * preview the precomputed heuristic on elements
    # if has_gui():
    #     sequence = sorted(final_printed, key=lambda e: heuristic_fn(final_printed, e, conf=None), reverse=True)
    #     remove_all_debug()
    #     draw_ordered(sequence, node_points)
    #     wait_for_user()

    plan = None
    min_remaining = len(all_elements)
    num_evaluated = max_backtrack = place_failures = transit_failures = transfer_failures = stiffness_failures = 0
    while queue and (elapsed_time(start_time) < max_time): #  and check_memory(): #max_memory):
        visits, priority, printed, element, current_command = heapq.heappop(queue)
        num_remaining = len(printed)
        num_evaluated += 1
        backtrack = num_remaining - min_remaining

        # if verbose:
        print('#'*10)
        print('Iteration: {} | Best: {}/{} | Printed: {} | Element: {} | Time: {:.3f} | BT : {} | Max BT: {}'.format(
            num_evaluated, min_remaining, len(all_elements), len(printed), element, elapsed_time(start_time), backtrack, max_backtrack))
        next_printed = printed - {element}

        if backtrack > max_backtrack:
            max_backtrack = backtrack
            # * debug visualize
            # draw_action(axis_pts_from_element, next_printed, element)
            notify('BT increased. Remaining: {} | place_failures {}, transit_failures {}, transfer_failures {}, stiffness_failures {}'.format(list(printed | {element}), place_failures, transit_failures, transfer_failures, stiffness_failures))
            if PAUSE_UPON_BT:
                remove_all_debug()
                set_renderer(enable=True)
                color_structure(element_from_index, printed, next_element=element, built_alpha=0.6)
                label_elements({k : element_from_index[k] for k in list(printed | {element})})
                # print('Blues are the remaining ones, green is the current one, blacks are the already removed ones.')
                cprint('BT increased. Remaining: {} | place_failures {}, transit_failures {}, transfer_failures {}, stiffness_failures {}'.format(
                    list(printed | {element}), place_failures, transit_failures, transfer_failures, stiffness_failures), 'cyan')
                # wait_for_duration(5)
                wait_if_gui()
                set_renderer(enable=False)

        if backtrack_limit < backtrack:
            cprint('backtrack {} exceeds limit {}, exit.'.format(backtrack, backtrack_limit), 'red')
            raise KeyboardInterrupt
            # break # continue

        # if RECORD_QUEUE:
        #     snapshot_state(queue_data, reason='queue_history', queue_log_cnt=QUEUE_COUNT)

        # * constraint checking
        if next_printed in visited:
            continue
        if not check_connected(connectors, grounded_elements, next_printed):
            continue
        # if stiffness and not test_stiffness(bar_struct, next_printed, checker=checker, fem_element_from_bar_id=fem_element_from_bar_id):
        #     if verbose:
        #         cprint('>'*5, 'red')
        #         cprint('Stiffness failure', 'red')
        #     stiffness_failures += 1
        #     continue

        if revisit and visits < MAX_REVISIT:
            heapq.heappush(queue, (visits + 1, priority, printed, current_command))

        with WorldSaver():
            command, = next(place_gen_fn(element, printed=next_printed, diagnosis=debug))
        if command is None:
            if verbose:
                cprint('<'*5, 'red')
                cprint('Place planning failure.', 'red')
            place_failures += 1
            continue

        # TODO: use pick conf
        # transit_start_conf = command.end_conf
        # transfer_start_conf = initial_conf
        # if motions and not lazy:
        #     # if not(bar_only and len(printed) == len(all_elements)):
        #     # if bar_only, skip to final conf transit motion
        #     with WorldSaver():
        #         transit_traj = compute_motion(command.end_robot, obstacles, element_from_index, printed,
        #                                       command.end_conf, transfer_start_conf,
        #                                       collisions=collision, attachments=[],
        #                                       max_time=max_time - elapsed_time(start_time), bar_only=bar_only, debug=debug)
        #     if transit_traj is None:
        #         transit_failures += 1
        #         if verbose:
        #             cprint('%'*5, 'red')
        #             cprint('Transit planning failure.', 'yellow')
        #         continue
        #     transit_traj.tag = 'transit'
        #     command.trajectories.append(transit_traj)

        #     with WorldSaver():
        #         transfer_traj = compute_motion(command.start_robot, obstacles, element_from_index, next_printed,
        #                                        transfer_start_conf, command.start_conf,
        #                                        collisions=collision, attachments=command.trajectories[0].attachments,
        #                                        max_time=max_time - elapsed_time(start_time), bar_only=bar_only, debug=debug)
        #     if transfer_traj is None:
        #         transfer_failures += 1
        #         if verbose:
        #             cprint('%'*5, 'red')
        #             cprint('Transfer planning failure.', 'yellow')
        #         continue
        #     transfer_traj.tag = 'transfer'
        #     command.trajectories.insert(0, transfer_traj)

        if num_remaining < min_remaining:
            min_remaining = num_remaining
            if verbose : cprint('New best: {}'.format(num_remaining), 'green')

        visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs
        if not next_printed:
            min_remaining = 0
            commands = retrace_commands(visited, next_printed, reverse=True)
            plan = flatten_commands(commands)

            # # * return-to-start transit
            # if motions and not lazy:
            #     transit_traj = compute_motion(plan[0].robot, obstacles, element_from_index, frozenset(),
            #                                  initial_conf, plan[0].start_conf,
            #                                  collisions=collision, attachments=command.trajectories[0].attachments,
            #                                  max_time=max_time - elapsed_time(start_time), bar_only=bar_only, debug=debug)
            #     if transit_traj is None:
            #         plan = None
            #         transit_failures += 1
            #     else:
            #         plan.insert(0, transit_traj)
            # TODO: lazy
            # if motions and lazy:
            #     with WorldSaver():
            #         plan = compute_motions(robot, obstacles, element_from_index, initial_conf, plan,
            #                                collisions=collision, max_time=max_time - elapsed_time(start_time),
            #                                bar_only=bar_only, debug=debug)
            break
            # if plan is not None:
            #     break

        add_successors(next_printed, command)

    data = {
        #'memory': get_memory_in_kb(), # May need to update instead
        'search_time' : elapsed_time(start_time),
        'num_evaluated': num_evaluated,
        'min_remaining': min_remaining,
        'max_backtrack': max_backtrack,
        'place_failures': place_failures,
        # 'stiffness_failures': stiffness_failures,
        # 'transit_failures': transit_failures,
        # 'transfer_failures': transfer_failures,
    }
    return plan, data
