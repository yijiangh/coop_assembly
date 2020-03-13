import heapq
import random
import time
from collections import namedtuple
from termcolor import cprint

# from pddlstream.utils import outgoing_from_edges
from pybullet_planning import INF, get_movable_joints, get_joint_positions, randomize, has_gui, \
    remove_all_debug, wait_for_user, elapsed_time, implies, LockRenderer, EndEffector, link_from_name

from coop_assembly.help_functions import METER_SCALE
from coop_assembly.planning import draw_element, check_connected
from coop_assembly.planning import TOOL_LINK_NAME, EE_LINK_NAME
from coop_assembly.planning.stream import get_goal_pose_gen_fn, get_bar_grasp_gen_fn, get_ik_gen_fn, get_pregrasp_gen_fn
from .utils import flatten_commands, Command

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
               max_time=INF, backtrack_limit=INF, revisit=False,
               collision=True, stiffness=True, motions=True, lazy=True, checker=None, **kwargs):
    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, joints)

    axis_pts_from_element = bar_struct.get_axis_pts_from_element()
    element_bodies = bar_struct.get_element_bodies()
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

    goal_pose_gen_fn = get_goal_pose_gen_fn(element_from_index)
    grasp_gen = get_bar_grasp_gen_fn(element_from_index, reverse_grasp=True, safety_margin_length=0.005)
    ik_gen = get_ik_gen_fn(end_effector, element_from_index, obstacles, collision=collision, verbose=True) #max_attempts=n_attempts

    final_conf = initial_conf # TODO: allow choice of config
    final_printed = all_elements
    queue = []
    visited = {final_printed: Node(None, None)}

    # outgoing_from_element = outgoing_from_edges(partial_orders)
    def add_successors(printed, conf):
        # only_ground = printed <= ground_elements
        num_remaining = len(printed) - 1
        # assert 0 <= num_remaining
        for element in randomize(printed):
            visits = 0
            priority = (num_remaining, random.random())
            heapq.heappush(queue, (visits, priority, printed, element, conf))
            # if not (printed): # and implies(is_ground(element, ground_nodes), only_ground):
            #     for directed in get_directions(element):
            #         visits = 0
            #         # bias = heuristic_fn(printed, directed, conf)
            #         # priority = (num_remaining, bias, random.random())
            #         priority = (num_remaining, random.random())
            #         heapq.heappush(queue, (visits, priority, printed, directed, conf))

    if check_connected(connectors, grounded_elements, all_elements):
    # and (not stiffness or test_stiffness(extrusion_path, element_from_id, final_printed, checker=checker)):
        add_successors(final_printed, final_conf)

    # if has_gui():
    #     sequence = sorted(final_printed, key=lambda e: heuristic_fn(final_printed, e, conf=None), reverse=True)
    #     remove_all_debug()
    #     draw_ordered(sequence, node_points)
    #     wait_for_user()

    plan = None
    min_remaining = len(all_elements)
    num_evaluated = max_backtrack = extrusion_failures = transit_failures = stiffness_failures = 0
    while queue and (elapsed_time(start_time) < max_time): #  and check_memory(): #max_memory):
        visits, priority, printed, element, current_conf = heapq.heappop(queue)
        num_remaining = len(printed)
        backtrack = num_remaining - min_remaining
        max_backtrack = max(max_backtrack, backtrack)
        if backtrack_limit < backtrack:
            break # continue
        num_evaluated += 1

        print('#'*10)
        print('Iteration: {} | Best: {} | Printed: {} | Element: {} | Time: {:.3f} | Visit: {}'.format(
            num_evaluated, min_remaining, len(printed), element, elapsed_time(start_time), visits))
        next_printed = printed - {element}

        if revisit and visits < MAX_REVISIT:
            heapq.heappush(queue, (visits + 1, priority, printed, element, current_conf))
        # next_nodes = compute_printed_nodes(ground_nodes, next_printed)

        # debug visualize
        # draw_action(axis_pts_from_element, next_printed, element)
        # if 3 < backtrack + 1:
        #    remove_all_debug()
        #    set_renderer(enable=True)
        #    draw_model(next_printed, node_points, ground_nodes)
        #    wait_for_user()

        if next_printed in visited and visits == 0:
            continue
        if not check_connected(connectors, grounded_elements, next_printed):
            cprint('>'*5, 'red')
            cprint('Connectivity failure', 'red')
            continue

        grasp, = next(grasp_gen(element))
        if grasp is None:
            cprint('$'*5, 'red')
            cprint('Pregrasp planning failure.', 'red')
            continue
        world_pose, = next(goal_pose_gen_fn(element))
        command, = next(ik_gen(element, world_pose, grasp, printed=next_printed))

        if command is None:
            cprint('<'*5, 'red')
            cprint('Pick planning failure.', 'red')
            extrusion_failures += 1
            continue
        # if motions and not lazy:
        #     motion_traj = compute_motion(robot, obstacles, element_bodies, printed,
        #                                  command.end_conf, current_conf, collisions=collisions,
        #                                  max_time=max_time - elapsed_time(start_time))
        #     if motion_traj is None:
        #         transit_failures += 1
        #         continue
        #     command.trajectories.append(motion_traj)

        if num_remaining < min_remaining:
            min_remaining = num_remaining
            cprint('New best: {}'.format(num_remaining), 'green')
            #if has_gui():
            #    # TODO: change link transparency
            #    remove_all_debug()
            #    draw_model(next_printed, node_points, ground_nodes)
            #    wait_for_duration(0.5)

        visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs
        if not next_printed:
            min_remaining = 0
            commands = retrace_commands(visited, next_printed, reverse=True)
            plan = flatten_commands(commands)

            # * return to start config transit
            # if motions and not lazy:
            #     motion_traj = compute_motion(robot, obstacles, element_bodies, frozenset(),
            #                                  initial_conf, plan[0].start_conf, collisions=collisions,
            #                                  max_time=max_time - elapsed_time(start_time))
            #     if motion_traj is None:
            #         plan = None
            #         transit_failures += 1
            #     else:
            #         plan.insert(0, motion_traj)
            # if motions and lazy:
            #     plan = compute_motions(robot, obstacles, element_bodies, initial_conf, plan,
            #                            collisions=collisions, max_time=max_time - elapsed_time(start_time))
            # break
            if plan is not None:
                break
        add_successors(next_printed, command.start_conf)

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