import heapq
import random
import time
from collections import namedtuple

# from pddlstream.utils import outgoing_from_edges
from pybullet_planning import INF, get_movable_joints, get_joint_positions, randomize, has_gui, \
    remove_all_debug, wait_for_user, elapsed_time, implies, LockRenderer
from coop_assembly.help_functions import METER_SCALE
from coop_assembly.planning import draw_element, check_connected
from .utils import flatten_commands

Node = namedtuple('Node', ['action', 'state'])

def draw_action(node_points, printed, element):
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
               collisions=True, stiffness=True, motions=True, lazy=True, checker=None, **kwargs):

    start_time = time.time()
    joints = get_movable_joints(robot)
    initial_conf = get_joint_positions(robot, joints)
    # element_from_id, node_points, ground_nodes = load_extrusion(extrusion_path)
    # id_from_element = get_id_from_element(element_from_id)

    axis_pts_from_element = bar_struct.get_axis_pts_from_element()
    element_bodies = bar_struct.get_element_bodies()
    all_elements = frozenset(element_bodies)
    grounded_elements = bar_struct.get_grounded_bar_keys()

    contact_from_connectors = bar_struct.get_connectors(scale=METER_SCALE)
    connectors = list(contact_from_connectors.keys())

    end_effector = EndEffector(robot, ee_link=link_from_name(robot, EE_LINK_NAME),
                               tool_link=link_from_name(robot, TOOL_LINK_NAME),
                               visual=False, collision=True)

    # if checker is None:
    #     checker = create_stiffness_checker(extrusion_path, verbose=False) # if stiffness else None

    # print_gen_fn = get_print_gen_fn(robot, obstacles, node_points, element_bodies, ground_nodes,
    #                                 supports=False, precompute_collisions=False,
    #                                 max_directions=MAX_DIRECTIONS, max_attempts=MAX_ATTEMPTS,
    #                                 collisions=collisions, **kwargs)
    # heuristic_fn = get_heuristic_fn(robot, extrusion_path, heuristic, checker=checker, forward=False)

    goal_pose_gen_fn = get_goal_pose_gen_fn(element_from_index)
    grasp_gen = get_bar_grasp_gen_fn(element_from_index, tool_pose=tool_pose, \
        reverse_grasp=True, safety_margin_length=0.02)
    ik_gen = get_ik_gen_fn(end_effector, element_from_index, obstacles, max_attempts=max_attempts, collision=True, \
        epsilon=epsilon, angle=angle,
        pos_step_size=pos_step_size, ori_step_size=ori_step_size)

    final_conf = initial_conf # TODO: allow choice of config
    final_printed = all_elements
    queue = []
    visited = {final_printed: Node(None, None)}

    # outgoing_from_element = outgoing_from_edges(partial_orders)
    def add_successors(printed, conf):
        # only_ground = printed <= ground_elements
        num_remaining = len(printed) - 1
        assert 0 <= num_remaining
        for element in randomize(printed):
            # outgoing_from_element[element] & printed
            priority = (num_remaining, random.random())
            heapq.heappush(queue, (visits, priority, printed, element, conf))
            # if not (printed): # and implies(is_ground(element, ground_nodes), only_ground):
            #     for directed in get_directions(element):
            #         visits = 0
            #         # bias = heuristic_fn(printed, directed, conf)
            #         # priority = (num_remaining, bias, random.random())
            #         priority = (num_remaining, random.random())
            #         heapq.heappush(queue, (visits, priority, printed, directed, conf))

    # if check_connected(ground_nodes, final_printed):
    # # and (not stiffness or test_stiffness(extrusion_path, element_from_id, final_printed, checker=checker)):
    #     add_successors(final_printed, final_conf)

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
        # element = get_undirected(all_elements, directed)
        num_remaining = len(printed)
        backtrack = num_remaining - min_remaining
        max_backtrack = max(max_backtrack, backtrack)
        if backtrack_limit < backtrack:
            break # continue
        num_evaluated += 1

        print('Iteration: {} | Best: {} | Printed: {} | Element: {} | | Time: {:.3f}'.format(
            num_evaluated, min_remaining, len(printed), element, elapsed_time(start_time)))
        next_printed = printed - {element}
        # next_nodes = compute_printed_nodes(ground_nodes, next_printed)

        draw_action(axis_pts_from_element, next_printed, element)
        # if 3 < backtrack + 1:
        #    remove_all_debug()
        #    set_renderer(enable=True)
        #    draw_model(next_printed, node_points, ground_nodes)
        #    wait_for_user()

        # we don't have directionality in assembly: (directed[0] not in next_nodes)
        # if (next_printed in visited) or \
        #    check_connected(connectors, grounded_elements, printed):
        #     continue

        command, = next(print_gen_fn(directed[0], element, extruded=next_printed), (None,))
        if command is None:
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
            print('New best: {}'.format(num_remaining))
            #if has_gui():
            #    # TODO: change link transparency
            #    remove_all_debug()
            #    draw_model(next_printed, node_points, ground_nodes)
            #    wait_for_duration(0.5)

        visited[next_printed] = Node(command, printed) # TODO: be careful when multiple trajs
        if not next_printed:
            min_remaining = 0
            #plan = retrace_trajectories(visited, next_printed, reverse=True)
            commands = retrace_commands(visited, next_printed, reverse=True)
            #commands = optimize_commands(robot, obstacles, element_bodies, extrusion_path, initial_conf, commands,
            #                             motions=motions, collisions=collisions)
            plan = flatten_commands(commands)

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
            # if plan is not None:
            #     break
        add_successors(next_printed, command.start_conf)
        if revisit:
            heapq.heappush(queue, (visits + 1, priority, printed, directed, current_conf))

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
