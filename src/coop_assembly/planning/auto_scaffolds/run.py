#!/usr/bin/env python

import time
import argparse
import os, sys
import numpy as np
import pybullet_planning as pp
from termcolor import cprint
from collections import defaultdict

# load pddlstream path
import coop_assembly.planning.load_pddlstream
from coop_assembly.planning.parsing import get_assembly_path, load_structure, unpack_structure
from coop_assembly.help_functions.shared_const import METER_SCALE

from coop_assembly.planning.utils import compute_z_distance, check_connected, Command
from coop_assembly.planning.visualization import set_camera
from coop_assembly.planning.stiffness import test_stiffness, conmech_model_from_bar_structure, create_stiffness_checker, evaluate_stiffness, draw_conmech_model
from coop_assembly.planning.validator import compute_plan_deformation
from coop_assembly.planning.stiffness import plan_stiffness

from pddlstream.language.stream import StreamInfo, PartialInputs, WildOutput, DEBUG
from pddlstream.language.constants import And, PDDLProblem, Equal, print_plan
from pddlstream.algorithms.downward import set_cost_scale, parse_action
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.utils import INF, Profiler, elapsed_time, read, randomize
from pddlstream.language.constants import print_plan, is_plan

DEFAULT_TRANS_TOL = 0.000115 # 0.001 | 0.000115 | INF

####################################################

def get_wild_remove_element_gen_fn(bar_struct, element_from_index, grounded_elements, connectors, debug=False, use_fluent=True, \
    connectivity=True, stiffness=True, checker_info=(None,None), verbose=False):
    def wild_gen_fn(robot_name, element, fluents=[]):
        next_printed = set()
        for fact in fluents:
            if fact[0] == 'assembled':
                #if fact[1] != element:
                next_printed.add(fact[1])
            else:
                raise NotImplementedError(fact[0])

        assert element in next_printed
        next_printed = next_printed - {element}
        if verbose:
            print('---')
            print('Remove E{} - fluent printed ({}): {}'.format(element, len(next_printed), sorted(next_printed)))

        # # * partial ordering
        # if ordering and (outgoing_from_element[element] & next_printed):
        #     # outgoing elements must be removed BEFORE element
        #     cprint('Ordering fails.', 'red')
        #     return

        # * connectivity
        if connectivity and not check_connected(connectors, grounded_elements, next_printed):
            if verbose: cprint('Connectivity fails.', 'red')
            return

        if stiffness and not test_stiffness(bar_struct, next_printed, checker=checker_info[0],
            fem_element_from_bar_id=checker_info[1], verbose=verbose):
            if verbose: cprint('Stiffness fails.', 'red')
            return

        if debug:
            num_outputs = 2 if use_fluent else 3
            outputs = [num_outputs*(None,)]
            yield outputs # For debugging
            return

        command = Command([])
        grasp = None
        outputs = [(command, grasp)]
        yield outputs
    # end wild_gen_fn
    return wild_gen_fn

def get_wild_add_element_gen_fn(bar_struct, element_from_index, grounded_elements, connectors, debug=False,
    use_fluent=True, connectivity=True, stiffness=True, allow_scaffolds=False, checker_info=(None,None), verbose=False):
    all_elements = set(element_from_index)
    def wild_gen_fn(robot_name, element, fluents=[]):
        next_printed = set()
        scaffold_printed = set()
        for fact in fluents:
            if fact[0] == 'assembled':
                next_printed.add(fact[1])
            else:
                raise NotImplementedError(fact[0])

        assert element not in next_printed
        next_printed = next_printed | {element}
        if verbose:
            print('---')
            print('Add E{} - fluent printed ({}): {}'.format(element, len(next_printed), sorted(next_printed)))

        # # * partial ordering
        # if ordering and (outgoing_from_element[element] & next_printed):
        #     # outgoing elements must be removed BEFORE element
        #     cprint('Ordering fails.', 'red')
        #     return

        # * connectivity, only for original structure for now
        if connectivity and not check_connected(connectors, grounded_elements, next_printed & all_elements):
            if verbose: cprint('Connectivity fails.', 'red')
            return

        if stiffness and not test_stiffness(bar_struct, next_printed, checker=checker_info[0],
            fem_element_from_bar_id=checker_info[1], verbose=verbose):
            if verbose: cprint('Stiffness fails.', 'red')
            return

        if debug:
            num_outputs = 2 if use_fluent else 3
            outputs = [num_outputs*(None,)]
            yield outputs # For debugging
            return

        command = Command([])
        grasp = None
        outputs = [(command, grasp)]
        yield outputs
    return wild_gen_fn

def get_wild_add_scaffold_element_gen_fn(bar_struct, element_from_index, grounded_elements, bar_id_from_scaffolds, debug=False,
    use_fluent=True, connectivity=True, stiffness=True, checker_info=(None,None), verbose=False):
    def wild_gen_fn(robot_name, scaffold_element, fluents=[]):
        next_printed = set()
        for fact in fluents:
            if fact[0] == 'assembled':
                next_printed.add(fact[1])
            else:
                raise NotImplementedError(fact[0])

        if verbose:
            print('---')
            print('Add Scaffold E{} - fluent printed ({}): {}'.format(scaffold_element, len(next_printed), sorted(next_printed)))

        # # * partial ordering
        # if ordering and (outgoing_from_element[element] & next_printed):
        #     # outgoing elements must be removed BEFORE element
        #     cprint('Ordering fails.', 'red')
        #     return

        # * connectivity
        if connectivity and not (bar_id_from_scaffolds[scaffold_element] & next_printed):
            if verbose: cprint('Scaffold connectivity fails.', 'red')
            return

        next_printed = next_printed | {scaffold_element}
        if stiffness and not test_stiffness(bar_struct, next_printed, checker=checker_info[0],
            fem_element_from_bar_id=checker_info[1], verbose=verbose):
            if verbose: cprint('Scaffold stiffness fails.', 'red')
            return

        if debug:
            num_outputs = 2 if use_fluent else 3
            outputs = [num_outputs*(None,)]
            yield outputs # For debugging
            return

        command = Command([])
        grasp = None
        outputs = [(command, grasp)]
        yield outputs
    # end wild_gen_fn
    return wild_gen_fn

####################################################

def compute_scaffold(checker, cm_nodes_from_bar_id):
    # * unpack bar structure data
    elem_ind_from_elem_tag = checker._sc_ins.get_elem_ind_from_elem_tag()
    scaffold_elements = elem_ind_from_elem_tag['scaffold']
    bar_id_from_scaffolds = defaultdict(set)
    for se_id in scaffold_elements:
        se_element = checker.model.elements[se_id]
        for bar_id, cm_node_ids in cm_nodes_from_bar_id.items():
            for cm_node_id in list(cm_node_ids):
                if cm_node_id in se_element.end_node_inds:
                    if bar_id != -1:
                        bar_id_from_scaffolds[se_id].add(bar_id)
    return bar_id_from_scaffolds

def get_pddlstream(bar_struct, element_from_index, grounded_elements, connectors,
        stiffness=True, allow_scaffold=True, forward=True, trans_tol=DEFAULT_TRANS_TOL,
        printed=None, removed=None,
        verbose=True, debug=False):
    printed = printed or set()
    removed = removed or set()
    checker, fem_element_from_bar_id = create_stiffness_checker(bar_struct, presample_scaffolds=allow_scaffold, trans_tol=trans_tol)
    # if connect_pb:
    #     draw_conmech_model(checker.model, fem_element_from_bar_id, elements=None);

    cm_nodes_from_bar_id = defaultdict(set)
    for bar_id, fem_es in fem_element_from_bar_id.items():
        for fem_e_id in fem_es:
            fem_element = checker.model.elements[fem_e_id]
            cm_nodes_from_bar_id[bar_id].update(fem_element.end_node_inds)

    all_scaffolds = set()
    bar_id_from_scaffolds = {}
    if allow_scaffold:
        bar_id_from_scaffolds.update(compute_scaffold(checker, cm_nodes_from_bar_id))
        all_scaffolds.update(bar_id_from_scaffolds)

    # * formulate pddl problem
    formu = 'forward' if forward else 'backward'
    if allow_scaffold:
        formu += '_scaffold'
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    domain_path = os.path.abspath(os.path.join(pwd_path, '..', 'pddl/stiffness/domain_{}.pddl'.format(formu)))
    stream_path = os.path.abspath(os.path.join(pwd_path, '..', 'pddl/stiffness/stream_{}.pddl'.format(formu)))
    print('Domain path: {}\nStream path: {}'.format(domain_path, stream_path))
    domain_pddl = read(domain_path)
    stream_pddl = read(stream_path)

    init = []
    init.extend([
        ('Robot', 'r0'),
    ])

    remaining = set(element_from_index) - removed - printed
    for e in remaining:
        init.extend([
            ('Element', e),
            ('Removed', e) if forward else ('Assembled', e),
        ])
    if allow_scaffold:
        for e in all_scaffolds:
            init.extend([
                ('ScaffoldElement', e),
                ('Removed', e),
            ])

    constant_map = {}
    checker_info = (checker, fem_element_from_bar_id)
    if forward:
        stream_map = {
            'sample-add': get_wild_add_element_gen_fn(bar_struct, element_from_index, grounded_elements, connectors,
                debug=False, stiffness=stiffness, checker_info=checker_info, verbose=False)
        }
    else:
        stream_map = {
            'sample-remove': get_wild_remove_element_gen_fn(bar_struct, element_from_index, grounded_elements, connectors,
                debug=False, stiffness=stiffness, checker_info=checker_info, verbose=False)
        }

    if allow_scaffold:
        stream_map['sample-add-scaffold'] = get_wild_add_scaffold_element_gen_fn(bar_struct, element_from_index,
            grounded_elements, bar_id_from_scaffolds, debug=False,
            use_fluent=True, connectivity=True, stiffness=True, checker_info=checker_info, verbose=False)

    if debug:
        stream_map = DEBUG

    goal_literals = []
    if forward:
        goal_literals.extend(('Assembled', e) for e in remaining)
    else:
        goal_literals.extend(('Removed', e) for e in remaining)

    # TODO require scaffolds to be removed at the end
    # if allow_scaffold:
    #     goal_literals.extend(('Removed', e) for e in all_scaffolds)

    goal = And(*goal_literals)

    # print('Init:', pddlstream_problem.init)
    # print('Goal:', pddlstream_problem.goal)

    pddlstream_problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)
    order_fn = get_order_fn(element_from_index, bar_id_from_scaffolds)

    return pddlstream_problem, order_fn

def get_order_fn(element_from_index, bar_id_from_scaffolds):
    from strips.utils import ha_applicable
    from pddlstream.language.conversion import obj_from_pddl

    def order_fn(state, goal, operators):
        operators = ha_applicable(state, goal, operators)
        actions = [op for op in operators if op.__class__.__name__ == 'Action'] # TODO: repair ugliness
        element_actions = []
        scaffold_actions = []
        height_from_action = {}
        for action in actions:
            name, args = parse_action(action.fd_action.name)
            args = [obj_from_pddl(arg).value for arg in args]
            if name == 'add_element':
                _, _, element, _ = args
                height = np.average(element_from_index[element].axis_endpoints, axis=0)[2]
                element_actions.append(action)
            elif name == 'add_scaffold_element':
                _, _, scaffold, _ = args
                height = np.average(np.vstack([element_from_index[element].axis_endpoints
                                               for element in bar_id_from_scaffolds[scaffold]]), axis=0)[2]
                scaffold_actions.append(action)
            else:
                raise NotImplementedError(name)
            height_from_action[action] = height
        actions = sorted(element_actions, key=height_from_action.__getitem__, reverse=False) + \
                  sorted(scaffold_actions, key=height_from_action.__getitem__, reverse=False)
        #actions = reversed(actions)
        #actions = randomize(actions)
        return actions
    return order_fn

def solve_pddlstream(pddlstream_problem, order_fn, costs=False, debug=False, verbose=True, forward=True):
    set_cost_scale(1)
    success_cost = 0 if costs else INF
    discrete_planner = {
        'search': 'eager', # eager | lazy | hill_climbing
        # lazy might be faster because it performs fewer heuristic evaluations but the solution quality might be lower
        # NOTE(caelan): eager is actually faster here because evaluating heuristic goal is cheap
        'evaluator': 'greedy',
        'heuristic': 'goal', # goal | add | ff
        #'heuristic': ['ff', get_bias_fn(element_from_index)],
        #'successors': 'all', # all | random
        # tiebreaker
        'successors': order_fn,
        # ? ordering scaffolds can come here
        # TODO: confirm that this is working correctly
    }
    profiler = Profiler(field='tottime', num=25)
    profiler.save()
    solution = solve_incremental(pddlstream_problem, max_time=60, planner=discrete_planner,
                                success_cost=success_cost, unit_costs=not costs,
                                max_planner_time=300, debug=debug, verbose=verbose)
    profiler.restore()

    plan, _, facts = solution
    # print('-'*10)
    print_plan(plan)
    assert is_plan(plan)

    # extract assembly plan
    if not forward:
        actions = [action for action in reversed(plan) if action.name == 'remove_element']
    else:
        # actions = [action for action in plan if action.name == 'add_element']
        actions = [action for action in plan if action.name == 'add_element' or action.name == 'add_scaffold_element']
    element_sequence = [ac.args[2] for ac in actions]

    return element_sequence

####################################################

def get_stiffness_deformation_fn(bar_struct, verbose=False, **kwargs):
    from coop_assembly.planning.visualization import draw_element, draw_ordered, draw_elements
    if pp.has_gui():
        # element_from_index, grounded_elements, contact_from_connectors, connectors = \
        #     unpack_structure(bar_struct, chosen_bars=None, scale=METER_SCALE, color=pp.apply_alpha(pp.RED, 0.1))
        set_camera([bar_struct.base_centroid(METER_SCALE)], camera_dist=1.25, scale=1.)
        endpts_from_element = bar_struct.get_axis_pts_from_element(scale=METER_SCALE)
        handles = draw_elements(endpts_from_element, color=pp.GREEN)
        #handles = draw_ordered(sorted(endpts_from_element), endpts_from_element)
        pp.wait_if_gui()

    checker, fem_element_from_bar_id = create_stiffness_checker(bar_struct, presample_scaffolds=False, trans_tol=INF) # INF | DEFAULT_TRANS_TOL
    #checker_info = (checker, fem_element_from_bar_id)
    cache = {}

    def fn(elements):
        if elements not in cache:
            max_displacement = 0.
            if elements:
                deformation = evaluate_stiffness(bar_struct, elements, checker=checker, fem_element_from_bar_id=fem_element_from_bar_id, verbose=verbose, **kwargs)
                if deformation.displacements:
                    max_displacement = max(np.linalg.norm(displacement[:3]) for displacement in deformation.displacements.values())
                else:
                    max_displacement = INF # Not connected to ground
            cache[elements] = max_displacement
        return cache[elements]

    if pp.has_gui():
        elements = frozenset(endpts_from_element)
        #deformation = fn(elements)
        deformation = evaluate_stiffness(bar_struct, elements, checker=checker,
                                         fem_element_from_bar_id=fem_element_from_bar_id, verbose=verbose, **kwargs)
        deformed_endpts_from_element = {element: (p1 + deformation.displacements[element], p2 + deformation.displacements[element])
                                        for element, (p1, p2) in endpts_from_element.item()}
        handles = draw_elements(deformed_endpts_from_element, color=pp.RED)
        pp.wait_if_gui()

    return fn

def search(start_state, goal_test, successor_fn, cost_fn=lambda *args: 1, heuristic_fn=lambda *args: 0, # priority_fn=None,
           operation=max, w=1, anytime=False, max_time=INF):
    #if priority_fn is None:
    priority_fn = lambda g, h: h if w == INF else operation(g, w*h)

    from heapq import heappush, heappop
    from collections import namedtuple
    Node = namedtuple('Node', ['g', 'parent']) # TODO: parent_action

    start_time = time.time()
    start_g = cost_fn(start_state)
    visited = {start_state: Node(start_g, None)}

    def retrace(s):
        if visited[s].parent is None:
            return [s]
        return retrace(visited[s].parent) + [s]

    goal_states = []
    start_h = heuristic_fn(start_state)
    start_p = priority_fn(start_g, start_h)
    queue = [(start_p, start_g, start_h, len(visited), start_state)]
    while queue and (elapsed_time(start_time) < max_time):
        _, current_g, current_h, _, current_state = heappop(queue)
        if visited[current_state].g < current_g:
            continue
        print('Cost: {:.6f} | Heuristic: {:.6f} | Visited: {} | Goal states: {} | Time: {:.3f}'.format(
            current_g, current_h, len(visited), len(goal_states), elapsed_time(start_time)))
        if goal_test(current_state):
            goal_states.append(current_state)
            if not anytime:
                break
            continue
        for next_state in successor_fn(current_state):
            if (next_state in visited) and (visited[next_state].g <= current_g):
                continue # Don't bother computing the cost fn
            next_cost = cost_fn(next_state)
            next_g = operation(current_g, next_cost)  # sum | max
            # print(current_g, next_cost, next_g)
            if (next_state not in visited) or (next_g < visited[next_state].g):
                visited[next_state] = Node(next_g, current_state)
                next_h = heuristic_fn(next_state)
                next_p = priority_fn(next_g, next_h)
                heappush(queue, (next_p, next_g, next_h, len(visited), next_state))

    if not goal_states:
        return None, INF
    goal_state = goal_states[-1]
    plan = retrace(goal_state)
    cost = visited[goal_state].g
    return plan, cost

def plan_elements(bar_struct, backward=False, w=INF, num=1):
    element_from_index, grounded_elements, contact_from_connectors, connectors = \
        unpack_structure(bar_struct, chosen_bars=None, scale=METER_SCALE, color=pp.apply_alpha(pp.RED, 0.1))

    connections = defaultdict(set) # Same as fem_element_from_bar_id
    for pair in connectors:
        for element1, element2 in [pair, reversed(pair)]: # -1 is ground
            connections[element1].add(element2)

    elements = frozenset(element_from_index)
    start_state = frozenset()
    goal_state = elements

    goal_test = lambda state: state == goal_state
    forward_successor_fn = lambda state: (state | {element} for element in randomize(elements - state)
                                          if (element in grounded_elements) or (state & connections[element])) # sorted | randomize
    backward_successor_fn = lambda state: (state - {element} for element in randomize(state)) # TODO: check connectivity

    operation = max
    cost_fn = get_stiffness_deformation_fn(bar_struct)
    zero_cost_fn = lambda state: 0 # TODO: lazy planning

    bucket_size = 1 # 1 | 5 | INF
    forward_heuristic_fn = lambda state: int(len(goal_state - state) / bucket_size)
    backward_heuristic_fn = lambda state: int(len(state) / bucket_size)

    successor_fn = forward_successor_fn
    heuristic_fn = forward_heuristic_fn
    if backward:
        start_state, goal_state = goal_state, start_state
        successor_fn = backward_successor_fn
        heuristic_fn = backward_heuristic_fn

    # TODO: hill climbing
    # TODO: lookahead by planning k steps in the future
    # TODO: local search
    profiler = Profiler(field='cumtime', num=25) # cumtime | tottime
    profiler.save()
    solutions = []
    for i in range(num):
        # TODO: prevent repeated sequences
        # TODO: cache evaluations
        print('{}/{}'.format(i, num))
        solution = search(start_state, goal_test, successor_fn,
                          cost_fn=cost_fn,
                          #cost_fn=zero_cost_fn,
                          heuristic_fn=heuristic_fn, operation=operation, w=w)
        plan, cost = solution
        print('Sequence: {}'.format(plan))
        print('Cost: {:.6f}'.format(cost))
        cost = INF if plan is None else operation(cost_fn(state) for state in plan)
        print('Cost: {:.6f}'.format(cost))
        #if plan is not None:
        solutions.append((plan, cost))
        if plan is not None:
            sequence = [state2 - state1 for state1, state2 in zip(plan[:-1], plan[1:])]
            print(sequence)

    print([cost for _, cost in solutions])
    profiler.restore()
    if not solutions:
        return None, INF
    best_solution = min(solutions, key=lambda pair: pair[1])
    print(best_solution[1])
    return best_solution

def plan_pyplanners(elements):
    from strips.operators import Action
    from strips.states import State, PartialState, Literal, STRIPSProblem
    from misc.utils import arg_info, currentframe
    from strips.utils import default_plan, SEARCHES, EVALUATORS, HEURISTICS, SUCCESSORS
    from planner.main import simple_debug, pause_debug
    class Removed(Literal): pass
    class Assembled(Literal): pass
    class Assemble(Action):
        def __init__(self, element):
            super(self.__class__, self).__init__(arg_info(currentframe()))
            self.conditions = {Removed(element)}
            self.effects = {Assembled(element), ~Removed(element)}
            self.cost = 2
            #self.cost = lambda state: 2
    initial = State(Removed(element) for element in elements)
    goal = PartialState(Assembled(element) for element in elements)
    operators = [Assemble(element) for element in elements]
    #problem = STRIPSProblem(initial, goal, operators)
    plan, state_space = default_plan(initial, goal, operators, search='eager', heuristic='goal')

####################################################

def visualize_translations(sequence, translations, trans_tol, title='plan deformation history'):
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(121)
    plt.plot([t[1] for t in translations])
    plt.plot([trans_tol for _ in translations])
    plt.title(title)

    plt.subplot(122)
    plt.plot(sequence)
    plt.title('sequence')
    plt.show()

def verify_plan(bar_struct, element_sequence, trans_tol=DEFAULT_TRANS_TOL, allow_scaffolds=True, verbose=False):
    translations = compute_plan_deformation(bar_struct, element_sequence, presample_scaffolds=allow_scaffolds, verbose=verbose)
    visualize_translations(element_sequence, translations, trans_tol, title='PDDLStream plan deformation history')
    return element_sequence

def progression_search(bar_struct, chosen_elements, trans_tol, verbose=False):
    progression_sequence = plan_stiffness(bar_struct, chosen_elements, trans_tol=trans_tol,
        stiffness=True, heuristic='z', max_time=pp.INF,
        max_backtrack=0, verbose=verbose)
    if progression_sequence:
        translations = compute_plan_deformation(bar_struct, progression_sequence)
        visualize_translations(progression_sequence, translations, trans_tol, title='Progression deformation history')
    return progression_sequence

####################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--connect_pb', action='store_true', help='Enables the viewer during planning (slow!)')
    parser.add_argument('-p', '--problem', default='1_exp_tets.json', # SP_arch
                        help='The name of the problem to solve')
    parser.add_argument('--stiffness', action='store_false', help='Enable stiffness constraint')
    parser.add_argument('--forward', action='store_false', help='Use forward formulation.')
    parser.add_argument('--allow_scaffold', action='store_false', help='Allow scaffolding.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode.')
    parser.add_argument('--trans_tol', type=float, default=DEFAULT_TRANS_TOL, help='translational deformation tolerance')
    parser.add_argument('--progression_search', action='store_true', help='run progression search without scaffolds')
    #
    parser.add_argument('--costs', action='store_true', help='Consider costs in pddlstream.')
    args = parser.parse_args()
    print('Arguments:', args)

    bar_struct, _ = load_structure(args.problem, connect_pb=args.connect_pb, viewer=args.connect_pb)
    if args.connect_pb:
        pp.create_plane(color=pp.WHITE)
        pp.draw_pose(pp.unit_pose())

    # transform model
    new_world_from_base = pp.Pose(point=np.array([0,0,0]))
    world_from_base = pp.Pose(point=bar_struct.base_centroid(METER_SCALE))
    # rotation = Pose(euler=Euler(roll=BASE_ROLL, pitch=BASE_PITCH, yaw=BASE_YAW))
    rotation = pp.unit_pose()
    tf = pp.multiply(new_world_from_base, rotation, pp.invert(world_from_base))
    bar_struct.transform(tf, scale=METER_SCALE)
    bar_struct.generate_grounded_connection()

    plan_elements(bar_struct)
    return

    element_from_index, grounded_elements, contact_from_connectors, connectors = \
        unpack_structure(bar_struct, chosen_bars=None, scale=METER_SCALE, color=pp.apply_alpha(pp.RED, 0.1))
    if args.connect_pb:
        set_camera([bar_struct.base_centroid(METER_SCALE)], camera_dist=1.25, scale=1.)
        pp.wait_if_gui()

    if not args.progression_search:
        pddlstream_problem, order_fn = get_pddlstream(bar_struct, element_from_index, grounded_elements, connectors,
                stiffness=args.stiffness, allow_scaffold=args.allow_scaffold, forward=args.forward, trans_tol=args.trans_tol,
                debug=args.debug, verbose=args.verbose)

        element_sequence = solve_pddlstream(pddlstream_problem, order_fn, costs=args.costs, debug=args.debug,
            verbose=args.verbose, forward=args.forward)

        if element_sequence:
            verify_plan(bar_struct, element_sequence, trans_tol=args.trans_tol, allow_scaffolds=args.allow_scaffold,
                verbose=args.verbose)
    else:
        all_elements = list(element_from_index.keys())
        progression_search(bar_struct, all_elements, args.trans_tol, verbose=args.verbose)

if __name__ == '__main__':
    main()
