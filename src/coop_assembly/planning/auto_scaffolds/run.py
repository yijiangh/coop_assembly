#!/usr/bin/env python

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

from pddlstream.utils import read
from pddlstream.language.stream import StreamInfo, PartialInputs, WildOutput, DEBUG
from pddlstream.language.constants import And, PDDLProblem, Equal, print_plan
from pddlstream.algorithms.downward import set_cost_scale, parse_action
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.utils import INF, Profiler
from pddlstream.language.constants import print_plan, is_plan

DEFAULT_TRANS_TOL = 0.000115 # 0.001 | 0.000115

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
    checker, fem_element_from_bar_id = create_stiffness_checker(bar_struct, presample_scaffolds=False, trans_tol=INF) # INF | DEFAULT_TRANS_TOL
    #checker_info = (checker, fem_element_from_bar_id)

    def fn(elements):
        if not elements:
            return 0.
        deformation = evaluate_stiffness(bar_struct, elements, checker=checker, fem_element_from_bar_id=fem_element_from_bar_id, verbose=verbose, **kwargs)
        if not deformation.displacements:
            return INF # Not connected to ground
        max_displacement = max([np.linalg.norm(displacement[:3]) for displacement in deformation.displacements.values()])
        return max_displacement
    return fn

def plan_elements(bar_struct, element_from_index):
    elements = frozenset(element_from_index)
    cost_fn = get_stiffness_deformation_fn(bar_struct)

    # from strips.operators import Action
    # from strips.states import State, PartialState, Literal, STRIPSProblem
    # from misc.utils import arg_info, currentframe
    # from strips.utils import default_plan, SEARCHES, EVALUATORS, HEURISTICS, SUCCESSORS
    # from planner.main import simple_debug, pause_debug
    # class Removed(Literal): pass
    # class Assembled(Literal): pass
    # class Assemble(Action):
    #     def __init__(self, element):
    #         super(self.__class__, self).__init__(arg_info(currentframe()))
    #         self.conditions = {Removed(element)}
    #         self.effects = {Assembled(element), ~Removed(element)}
    #         self.cost = 2
    #         #self.cost = lambda state: 2
    # initial = State(Removed(element) for element in elements)
    # goal = PartialState(Assembled(element) for element in elements)
    # operators = [Assemble(element) for element in elements]
    # #problem = STRIPSProblem(initial, goal, operators)
    # plan, state_space = default_plan(initial, goal, operators, search='eager', heuristic='goal')

    from heapq import heappush, heappop
    from collections import namedtuple
    Node = namedtuple('Node', ['g', 'parent'])

    start_s = frozenset()
    start_g = cost_fn(start_s)
    visited = {start_s: Node(start_g, None)}
    queue = [(start_g, start_s)]

    def retrace(s):
        if visited[s].parent is None:
            return [s]
        return retrace(visited[s].parent) + [s]

    goal_s = None
    while queue:
        current_g, current_s = heappop(queue)
        if visited[current_s].g < current_g:
            continue
        if current_s == elements:
            goal_s = current_s
            #sequence = retrace(current_s)
            break
        for element in (elements - current_s):
            next_s = current_s | {element}
            next_cost = cost_fn(next_s)
            next_g = max(current_g, next_cost) # sum | max
            print(current_g, next_cost, next_g)
            if (next_s not in visited) or (next_g < visited[next_s].g):
                visited[next_s] = Node(next_g, current_s)
                heappush(queue, (next_g, next_s))

    sequence = None if goal_s is None else retrace(goal_s)
    print('Sequence:', sequence)
    if sequence is not None:
        print(visited[elements].g)
        print('Cost: {:.6f}'.format(visited[sequence[-1]].g))
    input()

    return visited

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

    element_from_index, grounded_elements, contact_from_connectors, connectors = \
        unpack_structure(bar_struct, chosen_bars=None, scale=METER_SCALE, color=pp.apply_alpha(pp.RED, 0.1))
    if args.connect_pb:
        set_camera([bar_struct.base_centroid(METER_SCALE)], camera_dist=1.25, scale=1.)

    # plan_elements(bar_struct, element_from_index)
    # return

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
