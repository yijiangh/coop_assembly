import os, sys
import json
import time
import heapq
import random
import argparse
from itertools import combinations, product
from collections import namedtuple, defaultdict, deque
import numpy as np
from numpy.linalg import norm
from termcolor import cprint
from copy import copy

from compas.geometry import distance_point_point, distance_point_line, distance_point_plane
from compas.geometry import is_coplanar, subtract_vectors
from compas.datastructures import Network

from pybullet_planning import connect, elapsed_time, randomize, wait_if_gui, RED, BLUE, apply_alpha, add_line, draw_circle

from coop_assembly.help_functions.shared_const import INF, EPS
from coop_assembly.help_functions import tet_surface_area, tet_volume, distance_point_triangle, dropped_perpendicular_points
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions.tangents import tangent_from_point_one, planes_tangent_to_cylinder

from coop_assembly.data_structure import OverallStructure, BarStructure
from coop_assembly.geometry_generation.tet_sequencing import SearchState, compute_candidate_nodes
from coop_assembly.geometry_generation.utils import *

from coop_assembly.planning.parsing import get_assembly_path
from coop_assembly.planning.visualization import draw_element, GROUND_COLOR, BACKGROUND_COLOR, SHADOWS, set_camera
from coop_assembly.planning.utils import load_world

sys.path.append(os.environ['PDDLSTREAM_PATH'])
# here = os.path.abspath(os.path.dirname(__file__))
# sys.path.extend([
#     os.path.join(here, 'pddlstream/'),
# ])
from pddlstream.utils import incoming_from_edges

#######################################

def add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn,
                   printed, partial_orders=[], visualize=False):
    incoming_from_element = incoming_from_edges(partial_orders)
    remaining = all_elements - printed
    num_remaining = len(remaining) - 1
    #assert 0 <= num_remaining
    #bias_from_element = {}
    # TODO: print ground first
    for directed in randomize(compute_printable_directed(all_elements, ground_nodes, printed)):
        element = get_undirected(all_elements, directed)
        if not (incoming_from_element[element] <= printed):
            continue
        bias = heuristic_fn(printed, directed)
        priority = (num_remaining, bias, random.random())
        heapq.heappush(queue, (priority, printed, directed))

    # remaining = all_nodes - built_nodes
    # num_remaining = len(remaining) - 1
    # assert 0 <= num_remaining
    # candidate_nodes = compute_candidate_nodes(all_nodes, grounded_nodes, built_nodes)
    # if verbose : print('add successors: candidate nodes: {}'.format(candidate_nodes))
    # for node_id in candidate_nodes: # random.shuffle(list(candidate_nodes):
    #     # compute bias
    #     bias, tri_node_ids = heuristic_fn(built_nodes, node_id, built_triangles)
    #     priority = (num_remaining, bias, random.random())
    #     heapq.heappush(queue, (priority, built_nodes, built_triangles, node_id, tri_node_ids))

def generate_truss_progression(node_points, edges, ground_nodes, radius, heuristic_fn=None, partial_orders=[],
        check_collision=False, viewer=False, verbose=True):
    start_time = time.time()

    # * need to generate a visiting sequence for all the edges
    # heuristic: z
    # constraint: connectivity
    all_elements = frozenset(edges)
    ground_nodes = frozenset(ground_nodes)
    assert len(ground_nodes) == 3, 'the grounded nodes need to form a triangle.'
    heuristic_fn = heuristic_fn or get_search_heuristic_fn(node_points, edges, ground_nodes, forward=True, heuristic='z')

    initial_printed = frozenset()
    queue = []
    visited = {initial_printed : SearchState(None, None)}
    if check_connected(ground_nodes, all_elements):
        add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn,
                       initial_printed, partial_orders=partial_orders)
    else:
        cprint('full structure not grounded!', 'yellow')

    plan = None
    min_remaining = len(all_elements)
    num_evaluated = 0
    max_time = 10
    while queue and elapsed_time(start_time) < max_time:
        bias, printed, directed = heapq.heappop(queue)
        element = get_undirected(all_elements, directed)

        num_remaining = len(all_elements) - len(printed)
        backtrack = num_remaining - min_remaining
        # max_backtrack = max(max_backtrack, backtrack)
        # if backtrack_limit < backtrack:
        #     break # continue
        num_evaluated += 1
        if num_remaining < min_remaining:
            min_remaining = num_remaining

        next_printed = printed | {element}
        assert check_connected(ground_nodes, next_printed)

        if verbose:
            print('Iteration: {} | Min Remain: {} | Printed: {}/{} | Node: {}'.format(
                num_evaluated, min_remaining, len(printed), len(all_elements), element))

        # * check constraint
        if (next_printed in visited):
            if verbose: print('State visited before: {}'.format(next_printed))
            continue

        # * record history
        visited[next_printed] = SearchState(element, printed)
        if all_elements <= next_printed:
            min_remaining = 0
            plan = retrace_sequence(visited, next_printed)
            break

        # * continue to the next search level, add candidates to queue
        add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn,
                       next_printed, partial_orders=partial_orders)

    cprint('plan: {}'.format(plan), 'green')
    # TODO serialize plan and reparse

    # * convert axis lines into a BarStructure
    return generate_truss_from_points(node_points, ground_nodes, plan, radius)

##########################################
# forming bar structure
def generate_truss_from_points(node_points, ground_nodes, edge_seq, radius):
    printed = set()
    all_elements = set(edge_seq)
    node_neighbors = get_node_neighbors(edge_seq)
    visited_nodes = set(ground_nodes)
    print('>'*10)
    # the actual bar axis endpts are index by using the element's
    # corresponding edge (n1, n2) mapped into the node_points
    bar_from_elements = {}

    for _, element in enumerate(edge_seq):
        # next_printed = printed | {element}
        # unprinted = all_elements - next_printed
        n0, n1 = element
        n_neighbors = [list(set(node_neighbors[n0]) & printed), \
                       list(set(node_neighbors[n1]) & printed)]
        print('------')
        print('visited node: ', visited_nodes)
        print('printed: ', printed)
        print('n0 neighbors: ', set(node_neighbors[n0]))
        print('n1 neighbors: ', set(node_neighbors[n1]))
        print('node_neighbors: ', n_neighbors)

        # if n0 in visited_nodes and n1 not in visited_nodes:
        #     # * grounded node & no existing element neighbor at that node
        #     # simply add a new bar
        #     print('grounded! 0')
        # elif n0 not in visited_nodes and n1 in visited_nodes:
        #     print('grounded! 1')
        # elif n0 in visited_nodes and n1 in visited_nodes:

        for i in range(2):
            # fill in empty tuple for product
            if len(n_neighbors[i]) == 0:
                n_neighbors[i] = [()]
            elif len(n_neighbors[i]) == 1:
                n_neighbors[i] = [n_neighbors[i]]
            elif len(n_neighbors[i]) >= 2:
                n_neighbors[i] = list(combinations(n_neighbors[i], 2))

        # each of these should be segmented into 2 pairs
        neighbor_pairs = list(product(n_neighbors[0], n_neighbors[1]))
        cprint(neighbor_pairs, 'yellow')

        for contact_bars in randomize(neighbor_pairs):
            new_axis_endpts = compute_tangent_bar(bar_from_elements, node_points, element, contact_bars, radius)
            cprint('new axis pt: {}'.format(new_axis_endpts), 'cyan')
            if new_axis_endpts:
                # convert mil to meter
                h = draw_element({0 : map(lambda x : 1e-3*x, new_axis_endpts)}, 0)
                wait_if_gui()
                break

        bar_from_elements[element] = new_axis_endpts
        visited_nodes |= set([n0, n1])
        printed = printed | {element}

    b_struct = BarStructure()
    return b_struct

def compute_tangent_bar(bar_from_elements, node_points, element, in_contact_bars, radius):
    reverse = False
    contact_bars = copy(in_contact_bars)
    if len(in_contact_bars[1]) < len(in_contact_bars[0]):
        reverse = True
        contact_bars = copy(in_contact_bars[::-1])
    assert(len(contact_bars[1])<=2)
    cprint('contact_bars: {}'.format(contact_bars), 'blue')

    new_point = None
    if len(contact_bars[0]) == 0:
        new_point = node_points[element[0]] if not reverse else node_points[element[1]]

    axis_vector = None
    if len(contact_bars[0]) == 0 and len(contact_bars[1]) == 0:
        axis_endpts = (node_points[element[0]], node_points[element[1]])
        return axis_endpts

    elif len(contact_bars[0]) == 0 and len(contact_bars[1]) == 1:
        assert len(contact_bars[1]) > 0
        # point tangent to a cylinder
        contact_e = contact_bars[1][0]
        # print('contact_e', contact_e)
        # print([bar_from_elements[contact_e][0][i] for i in range(3)])
        # print([bar_from_elements[contact_e][1][i] for i in range(3)])

        bar_base_point = bar_from_elements[contact_e][0]
        bar_vector = bar_from_elements[contact_e][1] - bar_from_elements[contact_e][0]
        candidate_tan_planes = planes_tangent_to_cylinder(bar_base_point, bar_vector, new_point, 2*radius)
        if candidate_tan_planes is not None:
            axis_vector = np.array(randomize(candidate_tan_planes)[0][0])
            new_contact_pt, _ = dropped_perpendicular_points(new_point, new_point+axis_vector,
                                                bar_from_elements[contact_e][0], bar_from_elements[contact_e][1])
            return (new_point, np.array(new_contact_pt))

    elif len(contact_bars[0]) == 0 and len(contact_bars[1]) == 2:
        # one is floating or bare-grounded, use `first_tangent` strategy
        # intersecting tangents planes from the point to the two cylinders
        contact_e = contact_bars[1]
        for tangent_side in randomize(range(4)):
            axis_vector = np.array(tangent_from_point_one(bar_from_elements[contact_e[0]][0],
                                                          bar_from_elements[contact_e[0]][1] - bar_from_elements[contact_e[0]][0],
                                                          bar_from_elements[contact_e[1]][0],
                                                          bar_from_elements[contact_e[1]][1] - bar_from_elements[contact_e[1]][0],
                                                          new_point, 2*radius, 2*radius, tangent_side)[0])
            if axis_vector is not None:
                break
        if axis_vector is not None:
            # axis_vector, new_bar_len, pts_b1, pts_b2 = compute_new_bar_length(axis_vector, None, new_point, b_v1_1, b_v1_2, b_struct)
            new_contact_pt1, _ = np.array(dropped_perpendicular_points(new_point, new_point+axis_vector,
                                                bar_from_elements[contact_e[0]][0], bar_from_elements[contact_e[0]][1]))
            new_contact_pt2, _ = np.array(dropped_perpendicular_points(new_point, new_point+axis_vector,
                                                bar_from_elements[contact_e[1]][0], bar_from_elements[contact_e[1]][1]))
            axis_endpts = (new_point, new_point + axis_vector * max([norm(new_contact_pt1-new_point), norm(new_contact_pt2-new_point)]))

    elif len(contact_bars[0])==1 and len(contact_bars[1])==1:
        # deg 1 - deg 1 (4 configs)
        # ? uncovered
        raise NotImplementedError
    elif len(contact_bars[0])==1 and len(contact_bars[1])==2:
        # deg 1 - deg 2 (2x4 configs)
        # use `second_tangent` strategy
        raise NotImplementedError
    elif len(contact_bars[0])==2 and len(contact_bars[1])==2:
        # deg 2 - deg 2
        # use `third_tangent` strategy
        raise NotImplementedError

    return axis_endpts

    # vec_x, vec_y, vec_z = calculate_coord_sys(new_axis_end_pts, pt_mean)

    # b_v0 = b_struct.add_bar(0, new_axis_end_pts, "tube", (25.0, 2.0), vec_z, radius=radius)

    # # b_struct.vertex[b_v0].update({"index_sol":[sol_id]})
    # # b_struct.vertex[b_v0].update({"mean_point":pt_mean})

    # # * update contact point into BarS's edges
    # b_struct.connect_bars(b_v0, b_v1_1)
    # b_struct.connect_bars(b_v0, b_v1_2)

    # dpp_1 = compute_contact_line_between_bars(b_struct, b_v0, b_v1_1)
    # dpp_2 = compute_contact_line_between_bars(b_struct, b_v0, b_v1_2)

    # k_1 = list(b_struct.edge[b_v0][b_v1_1]["endpoints"].keys())[0]
    # k_2 = list(b_struct.edge[b_v0][b_v1_2]["endpoints"].keys())[0]
    # b_struct.edge[b_v0][b_v1_1]["endpoints"].update({k_1:(dpp_1[0], dpp_1[1])})
    # b_struct.edge[b_v0][b_v1_2]["endpoints"].update({k_2:(dpp_2[0], dpp_2[1])})

    # return b_struct, b_v0, new_axis_end_pts


#######################################
# heuristics

SEARCH_HEURISTIC = {
    'random',
    'z',
}

# TODO: minimize number of crossings
def get_search_heuristic_fn(node_points, edges, grounded_nodes, forward=True, penalty_cost=2.0, heuristic='z'):
    all_elements = frozenset(edges)
    sign = +1 if forward else -1

    def h_fn(printed, directed):
        # lower bias will be dequed first
        # iterate through all existing triangles and return the minimal cost one
        # Queue minimizes the statistic
        element = get_undirected(all_elements, directed)

        if heuristic == 'random':
            pass
        elif heuristic == 'z':
            return sign * compute_z_distance(node_points, element)
        else:
            raise NotImplementedError('truss-gen search heuristic ({}) not implemented, the only available ones are: {}'.format(
            heuristic, SEARCH_HEURISTIC))

    return h_fn

def retrace_sequence(visited, current_state, horizon=INF):
    # command = ((triangle__node_ids), new_node_id)
    command, prev_state = visited[current_state]
    if (prev_state is None) or (horizon == 0):
        # tracing reaches the end
        return []
    previous_tet_ids = retrace_sequence(visited, prev_state, horizon=horizon-1)
    return previous_tet_ids + [command]

#############################################################
def gen_truss(problem, viewer=False, radius=3.17, write=False, **kwargs):
    # radius in mm
    problem_path = get_assembly_path(problem)
    with open(problem_path) as json_file:
        data = json.load(json_file)
        cprint('Parsed from : {}'.format(problem_path), 'green')

    net = Network.from_data(data)

    # TODO waiting for compas update to use ordered dict for nodes
    # node_points, edges = net.to_nodes_and_edges()
    node_points = [np.array([net.node[v][c] for c in ['x', 'y', 'z']]) for v in range(net.number_of_nodes())]
    edges = [e for e in net.edges()]
    ground_nodes = [v for v, attr in net.nodes(True) if attr['fixed'] == True]

    print('parsed edges from to_node_and_edges: {}'.format(edges))
    # print('parsed edges from net.edges(): {}'.format(edges2))
    print('parsed node_points in Py3: {}'.format(node_points))
    print('parsed grounded_nodes in Py3: {}'.format(ground_nodes))
    # print('parsed lines in Py3: {}'.format(net.to_lines()))

    # if not verbose:
    #     # used when rpc call is made to get around stdout error
    #     sys.stdout = open(os.devnull, 'w')
    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    fixed_obstacles, robot = load_world()
    set_camera(node_points)

    for e in edges:
        p1 = node_points[e[0]] * 1e-3
        p2 = node_points[e[1]] * 1e-3
        add_line(p1, p2, color=apply_alpha(BLUE, 0.3), width=0.5)
    for v in ground_nodes:
        draw_circle(node_points[v]*1e-3, 0.01)

    b_struct = generate_truss_progression(node_points, edges, ground_nodes, radius, heuristic_fn=None,
        check_collision=False, viewer=False, verbose=True)

    if write:
        export_structure_data(b_struct.data, net.data, radius=radius, **kwargs)

#############################################################

HERE = os.path.abspath(os.path.dirname(__file__))
FILE_DIR = os.path.join(HERE, '..', '..', '..', 'tests', 'test_data')

def main():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='truss_one_tet_skeleton.json', help='The name of the problem to solve')
    parser.add_argument('-r', '--radius', default=3.17, help='Radius of bars in millimeter')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    parser.add_argument('-wr', '--write', action='store_true', help='Export results')
    # parser.add_argument('-db', '--debug', action='store_true', help='Debug verbose mode')
    args = parser.parse_args()
    print('Arguments:', args)

    gen_truss(args.problem, viewer=args.viewer, radius=args.radius, write=args.write, save_dir=FILE_DIR, file_name=args.problem)

if __name__ == '__main__':
    main()
