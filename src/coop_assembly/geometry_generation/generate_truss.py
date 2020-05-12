import os, sys
import json
import time
import heapq
import random
import argparse
from itertools import combinations
from collections import namedtuple, defaultdict
import numpy as np

from compas.geometry import distance_point_point, distance_point_line, distance_point_plane
from compas.geometry import is_coplanar

from pybullet_planning import connect

from coop_assembly.help_functions.shared_const import INF, EPS
from coop_assembly.help_functions import tet_surface_area, tet_volume, distance_point_triangle
from coop_assembly.data_structure import OverallStructure, BarStructure
from coop_assembly.planning.parsing import get_assembly_path
from coop_assembly.geometry_generation.tet_sequencing import SearchState, compute_candidate_nodes, retrace_sequence

#######################################

def generate_truss_from_points(o_struct, b_struct, radius, points, heuristic_fn=None,
    correct=True, check_collision=False, viewer=False, verbose=True):
    if not verbose:
        # used when rpc call is made to get around stdout error
        sys.stdout = open(os.devnull, 'w')
    connect(use_gui=viewer)

# def point2triangle_tet_sequencing(points, base_triangle_node_ids, heuristic_fn=None, verbose=False):

    all_nodes = frozenset(range(len(points)))
    ground_nodes = frozenset(base_triangle_node_ids)
    assert len(ground_nodes) == 3, 'the grounded nodes need to form a triangle.'
    heuristic_fn = heuristic_fn or get_search_heuristic_fn(points)

    initial_built_nodes = frozenset(ground_nodes)
    initial_built_triangles = set([frozenset(ground_nodes)])
    queue = []
    visited = {initial_built_nodes : SearchState(None, None)}
    add_successors(queue, all_nodes, ground_nodes, heuristic_fn, initial_built_nodes, initial_built_triangles, verbose=verbose)

    min_remaining = len(points)
    num_evaluated = 0
    max_time = 10
    start_time = time.time()
    while queue and (time.time() - start_time < max_time) :
        bias, built_nodes, built_triangles, node_id, tri_node_ids = heapq.heappop(queue)
        num_remaining = len(all_nodes) - len(built_nodes)
        num_evaluated += 1
        if num_remaining < min_remaining:
            min_remaining = num_remaining

        if verbose:
            print('Iteration: {} | Min Remain: {} | Built: {}/{} | Node: {} | Triangle: {}'.format(
                num_evaluated, min_remaining, len(built_nodes), len(all_nodes), node_id, list(tri_node_ids)))

        next_built_nodes = built_nodes | {node_id}
        next_built_triangles = built_triangles | set([frozenset(list(two_ends) + [node_id]) for two_ends in combinations(tri_node_ids, 2)])

        # * check constraint
        if (next_built_nodes in visited):
            if verbose: print('State visited before: {}'.format(next_built_nodes))
            continue

        # * record history
        visited[next_built_nodes] = SearchState((tri_node_ids, node_id), built_nodes)
        if all_nodes <= next_built_nodes:
            min_remaining = 0
            tet_ids = retrace_tet_sequence(visited, next_built_nodes)
            break

        # * continue to the next search level, add candidates to queue
        add_successors(queue, all_nodes, ground_nodes, heuristic_fn, next_built_nodes, next_built_triangles, verbose=verbose)

    for i in range(len(tet_ids)):
        tri_node_ids, node_id = tet_ids[i]
        tet_ids[i] = (list(tri_node_ids), node_id)

    if verbose: print('Resulting tet_ids: {}'.format(tet_ids))
    return tet_ids


#######################################

def add_successors(queue, all_nodes, grounded_nodes, heuristic_fn, built_nodes, verbose=False):
    remaining = all_nodes - built_nodes
    num_remaining = len(remaining) - 1
    assert 0 <= num_remaining
    candidate_nodes = compute_candidate_nodes(all_nodes, grounded_nodes, built_nodes)
    if verbose : print('add successors: candidate nodes: {}'.format(candidate_nodes))
    for node_id in candidate_nodes: # random.shuffle(list(candidate_nodes):
        # compute bias
        bias, tri_node_ids = heuristic_fn(built_nodes, node_id, built_triangles)
        priority = (num_remaining, bias, random.random())
        heapq.heappush(queue, (priority, built_nodes, built_triangles, node_id, tri_node_ids))

SEARCH_HEURISTIC = {
    'random',
}

# TODO: minimize number of crossings
def get_search_heuristic_fn(points, penalty_cost=2.0, heuristic='tet_surface_area'):
    assert penalty_cost >= 1.0, 'penalty cost should be bigger than 1.0, heuristic is computed by score *= penalty_cost'
    print('hey')
    def h_fn(built_nodes, node_id, built_triangles):
        # ? return (bias, chosen node ids)
        # lower bias will be dequed first
        # iterate through all existing triangles and return the minimal cost one
        dist_to_tri = []
        for tri in list(built_triangles):
            # tri_node_pts = [points[tri_id] for tri_id in list(tri)]
            # tet_pts = tri_node_pts + [points[node_id]]
            score = 0.0
            if heuristic == 'random':
                pass
            # elif heuristic == 'point2triangle_distance':
            #     score = distance_point_triangle(points[node_id], tri_node_pts)
            # elif heuristic == 'tet_surface_area':
            #     score = tet_surface_area(tet_pts)
            # elif heuristic == 'tet_volume':
            #     vol = tet_volume(tet_pts)
            #     score = vol if vol else penalty_cost
            else:
                raise NotImplementedError('truss-gen search heuristic ({}) not implemented, the only available ones are: {}'.format(
                heuristic, SEARCH_HEURISTIC))
            # planar_cost = penalty_cost if is_coplanar(tri_node_pts + [points[node_id]]) else 1.0
            # score *= planar_cost
            dist_to_tri.append(score)
        sorted_built_triangles = sorted(zip(dist_to_tri, built_triangles), key=lambda pair: pair[0])
        return sorted_built_triangles[0]
    return h_fn

def retrace_tet_sequence(visited, current_state, horizon=INF):
    # command = ((triangle__node_ids), new_node_id)
    command, prev_state = visited[current_state]
    if (prev_state is None) or (horizon == 0):
        # tracing reaches the end
        return []
    previous_tet_ids = retrace_tet_sequence(visited, prev_state, horizon=horizon-1)
    return previous_tet_ids + [command]

#############################################################
def gen_truss(problem, viewer=False):
    problem_path = get_assembly_path(problem)
    with open(problem_path) as json_file:
        data = json.load(json_file)
    o_struct = OverallStructure.from_data(data)

#############################################################

def main():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='one_tet_skeleton.json', help='The name of the problem to solve')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    parser.add_argument('-wr', '--write', action='store_true', help='Export results')
    parser.add_argument('-db', '--debug', action='store_true', help='Debug verbose mode')
    args = parser.parse_args()
    print('Arguments:', args)

    gen_truss(args.problem, viewer=args.viewer)

if __name__ == '__main__':
    main()
