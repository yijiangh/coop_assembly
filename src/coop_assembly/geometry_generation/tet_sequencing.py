import time
import heapq
from collections import defaultdict
from itertools import combinations
import random
from collections import namedtuple

from compas.geometry import distance_point_point, distance_point_line, distance_point_plane
from compas.geometry import is_coplanar
from coop_assembly.help_functions.shared_const import INF, EPS
from coop_assembly.help_functions import tet_surface_area, tet_volume, distance_point_triangle


def adjacent_from_edges(edges):
    """get a dict for graph connectivity {node_id : neighnor node ids}

    Parameters
    ----------
    edges : a list of 2-int-tuples
        [(node_1, node_2), ...]

    Returns
    -------
    undirected_edges : dict
        {node_id : neighnor node ids}
    """
    undirected_edges = defaultdict(set)
    for v1, v2 in edges:
        undirected_edges[v1].add(v2)
        undirected_edges[v2].add(v1)
    return undirected_edges

def compute_distance_from_grounded_node(elements, node_points, ground_node_ids):
    """Use the dijkstra algorithm to compute the graph distance from each node to the grounded nodes.
    Graph edge cost is set to be the Euclidean distance between two vertices.

    Parameters
    ----------
    elements : list of 2-int-tuples
        [(node_1, node_2), ...]
    node_points : list of points
        [[x,y,z], ...]
    ground_node_ids : list of int
        grounded node indices

    Returns
    -------
    cost_from_node : dict
        {node_id (int) : cost (float)}
    """
    neighbors = adjacent_from_edges(elements)
    edge_costs = {tuple(edge) : distance_point_point(node_points[edge[0]], node_points[edge[1]]) \
                  for edge in elements}
    # undirected edges

    edge_costs.update({edge[::-1]: distance for edge, distance in edge_costs.items()})

    cost_from_node = {}
    queue = []
    for node in ground_node_ids:
        cost = 0
        cost_from_node[node] = cost
        heapq.heappush(queue, (cost, node))

    while queue:
        cost1, node1 = heapq.heappop(queue)
        if cost_from_node[node1] < cost1:
            continue
        for node2 in neighbors[node1]:
            cost2 = cost1 + edge_costs[node1, node2]
            if cost2 < cost_from_node.get(node2, INF): # if no key found, return INF
                cost_from_node[node2] = cost2
                heapq.heappush(queue, (cost2, node2))
    return cost_from_node

def point2point_shortest_distance_tet_sequencing(points, cost_from_node):
    """generate a tet sequence by adding node one by one based on cost_from_node (ascending order)
    The base triangle for each node is chosen based on its Euclidean distance to the already added nodes (closest three).

    Parameters
    ----------
    points : a list of 3-float lists
        [description]
    cost_from_node : dict
        {node_id : cost}

    Returns
    -------
    tet_node_ids : list of 4-int lists
        [[base_tri node_id_0, base_tri node_id_1, base_tri node_id_2, newly added node_id], ...]
    """
    # TODO: add a penalty cost for coplanar nodes?
    added_pt_ids = []
    tet_node_ids = []
    cost_from_node = {int(k) : v for k, v in cost_from_node.items()}
    for pt_id, score in sorted(cost_from_node.items(), key=lambda item: item[1]):
        # if pt_id not in start_tri_ids and len(added_pt_ids) >= 3:
        if len(added_pt_ids) >= 3:
            dists = [distance_point_point(points[int(pt_id)], points[prev_node]) for prev_node in added_pt_ids]
            # use distance to previous nodes to sort previous nodes' indices
            sorted_added_ids = [added_id for dist, added_id in sorted(zip(dists, added_pt_ids)) \
                if added_id != pt_id]
            assert len(sorted_added_ids) >= 3
            # take the top three nodes
            closest_node_ids = sorted_added_ids[0:3]
            tet_node_ids.append((closest_node_ids, pt_id))
        assert pt_id not in added_pt_ids, 'pt_id: {} already in previously added pts: {}'.format(
            pt_id, added_pt_ids)
        added_pt_ids.append(pt_id)
    return tet_node_ids

#######################################

SearchState = namedtuple('SearchState', ['action', 'state'])

def compute_candidate_nodes(all_nodes, grounded_nodes, built_nodes):
    if grounded_nodes <= built_nodes:
        nodes = built_nodes | set(grounded_nodes) # union
    else:
        # build grounded ones first
        nodes = grounded_nodes
    return {node for node in set(all_nodes) - set(built_nodes)} # if_buildable()

def retrace_sequence(visited, current_state, horizon=INF):
    # command = ((triangle__node_ids), new_node_id)
    command, prev_state = visited[current_state]
    if (prev_state is None) or (horizon == 0):
        # tracing reaches the end
        return []
    previous_tet_ids = retrace_sequence(visited, prev_state, horizon=horizon-1)
    return previous_tet_ids + [command]

#######################################

PT2TRI_SEARCH_HEURISTIC = {
    'point2triangle_distance', 'tet_surface_area', 'tet_volume',
}

# TODO: minimize number of crossings
def get_pt2tri_search_heuristic_fn(points, penalty_cost=2.0, heuristic='tet_surface_area'):
    assert penalty_cost >= 1.0, 'penalty cost should be bigger than 1.0, heuristic is computed by score *= penalty_cost'
    print('hey')
    def h_fn(built_nodes, node_id, built_triangles):
        # return (bias, chosen triangle node ids)
        # lower bias will be dequed first
        # iterate through all existing triangles and return the minimal cost one
        dist_to_tri = []
        for tri in list(built_triangles):
            tri_node_pts = [points[tri_id] for tri_id in list(tri)]
            tet_pts = tri_node_pts + [points[node_id]]
            score = 0.0
            if heuristic == 'point2triangle_distance':
                score = distance_point_triangle(points[node_id], tri_node_pts)
            elif heuristic == 'tet_surface_area':
                score = tet_surface_area(tet_pts)
            elif heuristic == 'tet_volume':
                vol = tet_volume(tet_pts)
                score = vol if vol else penalty_cost
            else:
                raise NotImplementedError('point2triangle search heuristic ({}) not implemented, the only available ones are: {}'.format(
                heuristic, PT2TRI_SEARCH_HEURISTIC))
            planar_cost = penalty_cost if is_coplanar(tri_node_pts + [points[node_id]]) else 1.0
            score *= planar_cost
            dist_to_tri.append(score)
        sorted_built_triangles = sorted(zip(dist_to_tri, built_triangles), key=lambda pair: pair[0])
        return sorted_built_triangles[0]
    return h_fn

def add_successors(queue, all_nodes, grounded_nodes, heuristic_fn, built_nodes, built_triangles, verbose=False):
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



def point2triangle_tet_sequencing(points, base_triangle_node_ids, heuristic_fn=None, verbose=False):
    all_nodes = frozenset(range(len(points)))
    ground_nodes = frozenset(base_triangle_node_ids)
    assert len(ground_nodes) == 3, 'the grounded nodes need to form a triangle.'
    heuristic_fn = heuristic_fn or get_pt2tri_search_heuristic_fn(points)

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
            tet_ids = retrace_sequence(visited, next_built_nodes)
            break

        # * continue to the next search level, add candidates to queue
        add_successors(queue, all_nodes, ground_nodes, heuristic_fn, next_built_nodes, next_built_triangles, verbose=verbose)
    for i in range(len(tet_ids)):
        tri_node_ids, node_id = tet_ids[i]
        tet_ids[i] = (list(tri_node_ids), node_id)
    if verbose: print('Resulting tet_ids: {}'.format(tet_ids))
    return tet_ids
