import os, sys
import json
import time
import heapq
import random
import argparse
from itertools import combinations, product
from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from termcolor import cprint
import pybullet_planning as pp
import matplotlib.pyplot as plt

from compas.datastructures import Network

from pybullet_planning import connect, elapsed_time, randomize, wait_if_gui, RED, BLUE, apply_alpha, \
    add_line, draw_circle, reset_simulation, disconnect, draw_pose, unit_pose, LockRenderer

from coop_assembly.help_functions.shared_const import INF, EPS
from coop_assembly.help_functions.parsing import export_structure_data
from coop_assembly.help_functions.helpers_geometry import calculate_coord_sys, compute_contact_line_between_bars

from coop_assembly.data_structure import BarStructure, GROUND_INDEX
from coop_assembly.geometry_generation.tet_sequencing import SearchState
from coop_assembly.geometry_generation.utils import get_node_neighbors, get_element_neighbors, nodes_from_elements, \
    check_connected, incoming_from_edges, compute_printable, compute_z_distance, values_as_list, contain_directed, directed_index
from coop_assembly.geometry_generation.tangents import closest_point_segments
from coop_assembly.geometry_generation.validate import plot_contact_distance
from coop_assembly.help_functions import create_bar_body

from coop_assembly.planning.parsing import get_assembly_path
from coop_assembly.planning.visualization import draw_element, BACKGROUND_COLOR, SHADOWS, set_camera, \
    label_points, check_model
from coop_assembly import LOGGER

try:
    from multi_tangent.pybeamopt import BeamOpt, BeamOptSolver
except ImportError as err:
    print('multi_tangent not loaded')
    raise err

METHOD_OPTIONS = ['search', 'presolved_assignment']
EDGE_KEY = frozenset
# convert to meter
SCALE = 1e-3

#######################################

def add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn,
                   printed, partial_orders=[], visualize=False):
    incoming_from_element = incoming_from_edges(partial_orders)
    remaining = all_elements - printed
    num_remaining = len(remaining) - 1
    #assert 0 <= num_remaining
    #bias_from_element = {}
    # TODO: print ground first
    for element in randomize(compute_printable(all_elements, ground_nodes, printed)):
        # element = get_undirected(all_elements, directed)
        if not (incoming_from_element[element] <= printed):
            continue
        bias = heuristic_fn(printed, element)
        priority = (num_remaining, bias, random.random())
        heapq.heappush(queue, (priority, printed, element))

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
        check_collision=False, viewer=False, verbose=True, debug=False):
    """ node points assumed to be in millimiter unit
    """
    start_time = time.time()

    # * need to generate a visiting sequence for all the edges
    # heuristic: z
    # constraint: connectivity
    all_elements = frozenset(edges)
    ground_nodes = frozenset(ground_nodes)
    heuristic_fn = heuristic_fn or get_search_heuristic_fn(node_points, edges, ground_nodes, forward=True, heuristic='z')
    # assert len(ground_nodes) == 3, 'the grounded nodes need to form a triangle.'

    initial_printed = frozenset()
    queue = []
    visited = {initial_printed : SearchState(None, None)}
    if check_connected(ground_nodes, all_elements):
        add_successors(queue, all_elements, node_points, ground_nodes, heuristic_fn,
                       initial_printed, partial_orders=partial_orders)
    else:
        cprint('full structure not grounded!', 'yellow')

    # TODO can also do searching and geometric checking at the same time
    plan = None
    min_remaining = len(all_elements)
    num_evaluated = 0
    max_time = 10
    while queue and elapsed_time(start_time) < max_time:
        bias, printed, element = heapq.heappop(queue)
        # element = get_undirected(all_elements, directed)

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

        # TODO: generate bar while searching

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
    return generate_truss_from_points(node_points, ground_nodes, plan, radius, debug=debug)

##########################################
# forming bar structure
def generate_truss_from_points(node_points, ground_nodes, edge_seq, radius, debug=False):
    """build a double_tangent BarStructure from optimized bar endpoints.

    Parameters
    ----------
    node_points : [type]
        [description]
    ground_nodes : [type]
        [description]
    edge_seq : [type]
        list of edges indicating the enumerating sequence

    Returns
    -------
    [type]
        [description]
    """
    printed = set()
    # all_elements = set(edge_seq)
    node_neighbors = get_node_neighbors(edge_seq)
    element_neighbors = get_element_neighbors(edge_seq)
    visited_nodes = set(ground_nodes)

    # for debug viz
    bodies = {}
    handles = {}
    contact_handles = []

    # the actual bar axis endpts are index by using the element's
    # corresponding edge (n1, n2) mapped into the node_points
    # each entry's value (node pt id : axis pt, node pt id : axis pt)
    # ! bar_from_elements[edge] = {node : [x,y,z]}, initialize with the original line graph
    bar_from_elements = {}
    for e in edge_seq:
        bar_from_elements[e] = {
            e[0] : node_points[e[0]],
            e[1] : node_points[e[1]],
            }
    # ! connect_vars[set([edge1, edge2]), node] = 1 or 0
    connect_vars = {}
    contact_pairs = [] # defaultdict(list)
    collision_pairs = [] # defaultdict(list)

    for _, element in enumerate(edge_seq):
        # next_printed = printed | {element}
        # unprinted = all_elements - next_printed
        n0, n1 = element
        n_neighbors = {n0 : list(set(node_neighbors[n0]) & printed), \
                       n1 : list(set(node_neighbors[n1]) & printed)}
        print('------')

        # * find all potential tangent assignment of each node
        for i in [n0, n1]:
            if len(n_neighbors[i]) == 0:
                # this means this node is a valence-one node (no previous bars connected, newly formed cantilevering node)
                n_neighbors[i] = [()]
            elif len(n_neighbors[i]) == 1:
                n_neighbors[i] = [n_neighbors[i]]
            elif len(n_neighbors[i]) >= 2:
                # ! this is for 2-tangent systems for now
                n_neighbors[i] = list(combinations(n_neighbors[i], 2))
        neighbor_pairs = list(product(n_neighbors[n0], n_neighbors[n1]))
        cprint('produced neighnor pairs: {}'.format(neighbor_pairs), 'yellow')

        extension_score = np.inf

        # TODO: iterate through all neighboring pairs and use the pairs that produces the minial neighbor element extension
        # for contact_bars in randomize(neighbor_pairs):
        success = False
        for contact_bars in neighbor_pairs:
            cprint('^'*10)
            new_contact_pairs = []
            new_collision_pairs = []
            assert len(contact_bars) == 2
            for ni, contact_bars_at_node in enumerate(contact_bars):
                contacted_bars = set()
                for e in contact_bars_at_node:
                    if e != ():
                        contacted_bars.add(e)
                new_contact_pairs.extend([(element, e) for e in list(contacted_bars) if contain_directed(contact_pairs, (element, e))])

                no_collision_bars = set(n_neighbors[ni]).difference(contacted_bars)
                new_collision_pairs.extend([(element, e) for e in list(no_collision_bars) if contain_directed(collision_pairs, (element, e))])

            success, bar_from_elements = beam_opt_global(bar_from_elements, node_points, printed | {element}, ground_nodes, 
                                                         contact_pairs + new_contact_pairs, 
                                                         collision_pairs + new_collision_pairs, 
                                                         radius, debug=0, scale=1)
            if success:
                # * update contact and collision assignments
                contact_pairs = contact_pairs + new_contact_pairs, 
                collision_pairs = collision_pairs + new_collision_pairs, 

                # pp.remove_handles(values_as_list(handles))
                # list(map(lambda b : pp.remove_body(b), values_as_list(bodies)))
                new_bodies, new_handles = draw_bar_from_elements(bar_from_elements, radius)
                for nb_eid in new_bodies:
                    if nb_eid in bodies:
                        pp.remove_body(bodies[nb_eid])
                        pp.remove_handle(handles[nb_eid])
                bodies.update(new_bodies)
                handles.update(new_handles)
                wait_if_gui('A new bar is added.')

                break
        else:
            LOGGER.error('No feasible assignmen found.')

        visited_nodes |= set([n0, n1])
        printed = printed | {element}

        # for contact_e in update_element_axis_pts:
        #     if contact_e == element:
        #         continue
        #     c_node = (set(element) & set(contact_e)).pop()
        #     connect_vars[EDGE_KEY({element, contact_e}), c_node] = 1


    ## * compile bars into a BarStructure
    # TODO fix assumptions in BarStructure that everything is in millimeter
    # centroid_pt = np.average(node_points, axis=0)
    b_struct = compose_bar_structure(bar_from_elements, ground_nodes)

    if debug:
        wait_if_gui('Final bar assembly.')

    return b_struct, (bar_from_elements, connect_vars)

def compose_bar_structure(bar_from_elements, ground_nodes, contact_pairs, radius):
    b_struct = BarStructure()

    index_from_element = {}
    for e, pts in bar_from_elements.items():
        axis_endpts = list(pts.values())
        _ , _, vec_z = calculate_coord_sys(axis_endpts, [0,0,0])
        n0, n1 = e
        is_grounded = n0 in ground_nodes or n1 in ground_nodes
        # everything in BarStructure is stored in millimeter
        bar_key = b_struct.add_bar(None, [(1/SCALE * p).tolist() for p in axis_endpts], "tube", None, vec_z, radius=1/SCALE*radius, grounded=is_grounded)
        index_from_element[EDGE_KEY(e)] = bar_key

    # TODO use contact assignment here
    # for e, pts in bar_from_elements.items():
    #     # update contact point into BarS's edges
    #     for ne in element_neighbors[e]:
    for e0t, e1t in contact_pairs:
        e0, e1 = EDGE_KEY(e0t), EDGE_KEY(e1t)
        b_struct.connect_bars(index_from_element[e0], index_from_element[e1])
        contact_pts = compute_contact_line_between_bars(b_struct, index_from_element[e0], index_from_element[e1])
        b_struct.edge[index_from_element[e0]][index_from_element[e1]]["endpoints"].update({0:(list(contact_pts[0]), list(contact_pts[1]))})

    # * add grounded connector
    grounded_bars = list(b_struct.get_grounded_bar_keys())
    for ground_k in grounded_bars:
        b_struct.connect_bars(ground_k, GROUND_INDEX)
        # find the lower pt of the two
        axis_endpts = b_struct.get_bar_axis_end_pts(ground_k)
        if axis_endpts[0][2] > axis_endpts[1][2]:
            axis_endpts = axis_endpts[::-1]
        contact_pts = [axis_endpts[0], axis_endpts[0]-np.array([0,0,radius])]
        b_struct.edge[ground_k][GROUND_INDEX]["endpoints"].update({0:(list(contact_pts[0]), list(contact_pts[1]))})

    element_bodies = b_struct.get_element_bodies(color=apply_alpha(RED, 1.0))
    return b_struct

#############################################################

def beam_opt_global(bar_from_elements, node_points, edges, grounded_nodes, contact_pairs, collision_pairs, radius, scale=1.0, debug=False):
    """Use beam opt to compute nodal positions for given contact and collision assignments

    Parameters
    ----------
    bar_from_elements :
        # ! bar_from_elements[edge] = {node : [x,y,z]}, initialize with the original line graph
    node_points :
        num_nodes x 3 np array for original graph node poisitions
    edges : 
        [(n0,n1),...] for original graph connectivity
    grounded_nodes : 
        list of grounded node inds
    contact_pairs : _type_
        _description_
    collision_pairs : _type_
        _description_
    radius : float
        bar radius
    scale : float, optional
        _description_, by default 1.0
    debug : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # * fill in current bar end point as initial guess
    # initial_vert_points = np.array(beam_opt.vertices[beam_opt.edges])
    # for i, ei in enumerate(edges):
    #     initial_vert_points[i,0] = np.array(bar_from_elements[ei][ei[0]]) * scale
    #     initial_vert_points[i,1] = np.array(bar_from_elements[ei][ei[1]]) * scale
    # beam_opt.end_pts_init_positions = jnp.array(initial_vert_points.flatten())

    # * renumbering the verts and edges for the subassembly
    # vert_old_inds = list(nodes_from_elements(subassembly_edges))
    # local_verts = np.array(node_points)[vert_old_inds]
    local_lines = []
    boundary_vertices = []
    radius_list = []
    new_edge_from_old = {}
    contact_ts_bound = []
    for i, e in enumerate(edges):
        local_lines.append(bar_from_elements[e][e[0]])
        local_lines.append(bar_from_elements[e][e[1]])
        boundary_vertices.append(e[0] in grounded_nodes)
        boundary_vertices.append(e[1] in grounded_nodes)
        radius_list.append(radius)
        new_edge_from_old[e] = i

    for cp in contact_pairs:
        # TODO can restrict further based on initial contact point range
        contact_ts_bound.append([0.1, 0.9])
        contact_ts_bound.append([0.1, 0.9])

    bo_struct = BeamOpt()
    bo_struct.lines = local_lines
    bo_struct.boundary_vertices = boundary_vertices
    bo_struct.radius = radius_list

    contact_inds = []
    for e0,e1 in contact_pairs:
        e0_id, e1_id = directed_index(edges, tuple(e0)), directed_index(edges, tuple(e1))
        assert e0_id is not None and e1_id is not None
        assert e0_id != e1_id
        contact_inds.extend([e0_id, e1_id])
    collision_dists = []
    collision_inds = []
    for e0,e1 in collision_pairs:
        e0_id, e1_id = directed_index(edges, tuple(e0)), directed_index(edges, tuple(e1))
        assert e0_id is not None and e1_id is not None
        assert e0_id != e1_id
        collision_inds.extend([e0_id, e1_id])
        collision_dists.append(2*radius)
    bo_struct.contact_inds = contact_inds
    bo_struct.collision_inds = collision_inds
    bo_struct.collision_dists = collision_dists

    bo_struct.contact_ts_bound = contact_ts_bound

    solver = BeamOptSolver(bo_struct)
    solver.hessian_provided = True
    solver.silence = False
    solver.nlp_tol = 1e-8
    solver.maxtime = 100
    obj, xstar = solver.solve()
    if solver.nStatus != 0:
        LOGGER.error('Knitro status: {} | feasError {} | optError {}'.format(solver.nStatus, solver.feasError, solver.optError))
        return False, bar_from_elements

    # unpack computed x
    for i, e in enumerate(edges):
        bar_from_elements[e][e[0]] = xstar[i*6:i*6+3]
        bar_from_elements[e][e[1]] = xstar[i*6+3:i*6+6]

    # contact_handles = []
    # if True:
    #     # * draw contact lines
    #     neighbors = get_element_neighbors(edges)
    #     for i, ei in enumerate(edges):
    #         for nei in neighbors[ei]:
    #             new_nei = new_edges[new_edge_from_old[nei]]
    #             dpp = closest_point_segments(new_vertices[new_ei[0]], new_vertices[new_ei[1]],
    #                                          new_vertices[new_nei[0]], new_vertices[new_nei[1]])
    #             print('contact len: {:.4f} | target {:.4f}'.format(norm(dpp[0]-dpp[1])*(1/scale), radius*2))
    #             contact_handles.append(pp.add_line(dpp[0], dpp[1], color=pp.apply_alpha(pp.GREEN, 1.0), width=1.5))

    # update_element_axis_pts = {}
    # for i, (ei, new_ei) in enumerate(zip(edges, new_edges)):
    #     update_element_axis_pts[ei] = {
    #         ei[0] : ((1/scale)*new_vertices[new_ei[0],:]).tolist(),
    #         ei[1] : ((1/scale)*new_vertices[new_ei[1],:]).tolist(),
    #         }
    return True, bar_from_elements

def draw_assembly(vertices, edges, radius):
    bodies = []
    for e in edges:
        bodies.append(create_bar_body([vertices[e[0]], vertices[e[1]]], radius, scale=1e-3, color=pp.apply_alpha(pp.RED, 0.2)))
    return bodies

def draw_graph(vertices, edges, support_vertex_indices=None):
    handles = []
    support_vertex_indices = support_vertex_indices or []
    with pp.LockRenderer():
        set_camera(vertices)
        for e in edges:
            n1, n2 = e
            p1 = vertices[n1]
            p2 = vertices[n2]
            handles.append(pp.add_line(p1, p2, color=pp.apply_alpha(pp.BLUE, 0.6), width=0.5))
        for v in support_vertex_indices:
            handles.extend(pp.draw_circle(vertices[v], 0.01))
    return handles

def draw_bar_from_elements(bar_from_elements, radius):
    bodies = {}
    handles = {}
    with LockRenderer():
        for e, node_dict in bar_from_elements.items():
            p1, p2 = list(map(lambda x: np.array(x)*1e-3, node_dict.values()))
            h = pp.add_line(p1, p2, color=pp.apply_alpha(pp.BLACK, 1), width=0.5)
            b = create_bar_body([p1, p2], radius*1e-3, scale=1, color=pp.apply_alpha(pp.RED, 0.9))
            bodies[e] = b
            handles[e] = h
    return bodies, handles

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

    def h_fn(printed, element):
        # lower bias will be dequed first
        # iterate through all existing triangles and return the minimal cost one
        # Queue minimizes the statistic
        # element = get_undirected(all_elements, directed)

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

def compute_multi_tangent(problem_data, radius, viewer=False, debug=False, method='search', **kwargs):
    """[summary]

    Parameters
    ----------
    problem : [type]
        problem name, see `planning.parsing.get_assembly_path`
    viewer : bool, optional
        enable pybullet viewer, by default False
    radius : float, optional
        bar radius in millimeter, by default 10 mm

    Returns
    -------
    [type]
        [description]
    """
    net = Network.from_data(problem_data)
    # everything below this line is in meters
    node_points = [SCALE*np.array([net.node[v][c] for c in ['x', 'y', 'z']]) for v in range(net.number_of_nodes())]
    edges = [e for e in net.edges()]
    ground_nodes = [v for v, attr in net.nodes(True) if attr['fixed'] == True]

    # node_points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 1]])*1e3
    # edges = [(0,1), (1,2)]
    # ground_nodes = [0]

    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    with LockRenderer():
        set_camera(node_points)
        draw_pose(unit_pose(), length=0.01)
        # draw the ideal truss that we want to achieve
        label_points(node_points)
        for e in edges:
            n1, n2 = e
            p1 = node_points[n1]
            p2 = node_points[n2]
            add_line(p1, p2, color=apply_alpha(BLUE, 0.3), width=0.5)
        for v in ground_nodes:
            draw_circle(node_points[v], 0.01)

    if debug:
        wait_if_gui('Target line graph...')

    opt_data = None
    if method == 'search':
        b_struct, (x_vars, z_vars) = generate_truss_progression(node_points, edges, ground_nodes, radius, heuristic_fn=None,
            check_collision=False, viewer=False, verbose=True, debug=debug)

        x_vars_data = []
        for edge, pt_from_node in x_vars.items():
            for v, pt in pt_from_node.items():
                x_vars_data.append({'key' : (tuple(edge), v), 'value' : list(pt)})
        z_vars_data = []
        for ((e1, e2), n), val in z_vars.items():
            z_vars_data.append({'key' : ((tuple(e1), tuple(e2)), n), 'value' : val})
        opt_data = {
            'x_vars' : x_vars_data,
            'z_vars' : z_vars_data,
            }

    elif method == 'presolved_assignment':
        bar_from_elements = {}
        for e in edges:
            bar_from_elements[e] = {
                e[0] : node_points[e[0]],
                e[1] : node_points[e[1]],
                }

        problem_name = problem_data['problem_name'].split('.json')[0]
        assignment_filepath = os.path.join(FILE_DIR, problem_name+'_contact_assignment.json')
        with open(assignment_filepath) as json_file:
            assign_data = json.load(json_file)
            cprint('Contact assignments parsed from : {}'.format(assignment_filepath), 'green')
        contact_pairs = assign_data['contact_pairs'] 
        collision_pairs = assign_data['collision_pairs'] 

        success, bar_from_elements = beam_opt_global(bar_from_elements, node_points, edges, ground_nodes, 
                                                     contact_pairs, collision_pairs, 
                                                     radius, scale=1.0, debug=False)
        assert success
        b_struct = compose_bar_structure(bar_from_elements, ground_nodes, contact_pairs, radius)
    else:
        raise NotImplementedError('Unsupported method : {}'.format(method))

    cprint('Done.', 'green')
    return b_struct, opt_data

#############################################################

HERE = os.path.abspath(os.path.dirname(__file__))
FILE_DIR = os.path.join(HERE, '..', '..', '..', 'tests', 'test_data')

def main():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='truss_one_tet_skeleton.json', help='The name of the problem to solve')
    parser.add_argument('-m', '--method', choices=METHOD_OPTIONS, default='search', help='Radius of bars in meter')
    parser.add_argument('-r', '--radius', default=0.01, help='Radius of bars in meter')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    parser.add_argument('-w', '--write', action='store_true', help='Export results')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug verbose mode')
    parser.add_argument('--subset_bars', nargs='+', default=None, help='Plan for only subset of bar indices.')
    args = parser.parse_args()
    print('Arguments:', args)

    # export_file_name = args.problem.split('.json')[0]
    export_file_name = args.problem
    if 'skeleton' in export_file_name:
        export_file_name = export_file_name.split('_skeleton')[0] + '.json'
    if 'raw_bar_graph' in export_file_name:
        export_file_name = export_file_name.split('_raw_bar_graph')[0] + '.json'

    graph_data = None
    opt_data = None
    problem_path = get_assembly_path(args.problem)
    with open(problem_path) as json_file:
        graph_data = json.load(json_file)
        cprint('Design graph parsed from : {}'.format(problem_path), 'green')
    if 'data' in graph_data:
        graph_data = graph_data['data']
    graph_data['problem_name'] = args.problem

    b_struct, opt_data = compute_multi_tangent(graph_data, args.radius, viewer=args.viewer,
        method=args.method, debug=args.debug)

    fig = plot_contact_distance(b_struct)
    plt.show()
    # if args.write:
    #     fig.savefig(os.path.join(multi_tangent.DATA_DIRECTORY, 'plots', export_file_name + '_bar_structure_dist_plot.png'))
    check_model(b_struct, args.subset_bars, debug=args.debug)

    if args.write:
        export_structure_data(b_struct.to_data(), overall_struct_data=graph_data, save_dir=FILE_DIR,
            file_name=export_file_name, opt_data=opt_data, indent=0)
        # export_structure_data(b_struct.data, net.data, radius=radius, **kwargs)

    reset_simulation()
    disconnect()


if __name__ == '__main__':
    main()
