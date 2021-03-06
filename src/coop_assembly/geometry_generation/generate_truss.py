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

from compas.geometry import distance_point_point, distance_point_line, distance_point_plane, centroid_points
from compas.geometry import is_coplanar, subtract_vectors, angle_vectors
from compas.datastructures import Network

from pybullet_planning import connect, elapsed_time, randomize, wait_if_gui, RED, BLUE, GREEN, BLACK, apply_alpha, \
    add_line, draw_circle, remove_handles, add_segments, BROWN, YELLOW, reset_simulation, disconnect, draw_pose, unit_pose, \
    LockRenderer

from coop_assembly.help_functions.shared_const import INF, EPS
from coop_assembly.help_functions import tet_surface_area, tet_volume, distance_point_triangle, dropped_perpendicular_points
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions.helpers_geometry import find_points_extreme, calculate_coord_sys, compute_local_coordinate_system, \
    bar_sec_verts, compute_contact_line_between_bars

from coop_assembly.data_structure import OverallStructure, BarStructure, GROUND_INDEX
from coop_assembly.geometry_generation.tet_sequencing import SearchState, compute_candidate_nodes
from coop_assembly.geometry_generation.utils import *
from coop_assembly.geometry_generation.tangents import compute_tangent_from_two_lines, lines_tangent_to_cylinder, solve_one_one_tangent, \
    solve_second_tangent, solve_third_tangent

from coop_assembly.planning.parsing import get_assembly_path
from coop_assembly.planning.visualization import draw_element, GROUND_COLOR, BACKGROUND_COLOR, SHADOWS, set_camera, \
    label_points, check_model
from coop_assembly.planning.utils import load_world

METHOD_OPTIONS = ['search', 'shrink', 'bar_network']
EDGE = frozenset
SCALE = 1e-3

########################################
def from_bar_network(problem, debug=False, viewer=False):
    """build a BarStructure from a graph that's already a tangent-contact bar system (not necessarily double-tangent)
    """
    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)

    problem_path = get_assembly_path(problem)
    with open(problem_path) as json_file:
        data = json.load(json_file)
        cprint('Parsed from : {}'.format(problem_path), 'green')

    if 'data' in data:
        data = data['data']
    bar_network = Network.from_data(data)

    b_struct = BarStructure()
    centroid_pt = np.zeros(3)

    index_from_element = {}
    for v, v_data in bar_network.nodes(data=True):
        axis_endpts = v_data['axis_endpoints']
        _ , _, vec_z = calculate_coord_sys(axis_endpts, centroid_pt)
        is_grounded = v_data['fixed']
        radius = v_data['radius']
        bar_key = b_struct.add_bar(v, [p for p in axis_endpts], "tube", None, vec_z, radius=radius, grounded=is_grounded)
        index_from_element[v] = bar_key

    for e in bar_network.edges():
        b_struct.connect_bars(index_from_element[e[0]], index_from_element[e[1]])
        contact_pts = compute_contact_line_between_bars(b_struct, index_from_element[e[0]], index_from_element[e[1]])
        b_struct.edge[index_from_element[e[0]]][index_from_element[e[1]]]["endpoints"].update({0:(list(contact_pts[0]), list(contact_pts[1]))})

    # * add grounded connector
    b_struct.generate_grounded_connection()

    element_bodies = b_struct.get_element_bodies(color=apply_alpha(RED, 0.5))
    if debug:
        wait_if_gui('Final bar assembly.')

    return b_struct

#######################################

def generate_shrinked_truss(node_points, edges, edge_attributes, ground_nodes, radius, debug=False):
    """Directly turn a 3D line graph into truss by shrinking the edges to avoid collision
    """
    b_struct = BarStructure()
    centroid_pt = np.average(node_points, axis=0)

    all_elements = frozenset(edges)
    bar_from_elements = {}
    for e in all_elements:
        n1, n2 = e
        p0 = node_points[n1]
        p1 = node_points[n2]
        delta_p = (p1 - p0) / norm(p1 - p0)
        shrink = 0.0
        if 'shrink' in edge_attributes[e]:
            shrink = edge_attributes[e]['shrink']
        bar_from_elements[e] = {
            n1 : p0 + shrink*delta_p,
            n2 : p1 - shrink*delta_p
            }

    index_from_element = {}
    for e, pts in bar_from_elements.items():
        axis_endpts = list(pts.values())
        _ , _, vec_z = calculate_coord_sys(axis_endpts, centroid_pt)
        is_grounded = e[0] in ground_nodes or e[1] in ground_nodes
        bar_key = b_struct.add_bar(None, [p.tolist() for p in axis_endpts], "tube", None, vec_z, radius=radius, grounded=is_grounded)
        index_from_element[e] = bar_key

    element_neighbors = get_element_neighbors(list(bar_from_elements.keys()))
    for e, pts in bar_from_elements.items():
        # update contact point into BarS's edges
        for ne in element_neighbors[e]:
            b_struct.connect_bars(index_from_element[e], index_from_element[ne])
            contact_pts = compute_contact_line_between_bars(b_struct, index_from_element[e], index_from_element[ne], method='opt')
            b_struct.edge[index_from_element[e]][index_from_element[ne]]["endpoints"].update({0:(list(contact_pts[0]), list(contact_pts[1]))})

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

    element_bodies = b_struct.get_element_bodies(color=apply_alpha(RED, 0.5))
    if debug:
        wait_if_gui('Final bar assembly.')

    return b_struct

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
    """[summary]

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
    print('>'*10)
    # the actual bar axis endpts are index by using the element's
    # corresponding edge (n1, n2) mapped into the node_points
    # each entry's value (node pt id : axis pt, node pt id : axis pt)
    # since axis pt usually deviates from the ideal node pt
    # ! bar_from_elements[edge] = {node : [x,y,z]}
    bar_from_elements = {}
    # ! connect_vars[set([edge1, edge2]), node] = 1 or 0
    connect_vars = {}

    for _, element in enumerate(edge_seq):
        # temporal drawing cache for each iter
        handles = []

        # next_printed = printed | {element}
        # unprinted = all_elements - next_printed
        n0, n1 = element
        n_neighbors = {n0 : list(set(node_neighbors[n0]) & printed), \
                       n1 : list(set(node_neighbors[n1]) & printed)}
        print('------')
        # print('visited node: ', visited_nodes)
        # print('printed bars: ', printed)
        # print('existing node_neighbors: {}'.format(n_neighbors))
        # print('n#{} neighbors: {}'.format(n0, set(node_neighbors[n0])))
        # print('n#{} neighbors: {}'.format(n1, set(node_neighbors[n1])))

        # find neighnor elements of each node
        for i in [n0, n1]:
            # fill in empty tuple for product
            if len(n_neighbors[i]) == 0:
                n_neighbors[i] = [()]
            elif len(n_neighbors[i]) == 1:
                n_neighbors[i] = [n_neighbors[i]]
            elif len(n_neighbors[i]) >= 2:
                n_neighbors[i] = list(combinations(n_neighbors[i], 2))

        # each of these should be segmented into 2 pairs
        neighbor_pairs = list(product(n_neighbors[n0], n_neighbors[n1]))
        cprint('produced neighnor pairs: {}'.format(neighbor_pairs), 'yellow')

        chosen_new_axis_endpts = None
        chosen_contact_element_axis_pts = {}
        extension_score = np.inf
        # TODO: iterate through all neighboring pairs and use the pairs that produces the minial neighbor element extension
        # for contact_bars in randomize(neighbor_pairs):
        for contact_bars in neighbor_pairs:
            cprint('^'*10)
            new_axis_endpts, contact_elements = compute_tangent_bar(bar_from_elements, node_points, element, contact_bars, radius)
            print('new_axis_endpts: ', new_axis_endpts)
            # if new_axis_endpts:
            #     cprint('new axis pt: {} | contact elements : {}'.format(new_axis_endpts, contact_elements), 'cyan')
            #     break

            tmp_contact_elements = {}
            tmp_extension_scores = [0]
            score_fn = max
            for contact_e in contact_elements:
                neighbor_elements = list(set(element_neighbors[contact_e]) & set(printed))
                candidate_end_pts = []
                contact_e_axis_pts = list(bar_from_elements[contact_e].values())
                candidate_end_pts.extend(contact_e_axis_pts)
                # existing contact pts
                for e in neighbor_elements:
                    ne_contact_pts = dropped_perpendicular_points(*list(bar_from_elements[contact_e].values()),
                                                                  *list(bar_from_elements[e].values()))
                    candidate_end_pts.append(ne_contact_pts[0])

                    # draw new contact pts
                    add_line(np.array(ne_contact_pts[0])*1e-3, np.array(ne_contact_pts[1])*1e-3, color=apply_alpha(BLACK, 1))

                # new contact pts
                new_contact_pts = dropped_perpendicular_points(*new_axis_endpts.values(),
                                                               *bar_from_elements[contact_e].values())
                # print('new contact pt: ', new_contact_pts)

                candidate_end_pts.append(new_contact_pts[1])
                line_drawing = add_line(np.array(new_contact_pts[0])*1e-3, np.array(new_contact_pts[1])*1e-3, color=apply_alpha(BLACK, 1))
                handles.append(line_drawing)

                extended_end_pts = find_points_extreme(candidate_end_pts, list(bar_from_elements[contact_e].values()))
                tmp_extension_scores.append(abs(norm(extended_end_pts[0] - extended_end_pts[1]) - norm(contact_e_axis_pts[0]- contact_e_axis_pts[1])))

                tmp_contact_elements[contact_e] = {}
                c_n0, c_n1 = contact_e
                if norm(np.array(extended_end_pts[0]) - np.array(bar_from_elements[contact_e][c_n0])) > \
                   norm(np.array(extended_end_pts[0]) - np.array(bar_from_elements[contact_e][c_n1])):
                    tmp_contact_elements[contact_e][c_n1] = np.array(extended_end_pts[0])
                    tmp_contact_elements[contact_e][c_n0] = np.array(extended_end_pts[1])
                else:
                    tmp_contact_elements[contact_e][c_n0] = np.array(extended_end_pts[0])
                    tmp_contact_elements[contact_e][c_n1] = np.array(extended_end_pts[1])

            tmp_score = score_fn(tmp_extension_scores)
            print('new extension score: {} | {}'.format(tmp_score, extension_score))
            if tmp_score < extension_score:
                extension_score = tmp_score
                chosen_new_axis_endpts = new_axis_endpts
                chosen_contact_element_axis_pts = copy(tmp_contact_elements)

        for contact_e, end_pts in chosen_contact_element_axis_pts.items():
            # draw new axis pts
            end_pts = list(end_pts.values())
            add_line(end_pts[0]*1e-3, end_pts[1]*1e-3, color=apply_alpha(GREEN, 1))

        assert set(new_axis_endpts.keys()) == set(n_neighbors.keys())

        # update the new found bar axis
        bar_from_elements[element] = chosen_new_axis_endpts
        # update bar_from_elements, update corresponding entries
        bar_from_elements.update(chosen_contact_element_axis_pts)
        visited_nodes |= set([n0, n1])
        printed = printed | {element}

        for contact_e in chosen_contact_element_axis_pts :
            c_node = list(element&contact_e)[0]
            connect_vars[EDGE({element, contact_e}), c_node] = 1

        # update all bar element drawing
        with LockRenderer():
            for e in bar_from_elements.values():
                # convert mil to meter
                h = draw_element({0 : map(lambda x : 1e-3*x, list(e.values()))}, 0, width=3)
                circ_verts = bar_sec_verts(*list(e.values()), radius=radius)
                for v in circ_verts:
                    # assert(abs(v.dot(list(e.values())[0] - list(e.values())[1])) < 1e-8)
                    if abs(v.dot(list(e.values())[0] - list(e.values())[1])) > 1e-8:
                        cprint('circle dot: {}'.format(abs(v.dot(list(e.values())[0] - list(e.values())[1]))), 'red')

                ch1 = add_segments([(list(e.values())[0] + v)*1e-3 for v in circ_verts], closed=True, color=GREEN)
                ch2 = add_segments([(list(e.values())[1] + v)*1e-3 for v in circ_verts], closed=True, color=GREEN)
                # handles.extend([h] + ch1 + ch2)
                # handles.extend([h] + ch1 + ch2)

        if debug:
            wait_if_gui('A new bar is added.')
        remove_handles(handles)

    # * compile bars into a BarStructure
    b_struct = BarStructure()
    centroid_pt = np.average(node_points, axis=0)

    index_from_element = {}
    for e, pts in bar_from_elements.items():
        axis_endpts = list(pts.values())
        _ , _, vec_z = calculate_coord_sys(axis_endpts, centroid_pt)
        n0, n1 = e
        is_grounded = n0 in ground_nodes or n1 in ground_nodes
        bar_key = b_struct.add_bar(None, [p.tolist() for p in axis_endpts], "tube", None, vec_z, radius=radius, grounded=is_grounded)
        index_from_element[e] = bar_key

    for e, pts in bar_from_elements.items():
        # update contact point into BarS's edges
        for ne in element_neighbors[e]:
            b_struct.connect_bars(index_from_element[e], index_from_element[ne])
            contact_pts = compute_contact_line_between_bars(b_struct, index_from_element[e], index_from_element[ne])
            b_struct.edge[index_from_element[e]][index_from_element[ne]]["endpoints"].update({0:(list(contact_pts[0]), list(contact_pts[1]))})

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

    element_bodies = b_struct.get_element_bodies(color=apply_alpha(RED, 0.5))
    if debug:
        wait_if_gui('Final bar assembly.')

    return b_struct, (bar_from_elements, connect_vars)

def compute_tangent_bar(bar_from_elements, node_points, element, in_contact_bars, radius):
    """[summary]

    Parameters
    ----------
    bar_from_elements : dict
        {[element] : { node id : axis pt1}}
        element = (n1, n2), n1/2 indexs into `node_points`
    node_points : list of points
        the original list of points we want the bar system to achieve. We are pertubing the bar axis to approximately spanning these points.
    element : tuple of int
        (n1, n2) index into `node_points`
    in_contact_bars : [type]
        [description]
    radius : float
        bar radius in millimeter

    Returns
    -------
    dict
        Computed axis end pts for element :
        {n1 : axis pt 1, n2 : axis pt 2}, where element = (n1, n2)
    """
    # order the contact bar pairs based on number of bars in each entry
    contact_bars = copy(in_contact_bars)
    if len(in_contact_bars[1]) < len(in_contact_bars[0]):
        contact_bars = copy(in_contact_bars[::-1])
    assert(len(contact_bars[1])<=2)
    cprint('chosen element {}'.format(element), 'cyan')
    cprint('contact_bars: {}'.format(contact_bars), 'blue')

    # compute a new node vertex to achieve if any
    new_point = None
    # shared by all 0-x cases
    # fill in new_node_id and contact_v_id for them
    if len(contact_bars[0]) == 0:
        # * chose the pt that's not on existing line
        existing_nodes = set()
        for cb in contact_bars[1]:
            existing_nodes |= set(cb)
        new_node_id = list(set(element) - existing_nodes)[0]
        contact_v_id = list(set(element) - set([new_node_id]))[0]
        print('new node id {} | contact id {}'.format(new_node_id, contact_v_id))
        print('new node {} | contact node {}'.format(node_points[new_node_id], node_points[contact_v_id]))

        # * colinear check
        # colinear = True
        # for cb in contact_bars[1]:
        #     dist = distance_point_line(node_points[new_node_id], [node_points[cb[0]], node_points[cb[1]]])
        #     if dist > 1e-6:
        #         colinear = False
        # assert not colinear or len(contact_bars[1]) == 0
        new_point = node_points[new_node_id]
    cprint('new pt: {}'.format(new_point))

    # * each case needs to provide the following data:
    #   {new_node_id : new_point, contact_v_id : np.array(contact_pt)}, contact_e
    axis_vector = None
    if len(contact_bars[0]) == 0 and len(contact_bars[1]) == 0:
        cprint('0 - 0', 'yellow')
        new_point = node_points[new_node_id]
        contact_pt = node_points[contact_v_id]
        contact_e = []
    elif len(contact_bars[0]) == 0 and len(contact_bars[1]) == 1:
        cprint('0 - 1', 'yellow')
        assert len(contact_bars[1]) > 0
        # point tangent to a cylinder
        contact_e = contact_bars[1][0]
        supp_v_id = list(set(contact_e) - set(element))[0]
        contact_node_pt = bar_from_elements[contact_e][contact_v_id]
        # supp_node_pt = bar_from_elements[contact_e][supp_v_id]

        # t_pts = [point_M, d_e_add, d_e_sub]
        t_pts = lines_tangent_to_cylinder(bar_from_elements[contact_e], new_point, 2*radius)
        if t_pts is not None:
            normals = [np.array(t_pts[j]) for j in [1,2]]
            # randomly choose one of the contact normal (up or down)
            contact_pt = randomize(normals)[0] + contact_node_pt
            # print('contact pt: {}'.format(contact_pt))
            # sanity check: dropped points should be the same as above
            # contact_pt, _ = dropped_perpendicular_points(new_point, contact_pt, *bar_from_elements[contact_e].values())
            # print('contact pt: {}'.format(contact_pt))
        else:
            # colinear case, use the nearest axis pt as the contact point
            contact_axis_pts = list(bar_from_elements[contact_e].values())
            contact_pt = contact_axis_pts[0] if norm(contact_axis_pts[0]-new_point) < norm(contact_axis_pts[1]-new_point) else contact_axis_pts[1]
        # convert to a list, conforming to other cases' output
        contact_e = [contact_e]
    elif len(contact_bars[0]) == 0 and len(contact_bars[1]) == 2:
        # one is floating or bare-grounded, use `first_tangent` strategy
        # intersecting tangents planes from the point to the two cylinders
        cprint('0 - 2', 'yellow')
        contact_e = contact_bars[1]
        # supp_v_id_1 = list(set(contact_e[0]) - set(element))[0]
        # supp_v_id_2 = list(set(contact_e[1]) - set(element))[0]
        # randomly choose from the four tangent sides
        for tangent_side in randomize(range(4)):
            axis_vector = compute_tangent_from_two_lines(bar_from_elements[contact_e[0]].values(),
                                                         bar_from_elements[contact_e[1]].values(),
                                                         new_point, 2*radius, 2*radius, tangent_side)
            if axis_vector.any():
                break

        assert(axis_vector is not None)
        cprint('0-2: new_pt {} | axis vector {}'.format(new_point, axis_vector), 'red')
        if axis_vector is not None:
            # choose the contact pt that's with a longer bar length
            new_contact_pt1, _ = dropped_perpendicular_points(new_point, new_point+axis_vector, *bar_from_elements[contact_e[0]].values())
            new_contact_pt2, _ = dropped_perpendicular_points(new_point, new_point+axis_vector, *bar_from_elements[contact_e[1]].values())
            contact_pt = new_contact_pt1 if norm(new_contact_pt1-new_point) > norm(new_contact_pt2-new_point) else new_contact_pt2
    elif len(contact_bars[0])==1 and len(contact_bars[1])==1:
        # deg 1 - deg 1 (4 configs)
        cprint('1 - 1', 'yellow')
        contact_e1 = contact_bars[0][0]
        contact_e2 = contact_bars[1][0]
        contact_e = contact_bars[0] + contact_bars[1]

        # required output dict key data
        # contact / support vertex ids for the contact bars
        contact_v_id_1 = list(set(contact_e1) & set(element))[0]
        supp_v_id_1 = list(set(contact_e1) - set(element))[0]
        contact_v_id_2 = list(set(contact_e2) & set(element))[0]
        supp_v_id_2 = list(set(contact_e2) - set(element))[0]
        # print('contact v 1 {} | {} ||| contact v2 {} | {}'.format(contact_v_id_1, supp_v_id_1, contact_v_id_2, supp_v_id_2))

        line1 = [bar_from_elements[contact_e1][contact_v_id_1], bar_from_elements[contact_e1][supp_v_id_1]]
        line2 = [bar_from_elements[contact_e2][contact_v_id_2], bar_from_elements[contact_e2][supp_v_id_2]]
        # iterate through the two tangent cases
        for ind_1 in range(2):
            contact_pt1, vec_l1 = solve_one_one_tangent(line1, line2, radius, ind_1)
            if contact_pt1 is not None and vec_l1 is not None:
                break
        # add_line(contact_pt1*1e-3, vec_l1 + contact_pt1*1e-3, color=YELLOW, width=0.5)

        if vec_l1 is not None:
            contact_pt2, _ = dropped_perpendicular_points(contact_pt1, contact_pt1+vec_l1, *line2)
            new_node_id = contact_v_id_1
            new_point = contact_pt1
            contact_v_id = contact_v_id_2
            contact_pt = np.array(contact_pt2)
        else:
            assert False, 'no solution found!'
    elif len(contact_bars[0])==1 and len(contact_bars[1])==2:
        # deg 1 - deg 2
        # use `second_tangent` strategy
        cprint('1 - 2', 'yellow')
        contact_e = list(contact_bars[0]) + list(contact_bars[1])
        contact_e1 = contact_bars[0][0]
        # required output dict key data
        # contact / support vertex ids for the contact bars
        new_node_id = list(set(contact_e1) & set(element))[0]
        supp_v_id = list(set(contact_e1) - set(element))[0]
        contact_v_id = list(set(contact_bars[1][0]) & set(element))[0]
        assert contact_v_id == list(set(contact_bars[1][1]) & set(element))[0]

        contact_pt1 = bar_from_elements[contact_e1][new_node_id]
        supp_pt1 = bar_from_elements[contact_e1][supp_v_id]

        # pt_b_1, l1 = convert_pt_vec(bar_from_elements[contact_bars[1][0]])
        # pt_b_2, l2 = convert_pt_vec(bar_from_elements[contact_bars[1][1]])

        R = compute_local_coordinate_system(contact_pt1, supp_pt1)
        ex = R[:,1]
        ey = R[:,2]
        for ind in range(4):
            new_point, axis_vector = solve_second_tangent(contact_pt1, ex, ey, radius,
                list(bar_from_elements[contact_bars[1][0]].values()), list(bar_from_elements[contact_bars[1][1]].values()), 2*radius, 2*radius, ind)
            if axis_vector is not None:
                break
        assert(axis_vector is not None)

        if axis_vector is not None:
            new_contact_pt1, _ = dropped_perpendicular_points(new_point, new_point+axis_vector, *bar_from_elements[contact_bars[1][0]].values())
            new_contact_pt2, _ = dropped_perpendicular_points(new_point, new_point+axis_vector, *bar_from_elements[contact_bars[1][1]].values())
            # use the point with longer bar length
            contact_pt = new_contact_pt1 if norm(new_contact_pt1-new_point) > norm(new_contact_pt2-new_point) else new_contact_pt2

    elif len(contact_bars[0])==2 and len(contact_bars[1])==2:
        # 2 - 2 tangent bars on each end
        # use `third_tangent` strategy
        cprint('2 - 2', 'yellow')
        contact_e = list(contact_bars[0]) + list(contact_bars[1])

        # required output dict key data
        # contact / support vertex ids for the contact bars
        new_node_id = list(set(contact_bars[0][0]) & set(contact_bars[0][1]))[0]
        contact_v_id = list(set(contact_bars[1][0]) & set(contact_bars[1][1]))[0]
        assert set(element) == set([new_node_id, contact_v_id])

        # convenient alises
        line1 = bar_from_elements[contact_bars[0][0]].values()
        line2 = bar_from_elements[contact_bars[0][1]].values()
        line3 = bar_from_elements[contact_bars[1][0]].values()
        line4 = bar_from_elements[contact_bars[1][1]].values()

        # contact point 1
        pts_axis_1 = dropped_perpendicular_points(*line1, *line2)
        contact_pt1 = centroid_points(pts_axis_1)
        # contact point 2
        pts_axis_2 = dropped_perpendicular_points(*line3, *line4)
        contact_pt2 = centroid_points(pts_axis_2)
        pt_mid = centroid_points([contact_pt1, contact_pt2])

        R = compute_local_coordinate_system(contact_pt1, contact_pt2)
        ex = R[:,1]
        ey = R[:,2]

        # solve from mid point to both contact points
        # four tangent cases on each double-tangent end
        for ind_1, ind_2 in product(range(4), range(4)):
            ang, ref_point, vec_l1, vec_l2 = solve_third_tangent(pt_mid, ex, ey, radius, [line1, line2], [line3, line4], ind_1, ind_2, debug=True)
            if vec_l1 is not None and vec_l2 is not None:
                break
        assert(vec_l1 is not None and vec_l2 is not None)
        # pt3, vec_l1, vec_l2, ang_check = list(map(np.array, result))
        # add_line(pt3*1e-3, pt3 + vec_l1*1e-3, color=YELLOW, width=0.5)
        # add_line(pt3*1e-3, pt3 + vec_l2*1e-3, color=BLUE, width=0.5)

        # use the point with longer bar length as the end pt for the new bar
        candidate_end_pts = []
        for line in [line1, line2, line3, line4]:
            b_contact_pts, _ = dropped_perpendicular_points(ref_point, ref_point+vec_l1, *line)
            candidate_end_pts.append(b_contact_pts)
        extended_end_pts = list(map(np.array, find_points_extreme(candidate_end_pts, [ref_point, ref_point+vec_l1])))

        if norm(extended_end_pts[0] - node_points[new_node_id]) < norm(extended_end_pts[0] - node_points[contact_v_id]):
            new_point, contact_pt = extended_end_pts
        else:
            new_point, contact_pt = extended_end_pts[::-1]

    return {new_node_id : new_point, contact_v_id : np.array(contact_pt)}, contact_e

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
def gen_truss(problem_data, viewer=False, radius=3.17, debug=False, method='search', **kwargs):
    """[summary]

    Parameters
    ----------
    problem : [type]
        problem name, see `planning.parsing.get_assembly_path`
    viewer : bool, optional
        enable pybullet viewer, by default False
    radius : float, optional
        bar radius in millimeter, by default 3.17

    Returns
    -------
    [type]
        [description]
    """
    net = Network.from_data(problem_data)

    # TODO waiting for compas update to use ordered dict for nodes
    # node_points, edges = net.to_nodes_and_edges()
    node_points = [np.array([net.node[v][c] for c in ['x', 'y', 'z']]) for v in range(net.number_of_nodes())]
    # EDGE
    edges = [e for e in net.edges()]
    edge_attributes = {e[0] : e[1] for e in net.edges(True)}
    ground_nodes = [v for v, attr in net.nodes(True) if attr['fixed'] == True]

    # print('parsed edges from to_node_and_edges: {}'.format(edges))
    # print('parsed node_points: {}'.format(node_points))
    # print('parsed grounded_nodes: {}'.format(ground_nodes))

    # if not verbose:
    #     # used when rpc call is made to get around stdout error
    #     sys.stdout = open(os.devnull, 'w')
    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    # fixed_obstacles, robot = load_world()

    with LockRenderer():
        set_camera(node_points)
        draw_pose(unit_pose(), length=0.01)
        # draw the ideal truss that we want to achieve
        label_points([pt*SCALE for pt in node_points])
        for e in edges:
            n1, n2 = e
            p1 = node_points[n1] * SCALE
            p2 = node_points[n2] * SCALE
            add_line(p1, p2, color=apply_alpha(BLUE, 0.3), width=0.5)
        for v in ground_nodes:
            draw_circle(node_points[v]*1e-3, 0.01)

    if debug:
        wait_if_gui('Ideal truss...')

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
    elif method == 'shrink':
        b_struct = generate_shrinked_truss(node_points, edges, edge_attributes, ground_nodes, float(radius), debug=debug)
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
    parser.add_argument('-m', '--method', default='search', choices=METHOD_OPTIONS, help='Computing method')
    parser.add_argument('-r', '--radius', default=1.5, help='Radius of bars in millimeter')
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
    if args.method != 'bar_network':
        problem_path = get_assembly_path(args.problem)
        with open(problem_path) as json_file:
            graph_data = json.load(json_file)
            cprint('Design graph parsed from : {}'.format(problem_path), 'green')
        if 'data' in graph_data:
            graph_data = graph_data['data']

        b_struct, opt_data = gen_truss(graph_data, viewer=args.viewer, radius=args.radius,
            method=args.method, debug=args.debug)
    else:
        b_struct = from_bar_network(args.problem, viewer=args.viewer, debug=args.debug)

    if args.write:
        export_structure_data(b_struct.to_data(), overall_struct_data=graph_data, save_dir=FILE_DIR,
            file_name=export_file_name, opt_data=opt_data, indent=0)
        # export_structure_data(b_struct.data, net.data, radius=radius, **kwargs)

    check_model(b_struct, args.subset_bars, debug=args.debug)

    reset_simulation()
    disconnect()


if __name__ == '__main__':
    main()
