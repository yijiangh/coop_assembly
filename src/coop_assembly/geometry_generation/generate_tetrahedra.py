
'''

    ****       *****       ******       ****      ******  ******          **           **
   **  **      **  **      **          **  **       **    **              **           **
   **          *****       ****        ******       **    ****            **   *****   *****
   **  **      **  **      **          **  **       **    **              **  **  **   **  **
    ****   **  **  **  **  ******  **  **  **  **   **    ******          **   ******  *****


created on 30.06.2019
author: stefanaparascho

edited on 17.12.2019 by Yijiang Huang, yijiangh@mit.edu
'''
import sys, os
import random
import itertools
import math
import warnings

from compas.geometry import add_vectors, normalize_vector, scale_vector, \
    cross_vectors, subtract_vectors, Point, Vector
from compas.geometry import distance_point_point
from compas.geometry import rotate_points
from compas.geometry import angle_vectors
from compas.geometry import centroid_points

from coop_assembly.geometry_generation.tangents import first_tangent, second_tangent, third_tangent
from coop_assembly.help_functions.helpers_geometry import calculate_coord_sys, calculate_bar_z, \
    dropped_perpendicular_points, update_bar_lengths, correct_point, find_bar_ends, compute_contact_line_between_bars, \
    contact_to_ground
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE

from pybullet_planning import connect, reset_simulation, disconnect

def generate_first_triangle(o_struct, b_struct, radius, base_tri_pts, base_tri_ids, verbose=False):
    """[summary]

    Parameters
    ----------
    o_struct : [type]
        to be overwritten
    b_struct : [type]
        to be overwritten
    radius : float
        bar radius, in millimeter
    base_tri_pts : list of lists of 3-float
        [[x, y, z], [x, y, z], [x, y, z]]
    base_tri_ids : list of int
        point indices for the base triangle, used for bookkeeping indices
        in the OverallStructure vertex

    Returns
    -------
    (Bar_Structure, Overall_Structure)
        [description]
    """

    pt_0, pt_1, pt_2 = base_tri_pts

    vec_0   = normalize_vector(Point(*pt_0)-Point(*pt_1))
    vec_1   = normalize_vector(Point(*pt_1)-Point(*pt_2))
    vec_2   = normalize_vector(Point(*pt_2)-Point(*pt_0))
    c_0     = scale_vector(normalize_vector(cross_vectors(vec_0, vec_1)), 2*radius)
    c_1     = scale_vector(normalize_vector(cross_vectors(vec_1, vec_2)), 2*radius)
    c_2     = scale_vector(normalize_vector(cross_vectors(vec_2, vec_0)), 2*radius)

    # bar i: start point to raised end point
    end_pts_0 = (pt_0, add_vectors(pt_1, c_0))
    end_pts_1 = (pt_1, add_vectors(pt_2, c_1))
    end_pts_2 = (pt_2, add_vectors(pt_0, c_2))

    # ? overwriting the local frame's z axis above ???
    vec_z_0 = calculate_bar_z(end_pts_0)
    vec_z_1 = calculate_bar_z(end_pts_1)
    vec_z_2 = calculate_bar_z(end_pts_2)

    # add the three bars to the Bar_Structure as vertices,
    bar_type = 0
    crosec_type = "tube"
    crosec_values = (25.0, 2.0) # ? what does this cross section value mean?
    # these are vertex keys in the Bar_Structure network
    # * each bar is a vertex in the Bar_Structure
    # add_bar(self, _bar_type, _axis_endpoints, _crosec_type, _crosec_values, _zdir, _bar_parameters=[], radius=3.17,
    #   grounded=False, pb_scale=METER_SCALE)
    b_v0_key = b_struct.add_bar(bar_type, end_pts_0, crosec_type, crosec_values, vec_z_0, radius=radius) # , grounded=base_tri_ids[0] in grounded_nodes
    b_v1_key = b_struct.add_bar(bar_type, end_pts_1, crosec_type, crosec_values, vec_z_1, radius=radius) # , grounded=base_tri_ids[1] in grounded_nodes
    b_v2_key = b_struct.add_bar(bar_type, end_pts_2, crosec_type, crosec_values, vec_z_2, radius=radius) # , grounded=base_tri_ids[2] in grounded_nodes

    pt_m = [0,0,-1e13]

    # mean_pt used in SP' calculate_gripping_plane
    b_struct.node[b_v0_key].update({"mean_point":pt_m})
    b_struct.node[b_v1_key].update({"mean_point":pt_m})
    b_struct.node[b_v2_key].update({"mean_point":pt_m})

    # calculate contact point projected on bar axes, (Pi, P_{ci}) between bar i and bar i+1
    epts_0 = dropped_perpendicular_points(b_struct.node[b_v0_key]["axis_endpoints"][0],
                                          b_struct.node[b_v0_key]["axis_endpoints"][1],
                                          b_struct.node[b_v1_key]["axis_endpoints"][0],
                                          b_struct.node[b_v1_key]["axis_endpoints"][1])
    epts_1 = dropped_perpendicular_points(b_struct.node[b_v1_key]["axis_endpoints"][0],
                                          b_struct.node[b_v1_key]["axis_endpoints"][1],
                                          b_struct.node[b_v2_key]["axis_endpoints"][0],
                                          b_struct.node[b_v2_key]["axis_endpoints"][1])
    epts_2 = dropped_perpendicular_points(b_struct.node[b_v2_key]["axis_endpoints"][0],
                                          b_struct.node[b_v2_key]["axis_endpoints"][1],
                                          b_struct.node[b_v0_key]["axis_endpoints"][0],
                                          b_struct.node[b_v0_key]["axis_endpoints"][1])

    b_struct.connect_bars(b_v0_key, b_v1_key, _endpoints=epts_0)
    b_struct.connect_bars(b_v1_key, b_v2_key, _endpoints=epts_1)
    b_struct.connect_bars(b_v2_key, b_v0_key, _endpoints=epts_2)
    assert b_struct.edge[b_v0_key][b_v1_key]["grounded"] == False, "sanity check"

    b_struct.update_bar_lengths()

    tet_id = 0
    # tagging layer id using tet_id for partial ordering
    b_struct.node[b_v0_key].update({"layer":tet_id})
    b_struct.node[b_v1_key].update({"layer":tet_id})
    b_struct.node[b_v2_key].update({"layer":tet_id})

    # * OverallStructure updates
    # these are vertex's index in the Overall_Structure network
    o_v0_key = o_struct.add_node(pt_0, v_key=base_tri_ids[0], t_key=tet_id)
    o_v1_key = o_struct.add_node(pt_1, v_key=base_tri_ids[1], t_key=tet_id)
    o_v2_key = o_struct.add_node(pt_2, v_key=base_tri_ids[2], t_key=tet_id)
    if verbose:
        print('vertex key: {} added to the OverallStructure as the base triangle, original ids in the list: {}'.format(\
            [o_v0_key, o_v1_key, o_v2_key], base_tri_ids))

    # ? shouldn't these be assigned to tet #0 as well?
    # o_vi and o_vj's connection is "realized" by bar # b_v_key
    o_struct.add_bar(o_v0_key, o_v1_key, b_v0_key)
    o_struct.add_bar(o_v1_key, o_v2_key, b_v1_key)
    o_struct.add_bar(o_v0_key, o_v2_key, b_v2_key)

    # calculate and save the contact (tangent) point to each vertex
    o_struct.calculate_point(o_v0_key)
    o_struct.calculate_point(o_v1_key)
    o_struct.calculate_point(o_v2_key)

    return b_struct, o_struct


def generate_structure_from_points(o_struct, b_struct, radius, points, tet_node_ids,
    grounded_nodes=None, grounded_limit_distance=40, correct=False, check_collision=True, viewer=False, verbose=True):
    """generate double-tangent tet design from a given list of points and tet sequence indices.

    There are three types of parameters to be resolved at each step of the generation process:
    1. the node that a new bar connects to
    2. the two exisiting bars in the nodes that a new bar connects to
    3. one of the four possible sides of attachment to these bars

    where #1 above is specified in the given `tet_node_ids`) and the latter two are resolved
    in `add_tetra` function calls.

    Parameters
    ----------
    o_struct : [type]
        an empty OverallStructure to be written
    b_struct : [type]
        an empty Bar_Structure to be written
    radius : [type]
        [description]
    points : [type]
        [description]
    tet_node_ids : [type]
        [description]
    correct : bool, optional
        [description], by default True
    check_col : bool, optional
        [description], by default False
    """
    # * topology editing rule
    # sequential adding of tet onto one face of an existing tet
    # a new node is added at each step, with three more new elements (three-bar-group)

    # * connection
    # only two bars are joined at one point
    # double-tangent connection
    # each new bar connects to two existing bars in two joints

    # notation (see p56 of SP dissertation)
    # vertex point: approximated point for a node in an idealised geometry
    # node: a container of sub-nodes representing the physical realization of a vertex point
    # sub-node: a polygon (triangle) resulting when two new joints are added for a new bar
    # joint: a physical connection point where two bars touch and joined

    # input for the calculation of the three new bars:
    # 1. vertex points that represent the new node's location
    # 2. three pairs of bar axes of the existing bars that the three new bars will connect to their base
    # ! Note: needs to be cautious in relating bar index in the BarStructure to
    # ! an "edge" index in the OverallStructure

    # collision checks
    # parameters: connection side of the bar, existing bars of the node that the new bar is connecting to
    # the process iterates through over all four possible connection sides, and consequently runs through
    # all possible bar pairs that a new bar connect to in a side

    connect(use_gui=viewer)
    base_tri_ids = tet_node_ids[0][0]
    base_tri_pts = [points[node_id] for node_id in base_tri_ids]
    grounded_nodes = grounded_nodes or base_tri_pts

    if verbose:
        print('='*20)
        print('Generating the first triangle.')
    b_struct, o_struct = generate_first_triangle(o_struct, b_struct, radius, base_tri_pts, base_tri_ids, verbose=verbose)

    # if the first triangle is set as base, we don't add other subsequence connected bars as grounded bars
    for tet_id, (tri_node_ids, new_vertex_id) in enumerate(tet_node_ids):
        if verbose:
            print('='*20)
            print('Generating tet #{}: ({}) -> {}'.format(tet_id, tri_node_ids, new_vertex_id))

        # TODO: safe guarding base triangle has been added already
        vertex_pt = points[new_vertex_id]

        # ? does the order of the vertex in the base triangle matter?
        connected_edges_from_vert = {}
        for i, o_vert_id in enumerate(tri_node_ids):
            assert o_struct.has_node(o_vert_id), 'base triangle vertex {}: ({}) not added to the OverallStructure! key: {}'.format(
                i, o_vert_id, o_struct.node.keys())
            # all combination of two bars connected to the o_vertex, i.e.
            # existing neighboring bars to the vertex tri_node_ids[0]
            connected_o_edges = o_struct.vertex_connected_edges(o_vert_id)
            # pairs of OverallS edges (representing the pairs of existing bars that we can connect the new bar to)
            connected_o_edge_pairs = list(itertools.combinations(connected_o_edges, 2))
            # ? why reverse?
            connected_o_edge_pairs.reverse()
            connected_edges_from_vert[o_vert_id] = connected_o_edge_pairs
        assert(len(connected_edges_from_vert)==3)

        success = add_tetra(o_struct, b_struct, connected_edges_from_vert,
                            vertex_pt, new_vertex_id, radius,
                            correct=correct, check_collision=check_collision, verbose=verbose)
        if success is None:
            raise RuntimeError('Tet generation fails at #{} ({}) -> {}'.format(tet_id, tri_node_ids, new_vertex_id))
            # break
        else:
            new_bars = success[2]
            # tagging layer id using tet_id for partial ordering
            b_struct.node[new_bars[0]].update({"layer":tet_id+1})
            b_struct.node[new_bars[1]].update({"layer":tet_id+1})
            b_struct.node[new_bars[2]].update({"layer":tet_id+1})

    # tag grounded elements
    bar_endpts = b_struct.get_axis_pts_from_element(scale=1.0)
    for b, endpts in bar_endpts.items():
        pt0, pt1 = endpts
        for g_pt in grounded_nodes:
            if distance_point_point(pt0, g_pt) < grounded_limit_distance or \
               distance_point_point(pt1, g_pt) < grounded_limit_distance:
                b_struct.node[b].update({"grounded":True})
                break

    reset_simulation()
    disconnect()
    return b_struct, o_struct

def add_tetra(o_struct, b_struct, connected_edges_from_vert,
    new_vertex_pt, new_vertex_id, radius,
    bool_add=True, b_vert_ids=None, o_v_key=None, correct=True, check_collision=False, grounded_nodes=None, verbose=False):
    """adds a new point and tetrahedron to the structure
        input: nodes, bars from o_struct as vertex_key_integer and edge_vertex_key_tuples

    .. image:: ../images/three_bar_group_generation.png
        :scale: 60 %
        :align: center

    Parameters
    ----------
    o_struct : OverallStructure
        [description]
    b_struct : BarStructure
        [description]
    tri_node_ids : list of three int
    connected_edges_from_vert : dict
        {OverallS vertex key : list of OverallS's edges}
        dict keys: OverallStructure's vertex id triplets, representing the "ideal vertex" where multiple bars meet together.
        dict value: each entry is a list of OverallS's edges connected to ideal vertex tri_node_ids, each representing a potential new bar (edges in OverallS represents bars)
    new_vertex_pt : list, three floats
        [x, y, z] coordinate of the newly added ideal vertex in OverallS.
    new_vertex_id : int
        vertex key of the newly added ideal vertex in OverallS.
    radius : float
        radius of the bar, in millimeter
    bool_add : bool, optional
        generate new vertex, not using given b_vi, by default True
    b_vert_ids : list of three ints, optional
        BarS vertex ids, if specified, b_struct's corresponding vertices attributes will be updated, by default None
    o_v_key : int, optional
        if specified, o_struct's corresponding vertex pt will be updated, by default None
    correct : bool, optional
        perform angle/distance based vertex correction if True, by default True
    check_collision : bool, optional
        perform collision-based correction if True, by default False
    """

    # len_vec_min     = 500
    # len_vec_max     = 1400
    # len_vec         = (random.random()*(len_vec_max - len_vec_min))+len_vec_min
    max_len = 1800
    assert bool_add or (b_vert_ids is not None and len(b_vert_ids) == 3)
    assert len(connected_edges_from_vert) == 3

    # ! Note: needs to be cautious in relating bar index in the BarStructure to
    # ! an "edge" index in the OverallStructure
    tri_node_ids = list(connected_edges_from_vert.keys())
    # each is a list of o_edge id pairs (representing a pair of bars)
    comb_bars_1, comb_bars_2, comb_bars_3 = connected_edges_from_vert.values()

    # * finding the mean point of three contact midpoints
    jnd = 0
    bars1 = comb_bars_1[jnd]
    bars2 = comb_bars_2[jnd]
    bars3 = comb_bars_3[jnd]
    # print('Adding tet: bars 1 {} | bars 2 {} | bars 3 {}'.format(bars1, bars2, bars3))

    # vertex id in BarS
    # TODO: write a function to find mean point given bar ids
    # two bars at vertex 0
    b_v1_1 = o_struct.get_bar_vertex_key(bars1[0])
    b_v1_2 = o_struct.get_bar_vertex_key(bars1[1])

    # two bars at vertex 1
    b_v2_1 = o_struct.get_bar_vertex_key(bars2[0])
    b_v2_2 = o_struct.get_bar_vertex_key(bars2[1])

    # two bars at vertex 2
    b_v3_1 = o_struct.get_bar_vertex_key(bars3[0])
    b_v3_2 = o_struct.get_bar_vertex_key(bars3[1])

    # center points of the bar axes to obtain the central point of the base triangle
    dpp1 = compute_contact_line_between_bars(b_struct, b_v1_1, b_v1_2)
    contact_pt1 = centroid_points(dpp1)

    dpp2 = compute_contact_line_between_bars(b_struct, b_v2_1, b_v2_2)
    contact_pt2 = centroid_points(dpp2)

    dpp3 = compute_contact_line_between_bars(b_struct, b_v3_1, b_v3_2)
    contact_pt3 = centroid_points(dpp3)

    pt_mean = centroid_points([contact_pt1, contact_pt2, contact_pt3])

    # if new_vertex_pt:
    pt_new = new_vertex_pt

    if correct:
        # change new target vertex point position using the SP heuristic
        pt_new = correct_point(b_struct, o_struct, pt_new, [(b_v1_1, b_v1_2), (b_v2_1, b_v2_2), (b_v3_1, b_v3_2)], o_v_key=o_v_key)

    if verbose:
        print('------')
        print('Finding first tangent bar...')
    # enumerating all pairs of touch bars at vertex tri_node_ids[0]
    for j, bar_jnd_1 in enumerate(comb_bars_1):
        bars1 = bar_jnd_1
        b_v1_1 = o_struct.get_bar_vertex_key(bars1[0])
        b_v1_2 = o_struct.get_bar_vertex_key(bars1[1])

        if correct:
            # change new target vertex point position using the SP heuristic
            pt_new = correct_point(b_struct, o_struct, pt_new,
                                   [(b_v1_1, b_v1_2), (b_v2_1, b_v2_2), (b_v3_1, b_v3_2)], o_v_key=o_v_key)

        ret_ft = first_tangent(pt_new, contact_pt1, max_len,
                               b_v1_1, b_v1_2, b_struct, pt_mean, radius,
                               b_v0_n=None if bool_add else b_v0, check_collision=check_collision, verbose=verbose)

        if ret_ft is not None:
            # unboxing results
            b_struct, b_v0, _ = ret_ft
            if verbose:
                print('Bar chosen #{}: #B{}-(o{}) + #B{}-(o{})'.format(b_v0, b_v1_1, bars1[0], b_v1_2, bars1[1]))
            break
        else:
            if j == len(comb_bars_1)-1:
                # print("no point found for first tangent calculation - 430, add_tetra")
                raise RuntimeError("no point found for first tangent calculation - 430, add_tetra")

    if verbose:
        print('------')
        print('Finding second tangent bar...')
    for j, bar_jnd_2 in enumerate(comb_bars_2):
        bars2 = bar_jnd_2
        b_v2_1 = o_struct.get_bar_vertex_key(bars2[0])
        b_v2_2 = o_struct.get_bar_vertex_key(bars2[1])

        if correct:
            pt_new = correct_point(b_struct, o_struct, pt_new,
                                  [(b_v1_1, b_v1_2), (b_v2_1, b_v2_2), (b_v3_1, b_v3_2)], o_v_key=o_v_key)

        # b_v0 is the last added, first bar in the tet
        ret_st = second_tangent(contact_pt2, b_v2_1, b_v2_2,
                                b_struct, b_v0, pt_new, radius, max_len, pt_mean, b_v0_n=None, check_collision=check_collision, verbose=verbose)

        if ret_st:
            b_struct, b_v1, _, _ = ret_st
            if verbose:
                print('Bar chosen #{}: #B{}-(o{}) + #B{}-(o{})'.format(b_v1, b_v2_1, bars2[0], b_v2_2, bars2[1]))
            break
        else:
            if j == len(comb_bars_2) - 1:
                raise RuntimeError("no point found for second tangent calculation - 430, add_tetra")

    if verbose:
        print('------')
        print('Finding third tangent bar...')
    for j, bar_jnd_3 in enumerate(comb_bars_3):
        bars3 = bar_jnd_3
        b_v3_1 = o_struct.get_bar_vertex_key(bars3[0])
        b_v3_2 = o_struct.get_bar_vertex_key(bars3[1])

        if correct:
            pt_new = correct_point(b_struct, o_struct, pt_new,
                               [(b_v1_1, b_v1_2), (b_v2_1, b_v2_2), (b_v3_1, b_v3_2)], o_v_key=o_v_key)

        # b_v0, b_v1 are the two latest added vars in the tet
        ret_tt = third_tangent(b_struct, b_v0, b_v1, contact_pt3,
                               max_len, b_v3_1, b_v3_2, pt_mean, radius, b_v0_n=b_v2 if not bool_add else None, check_collision=check_collision, verbose=verbose)

        if ret_tt:
            b_struct, b_v2, _, _ = ret_tt
            if verbose:
                print('Bar chosen #{}: #B{}-(o{}) + #B{}-(o{})'.format(b_v2, b_v3_1, bars3[0], b_v3_2, bars3[1]))
            break
        else:
            # print("tangent 3 not found")
            if j == len(comb_bars_3) - 1:
                raise RuntimeError("no point found for third tangent calculation - add_tetra")

    # * BarStructure update
    if bool_add:
        # adding contact edge information in BarS
        b_struct.connect_bars(b_v0, b_v1)
        b_struct.connect_bars(b_v1, b_v2)
        b_struct.connect_bars(b_v2, b_v0)
    # contact edge coordinate
    dpp_1 = compute_contact_line_between_bars(b_struct, b_v1, b_v2)
    key = list(b_struct.edge[b_v1][b_v2]["endpoints"].keys())[0]
    b_struct.edge[b_v1][b_v2]["endpoints"].update({key:(dpp_1[0], dpp_1[1])})

    dpp_2 = compute_contact_line_between_bars(b_struct, b_v2, b_v0)
    key = list(b_struct.edge[b_v2][b_v0]["endpoints"].keys())[0]
    b_struct.edge[b_v2][b_v0]["endpoints"].update({key:(dpp_2[0], dpp_2[1])})

    dpp_3 = compute_contact_line_between_bars(b_struct, b_v0, b_v1)
    key = list(b_struct.edge[b_v0][b_v1]["endpoints"].keys())[0]
    b_struct.edge[b_v0][b_v1]["endpoints"].update({key:(dpp_3[0], dpp_3[1])})
    # grounded status update
    # if new_vertex_id in grounded_nodes:
    #     for new_bar_key in [b_v0, b_v1, b_v2]:
    #         b_struct.vertex[new_bar_key].update({'grounded' : True})

    # * OverallStructure update
    if bool_add:
        o_n_new = o_struct.add_node(pt_new, v_key=new_vertex_id)
        o_n1 = tri_node_ids[0]
        o_n2 = tri_node_ids[1]
        o_n3 = tri_node_ids[2]
        o_struct.add_bar(o_n_new, o_n1, b_v0)
        o_struct.add_bar(o_n_new, o_n2, b_v1)
        o_struct.add_bar(o_n_new, o_n3, b_v2)

    # adjust newly added bars' length
    find_bar_ends(b_struct, b_v0)
    find_bar_ends(b_struct, b_v1)
    find_bar_ends(b_struct, b_v2)

    # adjust neighbor bars' length
    find_bar_ends(b_struct, b_v1_1)
    find_bar_ends(b_struct, b_v1_2)
    find_bar_ends(b_struct, b_v2_1)
    find_bar_ends(b_struct, b_v2_2)
    find_bar_ends(b_struct, b_v3_1)
    find_bar_ends(b_struct, b_v3_2)

    return o_struct, b_struct, (b_v0, b_v1, b_v2)

##########################################
# TODO main
