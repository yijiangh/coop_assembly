
'''

    ****       *****       ******       ****      ******  ******          **           **
   **  **      **  **      **          **  **       **    **              **           **
   **          *****       ****        ******       **    ****            **   *****   *****
   **  **      **  **      **          **  **       **    **              **  **  **   **  **
    ****   **  **  **  **  ******  **  **  **  **   **    ******          **   ******  *****


created on 30.06.2019
author: stefanaparascho
'''

import math
import scipy.optimize
import numpy as np
from numpy.linalg import norm

from compas.geometry import add_vectors, subtract_vectors, cross_vectors, normalize_vector, scale_vector, vector_from_points, dot_vectors, length_vector
from compas.geometry import distance_point_point, distance_point_line, distance_line_line
from compas.geometry import is_point_on_segment
from compas.geometry import angle_vectors
from compas.geometry import centroid_points
from compas.geometry import project_points_plane

from coop_assembly.help_functions import dropped_perpendicular_points, find_points_extreme, check_dir, \
    calculate_coord_sys
# from coop_assembly.assembly_info_generation.fabrication_planes import calculate_gripping_plane
from coop_assembly.help_functions.shared_const import TOL
from coop_assembly.help_functions.helpers_geometry import compute_contact_line_between_bars, compute_local_coordinate_system


def compute_tangent_from_two_lines(line1, line2, ref_point, radius1, radius2, nb):
    """compute tangent bar axis vector for connecting a new point to two existing bars.
    This is done by computing tangent planes for two bars separately, and then take


    .. image:: ../images/first_tangent_plane_intersection.png

        :scale: 80 %
        :align: center

    Parameters
    ----------
    line1 : list of two points
        axis line for cylinder 1, [axis pt 0, axis pt 1]
    line2 : list of two points
        axis line for cylinder 2, [axis pt 0, axis pt 1]
    ref_point : point
        new vertex point Q
    radius1 : float
        radius for bar 1
    radius2 : float
        radius for bar 1
    nb : int
        tangent plane combination (two tangent planes per bar)

    Returns
    -------
    if
    list of one vector
        (not sure why SP needed a list around this single entry)
        a vector representing the new bar's axis
    """
    # planes1 in format [origin, axis vec, y axis], [origin, axis vec, y axis]
    planes1 = planes_tangent_to_cylinder(
        line1, ref_point, radius1, info='plane')
    # planes2 in format [normal, dot], [normal, dot]
    planes2 = planes_tangent_to_cylinder(
        line2, ref_point, radius2, info='contact')
    if planes1 == None or planes2 == None:
        print("Tangent planes not found")
        return None

    _, plane_x_axis, plane_y_axis = planes1[0] if nb == 0 or nb == 1 else planes1[1]
    plane_x_axis = normalize_vector(plane_x_axis)
    plane_y_axis = normalize_vector(plane_y_axis)
    normal = normalize_vector(planes2[nb%2][0])
    if not (abs(dot_vectors(plane_x_axis, normal)) < 1e-8 and abs(dot_vectors(plane_y_axis, normal)) < 1e-8):
        s = intersect_plane_plane_u(plane_x_axis, plane_y_axis, normal)
        s = normalize_vector(s)
        return s
    else:
        raise NotImplementedError("two tangent planes are parallel")

def lines_tangent_to_cylinder(cylinder_line, ref_point, radius):
    """Calculating tangent planes to a cylinder axis passing through the `ref_point`.
        This is one of the core geometric calculation routine.
        See SP dissertation 3.1.3.b (p. 74)

    .. image:: ../images/plane_tangent_to_one_cylinder.png
        :scale: 80 %
        :align: center

    Parameters
    ----------
    line_vect : list of two points
        [axis end 0, axis end 1]
        negating this direction would only swap the up/down up and down vectors.
    ref_point : point
        new point Q
    radius : float
        radius of the cylinder

    Returns
    -------
    list
        [contact point projected on the cylinder axis (point `M`), vector MB, -1 * vector MB], the latter two entries represent
        the tangent points' local coordinate in the plane [point M, e_x, e_y]
    """
    l_vect = normalize_vector(subtract_vectors(cylinder_line[1], cylinder_line[0]))
    base_point = cylinder_line[0]
    line_QMprime = subtract_vectors(ref_point, base_point)
    # * project out longitutude axis component of the base_point to obtain point M
    point_M = add_vectors(base_point, scale_vector(l_vect, dot_vectors(line_QMprime, l_vect)))
    line_QM = subtract_vectors(ref_point, point_M)

    e_x = normalize_vector(line_QM)
    e_y = cross_vectors(e_x, l_vect)
    if length_vector(line_QM) < 1e-6:
        # this means the ref_point is within the cylinder's body initially
        return None
    # sin(angle BQM)
    x = radius / length_vector(line_QM)
    # x coordinate in the local axis
    d_e1 = scale_vector(e_x, radius * x)
    # the radius of bar section (`radius`) has to be larger than line_QM (distance from point to bar axis),
    # otherwise the sqrt turns negative
    if x*x < 1.0:
        # y coordinate in the local axis
        d_e2 = scale_vector(e_y, radius*math.sqrt(1.0-x*x))
    else:
        # radius < line_QM: change ref_point
        return None

    # upper tangent point's delta x
    d_e_add = add_vectors(d_e1, d_e2)
    # lower tangent point's delta x
    d_e_sub = subtract_vectors(d_e1, d_e2)
    return [point_M, d_e_add, d_e_sub]

def planes_tangent_to_cylinder(cylinder_line, ref_point, radius, info='plane'):
    """find tangent planes of a cylinder passing through a given point, return tangent planes

    This is a convenient wrapper around `lines_tangent_to_cylinder` to get tangent planes.

    .. image:: ../images/plane_tangent_to_one_cylinder.png
        :scale: 80 %
        :align: center

    Parameters
    ----------
    cylinder_line : list of two points
        existing bar's axis vector [axis pt 0, axis pt 1]
        negating this order would only swap the up/down up and down vectors.
    ref_point : point
        point Q, the new floating point
    radius : float
        cylinder radius

    Returns
    -------
    None : if no solution found
    if info == 'plane'
        [[ref_point, l_vect, r1], [ref_point, l_vect, r2]] :
            list of two [ref_point, local_y, local_x]
            representing two tangent planes formed
            local x = QB
            local_y // line_vect
    if info == 'contact'
        [[upper_tang_pt, dot_1], [lower_tang_pt, dot_2]]
            upper_tang_pt is the upper contact (tangent) point
            dot1 = dot_vectors(ref_point, upper_tang_pt)
    """
    tangent_pts = lines_tangent_to_cylinder(cylinder_line, ref_point, radius)
    if tangent_pts is None:
        return None
    point_M, upper_delta_x, lower_delta_x = tangent_pts
    if info == 'plane':
        # r1 : B_up - ref_pt
        r1  = subtract_vectors(add_vectors(point_M, upper_delta_x), ref_point)
        r1  = normalize_vector(r1)
        # r1 : B_down - ref_pt
        r1  = subtract_vectors(add_vectors(point_M, upper_delta_x), ref_point)
        r2  = subtract_vectors(add_vectors(point_M, lower_delta_x), ref_point)
        r2  = normalize_vector(r2)
        l_vect  = normalize_vector(subtract_vectors(cylinder_line[1], cylinder_line[0]))
        return [[ref_point, l_vect, r1], [ref_point, l_vect, r2]]
    elif info == 'contact':
        dot_1 = dot_vectors(ref_point, upper_delta_x)
        dot_2 = dot_vectors(ref_point, lower_delta_x)
        return [[upper_delta_x, dot_1], [lower_delta_x, dot_2]]
    else:
        raise NotImplementedError('not supported info type: {}'.format(info))

def intersect_plane_plane_u(u_vect, v_vect, n_vect):
    """compute the spanning vector (line direction) for two intersecting planes
        plane 1:
            x axis: u_vect
            y axis: v_vect
        plane 2:
            z axis: abc_vect

    Essentially we are solving:
        <cross(u, v), x> = 0
        <n, x> = 0

    We use analytical solution: x = u - (<n,u>/<n,v>) v
    Notice that when <n, u> = 0 or <n, v> = 0, the solution space is essentially the span(u,v) itself.

    One might also consider directly uses `intersection_plane_plane` from `compas.geometry`

    Parameters
    ----------
    u_vect : list of 3 floats
        local x axis for plane 1
    v_vect : list of 3 floats
        local y axis for plane 1
    abc_vect : list of 3 floats
        local z axis for plane 2

    Returns
    -------
    vector
        directional vector for the intersecting line
    """
    n_vect = normalize_vector(n_vect)
    u_vect = normalize_vector(u_vect)
    v_vect = normalize_vector(v_vect)
    assert not (abs(dot_vectors(n_vect, u_vect)) < 1e-8 and abs(dot_vectors(n_vect, v_vect)) < 1e-8)
    A = dot_vectors(n_vect, u_vect) / dot_vectors(n_vect, v_vect)
    u_vect_sub  = subtract_vectors(u_vect, scale_vector(v_vect, A))
    return u_vect_sub


def compute_new_bar_length(vec_sol, compare_contact_pt, new_pt, b1_key, b2_key, b_struct):
    """compute proper axis end pts for a given new point and direction

    b_struct is not updated inside this function.

    Parameters
    ----------
    vec_sol : point
        directional point for the new bar axis
    compare_contact_pt : point
        this compare contact pt is only used to make sure that the output axis vector
        point FROM contact pt to the newly added point
    new_pt : point
        newly added point
    b1 : BarStructure vertex
    b2 : BarStructure vertex
    b1_key : int
    b2_key : int
    b_struct : BarStructure

    Returns
    -------
    None : if no feasible solution found
    [vec_sol, l_max, pts_b1, pts_b2]
        vec_sol : directional vector for the new bar's axis
        l_max : the new bar's length
        pts_b1 : modified contact bar 1' end points, extended to cover the contact of the new bar
        pts_b2 : modified contact bar 2' end points, ..
    """
    b1 = b_struct.node[b1_key]
    b2 = b_struct.node[b2_key]
    # compute contact pts with the two bars
    dpp = dropped_perpendicular_points(new_pt, add_vectors(new_pt, vec_sol), b1["axis_endpoints"][0], b1["axis_endpoints"][1])
    pt_x1, pt_1 = dpp
    dpp = dropped_perpendicular_points(new_pt, add_vectors(new_pt, vec_sol), b2["axis_endpoints"][0], b2["axis_endpoints"][1])
    pt_x2, pt_2 = dpp
    if pt_x1 is None or pt_x2 is None:
        return None

    # * compute all projected contact points on bar1
    pts_all_b1 = []
    # connected bars for b1
    b_vert_n = b_struct.vertex_neighbors(b1_key)
    pts_all_b1.append(b_struct.vertex[b1_key]["axis_endpoints"][0])
    pts_all_b1.append(b_struct.vertex[b1_key]["axis_endpoints"][1])
    for n in b_vert_n:
        pts_all_b1.append(dropped_perpendicular_points(
            b1["axis_endpoints"][0], b1["axis_endpoints"][1],
            b_struct.vertex[n]["axis_endpoints"][0], b_struct.vertex[n]["axis_endpoints"][1])[0])
    pts_all_b1.append(pt_1)
    # find the farthest pair of points among all contact projected pts
    pts_b1 = find_points_extreme(pts_all_b1, b1["axis_endpoints"])

    # * compute all projected contact points on bar2
    pts_all_b2 = []
    b_vert_n = b_struct.vertex_neighbors(b2_key)
    pts_all_b2.append(b_struct.vertex[b2_key]["axis_endpoints"][0])
    pts_all_b2.append(b_struct.vertex[b2_key]["axis_endpoints"][1])
    for n in b_vert_n:
        pts_all_b2.append(dropped_perpendicular_points(
            b2["axis_endpoints"][0], b2["axis_endpoints"][1], b_struct.vertex[n]["axis_endpoints"][0], b_struct.vertex[n]["axis_endpoints"][1])[0])
    pts_all_b2.append(pt_2)
    pts_b2 = find_points_extreme(pts_all_b2, b2["axis_endpoints"])

    if compare_contact_pt is not None:
        vec_test_dir_1 = subtract_vectors(compare_contact_pt, new_pt)
        if not check_dir(vec_sol, vec_test_dir_1):
            vec_sol = scale_vector(vec_sol, -1)

    # find which point is further
    lx1 = distance_point_point(new_pt, pt_x1)
    lx2 = distance_point_point(new_pt, pt_x2)
    l_max = lx1 if lx1 > lx2 else lx2

    return [vec_sol, l_max, pts_b1, pts_b2]

######################################################
# SP's tet group geometric functions

def first_tangent(new_pt, contact_pt, max_len, b_v1_1, b_v1_2, b_struct, pt_mean, radius,
        b_v0_n=None, check_collision=False):
    """0-2 case, zero existing bar at the new point, 2 bars existing at the other end

    SP disseration P129:
        two discrete parameters are used for adjusting the topology in case a collision is found:
        1. the connection side of the bar
        2. the existing bars in the node that a new bar is connecting to
        The process first iterates over the four possible connection sides (second image below)
        then runs through all possible bar pairs that a new bar can connect to in a node
        the check is performed sequentially for each of the three bars in a three-bar-group
        and stopped once a collision-free solution is found

    .. image:: ../images/perpendicular_bar_tangent_to_two_existing_bars.png
        :scale: 80%
        :align: center

    .. image:: ../images/connection_options_1to2.png
        :scale: 80%
        :align: center

    Parameters
    ----------
    new_pt : point
        OverallS new vertex point
    contact_pt : point
        contact pt used to make sure the direction is correct, see `contact_point` in `compute_new_bar_length`
    max_len : float
        max allowable length of bar, in mm
    b_v1_1 : int
        BarS vertex key for bar b1_1
    b_v1_2 : int
        BarS vertex key for bar b1_2
    b_struct : BarStructure
    pt_mean : point
        base triangle central point
    radius : float
        bar radius (millimeter)
    b_v0_n : int, optional
        index_sol attribute to indicate which tangent plane solution (four solutions in total), by default None
        If set to None, iterate all four solutions until found a feasible one.
    check_collision : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    b1_1 = b_struct.node[b_v1_1]
    b1_2 = b_struct.node[b_v1_2]

    sol_indices = range(4) if check_colisions else [b_struct.vertex[b_v0_n]["index_sol"][0] if b_v0_n else 0]
    # * iterating through combinations of tangent plane configurations
    for sol_i, sol_id in enumerate(sol_indices):
        new_bar_axis = compute_tangent_from_two_lines(b1_1["axis_endpoints"], b1_2["axis_endpoints"],
                                              new_pt, 2 * radius, 2 * radius, sol_id)

        if new_bar_axis is None:
            print("First tangent bar: bar #{} no solutions.".format(sol_id))
            if sol_i == len(sol_indices)-1:
                print("First tangent bar: all four tangent planes exhausted and no solution is found!")
                return None
            continue

        # directional vector pointing from contact pt to the newly added point
        # pts_b1_1, 2 are the updated axis pts for bar1 and bar2 to cover all related contact pts
        vec_sol_1, l1, pts_b1_1, pts_b1_2 = compute_new_bar_length(new_bar_axis[0], contact_pt, new_pt, b_v1_1, b_v1_2, b_struct)

        # new central axis end point (contact end)
        new_pt_e = add_vectors(new_pt, scale_vector(vec_sol_1, l1))
        new_axis_end_pts = (new_pt, new_pt_e)

        if not check_colisions:
            break

        # add extension for collision checking
        ext_len = 30
        new_axis_end_pts = (add_vectors(new_pt, scale_vector(normalize_vector(vector_from_points(new_pt_e, new_pt)), ext_len)), \
                            add_vectors(new_pt_e, scale_vector(normalize_vector(vector_from_points(new_pt, new_pt_e)), ext_len)))

        is_collided = check_colisions(b_struct, new_axis_end_pts, radius, bar_nb=b_v0_n)

        if is_collided:
            end_pts_check = b_struct.get_bar_axis_end_pts(b_v1_1)
            is_collided = check_colisions(b_struct, end_pts_check, radius, bar_nb=b_v0_n, bar_checking=b_v1_1)
            if is_collided:
                end_pts_check = b_struct.get_bar_axis_end_pts(b_v1_2)
                is_collided = check_colisions(b_struct, end_pts_check, radius, bar_nb=b_v0_n, bar_checking=b_v1_2)
        if not is_collided:
            print("First tangent bar: Collide: bar v#{}".format(len(b_struct.vertex)))
            if sol_i == len(sol_indices)-1:
                print("First tangent bar: no tangent 1 found in one bar combination.")
                return None
        else:
            break

    vec_x, vec_y, vec_z = calculate_coord_sys(new_axis_end_pts, pt_mean)
    if not b_v0_n:
        b_v0 = b_struct.add_bar(0, new_axis_end_pts, "tube", (25.0, 2.0), vec_z, radius=radius)
    else:
        b_v0 = b_v0_n
        b_struct.vertex[b_v0].update(
            {"axis_endpoints": new_axis_end_pts})

    b_struct.vertex[b_v0].update({"index_sol":[sol_id]})
    b_struct.vertex[b_v0].update({"mean_point":pt_mean})

    if not b_v0_n:
        b_struct.connect_bars(b_v0, b_v1_1)
        b_struct.connect_bars(b_v0, b_v1_2)

    dpp_1 = compute_contact_line_between_bars(b_struct, b_v0, b_v1_1)
    dpp_2 = compute_contact_line_between_bars(b_struct, b_v0, b_v1_2)

    # * update contact point into BarS's edges
    k_1 = list(b_struct.edge[b_v0][b_v1_1]["endpoints"].keys())[0]
    k_2 = list(b_struct.edge[b_v0][b_v1_2]["endpoints"].keys())[0]
    b_struct.edge[b_v0][b_v1_1]["endpoints"].update({k_1:(dpp_1[0], dpp_1[1])})
    b_struct.edge[b_v0][b_v1_2]["endpoints"].update({k_2:(dpp_2[0], dpp_2[1])})

    return b_struct, b_v0, new_axis_end_pts

def second_tangent(pt_mean_2, b_v2_1, b_v2_2, b_struct, b_v_old, new_point, radius, max_len, pt_mean, b_v0_n=None, check_collision=False):
    """1-2 case, one existing bar at the new point, 2 bars existing at the other end

    Arguments
    ---------
    b_v2_1 : BarS vertex
    b_v2_2 :
        lower two existing bars
    b_v_old : BarS vertex
        the first bar in the new tet group

    """
    b2_1 = b_struct.node[b_v2_1]
    b2_2 = b_struct.node[b_v2_2]

    # floating newly added bar in the tet (added by `first_tangent`)
    line = b_struct.vertex[b_v_old]["axis_endpoints"]
    # vec_l_0 = vector_from_points(line[0], line[1])
    # local coordinate system at the new point, used for parameterizing the contact pt
    R = compute_local_coordinate_system(*line)
    ex = R[:,1]
    ey = R[:,2]

    pt_b_1      = b2_1["axis_endpoints"][0]
    pt_b_1_2    = b2_1["axis_endpoints"][1]
    l_1         = vector_from_points(pt_b_1, pt_b_1_2)
    pt_b_2      = b2_2["axis_endpoints"][0]
    pt_b_2_2    = b2_2["axis_endpoints"][1]
    l_2         = vector_from_points(pt_b_2, pt_b_2_2)

    if not check_collision:
        if b_v0_n:
            ind = b_struct.vertex[b_v0_n]["index_sol"][0]
        else:
            ind = 0

        sols_test = compute_tangent_from_two_lines(b2_1["axis_endpoints"], b2_2["axis_endpoints"], new_point, 2 * radius, 2 * radius, ind)
        if not sols_test:
            return None

        ret_sst = solve_second_tangent(new_point, ex, ey, radius, pt_b_1, l_1, pt_b_2, l_2, 2*radius, 2*radius, ind)
        if ret_sst:
            pt2, vec_l = ret_sst
        else:
            return None

        solution = vec_l

        ret_cls = compute_new_bar_length(
            solution, pt_mean_2, pt2, b_v2_1, b_v2_2, b_struct)

        if not ret_cls:
            return None

        vec_sol_2, l2, pts_b2_1, pts_b2_2 = ret_cls
        pt2_e = add_vectors(pt2, scale_vector(vec_sol_2, l2))
        end_pts_0 = (pt2, pt2_e)
    else:
        # check collisions
        for ind in range(4):
            # safeguarding, assuming no first bar exists
            sols_test = compute_tangent_from_two_lines(
                b2_1["axis_endpoints"], b2_2["axis_endpoints"], new_point, 2 * radius, 2 * radius, ind)

            if ind == 3 and sols_test == None:
                return None
            if sols_test == None:
                continue

            ret_sst = solve_second_tangent(new_point, ex, ey, radius, pt_b_1, l_1, pt_b_2, l_2, 2*radius, 2*radius, ind)
            if ret_sst:
                pt2, vec_l = ret_sst
            else:
                return None
            solution = vec_l
            ret_cls = compute_new_bar_length(solution, pt_mean_2, pt2, b_v2_1, b_v2_2, b_struct)

            if not ret_cls:
                return None
            # new bar's axis vector, length, and update contact bars axis pts to cover the contact
            vec_sol_2, l2, pts_b2_1, pts_b2_2 = ret_cls

            pt2_e = add_vectors(pt2, scale_vector(vec_sol_2, l2))
            end_pts_0 = (pt2, pt2_e)

            ext_len = 30
            end_pts_0 = (add_vectors(pt2, scale_vector(normalize_vector(vector_from_points(pt2_e, pt2)), ext_len)),
                         add_vectors(pt2_e, scale_vector(normalize_vector(vector_from_points(pt2, pt2_e)), ext_len)))

            bool_col = check_colisions(b_struct, end_pts_0, radius, bar_nb=b_v0_n)

            if bool_col:
                end_pts_check = b_struct.vertex[b_v2_1]["axis_endpoints"]
                bool_col = check_colisions(
                    b_struct, end_pts_check, radius, bar_nb=b_v0_n, bar_checking=b_v2_1)
                if bool_col:
                    end_pts_check = b_struct.vertex[b_v2_2]["axis_endpoints"]
                    bool_col = check_colisions(
                        b_struct, end_pts_check, radius, bar_nb=b_v0_n, bar_checking=b_v2_2)
            if not bool_col:
                print("COLLIDE", len(b_struct.vertex))
            if ind == 3 and not bool_col:
                print("NO TANGENT 2 FOUND IN ONE BAR COMBINATION")
                return None
            if bool_col:
                break

    vec_x, vec_y, vec_z = calculate_coord_sys(end_pts_0, pt_mean)
    if not b_v0_n:
        # b_v0    = b_struct.add_bar(0, end_pts_0, "tube", (2*radius, 2.0), vec_z)
        b_v0    = b_struct.add_bar(0, end_pts_0, "tube", (25.0, 2.0), vec_z, radius=radius)
    else:
        b_v0    = b_v0_n
        b_struct.vertex[b_v0].update({"axis_endpoints": end_pts_0})

    b_struct.vertex[b_v0].update({"index_sol": [ind]})
    b_struct.vertex[b_v0].update({"mean_point":pt_mean})

    b2_1.update({"axis_endpoints" : pts_b2_1})
    b2_2.update({"axis_endpoints" : pts_b2_2})
    if not b_v0_n:
        b_struct.connect_bars(b_v0, b_v2_1)
        b_struct.connect_bars(b_v0, b_v2_2)

    dpp_1 = dropped_perpendicular_points(b_struct.vertex[b_v0]["axis_endpoints"][0], b_struct.vertex[b_v0]["axis_endpoints"][1], b_struct.vertex[b_v2_1]["axis_endpoints"][0], b_struct.vertex[b_v2_1]["axis_endpoints"][1])
    dpp_2 = dropped_perpendicular_points(b_struct.vertex[b_v0]["axis_endpoints"][0], b_struct.vertex[b_v0]["axis_endpoints"][1], b_struct.vertex[b_v2_2]["axis_endpoints"][0], b_struct.vertex[b_v2_2]["axis_endpoints"][1])
    #     b_struct.edge[b_v0][b_v2_1].update({"endpoints":[dpp_1[0], dpp_1[1]]})
    #     b_struct.edge[b_v0][b_v2_2].update({"endpoints":[dpp_2[0], dpp_2[1]]})
    k_1 = list(b_struct.edge[b_v0][b_v2_1]["endpoints"].keys())[0]
    k_2 = list(b_struct.edge[b_v0][b_v2_2]["endpoints"].keys())[0]
    b_struct.edge[b_v0][b_v2_1]["endpoints"].update({k_1:(dpp_1[0], dpp_1[1])})
    b_struct.edge[b_v0][b_v2_2]["endpoints"].update({k_2:(dpp_2[0], dpp_2[1])})

    return b_struct, b_v0, pt2, end_pts_0

def third_tangent(b_struct, b_v0, b_v1, pt_mean_3, max_len, b_v3_1, b_v3_2, pt_mean, radius, b_v0_n=None, check_collision=False):
    """2-2 case, two existing bar at the new point, 2 bars existing at the other end

       b_v0, b_v1 are the two latest added bars in the tet
    """
    # "base" two bars
    b3_1 = b_struct.node[b_v3_1]
    b3_2 = b_struct.node[b_v3_2]
    pt_b_3 = b3_1["axis_endpoints"][0]
    pt_b_4 = b3_2["axis_endpoints"][0]
    line_3 = b3_1["axis_endpoints"]
    line_4 = b3_2["axis_endpoints"]

    # lastest two bars
    line_1 = b_struct.node[b_v0]["axis_endpoints"]
    line_2 = b_struct.node[b_v1]["axis_endpoints"]
    pt_b_1 = line_1[0]
    pt_b_2 = line_2[0]

    l_1 = normalize_vector(vector_from_points(line_1[0], line_1[1]))
    l_2 = normalize_vector(vector_from_points(line_2[0], line_2[1]))
    l_3 = normalize_vector(vector_from_points(line_3[0], line_3[1]))
    l_4 = normalize_vector(vector_from_points(line_4[0], line_4[1]))

    # contact point 1
    pts_axis_1 = dropped_perpendicular_points(line_1[0], line_1[1], line_2[0], line_2[1])
    pt_axis_1 = centroid_points(pts_axis_1)
    # contact point 2
    pts_axis_2 = dropped_perpendicular_points(line_3[0], line_3[1], line_4[0], line_4[1])
    pt_axis_2 = centroid_points(pts_axis_2)
    pt_mid = centroid_points((pt_axis_1, pt_axis_2))
    # axis = vector_from_points(pt_axis_1, pt_axis_2)

    R = compute_local_coordinate_system(pt_axis_1, pt_axis_2)
    ex = R[:,1]
    ey = R[:,2]

    if not check_collision:
        if b_v0_n:
            ind_1 = b_struct.vertex[b_v0_n]["index_sol"][0]
            ind_2 = b_struct.vertex[b_v0_n]["index_sol"][1]
        else:
            ind_1 = 0
            ind_2 = 0

        # solve from mid point to both contact points
        ret_stt = solve_third_tangent(pt_mid, ex, ey, radius, pt_b_1, l_1, pt_b_2, l_2, pt_b_3, l_3, pt_b_4, l_4, ind_1, ind_2)

        if ret_stt:
            pt3, vec_l1, vec_l2, ang_check = ret_stt
        else:
            return None

        # pts_3.append(pt3)
        # solutions_1.append(vec_l1)
        # solutions_2.append(vec_l2)
        solution_1 = vec_l1
        solution_2 = vec_l2

        test_1 = compute_new_bar_length(
            solution_2, pt_mean_3, pt3, b_v3_1, b_v3_2, b_struct)
        test_2 = compute_new_bar_length(
            solution_1, pt_mean_3, pt3, b_v0, b_v1, b_struct)

        if not test_1 or not test_2:
            return None

        # for n in test_1:
        #     for m in test_2:
        #         if n[4] == m[4]:
        #             vec_sol_31, l31, pts_b3_11, pts_b3_21, ind = n
        #             vec_sol_32, l32, pts_b3_12, pts_b3_22, ind_2 = m
        vec_sol_31, l31, pts_b3_11, pts_b3_21 = test_1
        vec_sol_32, l32, pts_b3_12, pts_b3_22 = test_2

        pt3_e1 = add_vectors(pt3, scale_vector(vec_sol_31, l31))
        pt3_e2 = add_vectors(pt3, scale_vector(vec_sol_32, -1 * l32))

        end_pts_0 = (pt3_e2, pt3_e1)
    else:
        bool_test = False
        for i in range(4):
            for j in range(4):
                ind_1 = i
                ind_2 = j

                ret_stt = solve_third_tangent(pt_mid, ex, ey, radius, pt_b_1, l_1, pt_b_2, l_2, pt_b_3, l_3, pt_b_4, l_4, ind_1, ind_2)
                if ret_stt:
                    pt3, vec_l1, vec_l2, ang_check  = ret_stt
                else:
                    # ? premature return?
                    return None

                # pts_3.append(pt3)
                # solutions_1.append(vec_l1)
                # solutions_2.append(vec_l2)
                solution_1 = vec_l1
                solution_2 = vec_l2

                #for j in range(4):
                test_1 = compute_new_bar_length(solution_2, pt_mean_3, pt3, b_v3_1, b_v3_2, b_struct)
                test_2 = compute_new_bar_length(solution_1, pt_mean_3, pt3, b_v0, b_v1, b_struct)

                if not test_1 or not test_2:
                    return None

                vec_sol_31, l31, pts_b3_11, pts_b3_21  = test_1
                vec_sol_32, l32, pts_b3_12, pts_b3_22 = test_2

                pt3_e1 = add_vectors(pt3, scale_vector(vec_sol_31, l31))
                pt3_e2 = add_vectors(pt3, scale_vector(vec_sol_32, -1 * l32))

                end_pts_0 = (pt3_e2, pt3_e1)

                ext_len = 30
                end_pts_0 = (add_vectors(pt3_e2, scale_vector(normalize_vector(vector_from_points(pt3_e1, pt3_e2)), ext_len)),
                             add_vectors(pt3_e1, scale_vector(normalize_vector(vector_from_points(pt3_e2, pt3_e1)), ext_len)))

                bool_col = check_colisions(b_struct, end_pts_0, radius, bar_nb=b_v0_n)

                if bool_col:
                    end_pts_check = b_struct.vertex[b_v3_1]["axis_endpoints"]
                    bool_col = check_colisions(
                        b_struct, end_pts_check, radius, bar_nb=b_v0_n, bar_checking=b_v3_1)
                    if bool_col:
                        end_pts_check = b_struct.vertex[b_v3_2]["axis_endpoints"]
                        bool_col = check_colisions(
                            b_struct, end_pts_check, radius, bar_nb=b_v0_n, bar_checking=b_v3_2)

                # bool_col = True
                if not bool_col:
                    print("COLLIDE", len(b_struct.vertex))
                if i == 3 and j == 3 and not bool_col:
                    print("NO TANGENT 3 FOUND IN ONE BAR COMBINATION")
                    return None
                if bool_col:
                    bool_test = True
                    break
            if bool_test: break

    # end_pts_0 = [map(float, p) for p in end_pts_0]
    vec_x, vec_y, vec_z = calculate_coord_sys(end_pts_0, pt_mean)
    # pt_o        = centroid_points(end_pts_0)
    if not b_v0_n:
        # b_v0    = b_struct.add_bar(0, end_pts_0, "tube", (2*radius, 2.0), vec_z)
        b_v0    = b_struct.add_bar(0, end_pts_0, "tube", (25.0, 2.0), vec_z, radius=radius)
    else:
        b_v0    = b_v0_n
        b_struct.vertex[b_v0].update(
            {"axis_endpoints": end_pts_0})

    b_struct.vertex[b_v0].update({"index_sol": [ind_1, ind_2]})
    # b_struct.vertex[b_v0].update({"gripping_plane_no_offset":(pt_o, vec_x, vec_y, vec_z)})

    # calculate_gripping_plane(b_struct, b_v0, pt_mean)
    b_struct.vertex[b_v0].update({"mean_point":pt_mean})

    b3_1.update({"axis_endpoints" :  pts_b3_11})
    b3_2.update({"axis_endpoints" :  pts_b3_21})
    if not b_v0_n:
        b_struct.connect_bars(b_v0, b_v3_1)
        b_struct.connect_bars(b_v0, b_v3_2)

    dpp_1 = dropped_perpendicular_points(b_struct.vertex[b_v0]["axis_endpoints"][0], b_struct.vertex[b_v0]["axis_endpoints"][1], b_struct.vertex[b_v3_1]["axis_endpoints"][0], b_struct.vertex[b_v3_1]["axis_endpoints"][1])
    dpp_2 = dropped_perpendicular_points(b_struct.vertex[b_v0]["axis_endpoints"][0], b_struct.vertex[b_v0]["axis_endpoints"][1], b_struct.vertex[b_v3_2]["axis_endpoints"][0], b_struct.vertex[b_v3_2]["axis_endpoints"][1])

    #     b_struct.edge[b_v0][b_v3_1].update({"endpoints":[dpp_1[0], dpp_1[1]]})
    #     b_struct.edge[b_v0][b_v3_2].update({"endpoints":[dpp_2[0], dpp_2[1]]})
    k_1 = list(b_struct.edge[b_v0][b_v3_1]["endpoints"].keys())[0]
    k_2 = list(b_struct.edge[b_v0][b_v3_2]["endpoints"].keys())[0]
    b_struct.edge[b_v0][b_v3_1]["endpoints"].update({k_1:(dpp_1[0], dpp_1[1])})
    b_struct.edge[b_v0][b_v3_2]["endpoints"].update({k_2:(dpp_2[0], dpp_2[1])})

    return b_struct, b_v0, pt3, end_pts_0

#################################################

def solve_one_one_tangent(line1, line2, radius, ind_1, debug=False):
    """compute tanget bar to two bars, the "new point" is line1[0], different to `solve_tangent_one`.

    Assuming two bars have the same radius for now.

    Parameters
    ----------
    line1 : list of pts
        [contact pt, other pt], direction not important
    line2 : list of pts
        [contact pt, other pt], direction not important
    radius : float
        [description]
    ind_1 : int
        tangent plane index \in [0, 3]

    Returns
    -------
    [type]
        [description]
    """
    line1 = list(map(np.array, line1))
    line2 = list(map(np.array, line2))
    def compute_tan_pts(theta):
        # 3by3 matrix representing the bar's local coordinate
        R = compute_local_coordinate_system(*line1)
        v = np.array([0, np.cos(theta), np.sin(theta)])
        contact_pt1 = line1[0] + 2*radius*R.dot(v)

        # ! this does not conform to bar3 and bar4's own radius
        t_pts1 = lines_tangent_to_cylinder(line2, contact_pt1, 2*radius)

        if t_pts1 is not None:
            vec_l1 = np.array(line2[0]) + np.array(t_pts1[ind_1+1]) - contact_pt1
            vec_l1 /= norm(vec_l1)
            return np.array(contact_pt1), vec_l1
        else:
            return None, None

    def fn(theta):
        contact_pt1, vec_l1 = compute_tan_pts(theta)
        if contact_pt1 is None and vec_l1 is None:
            return 1e5
        delta_x1 = contact_pt1 - line1[0]
        delta_x1 /= norm(delta_x1)
        return abs(delta_x1.dot(vec_l1))

    res_opt = scipy.optimize.fminbound(fn, 0, 2*np.pi, full_output=True, disp=0 if not debug else 3)
    if abs(res_opt[1]) > 1e-3:
        return None, None

    contact_pt1, vec_l1 = compute_tan_pts(res_opt[0])
    return contact_pt1, vec_l1

def solve_second_tangent(new_point, ex, ey, radius, line1, line2, diameter_1, diameter_2, ind, debug=False):
    """find a line that passes through the circle of `radius` around new_point, while staying tangent
    to `line1` and `line2`.

    Parameters
    ----------
    new_point : [type]
        [description]
    ex : [type]
        [description]
    ey : [type]
        [description]
    radius : [type]
        [description]
    line1 : [type]
        [description]
    line2 : [type]
        [description]
    diameter_1 : [type]
        [description]
    diameter_2 : [type]
        [description]
    ind : [type]
        [description]
    debug : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """

    # try twice?
    # for i in range(2):
    r_c = 2*radius
    def compute_tan_pt(x):
        # sample (dx, dy) in the local coordinate system (ex, ey)
        delta_x = add_vectors(scale_vector(ex, x), scale_vector(ey, math.sqrt(r_c*r_c - x*x)))
        # point on the new bar
        ref_point = add_vectors(new_point, delta_x)
        vec = compute_tangent_from_two_lines(line1, line2, ref_point, diameter_1, diameter_2, ind)
        return ref_point, vec

    def fn(x):
        ref_point, vecs_l_all = compute_tan_pt(x)
        if vecs_l_all:
            vec_l = vecs_l_all[0]
        else:
            print("error in f")
            val = 1
            return val
        # we want to have the contact line orthogonal to the new axis
        val = abs(dot_vectors(normalize_vector(vec_l), normalize_vector(vector_from_points(new_point, ref_point))))
        return val

    res_opt = scipy.optimize.fminbound(fn, -2*radius, 2*radius, full_output=True, disp=0 if not debug else 3)
    if res_opt[1] > 0.1:
        return None

    x = float(res_opt[0])
    ret_fp2 = compute_tan_pt(x)
    if ret_fp2[1] is None:
        return None
    else:
        pt_2, vec_l = ret_fp2
        return pt_2, vec_l[0]

def solve_third_tangent(pt_mid, ex, ey, radius, line_pair1, line_pair2, ind_1, ind_2, debug=False):
    assert len(line_pair1) == 2 and len(line_pair2) == 2
    def compute_third_tan_pts(x):
        """ x is the local coordinate in (ex, ey) for the new axis point
        """
        x1  = x[0]
        x2  = x[1]
        delta_x1 = add_vectors(scale_vector(ex, x1), scale_vector(ey, x2))
        ref_point = add_vectors(pt_mid, delta_x1)
        # ! this does not conform to bar3 and bar4's own radius
        vec_l1 = compute_tangent_from_two_lines(*line_pair1, ref_point, 2*radius, 2*radius, ind_1)
        # ! this does not conform to bar3 and bar4's own radius
        vec_l2 = compute_tangent_from_two_lines(*line_pair2, ref_point, 2*radius, 2*radius, ind_2)
        if not vec_l1[0] or not vec_l2[0]:
            return None
        ang = angle_vectors(vec_l1[0], vec_l2[0])
        return ang, ref_point, vec_l1, vec_l2

    def fn(x):
        _, _, tfp_1, tfp_2 = compute_third_tan_pts(x)
        if tfp_1:
            vec_l1 = tfp_1[0]
        else:
            print("problem in opt 3 - 1")
            f = 180
            return f
        if tfp_2:
        #     vec_l2 = tfp_2[ind_2]
            vec_l2 = tfp_2[0]
        else:
            print("problem in opt 3 - 1")
            f = 180
            return f
        ang_v = angle_vectors(vec_l1, vec_l2, deg=True)
        if 180 - ang_v < 90:
            f = 180 - ang_v
        else:
            f = ang_v
        return f

    res_opt = scipy.optimize.fmin(fn, [0.0, 0.0], full_output=True, disp=0 if not debug else 3)
    if res_opt[1] > 0.1:
        return None

    # ret_fp3 = find_point_3(list(map(float, res_opt[0])), *args)
    ret_fp3 = compute_third_tan_pts(list(map(float, res_opt[0])))
    if not ret_fp3:
        return None
    else:
        ang, ref_point, vec_l1, vec_l2 = ret_fp3
        return ref_point, vec_l1[0], vec_l2[0], ang

##########################################

def check_colisions(b_struct, pts, radius, bar_nb=None, bar_checking=None):
    """SP's geometric collision checking function

    Parameters
    ----------
    b_struct : [type]
        [description]
    pts : [type]
        [description]
    radius : [type]
        [description]
    bar_nb : int, optional
        bar index to check against UNTIL, by default None
    bar_checking : [type], optional
        [description], by default None

    Returns
    -------
    bool
        True if no collision found, False otherwise
    """

    tol = TOL # | 50
    # print "bar_checking", bar_checking
    for b in b_struct.vertex:
        if not bar_nb:
            bar_nb = 1e14
        if bar_checking != None and b < 3:
            continue
        if b < bar_nb and b != bar_checking:
            pts_b = b_struct.vertex[b]["axis_endpoints"]
            dpp = dropped_perpendicular_points(pts[0], pts[1], pts_b[0], pts_b[1])
            dist = distance_point_point(*dpp)

            if 2*radius - dist > TOL and \
               is_point_on_segment(dpp[0], pts, tol=tol) and \
               is_point_on_segment(dpp[1], pts_b, tol=tol):
                # print("COLLISION: ", len(b_struct.vertex))
                return False
    return True
