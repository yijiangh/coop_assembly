
'''
created on 28.08.2019
'''

from __future__ import print_function

import random
import itertools
import math

from compas.geometry import Transformation, Frame, Rotation, Translation
from compas.geometry.basic import add_vectors, normalize_vector, vector_from_points, scale_vector, cross_vectors, subtract_vectors,\
    length_vector, length_vector, norm_vector
from compas.geometry.distance import distance_point_point, distance_point_line, distance_line_line, closest_point_on_line
from compas.geometry.transformations import rotate_points
from compas.geometry.angles import angle_vectors
from compas.geometry.average import centroid_points
from compas.geometry import translate_points, rotate_points_xy
from compas.geometry.queries import is_point_on_line

from coop_assembly.help_functions.helpers_geometry import calculate_bar_z, calculate_coord_sys, Frame_to_plane_data
from coop_assembly.help_functions.debug_utils import deprecation_error
from coop_assembly.help_functions.shared_const import TOL, IPY, EPS
from coop_assembly.help_functions.drawing import get_ideal_vert_keys_from_bar_vertex_key

def scale_frame(cp_plane, scale):
    return [scale_vector(cp_plane[0], scale)] + list(cp_plane[1:4])

def calculate_gripping_plane(b_struct, v, pt_mean, nb_rot=8, nb_trans=8,
    planes_rot=True, planes_trans=True):
    """calculate gripping planes for a given bar structure and the vertex key (representing a bar)

    Parameters
    ----------
    b_struct : [type]
        [description]
    v : [type]
        [description]
    pt_mean : [type]
        [description]
    nb_rot : int, optional
        number of rotational division, by default 8
    nb_trans : int, optional
        number of translational division, by default 8
    planes_rot : bool, optional
        [description], by default True
    planes_trans : bool, optional
        [description], by default True
    """

    end_pts_0 = b_struct.vertex[v]["axis_endpoints"]
    # local coordinate system
    vec_x, vec_y, vec_z = calculate_coord_sys(end_pts_0, pt_mean)
    # mid point of the bar axis
    pt_o = centroid_points(end_pts_0)

    b_struct.vertex[v].update({"gripping_plane":(pt_o, vec_x, vec_y, vec_z)})
    gripping_plane = b_struct.vertex[v]["gripping_plane"]

    frames_all = []

    if planes_trans:
        # extend both end points for 30 mm
        vec_bar = scale_vector(normalize_vector(subtract_vectors(end_pts_0[1], end_pts_0[0])), 30)
        pt1 = add_vectors(end_pts_0[0], vec_bar)
        vec_bar = scale_vector(vec_bar, -1)
        pt2 = add_vectors(end_pts_0[1], vec_bar)
        vec_n = subtract_vectors(pt2, pt1)
        len_vec = length_vector(vec_n)
        len_new = len_vec/(nb_trans-1)

        for i in range(nb_trans):
            origin = add_vectors(pt1, scale_vector(normalize_vector(vec_n), len_new*i))
            frame_n = [origin, gripping_plane[1], gripping_plane[2]]
            if not planes_rot:
                frames_all.append(frame_n)
                # if planes_flip == True:
                #     frame_n = [frame_n[0], scale_vector(frame_n[1], -1), scale_vector(frame_n[2], -1)]
                #     frames_all.append(frame_n)
            else:
                ang = math.radians(360/nb_rot)
                for n in range(nb_rot):
                    gripping_plane = frame_n
                    vecs_n = rotate_points([gripping_plane[1], gripping_plane[2]], angle=n*ang, axis=subtract_vectors(end_pts_0[1], end_pts_0[0]), origin=(0,0,0))
                    frame_n = [gripping_plane[0], vecs_n[0], vecs_n[1]]
                    frames_all.append(frame_n)
                    # if planes_flip == True:
                    #     frame_n = [frame_n[0], scale_vector(frame_n[1], -1), scale_vector(frame_n[2], -1)]
                    #     frames_all.append(frame_n)
    elif planes_rot:
        ang = math.radians(360/nb_rot)
        for n in range(nb_rot):
            vecs_n = rotate_points([gripping_plane[1], gripping_plane[2]], angle=n*ang, axis=subtract_vectors(end_pts_0[1], end_pts_0[0]), origin=(0,0,0))
            frame_n = [gripping_plane[0], vecs_n[0], vecs_n[1]]
            frames_all.append(frame_n)

            # if planes_flip == True:
            #     frame_n = [frame_n[0], scale_vector(frame_n[1], -1), scale_vector(frame_n[2], -1)]
            #     frames_all.append(frame_n)

    for i,f in enumerate(frames_all):
        z_vec = cross_vectors(f[1], f[2])
        frames_all[i].append(z_vec)

    b_struct.vertex[v].update({"gripping_planes_all" : frames_all})

def calculate_offset(o_struct, b_struct, bar_vkey, rot_angle=math.pi/6, trans_distance=30, seq=None, scale=1.0):
    """[summary]

    Example usage:
        by default, seq = [v for v in b_struct.vertex]
        for v in b_struct.vertex:
            pts.append(rg.Point3d(*b_struct.vertex[v]["mean_point"]))
            calculate_gripping_plane(b_struct, v, b_struct.vertex[v]["mean_point"], nb_rot=nb_rot, nb_trans=nb_trans)
            calculate_offset(o_struct, b_struct, v, offset_d1, offset_d2, seq)

    TODO: we can perform motion planning to solve for local disassembly motion

    Parameters
    ----------
    o_struct : [type]
        [description]
    b_struct : [type]
        [description]
    v_key : [type]
        vertex key in BarStructure, representing a physical bar
    d_o_1 : [type]
        [description]
    d_o_2 : [type]
        [description]
    seq : [type]
        [description]
    """
    seq = seq or list(range(bar_vkey))
    v_pos = seq.index(bar_vkey)
    int_v = 2 - v_pos % 3
    v_pos_max = v_pos + int_v # maximal bar index in the three-bar group
    assembled_bv = seq[0:v_pos_max+1]

    # * Find v_key bar's corresponding edge in OverallStructure
    o_edge_from_bv = get_ideal_vert_keys_from_bar_vertex_key(o_struct)
    o_edge = o_edge_from_bv[bar_vkey]

    # OverallStructure's edges are bars, find this bar's two end points' connected
    b_edge_from_o1 = find_connectors(o_struct, o_edge[0])
    b_edge_from_o2 = find_connectors(o_struct, o_edge[1])

    b_edge_assembled_from_o1 = [c for c in b_edge_from_o1 if c[0] in assembled_bv and c[1] in assembled_bv and (c[0] == bar_vkey or c[1] == bar_vkey)]
    b_edge_assembled_from_o2 = [c for c in b_edge_from_o2 if c[0] in assembled_bv and c[1] in assembled_bv and (c[0] == bar_vkey or c[1] == bar_vkey)]

    bar_1 = b_struct.vertex[bar_vkey]["axis_endpoints"]

    contact_vecs_from_o1 = [] # vectors of all connections to the bar in endpoint 1
    contact_pts_from_o1 = []
    contact_projected_pts_from_o1 = [] # points of connections on bar axis
    for c in b_edge_assembled_from_o1:
        # ep = b_struct.edge[c[0]][c[1]]["endpoints"][list(b_struct.edge[c[0]][c[1]]["endpoints"].keys())[0]]
        ep = list(b_struct.edge[c[0]][c[1]]["endpoints"].values())[0]
        if is_point_on_line(ep[0], bar_1, TOL):
            contact_vecs_from_o1.append(vector_from_points(ep[0], ep[1]))
            contact_projected_pts_from_o1.append(ep[0])
            contact_pts_from_o1.append(ep[1])
        elif is_point_on_line(ep[1], bar_1, TOL):
            contact_vecs_from_o1.append(vector_from_points(ep[1], ep[0]))
            contact_projected_pts_from_o1.append(ep[1])
            contact_pts_from_o1.append(ep[0])
        else:
            raise RuntimeError("Connector (BarS edge) |{}| end points not on bar {} axis".format(c, bar_vkey))

    # contact normals (from axis pt to contact pt)
    contact_vecs_from_o2 = [] # vectors of all connections to the bar in endpoint 2
    contact_pts_from_o2 = []
    # contact points projected on the axis
    contact_projected_pts_from_o2  = [] # points of connections on bar axis
    for c in b_edge_assembled_from_o2:
        # ep = b_struct.edge[c[0]][c[1]]["endpoints"][b_struct.edge[c[0]][c[1]]["endpoints"].keys()[0]]
        ep = list(b_struct.edge[c[0]][c[1]]["endpoints"].values())[0]
        if is_point_on_line(ep[0], bar_1, 0.1):
            contact_vecs_from_o2.append(vector_from_points(ep[0], ep[1]))
            contact_projected_pts_from_o2.append(ep[0])
            contact_pts_from_o2.append(ep[1])
        elif is_point_on_line(ep[1], bar_1, 0.1):
            contact_vecs_from_o2.append(vector_from_points(ep[1], ep[0]))
            contact_projected_pts_from_o2.append(ep[1])
            contact_pts_from_o2.append(ep[0])
        else:
            raise RuntimeError("no point found on axis - check function calculate_offset")

    # # body_pose = get_pose(b_struct.get_bar_pb_body(bar_vkey))
    # body_pose = scale_frame(b_struct.vertex[bar_vkey]["gripping_plane"], scale)
    # contact_pts = [scale_vector(p, scale) for p in contact_pts_from_o1 + contact_pts_from_o2]
    # contact_normals = [scale_vector(v, scale) for v in contact_vecs_from_o1 + contact_vecs_from_o2]

    # if not IPY:
    #     import numpy as np
    #     from numpy.linalg import norm
    #     import coop_assembly.assembly_info_generation.interlock as ipc
    # else:
    #     from compas.rpc import Proxy
    #     ipc = Proxy('coop_assembly.assembly_info_generation.interlock')

    # contact_change_dirs, _ = ipc.compute_local_disassembly_motion(body_pose, contact_pts, contact_normals)
    # w = 1.0
    # sum_v = [0,0,0,0,0,0]
    # for v in contact_change_dirs:
    #     sum_v = add_vectors(scale_vector(v, w), sum_v)

    # if norm_vector(sum_v[3:]) > EPS:
    #     print('sum_v: {} | {}'.format(sum_v, body_pose[0]))
    #     # axis, pt = velocity_to_rotation_fn(sum_v, body_pose[0])
    #     axis, pt = ipc.velocity_to_rotation(sum_v, body_pose[0])
    #     tf = Rotation.from_axis_and_angle(axis, rot_angle, point=pt)
    # else:
    #     tf = Translation(scale_vector(sum_v[:3], trans_distance*scale))

    d_o_1 = trans_distance
    d_o_2 = trans_distance
    # TODO: if grounded triangle, simply pregrasp from above

    ### calculate offset for first three bars (with one neighbour each)
    if len(contact_vecs_from_o1) == 1 and len(contact_vecs_from_o2) == 1:
        v1 = normalize_vector(contact_vecs_from_o1[0])
        v2 = normalize_vector(contact_vecs_from_o2[0])
        # same_dir    = check_dir(v1, v2)

        if angle_vectors(v1, v2, deg=True) < 90:
            # not locked on the both sides, translation-only
            vm      = scale_vector(normalize_vector(add_vectors(v1, v2)), -1.*d_o_1)
            # shift gripping plane
            pt_o    = b_struct.vertex[bar_vkey]["gripping_plane"][0]
            x_ax    = b_struct.vertex[bar_vkey]["gripping_plane"][1]
            y_ax    = b_struct.vertex[bar_vkey]["gripping_plane"][2]
            z_ax    = b_struct.vertex[bar_vkey]["gripping_plane"][3]
            pt_o_n  = translate_points([pt_o], vm)[0]
            b_struct.vertex[bar_vkey].update({"gripping_plane_offset":(pt_o_n, x_ax, y_ax, z_ax)})
        else:
            # not locked on the both sides, translation-only
            pt_1    = contact_pts_from_o1[0]
            pt_2    = contact_pts_from_o2[0]
            pt_o_n, vec_x_n, y_ax, vec_z = calculate_offset_pos_two_side_one_point_locked(b_struct, bar_vkey, pt_1, pt_2, v1, v2, d_o_1, d_o_2)
            #pt_o_n  = point_mean([pt_1_n, pt_2_n])
            b_struct.vertex[bar_vkey].update({"gripping_plane_offset":(pt_o_n, vec_x_n, y_ax, vec_z)})
    ### calculate offset for bars with neighbours only on one side
    elif (len(contact_vecs_from_o1) == 1 and len(contact_vecs_from_o2) == 0) or \
       (len(contact_vecs_from_o2) == 1 and len(contact_vecs_from_o1) == 0):
        if len(contact_vecs_from_o1) == 1:
            v1 = normalize_vector(contact_vecs_from_o1[0])
        else:
            v1 = normalize_vector(contact_vecs_from_o2[0])
        vm = scale_vector(v1, -1.*d_o_1)
        pt_o = b_struct.vertex[bar_vkey]["gripping_plane"][0]
        x_ax = b_struct.vertex[bar_vkey]["gripping_plane"][1]
        y_ax = b_struct.vertex[bar_vkey]["gripping_plane"][2]
        z_ax = b_struct.vertex[bar_vkey]["gripping_plane"][3]
        pt_o_n  = translate_points([pt_o], vm)[0]
        b_struct.vertex[bar_vkey].update({"gripping_plane_offset":(pt_o_n, x_ax, y_ax, z_ax)})
    elif (len(contact_vecs_from_o1) == 2 and len(contact_vecs_from_o2) == 0) or \
       (len(contact_vecs_from_o2) == 2 and len(contact_vecs_from_o1) == 0):
        # not locked on the both sides, translation-only
        if len(contact_projected_pts_from_o1) == 2:
            v1 = normalize_vector(contact_vecs_from_o1[0])
            v2 = normalize_vector(contact_vecs_from_o1[1])
        else:
            v1 = normalize_vector(contact_vecs_from_o2[0])
            v2 = normalize_vector(contact_vecs_from_o2[1])
        vm      = scale_vector(normalize_vector(add_vectors(v1, v2)), -1.*d_o_1)
        # shift gripping plane
        pt_o    = b_struct.vertex[bar_vkey]["gripping_plane"][0]
        x_ax    = b_struct.vertex[bar_vkey]["gripping_plane"][1]
        y_ax    = b_struct.vertex[bar_vkey]["gripping_plane"][2]
        z_ax    = b_struct.vertex[bar_vkey]["gripping_plane"][3]
        pt_o_n  = translate_points([pt_o], vm)[0]
        b_struct.vertex[bar_vkey].update({"gripping_plane_offset":(pt_o_n, x_ax, y_ax, z_ax)})
    ### calculate offset for other bars (with two neighbours each)
    elif len(contact_vecs_from_o1) == 2 and len(contact_vecs_from_o2) == 2:
        pt_o_n, vec_x_n, y_ax, vec_z  = calculate_offset_pos_two_side_two_point_locked(b_struct, bar_vkey, \
            contact_vecs_from_o1, contact_vecs_from_o2, contact_pts_from_o1, contact_pts_from_o2, d_o_1, d_o_2)

        #pt_o_n  = point_mean([pt_1_n, pt_2_n])
        b_struct.vertex[bar_vkey].update({"gripping_plane_offset":(pt_o_n, vec_x_n, y_ax, vec_z)})
        # return pt_o_n, vec_x_n, y_ax, vec_z

    gripping_frame = Frame(*scale_frame(b_struct.vertex[bar_vkey]["gripping_plane"], scale)[0:3])
    gripping_frame_offset = Frame(*scale_frame(b_struct.vertex[bar_vkey]["gripping_plane_offset"], scale)[0:3])
    # gripping_frame_offset = gripping_frame.transformed(tf)
    # b_struct.vertex[bar_vkey].update({"gripping_plane_offset":Frame_to_plane_data(gripping_frame_offset)})

    # * gripping_planes_all by applying transformation from gripping_plane
    tf = Transformation.from_frame_to_frame(gripping_frame, gripping_frame_offset)
    gripping_frames_all = [Frame(*scale_frame(plane, scale)[0:3]).transformed(tf) for plane in b_struct.vertex[bar_vkey]["gripping_planes_all"]]
    b_struct.vertex[bar_vkey].update({"gripping_planes_offset_all" : [Frame_to_plane_data(frame) for frame in gripping_frames_all]})

    # contact point projection on the central axis
    # vector connecting projected points on the bars
    # return contact_projected_pts_from_o1, contact_vecs_from_o1, contact_projected_pts_from_o2, contact_vecs_from_o2
    return tf

def calculate_offset_pos_two_side_one_point_locked(b_struct, v_key, pt_1, pt_2, v1, v2, d_o_1, d_o_2):
    """calculate offsetted plane when the bar's both sides are blocked by vector v1 and v2

    # ! Note: the old y axis is kept in this function, local x axis is updated

    Parameters
    ----------
    b_struct : [type]
        [description]
    v_key : int
        vertex key in BarStructure, representing a physical bar
    pt_1 : list
        first contact point's projection on the bar's axis
    pt_2 : list
        second contact point's projection on the bar's axis
    v1 : list
        first contact point - contact point vector
    v2 : list
        second contact point - contact point vector
    d_o_1 : float
        offset distance for end point #1
    d_o_2 : float
        offset distance for end point #2

    Returns
    -------
    tuple
        offset plane's origin, x-, y-, z-axis
    """

    pt_1_new  = add_vectors(pt_1, scale_vector(v1, -1.*d_o_1))
    pt_2_new  = add_vectors(pt_2, scale_vector(v2, -1.*d_o_2))

    vec_x_new = normalize_vector(vector_from_points(pt_1_new, pt_2_new))
    x_ax    = b_struct.vertex[v_key]["gripping_plane"][1]

    if not angle_vectors(x_ax, vec_x_new, deg=True) < 90:
        vec_x_new = scale_vector(vec_x_new, -1.)

    # transform gripping plane
    pt_o    = b_struct.vertex[v_key]["gripping_plane"][0]
    y_ax    = b_struct.vertex[v_key]["gripping_plane"][2]
    vec_z   = cross_vectors(vec_x_new, y_ax)
    l_n = (pt_1_new, pt_2_new)
    pt_o_new  = closest_point_on_line(pt_o, l_n)

    return pt_o_new, vec_x_new, y_ax, vec_z

def calculate_offset_pos_two_side_two_point_locked(b_struct, v_key, vecs_con_1, vecs_con_2, pts_con_1, pts_con_2, d_o_1, d_o_2):
    """calculate offsetted plane when the bar's both ends have two contact points

    """
    assert len(vecs_con_1) == 2 and len(pts_con_1) == 2
    assert len(vecs_con_2) == 2 and len(pts_con_2) == 2

    map(normalize_vector, vecs_con_1)
    map(normalize_vector, vecs_con_2)
    v1_1, v1_2 = vecs_con_1
    v2_1, v2_2 = vecs_con_2
    pt_1_1, pt_1_2 = pts_con_1
    pt_2_1, pt_2_2 = pts_con_2

    vm_1    = scale_vector(normalize_vector(add_vectors(v1_1, v1_2)), -1.*d_o_1)
    # original contact point (assuming two bars have the same radius)
    pt_1    = centroid_points([pt_1_1, pt_1_2])
    pt_1_new  = translate_points([pt_1], vm_1)[0]

    vm_2    = scale_vector(normalize_vector(add_vectors(v2_1, v2_2)), -1.*d_o_2)
    pt_2    = centroid_points([pt_2_1, pt_2_2])
    pt_2_new  = translate_points([pt_2], vm_2)[0]

    vec_x_new = normalize_vector(vector_from_points(pt_1_new, pt_2_new))
    x_ax    = b_struct.vertex[v_key]["gripping_plane"][1]

    if not angle_vectors(x_ax, vec_x_new, deg=True) < 90:
        vec_x_new = scale_vector(vec_x_new, -1.)

    pt_o    = b_struct.vertex[v_key]["gripping_plane"][0]
    y_ax    = b_struct.vertex[v_key]["gripping_plane"][2]
    vec_z   = cross_vectors(vec_x_new, y_ax)
    l_n = (pt_1_new, pt_2_new)
    pt_o_n  = closest_point_on_line(pt_o, l_n)

    return pt_o_n, vec_x_new, y_ax, vec_z

def find_connectors(o_struct, o_node_key):
    """Find BarS vertex keys that have multiple joints in the given o_node_key's ideal vertex

    Parameters
    ----------
    o_struct : OverallStructure
        [description]
    o_node_key : int
        OverallStructure's vertex key (ideal vertex where bars are joined together)

    Returns
    -------
    list of int
        BarS's vertex keys,
    """

    #self.vertex[n_key]
    # connected edges (bars) to the ideal vertex o_node_key
    o_edges = o_struct.vertex_connected_edges(o_node_key)
    # connected bars corresponding vertex key in BarStructure
    b_vertices = [o_struct.edge[e[0]][e[1]]["vertex_bar"] for e in o_edges]

    # BarStructure's edges describe the joints between pairs of bars
    b_edges = []
    for b_vert in b_vertices:
        b_edges.append(o_struct.struct_bar.vertex_connected_edges(b_vert))

    # select bars (BarS's vertices) that have more than one joints (BarS's edges)
    common_e = []
    for i, e1_edges in enumerate(b_edges):
        for j, e2_edges in enumerate(b_edges):
            if j > i:
                for e1 in e1_edges:
                    for e2 in e2_edges:
                        if e1 == e2 or e1 == e2[::-1]:
                            common_e.append(e1)
    return common_e
