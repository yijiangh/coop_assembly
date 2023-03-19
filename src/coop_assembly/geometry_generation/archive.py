import numpy as np
from collections import defaultdict
from numpy.linalg import norm as npnorm
from multi_tangent.utils import get_edge_direction

def computeOffsetBeamConstraintsPairs(vertices, edges, sinEps=1e-3):
    """_summary_
    Parameters
    ----------
    vertices: num_nodes x 3 np array
    edges:    num_edges x 2 np array
    sinEps : float, optional
        eps value for determining line parellel, by default 1e-3
    Returns
    -------
    mat_non_parallel_e_pairs
        2 x pair_size int matrix, non parallel connected line pair ids
    non_parallel_e_signs
        pair_size int list, non parallel connected line pair's sign, can take value 1 () or -1 ()
    mat_parallel_e_pairs
        2 x pair_size int matrix: parallel connected line pair ids
    mat_parallel_e_perp_dirs
        3 x pair_size int matrix: parallel connected line pair's perpedicular directions
    """
    beam_ids_from_vertex = defaultdict(set)
    for e_id, e in enumerate(edges):
        beam_ids_from_vertex[e[0]].add(e_id)
        beam_ids_from_vertex[e[1]].add(e_id)

    non_parallel_e_pairs = []
    non_parallel_e_signs = []
    parallel_e_pairs = [] 
    parallel_e_perp_dirs = []
    for v_id, v_edge_set in beam_ids_from_vertex.items():
        v_edges = list(v_edge_set)
        # * valence-2 vertex
        if len(v_edges) == 1:
            pass
        elif len(v_edges) == 2:
            e0, e1 = v_edges
            d0 = get_edge_direction(edges[e0,:], vertices)
            d1 = get_edge_direction(edges[e1,:], vertices)
            # if two lines are in parellel
            if npnorm(np.cross(d0, d1)) < sinEps:
                drt = np.array([0,0,1.0])
                # find a direction that's not parallel to d0
                if npnorm(np.cross(drt,d0)) < sinEps:
                    drt = np.array([1.0,0,0])
                parallel_e_pairs.append([e0,e1])
                parallel_e_perp_dirs.append(drt)
            else: # non-parellel
                non_parallel_e_pairs.append((e0,e1))
                non_parallel_e_signs.append(1)
        elif len(v_edges) == 3:
            # * valence-3 vertex
            parallel_index = -1
            for kd in range(len(v_edges)):
                e0 = v_edges[kd]
                e1 = v_edges[(kd+1) % 3]
                d0 = get_edge_direction(edges[e0,:], vertices)
                d1 = get_edge_direction(edges[e1,:], vertices)
                # record parallel_index at the first pair of parallel pair
                if npnorm(np.cross(d0, d1)) < sinEps:
                    parallel_index = kd
                    break

            # * none of the pair in the ring is parallel
            if parallel_index == -1:
                nr = np.zeros(3)
                for kd in range(3):
                    e0 = v_edges[kd]
                    e1 = v_edges[(kd+1)%3]
                    d0 = get_edge_direction(edges[e0,:], vertices)
                    d1 = get_edge_direction(edges[e1,:], vertices)
                    n = np.cross(d0,d1)
                    if np.dot(nr, n) >= 0:
                        non_parallel_e_signs.append(1)
                    else:
                        non_parallel_e_signs.append(-1)
                    non_parallel_e_pairs.append((e0,e1))
                    nr = n
            else:
                # * at least one pair in the ring is parellel
                e0 = v_edges[parallel_index]
                e1 = v_edges[(parallel_index+1)%3]
                e2 = v_edges[(parallel_index+2)%3]
                d0, d1, d2 = list(map(lambda x : get_edge_direction(edges[x,:], vertices), [e0,e1,e2]))
                # n = np.cross(d0,d1) # d0, d1 is parallel
                n12 = np.cross(d1,d2)
                n20 = np.cross(d2,d0)

                drt = np.array([0,0,1.0])
                # find a direction that's not parallel to d0
                if npnorm(np.cross(drt,d0)) < sinEps:
                    drt = np.array([1.0,0,0])
                    if npnorm(np.cross(drt,d0)) < sinEps:
                        raise ValueError('z and x axes both parallel')
                parallel_e_pairs.append([e0,e1])
                parallel_e_perp_dirs.append(drt)

                non_parallel_e_pairs.append((e1,e2))
                non_parallel_e_pairs.append((e2,e0))
                non_parallel_e_signs.append(1)
                if np.dot(n12, n20) >= 0:
                    non_parallel_e_signs.append(1)
                else:
                    non_parallel_e_signs.append(-1)
                    # non_parallel_e_signs.append(1)
        else:
            pass

    mat_non_parallel_e_pairs = np.empty((2, len(non_parallel_e_pairs)), dtype=int)
    for i, ep in enumerate(non_parallel_e_pairs):
        mat_non_parallel_e_pairs[:,i] = ep

    mat_parallel_e_pairs = np.empty((2, len(parallel_e_pairs)), dtype=int)
    for i, ep in enumerate(parallel_e_pairs):
        mat_parallel_e_pairs[:,i] = ep

    mat_parallel_e_perp_dirs = np.empty((3, len(parallel_e_perp_dirs)), dtype=float)
    for i, ed in enumerate(parallel_e_perp_dirs):
        mat_parallel_e_perp_dirs[:,i] = ed

    return mat_non_parallel_e_pairs, non_parallel_e_signs, \
        mat_parallel_e_pairs, mat_parallel_e_perp_dirs

def compute_tangent_bar(bar_from_elements, node_points, element, in_contact_bars, radius):
    """Compute double-tangent bar position for `element` using pure geometric logic

    Parameters
    ----------
    bar_from_elements : dict
        {[element] : { node id : axis pt1}}
        element = (n1, n2), n1/2 indexs into `node_points`
    node_points : list of points
        the original list of points we want the bar system to achieve. We are pertubing the bar axis to approximately spanning these points.
    element : tuple of int
        (n1, n2) index into `node_points`
    in_contact_bars : 2-tuple
        in-contact neighbor bars: ((e1_n1, e1_n2), (e2_n1, e2_n2))
        when no bar is assigned at one end, the tuple can be empty: ((), (e2_n1, e2_n2))
    radius : float
        bar radius in millimeter

    Returns
    -------
    update_element_axis_pts: dict
        updated contact element axis points
        {(e_n1, e_n2) : [axis pt 1, axis pt 2]}
    """
    # order the contact bar pairs based on number of bars in each entry
    contact_bars = copy(in_contact_bars)
    # smaller contact end point always comes first
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
    cprint('Valence-0 cantilevering point: {}'.format(new_point))

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

    # TODO these are pasted from the for-loop, should fix this
    new_axis_endpts = {new_node_id : new_point, contact_v_id : np.array(contact_pt)}
    contact_elements = contact_e

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

        # * new contact pts
        new_contact_pts = dropped_perpendicular_points(*new_axis_endpts.values(),
                                                       *bar_from_elements[contact_e].values())
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

    update_element_axis_pts = chosen_contact_element_axis_pts
    update_element_axis_pts[element] = [chosen_new_axis_endpts[element[0]], chosen_new_axis_endpts[element[1]]]
    return update_element_axis_pts

def beam_opt_tangent_bar(bar_from_elements, node_points, element, in_contact_bars, radius, scale=1.0, debug=False):
    """
    Inputs
    -------
    scale:
        scale for converting length unit, opt better operates in meter scale

    Returns
    -------
    update_element_axis_pts: dict
        updated contact element axis points
        {(e_n1, e_n2) : [axis pt 1, axis pt 2]}
    """
    subassembly_edges = [element]
    for contact_bars_at_node in in_contact_bars:
        for e in contact_bars_at_node:
            if e != ():
                subassembly_edges.append(e)
    if len(subassembly_edges) == 1:
        return {}, []

    # * renumbering the verts and edges for the subassembly
    vert_old_inds = list(nodes_from_elements(subassembly_edges))
    local_verts = np.array(node_points)[vert_old_inds]
    local_edges = []
    new_edge_from_old = {}
    for i, e in enumerate(subassembly_edges):
        local_edges.append((vert_old_inds.index(e[0]), vert_old_inds.index(e[1])))
        new_edge_from_old[e] = i
    local_edges = np.array(local_edges, dtype=int)

    beam_opt = BeamOpt(local_verts*scale, local_edges, [], radius*scale)
    beam_opt.weights =[1, 200, 1]

    # fill in current bar end point as initial guess
    initial_vert_points = np.array(beam_opt.vertices[beam_opt.edges])
    for i, ei in enumerate(subassembly_edges):
        initial_vert_points[i,0] = np.array(bar_from_elements[ei][ei[0]]) * scale
        initial_vert_points[i,1] = np.array(bar_from_elements[ei][ei[1]]) * scale
    beam_opt.end_pts_init_positions = jnp.array(initial_vert_points.flatten())

    var = beam_opt.computeInitVar()
    fun = computeBeamOptObjective(var, beam_opt)
    grad = computeBeamOptObjectiveGradient(var, beam_opt)
    v = jnp.ones(beam_opt.nP + beam_opt.nT)
    hessian = computeBeamOptObjectiveHessianP(var, v, beam_opt)

    new_vertices, new_edges, new_supports = beam_opt.runOptimization(trace=True)
    # new_V = beam_opt.xstar[beam_opt.nT: beam_opt.nP + beam_opt.nT].reshape(beam_opt.nB * 2, 3)
    assert new_edges.shape[0] == len(subassembly_edges)

    # plot_beam_opt_per_pair(beam_opt)
    # plot_f_trace(beam_opt)
    # plt.show()

    contact_handles = []
    if True:
        # * draw contact lines
        neighbors = get_element_neighbors(subassembly_edges)
        for i, ei in enumerate(subassembly_edges):
            new_ei = new_edges[new_edge_from_old[ei]]
            for nei in neighbors[ei]:
                new_nei = new_edges[new_edge_from_old[nei]]
                dpp = closest_point_segments(new_vertices[new_ei[0]], new_vertices[new_ei[1]],
                                             new_vertices[new_nei[0]], new_vertices[new_nei[1]])
                print('contact len: {:.4f} | target {:.4f}'.format(norm(dpp[0]-dpp[1])*(1/scale), radius*2))
                contact_handles.append(pp.add_line(dpp[0], dpp[1], color=pp.apply_alpha(pp.GREEN, 1.0), width=1.5))

    update_element_axis_pts = {}
    for i, (ei, new_ei) in enumerate(zip(subassembly_edges, new_edges)):
        update_element_axis_pts[ei] = {
            ei[0] : ((1/scale)*new_vertices[new_ei[0],:]).tolist(),
            ei[1] : ((1/scale)*new_vertices[new_ei[1],:]).tolist(),
            }
    return update_element_axis_pts, contact_handles

