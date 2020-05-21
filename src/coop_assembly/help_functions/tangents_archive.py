def lines_tangent_to_two_cylinder(base_point1, line_vect1, base_point2, line_vect2, ref_point, dist1, dist2):

    planes1     = p_planes_tangent_to_cylinder(base_point1, line_vect1, ref_point, dist1)
    planes2     = c_planes_tangent_to_cylinder(base_point2, line_vect2, ref_point, dist2)
    if planes1 == None or planes2 == None:
        return None
    s1  = intersect_plane_plane_u(planes1[0][1], planes1[0][2], planes2[0][0])
    s1  = normalize_vector(s1)
    s2  = intersect_plane_plane_u(planes1[0][1], planes1[0][2], planes2[1][0])
    s2  = normalize_vector(s2)
    s3  = intersect_plane_plane_u(planes1[1][1], planes1[1][2], planes2[0][0])
    s3  = normalize_vector(s3)
    s4  = intersect_plane_plane_u(planes1[1][1], planes1[1][2], planes2[1][0])
    s4  = normalize_vector(s4)

    return [s1, s2, s3, s4]

def tangent_from_point(base_point1, line_vect1, base_point2, line_vect2, ref_point, dist1, dist2):
    solutions = lines_tangent_to_two_cylinder(base_point1, line_vect1, base_point2, line_vect2, ref_point, dist1, dist2)
    if solutions == None:
        print("no solutions for tangent_from_point")
        return None
    sol_0   = solutions[0]
    sol_1   = solutions[1]
    sol_2   = solutions[2]
    sol_3   = solutions[3]

    return [sol_0, sol_1, sol_2, sol_3]

def tangent_through_two_points(base_point1, line_vect1, ref_point1, base_point2, line_vect2, ref_point2, dist1, dist2):
    ind = [0,1]
    sols_vec = []
    sols_pts = []
    # print("tangent_through_two_points", base_point1, line_vect1, ref_point1, base_point2, line_vect2, ref_point2, dist1, dist2)
    for i in ind:
        ret_p1 = p_planes_tangent_to_cylinder(base_point1, line_vect1, ref_point2, dist1 + dist2 + dist1 + dist2)
        ret1 = ret_p1[i]
        z_vec = cross_vectors(line_vect1, ret1[2])
        plane1 = (ret1[0], z_vec)
        # print("plane1", plane1)
        pp1 = project_points_plane([ref_point1], plane1)[0]
        vec_move = scale_vector(subtract_vectors(ref_point1, pp1), 0.5)
        new_pt = add_vectors(pp1, vec_move)

        for j in ind:
            ret_p2 = p_planes_tangent_to_cylinder(base_point2, line_vect2, ref_point1, dist1 + dist2 + dist1 + dist2)
            ret2 = ret_p2[j]
            z_vec = cross_vectors(line_vect2, ret2[2])
            plane2 = (ret2[0], z_vec)
            pp2 = project_points_plane([ref_point2], plane2)[0]
            vec_move = scale_vector(subtract_vectors(ref_point2, pp2), 0.5)
            pt2 = add_vectors(pp2, vec_move)

            sols_pts.append([new_pt, pt2])
            sol_vec = subtract_vectors(new_pt, pt2)
            sols_vec.append(sol_vec)
    return sols_pts

