from termcolor import cprint
import numpy as np

from compas.geometry import Transformation

from pybullet_planning import get_floating_body_collision_fn, multiply, get_pose, WorldSaver, set_pose, wait_for_user, \
    set_color, apply_alpha, remove_all_debug, interval_generator, add_line, draw_point, remove_handles, interpolate_poses, \
    get_distance
from pybullet_planning import Euler, Pose
from pybullet_planning import RED, BLUE
from compas_fab.backends.pybullet import pb_pose_from_Transformation, Frame_from_pb_pose

def get_pose_generator(epsilon, angle=np.pi/2, **kwargs):
    lower = [-epsilon]*3 + [-angle]*3
    upper = [epsilon]*3 + [angle]*3
    for [x, y, z, roll, pitch, yaw] in interval_generator(lower, upper, **kwargs):
        pose = Pose(point=[x,y,z], euler=Euler(roll=roll, pitch=pitch, yaw=yaw))
        yield pose

######################################

def sample_tf(body, obstacles, epsilon=10*1e-3, angle=np.pi/2, max_attempts=50, debug=False, max_distance=0, \
    **kwargs):
    pose_gen = get_pose_generator(epsilon)
    collision_fn = get_offset_collision_test(body, obstacles, max_distance=max_distance)
    world_from_bar = get_pose(body)
    for i in range(max_attempts):
        delta_pose = next(pose_gen)
        offset_pose = multiply(world_from_bar, delta_pose)
        # print('-------------')
        # print('attempt #', i)
        # if not collision_fn(offset_pose, diagnosis):
        #     return Transformation.from_frame(Frame_from_pb_pose(delta_pose))
        is_colliding = False
        offset_path = list(interpolate_poses(offset_pose, world_from_bar, **kwargs))
        for p in offset_path[:-1]: # *kwargs
            # TODO: if colliding at the world_from_bar pose, use local velocity + normal check
            # TODO: normal can be derived from
            # if get_distance(p[0], world_from_bar[0]) > 1e-3:
            if collision_fn(p):
                is_colliding = True
                break
        if not is_colliding:
            return offset_path
    return None

def offset_tf_from_contact(bar_vertex, contact_vecs_from_o1, contact_vecs_from_o2,
    contact_pts_from_o1, contact_pts_from_o2, rot_angle, trans_distance, scale=1.0, built_plate_z=0.0):

    # body_pose = get_pose(b_struct.get_bar_pb_body(bar_vkey))
    body_pose = scale_frame(bar_vertex["gripping_plane"], scale)
    contact_pts = [scale_vector(p, scale) for p in contact_pts_from_o1 + contact_pts_from_o2]
    contact_normals = [scale_vector(v, scale) for v in contact_vecs_from_o1 + contact_vecs_from_o2]

    bar_radius = bar_vertex['radius']
    bar_endpts = bar_vertex['axis_endpoints']
    if bar_vertex['grounded']:
        rot_axis = cross_vectors(add_vectors(bar_endpts[1], scale_vector(bar_endpts[0], -1)), [0,0,1])
        rotated_bar_axis_pts = rotate_points(bar_endpts, math.pi/2, axis=rot_axis, origin=bar_endpts[0])
        rotated_bar_axis = add_vectors(rotated_bar_axis_pts[1], scale_vector(rotated_bar_axis_pts[0], -1))

        lower_axis_pt = bar_endpts[1] if bar_endpts[0][2] - bar_endpts[1][2] > 0 else bar_endpts[0]
        intersec_line = (add_vectors(lower_axis_pt, scale_vector(rotated_bar_axis, -100)),
                         add_vectors(lower_axis_pt, scale_vector(rotated_bar_axis, 100)))
        ground_intersect_pt = intersection_line_plane(intersec_line, Plane([0,0,built_plate_z], [0,0,1]))

        dist = distance_point_point(ground_intersect_pt, lower_axis_pt)
        assert dist > bar_radius - TOL, 'bar penetrating into the ground, distance: {} | {}!'.format(dist, bar_radius)

        contact_pts.append(scale_vector(ground_intersect_pt, scale))
        contact_normals.append([0,0,1*scale])
        # if abs(dist-bar_radius) < TOL:
        #     contact_pts.append(ground_intersect_pt)
        #     contact_normals.append([0,0,1])

    for c_pt, c_v in zip(contact_pts, contact_normals):
        add_line(c_pt, add_vectors(c_pt, scale_vector(c_v, 0.2)), color=(1,0,0,1))
        draw_point(c_pt, color=(0,0,1,1), size=1e-2)

    if not IPY:
        import numpy as np
        from numpy.linalg import norm
        from scipy.linalg import null_space
        from scipy.optimize import linprog
        import coop_assembly.assembly_info_generation.interlock as ipc
    else:
        from compas.rpc import Proxy
        ipc = Proxy('coop_assembly.assembly_info_generation.interlock')

    contact_change_dirs, _ = ipc.compute_local_disassembly_motion(body_pose, contact_pts, contact_normals)

    kin_mat = np.vstack([v for v in contact_change_dirs]).T
    # ns = null_space(A_eq[3:, :])

    # n = len(contact_change_dirs)
    # c = [0] * n
    # A_eq = kin_mat[3:, :]
    # b_eq = [0] * 3
    # res = linprog(c, A_eq=A_eq, b_eq=b_eq)
    # print(res)
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linprog.html

    trans_only_vs = kin_mat[:, [j for j in range(kin_mat.shape[1]) if norm(kin_mat[3:, j]) < EPS]]
    if trans_only_vs.shape[1] == 0:
        # infeasible
        cprint('Rotation involved!', 'red')
        # sum_v = [0,0,0,0,0,0]
        # w = 1.0
        weights = np.random.uniform(0, 1.0, len(contact_change_dirs))
        # for w, v in zip(weights, contact_change_dirs):
        #     sum_v = add_vectors(scale_vector(v, w), sum_v)
        sum_v = kin_mat.dot(weights)

        # print('sum_v: {} | {}'.format(sum_v, body_pose[0]))

        axis, pt = ipc.velocity_to_rotation(sum_v, body_pose[0])
        tf = Rotation.from_axis_and_angle(axis, rot_angle, point=pt)
    else:
        cprint('translation only!', 'green')
        cprint('Available translation: {}'.format(trans_only_vs.shape[1]))
        # weights = np.random.uniform(0, 1.0, trans_only_vs.shape[1])
        weights = [1] * trans_only_vs.shape[1]
        # for i, w in enumerate(weights):
        #     sum_v += trans_only_vs[i]
        # weights_null = ns.dot(weights)
        sum_v = trans_only_vs.dot(weights)
        assert norm(sum_v[3:]) < 1e-10, 'norm: {} | ns: {} | sum_v: {}'.format(norm(sum_v[3:]), ns, sum_v)
        tf = Translation(scale_vector(sum_v[:3], trans_distance*scale))

    return tf

def get_offset_collision_test(body, obstacles, **kwargs):
    ee_collision_fn = get_floating_body_collision_fn(body, obstacles, **kwargs)
    def fn(pose, diagnosis=False):
        # world_from_offset_pose

        # world_from_tf = pb_pose_from_Transformation(tf)
        # world_from_bar = get_pose(body)
        # offset_pose = multiply(world_from_tf, world_from_bar)

        if diagnosis:
            print('offset pose:')
            set_pose(body, pose)
            set_color(body, apply_alpha(BLUE, 0.5))
            wait_for_user()

        is_colliding = ee_collision_fn(pose, diagnosis=diagnosis)
        if diagnosis:
            pcolor = 'red' if is_colliding else 'green'
            cprint('is colliding: {}'.format(is_colliding), pcolor)

        return is_colliding
    return fn
