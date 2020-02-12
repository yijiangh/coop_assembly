import numpy as np
from numpy.linalg import norm
import cdd

from pybullet_planning import get_pose, matrix_from_quat

def compute_feasible_region_from_block_dir(block_dirs, verbose=False):
    """ Compute extreme ray representation of feasible assembly region, given blocking direction vectors.

    The feasible assembly region is constrained by some hyperplanes, which use
    block_dirs as normals. cdd package allows us to convert the inequality
    representation to a generator (vertices and rays) of a polyhedron.

    Adapted from: https://github.com/yijiangh/compas_rpc_example

    More info on cddlib:
    https://pycddlib.readthedocs.io/en/latest/index.html
    Other packages on vertex enumeration:
    https://mathoverflow.net/questions/203966/computionally-efficient-vertex-enumeration-for-convex-polytopes

    Parameters
    ----------
    block_dirs : list of 3-tuples
        a list blocking directions.

    Returns
    -------
    f_rays: list of 3-tuples
        extreme rays of the feasible assembly region
    lin_set: list of int
        indices of rays that is linear (both directions)
    """
    mat_hrep = [] # "half-space" representation
    for vec in block_dirs:
        # For a polyhedron described as P = {x | A x <= b}
        # the H-representation is the matrix [b -A]
        mat_hrep.append([0, -vec[0], -vec[1], -vec[2]])
    mat = cdd.Matrix(mat_hrep, number_type='fraction')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    ext = poly.get_generators()
    lin_set = list(ext.lin_set) # linear set both directions

    nt = cdd.NumberTypeable('float')
    f_verts = []
    f_rays = []
    # linear_rays = []
    for i in range(ext.row_size):
        if ext[i][0] == 1:
            f_verts.append(tuple([nt.make_number(num) for num in ext[i][1:4]]))
        elif ext[i][0] == 0:
            # TODO: numerical instability?
            ray_vec = [nt.make_number(num) for num in ext[i][1:4]]
            ray_vec /= norm(ray_vec)
            f_rays.append(tuple(ray_vec))
            # if i in lin_set:
            #     lin_vec_set.append(tuple([- nt.make_number(num) for num in ext[i][1:4]]))

    # if f_verts:
    #     assert len(f_verts) == 1
        # np.testing.assert_almost_equal f_verts[0] == [0,0,0]

    # TODO: QR decomposition to make orthogonal
    if verbose:
        print('##############')
        print('ext:\n {}'.format(ext))
        print('ext linset:\n {}'.format(ext.lin_set))
        print('verts:\n {}'.format(f_verts))
        print('rays:\n {}'.format(f_rays))

    return f_rays, lin_set

def cross_prod_matrix(vec):
    """compute angular velovity tensor

    Ref: https://en.wikipedia.org/wiki/Angular_velocity#Angular_velocity_tensor

    Parameters
    ----------
    vec : list of three float

    Returns
    -------
    3x3 np matrix
    """
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]
                    ])

def compute_body_jacobian(contact_pt, com_pt):
    return np.hstack((np.eye(3), -cross_prod_matrix(np.array(contact_pt) - com_pt)))

# TODO: convert velocity described in local object frame to global frame

def contact_velocity_from_local_frame_velocity(contact_pts, body_pose, body_v):
    if len(body_pose) == 2:
        # pb pose
        pt_com, quat = body_pose
    elif len(body_pose) >= 3:
        # coop_assembly frame
        # (pt_o, vec_x, vec_y, vec_z)
        pt_com = body_pose[0]
    pt_vs = []
    for pt_c in contact_pts:
        Jc = compute_body_jacobian(pt_c, pt_com)
        pt_v = Jc.dot(body_v)
        pt_vs.append(pt_v.tolist())
    return pt_vs

def check_local_contact_feasibility(body_pose, contact_pts, block_dirs, body_vel, verbose=False):
    assert len(contact_pts) == len(block_dirs)
    assert len(body_vel) == 6
    if len(body_pose) == 2:
        # pb pose
        pt_com, quat = body_pose
    elif len(body_pose) >= 3:
        # coop_assembly frame
        # (pt_o, vec_x, vec_y, vec_z)
        pt_com = body_pose[0]

    violate = []
    for pt_c, n_c in zip(contact_pts, block_dirs):
        Jc = compute_body_jacobian(pt_c, pt_com)
        contact_val = np.array(n_c).dot(Jc).dot(body_vel)
        if verbose:
            print('contact val {} | pt {} | norm {}'.format(contact_val, pt_c, n_c))
        if contact_val < 0:
            violate.append((pt_c, n_c, contact_val))
    return False if len(violate) > 0 else True

def compute_local_disassembly_motion(body_pose, contact_pts, block_dirs, verbose=False):
    assert len(contact_pts) == len(block_dirs)
    if len(body_pose) == 2:
        # pb pose
        pt_com, quat = body_pose
    elif len(body_pose) >= 3:
        # coop_assembly frame
        # (pt_o, vec_x, vec_y, vec_z)
        pt_com = body_pose[0]

    mat_hrep = [] # "half-space" representation
    for pt_c, n_c in zip(contact_pts, block_dirs):
        # For a polyhedron described as P = {x | A x <= b}
        # the H-representation is the matrix [b -A]
        Jc = compute_body_jacobian(pt_c, pt_com)
        mat_hrep.append(np.hstack([[0], np.array(n_c).dot(Jc)]))
        # w = np.cross((np.array(pt_c) - np.array(pt_com)), n_c)
        # mat_hrep.append([0, -n_c[0], -n_c[1], -n_c[2], w[0], w[1], w[2]])
    mat = cdd.Matrix(mat_hrep, number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    ext = poly.get_generators()
    lin_set = list(ext.lin_set) # linear set both directions

    # TODO: check if f_rays' rotational part is rank-0,
    # if so, translation contact-breaking motion is possible

    nt = cdd.NumberTypeable('float')
    f_verts = []
    contact_change_dirs = []
    contact_maintain_dirs = []
    for i in range(ext.row_size):
        if ext[i][0] == 1:
            f_verts.append(tuple([nt.make_number(num) for num in ext[i][1:]]))
        elif ext[i][0] == 0:
            # TODO: numerical instability?
            ray_vec = [nt.make_number(num) for num in ext[i][1:]]
            ray_vec /= norm(ray_vec)
            # assert ray_vec[:3].dot(ray_vec[3:]) == 0
            if i in lin_set:
                contact_maintain_dirs.append(ray_vec)
            else:
                contact_change_dirs.append(ray_vec)

    if verbose:
        print('##############')
        print('ext:\n {}'.format(ext))
        print('ext linset:\n {}'.format(ext.lin_set))
        print('verts:\n {}'.format(f_verts))
        print('contact changing rays:\n {}'.format(contact_change_dirs))
        print('contact maintaining rays:\n {}'.format(contact_maintain_dirs))

    return contact_change_dirs, contact_maintain_dirs
