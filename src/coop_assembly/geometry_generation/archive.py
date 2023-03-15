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