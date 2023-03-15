import argparse, os
from collections import defaultdict
import json

import numpy as np
from numpy.linalg import norm as npnorm
import jax.numpy as jnp
from jax import grad, jit
from functools import partial
import scipy

import matplotlib.pyplot as plt
from matplotlib import collections as plt_collections

import pybullet_planning as pp
from beam_assembly import BeamAssembly
from coop_assembly.planning.visualization import draw_element, GROUND_COLOR, BACKGROUND_COLOR, SHADOWS, set_camera, \
    label_points, check_model
from coop_assembly.data_structure import BarStructure, GROUND_INDEX
from coop_assembly.help_functions.helpers_geometry import compute_contact_line_between_bars, calculate_coord_sys
from coop_assembly.geometry_generation.utils import get_element_neighbors

import multi_tangent
from multi_tangent import DATA_DIRECTORY
from multi_tangent.convert import bar_structure_from_beam_assembly
from multi_tangent.utils import get_edge_direction, LOGGER

def compute_cyclic_contact_assignment(edges):
    beam_ids_from_vertex = defaultdict(set)
    for e_id, e in enumerate(edges):
        beam_ids_from_vertex[e[0]].add(e_id)
        beam_ids_from_vertex[e[1]].add(e_id)
    contact_assignments = []
    for v_id, v_edge_set in beam_ids_from_vertex.items():
        v_edges = list(v_edge_set)
        if len(v_edges) > 1:
            contact_ring = []
            if len(v_edges) == 2:
                contact_ring.append(tuple(v_edges))
            else:
                for kd in range(len(v_edges)):
                    e0 = v_edges[kd]
                    e1 = v_edges[(kd+1) % len(v_edges)]
                    contact_ring.append((e0, e1))
            contact_assignments.append(contact_ring)
    return contact_assignments

def divide_contact_beam_pairs(vertices, edges, contact_assignments, sinEps=1e-3):
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
    non_parallel_e_pairs = []
    non_parallel_e_signs = []
    parallel_e_pairs = []
    parallel_e_perp_dirs = []

    for contact_ring in contact_assignments:
        dir_cross = []
        for i, (e0, e1) in enumerate(contact_ring):
            d0 = get_edge_direction(edges[e0,:], vertices)
            d1 = get_edge_direction(edges[e1,:], vertices)
            dir_cross.append(np.cross(d0,d1))
            # if two lines are in parellel
            if npnorm(np.cross(d0, d1)) < sinEps:
                drt = np.array([0,0,1.0])
                # find a direction that's not parallel to d0
                if npnorm(np.cross(drt,d0)) < sinEps:
                    drt = np.array([1.0,0,0])
                    if npnorm(np.cross(drt,d0)) < sinEps:
                        raise ValueError('z and x axes both parallel to e{}'.format(e0))
                parallel_e_pairs.append([e0,e1])
                parallel_e_perp_dirs.append(drt)
            else: # non-parellel
                non_parallel_e_pairs.append((e0,e1))
                if len(contact_ring) >= 3 and i == len(contact_ring)-1 and np.dot(dir_cross[-1], dir_cross[-2]) >= 0:
                        non_parallel_e_signs.append(-1)
                else:
                    non_parallel_e_signs.append(1)

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

########################

class BeamOpt:
    def __init__(self, vertices, edges, support_vertex_indices, radius, contact_assigments=None):
        """
        Inputs
        ======
        vertices: num_nodes x 3 np array
        edges:    num_edges x 2 np array
        support_vertex_indices: num_supp x 1 np array
        """
        # vertices' coordinate
        self.vertices = jnp.array(vertices.copy())

        # beams' vertex ids
        self.edges = jnp.array(edges.copy())

        # fixed vertex ids
        self.support_vertex_indices = support_vertex_indices.copy()

        self.end_pts_init_positions = (self.vertices[self.edges]).flatten().copy()

        # TODO allow certain edges to be frozen

        c_assigns = contact_assigments or compute_cyclic_contact_assignment(edges)
        [_non_parallel_e_pairs, _non_parallel_e_signs, _parallel_e_pairs, _parallel_e_perp_dirs] = \
            divide_contact_beam_pairs(vertices, edges, c_assigns, 0.2)

        # from multi_tangent.archive import computeOffsetBeamConstraintsPairs
        # [_non_parallel_e_pairs, _non_parallel_e_signs, _parallel_e_pairs, _parallel_e_perp_dirs] = \
        #     computeOffsetBeamConstraintsPairs(vertices, edges, 0.2)
        #     # assembly.computeOffsetBeamConstraintsPairs(0.2) # C++ version

        #beam beam connection
        # 2xpair non-parallel pairs
        self.non_parallel_e_pairs = jnp.array(_non_parallel_e_pairs)
        # non-parallel pair sign
        self.non_parallel_e_signs = jnp.array(_non_parallel_e_signs)

        # 2xpair parallel pairs
        self.parallel_e_pairs = jnp.array(_parallel_e_pairs)
        # parallel pairs' perpendicular directions
        self.parallel_e_perp_dirs = jnp.array(_parallel_e_perp_dirs)

        self.nB = self.edges.shape[0]
        self.nE_np = self.non_parallel_e_pairs.shape[1]
        self.nE_p = self.parallel_e_pairs.shape[1]

        # nT: number of variables for beam's intersection point
        self.nT = (self.nE_np + self.nE_p) * 2

        # nP: number of varibles Pi for beams' endpoints
        self.nP = self.end_pts_init_positions.shape[0]

        self.fix_var_ids = self.nT + np.array([[sv*3, sv*3+1, sv*3+2] for sv in self.support_vertex_indices], dtype=int).flatten()

        # TODO: multiple materials and radius
        self.radius = radius
        self.weights = [1, 30, 1]

        self.xstar = self.computeInitVar()
        self.x_trace = []

    def computeInitVar(self):
        t = jnp.ones(self.nT) * 0.5
        var = jnp.hstack([t, self.end_pts_init_positions]).copy()
        # reset trace
        self.x_trace = []
        return var

    def checkBeamDistanceConstraints(self, var):
        return computeBeamDistanceEnergy(var, self)

    def runOptimization(self, trace=True):
        """Run unconstrained optimization with Newton-CG for the given contact pattern

        Returns
        -------
        new_V: (num_beam x 2) x 3 np array for beam vertices, each row is [x,y,z]
        new_EV: (num_beam) x 2 np array for the edge assignment,
            the row ordering corresponds with the original edges, but the vertex indices mapping is reordered.
        new_fixed_vert_ids: list of int
            fixed vertex ids in new_V
        """
        var = self.computeInitVar()
        result = scipy.optimize.minimize(computeBeamOptObjective,
                                         var,
                                         args=self,
                                         jac=computeBeamOptObjectiveGradient,
                                         hessp=computeBeamOptObjectiveHessianP,
                                         method="Newton-CG",
                                         options={
                                            'disp':True,
                                            'return_all':trace,
                                                  })

        # print(f'Opt success: {result.success} | Status: {result.status} | {result.message} | n_iter: {result.nit}')
        # print(f'nfev, njev, nhev : {result.nfev}, {result.njev}, {result.nhev}')
        print('Distance constraint: {}'.format(self.checkBeamDistanceConstraints(result.x)))
        self.xstar = result.x
        new_V = result.x[self.nT: self.nP + self.nT].reshape(self.nB * 2, 3)
        new_EV = np.array([[2 * i, 2 * i + 1] for i in range(0, self.nB)], dtype=int)
        if trace:
            self.x_trace = result.allvecs

        # * map the old fixed node ids into the new ones
        old_fixed_beam_end_id = defaultdict(list)
        for gv in self.support_vertex_indices:
            for e_id, e in enumerate(self.edges):
                for i, end_id in enumerate(e):
                    if gv == end_id:
                        old_fixed_beam_end_id[gv].append((e_id, i))

        new_fixed_vert_ids = set()
        for _, e_endpts in old_fixed_beam_end_id.items():
            for e, i in e_endpts:
                new_fixed_vert_ids.add(new_EV[e][i])

        # new_assembly = BeamAssembly(new_V, new_EV, self.new_fixed_vert_ids)
        # new_assembly.setMaterial(self.material)
        # new_assembly.update()
        return new_V, new_EV, list(new_fixed_vert_ids)

    @property
    def new_fixed_vert_ids(self):
        """map the old fixed node ids into the new ones
        """
        new_EV = [[2 * x, 2 * x + 1] for x in range(0, self.nB)]
        old_fixed_beam_end_id = defaultdict(list)
        for gv in self.support_vertex_indices:
            for e_id, e in enumerate(self.edges):
                for i, end_id in enumerate(e):
                    if gv == end_id:
                        old_fixed_beam_end_id[gv].append((e_id, i))
        new_fixed_vert_ids = set()
        for _, e_endpts in old_fixed_beam_end_id.items():
            for e, i in e_endpts:
                new_fixed_vert_ids.add(new_EV[e][i])
        return list(new_fixed_vert_ids)


################################################

@partial(jit, static_argnums=(1,))
def computeBeamDistanceEnergy(var, beam_opt: BeamOpt):
    # non-parallel intersection point vars (line parameterization t)
    tE = var[0: beam_opt.nE_np * 2].reshape(2, beam_opt.nE_np)
    # parallel intersection point vars
    tE2 = var[beam_opt.nE_np * 2 : beam_opt.nE_np * 2 + beam_opt.nE_p * 2].reshape(2, beam_opt.nE_p)
    # end point per beam vars
    P = var[beam_opt.nT: beam_opt.nT + beam_opt.nP].reshape(beam_opt.nB, 2, 3)
    # nB x 3
    drt = P[:, 1, :] - P[:, 0, :]

    origin = P[:, 0, :]
    energy = 0
    # non-parallel pairs
    normal = jnp.cross(drt[beam_opt.non_parallel_e_pairs[0, :], :], drt[beam_opt.non_parallel_e_pairs[1, :], :])
    inv_length = 1.0 / jnp.linalg.norm(normal, axis = 1) * beam_opt.non_parallel_e_signs
    normal = jnp.diag(inv_length) @ normal
    # (nE_np x nE_np) * (nE_np, 3)
    # nB x 3
    p0 = jnp.diag(tE[0, :]) @ drt[beam_opt.non_parallel_e_pairs[0, :], :] + origin[beam_opt.non_parallel_e_pairs[0, :], :]
    p1 = jnp.diag(tE[1, :]) @ drt[beam_opt.non_parallel_e_pairs[1, :], :] + origin[beam_opt.non_parallel_e_pairs[1, :], :]
    energy = jnp.vdot(p1 - p0 - normal * beam_opt.radius * 2, p1 - p0 - normal * beam_opt.radius * 2)
    # TODO lower bound the distance to prevent penetration

    # parallel pairs
    normal = jnp.cross(beam_opt.parallel_e_perp_dirs.transpose(), drt[beam_opt.parallel_e_pairs[0, :], :])
    inv_length = 1.0 / jnp.linalg.norm(normal, axis = 1)
    normal = jnp.diag(inv_length) @ normal
    p0 = jnp.diag(tE2[0, :]) @ drt[beam_opt.parallel_e_pairs[0, :], :] + origin[beam_opt.parallel_e_pairs[0, :], :]
    p1 = jnp.diag(tE2[1, :]) @ drt[beam_opt.parallel_e_pairs[1, :], :] + origin[beam_opt.parallel_e_pairs[1, :], :]
    energy += jnp.vdot(p1 - p0 - normal * beam_opt.radius * 2, p1 - p0 - normal * beam_opt.radius * 2)
    # ? for ensuring parallel
    energy += jnp.vdot(normal, drt[beam_opt.parallel_e_pairs[1, :], :]) ** 2

    return energy

@partial(jit, static_argnums=(1,))
def computeBeamOptObjective(var, beam_opt: BeamOpt):
    t = var[0: beam_opt.nT].reshape(2, (beam_opt.nE_np + beam_opt.nE_p))
    Pf = var[beam_opt.nT: beam_opt.nT + beam_opt.nP]
    # P = var[beam_opt.nT: beam_opt.nT + beam_opt.nP].reshape(beam_opt.nB, 2, 3)
    # * origianl graph approximation energy
    energy1 = jnp.vdot(beam_opt.end_pts_init_positions - Pf, beam_opt.end_pts_init_positions - Pf)
    energy2 = computeBeamDistanceEnergy(var, beam_opt)
    # * parameter t range
    energy3 = jnp.vdot(jnp.clip(t, 0.1, 0.9) - t, jnp.clip(t, 0.1, 0.9) - t)

    # * Fix supports
    # energy4 = jnp.vdot(beam_opt.end_pts_init_positions[beam_opt.fix_var_ids] - Pf[beam_opt.fix_var_ids],
    #                    beam_opt.end_pts_init_positions[beam_opt.fix_var_ids] - Pf[beam_opt.fix_var_ids])

    summed_energy = beam_opt.weights[0] * energy1 + beam_opt.weights[1] * energy2 + beam_opt.weights[2] * energy3
    # + energy4*10
    return summed_energy

@partial(jit, static_argnums=(1,))
def computeBeamOptObjectiveGradient(var, beamOpt: BeamOpt):
    return grad(computeBeamOptObjective, 0)(var, beamOpt)

@partial(jit, static_argnums=(2,))
def computeGradientV(var, v, beamOpt: BeamOpt):
    gradient = computeBeamOptObjectiveGradient(var, beamOpt)
    return jnp.dot(gradient, v)

@partial(jit, static_argnums=(2,))
def computeBeamOptObjectiveHessianP(var, v, beamOpt: BeamOpt):
    return grad(computeGradientV, 0)(var, v, beamOpt)

##############################################

def compute_distance_energy_per_connector(var, beam_opt: BeamOpt):
    # non-parallel intersection point vars (line parameterization t)
    tE = var[0: beam_opt.nE_np * 2].reshape(2, beam_opt.nE_np)
    # parallel intersection point vars
    tE2 = var[beam_opt.nE_np * 2 : beam_opt.nE_np * 2 + beam_opt.nE_p * 2].reshape(2, beam_opt.nE_p)
    # end point per beam vars
    P = var[beam_opt.nT: beam_opt.nT + beam_opt.nP].reshape(beam_opt.nB, 2, 3)
    drt = P[:, 1, :] - P[:, 0, :]

    origin = P[:, 0, :]
    energy = 0
    # non-parallel pairs
    normal = jnp.cross(drt[beam_opt.non_parallel_e_pairs[0, :], :], drt[beam_opt.non_parallel_e_pairs[1, :], :])
    inv_length = 1.0 / jnp.linalg.norm(normal, axis = 1) * beam_opt.non_parallel_e_signs
    normal = jnp.diag(inv_length) @ normal
    p0 = jnp.diag(tE[0, :]) @ drt[beam_opt.non_parallel_e_pairs[0, :], :] + origin[beam_opt.non_parallel_e_pairs[0, :], :]
    p1 = jnp.diag(tE[1, :]) @ drt[beam_opt.non_parallel_e_pairs[1, :], :] + origin[beam_opt.non_parallel_e_pairs[1, :], :]
    # energy = jnp.vdot(p1 - p0 - normal * beam_opt.width, p1 - p0 - normal * beam_opt.width)
    np_vec_diff = p1 - p0 - normal * beam_opt.radius * 2

    # parallel pairs
    normal = jnp.cross(beam_opt.parallel_e_perp_dirs.transpose(), drt[beam_opt.parallel_e_pairs[0, :], :])
    inv_length = 1.0 / jnp.linalg.norm(normal, axis = 1)
    normal = jnp.diag(inv_length) @ normal
    p0 = jnp.diag(tE2[0, :]) @ drt[beam_opt.parallel_e_pairs[0, :], :] + origin[beam_opt.parallel_e_pairs[0, :], :]
    p1 = jnp.diag(tE2[1, :]) @ drt[beam_opt.parallel_e_pairs[1, :], :] + origin[beam_opt.parallel_e_pairs[1, :], :]
    # energy += jnp.vdot(p1 - p0 - normal * beam_opt.width, p1 - p0 - normal * beam_opt.width)
    p_vec_diff = p1 - p0 - normal * beam_opt.radius * 2
    # energy += jnp.vdot(normal, drt[beam_opt.parallel_e_pairs[1, :], :]) ** 2

    return np.vstack([np_vec_diff, p_vec_diff])

def plot_contact_distance(bar_struct: BarStructure):
    connectors = bar_struct.get_connectors()
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    # clean up duplicated bidirectional edges
    dist_data = dict()
    radius_lines = dict()
    count = 0
    for edge, c in connectors.items():
        n1, n2 = edge
        if n1 == GROUND_INDEX or n2 == GROUND_INDEX:
            continue
        reversed_edge = edge[::-1]
        if reversed_edge in dist_data:
            continue
        dist_data[edge] = (npnorm(np.array(c[0]) - np.array(c[1])))
        r1, r2 = bar_struct.node[n1]['radius'], bar_struct.node[n2]['radius']
        radius_lines[edge] = ([(count,r1+r2), (count+1,r1+r2)])
        count += 1

    # * histogram
    dist_list = list(dist_data.values())
    axes[0].hist(dist_list, 20, lw=2, ec="yellow", fc="green", alpha=0.5)
    axes[0].set_xlabel('Shortest distance between contact bars [m]')
    axes[0].set_ylabel('Count')

    # * line plot
    lc = plt_collections.LineCollection(list(radius_lines.values()), colors='blue', linewidths=2)
    axes[1].add_collection(lc)
    axes[1].plot(dist_list)
    axes[1].set_xlabel('Connector index')
    axes[1].set_ylabel('Dist between contact bars [m]')
    fig.suptitle('Contact dist from BarStructure')

    return fig

def plot_beam_opt_per_pair(beam_opt: BeamOpt):
    x_vecdiff = compute_distance_energy_per_connector(beam_opt.xstar, beam_opt)
    x_energy = npnorm(x_vecdiff, axis=1)

    fig, axes = plt.subplots(1,2, figsize=(12,6))

    # * histogram
    axes[0].hist(x_energy, 20, lw=2, ec="yellow", fc="green", alpha=0.5)
    axes[0].set_xlabel('$(E_1(t_1) - E_2(t_2)) \cdot 2r*n$')
    axes[0].set_ylabel('Count')

    # * line plot
    axes[1].plot([(beam_opt.radius*2)**2 for _ in range(len(x_energy))], color='blue', linewidth=2)
    axes[1].plot(x_energy, 'bo')
    axes[1].set_xlabel('Connector index')
    axes[1].set_ylabel('vec product')
    fig.suptitle('Contact dist from beam_opt $x_*$')

    return fig

def plot_f_trace(beam_opt: BeamOpt):
    f_sum_trace = [computeBeamOptObjective(x, beam_opt) for x in beam_opt.x_trace]
    f_beam_trace = [computeBeamDistanceEnergy(x, beam_opt) for x in beam_opt.x_trace]

    fig, axes = plt.subplots(figsize=(12,6))
    # * histogram
    # axes[0].hist(x_energy, 20, lw=2, ec="yellow", fc="green", alpha=0.5)
    # axes[0].set_xlabel('$(E_1(t_1) - E_2(t_2)) \cdot 2r*n$')
    # axes[0].set_ylabel('Count')
    # * line plot
    axes.plot(f_sum_trace, label='summed energy', color='blue', linewidth=2)
    axes.plot(f_beam_trace, label='beam distance energy', color='red', linewidth=2)
    axes.set_xlabel('iter')
    axes.set_ylabel('energy')
    axes.legend()
    # fig.suptitle('Contact dist from beam_opt $x_*$')
    return fig

def bar_structure_from_beam_assembly(
        old_edges, old_support_indices,
        new_vertices, new_edges,
        radius,
        to_meter_scale=1.0,
        tf_pose=pp.unit_pose(), shrink_radius=2e-3, file_name='new_bar_structure', save_file=True, add_ground_connectors=True):
    # * compile bars into a BarStructure (for coop_assembly)
    b_struct = BarStructure()
    centroid_pt = np.average(new_vertices, axis=0)

    # ! bar_from_elements[edge] = {node_id : [x,y,z]}
    bar_from_elements = {}
    # materials_from_elements = {}
    assert old_edges.shape[0] == new_edges.shape[0]
    for i, (np_old_e, new_e) in enumerate(zip(old_edges, new_edges)):
        old_e = tuple(np_old_e.tolist())
        bar_from_elements[old_e] = {
            old_e[0] : new_vertices[new_e[0]],
            old_e[1] : new_vertices[new_e[1]],
        }
        # materials_from_elements[old_e] = new_assembly.materials[i]

    index_from_element = {}
    for i, (e, pts) in enumerate(bar_from_elements.items()):
        axis_endpts = list(pts.values())
        _ , _, vec_z = calculate_coord_sys(axis_endpts, centroid_pt)
        n0, n1 = e
        is_grounded = n0 in old_support_indices or n1 in old_support_indices
        # radius = materials_from_elements[e].width / 2.0
        bar_key = b_struct.add_bar(None, [p.tolist() for p in axis_endpts], "tube", None, vec_z, radius=radius,
            grounded=is_grounded, pb_scale=to_meter_scale, shrink_radius=shrink_radius)
        index_from_element[e] = bar_key

    # TODO assign contact according to contact assignments
    element_neighbors = get_element_neighbors(list(bar_from_elements.keys()))
    for e, pts in bar_from_elements.items():
        # update contact point into BarS's edges
        for ne in element_neighbors[e]:
            b_struct.connect_bars(index_from_element[e], index_from_element[ne])
            contact_pts = compute_contact_line_between_bars(b_struct, index_from_element[e], index_from_element[ne])
            b_struct.edge[index_from_element[e]][index_from_element[ne]]["endpoints"].update({0:(list(contact_pts[0]), list(contact_pts[1]))})

    # * add grounded connector
    if add_ground_connectors:
        grounded_bars = list(b_struct.get_grounded_bar_keys())
        for ground_k in grounded_bars:
            b_struct.connect_bars(ground_k, GROUND_INDEX)
            # find the lower pt of the two
            axis_endpts = b_struct.get_bar_axis_end_pts(ground_k)
            if axis_endpts[0][2] > axis_endpts[1][2]:
                axis_endpts = axis_endpts[::-1]
            # radius = materials_from_elements[e].width / 2.0
            contact_pts = [axis_endpts[0], axis_endpts[0]-np.array([0,0,radius])]
            b_struct.edge[ground_k][GROUND_INDEX]["endpoints"].update({0:(list(contact_pts[0]), list(contact_pts[1]))})

    # * additional transformation
    # world_from_base = pp.Pose(point=b_struct.base_centroid(to_meter_scale))
    # tf = pp.multiply(tf_pose, pp.invert(world_from_base))
    # b_struct.transform(tf, scale=to_meter_scale)

    if save_file:
        graph_data = None
        opt_data = None
        export_structure_data(b_struct.to_data(), overall_struct_data=graph_data, save_dir=os.path.join(DATA_DIRECTORY, 'bar_structures'),
            file_name=file_name + '.json', opt_data=opt_data, indent=0)

    return b_struct

##############################################

def main():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='truss_one_tet_skeleton.json',
                        help='The name of the problem to solve')
    parser.add_argument('--file_format', default='bar_structure', # assembly
                        help='File formats.')
    parser.add_argument('--radius', default=None, type=float,
                        help='Bar radius')
    parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    parser.add_argument('-w', '--write', action='store_true', help='Export file.')
    # parser.add_argument('-c', '--collisions', action='store_false',
    #                     help='Disable collision checking with obstacles')
    parser.add_argument('-db', '--debug', action='store_true', help='Debug verbose mode')
    args = parser.parse_args()


    if args.file_format == 'assembly':
        assembly = BeamAssembly()
        assembly.loadFromJson(os.path.join(DATA_DIRECTORY, 'assembly', args.problem), 1, [0, 0, 0])
        vertices = np.array(assembly.vertices)
        edges = np.array(assembly.edges, dtype=int)
        support_vertex_indices = assembly.support_vertex_indices
        radius = args.radius or assembly.materials[0].width
    elif args.file_format == 'bar_structure':
        from coop_assembly.planning.parsing import get_assembly_path
        from compas.datastructures import Network
        # truss_one_tet_skeleton.json
        problem_path = get_assembly_path(args.problem)
        with open(problem_path) as json_file:
            graph_data = json.load(json_file)
            print('Design graph parsed from : {}'.format(problem_path))
        if 'data' in graph_data:
            graph_data = graph_data['data']

        net = Network.from_data(graph_data)
        vertices = np.array([[net.node[v][c] for c in ['x', 'y', 'z']] for v in range(net.number_of_nodes())])
        # ! scale conversion
        vertices *= 0.001
        assert vertices.shape[1] == 3
        edges = np.array([list(e) for e in net.edges()], dtype=int)
        assert edges.shape[1] == 2
        support_vertex_indices = [v for v, attr in net.nodes(True) if attr['fixed'] == True]
        radius = args.radius or 0.01
    else:
        pass

    beam_opt = BeamOpt(vertices, edges, support_vertex_indices, radius)
    beam_opt.weights =[1, 200, 1]
    var = beam_opt.computeInitVar()
    fun = computeBeamOptObjective(var, beam_opt)
    gradient = computeBeamOptObjectiveGradient(var, beam_opt)
    v = jnp.ones(beam_opt.nP + beam_opt.nT)
    hessian = computeBeamOptObjectiveHessianP(var, v, beam_opt)
    new_vertices, new_edges, new_supports = beam_opt.runOptimization()

    # fig = plot_beam_opt_per_pair(beam_opt)
    # plt.show()
    fig = plot_f_trace(beam_opt)
    plt.show()

    pp.connect(use_gui=args.viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    pp.draw_pose(pp.unit_pose(), length=1.0)
    # set_camera([[0,0,0]], camera_dist=0.5, scale=1.0)

    bar_struct = bar_structure_from_beam_assembly(
        edges, support_vertex_indices,
        new_vertices, new_edges, radius,
        shrink_radius=1e-3,
        file_name=args.problem.split('.json')[0], save_file=args.write, add_ground_connectors=1,
        # tf_pose=pp.Pose(euler=pp.Euler(roll=np.pi/2))
        )
    element_bodies = bar_struct.get_element_bodies(scale=1.0, color=pp.apply_alpha(pp.RED, 0.5))
    fig = plot_contact_distance(bar_struct)
    if args.write:
        export_file_name = args.problem
        if 'skeleton' in export_file_name:
            export_file_name = export_file_name.split('_skeleton')[0] + '.json'
        if 'raw_bar_graph' in export_file_name:
            export_file_name = export_file_name.split('_raw_bar_graph')[0] + '.json'
        fig.savefig(os.path.join(multi_tangent.DATA_DIRECTORY, 'plots', export_file_name + '_bar_structure_dist_plot.png'))
    plt.show()

    # * plot reference graph
    with pp.LockRenderer():
        set_camera(new_vertices)
        # pp.draw_pose(pp.unit_pose(), length=0.01)
        # label_points([pt*SCALE for pt in node_points])
        for e in edges:
            n1, n2 = e
            p1 = vertices[n1]
            p2 = vertices[n2]
            pp.add_line(p1, p2, color=pp.apply_alpha(pp.BLUE, 0.3), width=0.5)
        for v in support_vertex_indices:
            pp.draw_circle(vertices[v], 0.01)

    check_model(bar_struct, debug=0, to_meter_scale=1.0, contact_dist_tol=3e-3, collision_dist_tol=0.002)

    pp.reset_simulation()
    pp.disconnect()

if __name__ == '__main__':
    main()
