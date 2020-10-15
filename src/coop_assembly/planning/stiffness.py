import os
from collections import namedtuple, defaultdict, combinations
from termcolor import cprint
import numpy as np
from numpy.linalg import norm

try:
    from pyconmech.frame_analysis import StiffnessChecker
    from pyconmech.frame_analysis import GravityLoad, Node, Element, Support, Material, CrossSec, Material, Joint
except ImportError as e:
    cprint('{}, Not using conmech'.format(e), 'yellow')
    USE_CONMECH = False
    input("Press Enter to continue...")
else:
    USE_CONMECH = True

from pybullet_planning import HideOutput, INF, apply_alpha, RED, get_distance, angle_between, get_difference, get_angle

from compas.geometry import is_colinear
from coop_assembly.help_functions.shared_const import METER_SCALE, EPS
from coop_assembly.planning.parsing import load_structure, unpack_structure

TRANS_TOL = 0.0015
ROT_TOL = INF # 5 * np.pi / 180

####################################

# TODO: get_max_nodal_deformation
Deformation = namedtuple('Deformation', ['success', 'displacements', 'fixities', 'reactions'])
Displacement = namedtuple('Displacement', ['dx', 'dy', 'dz', 'theta_x', 'theta_y', 'theta_z'])
Reaction = namedtuple('Reaction', ['fx', 'fy', 'fz', 'mx', 'my', 'mz'])

####################################
# E, G12, fy, density
# element_tags = None is the fall-back material entry
WoodMaterial = Material(1050*1e4, 360*1e4, 1.3*1e4, 6, elem_tags=None, family='Wood', name='Wood', type_name='ISO')
SteelMaterial = Material(1050*1e4, 360*1e4, 1.3*1e4, 78.5, elem_tags=None, family='Wood', name='Wood', type_name='ISO')

def A_solid_cir(radius):
    return np.pi * radius**2

def Jx_solid_cir(radius):
    # https://en.wikipedia.org/wiki/Polar_moment_of_inertia
    return np.pi * radius**4 / 2

def Iy_solid_cir(radius):
    # https://www.engineeringtoolbox.com/area-moment-inertia-d_1328.html
    return np.pi * radius**4 / 4

def solid_cir_crosssec(r):
    A = A_solid_cir(r)
    Jx = Jx_solid_cir(r)
    Iy = Iy_solid_cir(r)
    return CrossSec(A, Jx, Iy, Iy, elem_tags=None, family='Circular', name='solid_circle')

####################################

def conmech_model_from_bar_structure(bar_struct, chosen_bars=None):
    # nodes, elements, supports, joints, materials, crosssecs, model_name, unit = \
    #     read_frame_json(json_file_path, verbose=verbose)
    element_from_index, grounded_elements, contact_from_connectors, connectors = \
        unpack_structure(bar_struct, chosen_bars=chosen_bars, scale=METER_SCALE)
    grounded_connectors = bar_struct.get_grounded_connector_keys()

    # Element = namedtuple('Element', ['index', 'axis_endpoints', 'radius', 'body', 'initial_pose', 'goal_pose',
    #                                  'grasps', 'goal_supports', 'layer'])

    # ! meter-based unit
    unit = "meter"
    model_name = bar_struct.name

    # TODO break the lines into pieces according to the connectors
    # axis points should all be unique
    cm_nodes = []
    cm_elements = []
    pts_from_element = defaultdict(set)
    for e_id, element in element_from_index.items():
        # Node(self, point, node_ind, is_grounded):
        # ! is_grounded is used to mark grounded nodes for construction purpose
        # structural supports are specified using the Support entries
        cm_nodes.append(Node(element.axis_endpoints[0], e_id*2, False))
        cm_nodes.append(Node(element.axis_endpoints[1], e_id*2+1, False))
        pts_from_element[e_id].update(element.axis_endpoints)

    supports = []
    for c_id, connector_endpts in contact_from_connectors.items():
        connector_node_inds = []
        for pt in connector_endpts:
            for n in cm_nodes:
                if get_distance(pt, n.point) < EPS:
                    connector_node_inds.append(n.node_ind)
                    break
            else:
                # unseen node
                n_id = len(cm_nodes)
                cm_nodes.append(Node(pt, n_id, False))
                connector_node_inds.append(n_id)
            for e_id in c_id:
                if is_colinear(pt, *element_from_index[e_id].axis_endpoints):
                    pts_from_element[e_id].add(pt)

        assert len(connector_node_inds) == 2
        assert connector_node_inds[0] != connector_node_inds[1], "#{}|{} - {}".format(c_id, connector_endpts, connector_node_inds)

        # * add connector element
        e_id = len(cm_elements)
        cm_elements.append(Element(connector_node_inds, e_id, elem_tag='connector', bending_stiff=True))

        if c_id in grounded_connectors:
            # TODO should only chosen one contact point as fixiety
            # Support(self, condition, node_ind)
            supports.extend([Support([1 for _ in range(6)], connector_node_inds[i]) for i in range(2)])

    for e_id, seg_pts in pts_from_element.items():
        sorted_pt_pairs = sorted(combinations(list(seg_pts), 2), key=lambda pt_pair: get_distance(*pt_pair))
        e_end_pts = element_from_index[e_id].axis_endpoints
        farthest_pts = sorted_pt_pairs[-1]
        if get_angle(get_difference(*farthest_pts), get_difference(*e_end_pts)) > np.pi/2:
            farthest_pts = farthest_pts[::-1]
        sorted_seg_pts = sorted(list(seg_pts), key=lambda pt: get_distance(pt, farthest_pts[0]))
        for i in range(len(sorted_seg_pts)-1):
            # Element(self, end_node_inds, elem_ind, elem_tag='', bending_stiff=True):
            # TODO break elements into segments according to connectors
            pt_pair = (sorted_seg_pts[i], sorted_seg_pts[i+1])
            # check if already in the node list, if not add it
            for pt in pt_pair:
                for n in cm_nodes:
                    if get_distance(pt, n.point) < EPS:
                        connector_node_inds.append(n.node_ind)
                        break
            cm_elements.append(Element([e_id*2, e_id*2+1], e_id, elem_tag='bar', bending_stiff=True))

    print(cm_nodes)
    print(cm_elements)
    input()

    # TODO Add rotational stiffness later
    # Joint(self, c_conditions, elem_tags):
    joints = []

    # TODO different material property and cross secs on Element and Connectors
    r = list(element_from_index.values())[0].radius
    crosssecs = [solid_cir_crosssec(r)]
    materials = [WoodMaterial]

    return cm_nodes, cm_elements, supports, joints, materials, crosssecs, model_name, unit

def create_stiffness_checker(bar_struct, verbose=False):
    if not USE_CONMECH:
        return None

    # * convert ball structure as
    nodes, elements, supports, joints, materials, crosssecs, model_name, unit = \
        conmech_model_from_bar_structure(bar_struct)

    with HideOutput():
        # checker = StiffnessChecker.from_json(json_file_path=extrusion_path, verbose=verbose)
        checker = StiffnessChecker(nodes=nodes, elements=elements, supports=supports, \
                   materials=materials, joints=joints, crosssecs=crosssecs, verbose=verbose, \
                   model_name=model_name, unit=unit, checker_engine="numpy")
        checker.set_loads(gravity_load=GravityLoad([0,0,-1.0]))
    #checker.set_output_json(True)
    #checker.set_output_json_path(file_path=os.getcwd(), file_name="stiffness-results.json")
    checker.set_nodal_displacement_tol(trans_tol=TRANS_TOL, rot_tol=ROT_TOL)
    return checker

def force_from_reaction(reaction):
    return reaction[:3]

def torque_from_reaction(reaction):
    return reaction[3:]

def evaluate_stiffness(extrusion_path, element_from_id, elements, checker=None, verbose=True):
    # TODO: check each connected component individually
    if not elements:
        return Deformation(True, {}, {}, {})

    if checker is None:
        checker = create_stiffness_checker(extrusion_path, verbose=False)

    #nodal_loads = checker.get_nodal_loads(existing_ids=[], dof_flattened=False) # per node
    #weight_loads = checker.get_self_weight_loads(existing_ids=[], dof_flattened=False) # get_nodal_loads = get_self_weight_loads?
    #for node in sorted(nodal_load):
    #    print(node, nodal_loads[node] - weight_loads[node])

    is_stiff = checker.solve(exist_element_ids=elements, if_cond_num=True)
    #print("has stored results: {0}".format(checker.has_stored_result()))
    success, nodal_displacement, fixities_reaction, element_reaction = checker.get_solved_results()
    assert is_stiff == success, "full structure not stiff!"
    displacements = {i: Displacement(*d) for i, d in nodal_displacement.items()}
    fixities = {i: Reaction(*d) for i, d in fixities_reaction.items()}
    reactions = {i: (Reaction(*d[0]), Reaction(*d[1])) for i, d in element_reaction.items()}

    #print("nodal displacement (m/rad):\n{0}".format(nodal_displacement)) # nodes x 7
    # TODO: investigate if nodal displacement can be used to select an ordering
    #print("fixities reaction (kN, kN-m):\n{0}".format(fixities_reaction)) # ground x 7
    #print("element reaction (kN, kN-m):\n{0}".format(element_reaction)) # elements x 13
    trans_tol, rot_tol = checker.get_nodal_deformation_tol()
    max_trans, max_rot, max_trans_vid, max_rot_vid = checker.get_max_nodal_deformation()
    # The inverse of stiffness is flexibility or compliance

    translation = np.max(np.linalg.norm([d[:3] for d in displacements.values()], ord=2, axis=1))
    rotation = np.max(np.linalg.norm([d[3:] for d in displacements.values()], ord=2, axis=1))
    is_stiff &= (translation <= trans_tol) and (rotation <= rot_tol)

    if verbose:
        print('Stiff: {} | Compliance: {:.5f}'.format(is_stiff, checker.get_compliance()))
        print('Max translation deformation: {0:.5f} / {1:.5} = {2:.5}, at node #{3}'.format(
            max_trans, trans_tol, max_trans / trans_tol, max_trans_vid))
        print('Max rotation deformation: {0:.5f} / {1:.5} = {2:.5}, at node #{3}'.format(
            max_rot, rot_tol, max_rot / rot_tol, max_rot_vid))

    #disc = 10
    #time_step = 1.0
    #orig_beam_shape = checker.get_original_shape(disc=disc, draw_full_shape=False)
    #beam_disp = checker.get_deformed_shape(exagg_ratio=1.0, disc=disc)
    return Deformation(is_stiff, displacements, fixities, reactions)


def test_stiffness(extrusion_path, element_from_id, elements, **kwargs):
    return evaluate_stiffness(extrusion_path, element_from_id, elements, **kwargs).success
