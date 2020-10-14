import os
from collections import namedtuple
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

from pybullet_planning import HideOutput, INF, apply_alpha, RED, get_distance

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

def conmech_model_from_bar_structure(bar_struct, grounded_connectors, chosen_bars=None):
    # nodes, elements, supports, joints, materials, crosssecs, model_name, unit = \
    #     read_frame_json(json_file_path, verbose=verbose)
    element_from_index, grounded_elements, contact_from_connectors, connectors = \
        unpack_structure(bar_struct, chosen_bars=chosen_bars, scale=METER_SCALE)

    # Element = namedtuple('Element', ['index', 'axis_endpoints', 'radius', 'body', 'initial_pose', 'goal_pose',
    #                                  'grasps', 'goal_supports', 'layer'])

    # ! meter-based unit
    unit = "meter"

    # TODO break the lines into pieces according to the connectors
    # axis points should all be unique
    node_ind = 0
    cm_nodes = []
    cm_elements = []
    for e_id, element in element_from_index.items():
        # Node(self, point, node_ind, is_grounded):
        cm_nodes.append(Node(element.axis_endpoints[0], e_id*2, False))
        cm_nodes.append(Node(element.axis_endpoints[1], e_id*2+1, False))
        # Element(self, end_node_inds, elem_ind, elem_tag='', bending_stiff=True):
        cm_elements.append(Element([e_id*2, e_id*2+1], e_id, elem_tag='bar', bending_stiff=True))

    n_id = len(cm_nodes)
    e_id = len(cm_elements)
    for c_id, connector_endpts in contact_from_connectors.items():
        connector_node_inds = []
        for i in range(2):
            pt = connector_endpts[i]
            for n in cm_nodes:
                if get_distance(pt, n.point) < EPS:
                    connector_node_inds[]


    # Support(self, condition, node_ind):
    # Joint(self, c_conditions, elem_tags):
    # CrossSec(self, A, Jx, Iy, Iz, elem_tags=None, family='unnamed', name='unnamed'):
    # Material(self, E, G12, fy, density, elem_tags=None, family='unnamed', name='unnamed', type_name='ISO'):

    return nodes, elements, supports, joints, materials, crosssecs, model_name, unit

def create_stiffness_checker(bar_structure_path, verbose=False):
    if not USE_CONMECH:
        return None
    bar_struct, o_struct = load_structure(bar_structure_path, viewer=False, color=apply_alpha(RED, 0))

    # * add grounded connector
    grounded_connectors = bar_struct.generate_grounded_connection()

    # * convert ball structure as
    nodes, elements, supports, joints, materials, crosssecs, model_name, unit = \
        conmech_model_from_bar_structure(bar_struct, grounded_connectors)

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
