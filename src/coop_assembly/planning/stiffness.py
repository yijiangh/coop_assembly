import heapq
import os
import random
import time
import json
import numpy as np
from numpy.linalg import norm

from collections import namedtuple, defaultdict
from itertools import combinations, product, combinations_with_replacement
from termcolor import cprint

try:
    from pyconmech.frame_analysis import StiffnessChecker
    from pyconmech.frame_analysis import GravityLoad, Node, Element, Support, Material, CrossSec, Material, Joint, Model, LoadCase
except ImportError as e:
    cprint('{}, Not using conmech'.format(e), 'yellow')
    USE_CONMECH = False
    input("Press Enter to continue...")
else:
    USE_CONMECH = True

from pybullet_planning import HideOutput, INF, apply_alpha, RED, get_distance, angle_between, get_difference, get_angle, has_gui
from pybullet_planning import RED, BLUE, GREEN, BLACK, TAN, add_line, set_color, apply_alpha, \
    set_camera_pose, add_text, draw_pose, get_pose, wait_for_user, wait_for_duration, get_name, wait_if_gui, remove_all_debug, remove_body, \
    remove_handles, remove_debug, LockRenderer, draw_point, elapsed_time, randomize

from compas.geometry import is_colinear
from coop_assembly.help_functions.shared_const import METER_SCALE, EPS
from coop_assembly.data_structure import GROUND_INDEX
from coop_assembly.planning.parsing import load_structure, unpack_structure, PICKNPLACE_DIRECTORY
from coop_assembly.planning.utils import check_connected, compute_z_distance

# TRANS_TOL = 0.003
# * SP Arch
# TRANS_TOL = 0.01
# * Other
TRANS_TOL = 0.005

ROT_TOL = INF # 5 * np.pi / 180

####################################

# TODO: get_max_nodal_deformation
Deformation = namedtuple('Deformation', ['success', 'displacements', 'fixities', 'reactions'])
Displacement = namedtuple('Displacement', ['dx', 'dy', 'dz', 'theta_x', 'theta_y', 'theta_z'])
Reaction = namedtuple('Reaction', ['fx', 'fy', 'fz', 'mx', 'my', 'mz'])

####################################
# E, G12, fy, density
# ! NOT USED NOW: fy is the material strength in the specified direction (local x direction)
# element_tags = None is the fall-back material entry

def wood_material(elem_tags=None):
    return Material(1050*1e4, 360*1e4, 1.3*1e4, 6, elem_tags=elem_tags, family='Wood', name='Wood', type_name='ISO')
# STEEL_MATERIAL = Material(1050*1e4, 360*1e4, 1.3*1e4, 78.5, elem_tags=None, family='Steel', name='Steel', type_name='ISO')

CONNECTOR_STENGTH_RATIO = 0.1
def connector_material(elem_tags=None):
    return Material(1050*1e4*CONNECTOR_STENGTH_RATIO, 360*1e4*CONNECTOR_STENGTH_RATIO, 1.3*1e4, 6,
        elem_tags=elem_tags, family='Connector', name='Glue', type_name='ISO')

ROTATIONAL_STIFFNESS = 100 # kn/rad

def A_solid_cir(radius):
    return np.pi * radius**2

def Jx_solid_cir(radius):
    # https://en.wikipedia.org/wiki/Polar_moment_of_inertia
    return np.pi * radius**4 / 2

def Iy_solid_cir(radius):
    # https://www.engineeringtoolbox.com/area-moment-inertia-d_1328.html
    return np.pi * radius**4 / 4

def solid_cir_crosssec(r, elem_tags=None):
    A = A_solid_cir(r)
    Jx = Jx_solid_cir(r)
    Iy = Iy_solid_cir(r)
    return CrossSec(A, Jx, Iy, Iy, elem_tags=elem_tags, family='Circular', name='solid_circle')

####################################

def find_nodes(new_pts, cm_nodes, tol=1e-6):
    node_inds = list()
    for pt in new_pts:
        # print('---\nPt: {}'.format(pt))
        for n in cm_nodes:
            if get_distance(pt, n.point) < tol:
                node_inds.append(n.node_ind)
                # print('Found: #{} | {}'.format(n.node_ind, n.point))
                break
        else:
            # unseen node
            n_id = len(cm_nodes)
            cm_nodes.append(Node(pt, n_id, False)) # grounded flag
            node_inds.append(n_id)
            # print('Not found: #{} | {}'.format(n_id, pt))
    return cm_nodes, node_inds


def conmech_model_from_bar_structure(bar_struct, debug=False, save_model=False):
    element_from_index, grounded_elements, contact_from_connectors, connectors = \
        unpack_structure(bar_struct, scale=METER_SCALE, color=apply_alpha(RED,0))
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
    pts_from_element = defaultdict(list)
    fem_element_from_bar_id = defaultdict(set)
    for e_id, element in element_from_index.items():
        # Node(self, point, node_ind, is_grounded):
        # ! is_grounded is used to mark grounded nodes for construction purpose
        # structural supports are specified using the Support entries
        cm_nodes.append(Node(element.axis_endpoints[0], e_id*2, False))
        cm_nodes.append(Node(element.axis_endpoints[1], e_id*2+1, False))
        pts_from_element[e_id].extend(element.axis_endpoints)

    supports = []
    connector_tags = set()
    for c_id, connector_endpts in contact_from_connectors.items():
        cm_nodes, node_inds = find_nodes(connector_endpts, cm_nodes)
        for pt in connector_endpts:
            for e_id in c_id:
                if e_id != GROUND_INDEX and is_colinear(pt, *element_from_index[e_id].axis_endpoints, tol=1e-6):
                    pts_from_element[e_id].append(pt)

        assert len(node_inds) == 2
        assert node_inds[0] != node_inds[1], "#{}|{} - {}".format(c_id, connector_endpts, node_inds)

        # * add connector element
        # Element(self, end_node_inds, elem_ind, elem_tag='', bending_stiff=True)
        e_id = len(cm_elements)
        e_tag = 'connector{}'.format(e_id)
        cm_elements.append(Element(tuple(node_inds), e_id, elem_tag=e_tag, bending_stiff=True))
        connector_tags.add(e_tag)
        for e in c_id:
            fem_element_from_bar_id[e].add(e_id)

        if c_id in grounded_connectors:
            # TODO should only chosen one contact point as fixiety
            # Support(self, condition, node_ind)
            supports.extend([Support([1 for _ in range(6)], node_inds[i]) for i in range(2)])

    bar_tags = set()
    for e_id, seg_pts in pts_from_element.items():
        sorted_pt_pairs = sorted(combinations(list(seg_pts), 2), key=lambda pt_pair: get_distance(*pt_pair))
        e_end_pts = element_from_index[e_id].axis_endpoints
        farthest_pts = sorted_pt_pairs[-1]
        if get_angle(get_difference(*farthest_pts), get_difference(*e_end_pts)) > np.pi/2:
            farthest_pts = farthest_pts[::-1]
        sorted_seg_pts = sorted(list(seg_pts), key=lambda pt: get_distance(pt, farthest_pts[0]))
        # print(sorted_seg_pts)
        cm_nodes, node_inds = find_nodes(list(sorted_seg_pts), cm_nodes)
        assert len(sorted_seg_pts) == len(node_inds)
        # print(node_inds)
        # elem_tag = 'bar{}'.format(e_id)
        elem_tag = 'bar{}'.format(e_id)
        bar_tags.add(elem_tag)
        seg_id = 0
        for i in range(len(sorted_seg_pts)-1):
            if node_inds[i] != node_inds[i+1]:
                new_element = Element((node_inds[i], node_inds[i+1]), len(cm_elements), elem_tag=elem_tag, bending_stiff=True)
                cm_elements.append(new_element)
                fem_element_from_bar_id[e_id].add(new_element.elem_ind)
                seg_id += 1
                # print(cm_elements[-1])
        # input()
    # TODO Add rotational stiffness later
    # Joint(self, c_conditions, elem_tags):
    node_c_conditions = [None, None, None] + [ROTATIONAL_STIFFNESS for _ in range(3)]
    # joint = Joint(node_c_conditions+node_c_conditions, list(bar_tags | connector_tags))
    joints = []

    # TODO different material property and cross secs on Element and Connectors
    r = list(element_from_index.values())[0].radius # in meter
    crosssecs = [solid_cir_crosssec(r, elem_tags=list(bar_tags | connector_tags))]
    materials = [wood_material(list(bar_tags)), connector_material(list(connector_tags))]

    model = Model(cm_nodes, cm_elements, supports, joints, materials, crosssecs, unit=unit, model_name=model_name)
    if save_model:
        model_path = os.path.join(PICKNPLACE_DIRECTORY, model.model_name.split(".json")[0] + '_conmech_model.json')
        model_data = model.to_data()
        model_data['fem_element_from_bar_id'] = {bar : list(fem_es) for bar, fem_es in fem_element_from_bar_id.items()}
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=1)
        cprint('Conmech model saved to: {}'.format(model_path), 'green')

    if debug and has_gui():
        h = []
        with LockRenderer(False):
            # for n in cm_nodes:
            #     draw_point(n.point, size=0.002, color=GREEN if n.is_grounded else BLACK)
            for e_id, elem_ids in fem_element_from_bar_id.items():
                print('E#{}: {}'.format(e_id, elem_ids))
                for elem_id in elem_ids:
                    e = cm_elements[elem_id]
                    node_inds = e.end_node_inds
                    if 'bar' in e.elem_tag:
                        th = add_line(cm_nodes[node_inds[0]].point, cm_nodes[node_inds[1]].point, color=(random.random(), random.random(), random.random(), 1), width=3)
                    elif 'connector' in e.elem_tag:
                        th = add_line(cm_nodes[node_inds[0]].point, cm_nodes[node_inds[1]].point, color=BLUE, width=3)
                    h.append(th)
                # input()
            for s in supports:
                ph = draw_point(cm_nodes[s.node_ind].point, size=0.02, color=RED)
                h.append(ph)
        wait_if_gui()
        remove_handles(h)

    return model, fem_element_from_bar_id

def create_stiffness_checker(bar_struct, verbose=False, **kwargs):
    if not USE_CONMECH:
        return None
    # * convert ball structure to a conmech model
    model, fem_element_from_bar_id = conmech_model_from_bar_structure(bar_struct, **kwargs)
    with HideOutput():
        # checker = StiffnessChecker.from_json(json_file_path=model_path, verbose=verbose)
        checker = StiffnessChecker(model, verbose=verbose, checker_engine="numpy")
        checker.set_loads(LoadCase(gravity_load=GravityLoad([0,0,-1.0])))
    #checker.set_output_json(True)
    #checker.set_output_json_path(file_path=os.getcwd(), file_name="stiffness-results.json")
    checker.set_nodal_displacement_tol(trans_tol=TRANS_TOL, rot_tol=ROT_TOL)
    return checker, fem_element_from_bar_id

def force_from_reaction(reaction):
    return reaction[:3]

def torque_from_reaction(reaction):
    return reaction[3:]

def evaluate_stiffness(bar_struct, elements, checker=None, fem_element_from_bar_id=None, verbose=True):
    # TODO: check each connected component individually
    if not elements:
        return Deformation(True, {}, {}, {})

    if checker is None or fem_element_from_bar_id is None:
        checker, fem_element_from_bar_id = create_stiffness_checker(bar_struct, verbose=verbose)

    #nodal_loads = checker.get_nodal_loads(existing_ids=[], dof_flattened=False) # per node
    #weight_loads = checker.get_self_weight_loads(existing_ids=[], dof_flattened=False) # get_nodal_loads = get_self_weight_loads?
    #for node in sorted(nodal_load):
    #    print(node, nodal_loads[node] - weight_loads[node])

    exist_element_ids = set()
    for bar in elements:
        exist_element_ids.update(fem_element_from_bar_id[bar])
    # print('fem elements len: ', len(exist_element_ids))

    is_stiff = checker.solve(exist_element_ids=list(exist_element_ids), if_cond_num=True)
    if not checker.has_stored_result():
        return Deformation(False, {}, {}, {})

    #print("has stored results: {0}".format(checker.has_stored_result()))
    success, nodal_displacement, fixities_reaction, element_reaction = checker.get_solved_results()
    assert is_stiff == success, "full structure not stiff!"
    assert success or checker.get_compliance() > 0.0, 'success {} | compliance {}'.format(success, checker.get_compliance())
    displacements = {i: Displacement(*d) for i, d in nodal_displacement.items()}
    fixities = {i: Reaction(*d) for i, d in fixities_reaction.items()}
    reactions = {i: (Reaction(*d[0]), Reaction(*d[1])) for i, d in element_reaction.items()}

    #print("nodal displacement (m/rad):\n{0}".format(nodal_displacement)) # nodes x 7
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
        # print('Max rotation deformation: {0:.5f} / {1:.5} = {2:.5}, at node #{3}'.format(
        #     max_rot, rot_tol, max_rot / rot_tol, max_rot_vid))

    #disc = 10
    #time_step = 1.0
    #orig_beam_shape = checker.get_original_shape(disc=disc, draw_full_shape=False)
    #beam_disp = checker.get_deformed_shape(exagg_ratio=1.0, disc=disc)
    return Deformation(is_stiff, displacements, fixities, reactions)


def test_stiffness(bar_struct, elements, **kwargs):
    return evaluate_stiffness(bar_struct, elements, **kwargs).success

##################################################

def plan_stiffness(bar_struct, elements, initial_position=None, checker=None, fem_element_from_bar_id=None, \
        stiffness=True, heuristic='z', max_time=INF, max_backtrack=0, verbose=False):
    """use the progression algorithm to plan a stiff sequence
    """
    start_time = time.time()
    # TODO the bar index gives the algorithm hints, try other starting point
    # TODO chosen bars
    element_from_index, grounded_elements, _, connectors = \
        unpack_structure(bar_struct, chosen_bars=None, scale=METER_SCALE, color=apply_alpha(RED,0.1))
    if stiffness and (checker is None or fem_element_from_bar_id is None):
        checker, fem_element_from_bar_id = create_stiffness_checker(bar_struct)

    # all_elements = frozenset(element_from_index)
    remaining_elements = frozenset(elements)
    min_remaining = len(remaining_elements)
    max_bt = stiffness_failures = 0
    queue = [(None, frozenset(), [])]
    while queue and (elapsed_time(start_time) < max_time):
        # TODO pop position and try distance heuristic
        _, printed, sequence = heapq.heappop(queue)
        num_remaining = len(remaining_elements) - len(printed)
        backtrack = num_remaining - min_remaining
        max_bt = max(max_bt, backtrack)
        if max_backtrack < backtrack:
            break # continue

        # * check constraints
        if not check_connected(connectors, grounded_elements, printed):
            continue
        if stiffness and not test_stiffness(bar_struct, printed, checker=checker, fem_element_from_bar_id=fem_element_from_bar_id, verbose=verbose):
            stiffness_failures += 1
            continue

        if printed == remaining_elements:
            # * Done!
            #from extrusion.visualization import draw_ordered
            # distance = compute_sequence_distance(node_points, sequence, start=initial_position, end=initial_position)
            cprint('Plan-stiffness success! Elements: {}, max BT: {}, stiffness failure: {}, Time: {:.3f}sec'.format(len(sequence), max_bt, stiffness_failures, elapsed_time(start_time))) #Distance: {:.3f}m,
            #local_search(extrusion_path, element_from_id, node_points, ground_nodes, checker, sequence,
            #             initial_position=initial_position, stiffness=stiffness, max_time=INF)
            #draw_ordered(sequence, node_points)
            #wait_for_user()
            return sequence

        # * add successors
        for element in randomize(remaining_elements - printed):
            new_printed = printed | {element}
            new_sequence = sequence + [element]
            num_remaining = len(remaining_elements) - len(new_printed)
            min_remaining = min(min_remaining, num_remaining)
            # Don't count edge length
            # distance = get_distance(position, node_points[node1]) if position is not None else None
            # distance = compute_sequence_distance(node_points, new_sequence)
            if heuristic == 'none':
                bias = None
            elif heuristic == 'random':
                bias = random.random()
            elif heuristic == 'z':
                bias = compute_z_distance(element_from_index, element)
            # elif heuristic == 'distance':
            #     bias = distance
            else:
                raise ValueError(heuristic)
            #bias = heuristic_fn(printed, element, conf=None) # TODO: experiment with other biases
            priority = (num_remaining, bias, random.random())
            heapq.heappush(queue, (priority, new_printed, new_sequence))

    cprint('Failed to find stiffness plan under tol {}! Elements: {}, Min remaining {}, Time: {:.3f}sec'.format(
        TRANS_TOL, len(remaining_elements), min_remaining, elapsed_time(start_time)), 'red')
    return None

##################################################
