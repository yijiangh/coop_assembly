import os
import pytest
from itertools import combinations
import numpy as np
from numpy.linalg import norm
from termcolor import cprint

from compas.datastructures import Network
from compas.geometry import scale_vector, closest_point_on_segment
# from compas_fab.backends.pybullet import pb_pose_from_Transformation

from coop_assembly.data_structure import BarStructure, OverallStructure
from coop_assembly.help_functions import find_point_id, tet_surface_area, \
    tet_volume, distance_point_triangle, dropped_perpendicular_points
from coop_assembly.geometry_generation.tet_sequencing import \
    compute_distance_from_grounded_node
from coop_assembly.geometry_generation.tet_sequencing import \
    get_pt2tri_search_heuristic_fn, \
    point2point_shortest_distance_tet_sequencing, \
    point2triangle_tet_sequencing
from coop_assembly.geometry_generation.execute import execute_from_points
from coop_assembly.geometry_generation.generate_truss import gen_truss
from coop_assembly.geometry_generation.tangents import lines_tangent_to_cylinder

from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE
from coop_assembly.planning.visualization import set_camera, SHADOWS, BACKGROUND_COLOR, label_elements
from coop_assembly.planning.utils import load_world, get_element_neighbors, get_connector_from_elements, check_connected, get_connected_structures, \
    flatten_commands
from coop_assembly.planning.visualization import color_structure, draw_ordered, draw_element, label_elements, label_connector, set_camera, draw_partial_ordered

# from coop_assembly.assembly_info_generation import calculate_gripping_plane, calculate_offset, contact_info_from_seq
from coop_assembly.geometry_generation.generate_tetrahedra import generate_structure_from_points

from pybullet_planning import connect, wait_for_user, set_camera_pose, create_plane, get_pose, set_pose, multiply, \
    set_color, RED, BLUE, GREEN, TAN, GREY, apply_alpha, set_point, Point, unit_pose, draw_pose, tform_from_pose, Pose, Point, Euler, tform_point, \
    pairwise_collision_info, draw_collision_diagnosis, pairwise_collision, dump_world, get_bodies, wait_if_gui, reset_simulation, remove_handles, \
    disconnect, add_line

@pytest.fixture
def save_dir():
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(here, 'test_data')

def np_norm(p):
    return norm(np.array(p))

def np_dot(p1, p2):
    return np.array(p1).dot(np.array(p2))

def diff_norm(p1, p2):
    return norm(np.array(p1) - np.array(p2))

@pytest.mark.tan
def test_lines_tangent_to_cylinder():
    # testing computing tangent planes passing a given point and a cylinder
    ref_point = np.array([1,0,0])
    cylinder_line = [np.array([0,0,0]), np.array([0,1,0])]
    dist = 0.5
    ptM, delta_up, delta_down = list(map(np.array, lines_tangent_to_cylinder(cylinder_line, ref_point, dist)))
    assert norm(ptM-np.zeros(3)) < 1e-12
    assert abs(norm(delta_up) - dist) < 1e-12
    assert abs(norm(delta_down) - dist) < 1e-12
    assert abs((ptM+delta_up-ref_point).dot(delta_up)) < 1e-12
    assert abs((ptM+delta_down-ref_point).dot(delta_down)) < 1e-12

    # flipping the line vector direction would only swap the up/down delta vectors
    ptM2, delta_up2, delta_down2 = list(map(np.array, lines_tangent_to_cylinder(cylinder_line[::-1], ref_point, dist)))
    assert norm(ptM - ptM2) < 1e-12
    assert norm(delta_up - delta_down2) < 1e-12
    assert norm(delta_down - delta_up2) < 1e-12

    for _ in range(10):
        theta = np.random.rand()*2*np.pi
        # random_y = np.random.random()*1e5
        random_y = 0
        r = dist
        assert(lines_tangent_to_cylinder(cylinder_line, np.array([r*np.cos(theta),random_y,r*np.sin(theta)]), dist) is None)
        r = dist+1e-8
        assert(lines_tangent_to_cylinder(cylinder_line, np.array([r*np.cos(theta),random_y,r*np.sin(theta)]), dist))

@pytest.mark.drop_contact
def test_contact_pts():
    # intersecting case
    line1 = [np.array([-1.,0,0]), np.array([1.,0,0])]
    line2 = [np.array([1.,1,0]), np.array([-1.,-1.,0])]
    pt0, pt1 = dropped_perpendicular_points(*line1, *line2)
    assert norm(pt0 - np.array([0.,0,0])) < 1e-12
    assert norm(pt0 - np.array([0.,0,0])) < 1e-12

    # parallel case
    line1 = [np.array([0.,0,0]), np.array([1.,0,0])]
    line2 = [np.array([0.,0,1]), np.array([1.,0,1])]
    pt0, pt1 = dropped_perpendicular_points(*line1, *line2)
    # any line in-between would be valid
    assert abs((pt0-pt1).dot(line1[0]-line1[1])) < 1e-12
    assert abs(norm(pt1-pt0) - 1.0) < 1e-12

    # colinear, non-touching case
    line1 = [np.array([0.,0,0]), np.array([1.,0,0])]
    line2 = [np.array([1.5,0,0]), np.array([2.,0,0])]
    pt0, pt1 = dropped_perpendicular_points(*line1, *line2)
    assert norm(pt0 - line1[1]) < 1e-12
    assert norm(pt1 - line2[0]) < 1e-12

    # colinear, touching case
    line1 = [np.array([0.,0,0]), np.array([1.,0,0])]
    line2 = [np.array([1.,0,0]), np.array([2.,0,0])]
    pt0, pt1 = dropped_perpendicular_points(*line1, *line2)
    assert norm(pt0 - line1[1]) < 1e-12
    assert norm(pt1 - line2[0]) < 1e-12

    # colinear, overlapped case
    line1 = [np.array([0.,0,0]), np.array([1.,0,0])]
    line2 = [np.array([0.5,0,0]), np.array([2.,0,0])]
    pt0, pt1 = dropped_perpendicular_points(*line1, *line2)
    assert norm(pt0 - line1[1]) < 1e-12
    assert norm(pt1 - line1[1]) < 1e-12

# @pytest.mark.cl_seg
def test_closest_pt_segment():
    pt = np.array([0.,1,0])
    line = [np.array([1.,0,0]), np.array([2.,0,0])]
    cl_pt = closest_point_on_segment(pt, line)
    assert norm(cl_pt - line[0]) < 1e-12

    pt = np.array([0.5,0,0])
    line = [np.array([0.,0,0]), np.array([1.,0,0])]
    cl_pt = closest_point_on_segment(pt, line)
    assert norm(cl_pt - pt) < 1e-12

@pytest.mark.gen_truss
@pytest.mark.parametrize('radius', [(3.17), ])
# @pytest.mark.parametrize('truss_problem', [(''), ])
def test_gen_truss(viewer, save_dir, truss_problem, radius, write):

    export_file_name = truss_problem
    if 'skeleton' in export_file_name:
        export_file_name = export_file_name.split('_skeleton')[0] + '.json'

    bar_struct = gen_truss(truss_problem, viewer=viewer, radius=radius, write=write, save_dir=save_dir, file_name=export_file_name, debug=False)
    reset_simulation()
    disconnect()

    connect(use_gui=viewer, shadows=True, color=BACKGROUND_COLOR)
    draw_pose(unit_pose())

    # built plate
    new_base = np.array([650, 0, 23])*1e-3
    # floor = create_plane(color=BLUE)
    # set_point(floor, Point(z=new_base[2]))

    bar_struct.get_element_bodies(apply_alpha(RED, 0.3))
    # focus camera
    endpts_from_element = bar_struct.get_axis_pts_from_element(scale=1e-3)
    set_camera([np.array(p[0]) for e, p in endpts_from_element.items()])
    wait_if_gui('before tf')

    bar_struct.transform(new_base, scale=1e-3)
    element_bodies = bar_struct.get_element_bodies(apply_alpha(RED, 0.3))
    endpts_from_element = bar_struct.get_axis_pts_from_element(scale=1e-3)
    set_camera([np.array(p[0]) for e, p in endpts_from_element.items()])
    print('new base_centroid:', bar_struct.base_centroid())
    wait_if_gui('after tf')

    endpts_from_element = bar_struct.get_axis_pts_from_element(scale=1e-3)
    handles = []
    handles.extend(label_elements(element_bodies))
    wait_if_gui('reconstructed truss axes labeled.')
    remove_handles(handles)

    elements = list(element_bodies.keys())
    contact_from_connectors = bar_struct.get_connectors(scale=1e-3)
    connectors = list(contact_from_connectors.keys())

    # TODO: sometimes there are excessive connectors
    # * connectors from bar
    print('Visualize connectors.')
    connector_from_elements = get_connector_from_elements(connectors, elements)
    for bar in bar_struct.vertices():
        handles = []
        bar_connectors = connector_from_elements[bar]
        for c in list(bar_connectors):
            handles.append(add_line(*contact_from_connectors[c], color=(1,0,0,1), width=2))
        color_structure(element_bodies, set(), next_element=bar, built_alpha=0.6)
        wait_if_gui('connector: {}'.format(bar_connectors))
        remove_handles(handles)

    # * neighbor elements from elements
    print('Visualize neighnor elements.')
    element_neighbors = get_element_neighbors(connectors, elements)
    for element, connected_bars in element_neighbors.items():
        color_structure(element_bodies, connected_bars, element, built_alpha=0.6)
        wait_if_gui('connected neighbors: {} | {}'.format(element, connected_bars))

    # TODO: some sanity check here
    # mutual collision checks
    is_collided = False
    for bar1, bar2 in connectors:
        b1_body = bar_struct.get_bar_pb_body(bar1, apply_alpha(RED, 0.5))
        b2_body = bar_struct.get_bar_pb_body(bar2, apply_alpha(TAN, 0.5))
        if b1_body is None or b2_body is None:
            continue

        assert len(get_bodies()) == len(element_bodies)
        # dump_world()

        if pairwise_collision(b1_body, b2_body):
            cr = pairwise_collision_info(b1_body, b2_body)
            draw_collision_diagnosis(cr, focus_camera=True)
            is_collided = True
            if not viewer:
                assert False, '{}-{} collision!'.format(b1_body, b2_body)
        print('-'*10)

    cprint('Valid: {}'.format(not is_collided), 'red' if is_collided else 'green')
    wait_if_gui('Done.')

@pytest.mark.gen_from_pts
@pytest.mark.parametrize('radius', [(3.17), ])
@pytest.mark.parametrize('pt_search_method', [('point2triangle'), ]) #('point2point'), ('point2triangle')
def test_generate_from_points(save_dir, points_library, viewer, file_spec, radius, pt_search_method, write):
    points, base_tri_pts = points_library[file_spec]
    print('\n' + '#'*10)
    print('Testing generate from point for set: {}, total # of pts: {}'.format(file_spec, len(points)))

    # affine transf, in millimeter
    tform = Pose(Point(0,0,0))
    for i, pt in enumerate(points):
        points[i] = tform_point(tform, pt)
    for i, pt in enumerate(base_tri_pts):
        base_tri_pts[i] = tform_point(tform, pt)

    start_tri_ids = [find_point_id(base_pt, points) for base_pt in base_tri_pts]
    assert len(start_tri_ids) == 3, 'start triangle should only have three points!'
    print('base triangle ids: {}'.format(start_tri_ids))

    if pt_search_method == 'point2point':
        cost_from_node = {}
        all_pt_ids = list(range(len(points)))
        elements = list(combinations(all_pt_ids, 2))
        cost_from_node = compute_distance_from_grounded_node(elements, points, start_tri_ids)
        tet_node_ids = point2point_shortest_distance_tet_sequencing(points, cost_from_node)
    elif pt_search_method == 'point2triangle':
        ordering_heuristic = 'tet_surface_area'
        penalty_cost = 2.0
        print('pt search strategy: {} | heuristic: {} | penalty cost: {}'.format(pt_search_method, ordering_heuristic, penalty_cost))
        heuristic_fn = get_pt2tri_search_heuristic_fn(points, penalty_cost, ordering_heuristic)
        tet_node_ids = point2triangle_tet_sequencing(points, start_tri_ids, heuristic_fn=heuristic_fn)
    else:
        raise NotImplementedError('search method not implemented!')

    # b_struct_data, o_struct_data = execute_from_points(points, tet_node_ids, radius, correct=False, check_collision=True, viewer=viewer)
    b_struct = BarStructure()
    o_struct = OverallStructure(b_struct)
    generate_structure_from_points(o_struct, b_struct, radius, points, tet_node_ids,
        correct=False, check_collision=True, viewer=viewer)

    if write:
        export_structure_data(b_struct.data, o_struct.data, save_dir=save_dir, file_name=file_spec+'_'+pt_search_method+'.json')

    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    # _, _ = load_world()
    dump_world()
    element_bodies = b_struct.get_element_bodies(color=apply_alpha(RED, 0))
    set_camera([attr['point_xyz'] for v, attr in o_struct.vertices(True)])

    handles = []
    handles.extend(label_elements(element_bodies))

    # * checking mutual collision between bars
    # TODO move this complete assembly collision sanity check to bar structure class
    contact_from_connectors = b_struct.get_connectors(scale=1e-3)
    connectors = list(contact_from_connectors.keys())
    for bar1, bar2 in connectors:
        b1_body = b_struct.get_bar_pb_body(bar1, apply_alpha(RED, 0.5))
        b2_body = b_struct.get_bar_pb_body(bar2, apply_alpha(TAN, 0.5))
        assert len(get_bodies()) == len(element_bodies)
        # dump_world()

        if pairwise_collision(b1_body, b2_body):
            cr = pairwise_collision_info(b1_body, b2_body)
            draw_collision_diagnosis(cr, focus_camera=True)
            if not viewer:
                assert False, '{}-{} collision!'.format(b1_body, b2_body)
        print('-'*10)

    if viewer:
        wait_for_user('Done.')
