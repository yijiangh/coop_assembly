import os
import pytest
from itertools import combinations
import numpy as np

from compas.datastructures import Network
from compas.geometry import scale_vector
from compas_fab.backends.pybullet import pb_pose_from_Transformation

from coop_assembly.data_structure import BarStructure, OverallStructure
from coop_assembly.help_functions import find_point_id, tet_surface_area, \
    tet_volume, distance_point_triangle
from coop_assembly.geometry_generation.tet_sequencing import \
    compute_distance_from_grounded_node
from coop_assembly.geometry_generation.tet_sequencing import \
    get_pt2tri_search_heuristic_fn, \
    point2point_shortest_distance_tet_sequencing, \
    point2triangle_tet_sequencing
from coop_assembly.geometry_generation.execute import execute_from_points
from coop_assembly.assembly_info_generation import calculate_gripping_plane, calculate_offset, contact_info_from_seq
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE
from coop_assembly.planning.visualization import set_camera, SHADOWS, BACKGROUND_COLOR, label_elements
from coop_assembly.planning.utils import load_world

from coop_assembly.geometry_generation.generate_tetrahedra import generate_structure_from_points

from pybullet_planning import connect, wait_for_user, set_camera_pose, create_plane, get_pose, set_pose, multiply, \
    set_color, RED, BLUE, GREEN, TAN, GREY, apply_alpha, set_point, Point, unit_pose, draw_pose, tform_from_pose, Pose, Point, Euler, tform_point, \
    pairwise_collision_info, draw_collision_diagnosis, pairwise_collision, dump_world, get_bodies

@pytest.fixture
def save_dir():
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(here, 'test_data')

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

    # create pybullet env
    # set_camera(points)
    # create_plane()

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
        export_structure_data(save_dir, b_struct.data, o_struct.data, file_name=file_spec+'_'+pt_search_method+'.json')

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
        b1_body = b_struct.get_bar_pb_body(bar1, apply_alpha(RED, 0.1))
        b2_body = b_struct.get_bar_pb_body(bar2, apply_alpha(TAN, 0.1))
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
