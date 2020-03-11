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
from coop_assembly.planning import set_camera

from pybullet_planning import connect, wait_for_user, set_camera_pose, create_plane, get_pose, set_pose, multiply, \
    set_color, RED, BLUE, apply_alpha, set_point, Point, unit_pose, draw_pose, tform_from_pose, Pose, Point, Euler, tform_point

@pytest.fixture
def save_dir():
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(here, 'test_data')


@pytest.mark.gen_from_pts
# @pytest.mark.parametrize('test_set_name', [('single_cube'), ('YJ_12_bars')])
# @pytest.mark.parametrize('test_set_name', [('YJ_12_bars')])
@pytest.mark.parametrize('test_set_name', [('single_tet')])
@pytest.mark.parametrize('radius', [(3.17), ])
# @pytest.mark.parametrize('pt_search_method', [('point2point'), ])
@pytest.mark.parametrize('pt_search_method', [('point2triangle'), ])
# @pytest.mark.parametrize('pt_search_method', [('point2point'), ('point2triangle')])
def test_generate_from_points(viewer, points_library, test_set_name, radius, pt_search_method, save_dir, write):

    points, base_tri_pts = points_library[test_set_name]
    print('\n' + '#'*10)
    print('Testing generate from point for set: {}, total # of pts: {}'.format(test_set_name, len(points)))

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
        tet_node_ids = point2triangle_tet_sequencing(points, start_tri_ids)
    else:
        raise NotImplementedError('search method not implemented!')

    b_struct_data, o_struct_data = execute_from_points(points, tet_node_ids, radius, correct=False, check_collision=True, viewer=viewer)
    if write:
        export_structure_data(save_dir, b_struct_data, o_struct_data, file_name=test_set_name+'_'+pt_search_method+'.json')

    if viewer:
        wait_for_user()

@pytest.mark.gen_grasp_planes
@pytest.mark.parametrize('test_file_name', [('YJ_12_bars_point2triangle.json'),])
def test_gen_grasp_planes(viewer, test_file_name, save_dir):
    b_struct_data, o_struct_data, _ = parse_saved_structure_data(os.path.join(save_dir, test_file_name))

    built_plate_z = -25

    connect(use_gui=viewer)
    floor = create_plane()
    set_point(floor, Point(x=1.2, z=built_plate_z*1e-3))
    # draw_pose(unit_pose())

    o_struct = OverallStructure.from_data(o_struct_data)
    b_struct = BarStructure.from_data(b_struct_data)
    b_struct.create_pb_bodies()
    o_struct.struct_bar = b_struct # TODO: better way to do this

    set_camera([attr['point_xyz'] for v, attr in o_struct.vertices(True)])

    nb_rot, nb_trans = 4, 4
    rot_angle = np.pi / 6
    trans_distance = 30

    for v in b_struct.vertex:
        bar_body = b_struct.get_bar_pb_body(v)
        set_color(bar_body, apply_alpha(RED, 0.1))

    seq = [v for v in b_struct.vertex]
    for v in b_struct.vertex:
        print('################')
        print('bar #{}'.format(v))

        bar_body = b_struct.get_bar_pb_body(v)
        world_from_bar = get_pose(bar_body)

        calculate_gripping_plane(b_struct, v, b_struct.vertex[v]["mean_point"], nb_rot=nb_rot, nb_trans=nb_trans)
        calculate_offset(o_struct, b_struct, v, rot_angle=rot_angle, trans_distance=trans_distance, sequence=seq, scale=1e-3,
            obstacles=[floor], built_plate_z=built_plate_z, method='SP') # sample

        # world_from_tf = pb_pose_from_Transformation(tf)
        # set_pose(bar_body, multiply(world_from_tf, world_from_bar))
        # set_color(bar_body, BLUE)
        # wait_for_user()

        set_pose(bar_body, world_from_bar)
        set_color(bar_body, apply_alpha(RED, 0.5))
