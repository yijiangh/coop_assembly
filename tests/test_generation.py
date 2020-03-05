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
from coop_assembly.assembly_info_generation.offset_motion import get_offset_collision_test
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE
from coop_assembly.planning import set_camera

from pybullet_planning import connect, wait_for_user, set_camera_pose, create_plane, get_pose, set_pose, multiply, \
    set_color, RED, BLUE, apply_alpha, set_point, Point, unit_pose, draw_pose

@pytest.fixture
def save_dir():
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(here, 'test_data')


@pytest.mark.gen_from_pts
# @pytest.mark.parametrize('test_set_name', [('single_cube'), ('YJ_12_bars')])
@pytest.mark.parametrize('test_set_name', [('YJ_12_bars')])
@pytest.mark.parametrize('radius', [(3.17), ])
# @pytest.mark.parametrize('pt_search_method', [('point2point'), ])
@pytest.mark.parametrize('pt_search_method', [('point2triangle'), ])
# @pytest.mark.parametrize('pt_search_method', [('point2point'), ('point2triangle')])
def test_generate_from_points(viewer, points_library, test_set_name, radius, pt_search_method, save_dir, write):

    points, base_tri_pts = points_library[test_set_name]
    print('\n' + '#'*10)
    print('Testing generate from point for set: {}, total # of pts: {}'.format(test_set_name, len(points)))

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
        calculate_offset(o_struct, b_struct, v, rot_angle=rot_angle, trans_distance=trans_distance, sequence=seq, scale=1e-3, obstacles=[floor], built_plate_z=built_plate_z, method='sample')

        # world_from_tf = pb_pose_from_Transformation(tf)
        # set_pose(bar_body, multiply(world_from_tf, world_from_bar))
        # set_color(bar_body, BLUE)
        # wait_for_user()

        set_pose(bar_body, world_from_bar)
        set_color(bar_body, apply_alpha(RED, 0.5))

@pytest.mark.ignore('')
@pytest.mark.collision_contact
@pytest.mark.parametrize('test_file_name', [('YJ_12_bars_point2triangle.json'),])
def test_collision_contact(viewer, test_file_name, save_dir):
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

    for bar_vkey in b_struct.vertex:
        print('-------------')
        print('Bar #{}'.format(bar_vkey))

        assembled_bv = list(range(bar_vkey))
        bar_vertex = b_struct.vertex[bar_vkey]
        bar_body = bar_vertex['pb_body']
        built_obstacles = [floor] + [b_struct.vertex[bv]['pb_body'] for bv in assembled_bv]

        for v in b_struct.vertex:
            body = b_struct.get_bar_pb_body(v)
            set_color(body, apply_alpha(RED, 0.1))
        for v in assembled_bv:
            body = b_struct.get_bar_pb_body(v)
            set_color(body, apply_alpha(RED, 0.4))
        set_color(bar_body, apply_alpha(BLUE, 0.4))

        contact_vec_1, contact_vec_2, contact_pt_1, contact_pt_2 = contact_info_from_seq(o_struct, b_struct, bar_vkey, assembled_bv, verbose=True)

        world_from_bar = get_pose(bar_body)
        collision_fn = get_body_collision_fn(bar_body, built_obstacles)
        cr = collision_fn(world_from_bar)
        print(cr)
        # wait_for_user()

        offset_collision_fn = get_offset_collision_test(bar_body, built_obstacles)
        is_collide = offset_collision_fn(world_from_bar, diagnosis=True)
        print('is_collide: ', is_collide)

def get_body_collision_fn(body, obstacles=[]):
    # a copied collision fn to return contact information
    from itertools import product
    from pybullet_planning.interfaces.robots.collision import pairwise_collision, pairwise_link_collision, expand_links, \
        pairwise_link_collision_info
    from pybullet_planning.interfaces.robots.link import get_links, get_link_pose, get_link_name
    from pybullet_planning.interfaces.robots.body import set_pose, get_body_name
    from pybullet_planning.interfaces.debug_utils.debug_utils import draw_collision_diagnosis

    moving_bodies = [body]
    # * body pairs
    check_body_pairs = list(product(moving_bodies, obstacles))  # + list(combinations(moving_bodies, 2))
    check_body_link_pairs = []
    for body1, body2 in check_body_pairs:
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        bb_link_pairs = product(links1, links2)
        for bb_links in bb_link_pairs:
            bbll_pair = ((body1, bb_links[0]), (body2, bb_links[1]))
            # if bbll_pair not in disabled_collisions and bbll_pair[::-1] not in disabled_collisions:
            check_body_link_pairs.append(bbll_pair)

    def collision_fn(pose):
        set_pose(body, pose)
        # * body - body check
        for (body1, link1), (body2, link2) in check_body_link_pairs:
            if pairwise_link_collision(body1, link1, body2, link2):
                cr = pairwise_link_collision_info(body1, link1, body2, link2)
                return cr
        return []
    return collision_fn
