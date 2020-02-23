import os
import pytest

from pybullet_planning import wait_for_user, connect, has_gui, wait_for_user, LockRenderer

from coop_assembly.data_structure import BarStructure, OverallStructure
from coop_assembly.assembly_info_generation.offset_motion import get_offset_collision_test
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE

from coop_assembly.planning import get_picknplace_robot_data
from coop_assembly.planning import load_world, set_camera
from coop_assembly.planning import color_structure

@pytest.fixture
def test_file_name():
    # @pytest.mark.parametrize('test_file_name', [('YJ_12_bars_point2triangle.json'),])
    return 'YJ_12_bars_point2triangle.json'

# @pytest.fixture
# def pkg_name():
#     return 'dms_ws_tet_bars'

# @pytest.fixture
# def dir_setup():
#     here = os.path.dirname(os.path.abspath(__file__))
#     test_data_dir = os.path.join(here, 'test_data')
#     result_dir = os.path.join(here, 'results')
#     return test_data_dir, result_dir

def load_structure(test_file_name, viewer):
    connect(use_gui=viewer)
    with LockRenderer():
        load_world()

        here = os.path.dirname(os.path.abspath(__file__))
        test_data_dir = os.path.join(here, 'test_data')

        b_struct_data, o_struct_data, _ = parse_saved_structure_data(os.path.join(test_data_dir, test_file_name))
        o_struct = OverallStructure.from_data(o_struct_data)
        b_struct = BarStructure.from_data(b_struct_data)
        b_struct.create_pb_bodies()
        o_struct.struct_bar = b_struct # TODO: better way to do this
        set_camera([attr['point_xyz'] for v, attr in o_struct.vertices(True)])
    return b_struct, o_struct

def test_load_robot(viewer):
    robot_data, ws_data = get_picknplace_robot_data()
    robot_urdf, _, tool_link_name, ee_link_name, joint_names, _ = robot_data
    assert ee_link_name == 'eef_tcp_frame'
    assert tool_link_name == 'robot_tool0'
    connect(use_gui=viewer)
    load_world()
    if has_gui():
        wait_for_user()

# @pytest.mark.choreo_wip
def test_grasp_gen_fn(viewer, test_file_name):

    for bar_vkey in b_struct.vertex:
        print('-------------')
        print('Bar #{}'.format(bar_vkey))

        assembled_bv = list(range(bar_vkey))
        bar_vertex = b_struct.vertex[bar_vkey]
        bar_body = bar_vertex['pb_body']
        built_obstacles = [floor] + [b_struct.vertex[bv]['pb_body'] for bv in assembled_bv]

        # contact_vec_1, contact_vec_2, contact_pt_1, contact_pt_2 = contact_info_from_seq(o_struct, b_struct, bar_vkey, assembled_bv, verbose=True)

        offset_collision_fn = get_offset_collision_test(bar_body, built_obstacles)
        is_collide = offset_collision_fn(world_from_bar, diagnosis=True)
        print('is_collide: ', is_collide)


@pytest.mark.choreo_wip
def test_draw_subset(viewer, test_file_name):
    bar_struct, _ = load_structure(test_file_name, viewer)

    element_bodies = bar_struct.get_element_bodies()
    printed = set([0,1,2,3])
    color_structure(element_bodies, printed, 4)

    wait_for_user()
