import os
import pytest

from pybullet_planning import wait_for_user, connect, has_gui, wait_for_user

from coop_assembly.choreo.robot_setup import get_picknplace_robot_data
from coop_assembly.choreo.utils import load_world

@pytest.fixture
def pkg_name():
    return 'dms_ws_tet_bars'

@pytest.fixture
def dir_setup():
    here = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(here, 'test_data')
    result_dir = os.path.join(here, 'results')
    return test_data_dir, result_dir

def test_load_robot(viewer):
    robot_data, ws_data = get_picknplace_robot_data()
    robot_urdf, _, tool_link_name, ee_link_name, joint_names, _ = robot_data
    assert ee_link_name == 'eef_tcp_frame'
    assert tool_link_name == 'robot_tool0'
    connect(use_gui=viewer)
    load_world()
    if has_gui():
        wait_for_user()

@pytest.mark.choreo_wip
def test_grasp_gen_fn():
    pass
