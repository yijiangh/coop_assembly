import os
import pytest

from coop_assembly.choreo import sequenced_picknplace_plan, visualize_picknplace_planning_results

@pytest.fixture
def pkg_name():
    return 'dms_ws_tet_bars'

@pytest.fixture
def dir_setup():
    here = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(here, 'test_data')
    result_dir = os.path.join(here, 'results')
    return test_data_dir, result_dir

@pytest.mark.choreo_call
# @pytest.mark.parametrize('solve_method', [('ladder_graph'), ('sparse_ladder_graph')])
@pytest.mark.parametrize('solve_method', [('sparse_ladder_graph')])
# @pytest.mark.parametrize('solve_method', [('ladder_graph')])
def test_choreo_seq_plan(viewer, solve_method, pkg_name, dir_setup):
    test_data_dir, result_dir = dir_setup
    assembly_json_path = os.path.join(test_data_dir, pkg_name, 'json', pkg_name + '.json')

    # TODO: o_struct.from_data(), replace assembly_json_path with that
    sequenced_picknplace_plan(assembly_json_path, solve_method=solve_method, scale=1e-3, viewer=viewer, viz_inspect=False, \
        save_dir=result_dir, sample_time=5, sparse_time_out=0, jt_res=0.1)

@pytest.mark.choreo_viz
def test_viz_choreo_planning_result(viewer, pkg_name, dir_setup):
    test_data_dir, result_dir = dir_setup
    assembly_json_path = os.path.join(test_data_dir, pkg_name, 'json', pkg_name + '.json')
    saved_file_path = os.path.join(result_dir, pkg_name + '_result_.json')
    visualize_picknplace_planning_results(assembly_json_path, saved_file_path, viewer=viewer)
