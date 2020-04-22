import pytest
from fixtures.gen_from_points import points_library

def pytest_addoption(parser):
    parser.addoption('--viewer', action='store_true', help='Enables the pybullet viewer')
    parser.addoption('--write', action='store_true', help='Export results')
    parser.addoption('--collision', action='store_false', help='disable collision checking')
    parser.addoption('--bar_only', action='store_true', help='only planning motion for the bars')
    parser.addoption('--stiffness', action='store_false', help='disable stiffness')
    parser.addoption('--motion', action='store_true', help='enable transit motion')
    parser.addoption('--watch', action='store_true', help='watch trajectories')
    parser.addoption('--revisit', action='store_true')
    parser.addoption('--problem', default='single_tet')
    parser.addoption('--rfn', help='result file name')
    parser.addoption('--n_trails', default=1)
    parser.addoption('--alg', default='incremental')
    parser.addoption('--debug_mode', action='store_true', help='debug verbose mode')

@pytest.fixture
def viewer(request):
    return request.config.getoption("--viewer")

@pytest.fixture
def write(request):
    return request.config.getoption("--write")

@pytest.fixture
def collision(request):
    return request.config.getoption("--collision")

@pytest.fixture
def bar_only(request):
    return request.config.getoption("--bar_only")

@pytest.fixture
def stiffness(request):
    return request.config.getoption("--stiffness")

@pytest.fixture
def motion(request):
    return request.config.getoption("--motion")

@pytest.fixture
def watch(request):
    return request.config.getoption("--watch")

@pytest.fixture
def revisit(request):
    return request.config.getoption("--revisit")

@pytest.fixture
def file_spec(request):
    return request.config.getoption("--problem")

@pytest.fixture
def n_trails(request):
    return request.config.getoption("--n_trails")

@pytest.fixture
def result_file_spec(request):
    return request.config.getoption("--rfn")

@pytest.fixture
def algorithm(request):
    return request.config.getoption("--alg")

@pytest.fixture
def debug_mode(request):
    return request.config.getoption("--debug_mode")
