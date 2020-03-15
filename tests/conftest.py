import pytest
from fixtures.gen_from_points import points_library

def pytest_addoption(parser):
    parser.addoption('--viewer', action='store_true', help='Enables the pybullet viewer')
    parser.addoption('--write', action='store_true', help='Export results')
    parser.addoption('--collision', action='store_false', help='disable collision checking')
    parser.addoption('--stiffness', action='store_false', help='disable stiffness')
    parser.addoption('--motion', action='store_true', help='enable transit motion')
    parser.addoption('--animate', action='store_false', help='animate trajectories')
    parser.addoption('--revisit', action='store_true')
    parser.addoption('--fn', default='single_tet')
    parser.addoption('--n_trails', default=10)

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
def stiffness(request):
    return request.config.getoption("--stiffness")

@pytest.fixture
def motion(request):
    return request.config.getoption("--motion")

@pytest.fixture
def animate(request):
    return request.config.getoption("--animate")

@pytest.fixture
def revisit(request):
    return request.config.getoption("--revisit")

@pytest.fixture
def file_spec(request):
    return request.config.getoption("--fn")

@pytest.fixture
def n_trails(request):
    return request.config.getoption("--n_trails")

#TODO: test a main function with argparse
# https://stackoverflow.com/questions/18160078/how-do-you-write-tests-for-the-argparse-portion-of-a-python-module
