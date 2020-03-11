import pytest
from fixtures.gen_from_points import points_library

def pytest_addoption(parser):
    parser.addoption('--viewer', action='store_true', help='Enables the pybullet viewer')
    parser.addoption('--write', action='store_true', help='Export results')
    parser.addoption('--collision', action='store_false', help='disable collision checking')
    parser.addoption('--stiffness', action='store_false', help='disable stiffness')
    parser.addoption('--motion', action='store_true', help='enable transit motion')

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
