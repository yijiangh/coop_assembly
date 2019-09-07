import pytest
from numpy.testing import assert_almost_equal, assert_equal

from compas.datastructures import Network
from coop_assembly.data_structure import Overall_Structure, Bar_Structure
from coop_assembly.geometry_generation import execute_from_points, main_gh_simple

def test_generate_from_points():
    points = [(866.02540378443905, 500.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1000.0, 0.0), (288.67513720801298, 500.0, 818.08450024563103), \
        (769.63370557686096, 1333.0446812766299, 544.74326601771895), (-160.70791990805799, 1388.88716089995, 907.16026675621299), \
        (117.857296010868, 1981.4332041832699, 151.32258105017999), (-798.20078255574094, 1580.5022372901301, 160.91197525036301), \
        (-560.70093861717601, 2300.5484023076901, 812.92987741903903), (-666.65502975497805, 2519.4354019355501, -176.536520678389), \
        (58.742324089034398, 2934.6909001409299, 527.67951271342201)]

    dict_nodes = {'5': [4, 2, 3], '4': [3, 0, 2], '7': [6, 5, 2], \
                  '6': [2, 4, 5], '10': [8, 6, 9], '3': [0, 2, 1], '9': [6, 7, 8], '8': [5, 7, 6]}
    supports_bars = [(0,1), (1,2), (2,0)]
    supports_nodes = [0, 1, 2, 4]
    load_bars = [(4,5)]
    load = (0, 0, -2000)
    radius = 10.0

    b_struct_data, o_struct_data = execute_from_points(
        points, dict_nodes, radius, support_nodes=supports_nodes, 
        support_bars=supports_bars, load_bars=load_bars, load=load)

    # workaround for "reconstructing" classes in GH 
    b_struct = Network.from_data(b_struct_data)
    o_struct = Network.from_data(o_struct_data)


@pytest.mark.proxy_compare
def test_compare_xfunc_rpc():
    points = [(150.01005777432357, -0.10444999026289396, 0.0), (71.876490367019642, 132.50681323244007, 0.0), (0.010057774323581724, -0.10444999026289396, 0.0), (64.100272158117349, 48.497333880525062, 165.82349611141939)]
    dict_nodes = {'3': [2, 1, 0]}
    radius = 10.0

    xfunc_b_data, xfunc_o_data = main_gh_simple(points, dict_nodes, radius, use_xfunc=True)
    rpc_b_data, rpc_o_data = main_gh_simple(points, dict_nodes, radius, use_xfunc=False)

    # print('xfunc\n')
    # for vkey, v in xfunc_o_data['vertex'].items():
    #     print(v['x'], v['y'], v['z'])
    # print('rpc\n')
    # for vkey, v in rpc_o_data['vertex'].items():
    #     print(v['x'], v['y'], v['z'])

    assert_equal(xfunc_b_data, rpc_b_data)
    assert_equal(xfunc_o_data, rpc_o_data)    