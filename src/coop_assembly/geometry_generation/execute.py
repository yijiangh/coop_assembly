
'''

    ****       *****       ******       ****      ******  ******          **           **
   **  **      **  **      **          **  **       **    **              **           **
   **          *****       ****        ******       **    ****            **   *****   *****
   **  **      **  **      **          **  **       **    **              **  **  **   **  **
    ****   **  **  **  **  ******  **  **  **  **   **    ******          **   ******  *****


created on 28.06.2019
author: stefanaparascho
'''

import pickle
import time
from termcolor import cprint
from itertools import combinations

from coop_assembly.data_structure import OverallStructure, BarStructure
from coop_assembly.geometry_generation.generate_tetrahedra import generate_first_triangle, generate_structure_from_points
from coop_assembly.help_functions.helpers_geometry import update_bar_lengths
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.assembly_info_generation import calculate_gripping_plane, calculate_offset, contact_info_from_seq
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE

from coop_assembly.planning.visualization import set_camera, SHADOWS, BACKGROUND_COLOR, label_elements

from pybullet_planning import connect, wait_if_gui, dump_world, apply_alpha, draw_collision_diagnosis, pairwise_collision, \
    pairwise_collision_info, get_bodies, RED, TAN, get_distance


def execute_from_points(points, tet_node_ids, radius, check_collision=True, correct=True, viewer=False, verbose=False, scale=1.0, write=False, \
        return_network=False, allowable_bar_collision_depth=1e-3, verify=False, grounded_nodes=None, grounded_limit_distance=40, **kwargs):
    """Main entry point for the design system, for direct, xfunc or rpc call

    Parameters
    ----------
    points : list of float lists
        [[x,y,z], ...]
    tet_node_ids : list
        [[(base triangle vertex ids), new vertex id], ...]
    radius : float
        rod radius in millimeter
    check_col : bool, optional
        [description], by default False
    correct : bool, optional
        [description], by default True
    viewer : bool, optional
        enable pybullet viewer if True, by default True

    Returns
    -------
    (Overall_Structure.data, Bar_Structure.data)
        Serialized version of the overall structure and bar structure
    """
    bar_struct = BarStructure()
    o_struct = OverallStructure(bar_struct)
    try:
        generate_structure_from_points(o_struct, bar_struct, radius, points, tet_node_ids,
            grounded_nodes=grounded_nodes, grounded_limit_distance=grounded_limit_distance,
            correct=correct, check_collision=check_collision, viewer=viewer, verbose=verbose)
    except RuntimeError as err_msg:
        cprint('No solution found!', 'red')
        cprint(err_msg, 'yellow')
        return

    endpts_from_element = bar_struct.get_axis_pts_from_element(scale=scale)
    cprint('Solution found!', 'green')

    if write:
        export_structure_data(bar_struct.to_data(), o_struct.to_data(), **kwargs)

    if verify:
        connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
        element_bodies = bar_struct.get_element_bodies(color=apply_alpha(RED, 0))
        set_camera([attr['point_xyz'] for v, attr in o_struct.nodes(True)])

        handles = []
        handles.extend(label_elements(element_bodies))

        # * checking mutual collision between bars
        # TODO move this complete assembly collision sanity check to bar structure class
        contact_from_connectors = bar_struct.get_connectors(scale=1e-3)
        connectors = list(contact_from_connectors.keys())
        check_cnt = 0
        # for bar1, bar2 in connectors:
        for bar1, bar2 in combinations(list(bar_struct.node.keys()), 2):
            if bar1 == -1 or bar2 == -1 or bar1 == bar2:
                continue
            b1_body = bar_struct.get_bar_pb_body(bar1, apply_alpha(RED, 0.1))
            b2_body = bar_struct.get_bar_pb_body(bar2, apply_alpha(TAN, 0.1))
            # assert len(get_bodies()) == len(element_bodies)

            if pairwise_collision(b1_body, b2_body):
                cr, = pairwise_collision_info(b1_body, b2_body)
                penetration_depth = get_distance(cr[5], cr[6])
                cprint('b#{} - b#{}, depth {}'.format(bar1, bar2, penetration_depth), 'red')
                if penetration_depth > allowable_bar_collision_depth:
                    assert False, 'Bar {}-{} collision! penetration distance {}'.format(b1_body, b2_body, penetration_depth)
                    # pass
            check_cnt += 1

        cprint('No collision in connectors found. ({} pairs checked)'.format(check_cnt), 'green')
        wait_if_gui('Done.')
    else:
        cprint('No pybullet collision checking performed.', 'yellow')

    bar_struct.generate_grounded_connection()
    grounded_elements = bar_struct.get_grounded_bar_keys()
    contact_from_connectors = bar_struct.get_connectors(scale=scale)

    if return_network:
        return bar_struct, o_struct
        # return (bar_struct.data, o_struct.data)
    else:
        # rpc can not transmit tuple-indexed dict and frozenset
        connector_data = [{'connection': list(c), 'endpoints' : [list(pt) for pt in pts]} for c, pts in contact_from_connectors.items()]
        return endpts_from_element, list(grounded_elements), connector_data
        # return contact_from_connectors
        # return res

def test_connect(viewer=False):
    """Just checking if we can sprawn the pybullet GUI.

    Parameters
    ----------
    viewer : bool, optional
        [description], by default False
    """
    connect(use_gui=viewer)
    dump_world()
    wait_if_gui()
    a = 10
    print('Hey, welcome to pybullet!')
    return a
