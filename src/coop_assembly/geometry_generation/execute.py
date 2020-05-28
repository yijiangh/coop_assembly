
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

from coop_assembly.data_structure import OverallStructure, BarStructure
from coop_assembly.geometry_generation.generate_tetrahedra import generate_first_triangle, generate_structure_from_points
from coop_assembly.help_functions.helpers_geometry import update_bar_lengths
from coop_assembly.help_functions.parsing import export_structure_data, parse_saved_structure_data
from coop_assembly.assembly_info_generation import calculate_gripping_plane, calculate_offset, contact_info_from_seq
from coop_assembly.help_functions.shared_const import HAS_PYBULLET, METER_SCALE

from coop_assembly.planning.visualization import set_camera, SHADOWS, BACKGROUND_COLOR, label_elements

from pybullet_planning import connect, wait_if_gui, dump_world, apply_alpha, draw_collision_diagnosis, pairwise_collision, \
    pairwise_collision_info, get_bodies, RED, TAN


def execute_from_points(points, tet_node_ids, radius, check_collision=False, correct=True, viewer=False, verbose=False, scale=1.0, write=False, \
        return_network=False, **kwargs):
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
    generate_structure_from_points(o_struct, bar_struct, radius, points, tet_node_ids,
        correct=correct, check_collision=check_collision, viewer=viewer, verbose=verbose)

    endpts_from_element = bar_struct.get_axis_pts_from_element(scale=scale)

    if write:
        export_structure_data(bar_struct.data, o_struct.data, **kwargs)

    connect(use_gui=viewer, shadows=SHADOWS, color=BACKGROUND_COLOR)
    element_bodies = bar_struct.get_element_bodies(color=apply_alpha(RED, 0))
    set_camera([attr['point_xyz'] for v, attr in o_struct.nodes(True)])

    handles = []
    handles.extend(label_elements(element_bodies))

    # * checking mutual collision between bars
    # TODO move this complete assembly collision sanity check to bar structure class
    contact_from_connectors = bar_struct.get_connectors(scale=1e-3)
    connectors = list(contact_from_connectors.keys())
    for bar1, bar2 in connectors:
        b1_body = bar_struct.get_bar_pb_body(bar1, apply_alpha(RED, 0.1))
        b2_body = bar_struct.get_bar_pb_body(bar2, apply_alpha(TAN, 0.1))
        assert len(get_bodies()) == len(element_bodies)

        if pairwise_collision(b1_body, b2_body):
            cr = pairwise_collision_info(b1_body, b2_body)
            draw_collision_diagnosis(cr, focus_camera=True)
            # if not viewer:
            assert False, '{}-{} collision!'.format(b1_body, b2_body)
        print('-'*10)

    print('No collision in connectors found.')
    wait_if_gui('Done.')

    # contact_from_connectors = bar_struct.get_connectors(scale=scale)
    # connectors = list(contact_from_connectors.keys())
    if return_network:
        return bar_struct, o_struct
        # return (bar_struct.data, o_struct.data)
    else:
        return endpts_from_element

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
    return a
