
'''

    ****       *****       ******       ****       ******  ******          **           **
   **  **      **  **      **          **  **        **    **              **           **
   **          *****       ****        ******        **    ****            **   *****   *****
   **  **      **  **      **          **  **        **    **              **  **  **   **  **
    ****   **  **  **  **  ******  **  **  **  **    **    ******          **   ******  *****


created on 28.06.2019
author: stefanaparascho
'''

from compas.utilities.colors import color_to_colordict
from compas.geometry import add_vectors, scale_vector

from coop_assembly.help_functions.helpers_geometry import dropped_perpendicular_points

def draw_bar_structure_gh(bar_struct):
    """draw bar structure network in Grasshopper

    Parameters
    ----------
    bar_struct : BarStructure

    Returns
    -------
    [type]
        [description]
    """
    from compas_ghpython.utilities import draw_points, draw_lines

    end_pts_from_v = {}
    bar_axis_lines_from_v = {}
    for key in bar_struct.nodes():
        end_pts_from_v[key] = draw_points([{"pos": bar_struct.vertex[key]["axis_endpoints"][0]},
                                           {"pos": bar_struct.vertex[key]["axis_endpoints"][1]}])
        bar_axis_lines_from_v[key] = draw_lines([{"start": bar_struct.vertex[key]["axis_endpoints"][0],
                                                  "end": bar_struct.vertex[key]["axis_endpoints"][1]}])

    connector_lines_from_edge = {}
    for u, v, attr in bar_struct.edges(True):
        dpp = list(attr["endpoints"].values())[0]
        if dpp != []:
            connector_lines_from_edge[frozenset([u,v])] = \
                draw_lines([{"start": dpp[0], "end": dpp[1]}])

    return end_pts_from_v, bar_axis_lines_from_v, connector_lines_from_edge

def get_o_edge_from_bar_vertex_key(o_struct):
    ideal_v_from_bv = {}
    for u, v, attr in o_struct.edges(True):
        ideal_v_from_bv[attr['vertex_bar']] = [u, v]
    return ideal_v_from_bv

