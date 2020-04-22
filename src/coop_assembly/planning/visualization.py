import colorsys
import numpy as np
from pybullet_planning import RED, BLUE, GREEN, BLACK, add_line, set_color, apply_alpha, get_visual_data, \
    set_camera_pose, add_text, draw_pose, get_pose, wait_for_user, wait_for_duration, get_name
from coop_assembly.help_functions.shared_const import METER_SCALE

BAR_LINE_WIDTH = 1.0
CONNECTOR_LINE_WIDTH = 1.0

GROUND_COLOR = 0.8*np.ones(3)
BACKGROUND_COLOR = [0.9, 0.9, 1.0] # 229, 229, 255
SHADOWS = True

###########################################

def draw_element(axis_endpts, element, color=RED):
    p1, p2 = axis_endpts[element]
    return add_line(p1, p2, color=color[:3], width=BAR_LINE_WIDTH)


def sample_colors(num, lower=0.0, upper=0.75): # for now wrap around
    return [colorsys.hsv_to_rgb(h, s=1, v=1) for h in reversed(np.linspace(lower, upper, num, endpoint=True))]


def draw_ordered(elements, axis_endpts):
    """colormap the sequence defined by the ``elements`` index list

    Purple elements are printed first, red elements are printed last
    (black elements have yet to be printed)

    Parameters
    ----------
    elements : list of int
        [description]
    axis_endpts : list of tuples
        [description]

    Returns
    -------
    pb debug handles
        [description]
    """
    colors = sample_colors(len(elements))
    handles = []
    for element, color in zip(elements, colors):
        handles.append(draw_element(axis_endpts, element, color=color))
    return handles


def color_structure(element_bodies, printed, next_element=None, built_alpha=1.0, remaining_alpha=0.2):
    """color element_bodies: printed ones to blue, next element to green, remaining ones to black

    Parameters
    ----------
    element_bodies : dict
        bar_v -> bar pb body
    printed : set
        set of printed elements' bar v key
    next_element : int
        [description], default to None
    built_alpha : float
        0-1., opacity value for built elements (printed + next_element), default to 1.0
    remaining_alpha : float
        0-1., opacity value for remaining elements, default to 0.2
    """
    element_colors = {}
    for element in printed:
        element_colors[element] = apply_alpha(BLUE, alpha=built_alpha)
    if next_element in element_bodies:
        element_colors[next_element] = apply_alpha(GREEN, alpha=built_alpha)
        remaining = set(element_bodies) - printed - {next_element}
    else:
        remaining = set(element_bodies) - printed
    for element in remaining:
        element_colors[element] = apply_alpha(BLACK, alpha=remaining_alpha)
    for element, color in element_colors.items():
        body = element_bodies[element]
        [shape] = get_visual_data(body)
        if color != shape.rgbaColor:
            set_color(body, color=color)

###########################################

def set_camera(node_points, camera_dir=[1, 1, 1], camera_dist=0.25):
    """pointing camera towards the centroid of a list of pts

    Parameters
    ----------
    node_points : [type]
        [description]
    camera_dir : list, optional
        [description], by default [1, 1, 1]
    camera_dist : float, optional
        [description], by default 0.25
    """
    centroid = np.average(node_points, axis=0) * METER_SCALE
    camera_offset = camera_dist * np.array(camera_dir)
    set_camera_pose(camera_point=centroid + camera_offset, target_point=centroid)

######################################################

def label_element(element_bodies, element, tag=None):
    element_body = element_bodies[element]
    tag = '-'+tag if tag is not None else ''
    return [
        add_text('b'+str(element)+tag, position=(0, 0, 0), parent=element_body),
        # add_text(element[1], position=(0, 0, +0.02), parent=element_body),
    ]


def label_elements(element_bodies, body_index=False):
    handles = []
    # +z points parallel to each element body
    for element, body in element_bodies.items():
        handles.extend(label_element(element_bodies, element, tag=get_name(body) if body_index else None))
        # handles.extend(draw_pose(get_pose(body), length=0.02))
        # wait_for_user()
    return handles


def label_connector(connector_pts, c, **kwargs):
    return [add_text('c'+str(c), position=(np.array(connector_pts[c][0])+np.array(connector_pts[c][1]))/2, **kwargs)]


def label_points(points, **kwargs):
    return [add_text(node, position=point, **kwargs) for node, point in enumerate(points)]

######################################################

import networkx as nx
import matplotlib.pyplot as plt

def visualize_collision_digraph(collision_facts):
    t_edges = {}
    p_edges = {}
    for fact in collision_facts:
        if fact[0] not in ['Collision', 'collision']:
            continue
        traj, e = fact[1].trajectories[0], fact[2]
        carried_element = traj.element
        if 'transit' in traj.tag:
            # transition traj
            t_edges[(e, carried_element)] = 1
        else:
            # place traj
            p_edges[(e, carried_element)] = 1
    # collision directional graph
    G = nx.DiGraph()
    for edge, weight in t_edges.items():
        G.add_edge(*edge, weight=weight)
    for edge, weight in p_edges.items():
        G.add_edge(*edge, weight=weight)

    # plotting
    plt.subplots()
    # nodal positions
    pos = nx.shell_layout(G)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=list(t_edges.keys()), width=3, edge_color='r')
    nx.draw_networkx_edges(
        G, pos, edgelist=list(p_edges.keys()), width=3, alpha=0.5, edge_color="b" #, style="dashed"
    )
    # labels
    nx.draw_networkx_labels(G, pos, font_size=20)

    plt.show()
