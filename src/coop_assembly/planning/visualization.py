import colorsys
import numpy as np
from pybullet_planning import RED, BLUE, GREEN, add_line, set_color, apply_alpha, get_visual_data

LINE_WIDTH = 1.0

###########################################

def draw_element(node_points, element, color=RED):
    n1, n2 = element
    p1 = node_points[n1]
    p2 = node_points[n2]
    return add_line(p1, p2, color=color[:3], width=LINE_WIDTH)


def sample_colors(num, lower=0.0, upper=0.75): # for now wrap around
    return [colorsys.hsv_to_rgb(h, s=1, v=1) for h in reversed(np.linspace(lower, upper, num, endpoint=True))]


def draw_ordered(elements, node_points):
    colors = sample_colors(len(elements))
    handles = []
    for element, color in zip(elements, colors):
        handles.append(draw_element(node_points, element, color=color))
    return handles

def color_structure(element_bodies, printed, next_element):
    """color element_bodies: printed ones to blue, next element to green, remaining ones to red

    Parameters
    ----------
    element_bodies : dict
        bar_v -> bar pb body
    printed : set
        set of printed elements' bar v key
    next_element : int
        [description]
    """
    element_colors = {}
    for element in printed:
        element_colors[element] = apply_alpha(BLUE, alpha=1)
    element_colors[next_element] = apply_alpha(GREEN, alpha=1)
    remaining = set(element_bodies) - printed - {next_element}
    for element in remaining:
        element_colors[element] = apply_alpha(RED, alpha=0.5)
    for element, color in element_colors.items():
        body = element_bodies[element]
        [shape] = get_visual_data(body)
        if color != shape.rgbaColor:
            set_color(body, color=color)
