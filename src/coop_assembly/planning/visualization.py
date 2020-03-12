import colorsys
import numpy as np
from pybullet_planning import RED, BLUE, GREEN, BLACK, add_line, set_color, apply_alpha, get_visual_data, \
    set_camera_pose, add_text, draw_pose, get_pose, wait_for_user, wait_for_duration
from coop_assembly.help_functions.shared_const import METER_SCALE

BAR_LINE_WIDTH = 1.0
CONNECTOR_LINE_WIDTH = 1.0

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

def label_element(element_bodies, element):
    element_body = element_bodies[element]
    return [
        add_text('b'+str(element), position=(0, 0, 0), parent=element_body),
        # add_text(element[1], position=(0, 0, +0.02), parent=element_body),
    ]


def label_elements(element_bodies):
    handles = []
    # +z points parallel to each element body
    for element, body in element_bodies.items():
        handles.extend(label_element(element_bodies, element))
        handles.extend(draw_pose(get_pose(body), length=0.02))
        # wait_for_user()
    return handles


def label_connector(connector_pts, c, **kwargs):
    return [add_text('c'+str(c), position=(np.array(connector_pts[c][0])+np.array(connector_pts[c][1]))/2, **kwargs)]


def label_points(points, **kwargs):
    return [add_text(node, position=point, **kwargs) for node, point in enumerate(points)]

###################################

def display_trajectories(trajectories, animate=True, time_step=0.02, video=False):
    # node_points, ground_nodes,
    if trajectories is None:
        return
    # set_extrusion_camera(node_points)
    # planned_elements = recover_sequence(trajectories)
    # colors = sample_colors(len(planned_elements))
    # if not animate:
    #     draw_ordered(planned_elements, node_points)
    #     wait_for_user()
    #     disconnect()
    #     return

    # video_saver = None
    # if video:
    #     handles = draw_model(planned_elements, node_points, ground_nodes) # Allows user to adjust the camera
    #     wait_for_user()
    #     remove_all_debug()
    #     wait_for_duration(0.1)
    #     video_saver = VideoSaver('video.mp4') # has_gui()
    #     time_step = 0.001
    # else:
    #     wait_for_user()

    #element_bodies = dict(zip(planned_elements, create_elements(node_points, planned_elements)))
    #for body in element_bodies.values():
    #    set_color(body, (1, 0, 0, 0))
    # connected_nodes = set(ground_nodes)
    printed_elements = []
    print('Trajectories:', len(trajectories))
    for i, trajectory in enumerate(trajectories):
        #wait_for_user()
        #set_color(element_bodies[element], (1, 0, 0, 1))
        last_point = None
        handles = []

        for _ in trajectory.iterate():
            # TODO: the robot body could be different
            # if isinstance(trajectory, PrintTrajectory):
            #     current_point = point_from_pose(trajectory.end_effector.get_tool_pose())
            #     if last_point is not None:
            #         # color = BLUE if is_ground(trajectory.element, ground_nodes) else RED
            #         color = colors[len(printed_elements)]
            #         handles.append(add_line(last_point, current_point, color=color, width=LINE_WIDTH))
            #     last_point = current_point
            if time_step is None:
                wait_for_user()
            else:
                wait_for_duration(time_step)

        # if isinstance(trajectory, PrintTrajectory):
        #     if not trajectory.path:
        #         color = colors[len(printed_elements)]
        #         handles.append(draw_element(node_points, trajectory.element, color=color))
        #         #wait_for_user()
        #     is_connected = (trajectory.n1 in connected_nodes) # and (trajectory.n2 in connected_nodes)
        #     print('{}) {:9} | Connected: {} | Ground: {} | Length: {}'.format(
        #         i, str(trajectory), is_connected, is_ground(trajectory.element, ground_nodes), len(trajectory.path)))
        #     if not is_connected:
        #         wait_for_user()
        #     connected_nodes.add(trajectory.n2)
        #     printed_elements.append(trajectory.element)

    # if video_saver is not None:
    #     video_saver.restore()
    wait_for_user()
