import colorsys
import numpy as np
from termcolor import cprint
from pybullet_planning import RED, BLUE, GREEN, BLACK, TAN, add_line, set_color, apply_alpha, get_visual_data, \
    set_camera_pose, add_text, draw_pose, get_pose, wait_for_user, wait_for_duration, get_name, wait_if_gui, remove_all_debug, remove_body, \
    remove_handles, pairwise_collision, pairwise_collision_info, draw_collision_diagnosis, has_gui

from coop_assembly.help_functions.shared_const import METER_SCALE
from coop_assembly.data_structure.utils import MotionTrajectory

BAR_LINE_WIDTH = 1.0
CONNECTOR_LINE_WIDTH = 1.0

GROUND_COLOR = 0.8*np.ones(3)
BACKGROUND_COLOR = [0.9, 0.9, 1.0] # 229, 229, 255
SHADOWS = True

###########################################

def draw_element(axis_endpts, element, color=RED, width=BAR_LINE_WIDTH):
    p1, p2 = axis_endpts[element]
    return add_line(p1, p2, color=color[:3], width=width)


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
    handles = []
    colors = sample_colors(len(elements))
    for element, color in zip(elements, colors):
        handles.append(draw_element(axis_endpts, element, color=color))
    return handles


def draw_partial_ordered(elements_from_layer, axis_endpts):
    handles = []
    colors = sample_colors(len(elements_from_layer.keys()))
    for layer, color in zip(elements_from_layer, colors):
        for e in list(elements_from_layer[layer]):
            e_id = e if isinstance(e, int) else e.index
            handles.append(draw_element(axis_endpts, e_id, color=color))
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
        if element not in element_bodies:
            cprint('element {} not in bodies, skipped'.format(element), 'yellow')
            continue
        if isinstance(element_bodies[element], int):
            body = element_bodies[element]
        else:
            body = element_bodies[element].body
        try:
            # TODO: might return nothing is pytest without -s ?
            [shape] = get_visual_data(body)
            if color != shape.rgbaColor:
                set_color(body, color=color)
        except:
            print('Color change failure.')
            pass

###########################################

def set_camera(node_points, camera_dir=[1, 1, 1], camera_dist=0.25, scale=METER_SCALE):
    """pointing camera towards the centroid of a list of pts
    Note that the node points are all assumed to be in millimeter scale, we do scaling inside this fn.

    Parameters
    ----------
    node_points : [type]
        [description]
    camera_dir : list, optional
        [description], by default [1, 1, 1]
    camera_dist : float, optional
        [description], by default 0.25
    """
    centroid = np.average(node_points, axis=0) * scale
    camera_offset = camera_dist * np.array(camera_dir)
    set_camera_pose(camera_point=centroid + camera_offset, target_point=centroid)

######################################################

def label_element(element_bodies, element, tag=None):
    if isinstance(element_bodies[element], int):
        element_body = element_bodies[element]
    else:
        element_body = element_bodies[element].body
    tag = '-'+tag if tag is not None else ''
    return [
        add_text('b'+str(element)+tag, position=(0, 0, 0), parent=element_body),
        # add_text(element[1], position=(0, 0, +0.02), parent=element_body),
    ]


def label_elements(element_bodies, indices=None, body_index=False):
    handles = []
    # +z points parallel to each element body
    keys = element_bodies.keys() if indices is None else indices
    for element in keys:
        if element not in element_bodies:
            cprint('element {} not in bodies, skipped'.format(element), 'yellow')
            continue
        body = element_bodies[element]
        handles.extend(label_element(element_bodies, element, tag=get_name(body) if body_index else None))
        # handles.extend(draw_pose(get_pose(body), length=0.02))
        # wait_for_user()
    return handles


def label_connector(connector_pts, c, **kwargs):
    return [add_text('c'+str(c), position=(np.array(connector_pts[c][0])+np.array(connector_pts[c][1]))/2, **kwargs)]


def label_points(points, **kwargs):
    return [add_text(node, position=point, **kwargs) for node, point in enumerate(points)]

#####################################################

def check_model(bar_struct, indices=None, debug=False):
    from coop_assembly.planning.utils import get_element_neighbors, get_connector_from_elements, check_connected, get_connected_structures
    elements = list(bar_struct.nodes()) if indices is None else indices

    element_bodies = bar_struct.get_element_bodies(indices=elements, color=apply_alpha(RED, 0.3))
    endpts_from_element = bar_struct.get_axis_pts_from_element(scale=1e-3)
    set_camera([np.array(p[0]) for e, p in endpts_from_element.items()],scale=1.)

    contact_from_connectors = bar_struct.get_connectors(scale=1e-3)
    connectors = list(contact_from_connectors.keys())

    # * grounded elements
    cprint('Visualize grounded elements.', 'yellow')
    grounded_elements = list(set(bar_struct.get_grounded_bar_keys()) & set(elements))
    remove_all_debug()
    for bar in grounded_elements:
        label_elements(element_bodies, [bar])
    color_structure(element_bodies, set(grounded_elements), next_element=None, built_alpha=0.6)
    if debug:
        wait_if_gui('grounded element: {}'.format(grounded_elements))

    # * connectors from bar
    cprint('Visualize connectors.', 'yellow')
    connector_from_elements = get_connector_from_elements(connectors, elements)
    for bar in elements:
        bar_connectors = connector_from_elements[bar]
        current_connectors = []
        remove_all_debug()
        for c in list(bar_connectors):
            if c[0] in elements and c[1] in elements:
                current_connectors.append(c)
                label_elements(element_bodies, c)
                add_line(*contact_from_connectors[c], color=(1,0,0,1), width=2)
        color_structure(element_bodies, set(), next_element=bar, built_alpha=0.6)
        if debug:
            wait_if_gui('connector: {}'.format(current_connectors))

    # * neighbor elements from elements
    print('Visualize neighnor elements.')
    element_neighbors = get_element_neighbors(connectors, elements)
    for element, connected_bars in element_neighbors.items():
        remove_all_debug()
        color_structure(element_bodies, connected_bars, element, built_alpha=0.6)
        label_elements(element_bodies, list(connected_bars) + [element])
        if debug:
            wait_if_gui('connected neighbors: {} | {}'.format(element, connected_bars))

    # TODO: some sanity check here
    # mutual collision checks
    is_collided = False
    for bar1, bar2 in connectors:
        if not (bar1 in elements and bar2 in elements):
            continue
        b1_body = bar_struct.get_bar_pb_body(bar1, apply_alpha(RED, 0.5))
        b2_body = bar_struct.get_bar_pb_body(bar2, apply_alpha(TAN, 0.5))
        if b1_body is None or b2_body is None:
            continue
        if pairwise_collision(b1_body, b2_body):
            cr = pairwise_collision_info(b1_body, b2_body)
            draw_collision_diagnosis(cr, focus_camera=True)
            is_collided = True
            if not has_gui():
                assert False, '{}-{} collision!'.format(b1_body, b2_body)
    cprint('Model valid: {}'.format(not is_collided), 'red' if is_collided else 'green')
    wait_if_gui('model checking finished.')


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
    print('Collision constraint graph: (i -> j) means assembling j collides with i.')
    print('A feasible solution should not have any cycle.')
    plt.subplots()
    # nodal positions
    pos = nx.shell_layout(G)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=list(t_edges.keys()), width=3, edge_color='y')
    nx.draw_networkx_edges(
        G, pos, edgelist=list(p_edges.keys()), width=3, alpha=0.5, edge_color="b" #, style="dashed"
    )
    # labels
    nx.draw_networkx_labels(G, pos, font_size=20)
    # identify cycles
    try:
        cycles = nx.find_cycle(G, orientation='original')
        print('cycle: {}'.format(cycles))
        nx.draw_networkx_edges(G, pos, edgelist=[c[0:2] for c in cycles], width=3, edge_color='r')
    except:
        pass

    plt.title('Collision constraint graph')
    plt.show()

###################################

def display_trajectories(trajectories, time_step=0.02, video=False, animate=True, element_from_index=None):
    """[summary]

    Parameters
    ----------
    trajectories : [type]
        [description]
    time_step : float, optional
        [description], by default 0.02
    video : bool, optional
        [description], by default False
    animate : bool, optional
        if set to False, display sequence colormap only, skip trajectory animation, by default True
    """
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

    video_saver = None
    if video:
        # handles = draw_model(planned_elements, node_points, ground_nodes) # Allows user to adjust the camera
        wait_if_gui()
        # remove_all_debug()
        # wait_for_duration(0.1)
        # video_saver = VideoSaver('video.mp4') # has_gui()
        # time_step = 0.001
    else:
        wait_if_gui('Ready to simulate trajectories.')

    remove_all_debug()
    #element_bodies = dict(zip(planned_elements, create_elements(node_points, planned_elements)))
    #for body in element_bodies.values():
    #    set_color(body, (1, 0, 0, 0))
    # connected_nodes = set(ground_nodes)
    # TODO: resolution depends on bar distance to convex hull of obstacles
    # TODO: fine resolution still results in collision?
    printed_elements = []
    print('Trajectories:', len(trajectories))
    for i, trajectory in enumerate(trajectories):
        #wait_for_user()
        #set_color(element_bodies[element], (1, 0, 0, 1))
        last_point = None
        handles = []

        bounding = None
        if printed_elements and 'tran' in trajectory.tag and element_from_index is not None:
            node_points = []
            for e in printed_elements:
                node_points.extend(element_from_index[e].axis_endpoints)
            from coop_assembly.planning.motion import create_bounding_mesh
            bounding = create_bounding_mesh(bodies=None, node_points=node_points,
                                            buffer=0.1)

        if isinstance(trajectory, MotionTrajectory):
            for attach in trajectory.attachments:
                set_color(attach.child, GREEN)

        for conf in trajectory.iterate():
            # TODO: the robot body could be different
            if time_step is None:
                wait_for_user('step sim.') #'{}'.format(conf))
            else:
                wait_for_duration(time_step)

        if isinstance(trajectory, MotionTrajectory):
            for attach in trajectory.attachments:
                set_color(attach.child, BLUE)
            is_connected = True
            print('{}) {:9} | Connected: {} | Ground: {} | Length: {}'.format(
                i, str(trajectory), is_connected, True, len(trajectory.path)))
                # is_ground(trajectory.element, ground_nodes)
        #     if not is_connected:
        #         wait_for_user()
        #     connected_nodes.add(trajectory.n2)

        if bounding is not None:
            remove_body(bounding)
        if 'retreat' in trajectory.tag:
            printed_elements.append(trajectory.element)

    # if video_saver is not None:
    #     video_saver.restore()
    wait_if_gui('Simulation finished.')
