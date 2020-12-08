import numpy as np
from collections import defaultdict, deque

##################################################

def reverse_element(element):
    return element[::-1]

def is_reversed(all_elements, element):
    assert (element in all_elements) != (reverse_element(element) in all_elements)
    return element not in all_elements

def get_undirected(all_elements, directed):
    is_reverse = is_reversed(all_elements, directed)
    assert (directed in all_elements) != is_reverse
    return reverse_element(directed) if is_reverse else directed

def get_other_node(node1, element):
    assert node1 in element
    return element[node1 == element[0]]

def get_directions(element):
    return {element, reverse_element(element)}

def compute_printed_nodes(ground_nodes, printed):
    return nodes_from_elements(printed) | set(ground_nodes)

def compute_printable(all_elements, ground_nodes, printed):
    nodes = compute_printed_nodes(ground_nodes, printed)
    for element in set(all_elements) - printed:
        # for directed in get_directions(element):
        node1, node2 = element
        if node1 in nodes or node2 in nodes:
            yield element

##################################################

def get_midpoint(node_points, element):
    return np.average([node_points[n] for n in element], axis=0)

def compute_z_distance(node_points, element):
    # Distance to a ground plane
    # Opposing gravitational force
    return get_midpoint(node_points, element)[2]

##################################################

def get_node_neighbors(elements):
    node_neighbors = defaultdict(set)
    for e in elements:
        n1, n2 = e
        node_neighbors[n1].add(e)
        node_neighbors[n2].add(e)
    return node_neighbors

def nodes_from_elements(elements):
    # TODO: always include ground nodes
    return {n for e in elements for n in e}

def get_element_neighbors(elements):
    node_neighbors = get_node_neighbors(elements)
    element_neighbors = defaultdict(set)
    for e in elements:
        n1, n2 = e
        element_neighbors[e].update(node_neighbors[n1])
        element_neighbors[e].update(node_neighbors[n2])
        element_neighbors[e].remove(e)
    return element_neighbors

def check_connected(ground_nodes, printed_elements):
    # TODO: could merge with Caelan's connected components algorithm
    if not printed_elements:
        return True
    node_neighbors = get_node_neighbors(printed_elements)
    queue = deque(ground_nodes)
    visited_nodes = set(ground_nodes)
    visited_elements = set()
    while queue:
        node1 = queue.popleft()
        for element in node_neighbors[node1]:
            visited_elements.add(element)
            node2 = get_other_node(node1, list(element))
            if node2 not in visited_nodes:
                queue.append(node2)
                visited_nodes.add(node2)
    return printed_elements <= visited_elements

#####################################################
# copy from pddlstream.utils

def incoming_from_edges(edges):
    incoming_vertices = defaultdict(set)
    for v1, v2 in edges:
        incoming_vertices[v2].add(v1)
    return incoming_vertices

def outgoing_from_edges(edges):
    outgoing_vertices = defaultdict(set)
    for v1, v2 in edges:
        outgoing_vertices[v1].add(v2)
    return outgoing_vertices
