'''

    ****       *****       ******       ****       ******  ******          **           **
   **  **      **  **      **          **  **        **    **              **           **
   **          *****       ****        ******        **    ****            **   *****   *****
   **  **      **  **      **          **  **        **    **              **  **  **   **  **
    ****   **  **  **  **  ******  **  **  **  **    **    ******          **   ******  *****


created on 28.06.2019
author: stefanaparascho

edited on 17.12.2019 by Yijiang Huang, yijiangh@mit.edu
'''
import os
from collections import defaultdict

from compas.datastructures import Network
from compas.geometry import add_vectors, scale_vector, cross_vectors, subtract_vectors
from compas.geometry import distance_point_point, closest_point_on_plane
from compas.geometry import centroid_points

from coop_assembly.help_functions.helpers_geometry import dropped_perpendicular_points

class OverallStructure(Network):
    """this class defines the overall structure in which a node is represented by a network.vertex and a bar by a network.edge does not include connectors - these are referenced through the additional bar_structure class

    Note: this data structure is only used to connect design intent (vertices to cover) with the the actual bar system.
    We only use the `BarStructure` in the assembly planning pipeline.

    The Overall_Structure is the abstract Network structure in which bars are modelled as edges and nodes (points where multiple bars would
    ideally come together) as vertices. This does not include the geometric information about the bars' actual position or endpoints,
    but only an idealised point where tetrahedra edges would be located.

    Each bar has two representations:
        (1) bar connecting a pair of ideal vertices: ``bar = (ideal vertex 0, ideal vertex 1)``
        (2) bar as a vertex in the bar structure: ``b_v  = o_struct.edge[bars[0]][bars[1]]["vertex_bar"], b_s = b_struct.vertex[b_v]``

    Additionaly, the network uses an attribute `tetrahedra` to keep track of
    {tet_index : [vertex keys]}

    SP dissertation section 3.5.2:

        `OverallStructure` is implemented to control "high-level" features, such as
        aggregations of bars into three-bar-groups. This data structure does not
        depict the reciprocal connections but abstracts one node where multiple bars
        come together in one "vertex point". `OverallStructure` uses vertices to represent
        nodes and edges to describe the topological function of bars, i.e. which nodes
        are connected through a bar.
        ..., while one edge always connects two vertices. In the given geometric system, one bar may be connected to multiple other bars, whereas one welded joint can only bridge two bars.

    This means that we need a more zoomed-in version of vertices, to describe multiple connection details (welded points) under one single ideal vertex.

    .. image:: ../images/node_subnode_joint.png
        :scale: 80 %
        :align: center

    This model is referred as **abstract data model** (see left sub-figure below).

    .. image:: ../images/data_structures.png
        :scale: 80 %
        :align: center

    Parameters
    ----------
    Network : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    def __init__(self, struct_bar=None):
        super(OverallStructure, self).__init__()
        self.struct_bar = struct_bar
        self.name = "Network_o"
        # vertex attributes
        self.update_default_node_attributes({"name" : "network_o"})
        self.update_default_node_attributes({"velocity_vector": (0.0,0.0,0.0)})
        self.update_default_node_attributes({"velocities_all": []})
        self.update_default_node_attributes({"fixed": False})
        # edge attributes
        self.update_default_edge_attributes({"name" : "network_o"})
        self.tetrahedra = {}
        self.t_key_max = 0

    #####################################
    # backward compatibility

    @property
    def vertex(self):
        return self.node

    def vertices(self, data=False):
        return self.nodes(data)

    def vertex_connected_edges(self, v):
        return self.connected_edges(v)

    def vertex_neighbors(self, v):
        return self.neighbors(v)

    #####################################
    # adding data

    def add_node(self, xyz = (0.0, 0.0, 0.0), v_key=None, t_key=None):
        """add vertex point to a tetrahedra index

        Parameters
        ----------
        xyz : tuple, optional
            point, by default (0.0, 0.0, 0.0)
        v_key : int, optional
            vertex index, by default None, in which case an int will automatically
            generated from the Network
        t_key : int, optional
            tet index, by default None

        Returns
        -------
        int
            vertex index
        """
        # add the vertex point to the network vertices
        v_key = super(OverallStructure, self).add_node(v_key)

        # update the tetrahedra dict
        self.vertex[v_key].update({"x":xyz[0], "y":xyz[1], "z":xyz[2], "point_xyz":xyz})
        if t_key != None:
            if t_key in self.tetrahedra:
                if v_key not in self.tetrahedra[t_key]["vertices"]:
                    self.tetrahedra[t_key]["vertices"].append(v_key)
            else:
                # create a new dict entry
                self.tetrahedra.update({t_key : {"vertices":[v_key]}})
        else:
            # if no t_key is specified, append it to the list
            if self.t_key_max in self.tetrahedra:
                if v_key not in self.tetrahedra[self.t_key_max]["vertices"]:
                    self.tetrahedra[self.t_key_max]["vertices"].append(v_key)
            else:
                self.tetrahedra.update({self.t_key_max:{"vertices":[v_key]}})
            self.t_key_max += 1
        return v_key

    def add_bar(self, v_key_1, v_key_2, bar_struct_vert_key, t_key=None):
        """create a network edge between vert 1 and vert 2,
        and assign the bar structure to the edge's attribute

        Parameters
        ----------
        v_key_1 : int
            vertex id for vertex point 1
        v_key_2 : int
            vertex id for vertex point 1
        bar_struct_vert_key : int
            bar structure vertex id, representing a bar
        t_key : int, optional
            tetrahedra index, by default None

        Returns
        -------
        [type]
            [description]
        """
        e_keys = self.add_edge(v_key_1, v_key_2)
        # update, the connection between v1 and v2 is "realized" by
        # the bar struct_bar.vertex[bar_struct_vert_key]
        self.edge[v_key_1][v_key_2].update({"vertex_bar":bar_struct_vert_key})
        self.struct_bar.vertex[bar_struct_vert_key].update({"o_edge":(v_key_1, v_key_2)})

        if t_key != None:
            if t_key in self.tetrahedra:
                if v_key_1 not in self.tetrahedra[t_key]["vertices"]:
                    self.tetrahedra[t_key]["vertices"].append(v_key_1)
                if v_key_2 not in self.tetrahedra[t_key]["vertices"]:
                    self.tetrahedra[t_key]["vertices"].append(v_key_2)
            else:
                self.tetrahedra.update({t_key:{"vertices":[v_key_1, v_key_2]}})
        else:
            # no tetrahedra index is provided, bump max tet key
            if self.t_key_max in self.tetrahedra:
                if v_key_1 not in self.tetrahedra[self.t_key_max]["vertices"]:
                    self.tetrahedra[self.t_key_max]["vertices"].append(v_key_1)
                if v_key_2 not in self.tetrahedra[self.t_key_max]["vertices"]:
                    self.tetrahedra[self.t_key_max]["vertices"].append(v_key_2)
            else:
                self.tetrahedra.update({self.t_key_max:{"vertices":[v_key_1, v_key_2]}})
        return e_keys

    def calculate_point(self, v_key):
        cons = self.connectors(v_key)
        pts = []
        for con in cons:
            points_1 = self.struct_bar.vertex[con[0]]["axis_endpoints"]
            points_2 = self.struct_bar.vertex[con[1]]["axis_endpoints"]
            dpp = dropped_perpendicular_points(points_1[0], points_1[1], points_2[0], points_2[1])
            pts.append(centroid_points(dpp)) # contact point is the mid point of P_{i} and P_{Ci}
        # ? why take centroid again?
        contact_pt = centroid_points(pts)
        self.vertex[v_key].update({"x":contact_pt[0], "y":contact_pt[1], "z":contact_pt[2], "point_xyz":contact_pt})
        return contact_pt

    def get_bar_vertex_key(self, bar):
        """return vertex key of a given bar in the BarStructure

        Parameters
        ----------
        bar : list of two int
            the ideal vertex indices that the bar is connecting to

        Returns
        -------
        int
            vertex key representing the bar in BarStructure
        """
        return self.edge[bar[0]][bar[1]]["vertex_bar"]

    def connectors(self, v_key, verbose=False):
        """[summary]

        Parameters
        ----------
        v_key : int
            OverallStructure vertex index

        Returns
        -------
        e_common : a list of 2-tuples
           BarStructure edges representing the physical joints within the vertex
        """
        #self.vertex[n_key]
        o_edges = self.vertex_connected_edges(v_key)
        # b_vertices is a list of BarStructure vertex ids that connect OverallStructure vertex e[0] and e[1]
        b_vertices = [self.edge[e[0]][e[1]]["vertex_bar"] for e in o_edges]

        # Bar_Structure edges represents contact points
        b_edges = []
        for b_vert in b_vertices:
            b_edges.append(self.struct_bar.vertex_connected_edges(b_vert))

        # ? maybe this is the polygon part?
        common_e = []
        for i, e_1 in enumerate(b_edges):
            for j, e_2 in enumerate(b_edges):
                if j>i:
                    if verbose: print('e_1: {}, e_2: {}'.format(e_1, e_2))
                    for e_1_1 in e_1:
                        for e_2_1 in e_2:
                            if (e_1_1[0] == e_2_1[0] and e_1_1[1] == e_2_1[1]) \
                                or (e_1_1[0] == e_2_1[1] and e_1_1[1] == e_2_1[0]):
                                common_e.append(e_1_1)
        return common_e

    # def nodes_vic(self, n_key, dist):
    #     # find nodes in a certain distance to node n_key

    #     point_base = self.vertex_coordinates(n_key)
    #     points = [self.vertex_coordinates(v_key) for v_key, args in self.vertices_iter(True)]
    #     n_key_new = n_key
    #     nodes_used = []
    #     nodes_vic = []
    #     nodes_vic, nodes_used = self.check_distance(n_key_new, nodes_used, nodes_vic, dist, point_base, n_key)

    #     return nodes_vic

    def check_distance(self, n_key, nodes_used, nodes_vic, dist, point_base, n_base):
        # checks distance for vertices while going through the network structure via neighbours (recursively)

        bool_check = True
        for n in self.neighbors(n_key):
            bool_check_l = True
            p = self.vertex_coordinates(n)
            if n not in nodes_used and n != n_base:
                d = distance_point_point(point_base, p)
                if d < dist:
                    nodes_vic.append(n)
                else:
                    bool_check = False
                    bool_check_l = False
                nodes_used.append(n)
                if bool_check_l: self.check_distance(n, nodes_used, nodes_vic, dist, point_base, n_base)
            if not bool_check: return nodes_vic, nodes_used

        return nodes_vic, nodes_used
