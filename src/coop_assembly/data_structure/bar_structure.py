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

import numpy as np
from collections import defaultdict
from compas.datastructures.network import Network
from compas.geometry import is_point_on_line
from compas.geometry import scale_vector
from coop_assembly.help_functions.helpers_geometry import dropped_perpendicular_points, find_points_extreme, \
    compute_contact_line_between_bars, create_bar_body, create_bar_flying_body
from coop_assembly.help_functions.shared_const import TOL, METER_SCALE

from pybullet_planning import create_plane, set_point, Point, get_pose, apply_alpha, RED, set_color, has_body, dump_world, get_bodies, \
    is_connected, remove_body

from .utils import Element, WorldPose

class BarStructure(Network):
    """This class encloses all the data that an assembly planner needs to know about the assembly. Each element
    is modeled as a graph node and edge models contact connection.

    SP:

        The Bar_Structure is some sort of an "inverted" network. It contains bars as nodes and the connections between bars as edges,
        these include the geometric information of each bar (endpoints) and their connecting points. However, this does not include information about which bars form a tetrahedron or which
        bars come together within a larger node, they only have information about where two bars are connected to one another.

    SP dissertation section 3.5.2:

        One bar may be connected to multiple other bars, whereas one welded joint can only bridge two bars.
        The nodes describe the bars, each of which can have multiple joints.
        The edges describe the joints between pairs of bars.
        `BarStructure` includes geometric information about the bars endpoints and the joint positions in the
        space.

    .. image:: ../images/node_subnode_joint.png
        :scale: 80 %
        :align: center

    This model is referred as **base data model**.

    .. image:: ../images/data_structures.png
        :scale: 80 %
        :align: center

    TODO: this data structure should be able to be derived from base class "VirtualJoint"

    Parameters
    ----------
    Network : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    # crosec_type     = "rectangle" / "tube" / "circle"
    # crosec_values = "rectangle" : (width, height) - height = dimension in z-axis direction
    #                                                            "tube"   : (outer diameter, thickness)
    #                                                            "circle"    : (diameter)
    def __init__(self, built_plate_z=0.0):
        super(BarStructure, self).__init__()
        self.support_point_max_key   = 0
        self.__load_point_max_key    = 0
        self.__connector_max_key     = 0
        self.name = "Network_b"
        # self._ground_key = self.add_ground(built_plate_z=built_plate_z)

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

    def add_bar(self, _bar_type, _axis_endpoints, _crosec_type, _crosec_values, _zdir, _bar_parameters=[], radius=3.17, grounded=False, pb_scale=METER_SCALE):
        v_key = self.add_node()
        bar_body = create_bar_body(_axis_endpoints, radius, scale=pb_scale)
        # goal_pose = get_pose(bar_body)
        self.node[v_key].update({"bar_type":_bar_type,
                                 "axis_endpoints":_axis_endpoints,
                                 "index_sol":None,    # tangent plane config (one out of four config)
                                 "mean_point":None,   # mean point used for local axis construction (SP uses this for gripping plane computation)
                                 "pb_body":bar_body,  # pybullet body
                                #  "goal_pose":goal_pose,
                                 'radius':radius,
                                 "grounded":grounded,
                                 "crosec_type":_crosec_type,
                                 "crosec_values":_crosec_values,
                                 "zdir":_zdir,
                                 "bar_parameters":_bar_parameters,
                                 "exchange_values":{},
                                 "layer":None, # tet group ids to indicate assembly partial ordering
                                  })
        return v_key
        # TODO: bisect search for local disassembly motion

    def create_pb_bodies(self, color=apply_alpha(RED, 0)):
        """create pybullet bodies for all elements, useful when the BarStructure is reconstructed from json data

        Parameters
        ----------
        color : tuple, optional
            [description], by default (1,1,1,0)
        """
        for v in self.nodes():
            self.get_bar_pb_body(v, color=color)

    def connect_bars(self, v_key1, v_key2, _endpoints=[], _connection_type=0, _connection_parameters=[], grounded=None):
        """create an edge connecting bar v_key1 and v_key2 or update edge attributes if edge exists already

        Parameters
        ----------
        v_key1 : int
            bar node id 1
        v_key2 : int
            bar node id 2
        _endpoints : list, optional
            [description], by default []
        _connection_type : int, optional
            [description], by default 0
        _connection_parameters : list, optional
            [description], by default []

        Returns
        -------
        [type]
            [description]
        """
        # !note that compas network edge is directional, thus (v_key2, v_key1) is not detected
        if self.has_edge(v_key1, v_key2):
            # edge exists already, updating edge attributes
            id = self.edge[v_key1][v_key2]["connections_count"]
            self.edge[v_key1][v_key2]["endpoints"].update( {id : _endpoints} )
            self.edge[v_key1][v_key2]["connection_type"].update( {id : _connection_type} )
            self.edge[v_key1][v_key2]["connection_parameters"].update( {id : _connection_parameters} )
            self.edge[v_key1][v_key2]["exchange_values"].update( {id : {}} )
            id += 1
            self.edge[v_key1][v_key2]["connections_count"] = id
            if grounded is not None:
                self.edge[v_key1][v_key2]["grounded"] = grounded
        else:
            # create an new edge
            has_key_v1 = v_key1 in self.node
            has_key_v2 = v_key2 in self.node
            self.add_edge(v_key1, v_key2, {"connections_count":1,
                                           "endpoints":{0:_endpoints},
                                           "connection_type":{0:_connection_type},
                                           "connection_parameters":{0:_connection_parameters},
                                           "exchange_values":{0:{}},
                                           "grounded":grounded or False})
            # avoid auto-created nodes
            if not has_key_v1:
                del self.node[v_key1]
            if not has_key_v2:
                del self.node[v_key2]
        return (v_key1, v_key2)

    def update_bar_lengths(self):
        """update each bar's length so that it can cover all the contact points specified in edges (contact joints)
        """
        for b in self.node:
            # edges are contact joints
            edges_con = self.vertex_connected_edges(b)
            list_pts = []
            # for each connnected joint
            for bar_vert_1, bar_vert_2 in edges_con:
                dpp = compute_contact_line_between_bars(self, bar_vert_1, bar_vert_2)
                self.edge[bar_vert_1][bar_vert_2]["endpoints"][0] = dpp
                points = self.edge[bar_vert_1][bar_vert_2]["endpoints"]
                for p in points.keys():
                    pair_points = points[p]
                    if pair_points != []:
                        for pt in pair_points:
                            if is_point_on_line(pt, self.node[b]["axis_endpoints"], TOL):
                                list_pts.append(pt)

            if len(list_pts) > 0:
                if len(list_pts) > 2:
                    pts_extr = find_points_extreme(list_pts, self.node[b]["axis_endpoints"])
                else:
                    pts_extr = list_pts
                # update axis end points
                self.node[b].update({"axis_endpoints":pts_extr})

    ##################################
    # individual get fn

    def get_bar_axis_end_pts(self, bar_v_key, scale=1.0):
        """return axis end points of a bar node

        Parameters
        ----------
        bar_v_key : int
            [description]

        Returns
        -------
        list of two points
            [description]
        """
        bar = self.node[bar_v_key]
        return (scale_vector(bar["axis_endpoints"][0], scale), scale_vector(bar["axis_endpoints"][1], scale))

    def get_connector_end_pts(self, b1, b2, scale=1.0):
        """return axis end points of a connection between bar ``b1`` and ``b2``

        Parameters
        ----------
        b1 : int
            bar node key
        b2 : int
            [description]

        Returns
        -------
        list of two points
            [description]
        """
        end_pts = list(self.edge[b1][b2]["endpoints"].values())[0]
        return (scale_vector(end_pts[0], scale), scale_vector(end_pts[1], scale))

    def get_bar_pb_body(self, bar_v_key, color=apply_alpha(RED, 0), regenerate=False, scale=METER_SCALE):
        """get pybullet body of a particular bar

        Parameters
        ----------
        bar_v_key : int
            [description]

        Returns
        -------
        int
            [description]
        """
        if bar_v_key not in self.node:
            # cprint('bar key not in the node {}'.format(bar_v_key))
            return None
        if regenerate or 'pb_body' not in self.node[bar_v_key] or \
            self.node[bar_v_key]['pb_body'] is None or \
            self.node[bar_v_key]['pb_body'] not in get_bodies():
            # if cannot find the body in the environment, useful when the env is recreated
            axis_pts = self.get_bar_axis_end_pts(bar_v_key)
            radius = self.node[bar_v_key]['radius']
            self.node[bar_v_key]['pb_body'] = create_bar_body(axis_pts, radius, scale=scale)
        set_color(self.node[bar_v_key]['pb_body'], color)
        return self.node[bar_v_key]['pb_body']

    ##################################
    # export dict info for planning

    def get_element_bodies(self, indices=None, color=apply_alpha(RED, 0), scale=METER_SCALE):
        """[summary]

        Returns
        -------
        dict
            bar vkey -> pb body
        """
        bar_keys = self.nodes() if indices is None else indices
        return {v : self.get_bar_pb_body(v, color, scale=scale) for v in bar_keys if len(self.node[v])>0}

    def set_body_color(self, color, indices=None):
        bar_keys = self.nodes() if indices is None else indices
        for k in bar_keys:
            set_color(self.node[k]['pb_body'], color)

    def get_element_from_index(self, indices=None, scale=1.0):
        element_from_index = {}
        bar_keys = self.nodes() if indices is None else indices
        for index in bar_keys:
            axis_pts = [np.array(pt) for pt in self.get_bar_axis_end_pts(index, scale=scale)]
            radius=self.node[index]['radius']*scale
            body = self.get_bar_pb_body(index, scale=scale)
            # goal_pose = self.node[index]['goal_pose']
            goal_pose = get_pose(body)
            layer = self.node[index]['layer']
            # all data in Element is in meter
            element_from_index[index] = Element(index=index, body=body,
                                                axis_endpoints=axis_pts,
                                                radius=radius,
                                                initial_pose=WorldPose(index, None),
                                                goal_pose=WorldPose(index, goal_pose),
                                                grasps=None,
                                                goal_supports=None,
                                                layer=layer)
        return element_from_index

    def get_axis_pts_from_element(self, scale=1.0):
        """[summary]

        Returns
        -------
        dict
            bar vkey -> ([x,y,z], [x,y,z])
        """
        return {v : self.get_bar_axis_end_pts(v, scale=scale) for v in self.nodes()}

    def get_connectors(self, scale=1.0):
        connectors = {}
        for b1, b2 in self.edges():
            connectors[(b1, b2)] = self.get_connector_end_pts(b1, b2, scale)
            # connectors[(b2, b1)] = self.get_connector_end_pts(b1, b2, scale)
        return connectors

    def get_grounded_bar_keys(self):
        # return frozenset(filter(lambda e: is_ground(e, ground_nodes), elements))
        return frozenset([bv for bv, attr in self.nodes(True) if attr['grounded']])

    def get_grounded_connector_keys(self):
        # return frozenset([bv for bv, attr in self.nodes(True) if attr['grounded']])
        pass

    ##################################
    # mutual collision check

    ##################################
    # tform
    def base_centroid(self, scale=1.0):
        node_points = []
        for _, pts in self.get_axis_pts_from_element(scale=scale).items():
            node_points.extend(pts)
        centroid = np.average(np.array(node_points), axis=0)
        min_z = np.min(node_points, axis=0)[2]  # - 1e-2
        return np.append(centroid[:2], [min_z])

    def transform(self, new_base_centroid, scale=1.0):
        old_base_centroid = self.base_centroid(scale)
        def recenter_point(point):
            return 1.0/scale * (scale*np.array(point) - old_base_centroid + new_base_centroid)

        # update vertex end pts
        for bar_k, bar_vals in self.node.items():
            if is_connected() and 'pb_body' in self.node[bar_k] and \
                self.node[bar_k]['pb_body'] in get_bodies():
                remove_body(self.node[bar_k]['pb_body'])
            self.node[bar_k]["axis_endpoints"] = (list(recenter_point(bar_vals["axis_endpoints"][0])),
                                                  list(recenter_point(bar_vals["axis_endpoints"][1]))
                                                  )
            self.node[bar_k]['pb_body'] = self.get_bar_pb_body(bar_k, regenerate=True, scale=scale)

        # update connector end pts
        for b1, b2 in self.edges():
            contact_pts = list(self.edge[b1][b2]["endpoints"].values())[0]
            self.edge[b1][b2]["endpoints"] = {0:(list(recenter_point(contact_pts[0])), list(recenter_point(contact_pts[1])))}

    # TODO: rotation, scaling: https://github.com/caelan/pb-construction/blob/master/extrusion/run.py#L75

    ##################################
    # structural model extraction
    # TODO: element into segments (axis-pt - connector pt)
    # existence of connector based on existence of neighbor element
