from __future__ import print_function

import argparse
import os
import sys

import numpy as np

sys.path.append(os.environ['PDDLSTREAM_PATH'])

# PDDLSTREAM_PATH

from pddlstream.utils import adjacent_from_edges
from examples.pybullet.utils.pybullet_tools.utils import connect, read_json, wait_for_user, disconnect, GREEN, add_line, \
    RED, draw_pose, Pose, aabb_from_points, draw_aabb, get_aabb_center, get_aabb_extent, AABB, get_distance, get_pairs, wait_if_gui, \
    INF, STATIC_MASS, set_color, quat_from_euler, apply_alpha, create_cylinder, set_point, set_quat, get_aabb, Euler

from itertools import combinations
from collections import defaultdict
from gurobipy import Model, GRB, quicksum, GurobiError

# https://github.com/yijiangh/coop_assembly/tree/feature/truss_gen/tests/test_data
# ls tests/test_data/*skeleton.json

ROOT_DIR = os.path.abspath(os.path.join(__file__, *[os.pardir]*4))
DATA_DIR = os.path.join(ROOT_DIR, 'tests', 'test_data')
print(DATA_DIR)

#sys.path.append(os.path.join(DATA_DIR, 'external', 'pybullet_planning', 'src', 'pybullet_planning'))

SCALE = 1e-2 # original units of millimeters

def parse_point(json_point):
    return SCALE*np.array([json_point[k] for k in ['x', 'y', 'z']])

def unbounded_var(model):
    return model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)

def np_var(model, d=3):
    return np.array([unbounded_var(model) for _ in range(d)])

def create_element(p1, p2, radius, color=apply_alpha(RED, alpha=1)):
    height = np.linalg.norm(p2 - p1)
    center = (p1 + p2) / 2
    # extents = (p2 - p1) / 2
    delta = p2 - p1
    x, y, z = delta
    phi = np.math.atan2(y, x)
    theta = np.math.acos(z / np.linalg.norm(delta))
    quat = quat_from_euler(Euler(pitch=theta, yaw=phi))
    # p1 is z=-height/2, p2 is z=+height/2

    # Visually, smallest diameter is 2e-3
    body = create_cylinder(radius, height, color=color, mass=STATIC_MASS)
    set_point(body, center)
    set_quat(body, quat)
    return body

def solve_gurobi(nodes, edges, max_distance, epsilon=0.001, num_solutions=INF,
                 max_time=1*60, verbose=True):
    # https://www.gurobi.com/documentation/9.0/refman/py_python_api_details.html

    model = Model(name='Construction')
    model.setParam(GRB.Param.OutputFlag, verbose)
    model.setParam(GRB.Param.TimeLimit, max_time)
    model.setParam(GRB.Param.NonConvex, 2) # PSDTol
    if num_solutions < INF:
        model.setParam(GRB.Param.SolutionLimit, num_solutions)

    x_vars = {}
    objective = []
    for edge in edges:
        for node in edge:
            var = np_var(model, d=3)
            x_vars[edge, node] = var
            difference = var - nodes[node]['point']
            objective.append(sum(difference*difference))
            #model.addConstr( ) #, "c0")
    model.setObjective(quicksum(objective), sense=GRB.MINIMIZE)

    #adjacent = adjacent_from_edges(edges) # vertices
    adjacent = defaultdict(set)
    for edge in edges:
        for node in edge:
            adjacent[node].add(edge)

    z_vars = {}
    for node, neighbors in adjacent.items():
        #num_tangents = min(len(neighbors) - 1, 2)
        #print(len(edges), num_tangents)
        for pair in map(frozenset, combinations(neighbors, r=2)):
            z_vars[pair, node] = model.addVar(vtype=GRB.BINARY)  # , name="x")

    z_var_from_edge = defaultdict(list)
    for (pair, node), var in z_vars.items():
        for edge in pair:
            z_var_from_edge[edge, node].append(var)
    for vars in z_var_from_edge.values():
        degree = len(vars) + 1
        num_tangents = min(degree - 1, 2)
        #print(len(vars), degree, num_tangents)
        model.addConstr(sum(vars) == num_tangents)

    for (edge1, node1), (edge2, node2) in combinations(x_vars, r=2):
        if edge1 == edge2:
            continue
        var1, var2 = x_vars[edge1, node1], x_vars[edge2, node2]
        difference = var2 - var1
        distance = edges[edge1]['radius'] + edges[edge2]['radius']
        model.addConstr(sum(difference*difference) >= distance**2)
        if node1 == node2:
            pair = frozenset({edge1, edge2})
            z_var = z_vars[pair, node1]
            model.addConstr(sum(difference*difference) <=
                            (distance + epsilon)**2 + (1 - z_var)*max_distance**2) # + epsilon
    # d = model.addVar(lb=0, ub=GRB.INFINITY)

    try:
        model.optimize()
    except GurobiError as e:
        raise e
    print('Objective: {:.3f}'.format(model.objVal))

    # https://www.gurobi.com/documentation/7.5/refman/optimization_status_codes.html
    if model.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):  # OPTIMAL | SUBOPTIMAL
        pass

    edge_points = defaultdict(list)
    for (edge, node), var in x_vars.items():
        point = np.array([v.x for v in var])
        edge_points[edge].append(point)
    bodies = []
    for edge, points in edge_points.items():
        point1, point2 = points
        bodies.append(create_element(point1, point2, edges[edge]['radius']))
        #add_line(point1, point2, color=GREEN)
    wait_if_gui()

def main():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--viewer', action='store_true')
    # parser.add_argument('-n', '--n_trails', default=1, help='number of trails')
    # parser.add_argument('-w', '--watch', action='store_true', help='watch trajectories')
    # parser.add_argument('-sm', '--step_sim', action='store_true', help='stepping simulation.')
    # parser.add_argument('-wr', '--write', action='store_true', help='Export results')
    # parser.add_argument('-db', '--debug', action='store_true', help='Debug verbose mode')
    args = parser.parse_args()
    print('Arguments:', args)

    skeletons = [file_name for file_name in os.listdir(DATA_DIR) if file_name.endswith('_skeleton.json')]
    print(len(skeletons), skeletons)
    #skeleton = skeletons[0]
    skeleton = 'cube_skeleton.json'

    json_data = read_json(os.path.join(DATA_DIR, skeleton))
    #print(json.dumps(json_data, sort_keys=True, indent=2))

    neighbors_from_node = {n: (set(neighbors)) for n, neighbors in json_data['adjacency'].items()} # sorted
    print(neighbors_from_node)
    edges = {frozenset({n1, n2}): info for n1, neighbors in json_data['edge'].items() for n2, info in neighbors.items()}
    for info in edges.values():
        info['radius'] *= SCALE
    print(edges)
    nodes = {n: {'point': parse_point(info), 'fixed': info['fixed']} for n, info in json_data['node'].items()}
    print(nodes)


    connect(use_gui=args.viewer)
    #floor = create_plane(color=GREEN)
    handles = draw_pose(Pose(), length=1)
    handles += [add_line(nodes[n1]['point'], nodes[n2]['point'], color=RED) for n1, n2 in edges]

    points = [info['point'] for info in nodes.values()]
    aabb = aabb_from_points(points)
    #hull = convex_hull(points)

    scale = 2.
    center = get_aabb_center(aabb)
    extent = get_aabb_extent(aabb)
    aabb = AABB(lower=center - scale*extent/2, upper=center + scale*extent/2)
    max_distance = get_distance(*aabb)
    #handles.extend(draw_aabb(aabb, color=GREEN))
    #wait_if_gui()

    solve_gurobi(nodes, edges, max_distance)

    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()
