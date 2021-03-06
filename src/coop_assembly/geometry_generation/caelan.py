from __future__ import print_function

import argparse
import os
import sys
import json
import math
import numpy as np

sys.path.append(os.environ['PDDLSTREAM_PATH'])

from pddlstream.utils import str_from_object, find_unique
# from examples.pybullet.utils.pybullet_tools.utils import connect, read_json, disconnect, add_line, \
from pybullet_planning import connect, read_json, disconnect, add_line, \
    RED, draw_pose, Pose, aabb_from_points, get_aabb_center, get_aabb_extent, AABB, get_distance, wait_if_gui, \
    INF, STATIC_MASS, quat_from_euler, set_point, set_quat, Euler, SEPARATOR, \
    draw_point, BLUE, safe_zip, apply_alpha, create_sphere, create_capsule, set_camera, create_cylinder, GREEN

from itertools import combinations, permutations
from collections import defaultdict, Counter
from gurobipy import Model, GRB, quicksum, GurobiError

# https://github.com/yijiangh/coop_assembly/tree/feature/truss_gen/tests/test_data
# ls tests/test_data/*skeleton.json

ROOT_DIR = os.path.abspath(os.path.join(__file__, *[os.pardir]*4))
DATA_DIR = os.path.join(ROOT_DIR, 'tests', 'test_data')

#sys.path.append(os.path.join(DATA_DIR, 'external', 'pybullet_planning', 'src', 'pybullet_planning'))

SCALE = 1e-1 # original units of millimeters

COORDINATES = ['x', 'y', 'z']

EDGE = frozenset
#EDGE = lambda nodes: frozenset(map(int, nodes))
#EDGE = lambda x, y: frozenset({x, y})
#def EDGE(nodes):
#    return frozenset(map(int, nodes))

##################################################

def parse_point(json_point):
    return SCALE*np.array([json_point[k] for k in COORDINATES])

def unbounded_var(model, lower=-GRB.INFINITY, upper=GRB.INFINITY, name=''):
    return model.addVar(lb=lower, ub=upper, name=name)

def np_var(model, name='', lower=None, upper=None):
    if lower is None:
        lower = len(COORDINATES)*[-GRB.INFINITY]
    if upper is None:
        upper = len(COORDINATES)*[GRB.INFINITY]
    return np.array([unbounded_var(model, lower=lower[i], upper=upper[i], name=name)
                     for i, k in enumerate(COORDINATES)])

def create_element(p1, p2, radius, capsule=True, color=apply_alpha(RED, alpha=1)):
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
    # TODO: boxes
    if capsule:
        body = create_capsule(radius, height, color=color, mass=STATIC_MASS)
    else:
        body = create_cylinder(radius, height, color=color, mass=STATIC_MASS)
    set_point(body, center)
    set_quat(body, quat)
    return body

def get_other(edge, node):
    return find_unique(lambda n: n != node, edge)

def get_length(edge, nodes):
    n1, n2 = list(edge)
    return get_distance(nodes[n1]['point'], nodes[n2]['point'])

def enumerate_steps(length, radius, fraction=0.25, spacing=1.5): # 1e-3 | 0.1
    step_size = spacing * radius
    num_steps = int(np.ceil(fraction * length / step_size))
    return np.linspace(start=0., stop=fraction, num=num_steps, endpoint=True)

##################################################

def center_viewer(nodes, pitch=-np.pi/8, distance=2):
    # TODO: be robust to SCALE
    centroid = np.average([nodes[node]['point'] for node in nodes], axis=0)
    centroid[2] = min(nodes[node]['point'][2] for node in nodes)
    set_camera(yaw=math.degrees(0), pitch=math.degrees(pitch), distance=distance, target_position=centroid)
    return draw_pose(Pose(point=centroid), length=1)

def visualize_structure(nodes, edges):
    for edge in edges:
        for n1, n2 in permutations(edge):
            point1, point2 = nodes[n1]['point'], nodes[n2]['point']
            for l in enumerate_steps(get_distance(point1, point2), edges[edge]['radius']):
                point = l*point1 + (1-l)*point2
                draw_point(point, size=2. * edges[edge]['radius'], color=BLUE)
                body = create_sphere(edges[edge]['radius'], color=apply_alpha(BLUE, 0.25), mass=STATIC_MASS)
                set_point(body, point)
    wait_if_gui()

def visualize_solution(edges, solution, alpha=0.25, **kwargs):
    bodies = []
    edge_points = defaultdict(list)
    for (edge, node), point in solution.items():
        edge_points[edge].append(point)
        #draw_point(point, size=2*edges[edge]['radius'], color=BLUE)

        point2 = solution[edge, get_other(edge, node)]
        for l in enumerate_steps(get_distance(point, point2), edges[edge]['radius'], **kwargs):
            trailing = (1 - l) * point + (l * point2)
            body = create_sphere(edges[edge]['radius'], color=apply_alpha(BLUE, alpha), mass=STATIC_MASS)
            set_point(body, trailing)
            bodies.append(body)

    # TODO: analyze collisions and proximity
    for edge, points in edge_points.items():
        point1, point2 = points
        print('{}: {:.3f}'.format(str_from_object(edge), get_distance(point1, point2)))
        bodies.append(create_element(point1, point2, edges[edge]['radius'], color=apply_alpha(RED, alpha)))
        #add_line(point1, point2, color=BLUE)
    wait_if_gui()
    return bodies

##################################################

def create_hint(nodes, edges, shrink=True):
    hint_solution = {}
    for edge in edges:
        for node in edge:
            node_point = nodes[node]['point']
            if shrink:
                other = get_other(edge, node)
                other_point = nodes[other]['point']
                fraction = 2*edges[edge]['radius'] / get_distance(node_point, other_point)
                point = fraction*other_point + (1 - fraction)*node_point
            else:
                point = node_point
            hint_solution[edge, node] = point
    return hint_solution

##################################################

def create_point_var(model, aabb, edge, node, hint_solution=None, hard=False):
    hint = hint_solution[edge, node]
    var = np_var(model, lower=aabb.lower, upper=aabb.upper)
    if hint_solution is not None:
        for v, hint in safe_zip(var, hint):
            # VarHintVal versus Start: Variables hints and MIP starts are similar in concept,
            # but they behave in very different ways
            if hard:
                v.start = hint
            else:
                v.varHintVal = hint  # case insensitive?
                # v.setAttr(GRB.Attr.VarHintVal, hint)
                # v.getAttr(GRB.Attr.VarHintVal) # TODO: error
    return var

def convex_combination(x1, x2, w=0.5):
    #assert 0 <= w <= 1
    return (1 - w) * x1 + (w * x2)

def x_from_var(var):
    return np.array([v.x for v in var])

def solve_gurobi(nodes, edges, aabb, hint_solution=None, min_tangents=2, optimize=True, diagnose=False, # 2 | INF
                 length_tolerance=SCALE*10, contact_tolerance=SCALE*1, buffer_tolerance=SCALE*0, # 10 | INF
                 num_solutions=1, max_time=1*60, verbose=True):
    # https://www.gurobi.com/documentation/9.0/refman/py_python_api_details.html
    # https://scicomp.stackexchange.com/questions/83/is-there-a-high-quality-nonlinear-programming-solver-for-python
    # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
    # https://pypi.org/project/mystic/
    # https://www.cvxpy.org/
    # https://github.com/coin-or/Ipopt

    # TODO: all elements meeting at a joint represented by a sphere
    print(SEPARATOR)
    assert contact_tolerance >= buffer_tolerance
    max_distance = get_distance(*aabb)

    if hint_solution is None:
        # TODO: diagnose initial infeasibility
        hint_solution = create_hint(nodes, edges)
    #visualize_solution(edges, hint_solution)

    model = Model(name='Construction')
    # https://www.gurobi.com/documentation/9.0/refman/parameters.html#sec:Parameters
    model.setParam(GRB.Param.OutputFlag, verbose)
    model.setParam(GRB.Param.TimeLimit, max_time)
    model.setParam(GRB.Param.NonConvex, 2) # PSDTol
    # https://www.gurobi.com/documentation/9.0/refman/num_tolerances_and_user_sc.html
    #model.setParam(GRB.Param.FeasibilityTol, SCALE*1e-3) # 1e-6
    if num_solutions < INF:
        model.setParam(GRB.Param.SolutionLimit, num_solutions)

    # TODO: name the variables
    x_vars = {}
    objective = []
    for edge in edges:
        for node in edge:
            x_vars[edge, node] = create_point_var(model, aabb, edge, node, hint_solution)
            difference = x_vars[edge, node] - nodes[node]['point']
            objective.append(sum(difference*difference))

        node1, node2 = edge
        length = get_distance(nodes[node1]['point'], nodes[node2]['point']) - 2*edges[edge]['radius']
        #steps = length / edges[edge]['radius']
        #print('Sphere approx: {:.3f}'.format(steps))
        #print('Edge: {} | Length: {:.3f}'.format(str_from_object(edge), length))

        difference = x_vars[edge, node2] - x_vars[edge, node1]
        # * length approximation
        if length_tolerance < INF:
            # TODO: make length_tolerance a function of the radius
            model.addConstr(sum(difference * difference) >= (length - length_tolerance) ** 2)
            model.addConstr(sum(difference * difference) <= (length + length_tolerance) ** 2)
    if optimize:
        # TODO: max deviation
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
        for pair in map(EDGE, combinations(neighbors, r=2)):
            # TODO: varHintVal
            z_vars[pair, node] = model.addVar(vtype=GRB.BINARY, name='')  # , name="x")

    z_var_from_edge = defaultdict(list)
    for (pair, node), var in z_vars.items():
        for edge in pair:
            z_var_from_edge[edge, node].append(var)
            # Constraint that one is less than the other

    for neighbors in z_var_from_edge.values():
        num_tangents = min(len(neighbors), min_tangents)
        #degree = len(neighbors) + 1
        #print('Neighbors: {} | Tangents: {}'.format(len(neighbors), num_tangents))
        model.addConstr(sum(neighbors) == num_tangents)
        if len(neighbors) == num_tangents:
            for z_var in neighbors:
                #model.addConstr(z_var == 1)
                z_var.lb = z_var.ub = 1
                #z_var.start = 1

    contact_vars = {}
    epsilon = SCALE*0 # 0 | 1
    for (edge1, node1), (edge2, node2) in combinations(x_vars, r=2):
        if edge1 == edge2:
            assert node1 != node2
            continue
        var1, var2 = x_vars[edge1, node1], x_vars[edge2, node2]
        other1 = get_other(edge1, node1)
        other2 = get_other(edge2, node2)
        distance = edges[edge1]['radius'] + edges[edge2]['radius']
        if node1 == node2:
            #l1_var = l2_var = 0
            # https://en.wikipedia.org/wiki/Linear_complementarity_problem

            # TODO: distance from the element line
            l1_var = np.full(var1.shape, model.addVar(lb=0, ub=0.5))
            point1_var = create_point_var(model, aabb, edge1, node1, hint_solution)
            contact1_var = (1 - l1_var) * var1 + (l1_var * x_vars[edge1, other1])
            #contact1_var = convex_combination(var1, x_vars[edge1, other1], w=l1_var)
            #model.addConstr(contact1_var == point1_var)
            diff1 = contact1_var - point1_var
            #model.addConstr(sum(diff1*diff1) < 1e-2) # TODO: doesn't work (as predicted)
            # * enforce point1 and contact 1 are the same points
            for x in diff1:
                model.addConstr(x >= -epsilon)
                model.addConstr(x <= epsilon)

            l2_var = np.full(var2.shape, model.addVar(lb=0, ub=0.5))
            point2_var = create_point_var(model, aabb, edge2, node2, hint_solution)
            contact2_var = (1 - l2_var) * var2 + (l2_var * x_vars[edge2, other2])
            for x in (contact2_var - point2_var):
                model.addConstr(x >= -epsilon)
                model.addConstr(x <= epsilon)

            # ! ideally
            # difference = contact1_var - contact2_var
            difference = point2_var - point1_var
            pair = EDGE({edge1, edge2})
            z_var = z_vars[pair, node1]
            contact_vars[edge1, edge2, node1] = (z_var, l1_var, point1_var, l2_var, point2_var)
            model.addConstr(sum(difference * difference) <=
                            (distance + contact_tolerance) ** 2 + (1 - z_var) * max_distance ** 2)

        if node1 == node2: # Only neighbors
            distance1 = get_distance(nodes[node1]['point'], nodes[other1]['point'])
            for l1 in enumerate_steps(distance1, edges[edge1]['radius']):
                point1_var = (1 - l1) * var1 + (l1 * x_vars[edge1, other1])
                distance2 = get_distance(nodes[node2]['point'], nodes[other2]['point'])
                for l2 in enumerate_steps(distance2, edges[edge2]['radius']):
                    point2_var = (1 - l2) * var2 + (l2 * x_vars[edge2, other2])
                    difference = point2_var - point1_var
                    model.addConstr(sum(difference * difference) >= (distance + buffer_tolerance) ** 2)

    # https://www.gurobi.com/documentation/9.0/ampl-gurobi/index.html
    # https://www.gurobi.com/resource/exporting-mps-files/
    # https://www.gurobi.com/products/optimization-modeling-language-resources-support/ampl/
    #model.write(filename='model.mps') # .mps, .rew, .lp, .rlp
    try:
        model.optimize()
    except GurobiError as e:
        raise e
    # TODO: forbid solutions (or apply other constraints)
    # https://www.gurobi.com/documentation/9.0/refman/attributes.html

    # https://www.gurobi.com/documentation/9.0/refman/optimization_status_codes.html
    #feasible = model.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) # OPTIMAL | SUBOPTIMAL
    #obj = model.objVal if feasible else INF
    print('Objective: {:.3f} | Solutions: {} | Status: {}'.format(model.objVal, model.solCount, model.status))

    # print('\nVars: {}'.format(len(model.getVars())))
    # for var in model.getVars(): # TODO: deviation from point
    #     print(var.VarName)
    #
    # print('\nConstraints: {}'.format(len(model.getConstrs())))
    # for constraint in model.getConstrs(): # TODO: constraint violation
    #     print(constraint.GenConstrType)
    #     print(constraint.QCName, constraint.QCSlack)
    #     print(constraint.GenConstrType, ) # model.ConstrVioIndex

    # TODO: scale values?
    if not model.solCount:
        # objective = model.feasRelax(relaxobjtype=0, # feasRelaxS
        #                             minrelax=True,
        #                             vars=[], lbpen=[], ubpen=[],
        #                             constrs=elastic_constraints,
        #                             rhspen=[1.]*len(elastic_constraints))

        if diagnose:
            model.setObjective(0.0)
            model.computeIIS() # gurobipy.GurobiError: Cannot compute IIS on a feasible model
            print('IIS is minimal\n' if model.IISMinimal else 'IIS is not minimal\n' )
            iss_constraints = {c.constrName for c in model.getConstrs() if c.IISConstr}
            print(iss_constraints)
        return None

    # https://www.gurobi.com/documentation/9.0/refman/attributes.html
    # print(model.X)
    # print(model.Xn)
    #print(model.ConstrVio, model.ConstrSVio, model.ConstrVioSum, model.ConstrSVioSum)

    for edge, neighbors in z_var_from_edge.items():
        print(str_from_object(edge), x_from_var(neighbors))
    for (edge1, edge2, node), (z_var, l1_var, point1_var, l2_var, point2_var) in contact_vars.items():
        if z_var.x != 1:
            continue
        distance = sum(edges[edge]['radius'] for edge in [edge1, edge2])
        point1 = x_from_var(point1_var)
        point2 = x_from_var(point2_var)
        # 2. * edges[edge]['radius']
        draw_point(point1, color=GREEN)
        draw_point(point2, color=GREEN)
        add_line(point1, point2, color=GREEN)
        print(str_from_object(pair), x_from_var(l1_var), x_from_var(l2_var),
              distance, get_distance(point1, point2))

    solution = {(edge, node): x_from_var(var)
                for (edge, node), var in x_vars.items()}
    visualize_solution(edges, solution)
    return solution

##################################################

def parse_structure(json_data, radius=3.17):
    # neighbors_from_node = {int(n): (set(map(int, neighbors))) for n, neighbors in json_data['adjacency'].items()} # sorted
    # print(neighbors_from_node)
    edges = {EDGE(map(int, {n1, n2})): info for n1, neighbors in json_data['edge'].items() for n2, info in neighbors.items()}
    for info in edges.values():
        if radius is not None:
            info['radius'] = radius
        info['radius'] *= SCALE
    print('\nElements:', len(edges), edges)

    nodes = {int(n): {'point': parse_point(info)} for n, info in json_data['node'].items()} # 'fixed': info['fixed']
    print('\nNodes:', len(nodes), nodes)
    return nodes, edges

def parse_solution(json_data):
    # TODO: recover tagents and contact points
    elements = {}
    solution = {}
    nodes = Counter()
    for info in json_data['node'].values():
        #create_element(point1, point2, elements[edge]['radius'], color=apply_alpha(RED, 0.25))
        #add_line(point1, point2, color=BLUE)
        edge = EDGE(map(int, info['o_edge']))
        nodes.update(edge)
        elements[edge] = info
        info['radius'] *= SCALE
        # TODO: confirm the endpoints are correct
        for node, point in safe_zip(info['o_edge'], info['axis_endpoints']):
            solution[edge, node] = SCALE*np.array(point)
    print('Elements:', len(elements), set(elements))
    print('Nodes:', len(nodes), sum(nodes.values()), nodes)

    #edges = {EDGE(map(int, {n1, n2})) for n1, neighbors in json_data['edge'].items() for n2, info in neighbors.items()}
    #print(len(edges), edges)
    return elements, solution

##################################################

def main():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--viewer', action='store_true', help='')
    # parser.add_argument('-n', '--n_trails', default=1, help='number of trails')
    args = parser.parse_args()
    print('Arguments:', args)

    #skeletons = [file_name for file_name in os.listdir(DATA_DIR) if file_name.endswith('_skeleton.json')]
    #print(len(skeletons), skeletons)
    #file_name = skeletons[0]
    #file_name = 'cube_skeleton.json'
    #file_name = '2_tets.json'
    file_name = 'truss_one_tet_skeleton.json'

    json_data = read_json(os.path.join(DATA_DIR, file_name))
    #json_data = json_data['bar_structure'] # bar_structure | overall_structure
    print(json.dumps(json_data, sort_keys=True, indent=2))
    # bar_structure: adjacency, attributes, edge (endpoints, grounded), node (axis_endpoints, goal_pose, radius, grounded)
    # overall_structure: adjacency, edge (radius), node (point)

    # for field in ['edge', 'node']:
    #     print(field, list(json_data[field].items())[0])

    connect(use_gui=args.viewer)
    #floor = create_plane(color=GREEN)

    # TODO: sub-structure
    solution = None
    if 'overall_structure' in json_data:
        nodes, edges = parse_structure(json_data['overall_structure'])
    else:
        nodes, edges = parse_structure(json_data)

    structure = 'overall_structure' # bar_structure | overall_structure
    if structure == 'bar_structure':
        edges, solution = parse_solution(json_data[structure])
        #visualize_solution(edges, solution)
    elif structure == 'overall_structure':
        pass
    else:
        raise NotImplementedError(structure)

    center_viewer(nodes)
    handles = [add_line(nodes[n1]['point'], nodes[n2]['point'], color=RED) for n1, n2 in edges]
    points = [info['point'] for info in nodes.values()]
    aabb = aabb_from_points(points)
    #hull = convex_hull(points)

    scale = 2.
    center = get_aabb_center(aabb)
    extent = get_aabb_extent(aabb)
    aabb = AABB(lower=center - scale*extent/2, upper=center + scale*extent/2)
    #handles.extend(draw_aabb(aabb, color=GREEN))
    #wait_if_gui()

    #visualize_structure(nodes, edges)
    solve_gurobi(nodes, edges, aabb, hint_solution=solution)

    disconnect()

if __name__ == '__main__':
    main()
