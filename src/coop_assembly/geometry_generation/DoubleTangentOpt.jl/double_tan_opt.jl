using Printf
using JSON
using LinearAlgebra: norm
using JuMP
using Gurobi
using Base.Iterators
import DataStructures: DefaultDict

include("utils.jl")

# set ENV["PYTHON"] = "... path of the python executable ..."
# run Pkg.build("PyCall")
# and re-launch Julia
using PyCall
PyPb = pyimport("pybullet_planning")

# using Makie

const ROOT_DIR = abspath(joinpath(@__FILE__, [".." for i in 1:5]...))
const DATA_DIR = joinpath(ROOT_DIR, "tests", "test_data")

# original units of millimeters
const SCALE = 1e-2
const COORDINATES = ["x", "y", "z"]

#########################################

function parse_point(json_point)
    return SCALE*[json_point[k] for k in COORDINATES]
end

function get_other(edge, node)
    return find_unique(n -> n != node, edge)
end

function enumerate_steps(edge, nodes, edges, fraction=0.1, spacing=1.5) # 1e-3 | 0.1
    step_size = spacing * edges[edge]["radius"]
    num_steps = ceil(Int, fraction * get_length(edge, nodes) / step_size)
    return range(0., stop=fraction, length=num_steps)
end

#########################################

function create_element(p1, p2, radius; color=PyPb.apply_alpha(PyPb.RED, alpha=1))
    height = norm(p2 - p1)
    center = (p1 + p2) / 2
    # extents = (p2 - p1) / 2
    delta = p2 - p1
    x, y, z = delta
    phi = atan2(y, x)
    theta = acos(z / norm(delta))
    quat = PyPb.quat_from_euler(PyPb.Euler(pitch=theta, yaw=phi))
    # p1 is z=-height/2, p2 is z=+height/2

    # Visually, smallest diameter is 2e-3
    # TODO: boxes
    #body = create_cylinder(radius, height, color=color, mass=STATIC_MASS)
    body = PyPb.create_capsule(radius, height, color=color, mass=PyPb.STATIC_MASS)
    PyPb.set_point(body, center)
    PyPb.set_quat(body, quat)
    return body
end

function visualize_solution(nodes, edges, solution)
    edge_points = Dict()
    for ((edge, node), point) in solution
        edge_points[edge] = []
        append!(edge_points[edge], point)
    #     draw_point(point, size=2*edges[edge]["radius"], color=BLUE)
    #     body = create_sphere(edges[edge]["radius"], color=apply_alpha(BLUE, 0.25), mass=STATIC_MASS)
    #     set_point(body, point)

    #     point2 = solution[edge, get_other(edge, node)]
    #     for l in enumerate_steps(edge, nodes, edges)
    #         trailing = (1 - l) * point + (l * point2)
    #         body = create_sphere(edges[edge]["radius"], color=apply_alpha(BLUE, 0.25), mass=STATIC_MASS)
    #         set_point(body, trailing)
    #     end
    end

    # TODO: analyze collisions and proximity
    #bodies = []
    for (edge, points) in edge_points
        point1, point2 = points
        println("$edge: $(norm(point1-point2))")
        element = create_element(point1, point2, edges[edge]["radius"]; color=PyPb.apply_alpha(RED, 0.25))
        #bodies.append(element)
        PbPb.add_line(point1, point2, color=BLUE)
    end
    PyPb.wait_if_gui()
end

#########################################

function solve(nodes, edges, aabb;
               min_tangents=2, # 2 | INF
               length_tolerance=SCALE*10, contact_tolerance=SCALE*1, buffer_tolerance=SCALE*0,
               num_solutions=1, max_time=1*60, verbose=true)
    # * Define the model
    # define name of the model, it could be anything, not necessarily "model"
    model = Model(with_optimizer(Gurobi.Optimizer))

    # https://www.gurobi.com/documentation/9.0/refman/parameters.html
    # model.setParam(GRB.Param.NonConvex, 2) # PSDTol
    set_optimizer_attribute(model, "OutputFlag", verbose)
    set_optimizer_attribute(model, "TimeLimit", max_time)
    if num_solutions < Inf
        set_optimizer_attribute(model, "SolutionLimit", num_solutions)
    end

    # * Variables and Objective
    x_vars = Dict()
    objective = Dict()
    for edge in keys(edges)
        for node in edge
            # var = np_var(model, lower=aabb.lower, upper=aabb.upper)
            vars = @variable(model, [i=1:3],
                lower_bound=aabb[1][i], upper_bound=aabb[2][i],
                base_name="x[($(Tuple(edge)),$node]")
            # * create start values
            # hint = hint_solution[edge, node]
            # for v, hint in safe_zip(var, hint):
                # VarHintVal versus Start: Variables hints and MIP starts are similar in concept,
                # but they behave in very different ways
                # v.VarHintVal = hint # case insensitive?
                #v.setAttr(GRB.Attr.VarHintVal, hint)
                #v.getAttr(GRB.Attr.VarHintVal) # TODO: error
            # end
            # set_start_value(x, val)
            difference = vars .- nodes[node]["point"]
            x_vars[edge, node] = vars
            objective[edge, node] = sum(difference.*difference)
        end

        node1, node2 = edge
        length = norm(nodes[node1]["point"] - nodes[node2]["point"]) - 2*edges[edge]["radius"]
        steps = length / edges[edge]["radius"]
        @printf "Sphere approx: %.3f\n" steps

        @printf "Edge: %s | Length: %.3f\n" edge length
        difference = x_vars[edge, node2] - x_vars[edge, node1]
        if length_tolerance < Inf
            # TODO: make length_tolerance a function of the radius
            @constraint(model, sum(difference.^2) >= (length - length_tolerance)^2)
            @constraint(model, sum(difference.^2) <= (length + length_tolerance)^2)
        end
    end
    @objective(model, Min, sum(values(objective)))

    adjacent = DefaultDict([])
    for edge in keys(edges)
        for node in edge
            append!(adjacent[node], edge)
        end
    end

    z_vars = Dict()
    for (node, neighbors) in adjacent
        # num_tangents = min(len(neighbors) - 1, 2)
        # print(len(edges), num_tangents)
        for pair in map(Tuple, partition(neighbors, 2))
            var = @variable(model, binary=true)
            set_name(var, "z[$(Tuple(pair)), $(node)]")
            z_vars[pair, node] = var
        end
    end

    z_var_from_edge = DefaultDict(Dict())
    for ((pair, node), var) in z_vars
        for (i, edge) in enumerate(pair)
            z_var_from_edge[edge, node][i] = var
        end
    end

    # * Constraints
    # contact degree constraint
    for ((edge, node), neighbors) in z_var_from_edge
        num_tangents = min(length(neighbors), min_tangents)
        # degree = len(neighbors) + 1
        @printf("Neighbors: %s | Tangents: %s\n", length(neighbors), num_tangents)
        # model.addConstr(sum(neighbors) == num_tangents)
        ne_con = @constraint(model, sum(values(neighbors)) == num_tangents)
        set_name(ne_con, "NeCon[$(Tuple(edge)), $node]")

        if length(neighbors) == num_tangents
            for z_var in values(neighbors)
                #model.addConstr(z_var == 1)
                #z_var.start = 1
                # z_var.lb = z_var.ub = 1
                fix(z_var, 1; force = true)
            end
        end
    end

    # contact/collision constraint
    for ((edge1, node1), (edge2, node2)) in partition(x_vars, 2)
        if edge1 == edge2
            @assert node1 != node2
            continue
        end
        var1, var2 = x_vars[edge1, node1], x_vars[edge2, node2]
        difference = var2 - var1
        radius_distance = edges[edge1]["radius"] + edges[edge2]["radius"]

        if node1 == node2
            pair = Set((edge1, edge2))
            z_var = z_vars[pair, node1]
            # model.addConstr(sum(difference * difference) <=
                            # (distance + contact_tolerance) ** 2 + (1 - z_var) * max_distance ** 2)
            # ! contact constraint
            contact_con = @constraint(model, sum(difference .* difference) <= (radius_distance + contact_tolerance)^2 + (1 - z_var) * max_distance^2)
            set_name(contact_con, "ContactCon[$(Tuple(edge1)),$node1,$(Tuple(edge2))]")

            other1 = get_other(edge1, node1)
            for l1 in enumerate_steps(edge1, nodes, edges)
                point1 = (1 - l1) * var1 + (l1 * x_vars[edge1, other1])
                other2 = get_other(edge2, node2)
                for l2 in enumerate_steps(edge2, nodes, edges)
                    point2 = (1 - l2) * var2 + (l2 * x_vars[edge2, other2])
                    difference = point2 - point1
                    # ! collision constraint
                    collision_con = @constraint(model, sum(difference .* difference) >= (distance + buffer_tolerance)^2)  # Only neighbors
                    set_name(collision_con, "CollisionCon[$(Tuple(edge1)),$node1,$(Tuple(edge2))]")
                end
            end
        end
    end

    ###################################

    status=optimize!(model) # time to optimize!
    @printf("Objective: %.3f | # Solutions: %d | Status: %s", objective_value(model), result_count(model), termination_status(model))

    if result_count(model) <= 0
        return
    end

    for (edge, neighbors) in z_var_from_edge
        println("Edge $(Tuple(edge)) : neighbors $(Tuple(neighbors))")
    end

    solution = Dict((edge, node) => [v.x for v in var]
                for ((edge, node), var) in x_vars)
    println(solution)
    # visualize_solution(nodes, edges, solution)
end

function main(viewer=true)
    # file_name = "cube_skeleton.json"
    file_name = "2_tets.json"

    file_path = joinpath(DATA_DIR, file_name)
    json_data = JSON.parsefile(file_path)
    json_data = json_data["overall_structure"] # bar_structure | overall_structure
    # bar_structure: adjacency, attributes, edge, node
    # overall_structure: adjacency, edge, node

    neighbors_from_node = Dict(tryparse(Int,n) => Set(keys(neighbors)) for (n, neighbors) in json_data["adjacency"])
    # println("neighnor from node", neighbors_from_node)

    edges = Dict(Set([tryparse(Int,n1), tryparse(Int,n2)]) => info for (n1, neighbors) in json_data["edge"] for (n2, info) in neighbors)
    for info in values(edges)
        info["radius"] = 3.17
        info["radius"] *= SCALE
    end
    # println("edges: ", edges)

    nodes = Dict(tryparse(Int, n) => Dict("point" => parse_point(info)) for (n, info) in json_data["node"]) # "fixed": info["fixed"]

    points = [info["point"] for info in values(nodes)]
    aabb = PyPb.aabb_from_points(points)
    # println("aabb: ", aabb)

    scale = 2.
    center = PyPb.get_aabb_center(aabb)
    extent = PyPb.get_aabb_extent(aabb)
    aabb = PyPb.AABB(lower=center - scale*extent/2, upper=center + scale*extent/2)
    println("aabb: ", aabb)

    # PyPb.connect(use_gui=true)
    # handles = PyPb.draw_pose(PyPb.Pose(), length=1)
    # append!(handles, [PyPb.add_line(nodes[n1]["point"], nodes[n2]["point"], color=PyPb.RED) for (n1, n2) in keys(edges)])
    # catch
    #     println("Errored, pybullet disconnected.")
    #     PyPb.disconnect()
    # end

    solve(nodes, edges, aabb)
    # PyPb.wait_if_gui("Finished.")
    # PyPb.disconnect()
end

main()

    # plotting
    # http://juliaplots.org/MakieReferenceImages/gallery//fluctuation_3d/index.html
    # scene = Scene()
