using Printf
using JSON
using LinearAlgebra: norm
using JuMP
using Gurobi
using Base.Iterators
import DataStructures: DefaultDict
using IterTools
using Statistics

using ArgParse
using Crayons.Box

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

function get_length(edge, nodes)
    n1, n2 = edge
    return norm(nodes[n1]["point"]-nodes[n2]["point"])
end

function enumerate_steps(edge, nodes, edges, fraction=0.1, spacing=1.5) # 1e-3 | 0.1
    step_size = spacing * edges[edge]["radius"]
    num_steps = ceil(Int, fraction * get_length(edge, nodes) / step_size)
    return range(0., stop=fraction, length=num_steps)
end

##################################################

function create_hint(nodes, edges, shrink=true)
    hint_solution = Dict()
    for edge in keys(edges)
        for node in edge
            node_point = nodes[node]["point"]
            if shrink
                other = get_other(edge, node)
                other_point = nodes[other]["point"]
                fraction = 2*edges[edge]["radius"] / norm(node_point - other_point)
                point = fraction*other_point + (1 - fraction)*node_point
            else
                point = node_point
            end
            hint_solution[edge, node] = point
        end
    end
    return hint_solution
end

#########################################

function create_element(p1, p2, radius; color=PyPb.apply_alpha(PyPb.RED, alpha=1))
    height = norm(p2 - p1)
    center = (p1 + p2) / 2
    # extents = (p2 - p1) / 2
    delta = p2 - p1
    x, y, z = delta
    phi = atan(y, x)
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

function center_viewer(nodes, pitch=-π/8, distance=2)
    # TODO: be robust to SCALE
    centroid = mean([nodes[node]["point"] for node in keys(nodes)]; dims=1)[1]
    centroid[2] = min([nodes[node]["point"][3] for node in keys(nodes)]...)
    PyPb.set_camera(yaw=deg2rad(0), pitch=deg2rad(pitch), distance=distance, target_position=centroid)
    return PyPb.draw_pose(PyPb.Pose(point=centroid), length=1)
end

function visualize_solution(edges, solution, alpha=0.25)
    bodies = []
    edge_points = DefaultDict(Vector{Vector})
    for ((edge, node), point) in solution
        push!(edge_points[edge], point)
    #     draw_point(point, size=2*edges[edge]["radius"], color=BLUE)
    #     body = create_sphere(edges[edge]["radius"], color=apply_alpha(BLUE, 0.25), mass=STATIC_MASS)
    #     set_point(body, point)

        # point2 = solution[edge, get_other(edge, node)]
        # for l in enumerate_steps(edge, nodes, edges)
        #     trailing = (1 - l) * point + (l * point2)
        #     body = create_sphere(edges[edge]["radius"], color=apply_alpha(BLUE, 0.25), mass=STATIC_MASS)
        #     set_point(body, trailing)
        # end
    end

    # TODO: analyze collisions and proximity
    for (edge, points) in edge_points
        point1, point2 = points
        point1 = points[1]
        point2 = points[2]
        println("E#$edge: $point1 | $point2 | L: $(norm(point1-point2))")
        element = create_element(point1, point2, edges[edge]["radius"]; color=PyPb.apply_alpha(PyPb.RED, 0.25))
        push!(bodies, create_element(point1, point2, edges[edge]["radius"], color=PyPb.apply_alpha(PyPb.RED, alpha)))
        # PbPb.add_line(point1, point2, color=BLUE)
    end
end

#########################################

function solve(nodes, edges, aabb;
               hint_solution=nothing,
               min_tangents=2, # 2 | INF
               length_tolerance=SCALE*10, contact_tolerance=SCALE*1, buffer_tolerance=SCALE*0,
               num_solutions=1, max_time=1*60, verbose=true)
    # println(String(['-' for i=1:20]...))
    max_distance = norm(aabb[2] - aabb[1])

    if hint_solution === nothing
        # TODO: diagnose initial infeasibility
        hint_solution = create_hint(nodes, edges)
    end

    # * Define the model
    # define name of the model, it could be anything, not necessarily "model"
    model = Model(with_optimizer(Gurobi.Optimizer))

    # https://www.gurobi.com/documentation/9.0/refman/parameters.html
    set_optimizer_attribute(model, "NonConvex", 2)
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
                base_name="x[($(Tuple(edge)),$node]",
                start=hint_solution[edge, node][i])
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

    adjacent = DefaultDict(Set)
    for edge in keys(edges)
        for node in edge
            push!(adjacent[node], edge)
        end
    end

    z_vars = Dict()
    for (node, neighbors) in adjacent
        # num_tangents = min(len(neighbors) - 1, 2)
        # print(len(edges), num_tangents)
        for pair in map(Set, subsets(collect(neighbors), 2))
            var = @variable(model, binary=true)
            set_name(var, "z[$(Tuple(pair)), $(node)]")
            z_vars[pair, node] = var
        end
    end
    @show adjacent

    z_var_from_edge = DefaultDict(Dict())
    for ((pair, node), var) in z_vars
        for (i, edge) in enumerate(pair)
            z_var_from_edge[edge, node][i] = var
        end
    end

    # * Constraints
    # ! contact degree constraint (linear)
    for ((edge, node), neighbors) in z_var_from_edge
        num_tangents = min(length(neighbors), min_tangents)
        # degree = len(neighbors) + 1
        @printf("Neighbors: %s | Tangents: %s\n", length(neighbors), num_tangents)
        # model.addConstr(sum(neighbors) == num_tangents)
        ne_con = @constraint(model, sum(values(neighbors)) == num_tangents)
        set_name(ne_con, "NeCon[$(Tuple(edge)), $node]")

        if length(neighbors) == num_tangents
            for z_var in values(neighbors)
                fix(z_var, 1; force = true)
                # set_lower_bound(z_var, 1)
                # set_upper_bound(z_var, 1)
            end
        end
    end

    # ! contact/collision constraint (MI-Quadratic)
    contact_vars = Dict()
    for ((edge1, node1), (edge2, node2)) in subsets(collect(keys(x_vars)), 2)
        if edge1 == edge2
            @assert node1 != node2
            continue
        end
        var1, var2 = x_vars[edge1, node1], x_vars[edge2, node2]
        difference = var2 - var1
        radius_distance = edges[edge1]["radius"] + edges[edge2]["radius"]

        if node1 == node2
            pair = Set([edge1, edge2])
            z_var = z_vars[pair, node1]
            # ! contact constraint
            contact_con = @constraint(model, sum(difference .* difference) <= (radius_distance + contact_tolerance)^2 + (1 - z_var) * max_distance^2)
            set_name(contact_con, "ContactCon[$(Tuple(edge1)),$node1,$(Tuple(edge2))]")

            # ! collision constraint
            other1 = get_other(edge1, node1)
            for l1 in enumerate_steps(edge1, nodes, edges)
                point1 = (1 - l1) * var1 + (l1 * x_vars[edge1, other1])
                other2 = get_other(edge2, node2)
                for l2 in enumerate_steps(edge2, nodes, edges)
                    point2 = (1 - l2) * var2 + (l2 * x_vars[edge2, other2])
                    difference = point2 - point1
                    # Only neighbors
                    collision_con = @constraint(model, sum(difference .* difference) >= (radius_distance + buffer_tolerance)^2)
                    set_name(collision_con, "CollisionCon[$(Tuple(edge1)),$node1,$(Tuple(edge2))]")

            # contact_vars[edge1, edge2, node1] = (z_var, l1_var, point1_var, l2_var, point2_var)
                end
            end
        end
    end

    ###################################

    status=optimize!(model) # time to optimize!
    @printf("Objective: %s | # Solutions: %d | Status: %s", objective_value(model), result_count(model), termination_status(model))

    if result_count(model) <= 0 || !has_values(model)
        error("The model was not solved correctly.")
        return
    end

    for (edge, neighbors) in z_var_from_edge
        println("Edge $(Tuple(edge)) : neighbors $(value.(values(neighbors)))")
    end

    # * draw contact lines
    # for ((edge1, edge2, node), (z_var, l1_var, point1_var, l2_var, point2_var)) in contact_vars
    #     if value.(z_var) != 1
    #         continue
    #     end
    #     distance = sum(edges[edge]["radius"] for edge in [edge1, edge2])
    #     point1 = value.(point1_var)
    #     point2 = value.(point2_var)
    #     # 2. * edges[edge]['radius']
    #     draw_point(point1, color=GREEN)
    #     draw_point(point2, color=GREEN)
    #     add_line(point1, point2, color=GREEN)
    #     print(str_from_object(pair), value.(l1_var), value.(l2_var),
    #           radius_distance, norm(point1-point2))

    solution = Dict((edge, node) => value.(var)
                for ((edge, node), var) in x_vars)
    println(solution)
    visualize_solution(edges, solution)
end

function main(viewer=true)
    # file_name = "cube_skeleton.json"
    # file_name = "2_tets.json"
    file_name = "single_tet_point2triangle.json"

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

    try
        PyPb.connect(use_gui=true)

        handles = PyPb.draw_pose(PyPb.Pose(), length=1)
        append!(handles, [PyPb.add_line(nodes[n1]["point"], nodes[n2]["point"], color=PyPb.RED) for (n1, n2) in keys(edges)])
        center_viewer(nodes)

        solve(nodes, edges, aabb)
        println(GREEN_FG, "Solution found!")
        PyPb.wait_for_user("Finished.")
    finally
        PyPb.disconnect()
    end
end

#############################################

parser = ArgParseSettings()
@add_arg_table! parser begin
    # "name"
	#     help = "Name to greet"
  	#     arg_type = String
    "--viewer"
        help = "Enable pybullet viewer"
        arg_type = Bool
        default = false
end

args = parse_args(parser)

main(args["viewer"])

    # plotting
    # http://juliaplots.org/MakieReferenceImages/gallery//fluctuation_3d/index.html
    # scene = Scene()
