using Printf
using JSON
using LinearAlgebra: norm
using JuMP
#
using Gurobi
using Ipopt
#
using Base.Iterators
import DataStructures: DefaultDict
using IterTools
using Statistics

using ArgParse
using Crayons.Box

# set ENV["PYTHON"] = "... path of the python executable ..."
# run Pkg.build("PyCall")
# and relaunch Julia
# OR
# in the conda env, python -c "import julia; julia.install()"
# then relaunch Julia
using PyCall
PyPb = pyimport("pybullet_planning")
PyCoopDataStructures = pyimport("coop_assembly.data_structure")
# PyCompasDataStructures = pyimport("compas.datastructures")

include("utils.jl")
include("visualize.jl")

#####################################

const ROOT_DIR = abspath(joinpath(@__FILE__, [".." for i in 1:5]...))
const DATA_DIR = joinpath(ROOT_DIR, "tests", "test_data")

# original units of millimeters
const SCALE = 1e-2
const COORDINATES = ["x", "y", "z"]

# edge id format
const EDGE_ID = Set

const LENGTH_TOLERANCE = SCALE * 10.0
const BUFFER_TOLERANCE = - SCALE * 0.1
const CONTACT_TOLERANCE = SCALE * 0.1
const MIN_TANGENTS = 2 # | Inf

const OPT_OPTIONS = ["optimize", "check_feasible"]

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

########################################

"""
- edge: EDGE_ID((node_id1, node_id2))
- nodes: Dict( node_id => node info)
- edges: Dict( edge_id => edge info)
- fraction: the fraction of the full bar length to consider
- spacing: spacing ratio, step_size = bar radius if `spacing = 1`
"""
function enumerate_steps(edge, nodes, edge_from_id, fraction=0.1, spacing=1.0) # 1e-3 | 0.1 | 1.5
    step_size = spacing * edge_from_id[edge]["radius"]
    num_steps = ceil(Int, fraction * get_length(edge, nodes) / step_size)
    return range(0., stop=fraction, length=num_steps)
end

##################################################

"""
create an initial guess using the given original graph

Return:
- Dict: (edge, node) => point coordinate
"""
function create_x_hint(nodes, edges, shrink=true)
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

"""
validate and visualize solutions
"""
function visualize_solution(nodes, edges, x_sol, z_sol; alpha=0.25, min_tangents=MIN_TANGENTS)
    render_lock = PyPb.LockRenderer()
    bodies = []
    edge_points = DefaultDict(Vector{Vector})
    for ((edge, node), point) in x_sol
        push!(edge_points[edge], point)
        PyPb.draw_point(point, size=2*edges[edge]["radius"], color=PyPb.BLUE)
        body = PyPb.create_sphere(edges[edge]["radius"], color=PyPb.apply_alpha(PyPb.BLUE, 0.25), mass=PyPb.STATIC_MASS)
        PyPb.set_point(body, point)

        point2 = x_sol[edge, get_other(edge, node)]
        for l in enumerate_steps(edge, nodes, edges)
            trailing = convex_combination(point, point2, l)
            body = PyPb.create_sphere(edges[edge]["radius"], color=PyPb.apply_alpha(PyPb.BLUE, 0.25), mass=PyPb.STATIC_MASS)
            PyPb.set_point(body, trailing)
        end
    end

    # TODO: analyze collisions and proximity
    for (edge, points) in edge_points
        point1, point2 = points
        # println("E#$edge: $point1 | $point2 | L: $(norm(point1-point2))")
        element = create_element(point1, point2, edges[edge]["radius"]; color=PyPb.apply_alpha(PyPb.RED, 0.25))
        push!(bodies, create_element(point1, point2, edges[edge]["radius"], color=PyPb.apply_alpha(PyPb.RED, alpha)))
        PyPb.add_line(point1, point2, color=PyPb.BLUE)
    end

    # * check contact number constraint
    z_sol_from_edge = DefaultDict(Vector)
    for ((pair, node), var) in z_sol
        for (i, edge) in enumerate(pair)
            push!(z_sol_from_edge[edge, node], var)
        end
    end
    for ((edge, node), neighbors) in z_sol_from_edge
        num_tangents = min(length(neighbors), min_tangents)
        # @warn sum(values(neighbors)) == num_tangents "E#$edge-N#$node: #contact neighbors $(sum(values(neighbors))) != required $(num_tangents)"
        if sum(values(neighbors)) != num_tangents
            println(RED_FG("E#$(Tuple(edge))-N#$node: #contact neighbors $(sum(values(neighbors))) != required $(num_tangents)"))
        end
    end

    # * draw contact lines and check contact/collision distance constraint
    for (((edge1, edge2), node), z_var) in z_sol
        contact_pt1, contact_pt2 = closest_point_segments(edge_points[edge1], edge_points[edge2])
        contact_distance = norm(contact_pt1-contact_pt2)
        radius_distance = sum(edges[edge]["radius"] for edge in [edge1, edge2])

        var1, var2 = x_sol[edge1, node], x_sol[edge2, node]
        approx_distance = norm(var2 - var1)

        println(BLUE_FG("(($(Tuple(edge1)), $(Tuple(edge2)), $node) : z_var $z_var"))
        println(BLUE_FG("End_pt distance $(approx_distance) | Accurate contact distance $(contact_distance)"))
        if z_var ≈ 0.0
            # ! collision constraint
            # collision_con = @constraint(model, sum(difference .* difference) >= (radius_distance + buffer_tolerance)^2)
            # @assert
            # @warn contact_distance ≥ (radius_distance + BUFFER_TOLERANCE) "contact_distance $(contact_distance) should be bigger than radius_distance $(radius_distance)!"
            if !(contact_distance ≥ (radius_distance + BUFFER_TOLERANCE))
                println(RED_FG("contact_distance $(contact_distance) should be bigger than radius_distance $(radius_distance)!"))
            end
            continue
        else
            # ! contact constraint
            # contact_con = @constraint(model, sum(difference .* difference) <= (radius_distance + contact_tolerance)^2 + (1 - z_var) * max_distance^2)
            # @warn (radius_distance + BUFFER_TOLERANCE) ≤ contact_distance ≤ (radius_distance + CONTACT_TOLERANCE) "contact_distance $(contact_distance) should be with range [$(radius_distance + BUFFER_TOLERANCE), $(radius_distance + CONTACT_TOLERANCE)]!"
            if !((radius_distance + BUFFER_TOLERANCE) ≤ contact_distance ≤ (radius_distance + CONTACT_TOLERANCE))
                println(RED_FG("contact_distance $(contact_distance) should be with range [$(radius_distance + BUFFER_TOLERANCE), $(radius_distance + CONTACT_TOLERANCE)]!"))
            end

            PyPb.draw_point(contact_pt1, color=PyPb.GREEN)
            PyPb.draw_point(contact_pt2, color=PyPb.GREEN)
            PyPb.add_line(contact_pt1, contact_pt2, color=PyPb.GREEN)
            # print(str_from_object(pair), value.(l1_var), value.(l2_var),
            #       radius_distance, norm(point1-point2))
        end
    end
    render_lock.restore()
end

#########################################

"""
- length_tolerance: tolerance for bar's adherence to the original graph edge length
- contact_tolerance : eps added to the contact constraint
- buffer_tolerance : eps added to the collision constraint
"""
function solve(nodes, edges, aabb;
               opt_option="optimize",
               hint_x_solution=nothing,
               hint_z_solution=nothing,
               diagnose=true,
               min_tangents=MIN_TANGENTS,
               length_tolerance=LENGTH_TOLERANCE, contact_tolerance=CONTACT_TOLERANCE, buffer_tolerance=BUFFER_TOLERANCE,
               num_solutions=1, max_time=1*60, verbose=true)
    println("-"^20)
    max_distance = norm(aabb[2] - aabb[1])

    @assert opt_option in OPT_OPTIONS
    optimize = opt_option=="optimize" ? true : false

    if hint_x_solution === nothing
        hint_x_solution = create_x_hint(nodes, edges)
    end

    # * Define the model
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
            vars = @variable(model, [i=1:3],
                lower_bound=aabb[1][i], upper_bound=aabb[2][i],
                base_name="x[($(edge),$node]",
                start=hint_x_solution[edge, node][i])
            difference = vars .- nodes[node]["point"]
            x_vars[edge, node] = vars
            objective[edge, node] = sum(difference.*difference)
        end

        # difference = x_vars[edge, node2] - x_vars[edge, node1]
        # if length_tolerance < Inf
        #     # TODO: make length_tolerance a function of the radius
        #     @constraint(model, sum(difference.^2) >= (length - length_tolerance)^2)
        #     @constraint(model, sum(difference.^2) <= (length + length_tolerance)^2)
        # end
    end

    if optimize
        println(BLUE_FG("Optimizing model."))
        @objective(model, Min, sum(values(objective)))
    else
        println(YELLOW_FG("Checking feasiblity."))
        # https://github.com/jump-dev/JuMP.jl/issues/693#issuecomment-272638604
        @objective(model, Min, 0)
    end

    # collect connectivity info
    adjacent_edge_from_node = DefaultDict(Set)
    for edge in keys(edges)
        for node in edge
            push!(adjacent_edge_from_node[node], edge)
        end
    end
    # @show adjacent_edge_from_node

    # * create z_variables
    z_vars = Dict()
    for (node, neighbors) in adjacent_edge_from_node
        # num_tangents = min(len(neighbors) - 1, 2)
        # print(len(edges), num_tangents)
        for pair in map(Set, subsets(collect(neighbors), 2))
            z_vars[pair, node] = @variable(model, binary=true, base_name="z[$(pair), $(node)]")
        end
    end

    # * Constraints
    # ! contact degree constraint (linear)
    z_var_from_edge = DefaultDict(Vector)
    for ((pair, node), var) in z_vars
        for (i, edge) in enumerate(pair)
            push!(z_var_from_edge[edge, node], var)
        end
    end
    for ((edge, node), neighbors) in z_var_from_edge
        num_tangents = min(length(neighbors), min_tangents)
        # degree = len(neighbors) + 1
        println("E#$(Tuple(edge))-N$(node) : Neighbors: $(length(neighbors)) | Tangents: $(num_tangents)")

        ne_con = @constraint(model, sum(values(neighbors)) == num_tangents)
        set_name(ne_con, "NeCon[$(edge), $node]")

        # if only has two neighboring bars, must touch both of them
        if length(neighbors) == num_tangents
            for z_var in values(neighbors)
                fix(z_var, 1; force = true)
                @assert is_fixed(z_var)
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
        # ! use end point to approximate
        # TODO: introduce contact point parameterization
        var1, var2 = x_vars[edge1, node1], x_vars[edge2, node2]
        difference = var2 - var1
        radius_distance = edges[edge1]["radius"] + edges[edge2]["radius"]

        if node1 == node2
            pair = Set([edge1, edge2])
            z_var = z_vars[pair, node1]
            # ! contact constraint
            contact_con = @constraint(model, sum(difference .* difference) <= (radius_distance + contact_tolerance)^2 + (1 - z_var) * max_distance^2)
            set_name(contact_con, "ContactCon[$(edge1),$node1,$(edge2)]")

            # ! collision constraint
            other1 = get_other(edge1, node1)
            for l1 in enumerate_steps(edge1, nodes, edges)
                # point1 = (1 - l1) * var1 + (l1 * x_vars[edge1, other1])
                point1 = convex_combination(var1, x_vars[edge1, other1], l1)
                other2 = get_other(edge2, node2)
                for l2 in enumerate_steps(edge2, nodes, edges)
                    # point2 = (1 - l2) * var2 + (l2 * x_vars[edge2, other2])
                    point2 = convex_combination(var2, x_vars[edge2, other2], l2)
                    # Only neighbors
                    difference = point2 - point1
                    collision_con = @constraint(model, sum(difference .* difference) >= (radius_distance + buffer_tolerance)^2)
                    set_name(collision_con, "CollisionCon[$(edge1),$node1,$(edge2)]")

                    # contact_vars[edge1, edge2, node1] = (z_var, l1_var, point1_var, l2_var, point2_var)
                    contact_vars[edge1, edge2, node1] = z_var
                end
            end
        end
    end

    if !optimize
        if hint_x_solution !== nothing
            for ((edge, node), x_var) in x_vars
                xf_con = @constraint(model, x_vars[edge, node] .== hint_x_solution[edge, node])
            end
        end
        if hint_z_solution !== nothing
            for ((pair, node), z_var) in z_vars
                zf_con = @constraint(model, z_vars[pair, node] .== hint_z_solution[pair, node])
            end
        end
    end

    ###################################

    println("="^20)
    status=optimize!(model) # time to optimize!
    println("="^20)

    @printf("Status: %s\n", termination_status(model))
    if result_count(model) <= 0 || termination_status(model) == MOI.INFEASIBLE # !has_values(model)
        @error "The model was not solved correctly."

        if diagnose
            compute_conflict!(model)
            @show MOI.ConflictStatus()
            if MOI.get(model, MOI.ConflictStatus()) != MOI.CONFLICT_FOUND
                error("No conflict could be found for an infeasible model.")
            end
            @show MOI.ConstraintConflictStatus()
            # Both constraints should participate in the conflict.
            # MOI.get(model, MOI.ConstraintConflictStatus(), c1)
            # MOI.get(model, MOI.ConstraintConflictStatus(), c2)
        end

        return nothing, nothing
    else
        @printf("Objective: %s | # Solutions: %d", objective_value(model), result_count(model))
    end

    # for (edge, neighbors) in z_var_from_edge
    #     println("Edge $(EDGE_ID(edge)) : neighbors $(value.(values(neighbors)))")
    # end

    x_sol = Dict((edge, node) => value.(var)
                for ((edge, node), var) in x_vars)

    z_sol = Dict((edge_pair, node) => value.(var)
                for ((edge_pair, node), var) in z_vars)

    println("X solution: ", x_sol)
    println("Z solution: ", z_sol)
    return x_sol, z_sol
end

function main(file_name, args)
    graph_data = nothing
    saved_x_vars = nothing
    saved_z_vars = nothing
    sol_b_struct = nothing

    if args["check_feasible_file"] !== nothing
        saved_sol_path = joinpath(DATA_DIR, args["check_feasible_file"])
        sol_data = JSON.parsefile(saved_sol_path)
        println(GREEN_FG("Solution parsed from $(saved_sol_path)!"))

        graph_data = sol_data["overall_structure"]
        sol_b_struct = PyCoopDataStructures.BarStructure.from_data(sol_data["bar_structure"]["data"])
        #
        saved_x_vars = Dict()
        for data in sol_data["opt_data"]["x_vars"]
            # (edge, node) => value.(var)
            edge, node = data["key"]
            saved_x_vars[EDGE_ID(edge), node] = SCALE * data["value"]
        end
        saved_z_vars = Dict()
        for data in sol_data["opt_data"]["z_vars"]
            # (edge_pair, node) => value.(var)
            (e1, e2), node = data["key"]
            saved_z_vars[EDGE_ID((EDGE_ID(e1), EDGE_ID(e2))), node] = data["value"]
        end
    else
        file_path = joinpath(DATA_DIR, file_name)
        @assert Base.Filesystem.ispath(file_path)
        graph_data = JSON.parsefile(file_path)
        println(GREEN_FG("Design graph parsed from $(file_path)!"))
    end

    # bar_structure: adjacency, attributes, edge, node
    # overall_structure: adjacency, edge, node
    neighbors_from_node = Dict(tryparse(Int,n) => Set(keys(neighbors)) for (n, neighbors) in graph_data["adjacency"])

    # * original graph info
    edges = Dict(EDGE_ID([tryparse(Int,n1), tryparse(Int,n2)]) => info for (n1, neighbors) in graph_data["edge"] for (n2, info) in neighbors)
    for info in values(edges)
        # ! enforce uniform bar radius for now
        info["radius"] = 3.17
        info["radius"] *= SCALE
    end
    nodes = Dict(tryparse(Int, n) => Dict("point" => parse_point(info)) for (n, info) in graph_data["node"]) # "fixed": info["fixed"]

    # compute AABB for box domain for x vars
    points = [info["point"] for info in values(nodes)]
    aabb = PyPb.aabb_from_points(points)
    bb_scale = 2.
    center = PyPb.get_aabb_center(aabb)
    extent = PyPb.get_aabb_extent(aabb)
    aabb = PyPb.AABB(lower=center - bb_scale*extent/2, upper=center + bb_scale*extent/2)
    println("aabb: ", aabb)

    viewer = args["viewer"]
    try
        PyPb.connect(use_gui=viewer)

        render_lock = PyPb.LockRenderer()
        handles = PyPb.draw_pose(PyPb.Pose(), length=1)
        append!(handles, [PyPb.add_line(nodes[n1]["point"], nodes[n2]["point"], color=PyPb.RED) for (n1, n2) in keys(edges)])
        center_viewer(nodes)
        render_lock.restore()

        if args["check_feasible_file"] !== nothing
            # parsed a saved solution
            # element_from_index = sol_b_struct.get_element_from_index(scale=SCALE, color=PyPb.apply_alpha(PyPb.RED, 0.3)) # |indices=chosen_bars,

            # sol_b_struct.get_element_bodies(color=PyPb.RED, regenerate=true, scale=SCALE)
            # contact_from_connectors = sol_b_struct.get_connectors(scale=SCALE)
            # grounded_elements = sol_b_struct.get_grounded_bar_keys()
            # sol_b_struct.set_body_color(PyPb.RED) #, indices=chosen_bars
            # TODO convert parsed solution to x_vars and z_vars

            visualize_solution(nodes, edges, saved_x_vars, saved_z_vars)
            PyPb.wait_if_gui("Solution reconstructed.")
            # return
        end

        x_sol, z_sol = solve(nodes, edges, aabb;
            opt_option=args["opt_option"],
            num_solutions=args["num_solutions"],
            diagnose=args["diagnose"],
            hint_x_solution=saved_x_vars,
            hint_z_solution=saved_z_vars)

        if x_sol !== nothing && z_sol !== nothing
            println(GREEN_FG("Solution found!"))
            visualize_solution(nodes, edges, x_sol, z_sol)
        else
            println(RED_FG("No solution found!"))
        end

        PyPb.wait_if_gui("Finished.")
    finally
        PyPb.disconnect()
    end
end

#############################################

parser = ArgParseSettings()
@add_arg_table! parser begin
    "--problem"
	    help = "json file name for the original graph to be solved upon"
        arg_type = String
        default = "truss_one_tet_skeleton.json"
        # file_name = "cube_skeleton.json"
        # file_name = "2_tets.json"
        # file_name = "single_tet_point2triangle.json"
    "--check_feasible_file"
	    help = "json file name for a computed solution, for viz or feasiblity checking"
        arg_type = String
        default = nothing
    "--viewer"
        help = "Enable pybullet viewer"
        action = :store_true
    "--opt_option"
        arg_type = String
        default = "optimize"
    "--diagnose"
        help = "Show diagnosis info if the optimization is terminated due to infeasiblibility"
        action = :store_true
    "--num_solutions"
        help = "Limits the number of feasible MIP solutions found. https://www.gurobi.com/documentation/9.0/refman/solutionlimit.html"
        arg_type = Int
        default = 1
    "--write"
        help = "export the newly found solution"
        action = :store_false
end

args = parse_args(parser)
println(YELLOW_FG("Args: $(args)"))
main(args["problem"], args)
