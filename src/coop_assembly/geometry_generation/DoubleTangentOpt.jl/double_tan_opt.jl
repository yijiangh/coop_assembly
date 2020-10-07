using JSON
using LinearAlgebra
using JuMP
using Gurobi

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

function parse_point(json_point)
    return SCALE*[json_point[k] for k in COORDINATES]
end

function solve(nodes, edges, aabb;
               min_tangents=2, # 2 | INF
               length_tolerance=SCALE*10, contact_tolerance=SCALE*1, buffer_tolerance=SCALE*0,
               num_solutions=1, max_time=1*60, verbose=true)
    # m = 5
    # n = 10
    # A = randn(m,n)
    # b = randn(m)
    # M = 1
    # k = convert(Int64, round(m/3))

    # # Renaming a bunch of variables
    # S = A'*A
    # c = -2*A'*b
    # d = norm(b)^2

    # # * Define the model
    # # define name of the model, it could be anything, not necessarily "model"
    # model = Model(with_optimizer(Gurobi.Optimizer))

    # # *Variables
    # @variable(model, x[1:n]) # define variable x
    # @variable(model, y[1:n], Bin) # define the binary variable y

    # # *Objective
    # # by this command, we are programatically defining a quadratic objective to be minimized
    # sense = MOI.MIN_SENSE
    # @objective(model, sense, sum(S[i,j]*x[i]*x[j] for i in 1:n, j in 1:n)+ sum(c[i]*x[i] for i in 1:n) + d)

    # # * Constraints
    # @constraint(model, con_lb[i=1:n], -M*y[i] <= x[i]) # lower bound constraint

    # @constraint(model, con_ub[i=1:n], x[i] <= M*y[i]) # upper bound constraint

    # @constraint(model, con_bd_sum, sum(y[i] for i in 1:n) <= k) # cardinality constraint in terms of y
end

function main()
    # file_name = 'cube_skeleton.json'
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

    nodes = Dict(tryparse(Int, n) => Dict("point" => parse_point(info)) for (n, info) in json_data["node"]) # 'fixed': info['fixed']

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
