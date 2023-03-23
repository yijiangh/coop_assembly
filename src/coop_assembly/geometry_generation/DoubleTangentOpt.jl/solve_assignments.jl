using JSON
using Crayons.Box
using JuMP, KNITRO
using IterTools
import DataStructures: DefaultDict

include("utils.jl")

ROOT_DIR = abspath(joinpath(@__FILE__, [".." for i in 1:5]...))
DATA_DIR = joinpath(ROOT_DIR, "tests", "test_data")
# edge id format
const EDGE_ID = Set

#############################

min_tangents = 2
file_name = "cube_skeleton.json"
# file_name = "truss_one_tet_skeleton.json"
# file_name = "compas_tree_skeleton.json"
# file_name = "robarch_tree_skeleton.json"

problem_name = split(file_name, ".json")[1]
file_path = joinpath(DATA_DIR, file_name)
@assert Base.Filesystem.ispath(file_path)
raw_data = JSON.parsefile(file_path)
graph_data = haskey(raw_data, "data") ? raw_data["data"] : raw_data
println(GREEN_FG("Design graph parsed from $(file_path)!"))

neighbors_from_node = Dict(tryparse(Int,n) => Set(keys(neighbors)) for (n, neighbors) in graph_data["adjacency"])

# * original graph info
edges = Dict(EDGE_ID([tryparse(Int,n1), tryparse(Int,n2)]) => info for (n1, neighbors) in graph_data["edge"] for (n2, info) in neighbors)
for info in values(edges)
    # ! enforce uniform bar radius for now
    info["radius"] = 3.17
    info["radius"] *= SCALE
end
nodes = Dict(tryparse(Int, n) => Dict("point" => parse_point(info)) for (n, info) in graph_data["node"]) # "fixed": info["fixed"]

model = Model(KNITRO.Optimizer)

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

@objective(model, Min, 0)
set_optimizer_attribute(model, "outlev", 1)
status=optimize!(model)

println(termination_status(model))

z_sol = Dict((edge_pair, node) => value.(var)
            for ((edge_pair, node), var) in z_vars)

contact_pairs = []
collision_constr_from_node = DefaultDict(Set)
for (node, neighbors) in adjacent_edge_from_node
    for pair in map(Set, subsets(collect(neighbors), 2))
        push!(collision_constr_from_node[node], pair)
    end
end

for ((edge_pair, node), val) in z_sol
    if val ≈ 1.0
        e1, e2 = edge_pair
        tuple_pair = (collect(e1), collect(e2))
        push!(contact_pairs, tuple_pair)
        if edge_pair in collision_constr_from_node[node]
            pop!(collision_constr_from_node[node], edge_pair)
        end
    end
end

collision_pairs = []
for (node, collision_set) in collision_constr_from_node
    if !isempty(collision_set)
        append!(collision_pairs, map(x -> [collect(v) for v in collect(x)], collect(collision_set)))
    end
end

assignment_data = Dict("contact_pairs" => contact_pairs, "collision_pairs" => collision_pairs)
result_file_path = joinpath(DATA_DIR, problem_name*"_contact_assignment.json")
open(result_file_path,"w") do f 
    JSON.print(f, assignment_data) 
end