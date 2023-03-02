using JSON
using Test, SafeTestsets
using Zygote, FiniteDifferences, ForwardDiff
using Nonconvex

const ROOT_DIR = abspath(joinpath(@__FILE__, [".." for i in 1:5]...))
const DATA_DIR = joinpath(ROOT_DIR, "tests", "test_data")

include("beam_dist_energy.jl")

# @testset "BeamDistEnergy_grad" begin
    file_name = "truss_one_tet_skeleton.json"
    file_path = joinpath(DATA_DIR, file_name)
    @assert Base.Filesystem.ispath(file_path)
    graph_data = JSON.parsefile(file_path)
    println("Design graph parsed from $(file_path)!")

    edges_dict = Dict([tryparse(Int,n1), tryparse(Int,n2)] => info for (n1, neighbors) in graph_data["edge"] for (n2, info) in neighbors)
    edges = Matrix{Int}(undef, length(edges_dict), 2)
    for (i, e) in enumerate(keys(edges_dict))
        edges[i,:] = map(x->Int(x)+1, e)
    end

    nodes_dict = Dict(tryparse(Int, n) => parse_point(info) for (n, info) in graph_data["node"]) # "fixed": info["fixed"]
    nodes = Matrix{Float64}(undef, length(nodes_dict), 3)
    for (i, node) in nodes_dict
        nodes[i+1,:] = node
    end
    radius = eltype(nodes)(3.17 * 1e-2)

    beam_dist_energy = BeamDistanceEnergy(edges, nodes, radius)

    x0 = Vector{Float64}(undef, 2*3*length(edges))
    for i in 1:size(edges)[1]
        v0, v1 = edges[i,:]
        x0[i*6-5:i*6-3] = nodes[v0,:]
        x0[i*6-2:i*6] = nodes[v1,:]
    end

    f = x -> beam_dist_energy(x)

    val1, grad1 = NonconvexCore.value_gradient(f, x0)
    val2, grad2 = f(x0), Zygote.gradient(f, x0)[1]
    grad3 = FiniteDifferences.grad(central_fdm(5, 1), f, x0)[1]

    @test val1 .== val2
    @test norm(grad1 - grad2) == 0
    @test norm(grad1 - grad3) <= 1e-8
# end
