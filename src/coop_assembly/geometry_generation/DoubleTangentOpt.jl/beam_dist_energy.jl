using UnPack
using LinearAlgebra
import DataStructures: DefaultDict
using ChainRulesCore

include("utils.jl")

mutable struct BeamDistanceEnergy
    np_e_pairs::Matrix{Int}
    np_e_signs::Vector{Int}
    p_e_pairs::Matrix{Int}
    p_e_perp_dirs::Matrix{Float64}
    nB::Int
    nE_np::Int
    nE_p::Int
    nT::Int
    nP::Int
    radius::Float64
end

function BeamDistanceEnergy(
    edges::AbstractMatrix{iT},
    nodes::AbstractMatrix{T},
    radius::T
    ) where {iT, T}
    np_e_pairs, np_e_signs, p_e_pairs, p_e_perp_dirs = compute_offset_beam_constraint_pairs(edges, nodes)
    nB = size(edges,1)
    nE_np = size(np_e_pairs,2)
    nE_p = size(p_e_pairs,2)
    # nT: number of variables for beam's intersection point
    nT = (nE_np + nE_p)*2
    nP = 2*3*nB

    return BeamDistanceEnergy(np_e_pairs, np_e_signs, p_e_pairs, p_e_perp_dirs,
        nB, nE_np, nE_p, nT, nP, radius)
end

function (be::BeamDistanceEnergy)(x::AbstractVector)
    @unpack np_e_pairs, np_e_signs, p_e_pairs, p_e_perp_dirs,
        nB, nE_np, nE_p, nT, nP = be
    T_np = reshape(x[1:nE_np*2], 2, nE_np)
    T_p = reshape(x[nE_np*2+1 : nE_np*2+1 + nE_p*2-1], 2, nE_p)
    P = reshape(x[nT : nT + nP-1], nB, 2, 3)
    drt = P[:,2,:] - P[:,1,:] # nB x 3

    origin = P[:,1,:]
    energy = 0.0
    # non-parallel pairs
    if nE_np > 0
        # np_normal = similar(drt[np_e_pairs[1,:], :])
        # for (i, (d1, d2)) in enumerate(zip(eachcol(drt[np_e_pairs[1,:], :]), eachcol(drt[np_e_pairs[2,:],:])))
        #     np_normal[:,i] = cross(d1,d2)
        # end
        np_normal = hcat(cross.(eachcol(drt[np_e_pairs[1,:], :]), eachcol(drt[np_e_pairs[2,:],:]))...)
        inv_length = 1.0 / map(norm, eachslice(np_normal, dims=2)) .* np_e_signs
        np_normal = Diagonal(inv_length) * np_normal
        p0 = Diagonal(T_np[1,:]) * drt[np_e_pairs[1,:],:] + origin[np_e_pairs[1,:],:]
        p1 = Diagonal(T_np[2,:]) * drt[np_e_pairs[2,:],:] + origin[np_e_pairs[2,:],:]
        contact_delta = p1 - p0 - np_normal * radius * 2
        energy = sum(x->sum(x.^2), eachcol(contact_delta))
    end

    # parallel pairs
    # p_normal = cross.(eachcol(p_e_perp_dirs.T), eachcol(drt[p_e_pairs[1,:],:]))
    # if nE_p > 0
    #     p_normal = similar(drt[p_e_pairs[1,:], :])
    #     for (i, (d1, d2)) in enumerate(zip(eachrow(p_e_perp_dirs), eachcol(drt[p_e_pairs[1,:],:])))
    #         p_normal[:,i] = cross(d1,d2)
    #     end
    #     # p_normal = cross(p_e_perp_dirs.T, drt[p_e_pairs[1,:],:])
    #     inv_length = 1.0 / map(norm, eachslice(p_normal, dims=2))
    #     p_normal = Diagonal(inv_length) * p_normal
    #     p0 = Diagonal(T_p[1,:]) * drt[p_e_pairs[1,:],:] + origin[p_e_pairs[1,:],:]
    #     p1 = Diagonal(T_p[2,:]) * drt[p_e_pairs[2,:],:] + origin[p_e_pairs[2,:],:]
    #     contact_delta = p1 - p0 - p_normal * radius * 2
    #     energy = sum(x->sum(x.^2), eachcol(contact_delta))
    # end
    return energy
end

function ChainRulesCore.rrule(bde::BeamDistanceEnergy, x::Vector)
    return bde(x), Δ -> (NoTangent(), ForwardDiff.gradient(bde, x) * Δ)
end

# function ChainRulesCore.rrule(
#     bde::BeamDistanceEnergy, x::AbstractVector
# ) where {T}
#     function pullback_fn(Δ)
#     end
#     return bde(x), pullback_fn
# end

"""

Returns
-------
- non_parallel_e_pairs
    2 x pair_size int matrix, non parallel connected line pair ids
- non_parallel_e_signs
    pair_size int array, non parallel connected line pair's sign, can take value 1 () or -1 ()
- parallel_e_pairs
    2 x pair_size int matrix: parallel connected line pair ids
- parallel_e_perp_dirs
    3 x pair_size float matrix: parallel connected line pair's perpedicular directions
"""
function compute_offset_beam_constraint_pairs(edges::AbstractMatrix{T}, nodes::AbstractMatrix{iT}) where {iT, T}
    adjacent_edge_from_node = DefaultDict(Set)
    for (ei, edge) in enumerate(eachrow(edges))
        for node in edge
            push!(adjacent_edge_from_node[node], ei)
        end
    end

    non_parallel_e_pairs = []
    non_parallel_e_signs = []
    parallel_e_pairs = []
    parallel_e_perp_dirs = []
    for (vid, v_edge_set) in adjacent_edge_from_node
        v_edges = collect(v_edge_set)
        if length(v_edges) == 1
            # do nothing
        elseif length(v_edges) == 2
            e0, e1 = v_edges
            d0 = get_edge_direction(edges[e0,:], nodes)
            d1 = get_edge_direction(edges[e1,:], nodes)
            if norm(cross(d0, d0)) < eps()
                drt = [.0, .0, 1.0]
                if norm(cross(drt, d0)) < eps()
                    drt = [1.0, .0, .0]
                end
                push!(parallel_e_pairs, [e0, e1])
                push!(parallel_e_perp_dirs, drt)
            else
                push!(non_parallel_e_pairs, [e0, e1])
                push!(non_parallel_e_signs, 1)
            end
        elseif length(v_edges) == 3
            # * valence-3 vertex
            parallel_index = -1
            for kd in eachindex(v_edges)
                e0 = v_edges[kd]
                e1 = v_edges[kd % 3 + 1]
                d0 = get_edge_direction(edges[e0,:], nodes)
                d1 = get_edge_direction(edges[e1,:], nodes)
                # record parallel_index at the first pair of parallel pair
                if norm(cross(d0, d1)) < eps()
                    parallel_index = kd
                    break
                end
            end

            # * none of the pair in the ring is parallel
            if parallel_index == -1
                nr = zeros(3)
                for kd in 1:3
                    e0 = v_edges[kd]
                    e1 = v_edges[kd%3 + 1]
                    d0 = get_edge_direction(edges[e0,:], nodes)
                    d1 = get_edge_direction(edges[e1,:], nodes)
                    n = cross(d0,d1)
                    if dot(nr, n) >= 0
                        push!(non_parallel_e_signs, 1)
                    else
                        push!(non_parallel_e_signs, -1)
                    end
                    push!(non_parallel_e_pairs, [e0,e1])
                    nr = n
                end
            else
                # * at least one pair in the ring is parellel
                e0 = v_edges[parallel_index]
                e1 = v_edges[parallel_index%3+1]
                e2 = v_edges[parallel_index%3+2]
                d0, d1, d2 = map(ex -> get_edge_direction(edges[ex,:], nodes), [e0,e1,e2])
                n = cross(d0,d1)
                n12 = cross(d1,d2)
                n20 = cross(d2,d0)

                push!(non_parallel_e_pairs, [e1,e2])
                push!(non_parallel_e_pairs, [e2,e0])
                push!(non_parallel_e_signs, 1)
                if dot(n12, n20) >= 0
                    push!(non_parallel_e_signs, 1)
                else
                    push!(non_parallel_e_signs, -1)
                end
            end
        else
            @error("Unsupported valence number $(length(v_edges))")
        end
    end
    mat_non_parallel_e_pairs = Matrix{Int}(undef, 2, length(non_parallel_e_pairs))
    for (i, ep) in enumerate(non_parallel_e_pairs)
        mat_non_parallel_e_pairs[:,i] = ep
    end

    mat_parallel_e_pairs = Matrix{Int}(undef, 2, length(parallel_e_pairs))
    for (i, ep) in enumerate(parallel_e_pairs)
        mat_parallel_e_pairs[:,i] = ep
    end

    mat_parallel_e_perp_dirs = Matrix{eltype(parallel_e_perp_dirs)}(undef, 3, length(parallel_e_perp_dirs))
    for (i, ep) in enumerate(parallel_e_perp_dirs)
        mat_parallel_e_perp_dirs[:,i] = ep
    end

    return mat_non_parallel_e_pairs, non_parallel_e_signs, mat_parallel_e_pairs, mat_parallel_e_perp_dirs
end

