using JuMP
using Ipopt
using LinearAlgebra: norm

#########################################
# original units of millimeters
const SCALE = 1e-2
const COORDINATES = ["x", "y", "z"]

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

#############################################

# ported from pddlstream.utils
# https://github.com/caelan/pddlstream/blob/stable/pddlstream/utils.py

"""
find an unique element in a sequence satisfying a `test`
"""
function find_unique(test_fn, sequence)
    found, value = false, nothing
    for item in sequence
        if test_fn(item)
            if found
                throw(error("Both elements $value and $item satisfy the test"))
            end
            found, value = true, item
        end
    end
    if !found
        throw(error("Unable to find an element satisfying the test"))
    end
    return value
end

#############################################

function convex_combination(x1, x2, w=0.5)
    return (1-w) * x1 + w * x2
end

"""
Compute closest line between two line segments, using an unconstrained quadratic optimization
"""
function closest_point_segments(line1, line2; optimize=true)
    if optimize
        m = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
        # m = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => false))
        @variable(m, 0 <= γ1 <= 1)
        @variable(m, 0 <= γ2 <= 1)
        @objective(m, Min, sum((convex_combination(line1..., γ1) - convex_combination(line2..., γ2)).^2))

        optimize!(m)
        # distance = objective_value(m)
        g = [value.(γ1), value.(γ2)]
    else
        # TODO we directly derive the gradient
        ΔX1 = line1[2] - line1[1]
        ΔX2 = line2[2] - line2[1]
        b = line1[1] - line2[1]
        A = [ΔX1 -ΔX2]
        g = A'*A \ (A'*b)
    end
    return convex_combination(line1..., g[1]), convex_combination(line2..., g[2])
end

function normalize_vector(v)
    vnorm = norm(v)
    if vnorm < eps()
        @warn "normalize vector with norm close to zero"
        vnorm = eps()
    end
    return v / vnorm
end

function get_edge_direction(edge::AbstractArray, nodes::AbstractMatrix)
    v0, v1 = edge
    p0, p1 = nodes[v0,:], nodes[v1,:]
    return normalize_vector(p1 - p0)
end
