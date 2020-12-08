using LinearAlgebra
using JuMP
using Gurobi

model = Model(with_optimizer(Gurobi.Optimizer))
# set_optimizer_attribute(model, "NonConvex", 2)

# Create variables
@variable(model, x) # define variable x
@variable(model, y) # define variable x
@variable(model, z, base_name="z") # define variable x

# Set objective: x
@objective(model, Max, 1.0*x)

# Add constraint: x + y + z = 1
@constraint(model, lcon, x + y + z == 1.0)

# Add second-order cone: x^2 + y^2 <= z^2
# https://jump.dev/JuMP.jl/stable/constraints/#Quadratic-constraints-1
@constraint(model, qcon1, [z, x, y] in SecondOrderCone())
# @constraint(model, qcon1, x^2 + y^2 <= z^2)

# Add rotated cone: x^2 <= yz
@constraint(model, qcon2, [y, z, x, 0] in RotatedSecondOrderCone())
# @constraint(model, qcon2, x^2 <= y*z)

# # Run the optimizer
# # -----------------

status=optimize!(model) # time to optimize!
termination_status(model)

# Let us look at the important outputs
# ------------------------------------
println("******************************************************")
println("optimal objective value is = ", objective_value(model))
println("optimal x is = ",  value.(x))
println("optimal y is =", value.(y))
println("optimal y is =", value.(z))
