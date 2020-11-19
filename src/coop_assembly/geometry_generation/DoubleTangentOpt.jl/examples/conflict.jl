using JuMP
using Gurobi
using JuMP
using Gurobi
using CPLEX

# ! You must use a solver that supports conflict refining/IIS computation, like CPLEX or Gurobi
model = Model(with_optimizer(CPLEX.Optimizer))
@variable(model, x >= 0)
@constraint(model, c1, x >= 2)
@constraint(model, c2, x <= 1)
optimize!(model)

# termination_status(model) will likely be MOI.INFEASIBLE,
# depending on the solver

compute_conflict!(model)
if MOI.get(model, MOI.ConflictStatus()) != MOI.CONFLICT_FOUND
    error("No conflict could be found for an infeasible model.")
end

# Both constraints should participate in the conflict.
MOI.get(model, MOI.ConstraintConflictStatus(), c1)
MOI.get(model, MOI.ConstraintConflictStatus(), c2)
