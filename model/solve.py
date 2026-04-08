# solve.py
import pyomo.environ as pyo
import pyomo.dae as dae

def solve_model(model):
    """
    Discretize and solve the reactor kinetics model.
    
    Parameters
    ----------
    model: Pyomo model to solve
    
    Returns
    ----------
    model: Solved model
    results: Solver results
    """
    print("Solving model with IPOPT...")
    
    # Solve model
    solver = pyo.SolverFactory('ipopt')
    solver.options['tol']               = 1e-4
    solver.options['acceptable_tol']    = 1e-4
    results = solver.solve(model, tee=True)
    
    print(f"Solver termination condition: {results.solver.termination_condition}")
    print(f"Solver status: {results.solver.status}")
    
    return model, results

def solve_model_robust(model, max_iter=200, tol=1e-5, verbose=False):
    """
    Discretize and solve the Pyomo DAE model using IPOPT with robust solver settings.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The Pyomo model containing DAE and algebraic constraints.
    max_iter : int, optional
        Maximum number of solver iterations (default: 200).
    tol : float, optional
        Solver tolerance for convergence (default: 1e-5).
    verbose : bool, optional
        If True, print progress messages and solver logs (default: True).

    Returns
    -------
    model : pyo.ConcreteModel
        The solved model with updated variable values.
    results : SolverResults
        The IPOPT solver result object containing status and termination info.
    """
    if verbose:
        print("Solving model with IPOPT (robust settings)...")

    # Configure IPOPT solver
    solver = pyo.SolverFactory('ipopt')
    solver_options = {
        'max_iter': max_iter,
        'tol': tol,
        'constr_viol_tol': tol,
        'acceptable_tol': tol,
        'acceptable_iter': 5,
        'mu_strategy': 'adaptive',
        'mu_init': 1e-5,
        'honor_original_bounds': 'no',
        'nlp_scaling_method': 'gradient-based',
        'obj_scaling_factor': 1.0,
        'print_level': 5 if verbose else 0,

        'warm_start_init_point': 'yes',
        'warm_start_bound_push': 1e-8,
        'warm_start_mult_bound_push': 1e-8,
        'bound_relax_factor': 1e-6,      
        'start_with_resto': 'no',
        'resto_failure_feasibility_threshold': 1e-2, # Accept slightly infeasible resto steps
        'required_infeasibility_reduction': 0.1,     # Don't demand perfection in resto
        'alpha_for_y': 'bound-mult',                 # Helps with dual infeasibility spikes
    }
    solver.options['nlp_scaling_method'] = 'user-scaling'
    for k, v in solver_options.items():
        solver.options[k] = v

    # solver.options['bound_push']        = 1e-8   # default is 1e-2, too loose
    # solver.options['bound_frac']        = 1e-8
    # solver.options['slack_bound_push']  = 1e-8
    # solver.options['slack_bound_frac']  = 1e-8
    try:
        results = solver.solve(model, tee=verbose)
    except Exception as e:
        if verbose:
            print(f"First solve attempt failed: {e}")
            print("Retrying with more relaxed settings...")

        solver_options.update({
            'max_iter': 100,
            'tol': 1e-3,
            'mu_init': 1e-3,
            'bound_relax_factor': 1e-4,
        })
        for k, v in solver_options.items():
            solver.options[k] = v

        results = solver.solve(model, tee=verbose)

    if verbose:
        print(f"Solver termination condition: {results.solver.termination_condition}")
        print(f"Solver status: {results.solver.status}")

    return model, results
