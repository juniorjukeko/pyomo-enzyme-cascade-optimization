import pyomo.environ as pyo
import pyomo.dae as dae
import numpy as np
import params_initialization
import model.pore_bvp_single as bvp_single
import model.pore_bvp_joint as bvp_joint
from model.reactor_ivp import add_reactor_odes
from model.utils import initialize_pore_profiles

def build_reactor_model(immobilization='co-immobilization', decay_coef={'kA':0, 'kB':0}, **kwargs):
    model = pyo.ConcreteModel()
    
    # Model parameters indexing - one stage, 3 substrates
    model.Stage      = pyo.Set(initialize=[1])  # Single stage
    model.Components = pyo.Set(initialize=['S1', 'S2', 'S3']) # Substrate components

    # 1. Load parameters
    model = params_initialization.load_parameters(model)  # load_parameters.py

    # 2a. Establish model time and space domains (independent vars.)
    model.time  = dae.ContinuousSet(bounds=(0,model.tf))     # Reaction time
    model.x     = dae.ContinuousSet(bounds=(0, model.L))     # Pore x-spatial dimension
  
    # 2b. State variables (for IVP) index order:-> Components, time
    model.S_0 = pyo.Var(model.Components, model.time, domain=pyo.Reals, bounds=(-50, 2000))
    model.dS_0dt = dae.DerivativeVar(model.S_0, wrt=model.time) # first-derivative of S_0 on time

    # 2c. Pore-scale variables (for BVP) index order:-> Components, x, time   
    if immobilization == "single":
        # Pore variables for Particle A (Enzyme A only)
        model.S_n_A = pyo.Var(model.Components, model.x, model.time, domain=pyo.Reals, bounds=(-50, 2000))
        model.dS_ndx_A = dae.DerivativeVar(model.S_n_A, wrt=model.x)
        model.d2S_ndx2_A = dae.DerivativeVar(model.dS_ndx_A, wrt=model.x)

        # Pore variables for Particle B (Enzyme B only)
        model.S_n_B = pyo.Var(model.Components, model.x, model.time, domain=pyo.Reals, bounds=(-50, 2000))
        model.dS_ndx_B = dae.DerivativeVar(model.S_n_B, wrt=model.x)
        model.d2S_ndx2_B = dae.DerivativeVar(model.dS_ndx_B, wrt=model.x)
        
    elif immobilization == "co-immobilization":
        model.S_n = pyo.Var(model.Components, model.x, model.time, domain=pyo.Reals, bounds=(-50, 2000))  # S_n, Pore concentration S = f(i, x, t)
        model.dS_ndx   = dae.DerivativeVar(model.S_n, wrt=model.x)      # first-derivative of S_n
        model.d2S_ndx2 = dae.DerivativeVar(model.dS_ndx, wrt=model.x)   # second-derivative of S_n

    # 3. Discretize time and space (x) dimensions
    print("Discretizing model...")
    discretizer_col = pyo.TransformationFactory('dae.collocation')
    discretizer_col.apply_to(model, wrt=model.time, nfe=60, ncp=2, scheme='LAGRANGE-RADAU')
    discretizer_col.apply_to(model, wrt=model.x, nfe=10, ncp=3, scheme='LAGRANGE-RADAU')
    
    # discretizer_fd = pyo.TransformationFactory('dae.finite_difference')
    # discretizer_fd.apply_to(model, wrt=model.time, nfe=100)
    # discretizer_fd.apply_to(model, wrt=model.x, nfe=50)

    # 4. Add constraints and objectives
    bvp_kwargs = kwargs.get('bvp_kwargs')
    if immobilization == 'single':
        bvp_single.add_bvp_constraints(model, decay_coef=decay_coef, bvp_kwargs=bvp_kwargs)
    elif immobilization == 'co-immobilization':
        bvp_joint.add_bvp_constraints(model, immobilization=immobilization, decay_coef=decay_coef, bvp_kwargs=bvp_kwargs)
    else:
        raise ValueError(f"Unknown immobilization scheme: {immobilization}")
        
    add_reactor_odes(model)
    
    # 5. Apply the Pore Initial Guess
    print("Applying initial guesses for BVP...")
    # Constant line for S_0 initial guess
    for c in model.Components:
        # Fetch the initial concentration value
        # Note: If your params_initialization.py defines S_initial with a Stage index, 
        # change this to: init_val = pyo.value(model.S_initial[1, c])
        init_val = pyo.value(model.S_initial[c]) 
        for t in model.time:
            model.S_0[c, t].set_value(init_val)
            
            if immobilization == 'co-immobilization':
                model.S_n[c, 0, t].set_value(init_val)
            
            elif immobilization == 'single':
                model.S_n_A[c, 0, t].set_value(init_val)
                model.S_n_B[c, 0, t].set_value(init_val)

    # Warm profile for S_0 and S_n initial guess
    # initialize_pore_profiles(model, immobilization=immobilization)
            
    return model

if __name__ == "__main__":
    from model.solve import solve_model_robust, solve_model
    import visualization.model_viz as m_viz
    
    # Testing with default configurations
    bvp_kwargs_single = {
        'default_fun': 'linear', 
        'adjust_Np': True,
        'enzymeA': {
            'kinetics_type': 'first_order',
            'start': 0.5, 
            'end': 0.5, 
        },
        'enzymeB': {
            'kinetics_type': 'first_order',
            'start': 0.5, 
            'end': 0.5,
        }
    }
    bvp_kwargs_joint = {
        'default_fun': 'step', 
        'adjust_Np': False,
        'enzymeA': {
            'kinetics_type': 'first_order',
            'start': 0, 
            'end': 1, 
            'x_step_up': 0.0, 
            'x_step_down': 0.4, 
            'smoothness': 200
        },
        'enzymeB': {
            'kinetics_type': 'first_order',
            'start': 0, 
            'end': 1,
            'x_step_up': 0.7, 
            'x_step_down': 1.0,
            'smoothness': 200
        }
    }
    
    try:
        print("Building test model...")
        # decay_coef = {'kA': 0.00, 'kB': 0.00}
        decay_coef = {'kA': 0.0018, 'kB': 0.0024}
        
        # Select immobilization type here: 'single' or 'co-immobilization'
        test_type = 'co-immobilization'
        bvp_kwargs = bvp_kwargs_single if test_type == 'single' else bvp_kwargs_joint
        
        test_model = build_reactor_model(immobilization=test_type, 
                                         decay_coef=decay_coef, 
                                         bvp_kwargs=bvp_kwargs)
        
        print("Solving test model...")
        solved_model, solver_results = solve_model_robust(test_model, max_iter=200, tol=1e-4, verbose=True)
        # solved_model, solver_results = solve_model(test_model)
        
        # Print some basic results
        if solver_results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("[DONE] Optimization successful!")
            # Result print
            print("Final S2 yield:", test_model.S_0['S2',test_model.time.last()]()/test_model.S_0['S1',test_model.time.first()]())
            print("Final S3 yield:", test_model.S_0['S3',test_model.time.last()]()/test_model.S_0['S1',test_model.time.first()]())
            
            # Generate individual plot
            print("\n1. Plotting enzyme profiles...") 
            m_viz.plot_enzyme_pore_profiles(solved_model, immobilization=test_type) 
            m_viz.plot_enzyme_decay_profiles(solved_model, decay_coef) 
            print("\n2. Plotting substrate concentrations...") 
            m_viz.plot_substrate_time_profiles(solved_model) 
            
        else:
            print("[FAIL] Optimization failed or did not converge optimally")
            print(f"Termination condition: {solver_results.solver.termination_condition}")
            
    except ImportError as e:
        print(f"Could not import model builder: {e}")
        print("This is normal if model.py is not in the same directory.")
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("Test completed.")