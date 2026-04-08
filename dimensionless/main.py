# main.py
import pyomo.environ as pyo
import pyomo.dae as dae

import params_initialization
import model.pore_bvp_single as bvp_single
import model.pore_bvp_joint as bvp_joint
from model.reactor_ivp import add_reactor_odes
from model.utils import initialize_concentration_profiles

def build_reactor_model(immobilization='co-immobilization', decay_coef={'kA':0, 'kB':0}, **kwargs):
    model = pyo.ConcreteModel()
    
    # Model parameters indexing - one stage, 3 substrates
    model.Stage      = pyo.Set(initialize=[1])  # Single stage
    model.Components = pyo.Set(initialize=['S1', 'S2', 'S3'])
    
    # 1. Load parameters
    model = params_initialization.load_parameters(model)  # load_parameters.py
    
    # Add Reference Concentration
    model.S_ref = pyo.Param(initialize=model.S_initial['S1'])
    
    # 2a. Change spatial domain to xi (0 to 1)
    model.xi = dae.ContinuousSet(bounds=(0, 1.0)) # Replaces model.x
    model.time  = dae.ContinuousSet(bounds=(0,model.tf))
    
    # 2b. Dimensionless State Variables (u)
    model.u_0 = pyo.Var(model.Components, model.time, domain=pyo.Reals, bounds=(-1e-1, 1.2)) 
    model.du_0dt = dae.DerivativeVar(model.u_0, wrt=model.time) 

    # 2c. Pore-scale variables (for BVP) index order:-> Components, xi, time  
    if immobilization == "single":
        # Pore variables for Particle A (Enzyme A only)
        model.u_n_A = pyo.Var(model.Components, model.xi, model.time, domain=pyo.Reals, bounds=(-1e-1, 1.2))
        model.du_ndxi_A = dae.DerivativeVar(model.u_n_A, wrt=model.xi)
        model.d2u_ndxi2_A = dae.DerivativeVar(model.du_ndxi_A, wrt=model.xi)

        # Variables for Particle B (Enzyme B only)
        model.u_n_B = pyo.Var(model.Components, model.xi, model.time, domain=pyo.Reals, bounds=(-1e-1, 1.2))
        model.du_ndxi_B = dae.DerivativeVar(model.u_n_B, wrt=model.xi)
        model.d2u_ndxi2_B = dae.DerivativeVar(model.du_ndxi_B, wrt=model.xi)    
        
    elif immobilization == "co-immobilization":
        model.u_n = pyo.Var(model.Components, model.xi, model.time, domain=pyo.Reals, bounds=(-1e-1, 1.1))
        model.du_ndxi = dae.DerivativeVar(model.u_n, wrt=model.xi)
        model.d2u_ndxi2 = dae.DerivativeVar(model.du_ndxi, wrt=model.xi)

    # 3. Discretize time and space (xi) dimensions
    print("Discretizing model...")
    discretizer1 = pyo.TransformationFactory('dae.collocation')
    discretizer1.apply_to(model, wrt=model.time, nfe=40, ncp=2, scheme='LAGRANGE-RADAU')
    discretizer1.apply_to(model, wrt=model.xi, nfe=40, ncp=3, scheme='LAGRANGE-RADAU') # Increase if needed
    
    # discretizer2 = pyo.TransformationFactory('dae.finite_difference')
    # discretizer2.apply_to(model, wrt=model.time, nfe=200, scheme='BACKWARD')
    # discretizer2.apply_to(model, wrt=model.xi, nfe=100, scheme='BACKWARD')
    
    # 4. Add constraints and objectives
    bvp_kwargs = kwargs.get('bvp_kwargs')
    if immobilization == 'single':
        bvp_single.add_bvp_constraints(model, decay_coef=decay_coef, bvp_kwargs=bvp_kwargs)
    elif immobilization == 'co-immobilization':
        bvp_joint.add_bvp_constraints(model, decay_coef=decay_coef, bvp_kwargs=bvp_kwargs)
    add_reactor_odes(model)
    
    # 5. Apply the Pore Initial Guess
    print("Applying initial guesses for BVP...")
    # Constant line for S_0 initial guess
    # for t in model.time:
    #     model.u_0['S1', t].set_value(0.5) # Example initial S1 bulk/2
    #     model.u_0['S2', t].set_value(1e-2)
    #     model.u_0['S3', t].set_value(1e-2)  
    
    # Warm profile for S_0 and S_n initial guess
    initialize_concentration_profiles(model, immobilization='co-immobilization')
    
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
        'adjust_Np': True,
        'enzymeA': {
            'kinetics_type': 'first_order',
            'start': 0, 
            'end': 1, 
            'x_step_up': 0.0, 
            'x_step_down': 0.4, 
            'smoothness': 300
        },
        'enzymeB': {
            'kinetics_type': 'first_order',
            'start': 0, 
            'end': 1,
            'x_step_up': 0.6, 
            'x_step_down': 1.0,
            'smoothness': 300
        }
    }
    try:
        print("Building test model...")
        # decay_coef = {'kA': 0.0018, 'kB': 0.0024} # Enzyme A, Enzyme B decay kinetics coefficient
        decay_coef = {'kA': 0.000, 'kB': 0.000}
        test_model = build_reactor_model(immobilization='co-immobilization', 
                                         decay_coef=decay_coef, bvp_kwargs=bvp_kwargs_joint)
        
        print("Solving test model...")
        # solved_model, solver_results = solve_model_robust(test_model, max_iter=200, tol=1e-4, verbose=True)
        solved_model, solver_results = solve_model(test_model)
        
        # Print some basic results
        if solver_results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("[DONE] Optimization successful!")
            # Result print
            print("Final S2 yield:", test_model.u_0['S2',test_model.time.last()]()/test_model.u_0['S1',test_model.time.first()]())
            print("Final S3 yield:", test_model.u_0['S3',test_model.time.last()]()/test_model.u_0['S1',test_model.time.first()]())
            
            # Generate individual plot
            print("\n1. Plotting enzyme profiles...") 
            m_viz.plot_enzyme_pore_profiles(solved_model,immobilization='co-immobilization') 
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
