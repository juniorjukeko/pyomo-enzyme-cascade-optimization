# enzyme_ratio_optim.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import pyomo.environ as pyo
from main import build_reactor_model
from model.solve import solve_model, solve_model_robust

def run_enzyme_ratio_study(decay_coef={'kA': 0, 'kB': 0}, 
                          bvp_kwargs_template=None, total_enzyme=10, 
                          EA_range= np.linspace(0.1*10, 0.9*10, 10), 
                          save_results=False):
    """
    DOCS
    """
    
    if bvp_kwargs_template is None:
        bvp_kwargs_template = {
            'default_fun': 'linear',
            'adjust_Np': False,
            'enzymeA': {'fun': 'linear', 'start': 1, 'end': 0},
            'enzymeB': {'fun': 'linear', 'start': 0, 'end': 1}
        }
    
    results = []
    
    for i, EA_max in enumerate(EA_range):
        EB_max = total_enzyme - EA_max
        
        print(f"\n{'='*60}")
        print(f"Running configuration {i+1}/{len(EA_range)} | EA_max: {EA_max:.2f}, EB_max: {EB_max:.2f}")
        
        try:
            # Build a NEW model for each configuration (not cloned)
            test_model = build_reactor_model(immobilization='co-immobilization', 
                                             decay_coef=decay_coef, 
                                             bvp_kwargs=bvp_kwargs_template)
            
            # Set enzyme values BEFORE any expressions are created
            test_model.EA_max.set_value(EA_max)
            test_model.EB_max.set_value(EB_max)
            
            # print(f"DEBUG: EA value: {test_model.EA_max.value}")
            # print(f"DEBUG: EB value: {test_model.EB_max.value}")
            
            # Solve model
            solved_model, solver_results = solve_model_robust(test_model, tol=1e-3, verbose=False)
            # solved_model, solver_results = solve_model(test_model)
            
            if solver_results.solver.termination_condition == pyo.TerminationCondition.optimal:
                # Extract results
                from pyomo.environ import value
                S1_initial = value(solved_model.S_0['S1', solved_model.time.first()])
                S2_final = value(solved_model.S_0['S2', solved_model.time.last()])
                S3_final = value(solved_model.S_0['S3', solved_model.time.last()])
                
                S2_yield = S2_final / S1_initial
                S3_yield = S3_final / S1_initial
                
                # Store results
                config_info = {
                    'config_id': i,
                    'EA_max': EA_max,
                    'EB_max': EB_max,
                    'EA_ratio': EA_max / total_enzyme,
                    'EB_ratio': EB_max / total_enzyme,
                    'S2_yield': S2_yield,
                    'S3_yield': S3_yield,
                    'S2_final': S2_final,
                    'S3_final': S3_final,
                    'converged': True,
                    'solver_status': 'optimal'
                }
                
                print(f"[SUCCESS] Success: S3 yield = {S3_yield:.4f}")
                
            else:
                config_info = {
                    'config_id': i,
                    'EA_max': EA_max,
                    'EB_max': EB_max,
                    'EA_ratio': EA_max / total_enzyme,
                    'EB_ratio': EB_max / total_enzyme,
                    'S2_yield': np.nan,
                    'S3_yield': np.nan,
                    'S2_final': np.nan,
                    'S3_final': np.nan,
                    'converged': False,
                    'solver_status': str(solver_results.solver.termination_condition)
                }
                print(f"[FAIL] Failed: {solver_results.solver.termination_condition}")  
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            config_info = {
                'config_id': i,
                'EA_max': EA_max,
                'EB_max': EB_max,
                'EA_ratio': EA_max / total_enzyme,
                'EB_ratio': EB_max / total_enzyme,
                'S2_yield': np.nan,
                'S3_yield': np.nan,
                'S2_final': np.nan,
                'S3_final': np.nan,
                'converged': False,
                'solver_status': f'Error: {str(e)}'
            }
        
        results.append(config_info)
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    results_df['immobilization'] = 'co-immobilization'
    results_df['decay_kA'] = decay_coef.get('kA', 0)
    results_df['decay_kB'] = decay_coef.get('kB', 0)
    results_df['enzyme_profile'] = str(bvp_kwargs_template)
    
    if save_results:
        filename = f"enzyme_ratio_opt_co_immobilization_single.csv"
        results_df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
    
    return results_df

def compare_profiles_study(profiles_to_compare, decay_coef={'kA': 0, 'kB': 0},
                          total_enzyme=10, EA_range= np.linspace(0.1*10, 0.9*10, 10), 
                          save_results=False):
    """
    DOCS
    """
    
    all_results = []
    
    for profile_name, bvp_kwargs in profiles_to_compare.items():
        print(f"\n{'#'*80}")
        print(f"Running study for profile: {profile_name}")
        print(f"{'#'*80}")
        
        # Use the fixed run_enzyme_ratio_study for each profile
        results_df = run_enzyme_ratio_study(
            decay_coef=decay_coef,
            bvp_kwargs_template=bvp_kwargs,
            total_enzyme=total_enzyme,
            EA_range=EA_range,
            save_results=False  # We'll save the combined results instead
        )
        
        results_df['profile_name'] = profile_name
        all_results.append(results_df)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    if save_results:
        filename = f"enzyme_ratio_opt_co_immobilization_multicomp.csv"
        combined_results.to_csv(filename, index=False)
        print(f"\nCombined results saved to: {filename}")
    
    return combined_results
