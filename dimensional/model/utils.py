import pyomo.environ as pyo

import pyomo.environ as pyo

def get_enzyme_decay_rule(decay_constant):
    """
    Returns a Pyomo rule for enzyme decay over time.
    """
    if decay_constant > 0:
        return lambda m, t: pyo.exp(-decay_constant * t)
    else:
        return lambda m, t: 1.0

def get_enzyme_kinetics(E_density, decay, k_cat, S, Km=None, kinetics_type='first_order'):
    """
    Returns the (LHS_multiplier, RHS_rate) to construct numerically stable PDEs 
    for dimensional modeling.
    Formulation: LHS_multiplier * (D * d2S/dx2) == RHS_rate
    """
    if kinetics_type == 'first_order':
        # Standard First Order: D * d2S/dx2 = E * decay * k * S
        lhs_multiplier = 1.0
        rhs_rate = E_density * decay * k_cat * S
        return lhs_multiplier, rhs_rate
        
    elif kinetics_type == 'michaelis_menten':
        # Michaelis-Menten: D * d2S/dx2 = (E * decay * k * S) / (Km + S)
        if Km is None:
            raise ValueError("Km must be provided for Michaelis-Menten kinetics.")
            
        lhs_multiplier = 1.0
        # The equation safely uses the fraction because dimensional S does not trigger scaling bombs
        rhs_rate = (E_density * decay * k_cat * S) / (Km + S)
        return lhs_multiplier, rhs_rate
        
    else:
        raise ValueError(f"Unknown kinetics type: {kinetics_type}")

def get_enzyme_profile_rule(model, E_max, start=1, end=0, fun='linear', **kwargs):
    """
    Add configurable enzyme distribution
    
    Parameters:
    fun: enzyme density distribution function throughout pore length (density between 0 and maximum enzyme density m.EA & m.EB)
            - linear    : linear function
            - step      : step function (must define x_step as 0 to 1)
    start: 
        - linear    : Density at x=0 as fraction of maximum enzyme density (0 to 1)
        - step      : Baseline enzyme density (before step-up and after step-down) as fraction of maximum enzyme density (0 to 1) 
    end: 
        - linear    : Density at x=L as fraction of maximum enzyme density (0 to 1)
        - step      : Plateau enzyme density (after step-up and before step-down) as fraction of maximum enzyme density (0 to 1)
    kwargs: Additional parameters for specific distributions
        - For 'step': 
            x_step_up: Fraction of L where step up begins (0 to 1)
            x_step_down: Fraction of L where step down begins (0 to 1)
            smoothness: Smoothness factor for transitions (lower = smoother, default=100)
    """
    # Validate start and end parameters
    if (start < 0 or start > 1) or (end < 0 or end > 1):
        raise ValueError("'start' and 'end' must be between 0 and 1")

    if fun == 'linear':
        def profile_rule(m, x):
            return E_max * (start + (end - start) * (x/m.L))
        return pyo.Expression(model.x, rule=profile_rule)
    
    elif fun == 'step':
        x_step_up   = kwargs.get('x_step_up', 0.3)      # Start of step-up transition
        x_step_down = kwargs.get('x_step_down', 0.7)    # Start of step-down transition
        smoothness  = kwargs.get('smoothness', 100.0)   # Smoothness factor
        
        # Validate step parameters
        if x_step_up < 0 or x_step_up > 1:
            raise ValueError("x_step_up must be between 0 and 1")
        if x_step_down < 0 or x_step_down > 1:
            raise ValueError("x_step_down must be between 0 and 1")
        if x_step_up >= x_step_down:
            raise ValueError("x_step_up must be less than x_step_down")

        def profile_rule(m, x):
            x_frac = x / m.L

            # Smooth step-up transition (sigmoid function)
            step_up_transition   = 1.0 / (1.0 + pyo.exp(-smoothness * (x_frac - x_step_up)))  
            
            # Smooth step-down transition (sigmoid function)
            step_down_transition = 1.0 / (1.0 + pyo.exp(-smoothness * (x_frac - x_step_down)))
            
            # Combined profile:
            step_profile = (start * E_max * (1.0 - step_up_transition) +
                            end * E_max * (step_up_transition - step_down_transition) +
                            start * E_max * step_down_transition)
            return step_profile
        return pyo.Expression(model.x, rule=profile_rule)
                
    else:
        raise ValueError(f"Unsupported profile type: {fun}")


def calculate_pore_count_coefficient(model, EA_profile, EB_profile, EA_max, EB_max):
    """
    Normalize pore count so that total enzyme loading (eA + eB) 
    is equal to the reference uniform loading co-immobilization (beta-SID).
    Calculations are based on area under enzyme profile curve.
    """

    x_values = sorted(list(model.x))
    total_EA = 0.0
    total_EB = 0.0
    count = 0

    for x in x_values:
        EA_val = pyo.value(EA_profile[x])
        EB_val = pyo.value(EB_profile[x])
        total_EA += EA_val
        total_EB += EB_val
        count += 1

    if count == 0:
        return 1.0

    avg_EA = total_EA / count
    avg_EB = total_EB / count

    avg_total = avg_EA + avg_EB

    ref_total = pyo.value(EA_max) + pyo.value(EB_max)

    pore_count_coef = ref_total / avg_total if avg_total > 0 else 1.0

    print("Pore count coefficient (EA + EB normalization):")
    print(f"  Avg EA: {avg_EA:.4e} |  Avg EB: {avg_EB:.4e}")
    print(f"  Avg total enzyme: {avg_total:.6e} | Reference total enzyme: {ref_total:.4e}")
    print(f"  Pore count coefficient: {pore_count_coef:.2f}")

    return pore_count_coef

def initialize_pore_profiles(model, immobilization='co-immobilization'):
    """
    Initializes spatial concentration gradients and bulk variables 
    based on the selected immobilization scheme.
    """
    x_first = model.x.first()
    x_last  = model.x.last()
    S_bulk_init = {comp: pyo.value(model.S_initial[comp]) for comp in model.Components}
    
    k_guess_1 = 0.05  
    k_guess_2 = 0.02  
    
    for t in model.time:
        # 1. Time-dependent bulk guesses (Same for both schemes)
        guess_S1_t = S_bulk_init['S1'] * pyo.exp(-k_guess_1 * t)
        
        if k_guess_1 == k_guess_2: 
            guess_S2_t = S_bulk_init['S1'] * k_guess_1 * t * pyo.exp(-k_guess_1 * t)
        else:
            guess_S2_t = S_bulk_init['S1'] * (k_guess_1 / (k_guess_2 - k_guess_1)) * (pyo.exp(-k_guess_1 * t) - pyo.exp(-k_guess_2 * t))
        guess_S2_t += S_bulk_init['S2']
        
        guess_S3_t = S_bulk_init['S1'] + S_bulk_init['S2'] + S_bulk_init['S3'] - guess_S1_t - guess_S2_t
        
        bulk_guesses = {
            'S1': max(guess_S1_t, 1e-8),
            'S2': max(guess_S2_t, 1e-8),
            'S3': max(guess_S3_t, 1e-8)
        }
        
        for comp in model.Components:
            model.S_0[comp, t].set_value(bulk_guesses[comp])
            
        # 2. Apply spatial gradients depending on the scheme
        for x in model.x:
            x_norm = (x - x_first) / (x_last - x_first)
            
            if immobilization == 'co-immobilization':
                # Single shared pore
                model.S_n['S1', x, t].set_value(max(bulk_guesses['S1'] * (1.0 - 0.5 * x_norm), 1e-8))
                model.S_n['S2', x, t].set_value(max(bulk_guesses['S2'] * (1.0 + 0.1 * x_norm), 1e-8))
                model.S_n['S3', x, t].set_value(max(bulk_guesses['S3'] * (1.0 + 0.1 * x_norm), 1e-8))
                
            elif immobilization == 'single':
                # Particle A (Enzyme A only: S1 -> S2)
                model.S_n_A['S1', x, t].set_value(max(bulk_guesses['S1'] * (1.0 - 0.5 * x_norm), 1e-8)) # S1 consumed
                model.S_n_A['S2', x, t].set_value(max(bulk_guesses['S2'] * (1.0 + 0.1 * x_norm), 1e-8)) # S2 generated
                model.S_n_A['S3', x, t].set_value(max(bulk_guesses['S3'], 1e-8))                       # S3 flat
                
                # Particle B (Enzyme B only: S2 -> S3)
                model.S_n_B['S1', x, t].set_value(max(bulk_guesses['S1'], 1e-8))                        # S1 flat
                model.S_n_B['S2', x, t].set_value(max(bulk_guesses['S2'] * (1.0 - 0.5 * x_norm), 1e-8)) # S2 consumed
                model.S_n_B['S3', x, t].set_value(max(bulk_guesses['S3'] * (1.0 + 0.1 * x_norm), 1e-8)) # S3 generated