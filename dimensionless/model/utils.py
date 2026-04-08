import pyomo.environ as pyo

def get_enzyme_decay_rule(decay_constant):
    """
    Returns a Pyomo rule for enzyme decay over time.
    """
    if decay_constant > 0:
        return lambda m, t: pyo.exp(-decay_constant * t)
    else:
        return lambda m, t: 1.0

def get_enzyme_kinetics(E_density, decay, k_cat, u, S_ref=None, Km=None, kinetics_type='first_order'):
    """
    Returns the (LHS_multiplier, RHS_rate) to construct numerically stable PDEs.
    Formulation: LHS_multiplier * (D / L^2 * d2u/dxi2) == RHS_rate
    """
    if kinetics_type == 'first_order':
        # Standard First Order: D/L^2 * d2u/dxi2 = E * decay * k * u
        lhs_multiplier = 1.0
        rhs_rate = E_density * decay * k_cat * u
        return lhs_multiplier, rhs_rate
        
    elif kinetics_type == 'michaelis_menten':
        # Michaelis-Menten: D/L^2 * d2u/dxi2 = E * decay * k * u / (Km + u * S_ref)
        # TRICK: Multiply (Km + u * S_ref) to the LHS to prevent division-by-zero crashes.
        if S_ref is None or Km is None:
            raise ValueError("S_ref and Km must be provided for Michaelis-Menten kinetics.")
            
        lhs_multiplier = 1
        rhs_rate = E_density * decay * k_cat * u / (Km + u * S_ref)
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
        def profile_rule(m, xi):
            return E_max * (start + (end - start) * xi)
        return pyo.Expression(model.xi, rule=profile_rule)
    
    elif fun == 'step':
        xi_step_up   = kwargs.get('x_step_up', 0.3)      # Start of step-up transition
        xi_step_down = kwargs.get('x_step_down', 0.7)    # Start of step-down transition
        smoothness  = kwargs.get('smoothness', 100.0)   # Smoothness factor
        
        # Validate step parameters
        if xi_step_up < 0 or xi_step_up > 1:
            raise ValueError("x_step_up must be between 0 and 1")
        if xi_step_down < 0 or xi_step_down > 1:
            raise ValueError("x_step_down must be between 0 and 1")
        if xi_step_up >= xi_step_down:
            raise ValueError("x_step_up must be less than x_step_down")
        
        def profile_rule(m, xi):

            # Smooth step-up transition (sigmoid function)
            step_up_transition   = 1.0 / (1.0 + pyo.exp(-smoothness * (xi - xi_step_up)))  
            # Smooth step-down transition (sigmoid function)
            step_down_transition = 1.0 / (1.0 + pyo.exp(-smoothness * (xi - xi_step_down)))
            
            # Combined profile:
            # - Before step_up: Constant at start value (before step-up) -> start * E_max
            # - During step_up transition: Smooth step-up transition to (end)
            # - Plateau: Constant at step_value (plateau) -> end * E_max  
            # - During step_down transition: Smooth step-down transition to (start)
            # - After step_down: Constant at start value -> start * E_max

            # (before step_up + (transition->plateau->transition) + after step_down)
            step_profile = (start * E_max * (1.0 - step_up_transition) +
                            end * E_max * (step_up_transition - step_down_transition) +
                            start * E_max * step_down_transition)
            return step_profile
        return pyo.Expression(model.xi, rule=profile_rule)
                
    else:
        raise ValueError(f"Unsupported profile type: {fun}")

def calculate_pore_count_coefficient(model, EA_profile, EB_profile, EA_max, EB_max):
    """
    Normalize pore count so that total enzyme loading (eA + eB) 
    is equal to the reference uniform loading co-immobilization (beta-SID).
    Calculations are based on area under enzyme profile curve.
    Reference system:
        EA(x) = EA_max
        EB(x) = EB_max
    """

    xi_values = sorted(list(model.xi))
    # print(x_values)
    total_EA = 0.0
    total_EB = 0.0
    count = 0

    for xi in xi_values:
        EA_val = pyo.value(EA_profile[xi])
        EB_val = pyo.value(EB_profile[xi])
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

def initialize_concentration_profiles(model, immobilization='co-immobilization'):
    """
    Initializes dimensionless variables u_n (or u_n_A/u_n_B) and u_0 with a time-dependent 
    pseudo-kinetic profile and a spatial concentration gradient.
    """
    xi_first = model.xi.first()
    xi_last  = model.xi.last()
    
    # 1. Dimensionless initial bulk concentrations (u = S / S_ref)
    u_bulk_init = {comp: pyo.value(model.S_initial[comp]) / pyo.value(model.S_ref) for comp in model.Components}
    
    # Rough pseudo-kinetic constants for the guess
    k_guess_1 = 0.005  # Rate of S1 depletion
    k_guess_2 = 0.01  # Rate of S2 depletion into S3
    
    for t in model.time:
        # 2. Time-dependent bulk guesses (Dimensionless consecutive reaction)
        guess_u1_t = u_bulk_init['S1'] * pyo.exp(-k_guess_1 * t)
        
        # S2 rises and falls (approximated)
        if k_guess_1 == k_guess_2: 
            guess_u2_t = u_bulk_init['S1'] * k_guess_1 * t * pyo.exp(-k_guess_1 * t)
        else:
            guess_u2_t = u_bulk_init['S1'] * (k_guess_1 / (k_guess_2 - k_guess_1)) * (pyo.exp(-k_guess_1 * t) - pyo.exp(-k_guess_2 * t))
        guess_u2_t += u_bulk_init['S2'] # Add initial
        
        # S3 is the remainder (mass balance)
        guess_u3_t = u_bulk_init['S1'] + u_bulk_init['S2'] + u_bulk_init['S3'] - guess_u1_t - guess_u2_t
        
        # Apply soft bounds (e.g., 1e-6) to prevent mathematically exact zero guesses
        bulk_guesses = {
            'S1': max(guess_u1_t, 1e-6),
            'S2': max(guess_u2_t, 1e-6),
            'S3': max(guess_u3_t, 1e-6)
        }
        
        # Assign time-dependent bulk guesses
        for comp in model.Components:
            model.u_0[comp, t].set_value(bulk_guesses[comp])
            
        # 3. Apply spatial gradients to the time-dependent bulk guess
        for xi in model.xi:
            # Normalized distance from 0 to 1
            xi_norm = (xi - xi_first) / (xi_last - xi_first)
            
            # Quadratic smoothing profiles to satisfy zero-flux boundary condition at pore end (xi_norm = 1)
            # Derivative of these at xi=1 is exactly 0
            decay_profile = 1.0 - 0.1 * xi_norm + 0.05 * (xi_norm**2)
            build_profile = 1.0 + 0.1 * xi_norm - 0.05 * (xi_norm**2)
            
            if immobilization == 'co-immobilization':
                # Single shared pore
                model.u_n['S1', xi, t].set_value(max(bulk_guesses['S1'] * decay_profile, 1e-6))
                model.u_n['S2', xi, t].set_value(max(bulk_guesses['S2'] * build_profile, 1e-6))
                model.u_n['S3', xi, t].set_value(max(bulk_guesses['S3'] * build_profile, 1e-6))
                
            elif immobilization == 'single':
                # Particle A (Enzyme A only: S1 -> S2)
                model.u_n_A['S1', xi, t].set_value(max(bulk_guesses['S1'] * decay_profile, 1e-6)) # S1 consumed
                model.u_n_A['S2', xi, t].set_value(max(bulk_guesses['S2'] * build_profile, 1e-6)) # S2 generated
                model.u_n_A['S3', xi, t].set_value(max(bulk_guesses['S3'], 1e-6))                 # S3 flat (inert)
                
                # Particle B (Enzyme B only: S2 -> S3)
                model.u_n_B['S1', xi, t].set_value(max(bulk_guesses['S1'], 1e-6))                 # S1 flat (inert)
                model.u_n_B['S2', xi, t].set_value(max(bulk_guesses['S2'] * decay_profile, 1e-6)) # S2 consumed
                model.u_n_B['S3', xi, t].set_value(max(bulk_guesses['S3'] * build_profile, 1e-6)) # S3 generated