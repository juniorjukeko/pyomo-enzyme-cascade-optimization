# pore_bvp_single.py
import pyomo.environ as pyo
from .utils import get_enzyme_profile_rule, get_enzyme_decay_rule, get_enzyme_kinetics

def add_bvp_constraints(model, decay_coef={'kA':0, 'kB':0}, bvp_kwargs=None): 
    if bvp_kwargs is None:
        bvp_kwargs = {}
        
    # --- 1. Extract Enzyme-Specific Settings ---
    default_fun    = bvp_kwargs.get('default_fun', 'linear')
    enzymeA_kwargs = bvp_kwargs.get('enzymeA', {})
    enzymeB_kwargs = bvp_kwargs.get('enzymeB', {})
    
    # INDEPENDENT KINETIC TYPES
    kinetics_type_A = enzymeA_kwargs.get('kinetics_type', 'first_order')
    kinetics_type_B = enzymeB_kwargs.get('kinetics_type', 'first_order')
    
    # Safely extract dimensional reference constants (Required if MM is chosen)
    Km_A  = getattr(model, 'kM_A', None) 
    Km_B  = getattr(model, 'kM_B', None)

    # --- 2. Create symbolic enzyme decay profiles ---
    model.decay_A = pyo.Expression(model.time, rule=get_enzyme_decay_rule(decay_coef.get('kA', 0.0)))
    model.decay_B = pyo.Expression(model.time, rule=get_enzyme_decay_rule(decay_coef.get('kB', 0.0)))

    # --- 3. Build enzyme density profile along pore ---
    model.EA_x_profile = get_enzyme_profile_rule(
        model, model.EA_max,
        start=enzymeA_kwargs.get('start', 0.5), end=enzymeA_kwargs.get('end', 0.5),
        fun=enzymeA_kwargs.get('fun', default_fun),
        **{k: v for k, v in enzymeA_kwargs.items() if k not in ['fun', 'start', 'end', 'kinetics_type']}
    )
    
    model.EB_x_profile = get_enzyme_profile_rule(
        model, model.EB_max,
        start=enzymeB_kwargs.get('start', 0.5), end=enzymeB_kwargs.get('end', 0.5),
        fun=enzymeB_kwargs.get('fun', default_fun),
        **{k: v for k, v in enzymeB_kwargs.items() if k not in ['fun', 'start', 'end', 'kinetics_type']}
    )

    # --- 4a. BVP: Enzyme A only: S1 -> S2 ---
    def typeA_S1_pde_rule(m, x, t): # S1 is CONSUMED here
        lhs_mult, rhs_rate = get_enzyme_kinetics(
            m.EA_x_profile[x], m.decay_A[t], m.kA, m.S_n_A['S1', x, t],
            Km=Km_A, kinetics_type=kinetics_type_A
        )
        return lhs_mult * m.D['S1'] * m.d2S_ndx2_A['S1', x, t] == rhs_rate
    model.A_S1_pde = pyo.Constraint(model.x, model.time, rule=typeA_S1_pde_rule)

    def typeA_S2_pde_rule(m, x, t): # S2 is GENERATED here
        lhs_mult, rhs_rate = get_enzyme_kinetics(
            m.EA_x_profile[x], m.decay_A[t], m.kA, m.S_n_A['S1', x, t],
            Km=Km_A, kinetics_type=kinetics_type_A
        )
        return lhs_mult * m.D['S2'] * m.d2S_ndx2_A['S2', x, t] == -rhs_rate
    model.A_S2_pde = pyo.Constraint(model.x, model.time, rule=typeA_S2_pde_rule)

    def typeA_S3_pde_rule(m, x, t): # S3 does nothing but diffuse
        return m.D['S3'] * m.d2S_ndx2_A['S3', x, t] == 0
    model.A_S3_pde = pyo.Constraint(model.x, model.time, rule=typeA_S3_pde_rule)

    # --- 4b. BVP: Enzyme B only: S2 -> S3 ---
    def typeB_S1_pde_rule(m, x, t): # S1 does nothing but diffuse
        return m.D['S1'] * m.d2S_ndx2_B['S1', x, t] == 0
    model.B_S1_pde = pyo.Constraint(model.x, model.time, rule=typeB_S1_pde_rule)

    def typeB_S2_pde_rule(m, x, t): # S2 is CONSUMED here
        lhs_mult, rhs_rate = get_enzyme_kinetics(
            m.EB_x_profile[x], m.decay_B[t], m.kB, m.S_n_B['S2', x, t],
            Km=Km_B, kinetics_type=kinetics_type_B
        )
        return lhs_mult * m.D['S2'] * m.d2S_ndx2_B['S2', x, t] == rhs_rate
    model.B_S2_pde = pyo.Constraint(model.x, model.time, rule=typeB_S2_pde_rule)

    def typeB_S3_pde_rule(m, x, t): # S3 is GENERATED here
        lhs_mult, rhs_rate = get_enzyme_kinetics(
            m.EB_x_profile[x], m.decay_B[t], m.kB, m.S_n_B['S2', x, t],
            Km=Km_B, kinetics_type=kinetics_type_B
        )
        return lhs_mult * m.D['S3'] * m.d2S_ndx2_B['S3', x, t] == -rhs_rate
    model.B_S3_pde = pyo.Constraint(model.x, model.time, rule=typeB_S3_pde_rule)

    # --- 5. Boundary conditions & Flux Rules --- 
    def bc1_A_rule(m, component, t):
        return m.S_n_A[component, m.x.first(), t] == m.S_0[component, t]
    model.bc1_A = pyo.Constraint(model.Components, model.time, rule=bc1_A_rule)

    def bc1_B_rule(m, component, t):
        return m.S_n_B[component, m.x.first(), t] == m.S_0[component, t]
    model.bc1_B = pyo.Constraint(model.Components, model.time, rule=bc1_B_rule)

    def bc2_A_rule(m, component, t):
        return m.dS_ndx_A[component, m.x.last(), t] == 0
    model.bc2_A = pyo.Constraint(model.Components, model.time, rule=bc2_A_rule)

    def bc2_B_rule(m, component, t):
        return m.dS_ndx_B[component, m.x.last(), t] == 0
    model.bc2_B = pyo.Constraint(model.Components, model.time, rule=bc2_B_rule)

    def flux_rule(m, component, t):
        Np_A = m.Np / 2  
        Np_B = m.Np / 2  
        flux_A = -m.D[component] * m.dS_ndx_A[component, m.x.first(), t] * m.A * Np_A
        flux_B = -m.D[component] * m.dS_ndx_B[component, m.x.first(), t] * m.A * Np_B
        return flux_A + flux_B
        
    model.flux = pyo.Expression(model.Components, model.time, rule=flux_rule)