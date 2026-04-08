import pyomo.environ as pyo
import pyomo.dae as dae

def add_reactor_odes(model):
    """
    Add reactor-scale ODEs.
    Independent of immobilization scheme because the pore physics differences 
    are already encoded in the BVP flux calculation.
    """
    # --- Reactor mass balances ---
    
    # S1 consumed to form S2
    def S1_reactor_ivp_rule(m, t):
        return m.S_ref * m.du_0dt['S1', t] == -m.flux['S1', t]

    # S2 produced from S1 and consumed to S3
    def S2_reactor_ivp_rule(m, t):
        return m.S_ref * m.du_0dt['S2', t] == m.flux['S1', t] - m.flux['S2', t]

    # S3 produced from S2
    def S3_reactor_ivp_rule(m, t):
        return m.S_ref * m.du_0dt['S3', t] == m.flux['S2', t]

    model.S1_reactor_ivp = pyo.Constraint(model.time, rule=S1_reactor_ivp_rule)
    model.S2_reactor_ivp = pyo.Constraint(model.time, rule=S2_reactor_ivp_rule)
    model.S3_reactor_ivp = pyo.Constraint(model.time, rule=S3_reactor_ivp_rule)
    
    # Total concentration cannot exceed initial S1 at any time
    # def mass_balance_rule(m, t):
    #     return m.S_0['S1', t] + m.S_0['S2', t] + m.S_0['S3', t] == m.S_initial['S1']
    # model.mass_balance = pyo.Constraint(model.time, rule=mass_balance_rule)
    # --- Initial conditions ---
    def ic_u_0_rule(m, component):
        return m.u_0[component, 0] == m.S_initial[component] / m.S_ref

    model.ic_u_0 = pyo.Constraint(model.Components, rule=ic_u_0_rule)