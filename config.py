"""
config.py

Central configuration file for the alpha model simulation constants.
Contains geometric, kinetic, physical property, and batch reactor parameters, 
and numerical solver settings.
"""
# -------------------------------
# Batch reactor parameters settings
# -------------------------------
INITIAL_SUBSTRATE_CONC = {
    "S1_0"  : 1000,
    "S2_0"  : 0.1,
    "S3_0"  : 0.1
}                           # S_i,0 (t=0), mM, initial substrate S1, S2, S3 concentration in reactor bulk
REACTION_TIME   = 480        # t_f, min, total reaction (simulation) time
PORES_COUNT     = 5*10**14  # N_n, -, total pores count

# -------------------------------
# Physical property parameters settings
# -------------------------------
DIFFUSIFITY_CONST = {
    "S1"  : 10**(-8),
    "S2"  : 5*10**(-6),
    "S3"  : 5*10**(-6)
}           # D_i, dm²/min, Diffusivity constant

# -------------------------------
# Geometric parameters settings
# -------------------------------
PORE_AREA               = 8*10**(-15)   # A, dm², pore cross-sectional area
PORE_LENGTH             = 2*10**(-4)    # L, dm, pore length
MAX_ENZYME_SURFACE_DENSITY  = {
    "Enzyme_A"  : 10,
    "Enzyme_B"  : 10,
}           # E_j^max, μmol/dm², MAXIMUM enzyme surface density inside pores

# -------------------------------
# Kinetic parameters settings
# -------------------------------
# First order kinetics
REACTION_CONST          = {
    "Enzyme_A"    : 30, #30,
    "Enzyme_B"    : 80  #80 
}           # k_j, dm²/μmol/min, first-order kinetic constant

# Michaelis-Menten kinetics
MICHAELIS_MENTEN_CONST  = {
    "Enzyme_A"  : 30,
    "Enzyme_B"  : 30
}           # K_j, mM, Michaelis-Menten constant
