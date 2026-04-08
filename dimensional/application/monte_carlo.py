import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def generate_parameter_samples(N, bounds, method="log", seed=None, save_path=None):
    """
    Generate Monte Carlo parameter samples (f_FOK). Uniform sampling of log-space of bounds.

    Parameters
    ----------
    N : int
        Number of samples
    bounds : dict
        Parameter bounds:
        {
            'E_A_max': (min, max),
            'E_B_max': (min, max),
            'k_A': (min, max),
            'k_B': (min, max),
            'D_1': (min, max),
            'D_2': (min, max)
        }
    method : str
        'log'  -> logarithmic uniform sampling
        'lhs'  -> Latin Hypercube Sampling in log-space
    seed : int or None
        Random seed

    Returns
    -------
    pd.DataFrame
    """
    from scipy.stats import qmc
    rng = np.random.default_rng(seed)
    independent_params = ['k_A', 'k_B', 'D_1', 'D_2']

    if method == "log":

        # sample E_A only
        E_A = rng.uniform(1, 9, N)
        E_B = 10 - E_A

        data = {
            'E_A_max': E_A, 'E_B_max': E_B
        }

        for p in independent_params:
            low, high = bounds[p]
            data[p] = 10 ** rng.uniform(
                np.log10(low), np.log10(high),
                N
            )

    elif method == "lhs":

        sampler = qmc.LatinHypercube(d=len(independent_params)+1, seed=seed)
        sample = sampler.random(N)

        # first dimension for E_A
        E_A = 1 + sample[:, 0] * (9 - 1)
        E_B = 10 - E_A

        data = {
            'E_A_max': E_A, 'E_B_max': E_B
        }

        # remaining parameters log-scaled
        for i, p in enumerate(independent_params, start=1):
            low, high = bounds[p]

            log_low = np.log10(low)
            log_high = np.log10(high)

            scaled = log_low + sample[:, i] * (log_high - log_low)
            data[p] = 10 ** scaled

    else:
        raise ValueError("method must be 'log' or 'lhs'")
    
    df = pd.DataFrame(data)
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Parameter samples saved to: {save_path}")
        
    return df


def run_monte_carlo_simulation(
    samples_csv,
    bvp_profile_1, bvp_profile_2,
    decay_coef=None,
    n_runs=None,
    bvp1_col='bvp1_S3_yield', bvp2_col='bvp2_S3_yield',
    save_path=None,
    verbose=False, solver_tol=1e-3,
):
    """
    Run Monte Carlo simulation over sampled parameter sets for two SID profiles.

    Reads a samples CSV (produced by generate_parameter_samples), appends two result
    columns (bvp1_S3_yield, bvp2_S3_yield) if they don't already exist, and fills in
    missing results row-by-row up to n_runs attempts. Saves progress after every row so
    partial results are never lost.

    Parameters
    ----------
    samples_csv : str
        Path to the CSV of parameter samples. Expected columns:
        E_A_max, E_B_max, k_A, k_B, D_1, D_2.
        If bvp1_col / bvp2_col columns already exist (from a prior run), only rows
        where BOTH are NaN are retried.
    bvp_profile_1 : dict
        bvp_kwargs dict for profile 1 (passed directly to build_reactor_model).
    bvp_profile_2 : dict
        bvp_kwargs dict for profile 2 (passed directly to build_reactor_model).
    decay_coef : dict, optional
        Enzyme decay coefficients, e.g. {'kA': 0.004, 'kB': 0.002}. Defaults to {'kA': 0, 'kB': 0} (no decay).
    n_runs : int or None
        Maximum number of *incomplete* rows to process in this call.
        Pass None to process all incomplete rows.
    bvp1_col : str
        Column name for profile-1 S3 yield results. Default 'bvp1_S3_yield'.
    bvp2_col : str
        Column name for profile-2 S3 yield results. Default 'bvp2_S3_yield'.
    save_path : str or None
        Where to write the updated CSV after each row. Defaults to overwriting
        samples_csv in-place so progress is preserved on interruption.
    verbose : bool
        Print progress to stdout.
    solver_tol : float
        Tolerance passed to solve_model_robust.

    Returns
    -------
    pd.DataFrame
        The updated samples DataFrame with bvp1_col and bvp2_col filled in.
    """
    import pyomo.environ as pyo
    from pyomo.environ import value as pyo_value
    from main import build_reactor_model
    from model.solve import solve_model_robust

    if decay_coef is None:
        decay_coef = {'kA': 0, 'kB': 0}

    if save_path is None:
        save_path = samples_csv

    # 1. Load samples
    df = pd.read_csv(samples_csv)

    # Add result columns if absent
    if bvp1_col not in df.columns:
        df[bvp1_col] = np.nan
    if bvp2_col not in df.columns:
        df[bvp2_col] = np.nan

    # Identify rows where both results are still missing
    pending_mask = df[bvp1_col].isna() & df[bvp2_col].isna()
    pending_indices = df.index[pending_mask].tolist()

    if n_runs is not None:
        pending_indices = pending_indices[:n_runs]

    total = len(pending_indices)
    print(f"[MC] Total pending rows to run: {total}")
    if verbose:  
        print(f"[MC] Decay coefficients: kA={decay_coef.get('kA',0)}, kB={decay_coef.get('kB',0)}")

    profiles = {
        bvp1_col: bvp_profile_1,
        bvp2_col: bvp_profile_2,
    }

    # 2. Loop for MC runs
    for run_num, idx in enumerate(pending_indices, start=1):
        row = df.loc[idx]

        EA_max = float(row['E_A_max'])
        EB_max = float(row['E_B_max'])
        kA_val = float(row['k_A'])
        kB_val = float(row['k_B'])
        D1_val = float(row['D_1'])
        D2_val = float(row['D_2'])

        if verbose:
            print(f"\n{'='*65}")
            print(f"[MC] Run {run_num}/{total}  (df index={idx})")
            print(f"     EA={EA_max:.3f}  EB={EB_max:.3f}  "
                  f"kA={kA_val:.4g}  kB={kB_val:.4g}  "
                  f"D1={D1_val:.3e}  D2={D2_val:.3e}")

        for col_name, bvp_kwargs in profiles.items():
            if verbose:
                print(f"  -> Profile: {col_name}")

            try:
                model = build_reactor_model(
                    immobilization='co-immobilization',
                    decay_coef=decay_coef,
                    bvp_kwargs=bvp_kwargs,
                )

                # Override sampled parameters
                model.EA_max.set_value(EA_max)
                model.EB_max.set_value(EB_max)
                model.kA.set_value(kA_val)
                model.kB.set_value(kB_val)

                # D_1 → S1 diffusivity, D_2 → S2, S3 diffusivity (CHANGE IF NEEDED)
                model.D['S1'].set_value(D1_val)
                model.D['S2'].set_value(D2_val)
                model.D['S3'].set_value(D2_val)

                solved_model, solver_results = solve_model_robust(
                    model, tol=solver_tol, verbose=False
                )

                if solver_results.solver.termination_condition == pyo.TerminationCondition.optimal:
                    S1_init = pyo_value(solved_model.S_0['S1', solved_model.time.first()])
                    S3_end  = pyo_value(solved_model.S_0['S3', solved_model.time.last()])
                    s3_yield = S3_end / S1_init
                    df.at[idx, col_name] = s3_yield
                    if verbose:
                        print(f"     [OK] S3_yield = {s3_yield:.4f}")
                else:
                    df.at[idx, col_name] = np.nan
                    if verbose:
                        print(f"     [FAIL] {solver_results.solver.termination_condition}")

            except Exception as exc:
                df.at[idx, col_name] = np.nan
                if verbose:
                    import traceback
                    print(f"     [ERROR] {exc}")
                    traceback.print_exc()

        # Save after every row so progress survives interruptions
        df.to_csv(save_path, index=False)
        if verbose:
            completed = df[[bvp1_col, bvp2_col]].notna().all(axis=1).sum()
            print(f"  -> Saved. Completed rows so far: {completed}/{len(df)}")

    if verbose:
        completed = df[[bvp1_col, bvp2_col]].notna().all(axis=1).sum()
        print(f"\n[MC] Done. Total completed rows: {completed}/{len(df)}")
        print(f"[MC] Results saved to: {save_path}")

    return df
