# model_viz.py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib.gridspec import GridSpec
matplotlib.use('TkAgg')
import pyomo.environ as pyo

def plot_enzyme_decay_profiles(model, decay_coef=None, ax=None, save_path=None):
    """
    Plot enzyme decay profiles (decay_A and decay_B) with scientific-style annotations.
    (This function remains largely unchanged as time t is already dimensional).
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        fig = ax.figure

    # --- Extract data ---
    t_values = sorted(list(model.time))
    decay_A_vals = [pyo.value(model.decay_A[t]) for t in t_values]
    decay_B_vals = [pyo.value(model.decay_B[t]) for t in t_values]

    # --- Plot decay curves ---
    lineA, = ax.plot(t_values, decay_A_vals, 'r-', linewidth=3, label='Enzyme A')
    lineB, = ax.plot(t_values, decay_B_vals, 'b-', linewidth=3, label='Enzyme B')

    # --- Styling ---
    ax.set_xlabel('Time (min)', fontsize=16, family='serif')
    ax.set_ylabel('Relative Enzyme Activity', fontsize=16, family='serif')
    ax.set_title('Enzyme Decay Kinetics Over Time', fontsize=18, family='serif')
    ax.legend(fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(0, pyo.value(model.tf))

    # --- Extract decay coefficients ---
    kA_coef = decay_coef.get('kA', 0) if decay_coef else 0
    kB_coef = decay_coef.get('kB', 0) if decay_coef else 0

    # --- Compute half-lives ---
    half_life_A = np.log(2) / kA_coef if kA_coef > 0 else np.nan
    half_life_B = np.log(2) / kB_coef if kB_coef > 0 else np.nan

    # --- Compose annotation string ---
    annotation_text = (
        rf"$k_d^A  = {kA_coef:.5f}\ \mathrm{{min^{{-1}}}}$" + "\n" +
        rf"$t_{{1/2}}^A = {half_life_A:.1f}\ \mathrm{{min}}$" + "\n" +
        rf"$k_d^B  = {kB_coef:.5f}\ \mathrm{{min^{{-1}}}}$" + "\n" +
        rf"$t_{{1/2}}^B = {half_life_B:.1f}\ \mathrm{{min}}$"
    )

    # --- Add semi-transparent square bounding box ---
    ax.text(
        0.02, 0.05, annotation_text,
        transform=ax.transAxes,
        fontsize=14,
        fontfamily='serif',
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', alpha=0.90)
    )

    # --- Save or show ---
    if save_path and created_fig:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enzyme decay plot saved to: {save_path}")

    if created_fig:
        plt.tight_layout()
        plt.show()

    return fig

def plot_enzyme_pore_profiles(model, immobilization, ax=None, save_path=None):
    """
    Plot enzyme surface density along the pore. 
    Adapts dimensionless model.xi to dimensional x for the axis.
    """
    # 1. Extract dimensionless space array (xi)
    xi_values = sorted(list(model.xi))
    
    # 2. Scale it back to dimensional physical length (x)
    L_val = pyo.value(model.L)
    x_values = [xi * L_val for xi in xi_values]

    # Determine EA and EB values based on immobilization (using xi index)
    EA_values = [pyo.value(model.EA_x_profile[xi]) for xi in xi_values]
    EB_values = [pyo.value(model.EB_x_profile[xi]) for xi in xi_values]
    EA_max, EB_max = max(EA_values), max(EB_values)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_A, ax_B = axes

    # Plot title
    if immobilization=='single':
        title_text = f"Enzyme Density Along Pore Length\n (single immobilization)"
    elif immobilization=='co-immobilization':
        title_text = f"Enzyme Density Along Pore Length\n (co-immobilization)"
    
    # Plot Enzyme A
    ax_A.plot(x_values, EA_values, 'r-', linewidth=4, label='Enzyme A')
    ax_A.set_ylabel('Enzyme A (μmol/dm²)', fontsize=14, family='serif')
    ax_A.set_ylim(0, 10)
    ax_A.set_yticks([5])
    ax_A.grid(True, alpha=0.3)
    ax_A.text(0.02, 0.05, f"Max EA = {EA_max:.2f}", transform=ax_A.transAxes,
              fontsize=12, family='serif', verticalalignment='bottom', horizontalalignment='left')

    # Plot Enzyme B
    ax_B.plot(x_values, EB_values, 'b-', linewidth=4, label='Enzyme B')
    ax_B.set_xlabel('Pore Position x (dm)', fontsize=14, family='serif')
    ax_B.set_ylabel('Enzyme B (μmol/dm²)', fontsize=14, family='serif')
    ax_B.set_ylim(0, 10)
    ax_B.set_yticks([5])
    ax_B.grid(True, alpha=0.3)
    ax_B.text(0.02, 0.05, f"Max EB = {EB_max:.2f}", transform=ax_B.transAxes,
              fontsize=12, family='serif', verticalalignment='bottom', horizontalalignment='left')

    # Shared x-axis formatting (Use physical length L_val)
    ax_B.set_xlim(0, L_val)
    ax_B.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax_B.ticklabel_format(axis='x', style='scientific', scilimits=(-3, 3))
    ax_B.tick_params(axis='both', which='major', labelsize=12)

    # Overall figure title
    fig.suptitle(title_text, fontsize=16, family='serif')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enzyme profile plot saved to: {save_path}")

    return fig

def plot_substrate_time_profiles(model, save_path=None):
    """
    Plot S1, S2, S3 physical concentrations over time.
    Translates the dimensionless u_0 variable back into physical S_0.
    """
    # Extract time
    t_values = sorted(list(model.time))
    
    # Extract the scaling factor (S_ref) to convert u back to S
    S_ref_val = pyo.value(model.S_ref)
    
    # Extract dimensionless u_0 and multiply by S_ref
    S1_values = [pyo.value(model.u_0['S1', t]) * S_ref_val for t in t_values]
    S2_values = [pyo.value(model.u_0['S2', t]) * S_ref_val for t in t_values]
    S3_values = [pyo.value(model.u_0['S3', t]) * S_ref_val for t in t_values]

    # Final and initial values
    final_S1, final_S2, final_S3 = S1_values[-1], S2_values[-1], S3_values[-1]
    initial_S1 = S1_values[0]

    # Yield calculations
    Y_S2 = final_S2 / initial_S1
    Y_S3 = final_S3 / initial_S1

    # Create figure layout (2/3 plot, 1/3 info)
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 3, width_ratios=[2, 0.05, 1], figure=fig)

    ax = fig.add_subplot(gs[0])  # main plot
    ax_text = fig.add_subplot(gs[2])  # text panel

    # Plot concentration profiles
    ax.plot(t_values, S1_values, 'k-', linewidth=3, label='S1 (Substrate)')
    ax.plot(t_values, S2_values, 'b-', linewidth=3, label='S2 (Intermediate)')
    ax.plot(t_values, S3_values, 'r-', linewidth=3, label='S3 (Product)')

    # Axes settings
    ax.set_xlabel('Reaction time, t (min)', fontsize=16, family='serif')
    ax.set_ylabel('Concentration (μM)', fontsize=16, family='serif')
    ax.set_title('Substrate Concentrations Over Reaction Time', fontsize=18, family='serif')
    ax.legend(fontsize=14)
    ax.set_xlim(0, pyo.value(model.tf))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Remove axes for right panel
    ax_text.axis('off')

    # Text (scientific journal style)
    text_str = (
        f"$\\bf{{Final\\ Concentrations}}$\n"
        f"$S_1$ = {final_S1:.2f} μM\n"
        f"$S_2$ = {final_S2:.2f} μM\n"
        f"$S_3$ = {final_S3:.2f} μM\n\n"
        f"$\\bf{{Yields}}$\n"
        f"$Y_m(S_2)$ = {Y_S2:.3f}\n"
        f"$Y_m(S_3)$ = {Y_S3:.3f}"
    )

    # Place text in upper right
    ax_text.text(
        0.0, 0.95, text_str,
        transform=ax_text.transAxes,
        fontsize=16, fontfamily='serif', va='top', ha='left'
    )

    plt.tight_layout()

    # Save if needed
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Substrate concentration plot saved to: {save_path}")

    plt.show()
    return fig