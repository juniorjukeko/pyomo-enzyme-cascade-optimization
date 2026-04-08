import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

def plot_parameter_distributions(df, save_path=None):

    # calculate f_FOK
    f_FOK = (df['E_A_max']*df['k_A'] * df['D_1'])/(df['E_B_max']*df['k_B']*df['D_2'])

    plot_data = {
        'E_A_max': df['E_A_max'],
        'k_A': df['k_A'],
        'D_1': df['D_1'],
        'k_B': df['k_B'],
        'D_2': df['D_2'],
        'f_FOK': f_FOK
    }

    colors = ['b', 'b', 'b', 'orange', 'orange', 'k']

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (name, values), color in zip(axes, plot_data.items(), colors):

        ax.hist(
            values,
            bins=40,
            color=color,
            alpha=0.8,
            linewidth=1.5
        )

        ax.set_title(name, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)

        if name != 'E_A_max':
            ax.set_xscale('log')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enzyme profile plot saved to: {save_path}")

    return fig

def plot_mc_pairplot(
    samples_csv,
    bvp1_col='bvp1_S3_yield', bvp2_col='bvp2_S3_yield',
    bvp1_label=None, bvp2_label=None,
    alpha=0.6,
    s=20,
    save_path=None,
    show=True,
):
    """
    Pairplot of S3 yield for two BVP profiles from a Monte Carlo simulation CSV.

    Only rows where both yield values are in (0, 1] are included. Each point is
    coloured by f_FOK = (E_A_max * k_A * D_1) / (E_B_max * k_B * D_2).
    A y = x diagonal reference line is drawn to show which profile wins.

    Parameters
    ----------
    samples_csv : str
        Path to the CSV produced by run_monte_carlo_simulation.
        Required columns: E_A_max, E_B_max, k_A, k_B, D_1, D_2,
        plus bvp1_col and bvp2_col.
    bvp1_col : str
        Column name for profile-1 S3 yield. Default 'bvp1_S3_yield'.
    bvp2_col : str
        Column name for profile-2 S3 yield. Default 'bvp2_S3_yield'.
    bvp1_label : str or None
        X-axis label suffix for profile 1. Defaults to bvp1_col.
    bvp2_label : str or None
        Y-axis label suffix for profile 2. Defaults to bvp2_col.
    alpha : float
        Dot transparency. Default 0.6.
    s : float
        Dot size. Default 20.
    save_path : str or None
        If given, saves the figure to this path (e.g. 'mc_pairplot.pdf').
    show : bool
        Call plt.show() at the end. Default True.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    # 1. load data
    df = pd.read_csv(samples_csv)

    required_cols = ['E_A_max', 'E_B_max', 'k_A', 'k_B', 'D_1', 'D_2',
                     bvp1_col, bvp2_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # filter yields
    df = df.dropna(subset=[bvp1_col, bvp2_col])
    valid = (
        df[bvp1_col].between(0, 1.05, inclusive='both') &
        df[bvp2_col].between(0, 1.05, inclusive='both')
    )
    df = df[valid].copy()

    n_total  = pd.read_csv(samples_csv).dropna(subset=[bvp1_col, bvp2_col]).shape[0]
    n_valid  = len(df)
    n_drop   = n_total - n_valid
    print(f"[plot_mc_pairplot] Valid rows: {n_valid}  |  Dropped (out of [0,1]): {n_drop}")

    if n_valid == 0:
        raise ValueError("No valid rows remain after filtering yields to [0, 1].")

    # 2. Compute f_FOK
    numerator   = df['E_A_max'] * df['k_A'] * df['D_1']
    denominator = df['E_B_max'] * df['k_B'] * df['D_2']
    df['f_FOK'] = numerator / denominator

    # Log-scale for symmetric diverging colour mapping around f_FOK = 1
    log_f = np.log10(df['f_FOK'].clip(lower=1e-6))
    vmax  = np.abs(log_f).max()
    vmax  = max(vmax, 0.01)          # guard against all-zero edge case

    norm  = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap  = plt.cm.bwr_r             # blue (high) – red (low)

    # 3. Plot Figure
    fig = plt.figure(figsize=(5.5, 5))
    ax  = fig.add_subplot(111)

    sc = ax.scatter(
        df[bvp1_col], df[bvp2_col],
        c=log_f, cmap=cmap,
        norm=norm,
        s=s,
        alpha=alpha,
        linewidths=0.5, edgecolors='black',
        zorder=3,
    )

    # Diagonal reference line  (y = x)
    ax.plot([0, 1], [0, 1], color='black', linewidth=2,
            linestyle='--', zorder=4, label='$y = x$')

    # Colour bar
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(r'$\log_{10}(f_{\mathrm{FOK}})$', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Formatting
    x_label = bvp1_label if bvp1_label is not None else bvp1_col
    y_label = bvp2_label if bvp2_label is not None else bvp2_col

    ax.set_xlabel(
        r'$\it{Y}_{\mathrm{' + x_label.replace('_', r'\_') + r'}}$ / (-)',
        fontsize=18
    )
    ax.set_ylabel(
        r'$\it{Y}_{\mathrm{' + y_label.replace('_', r'\_') + r'}}$ / (-)',
        fontsize=18
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_aspect('equal', adjustable='box')

    # Annotation: count in each region
    # n_above = (df[bvp2_col] > df[bvp1_col]).sum()
    # n_below = (df[bvp2_col] < df[bvp1_col]).sum()
    # ax.text(0.04, 0.96, f'SID-2 better: {n_above}', transform=ax.transAxes,
    #         fontsize=10, va='top', color='dimgray')
    # ax.text(0.96, 0.04, f'SID-1 better: {n_below}', transform=ax.transAxes,
    #         fontsize=10, ha='right', color='dimgray')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[plot_mc_pairplot] Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig, ax


