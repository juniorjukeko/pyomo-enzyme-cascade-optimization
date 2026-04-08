# Pyomo-based Frameworks for the Modeling and Optimization of Enzymatic Cascades

## Overview

This project provides a mechanistic, first-principles framework for modeling and optimizing enzymatic cascades immobilized within porous particles. It explicitly models coupled reaction-diffusion phenomena inside the pores and the surrounding batch reactor.

By migrating the original SciPy-based numerical approach to a Pyomo-based algebraic modeling implementation, this framework allows for complex, large-scale dynamic optimization. Key enhancements in this repository include:

- **Time-dependent Enzyme Deactivation:** Optional integration of first-order decay kinetics to simulate simultaneous enzyme deactivation over time.
- **Advanced Spatial Immobilization Distributions (SIDs):** Expanded parametrizations that support flexible spatial patterns, including egg-shell, egg-white, and egg-yolk-type distributions.
- **Robust Solver Integration:** Automatic collocation discretization and integration with large-scale non-linear solvers (like IPOPT) for efficient optimization.

---

## Main References

This project builds upon the theoretical foundation and initial codebase established by Paschalidis et al.:

- **Main Paper:** L. Paschalidis, S. Arana-Peña, V. Sieber, and J. Burger, "Mechanistic modeling, parametric study, and optimization of immobilization of enzymatic cascades in porous particles," *React. Chem. Eng.*, vol. 8, no. 9, pp. 2234–2244, 2023.
- **Original SciPy Repository:** [TUM-CS-CTV/ImmobilizationMPO](https://github.com/TUM-CS-CTV/ImmobilizationMPO)

---

## Dimensionless Conversion Equations

To improve the numerical stability of the boundary-value and initial-value problems during optimization, dimensional variables are scaled into dimensionless forms. The standard conversion equations used in this framework's dimensionless approach are:

### Spatial Dimension (Pore Length)

$$\xi = \frac{x}{L}$$

where $x$ is the dimensional pore depth and $L$ is the total pore length.

### Substrate Concentration

$$u_{i} = \frac{S_{i}}{S_{\text{ref}}}$$

where $S_{i}$ is the concentration of component $i$, and $S_{\text{ref}}$ is a reference concentration, typically $S_{1,0}(t=0)$.

### Enzyme Surface Density

$$E_{n,j}^{*}(\xi) = \frac{E_{j,n}(x)}{E_j^{\max}}$$

where $E_{j,n}(x)$ is the dimensional enzyme density and $E_j^{\max}$ is the maximum surface density.
