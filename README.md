# Pyomo-based Frameworks for The Modeling and Optimization of Enzymatic Cascades

## Overview
[cite_start]This project provides a mechanistic, first-principles framework for modeling and optimizing enzymatic cascades immobilized within porous particles[cite: 34]. [cite_start]It explicitly models coupled reaction-diffusion phenomena inside the pores and the surrounding batch reactor[cite: 34]. 

[cite_start]By migrating the original SciPy-based numerical approach to a **Pyomo-based algebraic modeling implementation**, this framework allows for complex, large-scale dynamic optimization[cite: 41, 42]. Key enhancements in this repository include:
* [cite_start]**Time-dependent Enzyme Deactivation:** Optional integration of first-order decay kinetics to simulate simultaneous enzyme deactivation over time[cite: 48].
* [cite_start]**Advanced Spatial Immobilization Distributions (SIDs):** Expanded parametrizations that support flexible spatial patterns, including egg-shell, egg-white, and egg-yolk-type distributions[cite: 49].
* [cite_start]**Robust Solver Integration:** Automatic collocation discretization and integration with large-scale non-linear solvers (like IPOPT) for efficient optimization[cite: 42, 43].

## Main References
[cite_start]This project builds upon the theoretical foundation and initial codebase established by Paschalidis et al.[cite: 34, 38]:
* **Main Paper:** L. Paschalidis, S. Arana-Peña, V. Sieber, and J. Burger, "Mechanistic modeling, parametric study, and optimization of immobilization of enzymatic cascades in porous particles," *React. Chem. Eng.*, vol. 8, no. [cite_start]9, pp. 2234-2244, 2023[cite: 521]. 
* [cite_start]**Original SciPy Repository:** [TUM-CS-CTV/ImmobilizationMPO](https://github.com/TUM-CS-CTV/ImmobilizationMPO)[cite: 40].

## Dimensionless Conversion Equations
To improve the numerical stability of the boundary-value and initial-value problems during optimization, dimensional variables are often scaled into dimensionless forms. The standard conversion equations used in this framework's dimensionless approach are:

* **Spatial Dimension (Pore Length):** $$\xi = \frac{x}{L}$$
  *(where $x$ is the dimensional pore depth and $L$ is the total pore length)*

* **Substrate Concentration:** $$u_{i} = \frac{S_{i}}{S_{ref}}$$
  *(where $S_{i}$ is the concentration of component $i$, and $S_{ref}$ is a reference concentration, typically $S_{1,0}(t=0)$)*

* **Enzyme Surface Density:** $$E_{n, j}^*(\xi) = \frac{E_{j,n}(x)}{E_j^{max}}$$
  *(where $E_{j,n}(x)$ is the dimensional enzyme density and $E_j^{max}$ is the maximum surface density)*
