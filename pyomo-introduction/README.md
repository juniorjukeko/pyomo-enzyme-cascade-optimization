## Roadmap for Learning Pyomo

### 1. Linear Models (Optimization Fundamentals)
**Objective:** Understand Pyomo syntax and model structure using simple linear programming (LP) and mixed-integer linear programming (MILP) examples.

**To Do:**
- [ ] Review basic Pyomo components: `ConcreteModel`, `Var`, `Objective`, `Constraint`, `SolverFactory`.
- [ ] Solve sample cases
- [ ] Visualize and interpret results (use `matplotlib` or `pandas`).
- [ ] Write notes on common solver messages and how to debug infeasible models.

### 2. Simulation (Nonlinear Models)
**Objective:** Move from algebraic optimization to nonlinear, steady-state, or parametric process simulations (reaction or mass balance models).

**To Do:**
- [ ] Learn how to define nonlinear expressions using `pyo.Expression` and `pyo.Constraint`.
- [ ] Solve sample cases (steady-state reactor, kinetic rate, parametric variations).
- [ ] Parametric optimization.

### 3. Differential-Algebraic Equations (DAE)
**Objective:** Extend models to dynamic behavior using Pyomo.DAE, moving to time-dependent or spatially distributed process models.

**To Do:**
- [ ] Learn Pyomo.DAE components: `ContinuousSet`, `DerivativeVar`, and discretization methods (`Collocation`, `FiniteDifference`).
- [ ] Solve ODE-DAE systems that are related to reaction kinetics or diffusion
- [ ] Visualize dynamic profiles vs. spatial variables (time or position).
- [ ] Explore model scaling, initialization strategies, solver options (maybe).

---
## C. Notes & Documentation
Each topic (Linear, Simulation, DAE) should have:
- A dedicated Jupyter notebook.
- Example code with comments.
- Reflections on challenges, insights, and lessons learned.

---
## D. References & Resources
- [Pyomo Docs](http://www.pyomo.org/documentation)
- Link 2, etc.
