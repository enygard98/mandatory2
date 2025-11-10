# matmek4270-mandatory2

[![MAT-MEK4270 mandatory 2](https://github.com/enygard98/mandatory2/actions/workflows/main.yml/badge.svg)](https://github.com/enygard98/mandatory2/actions/workflows/main.yml)


Please see instructions in [the book](https://matmek-4270.github.io/matmek4270-book/mandatory2.html).


# Some further instructions for this repo

This repository contains a small framework in `galerkin.py` for assembling mass/stiffness-like operators and projecting/solving with different polynomial/trigonometric bases and boundary-condition lifts. The code is using classes for reuse of code and clarity, but is kept as simple as possible for educational purposes.

Big picture (architecture)
- Mapping helpers: `map_reference_domain`, `map_true_domain`, `map_expression_true_domain` convert points/expressions between the true domain and a basis-specific reference domain; always use these when mixing domains.
- Base space: `FunctionSpace(N, domain)` defines common behavior: mesh generation, evaluating basis and derivatives, numerical inner products, and a generic mass matrix via numerical quad.
- Concrete spaces
  - `Legendre` and `Chebyshev` use orthogonal polynomials on reference `(-1, 1)`; `eval` uses fast `np.polynomial.legendre.legval` / `chebval` on mapped coordinates.
  - `Trigonometric` base has reference `(0, 1)` and adds a boundary lift in `eval` via `self.B.Xl(X)`; `Sines` is implemented, `Cosines` is a placeholder.
  - `Composite` spaces enforce boundary conditions via a lift `B` and a stencil matrix `S`, e.g. Dirichlet: ψ_i = Q_i − Q_{i+2}. Mass is assembled as `S · M · S^T` where `M` is diagonal in the orthogonal basis. Implementations include `DirichletChebyshev` (done) and placeholders for Legendre/Chebyshev Neumann and Legendre Dirichlet.
- Boundary lifts: `Dirichlet` and `Neumann` construct an affine or quadratic function `B.x` on the physical domain and its reference-evaluator `B.Xl`. Solutions are represented as u(x) = Σ û_i ψ_i(x) + B.x.
- Weak forms: `BasisFunction`, `TrialFunction`, `TestFunction` provide a tiny domain-specific language (DSL). `inner(Trial.diff(k), Test)` builds derivative-weighted bilinear forms; `assemble_generic_matrix` integrates with proper weights and symmetry.

Key files and examples
- `galerkin.py` is the only source file and contains:
  - Example workflows: `test_project`, `test_convection_diffusion`, `test_helmholtz` show projection and PDE assembly patterns.
  - Space definitions and utilities listed above.

Developer workflows
- Dependencies: numpy, scipy, sympy. Install them before running.
- Run examples/tests: execute `python galerkin.py` to run the three demo tests in `__main__`.
- Assemble a PDE (pattern):
  - Choose space `V = DirichletChebyshev(N, domain, bc)` (or similar).
  - Define trial/test: `u = TrialFunction(V)`, `v = TestFunction(V)`.
  - Build operator: e.g., `A = inner(u.diff(2), v) + inner(u, v)`.
  - Build RHS: subtract lift contribution, e.g., `b = inner(f - (V.B.x.diff(x, 2) + V.B.x), v)`.
  - Solve for coefficients: `np.linalg.solve(A, b)` (or `sparse.linalg.spsolve`).

Project conventions and gotchas
- Reference domains: Legendre/Chebyshev map to `(-1, 1)`; Trigonometric maps to `(0, 1)`. Always map x before polynomial evaluation (`eval` methods handle this).
- Coefficient lengths: simple spaces use N+1 coefficients; composite spaces expand with stencil `S` of shape `(N+1, N+3)` and use diagonal `M` of length `N+3` in the orthogonal basis.
- Chebyshev weighting: `inner_product` performs the substitution x = cos(θ) and integrates on `[0, π]` to handle the weight 1/√(1−x²); do not use uniform quad there.
- Derivatives of trigonometric bases: Sine derivative scale is `((j+1)π)^k` with alternating sign for odd k; see `Sines.derivative_basis_function` for the exact pattern.
- Boundary lifts: `V.eval(uh, xj)` returns `P @ uh + V.B.Xl(X)` for spaces with lifts. When forming `b`, include the differential operator applied to `B.x` on the RHS, as shown in tests.

Placeholders intentionally left for students (do not remove without intent)
- `Cosines`, `DirichletLegendre.basis_function`, `NeumannLegendre`, `NeumannChebyshev`, and some `L2_norm_sq`/`mass_matrix` overrides are `NotImplemented`. Mirror the implemented counterparts:
  - Cosines similar to Sines but with cos(kπx) including the k=0 mode handling and L2 norms.
  - Dirichlet/Neumann Legendre/Chebyshev composite spaces follow the stencil S approach used in `DirichletChebyshev` and lifting via `B`.
  - The stencil matrix for the Neumann conditions can be derived using similar logic to the Dirichlet case, ensuring the homogeneous derivative conditions are met at the boundaries. For this use the value of the derivative at the boundary, see [lecture 11](https://matmek-4270.github.io/matmek4270-book/lecture11.html#neumann-boundary-conditions).
- The demos in `__main__` assume these will be implemented; until then, some tests will fail. Use `FunctionSpace.mass_matrix()` as a generic fallback if you don’t provide a closed-form one.

External APIs and patterns to reuse
- SymPy: build exact expressions and derivatives (`x = sp.Symbol("x")`), then `sp.lambdify` for numerical integration and evaluation.
- SciPy: `scipy.integrate.quad` is used; for Chebyshev, pass `weight='alg', wvar=(-0.5, -0.5)`.
- Numpy polynomials: prefer `np.polynomial.legendre.legval`/`chebval` for fast evaluation of polynomial series.

When adding new spaces or operators
- Subclass `FunctionSpace` (or `Composite` if enforcing BCs via stencil). Implement `basis_function(j, sympy=False)`, `derivative_basis_function(j, k)`, optional `weight`, and `eval` if you can use a faster evaluator.
- Respect domain mappings and return shapes consistent with `(N+1,)` coefficient vectors (or stencil-expanded shapes in composite spaces).
