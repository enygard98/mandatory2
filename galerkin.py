import numpy as np
import scipy.sparse as sparse
import scipy.special as sc
import sympy as sp
from numpy.polynomial import Chebyshev as Cheb
from numpy.polynomial import Legendre as Leg
from scipy.integrate import quad

x = sp.Symbol("x")


def map_reference_domain(x, d, r):
    return r[0] + (r[1] - r[0]) * (x - d[0]) / (d[1] - d[0])


def map_true_domain(x, d, r):
    return d[0] + (d[1] - d[0]) * (x - r[0]) / (r[1] - r[0])


def map_expression_true_domain(u, x, d, r):
    if d != r:
        u = sp.sympify(u)
        xm = map_true_domain(x, d, r)
        u = u.replace(x, xm)
    return u


class FunctionSpace:
    def __init__(self, N, domain=(-1, 1)):
        self.N = N
        self._domain = domain

    @property
    def domain(self):
        return self._domain

    @property
    def reference_domain(self):
        raise RuntimeError

    @property
    def domain_factor(self):
        d = self.domain
        r = self.reference_domain
        return (d[1] - d[0]) / (r[1] - r[0])

    def mesh(self, N=None):
        d = self.domain
        n = N if N is not None else self.N
        return np.linspace(d[0], d[1], n + 1)

    def weight(self, x=x):
        return 1

    def basis_function(self, j, sympy=False):
        raise RuntimeError

    def derivative_basis_function(self, j, k=1):
        raise RuntimeError

    def evaluate_basis_function(self, Xj, j):
        return self.basis_function(j)(Xj)

    def evaluate_derivative_basis_function(self, Xj, j, k=1):
        return self.derivative_basis_function(j, k=k)(Xj)

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh

    def eval_basis_function_all(self, Xj):
        P = np.zeros((len(Xj), self.N + 1))
        for j in range(self.N + 1):
            P[:, j] = self.evaluate_basis_function(Xj, j)
        return P

    def eval_derivative_basis_function_all(self, Xj, k=1):
        Xj = np.atleast_1d(Xj)
        P = np.zeros((len(Xj), self.N + 1))
        for j in range(self.N + 1):
            P[:, j] = self.evaluate_derivative_basis_function(Xj, j, k=k)
        return P

    def inner_product(self, u):
        us = map_expression_true_domain(u, x, self.domain, self.reference_domain)
        # map besselj via scipy.special.jv when lambdifying
        us = sp.lambdify(x, us, modules=[{"besselj": sc.jv}, "numpy"])
        uj = np.zeros(self.N + 1)
        h = self.domain_factor
        r = self.reference_domain
        for i in range(self.N + 1):
            psi = self.basis_function(i)

            def uv(Xj):
                return us(Xj) * psi(Xj)

            uj[i] = float(h) * quad(uv, float(r[0]), float(r[1]))[0]
        return uj

    def mass_matrix(self):
        return assemble_generic_matrix(TrialFunction(self), TestFunction(self))


class Legendre(FunctionSpace):
    def __init__(self, N, domain=(-1, 1)):
        FunctionSpace.__init__(self, N, domain=domain)

    @property
    def reference_domain(self):
        return (-1, 1)

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.legendre(j, x)
        return Leg.basis(j)

    def derivative_basis_function(self, j, k=1):
        return self.basis_function(j).deriv(k)

    def L2_norm_sq(self, N):
        # Return the L2 norm squared of Legendre polynomials on [-1,1]:
        # \int_{-1}^1 P_j(x)^2 dx = 2 / (2 j + 1)
        return np.array([2.0 / (2 * i + 1) for i in range(N)])

    def mass_matrix(self):
        # Construct diagonal mass matrix on the reference domain using
        # the analytic L2 norms of Legendre polynomials.
        vals = self.L2_norm_sq(self.N + 1)
        return sparse.diags([vals], [0], shape=(self.N + 1, self.N + 1), format="csr")

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        return np.polynomial.legendre.legval(Xj, uh)


class Chebyshev(FunctionSpace):
    def __init__(self, N, domain=(-1, 1)):
        FunctionSpace.__init__(self, N, domain=domain)

    @property
    def reference_domain(self):
        return (-1, 1)

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.cos(j * sp.acos(x))
        return Cheb.basis(j)

    def derivative_basis_function(self, j, k=1):
        return self.basis_function(j).deriv(k)

    def weight(self, x=x):
        return 1 / sp.sqrt(1 - x**2)

    def L2_norm_sq(self, N):
        # For Chebyshev polynomials T_j with weight w(x)=1/sqrt(1-x^2) on [-1,1]:
        # \int_{-1}^1 T_0(x)^2 w(x) dx = pi,
        # \int_{-1}^1 T_j(x)^2 w(x) dx = pi/2 for j >= 1
        if N <= 0:
            return np.array([])
        arr = np.empty(N)
        arr[0] = np.pi
        if N > 1:
            arr[1:] = np.pi / 2.0
        return arr

    def mass_matrix(self):
        # Construct diagonal mass matrix on the reference domain using
        # the analytic L2 norms of Chebyshev polynomials with weight
        # w(x)=1/sqrt(1-x^2).
        vals = self.L2_norm_sq(self.N + 1)
        return sparse.diags([vals], [0], shape=(self.N + 1, self.N + 1), format="csr")

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        return np.polynomial.chebyshev.chebval(Xj, uh)

    def inner_product(self, u):
        us = map_expression_true_domain(u, x, self.domain, self.reference_domain)
        # change of variables to x = cos(theta)
        us_theta = sp.simplify(us.subs(x, sp.cos(x)), inverse=True)
        # map besselj via scipy.special.jv when lambdifying
        us_f = sp.lambdify(x, us_theta, modules=[{"besselj": sc.jv}, "numpy"])

        uj = np.zeros(self.N + 1)
        h = float(self.domain_factor)

        # build basis functions *per integer index* (avoid a symbolic index k)
        basis_funcs = []
        for i in range(self.N + 1):
            psi_sym = sp.simplify(self.basis_function(i, True).subs(x, sp.cos(x), inverse=True))
            basis_funcs.append(sp.lambdify(x, psi_sym, modules=[{"besselj": sc.jv}, "numpy"]))

        for i in range(self.N + 1):
            def uv(theta, j=i):
                return us_f(theta) * basis_funcs[j](theta)
            uj[i] = h * quad(uv, 0, np.pi)[0]

        return uj


class Trigonometric(FunctionSpace):
    """Base class for trigonometric function spaces"""

    @property
    def reference_domain(self):
        return (0, 1)

    def mass_matrix(self):
        return sparse.diags(
            [self.L2_norm_sq(self.N + 1)], [0], (self.N + 1, self.N + 1), format="csr"
        )

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)


class Sines(Trigonometric):
    def __init__(self, N, domain=(0, 1), bc=(0, 0)):
        Trigonometric.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.sin((j + 1) * sp.pi * x)
        return lambda Xj: np.sin((j + 1) * np.pi * Xj)

    def derivative_basis_function(self, j, k=1):
        scale = ((j + 1) * np.pi) ** k * {0: 1, 1: -1}[(k // 2) % 2]
        if k % 2 == 0:
            return lambda Xj: scale * np.sin((j + 1) * np.pi * Xj)
        else:
            return lambda Xj: scale * np.cos((j + 1) * np.pi * Xj)

    def L2_norm_sq(self, N):
        return 0.5


class Cosines(Trigonometric):
    def __init__(self, N, domain=(0, 1), bc=(0, 0)):
        Trigonometric.__init__(self, N, domain=domain)
        # Cosine basis matches Neumann boundary conditions (derivative specified)
        self.B = Neumann(bc, domain, self.reference_domain)

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.cos(j * sp.pi * x)
        return lambda Xj: np.cos(j * np.pi * Xj)

    def derivative_basis_function(self, j, k=1):
        # derivative of cos(j*pi*x) = (j*pi)^k * cos or sin with alternating signs
        scale = (j * np.pi) ** k * {0: 1, 1: -1}[(k // 2) % 2]
        if k % 2 == 0:
            return lambda Xj: scale * np.cos(j * np.pi * Xj)
        else:
            # odd derivatives introduce a negative sign compared to sin pattern
            return lambda Xj: -scale * np.sin(j * np.pi * Xj)

    def L2_norm_sq(self, N):
        # On (0,1): \int_0^1 cos(0)^2 dx = 1, and for j>=1 \int_0^1 cos(j pi x)^2 dx = 1/2
        if N <= 0:
            return np.array([])
        arr = np.empty(N)
        arr[0] = 1.0
        if N > 1:
            arr[1:] = 0.5
        return arr


# Create classes to hold the boundary function


class Dirichlet:
    def __init__(self, bc, domain, reference_domain):
        d = domain
        r = reference_domain
        h = d[1] - d[0]
        self.bc = bc
        self.x = (
            bc[0] * (d[1] - x) / h + bc[1] * (x - d[0]) / h
        )  # in physical coordinates
        self.xX = map_expression_true_domain(
            self.x, x, d, r
        )  # in reference coordinates
        # map besselj via scipy.special.jv when lambdifying
        self.Xl = sp.lambdify(x, self.xX, modules=[{"besselj": sc.jv}, "numpy"])


class Neumann:
    def __init__(self, bc, domain, reference_domain):
        d = domain
        r = reference_domain
        h = d[1] - d[0]
        self.bc = bc
        self.x = bc[0] / h * (d[1] * x - x**2 / 2) + bc[1] / h * (
            x**2 / 2 - d[0] * x
        )  # in physical coordinates
        self.xX = map_expression_true_domain(
            self.x, x, d, r
        )  # in reference coordinates
        # map besselj via scipy.special.jv when lambdifying
        self.Xl = sp.lambdify(x, self.xX, modules=[{"besselj": sc.jv}, "numpy"])


class Composite(FunctionSpace):
    r"""Base class for function spaces created as linear combinations of orthogonal basis functions

    The composite basis functions are defined using the orthogonal basis functions
    (Chebyshev or Legendre) and a stencil matrix S. The stencil matrix S is used
    such that basis function i is

    .. math::

        \psi_i = \sum_{j=0}^N S_{ij} Q_j

    where :math:`Q_i` can be either the i'th Chebyshev or Legendre polynomial

    For example, both Chebyshev and Legendre have Dirichlet basis functions

    .. math::

        \psi_i = Q_i-Q_{i+2}

    Here the stencil matrix will be

    .. math::

        s_{ij} = \delta_{ij} - \delta_{i+2, j}, \quad (i, j) \in (0, 1, \ldots, N) \times (0, 1, \ldots, N+2)

    Note that the stencil matrix is of shape :math:`(N+1) \times (N+3)`.
    """

    def eval(self, uh, xj):
        xj = np.atleast_1d(xj)
        Xj = map_reference_domain(xj, self.domain, self.reference_domain)
        P = self.eval_basis_function_all(Xj)
        return P @ uh + self.B.Xl(Xj)

    def mass_matrix(self):
        M = sparse.diags(
            [self.L2_norm_sq(self.N + 3)],
            [0],
            shape=(self.N + 3, self.N + 3),
            format="csr",
        )
        return self.S @ M @ self.S.T

    def derivative_basis_function(self, j, k=1):
        """Evaluate k-th derivative of composite basis psi_j = sum_l S_{j,l} Q_l.

        This constructs a numeric evaluator by summing the k-th derivatives
        of the underlying orthogonal polynomials Q_l multiplied by the
        stencil coefficients S_{j,l}.
        """
        row = self.S.getrow(j).tocoo()
        cols = row.col
        data = row.data

        # If composite built from Legendre
        if issubclass(self.__class__, Legendre) or isinstance(self, Legendre):
            def psi_deriv(Xj):
                Xj = np.atleast_1d(Xj)
                val = np.zeros_like(Xj, dtype=float)
                for col, coeff in zip(cols, data):
                    c = np.zeros(col + 1)
                    c[col] = 1.0
                    dc = np.polynomial.legendre.legder(c, m=k)
                    val = val + coeff * np.polynomial.legendre.legval(Xj, dc)
                return val

            return psi_deriv

        # If composite built from Chebyshev
        if issubclass(self.__class__, Chebyshev) or isinstance(self, Chebyshev):
            def psi_deriv(Xj):
                Xj = np.atleast_1d(Xj)
                val = np.zeros_like(Xj, dtype=float)
                for col, coeff in zip(cols, data):
                    c = np.zeros(col + 1)
                    c[col] = 1.0
                    dc = np.polynomial.chebyshev.chebder(c, m=k)
                    val = val + coeff * np.polynomial.chebyshev.chebval(Xj, dc)
                return val

            return psi_deriv

        raise RuntimeError("Composite basis must be built from Legendre or Chebyshev")


class DirichletLegendre(Composite, Legendre):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0)):
        Legendre.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags((1, -1), (0, 2), shape=(N + 1, N + 3), format="csr")

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.legendre(j, x) - sp.legendre(j + 2, x)
        # numpy.polynomial.Legendre.basis returns a callable-like object
        Pj = Leg.basis(j)
        Pj2 = Leg.basis(j + 2)
        return lambda Xj: Pj(Xj) - Pj2(Xj)


class NeumannLegendre(Composite, Legendre):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0), constraint=0):
        # Build Legendre-based composite space satisfying Neumann BCs.
        Legendre.__init__(self, N, domain=domain)
        self.B = Neumann(bc, domain, self.reference_domain)
        # Use symmetric stencil similar to NeumannChebyshev:
        # psi_i = Q_{i+1} - Q_{i-1} (missing terms treated as zero)
        self.S = sparse.diags((-1, 1), (-1, 1), shape=(N + 1, N + 3), format="csr")

    def basis_function(self, j, sympy=False):
        # psi_j = P_{j+1} - P_{j-1}
        if sympy:
            return sp.legendre(j + 1, x) - sp.legendre(j - 1, x)
        # numeric variant: guard negative index
        if j - 1 >= 0:
            return Leg.basis(j + 1) - Leg.basis(j - 1)
        else:
            return Leg.basis(j + 1)


class DirichletChebyshev(Composite, Chebyshev):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0)):
        Chebyshev.__init__(self, N, domain=domain)
        self.B = Dirichlet(bc, domain, self.reference_domain)
        self.S = sparse.diags((1, -1), (0, 2), shape=(N + 1, N + 3), format="csr")

    def basis_function(self, j, sympy=False):
        if sympy:
            return sp.cos(j * sp.acos(x)) - sp.cos((j + 2) * sp.acos(x))
        return Cheb.basis(j) - Cheb.basis(j + 2)


class NeumannChebyshev(Composite, Chebyshev):
    def __init__(self, N, domain=(-1, 1), bc=(0, 0), constraint=0):
        Chebyshev.__init__(self, N, domain=domain)
        # Neumann composite: use Neumann boundary and stencil that maps
        # psi_i = Q_{i+1} - Q_{i-1} (missing terms are treated as zero)
        self.B = Neumann(bc, domain, self.reference_domain)
        # stencil offsets -1 and +1 over a total column count of N+3
        self.S = sparse.diags((-1, 1), (-1, 1), shape=(N + 1, N + 3), format="csr")

    def basis_function(self, j, sympy=False):
        # psi_j = Q_{j+1} - Q_{j-1} where Q_k are Chebyshev polynomials
        if sympy:
            # If j is integer-like, handle small indices explicitly to avoid
            # symbolic cancellation when j is symbolic.
            try:
                j_val = int(j)
            except Exception:
                return sp.cos((j + 1) * sp.acos(x)) - sp.cos((j - 1) * sp.acos(x))
            if j_val - 1 < 0:
                return sp.cos((j_val + 1) * sp.acos(x))
            return sp.cos((j_val + 1) * sp.acos(x)) - sp.cos((j_val - 1) * sp.acos(x))
        # numeric variant: guard negative index
        if j - 1 >= 0:
            return Cheb.basis(j + 1) - Cheb.basis(j - 1)
        else:
            return Cheb.basis(j + 1)


class BasisFunction:
    def __init__(self, V, diff=0, argument=0):
        self._V = V
        self._num_derivatives = diff
        self._argument = argument

    @property
    def argument(self):
        return self._argument

    @property
    def function_space(self):
        return self._V

    @property
    def num_derivatives(self):
        return self._num_derivatives

    def diff(self, k):
        return self.__class__(self.function_space, diff=self.num_derivatives + k)


class TestFunction(BasisFunction):
    def __init__(self, V, diff=0):
        BasisFunction.__init__(self, V, diff=diff, argument=0)


class TrialFunction(BasisFunction):
    def __init__(self, V, diff=0):
        BasisFunction.__init__(self, V, diff=diff, argument=1)


def assemble_generic_matrix(u, v):
    assert isinstance(u, TrialFunction)
    assert isinstance(v, TestFunction)
    V = v.function_space
    assert u.function_space == V
    r = V.reference_domain
    D = np.zeros((V.N + 1, V.N + 1))
    cheb = V.weight() == 1 / sp.sqrt(1 - x**2)
    symmetric = True if u.num_derivatives == v.num_derivatives else False
    w = {"weight": "alg" if cheb else None, "wvar": (-0.5, -0.5) if cheb else None}

    def uv(Xj, i, j):
        return V.evaluate_derivative_basis_function(
            Xj, i, k=v.num_derivatives
        ) * V.evaluate_derivative_basis_function(Xj, j, k=u.num_derivatives)

    for i in range(V.N + 1):
        for j in range(i if symmetric else 0, V.N + 1):
            D[i, j] = quad(uv, float(r[0]), float(r[1]), args=(i, j), **w)[0]
            if symmetric:
                D[j, i] = D[i, j]
    return D


def inner(u, v: TestFunction):
    V = v.function_space
    h = V.domain_factor
    if isinstance(u, TrialFunction):
        num_derivatives = u.num_derivatives + v.num_derivatives
        if num_derivatives == 0:
            return float(h) * V.mass_matrix()
        else:
            return float(h) ** (1 - num_derivatives) * assemble_generic_matrix(u, v)
    return V.inner_product(u)


def project(ue, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    b = inner(ue, v)
    A = inner(u, v)
    uh = sparse.linalg.spsolve(A, b)
    return uh


def L2_error(uh, ue, V, kind="norm"):
    d = V.domain
    # map besselj via scipy.special.jv when lambdifying
    uej = sp.lambdify(x, ue, modules=[{"besselj": sc.jv}, "numpy"])

    def uv(xj):
        return (uej(xj) - V.eval(uh, xj)) ** 2

    return np.sqrt(quad(uv, float(d[0]), float(d[1]))[0])


def test_project():
    ue = sp.besselj(0, x)
    domain = (0, 10)
    for space in (Chebyshev, Legendre):
        V = space(16, domain=domain)
        u = project(ue, V)
        err = L2_error(u, ue, V)
        print(f"test_project: L2 error = {err:2.4e}, N = {V.N}, {V.__class__.__name__}")
        assert err < 1e-6


def test_helmholtz():
    ue = sp.besselj(0, x)
    f = ue.diff(x, 2) + ue
    domain = (0, 10)
    for space in (
        NeumannChebyshev,
        NeumannLegendre,
        DirichletChebyshev,
        DirichletLegendre,
        Sines,
        Cosines,
    ):
        if space in (NeumannChebyshev, NeumannLegendre, Cosines):
            bc = ue.diff(x, 1).subs(x, domain[0]), ue.diff(x, 1).subs(x, domain[1])
        else:
            bc = ue.subs(x, domain[0]), ue.subs(x, domain[1])
        N = 60 if space in (Sines, Cosines) else 12
        V = space(N, domain=domain, bc=bc)
        u = TrialFunction(V)
        v = TestFunction(V)
        A = inner(u.diff(2), v) + inner(u, v)
        b = inner(f, v)
        u_tilde = np.linalg.solve(A, b)
        err = L2_error(u_tilde, ue, V)
        print(f"test_helmholtz: L2 error = {err:2.4e}, N = {N}, {V.__class__.__name__}")
        assert err < 4


def test_convection_diffusion():
    eps = 0.05
    ue = (sp.exp(-x / eps) - 1) / (sp.exp(-1 / eps) - 1)
    f = 0
    domain = (0, 1)
    for space in (DirichletLegendre, DirichletChebyshev, Sines):
        N = 50 if space is Sines else 16
        V = space(N, domain=domain, bc=(0, 1))
        u = TrialFunction(V)
        v = TestFunction(V)
        A = inner(u.diff(2), v) + (1 / eps) * inner(u.diff(1), v)
        b = inner(f - ((1 / eps) * V.B.x.diff(x, 1)), v)
        u_tilde = np.linalg.solve(A, b)
        err = L2_error(u_tilde, ue, V)
        print(
            f"test_convection_diffusion: L2 error = {err:2.4e}, N = {N}, {V.__class__.__name__}"
        )
        assert err < 1e-3


if __name__ == "__main__":
    test_project()
    test_convection_diffusion()
    test_helmholtz()
