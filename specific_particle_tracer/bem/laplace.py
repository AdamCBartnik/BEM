"""Exterior Laplace boundary element solver: given a conductor surface mesh
and Dirichlet data on it, solve for the induced surface charge (Neumann
data), and evaluate the resulting potential/field anywhere in the exterior
region.

Meant to be used with the superposition trick for a cathode's static field:
the true total potential diverges at infinity (it contains the uniform
asymptotic gun field itself), which a finite, truncated mesh cannot
represent as a boundary condition. Instead, solve only for the
*perturbation* phi_pert = phi_total - phi_inf. Its boundary data is
phi_pert = -phi_inf on the real conductor surface (so that phi_pert +
phi_inf, the total, comes out to the required 0 on a grounded conductor),
and phi_pert itself decays like a dipole field at large distance -- a
boundary condition a finite mesh represents well, unlike the diverging
total field. See bem.fields.HemisphericalTipBEMField for the concrete
hemisphere-tip application of this.
"""

import numpy as np
import bempp_cl.api as bempp_api


class ExteriorLaplaceSolution:
    """The solution (Dirichlet + solved Neumann data) of an exterior Laplace
    Dirichlet problem on one conductor surface mesh: phi harmonic outside
    the surface, matching given Dirichlet data g on the surface, decaying
    at infinity.

    Build with `ExteriorLaplaceSolution.solve(vertices, elements, dirichlet_fn)`.
    """

    def __init__(self, space, dirichlet_fun, neumann_fun):
        self.space = space
        self.dirichlet_fun = dirichlet_fun
        self.neumann_fun = neumann_fun

    @classmethod
    def solve(cls, vertices, elements, dirichlet_fn, gmres_tol=1e-8):
        """Solve for the surface Neumann data given Dirichlet data g.

        Uses the standard direct boundary-integral formulation for the
        exterior Dirichlet problem: (0.5*I + K) g = V t, solved for the
        unknown Neumann trace t by GMRES (K the double-layer operator, V
        the single-layer operator, I the identity/mass matrix).

        Parameters
        ----------
        vertices, elements : the mesh, BEMpp convention (see bem.mesh).
        dirichlet_fn : callable(x, y, z) -> float
            The Dirichlet (potential) data on the surface.
        gmres_tol : float
            Relative residual tolerance for the GMRES solve.
        """
        grid = bempp_api.Grid(vertices, elements)
        space = bempp_api.function_space(grid, "P", 1)

        # jit=False: dirichlet_fn is an arbitrary Python closure (not
        # something numba's nopython mode can compile), and these
        # boundary/potential evaluations are dominated by the BEM matrix
        # assembly and linear solve anyway, not by this per-point callback.
        @bempp_api.real_callable(jit=False)
        def _dirichlet_data(x, n, domain_index, result):
            result[0] = dirichlet_fn(x[0], x[1], x[2])

        dirichlet_fun = bempp_api.GridFunction(space, fun=_dirichlet_data)

        slp = bempp_api.operators.boundary.laplace.single_layer(space, space, space)
        dlp = bempp_api.operators.boundary.laplace.double_layer(space, space, space)
        identity = bempp_api.operators.boundary.sparse.identity(space, space, space)

        rhs = (0.5 * identity + dlp) * dirichlet_fun
        neumann_fun, info = bempp_api.linalg.gmres(slp, rhs, tol=gmres_tol)
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info})")

        return cls(space, dirichlet_fun, neumann_fun)

    def potential(self, points):
        """Perturbation potential phi_pert at `points`, shape (..., 3).

        Returns an array of shape points.shape[:-1], via the exterior
        representation formula phi_pert(x) = (S t)(x) - (D g)(x).
        """
        points = np.asarray(points, dtype=float)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3).T  # bempp point convention: (3, n)

        slp_pot = bempp_api.operators.potential.laplace.single_layer(self.space, flat)
        dlp_pot = bempp_api.operators.potential.laplace.double_layer(self.space, flat)
        phi = (slp_pot * self.neumann_fun - dlp_pot * self.dirichlet_fun).ravel()
        return phi.reshape(shape)

    def field(self, points, fd_step):
        """Perturbation field E_pert = -grad(phi_pert) at `points`, via
        central finite differences with step `fd_step` [m].

        BEMpp's potential operators only give the potential itself, not its
        gradient, so this evaluates all 6 stencil offsets (for all points
        at once, to amortize potential-operator assembly) rather than
        differentiating analytically.

        Parameters
        ----------
        points : ndarray, shape (..., 3)
        fd_step : float

        Returns
        -------
        E : ndarray, same shape as `points`.
        """
        points = np.asarray(points, dtype=float)
        shape = points.shape
        flat = points.reshape(-1, 3)
        n = flat.shape[0]

        stacked = np.tile(flat, (6, 1))
        for axis in range(3):
            stacked[2 * axis * n : (2 * axis + 1) * n, axis] += fd_step
            stacked[(2 * axis + 1) * n : (2 * axis + 2) * n, axis] -= fd_step

        phi = self.potential(stacked).reshape(6, n)
        grad = np.empty((n, 3))
        for axis in range(3):
            grad[:, axis] = (phi[2 * axis] - phi[2 * axis + 1]) / (2.0 * fd_step)

        return (-grad).reshape(shape)
