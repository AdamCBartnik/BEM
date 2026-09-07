"""Exterior Laplace boundary element solver, via the indirect (charge-
simulation) method: represent the potential as phi_pert(x) = S[sigma](x)
for x exterior, where S is the single-layer potential and sigma an unknown
surface density, solved so that S[sigma] matches given Dirichlet data on
the boundary -- a first-kind boundary integral equation V sigma = g (V the
single-layer boundary operator).

This is deliberately simpler than the "direct" formulation (solve for a
Neumann trace t given the Dirichlet trace g, reconstruct the potential as
S[t] - D[g]) this project tried first: reconstructing a *field* from that
mixed representation needs the gradient of the double-layer potential, a
hypersingular (~1/r^3) kernel that turned out to be unusable close to a
panel (see git history, and bem.panel_field's module docstring). Indirect/
CSM needs only the single-layer potential's own gradient (~1/r^2, the same
kernel this project's Coulomb-law term already validated cleanly), at the
cost of solving a first-kind rather than second-kind system -- generally
more ill-conditioned, so if GMRES struggles or accuracy stays poor close
to a panel, a first thing to try is a dedicated near-singular quadrature
transform (Telles' or the Johnston-Elliott sinh transform) on this
module's own Coulomb integral, not reverting to the direct formulation.

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

from .panel_field import evaluate_coulomb_field


class ExteriorLaplaceSolution:
    """The solution (Dirichlet data + solved surface charge density) of an
    exterior Laplace Dirichlet problem on one conductor surface mesh: phi
    harmonic outside the surface, matching given Dirichlet data g on the
    surface, decaying at infinity.

    Build with `ExteriorLaplaceSolution.solve(vertices, elements, dirichlet_fn)`.
    """

    def __init__(self, space, dirichlet_fun, sigma_fun):
        self.space = space
        self.dirichlet_fun = dirichlet_fun
        self.sigma_fun = sigma_fun

    @classmethod
    def solve(cls, vertices, elements, dirichlet_fn, gmres_tol=1e-8):
        """Solve for the surface charge density sigma whose single-layer
        potential matches the given Dirichlet data g (the indirect/charge-
        simulation method -- see module docstring).

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
        sigma_fun, info = bempp_api.linalg.gmres(slp, dirichlet_fun, tol=gmres_tol)
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info})")

        return cls(space, dirichlet_fun, sigma_fun)

    def potential(self, points):
        """Perturbation potential phi_pert = S[sigma] at `points`, shape
        (..., 3). Returns an array of shape points.shape[:-1]."""
        points = np.asarray(points, dtype=float)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3).T  # bempp point convention: (3, n)

        slp_pot = bempp_api.operators.potential.laplace.single_layer(self.space, flat)
        phi = (slp_pot * self.sigma_fun).ravel()
        return phi.reshape(shape)

    def field(self, points, refine_ratio=1.0, max_depth=6):
        """Perturbation field E_pert = -grad(phi_pert) at `points`, by
        direct adaptive-quadrature integration of the Coulomb kernel over
        every mesh panel -- see `bem.panel_field` (no finite differences).

        DOF index i corresponds exactly to `vertices[:, i]` for the P1
        spaces used throughout this project (checked against BEMpp's
        `space.cell_dofs`), so `GridFunction.coefficients` can be used
        directly as nodal values.

        Parameters
        ----------
        points : ndarray, shape (..., 3)
        refine_ratio, max_depth : see `panel_field.evaluate_coulomb_field`.

        Returns
        -------
        E : ndarray, same shape as `points`.
        """
        grid = self.space.grid
        sigma_nodal = np.real(self.sigma_fun.coefficients)
        return evaluate_coulomb_field(
            grid.vertices, grid.elements, sigma_nodal, points, refine_ratio=refine_ratio, max_depth=max_depth
        )
