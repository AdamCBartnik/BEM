"""BEM counterpart to `fields.HemisphericalTipField`: the static field for a
grounded hemispherical tip on an infinite flat cathode, sitting in a
uniform asymptotic field Ez, computed by solving a boundary element Laplace
problem instead of using the closed-form solution.

This exists to validate the analytic solution (and, eventually, to extend
to shapes that don't have one) -- `HemisphericalTipBEMField.evaluate` is a
drop-in comparison against `fields.HemisphericalTipField.evaluate`.

Two things make this trickier than a plain "mesh the conductor and solve"
BEM problem:

1. The real cathode plane is infinite, but a mesh must be finite, and the
   diverging (non-decaying) part of the total field -- the uniform
   asymptotic Ez itself -- is exactly what a finite mesh can't represent as
   a boundary condition. Solved with the superposition trick in
   `bem.laplace`: solve only for the perturbation phi_pert = phi_total -
   phi_inf, which decays like a dipole field and truncates well, then add
   the uniform field back analytically.

2. Truncating the *plane* itself (meshing a finite disk of it) introduces
   an artificial free edge/rim, which the direct boundary-integral
   formulation in `bem.laplace` isn't valid for -- it assumes a closed
   surface (no boundary), and empirically produces a badly-conditioned or
   even non-convergent GMRES solve on an open, rimmed mesh. This is
   sidestepped here (as it is in the analytic solution) rather than solved
   in general: a hemispherical tip's mirror image through z=0 is exactly
   the lower hemisphere, so the tip's full mirrored shape is simply the
   complete sphere of radius R. Solving the closed-sphere problem with
   Dirichlet data phi_pert = -phi_inf = Ez*z (odd under z -> -z) gives a
   solution that is itself odd about z=0, which is automatically zero on
   the z=0 plane everywhere outside the tip -- exactly the boundary
   condition the surrounding flat cathode needs, with no separate plane
   image and no plane meshing at all. This is precisely the same symmetry
   argument `fields.HemisphericalTipField`'s docstring gives for why its
   analytic solution needs no separate plane image either.

   A tip shape without that convenient mirror symmetry will eventually need
   the plane genuinely truncated and the open-surface case handled (e.g. by
   closing the mesh with an artificial distant boundary, or using an
   open-surface BIE formulation) -- left for when a non-symmetric geometry
   is actually attempted. A shape *carved out of* the plane (a symmetric
   dimple/well) would still enjoy this same mirror trick -- its mirror
   image is a closed cavity rather than a closed bump, which just flips the
   problem to an interior Dirichlet solve -- but an asymmetric carved shape
   has the identical open-surface problem an asymmetric bump does.

Known limitation: `evaluate` is only validated (against
`fields.HemisphericalTipField`) away from the tip surface -- error is a
fraction of a percent by r ~ 3R and shrinks further with mesh resolution,
but grows to tens of percent right at r ~ R. That's because `evaluate` gets
the field by finite-differencing the *potential*, and finite-differencing
right next to (or on) a boundary element is a well-known hard case for BEM
(near-singular quadrature) -- the potential itself is only resolved to
mesh/quadrature accuracy, and differencing amplifies that error close to
the surface. The right fix is to read the field directly off the already-
solved surface charge (Neumann trace) instead of differencing the
potential -- on a grounded conductor the field is purely normal, with
magnitude set by the surface charge density -- but that requires correctly
mapping the solved Neumann `GridFunction`'s DOF coefficients back to
physical surface points (its coefficients are not simply per-input-vertex
values), which isn't done yet. Since particles are emitted essentially at
r = R, this matters for any real usage and is the natural next piece of
this module, not yet attempted.
"""

import numpy as np

from .mesh import sphere_mesh
from .laplace import ExteriorLaplaceSolution


class HemisphericalTipBEMField:
    """BEM solve for `fields.HemisphericalTipField`'s geometry: a grounded
    hemispherical tip of radius R on an infinite flat cathode, in a uniform
    asymptotic field Ez.

    Only the full sphere (radius R) is meshed, not the surrounding plane --
    see the module docstring for why that's exact for this particular
    (mirror-symmetric) shape.

    Parameters
    ----------
    Ez : float
        Asymptotic field [V/m] far from the tip, same sign convention as
        `fields.GunField.Ez`/`fields.HemisphericalTipField.Ez`.
    R : float
        Tip radius [m].
    n_theta, n_phi : int, optional
        Mesh resolution -- see `bem.mesh.sphere_mesh`.
    gmres_tol : float, optional
        Relative residual tolerance for the boundary-integral GMRES solve.
    fd_step : float or None, optional
        Finite-difference step [m] used to get the field from the solved
        potential (see `ExteriorLaplaceSolution.field`). Default 1e-3 * R,
        picked empirically: small enough to resolve the field's curvature,
        large enough that the difference isn't swamped by the BEM solve's
        own (mesh- and quadrature-limited) potential error -- see the
        module docstring's near-surface caveat, which this step size does
        not fix.
    xp : module, optional
        Only used to shape/type the returned field array the same way
        `fields.HemisphericalTipField` does; the BEM solve itself always
        runs on numpy/bempp.
    """

    def __init__(self, Ez, R, n_theta=40, n_phi=48, gmres_tol=1e-8, fd_step=None, xp=np):
        self.Ez = float(Ez)
        self.R = float(R)
        self.xp = xp
        self.fd_step = fd_step if fd_step is not None else 1e-3 * self.R

        vertices, elements = sphere_mesh(self.R, n_theta=n_theta, n_phi=n_phi)
        Ez_ = self.Ez
        # phi_inf(z) = -Ez*z, so the Dirichlet data forcing the total
        # potential to zero on the (grounded) sphere is g = -phi_inf = Ez*z.
        self.solution = ExteriorLaplaceSolution.solve(
            vertices, elements, dirichlet_fn=lambda x, y, z: Ez_ * z, gmres_tol=gmres_tol
        )

    def evaluate(self, position):
        """Evaluate the field at the given positions.

        Parameters
        ----------
        position : ndarray, shape (..., 3)

        Returns
        -------
        E : ndarray, shape (..., 3)
            Electric field [V/m] at each position. Zero outside the region
            this solve is valid for (r < R, or z < 0 away from the tip) --
            same convention as `fields.HemisphericalTipField.evaluate`.
        """
        position = np.asarray(position, dtype=float)
        r = np.linalg.norm(position, axis=-1)
        valid = (r >= self.R * (1.0 - 1e-6)) & (position[..., 2] >= 0.0)

        E_uniform = np.zeros_like(position)
        E_uniform[..., 2] = self.Ez

        E_pert = self.solution.field(position, self.fd_step)

        E = E_uniform + E_pert
        E = np.where(valid[..., None], E, 0.0)
        return self.xp.asarray(E)
