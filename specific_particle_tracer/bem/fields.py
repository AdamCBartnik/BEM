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

Known limitation: `evaluate` gets its field from `bem.panel_field` (direct
Coulomb-law integration of the solved surface charge over every mesh
panel -- see `bem.laplace`/`bem.panel_field` for why this indirect/charge-
simulation formulation was chosen over the mixed direct one). That's
accurate close to the tip -- sub-percent error by r ~ 1.05R, a few percent
by r ~ 1.01R -- but still not accurate *exactly* on the mesh surface
(r = R exactly), where error is still of order 100%. The mesh is flat-
faceted, not curved, so r = R (the analytic sphere's own surface) sits
just barely *outside* the discretized geometry except at mesh vertices --
a standoff that shrinks, rather than grows, under mesh refinement, so this
isn't fixed by a finer mesh either. If exact on-surface accuracy is
needed, the next things to try, in rough order of effort: (a) a dedicated
near-singular quadrature transform (Telles' or the Johnston-Elliott sinh
transform) in `bem.panel_field`, since the current adaptive-subdivision
scheme plateaus rather than converging as the standoff shrinks; or (b) the
closed-form jump relation for a single-layer potential's normal
derivative, which decomposes the on-surface field into sigma/2 (known)
plus the *weakly* singular (not hypersingular) adjoint-double-layer
operator applied to sigma -- both already directly available from bempp,
and this sidesteps near-panel quadrature for the on-surface case entirely.
Neither is implemented yet. Since particles are emitted essentially at
r = R, this matters for any real usage.
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
        Relative residual tolerance for the boundary-integral GMRES solve
        (a first-kind system -- see `bem.laplace` -- so this solve is
        noticeably slower than the direct formulation's was, though it
        still converges fine at the resolutions tried so far).
    refine_ratio, max_depth : optional
        Passed through to `bem.panel_field.evaluate_coulomb_field`.
    xp : module, optional
        Only used to shape/type the returned field array the same way
        `fields.HemisphericalTipField` does; the BEM solve itself always
        runs on numpy/bempp.
    """

    def __init__(self, Ez, R, n_theta=40, n_phi=48, gmres_tol=1e-8, refine_ratio=1.0, max_depth=6, xp=np):
        self.Ez = float(Ez)
        self.R = float(R)
        self.xp = xp
        self.refine_ratio = refine_ratio
        self.max_depth = max_depth

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

        E_pert = self.solution.field(position, refine_ratio=self.refine_ratio, max_depth=self.max_depth)

        E = E_uniform + E_pert
        E = np.where(valid[..., None], E, 0.0)
        return self.xp.asarray(E)
