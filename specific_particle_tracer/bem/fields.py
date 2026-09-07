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
   a boundary condition. Solved with the superposition trick: solve only
   for the perturbation phi_pert = phi_total - phi_inf, which decays like a
   dipole field and truncates well, then add the uniform field back
   analytically.

2. Truncating the *plane* itself introduces an artificial free edge/rim
   that the direct boundary-integral formulation isn't valid for. Instead
   of solving that in general, this shape's own mirror symmetry sidesteps
   it: a hemispherical tip's mirror image through z=0 is exactly the lower
   hemisphere, so the tip's full mirrored shape is simply the complete
   sphere of radius R. Solving the closed-sphere problem with Dirichlet
   data phi_pert = -phi_inf = Ez*z (odd under z -> -z) gives a solution
   that is itself odd about z=0, automatically zero on the z=0 plane
   everywhere outside the tip -- exactly the boundary condition the
   surrounding flat cathode needs, with no separate plane image and no
   plane meshing at all. This is precisely the same symmetry argument
   `fields.HemisphericalTipField`'s docstring gives for why its analytic
   solution needs no separate plane image either.

   A tip shape without that convenient mirror symmetry will eventually need
   the plane genuinely truncated and the open-surface case handled -- left
   for when a non-symmetric geometry is actually attempted. A shape
   *carved out of* the plane (a symmetric dimple/well) would still enjoy
   this same mirror trick -- its mirror image is a closed cavity rather
   than a closed bump, which just flips the problem to an interior
   Dirichlet solve -- but an asymmetric carved shape has the identical
   open-surface problem an asymmetric bump does.

Axisymmetric, not general 3D
-----------------------------
Both this shape and its excitation (a uniform field along the symmetry
axis) are axisymmetric, so this class solves via `bem.axisymmetric`
(a 1D generating-profile mesh, azimuthal integration done in closed form
via elliptic integrals) rather than the general 3D machinery in
`bem.mesh`/`bem.laplace`/`bem.panel_field`. That general 3D path is kept
for when a future geometry or excitation isn't axisymmetric -- it remains
tested and usable, just not what this class uses by default. The
axisymmetric solve is both far cheaper (tens of profile nodes instead of
thousands of triangles -- direct linear solve, not GMRES) and at least as
accurate: on-surface field error (r = R exactly, where particles are
actually emitted) shrinks monotonically with profile resolution, matching
the general 3D approach's accuracy at a given resolution while costing
roughly two orders of magnitude less per field evaluation -- see
`bem.axisymmetric`'s module docstring for the near-surface regularization
this shares with `bem.panel_field`, for a normalization bug (a missing
factor of 2*pi) that a first version of this class's validation caught via
a basic shell-theorem sanity check (not the point-by-point comparisons
that had already passed), and for the fully-vectorized (numpy or cupy)
field evaluation `evaluate` uses -- `xp='gpu'` here runs the field
evaluation itself on the GPU, not just the array bookkeeping around it.
"""

import numpy as np

from .mesh import sphere_profile
from .axisymmetric import AxisymmetricBEMSolution


class HemisphericalTipBEMField:
    """BEM solve for `fields.HemisphericalTipField`'s geometry: a grounded
    hemispherical tip of radius R on an infinite flat cathode, in a uniform
    asymptotic field Ez.

    The full sphere (radius R) is used as the generating profile, not just
    the surrounding plane -- see the module docstring for why that's exact
    for this particular (mirror-symmetric) shape.

    Parameters
    ----------
    Ez : float
        Asymptotic field [V/m] far from the tip, same sign convention as
        `fields.GunField.Ez`/`fields.HemisphericalTipField.Ez`.
    R : float
        Tip radius [m].
    n_theta : int, optional
        Number of profile nodes from pole to pole -- see
        `bem.mesh.sphere_profile`.
    n_subdiv : int, optional
        Fixed (non-adaptive) sub-panels per profile segment used by the
        cached field-evaluation quadrature -- see
        `bem.axisymmetric.evaluate_axisymmetric_field`.
    xp : module, optional
        numpy or cupy. The BEM *solve* always runs on numpy/CPU (a small,
        one-time direct linear solve -- no benefit from the GPU there),
        but `evaluate`'s field computation runs entirely on this backend,
        including the elliptic-integral kernel evaluations themselves
        (see `bem.axisymmetric._ellip_ke`).
    """

    def __init__(self, Ez, R, n_theta=40, n_subdiv=8, xp=np):
        self.Ez = float(Ez)
        self.R = float(R)
        self.xp = xp
        self.n_subdiv = n_subdiv

        profile = sphere_profile(self.R, n_theta=n_theta)
        Ez_ = self.Ez
        # phi_inf(z) = -Ez*z, so the Dirichlet data forcing the total
        # potential to zero on the (grounded) sphere is g = -phi_inf = Ez*z.
        self.solution = AxisymmetricBEMSolution.solve(profile, dirichlet_fn=lambda rho, z: Ez_ * z)

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
        xp = self.xp
        position = xp.asarray(position, dtype=float)
        r = xp.linalg.norm(position, axis=-1)
        valid = (r >= self.R * (1.0 - 1e-6)) & (position[..., 2] >= 0.0)

        E_uniform = xp.zeros_like(position)
        E_uniform[..., 2] = self.Ez

        E_pert = self.solution.field(position, n_subdiv=self.n_subdiv, xp=xp)

        E = E_uniform + E_pert
        return xp.where(valid[..., None], E, 0.0)
