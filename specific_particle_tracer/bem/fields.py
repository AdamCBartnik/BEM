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
   *carved out of* the plane (a symmetric dimple/well) does NOT enjoy this
   same mirror trick, despite first appearances -- worked out directly
   while adding `CylindricalWellBEMField` below. A protrusion's vacuum
   domain (z>=0, exterior to the bump) is disjoint from its own mirror
   image (which sits entirely in z<=0), so the union is a clean closed
   surface with the plane relegated to an internal seam. A depression's
   vacuum domain already contains all of z>=0 (nothing excludes it
   there), so its mirror image (the well's own reflection, sitting in
   0<=z<=H) is a *subset* of the already-included z>=0 half, not a
   disjoint complement -- domain and mirror image overlap instead of
   tiling space, and the construction just doesn't produce a well-posed
   single surface to solve on. A well's field solve therefore does need
   the plane genuinely truncated (see `CylindricalWellBEMField` and
   `bem.mesh.cylindrical_well_real_profile`) -- the same open-surface
   treatment an asymmetric bump would need, not a shortcut around it.

Axisymmetric, not general 3D
-----------------------------
Both this shape and its excitation (a uniform field along the symmetry
axis) are axisymmetric, so this class solves via `bem.axisymmetric`
(a 1D generating-profile mesh, azimuthal integration done in closed form
via elliptic integrals). An earlier general 3D triangulated-mesh path
(via bempp-cl, direct + GMRES) was tried first and dropped once every
geometry this project actually needed -- axisymmetric bumps, axisymmetric
wells, and the genuinely off-axis image-charge job via `bem.image_charge`'s
azimuthal-Fourier-mode reduction -- turned out to fit a 1D reduction of
some kind; see git history for that path if a truly non-axisymmetric
geometry or excitation is ever needed. The axisymmetric solve is both far
cheaper (tens of profile nodes instead of thousands of triangles -- direct
linear solve, not GMRES) and at least as accurate: on-surface field error
(r = R exactly, where particles are actually emitted) shrinks monotonically
with profile resolution, matching the general 3D approach's accuracy (as
measured before that path was dropped) at a given resolution while costing
roughly two orders of magnitude less per field evaluation -- see
`bem.axisymmetric`'s module docstring for the near-surface regularization
this needs (the same "summing large, nearly-canceling contributions"
problem any triangulated-mesh Coulomb kernel hits too), for a normalization
bug (a missing factor of 2*pi) that a first version of this class's
validation caught via a basic shell-theorem sanity check (not the
point-by-point comparisons that had already passed), and for the
fully-vectorized (numpy or cupy)
field evaluation `evaluate` uses -- `xp='gpu'` here runs the field
evaluation itself on the GPU, not just the array bookkeeping around it.
"""

import numpy as np

from .mesh import sphere_profile, cylindrical_well_real_profile, _n_for_length
from .axisymmetric import AxisymmetricBEMSolution


def _cylindrical_well_contains(rho, z, R, H, bottom_fillet_radius, rim_fillet_radius, xp, eps=None):
    """True where (rho, z) is inside (or numerically on) the *real*
    cylindrical-well conductor boundary -- see
    `bem.mesh.cylindrical_well_real_profile`'s docstring for the shape,
    including its optional corner rounding. Reduces exactly to the
    simple two-piece sharp-corner rule (`bem.geometry.
    CylindricalWellBEMGeometry.kill_mask`'s original test, and this
    class's `evaluate`/`potential`'s) when both radii are 0 -- deliberately
    reusing those two branches' *exact* original expressions (including
    their own, not-quite-symmetric epsilon convention) rather than folding
    them into the fillet-region logic below, so the un-rounded case is
    guaranteed byte-identical to before this rounding feature existed.

    Shared by `CylindricalWellBEMField.evaluate`/`potential` (to zero the
    field/potential inside the conductor) and
    `bem.geometry.CylindricalWellBEMGeometry.kill_mask` (to kill a
    particle that re-enters it) -- both used to test against the old,
    always-sharp-cornered rule directly; once corners can be rounded,
    that rule is wrong exactly in the rounded region (confirmed directly:
    it shows up as a visibly wrong, sharp-edged "inside conductor" patch
    in a static-potential plot, sitting a bit outside where the actual
    rounded surface is).

    `eps` biases each fillet's own boundary very slightly toward "inside"
    (default 1e-6*R) so a particle sitting exactly on the rounded surface
    doesn't dither between the two sides step to step.
    """
    if eps is None:
        eps = 1e-6 * R
    outside = rho >= R * (1.0 - 1e-6)
    material = xp.where(outside, z <= 0.0, z <= -H * (1.0 + 1e-6))

    if rim_fillet_radius > 0.0:
        center_rho, center_z = R + rim_fillet_radius, -rim_fillet_radius
        in_region = (rho >= R - eps) & (rho <= R + rim_fillet_radius + eps) & (z >= -rim_fillet_radius - eps) & (z <= eps)
        dist = xp.hypot(rho - center_rho, z - center_z)
        material = xp.where(in_region, dist <= rim_fillet_radius, material)

    if bottom_fillet_radius > 0.0:
        center_rho, center_z = R - bottom_fillet_radius, -H + bottom_fillet_radius
        in_region = (
            (rho <= R + eps) & (rho >= R - bottom_fillet_radius - eps)
            & (z <= -H + bottom_fillet_radius + eps) & (z >= -H - eps)
        )
        dist = xp.hypot(rho - center_rho, z - center_z)
        material = xp.where(in_region, dist >= bottom_fillet_radius, material)

    return material


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
    max_length : float, optional
        Cap [m] on every profile segment's length, pole to pole -- see
        `bem.mesh.sphere_profile`/`bem.mesh._n_for_length`. Default is
        `R * np.pi / 39` (matching this class's old default of
        `n_theta=40`).
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

    def __init__(self, Ez, R, max_length=None, n_subdiv=8, xp=np):
        self.Ez = float(Ez)
        self.R = float(R)
        self.xp = xp
        self.n_subdiv = n_subdiv

        max_length = self.R * np.pi / 39.0 if max_length is None else max_length
        n_theta = _n_for_length(self.R * np.pi, max_length)
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

    def potential(self, position):
        """Potential [V] at the given positions: Phi_uniform + Phi_pert
        (from the BEM solve), same masking convention as `evaluate` (0
        inside the conductor -- physically correct, not just a
        placeholder). Always numpy, regardless of `self.xp` -- like
        `bem.axisymmetric.AxisymmetricBEMSolution.potential` itself
        (an adaptive-quadrature reference path), this isn't the
        vectorized/GPU-capable path `evaluate` is; fine for a one-off
        density plot (see `plotting.static_potential_grid`), not meant
        for a hot loop.

        Parameters
        ----------
        position : ndarray, shape (..., 3)

        Returns
        -------
        V : ndarray, shape position.shape[:-1]
        """
        position = np.asarray(position, dtype=float)
        r = np.linalg.norm(position, axis=-1)
        valid = (r >= self.R * (1.0 - 1e-6)) & (position[..., 2] >= 0.0)
        V_uniform = -self.Ez * position[..., 2]
        V_pert = self.solution.potential(position)
        return np.where(valid, V_uniform + V_pert, 0.0)


class CylindricalWellBEMField:
    """BEM solve for a cylindrical well (radius R, depth H) cut into an
    infinite flat cathode, in a uniform asymptotic field Ez.

    Unlike `HemisphericalTipBEMField`, this shape can't avoid meshing the
    plane -- see the module docstring for why a depression's mirror image
    overlaps its own real domain instead of tiling space the way a
    protrusion's does. So this solves the real, open, truncated-plane
    profile directly (`bem.mesh.cylindrical_well_real_profile`): the same
    finite-`plane_radius` truncation approximation
    `bem.mesh.hemisphere_tip_image_profile`'s image surfaces already rely
    on for the image-charge job (there, already validated to agree with
    the exact closed-form solution to well under 1% near the shape), now
    used for the field job too.

    Parameters
    ----------
    Ez : float
        Asymptotic field [V/m], same convention as `fields.GunField.Ez`.
    R, H : float
        Well radius and depth [m] (well bottom is the disk z=-H, rho<=R).
    plane_radius : float, optional
        Radius the surrounding (truncated) plane is meshed out to.
        Default 20*max(R, H), matching this project's default far-field
        handoff distance (see `bem.geometry.CylindricalWellBEMGeometry`)
        -- past that distance the field is treated as a plain flat
        cathode's anyway, so there is little to gain from meshing the
        plane any farther than that.
    max_length : float, optional
        Global cap [m] on every profile segment's length -- see
        `bem.mesh.cylindrical_well_real_profile`. Default `min(R, H)/19`
        (matching this class's old `n_bottom=n_wall=20`). Safe to apply
        uniformly out to the plane's far edge despite `plane_radius`
        being a large multiple of `R`/`H`, since the plane region is
        *graded* (see `bem.mesh.graded_annulus_profile`): it only
        resolves this finely right at its inner edge, next to the
        feature, and coarsens geometrically from there.
    bottom_fillet_radius, rim_fillet_radius : float or None, optional
        Rounding radii [m] for the bottom-of-well/rim corners, passed
        straight through to `bem.mesh.cylindrical_well_real_profile` --
        see that function's docstring, and `bem.geometry.
        CylindricalWellBEMGeometry`'s, for why a real fabricated well
        may not have perfectly sharp corners. Default None (0, sharp)
        for either, matching this class's original shape.
    bottom_max_length, bottom_fillet_max_length, wall_max_length,
    rim_fillet_max_length, plane_max_length : float, optional
        Per-region overrides for the bottom disk / bottom-corner fillet /
        wall / rim fillet / plane, passed straight through to
        `bem.mesh.cylindrical_well_real_profile`.
    n_subdiv : int, optional
        Passed to `bem.axisymmetric.evaluate_axisymmetric_field`.
    xp : module, optional
        numpy or cupy -- see `HemisphericalTipBEMField`'s docstring for
        which part of the computation this controls (the solve itself is
        always numpy/CPU; only `evaluate` runs on `xp`).
    """

    def __init__(
        self, Ez, R, H, plane_radius=None, max_length=None,
        bottom_fillet_radius=None, rim_fillet_radius=None,
        bottom_max_length=None, bottom_fillet_max_length=None, wall_max_length=None,
        rim_fillet_max_length=None, plane_max_length=None,
        n_subdiv=8, xp=np,
    ):
        self.Ez = float(Ez)
        self.R = float(R)
        self.H = float(H)
        self.bottom_fillet_radius = 0.0 if bottom_fillet_radius is None else float(bottom_fillet_radius)
        self.rim_fillet_radius = 0.0 if rim_fillet_radius is None else float(rim_fillet_radius)
        self.plane_radius = 20.0 * max(self.R, self.H) if plane_radius is None else plane_radius
        self.xp = xp
        self.n_subdiv = n_subdiv

        max_length = min(self.R, self.H) / 19.0 if max_length is None else max_length
        profile = cylindrical_well_real_profile(
            self.R, self.H, self.plane_radius, max_length,
            bottom_fillet_radius=bottom_fillet_radius, rim_fillet_radius=rim_fillet_radius,
            bottom_max_length=bottom_max_length, bottom_fillet_max_length=bottom_fillet_max_length,
            wall_max_length=wall_max_length, rim_fillet_max_length=rim_fillet_max_length,
            plane_max_length=plane_max_length,
        )
        Ez_ = self.Ez
        self.solution = AxisymmetricBEMSolution.solve(profile, dirichlet_fn=lambda rho, z: Ez_ * z)

    def evaluate(self, position):
        """Evaluate the field at the given positions.

        Parameters
        ----------
        position : ndarray, shape (..., 3)

        Returns
        -------
        E : ndarray, shape (..., 3)
            Electric field [V/m] at each position. Zero inside the solid
            conductor -- see `_cylindrical_well_contains` for the exact
            (possibly corner-rounded) boundary.
        """
        xp = self.xp
        position = xp.asarray(position, dtype=float)
        rho = xp.linalg.norm(position[..., :2], axis=-1)
        z = position[..., 2]
        valid = ~_cylindrical_well_contains(
            rho, z, self.R, self.H, self.bottom_fillet_radius, self.rim_fillet_radius, xp
        )

        E_uniform = xp.zeros_like(position)
        E_uniform[..., 2] = self.Ez

        E_pert = self.solution.field(position, n_subdiv=self.n_subdiv, xp=xp)

        E = E_uniform + E_pert
        return xp.where(valid[..., None], E, 0.0)

    def potential(self, position):
        """Potential [V] at the given positions, same masking convention
        as `evaluate` (0 inside the conductor -- physically correct).
        Always numpy, regardless of `self.xp` -- see
        `HemisphericalTipBEMField.potential`'s docstring for why (a
        diagnostic/plotting path, not the vectorized/GPU one).

        Parameters
        ----------
        position : ndarray, shape (..., 3)

        Returns
        -------
        V : ndarray, shape position.shape[:-1]
        """
        position = np.asarray(position, dtype=float)
        rho = np.linalg.norm(position[..., :2], axis=-1)
        z = position[..., 2]
        valid = ~_cylindrical_well_contains(
            rho, z, self.R, self.H, self.bottom_fillet_radius, self.rim_fillet_radius, np
        )
        V_uniform = -self.Ez * z
        V_pert = self.solution.potential(position)
        return np.where(valid, V_uniform + V_pert, 0.0)
