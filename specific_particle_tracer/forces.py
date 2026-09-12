"""Pairwise Coulomb repulsion (with Plummer softening) and image-charge
attraction toward a perfectly-conducting cathode, for the shapes in
fields.py/geometry.py.

Groups do not interact with each other, so all pairwise sums are only
taken within the last axis of a (n_groups, n_emit, 3) position array. This
keeps the whole thing a batch of small, independent N-body problems, which
vectorizes well and is the natural unit of work for the GPU backend.
"""

import numpy as np

from .constants import COULOMB_CONSTANT


def _pairwise_force(position_i, position_j, charge_i, charge_j, active, plummer_radius, xp):
    """Sum_j k*charge_i*charge_j*(r_i-r_j)/(|r_i-r_j|^2+a^2)^1.5, batched
    over an (n_groups, n_emit, ...) leading shape, masked to active-active
    pairs on both `position_i` (i) and `position_j` (j).
    """
    diff = position_i[:, :, None, :] - position_j[:, None, :, :]  # r_i - r_j
    dist2 = xp.sum(diff * diff, axis=-1) + plummer_radius**2
    dist3 = dist2 * xp.sqrt(dist2)

    pair_mask = active[:, :, None] & active[:, None, :]
    coeff = COULOMB_CONSTANT * charge_i[:, :, None] * charge_j[:, None, :] / dist3
    coeff = xp.where(pair_mask, coeff, 0.0)

    return xp.einsum("gij,gijc->gic", coeff, diff)


def coulomb_force(position, charge, active, plummer_radius, xp=np):
    """Plummer-softened pairwise Coulomb repulsion between the (real)
    particles of each group.

    Parameters
    ----------
    position : ndarray, shape (n_groups, n_emit, 3)
        Positions [m].
    charge : ndarray, shape (n_groups, n_emit)
        Signed per-particle charge [C].
    active : ndarray of bool, shape (n_groups, n_emit)
        Only particles that are currently active (born) exert and feel
        forces.
    plummer_radius : float
        Softening length [m].
    xp : module, optional
        Array backend (numpy or cupy). Default numpy.

    Returns
    -------
    force : ndarray, shape (n_groups, n_emit, 3)
        Net Coulomb force [N] on each particle from the other members of
        its own group. The i == j (self) term is automatically zero since
        r_i - r_i = 0.
    """
    return _pairwise_force(position, position, charge, charge, active, plummer_radius, xp)


def image_charge_force(position, charge, active, z0, plummer_radius, xp=np):
    """Attraction of each particle toward the image charges induced in a
    flat, perfectly-conducting cathode, restricted (like Coulomb) to the
    particles within one group.

    The cathode is treated, for this purpose only, as if it sat at
    z = -z0 rather than z = 0: the image of a particle at height z is a
    charge of the opposite sign at z_image = -2*z0 - z, i.e. even a
    particle sitting exactly on the real surface (z = 0) has a finite
    2*z0 separation from its image, which regularizes the force that would
    otherwise diverge right at the surface.

    Every real particle in a group interacts with the image of every real
    particle in that group, including its own image (unlike the Coulomb
    sum, this self term is generally nonzero and is the dominant
    contribution for a single emitted particle).

    Parameters
    ----------
    position : ndarray, shape (n_groups, n_emit, 3)
    charge : ndarray, shape (n_groups, n_emit)
    active : ndarray of bool, shape (n_groups, n_emit)
    z0 : float
        Effective image-plane offset [m] (default 3.0 nm at the tracker
        level).
    plummer_radius : float
        Same softening length used for the real-real Coulomb sum, applied
        here too for robustness (e.g. if a user sets z0 = 0).
    xp : module, optional

    Returns
    -------
    force : ndarray, shape (n_groups, n_emit, 3)
        Net image-charge force [N] on each particle.
    """
    image_position = position.copy()
    image_position[..., 2] = -2.0 * z0 - position[..., 2]

    return _pairwise_force(position, image_position, charge, -charge, active, plummer_radius, xp)


def hemispherical_tip_image_force(position, charge, active, sphere_radius, plummer_radius, *, plane_z0, xp=np):
    """Attraction of each particle toward the image charges induced in a
    grounded hemispherical tip of radius `sphere_radius` at the origin on
    an infinite grounded plane, restricted (like Coulomb) to the particles
    within one group.

    3 images per real charge, not just 1. For a real charge q at position
    r0 (distance d = |r0| > a from the origin, with a = `sphere_radius`):

      1. Sphere image: charge -q*a/d at a^2/d^2 * r0 (the classic
         single-image construction for a grounded sphere -- makes the
         sphere r=a an exact equipotential by itself).
      2. Plane image: charge -q at r0 reflected through z=-plane_z0.
      3. Plane image of the sphere image: charge +q*a/d at the sphere
         image's position, also reflected through z=-plane_z0.

    All three sit strictly inside the solid conductor (the sphere images
    at radius a^2/d < a, i.e. inside the tip; the plane images below the
    plane, inside the bulk cathode), which is what makes them valid images
    in the first place. Unlike using the sphere image alone, this
    reproduces the correct flat-mirror-like behavior for a particle
    anywhere on the flat part of the cathode, not just right at the tip.

    Both surfaces are recessed by the same z0 this project's other
    image-charge paths use, so that no particle ever sees a divergent
    force at a real surface: pass `sphere_radius` = R - z0 (R the physical
    tip radius) *and* `plane_z0` = z0, matching `image_charge_force`'s own
    z_image = -2*z0 - z convention for the flat cathode. Recessing only
    the sphere (`plane_z0=0`) would leave the flat part of the cathode
    unregularized -- a particle approaching z=0 out beyond the tip would
    feel an unbounded plane-image attraction there while an otherwise
    identical particle in `geometry.FlatCathode` felt a finite one.

    Exactness, with the plane recessed: the {q, plane image} and {sphere
    image, its plane image} pairs are each antisymmetric about z=-plane_z0,
    so the potential is *exactly* zero on the recessed plane; and {q,
    sphere image} makes it exactly zero on the sphere r=a. The two
    reflected charges do not separately vanish on that sphere, though,
    since a sphere centered at the origin isn't symmetric about
    z=-plane_z0 -- so with plane_z0 > 0 the sphere condition is satisfied
    only to O(plane_z0/a) (measured: ~2% of the source's own near-surface
    scale at this project's default z0/R = 3/50). That residual is the
    same order as the z0 recession itself, i.e. part of the regularization
    rather than an error on top of it; `plane_z0=0` recovers the
    simultaneously-exact idealization (zero on both the sphere r=a and the
    plane z=0), which is what `bem.image_charge`'s own validation tests
    against this function historically compared to.

    Parameters
    ----------
    position : ndarray, shape (n_groups, n_emit, 3)
    charge : ndarray, shape (n_groups, n_emit)
    active : ndarray of bool, shape (n_groups, n_emit)
    sphere_radius : float
        Image-sphere radius [m] (R - z0 at the tracker level).
    plummer_radius : float
        Same softening length used for the real-real Coulomb sum.
    plane_z0 : float, keyword-only
        Recession [m] of the image plane below the real cathode surface
        (z0 at the tracker level) -- required rather than defaulted, since
        silently defaulting it to 0 is exactly the unregularized-plane
        case described above.
    xp : module, optional

    Returns
    -------
    force : ndarray, shape (n_groups, n_emit, 3)
        Net image-charge force [N] on each particle.
    """
    a = sphere_radius
    d2 = xp.sum(position * position, axis=-1)
    d = xp.sqrt(d2)
    d_safe = xp.where(d > 0.0, d, a)  # particles should never sit at the origin

    scale = (a * a) / (d_safe * d_safe)
    sphere_image_position = position * scale[..., None]
    sphere_image_charge = -charge * (a / d_safe)

    plane_image_position = position.copy()
    plane_image_position[..., 2] = -2.0 * plane_z0 - position[..., 2]

    plane_of_sphere_image_position = sphere_image_position.copy()
    plane_of_sphere_image_position[..., 2] = -2.0 * plane_z0 - sphere_image_position[..., 2]

    f_sphere = _pairwise_force(position, sphere_image_position, charge, sphere_image_charge, active, plummer_radius, xp)
    f_plane = _pairwise_force(position, plane_image_position, charge, -charge, active, plummer_radius, xp)
    f_plane_of_sphere = _pairwise_force(
        position, plane_of_sphere_image_position, charge, -sphere_image_charge, active, plummer_radius, xp,
    )

    return f_sphere + f_plane + f_plane_of_sphere


def _hemispherical_tip_images(source_position, source_charge, sphere_radius, plane_z0):
    """The same 3 image charges `hemispherical_tip_image_force` sums the
    force from (see that function's docstring for the construction), as
    (position, charge) pairs, for one scalar (3,)-shaped source -- used
    by `HemisphericalTipImageSolution.image_potential`, which (like
    `bem.image_charge.ImageChargeBEMSolution.image_field`) is a
    numpy-only single-source reference path, not the batched/xp-generic
    one `hemispherical_tip_image_force` itself needs for the tracker's
    hot loop -- so this is a second copy of the construction, not a
    shared one; a test cross-checks image_potential's gradient against
    hemispherical_tip_image_force's own force directly (rather than each
    against their own math alone) so the two can't silently drift apart."""
    a = sphere_radius
    d = np.linalg.norm(source_position)
    d_safe = d if d > 0.0 else a

    scale = (a * a) / (d_safe * d_safe)
    sphere_image_position = source_position * scale
    sphere_image_charge = -source_charge * (a / d_safe)

    plane_image_position = source_position.copy()
    plane_image_position[2] = -2.0 * plane_z0 - source_position[2]

    plane_of_sphere_image_position = sphere_image_position.copy()
    plane_of_sphere_image_position[2] = -2.0 * plane_z0 - sphere_image_position[2]

    return (
        (sphere_image_position, sphere_image_charge),
        (plane_image_position, -source_charge),
        (plane_of_sphere_image_position, -sphere_image_charge),
    )


class HemisphericalTipImageSolution:
    """Closed-form counterpart to
    `bem.image_charge.ImageChargeBEMSolution`, for `geometry.HemisphericalTip`'s
    exact 3-image system (see `hemispherical_tip_image_force`'s docstring
    for the construction, reused here via `_hemispherical_tip_images`).
    Same `.image_potential` interface as that BEM class (source, query,
    d_lo, d_hi, n_max), so plotting code written against one works
    unchanged against the other (see `geometry.HemisphericalTip.image_solution`)
    -- `d_lo`, `d_hi`, `n_max` are accepted but unused: there is no BEM
    blend distance or mode count for an exact closed-form solution.
    """

    def __init__(self, sphere_radius, plane_z0, real_radius=None):
        self.sphere_radius = float(sphere_radius)
        # Same recessed image plane `hemispherical_tip_image_force` uses
        # (see its docstring) -- required, not defaulted, for the same
        # reason it is there.
        self.plane_z0 = float(plane_z0)
        # Only used for image_potential's masking (see that method's
        # docstring) -- the image construction itself only ever needs
        # sphere_radius (the reduced, R - z0 radius). Falls back to
        # sphere_radius if not given, i.e. masks against the reduced
        # geometry as image_potential originally did.
        self.real_radius = self.sphere_radius if real_radius is None else float(real_radius)

    def image_potential(self, source_position, source_charge, query_position, d_lo=None, d_hi=None, n_max=None):
        """Potential [V] due to the conductor's induced response (the 3
        exact image charges) to one point charge, evaluated at arbitrary
        query points -- see `bem.image_charge.ImageChargeBEMSolution.
        image_potential`'s docstring for the excludes-the-source's-own-
        self-potential convention this matches.

        Parameters
        ----------
        source_position : array, shape (3,)
        source_charge : float
        query_position : array, shape (..., 3)
        d_lo, d_hi, n_max : unused (see class docstring).

        Returns
        -------
        V : array, shape query_position.shape[:-1]. 0 inside the *real*
            conductor (r <= real_radius or z <= 0), not the reduced
            geometry the image construction itself uses -- no real
            particle ever occupies the thin (z0-scale) sliver between
            the two, so masking against the real boundary is what
            actually matches physical space, matching
            `ImageChargeBEMSolution.image_potential`'s convention
            (physically correct there, not just a mask, since a grounded
            conductor's interior sits at its own boundary potential).
        """
        source_position = np.asarray(source_position, dtype=float)
        query_position = np.asarray(query_position, dtype=float)

        V = np.zeros(query_position.shape[:-1])
        for image_position, image_charge in _hemispherical_tip_images(
            source_position, float(source_charge), self.sphere_radius, self.plane_z0
        ):
            diff = query_position - image_position
            r = np.linalg.norm(diff, axis=-1)
            V = V + COULOMB_CONSTANT * image_charge / r

        r_query = np.linalg.norm(query_position, axis=-1)
        inside = (r_query <= self.real_radius) | (query_position[..., 2] <= 0.0)
        return np.where(inside, 0.0, V)


class FlatCathodeImageSolution:
    """Closed-form counterpart to `bem.image_charge.ImageChargeBEMSolution`,
    for `geometry.FlatCathode`'s single-image system (see
    `image_charge_force`'s docstring). Same `.image_potential` interface
    as that BEM class -- see `HemisphericalTipImageSolution`'s docstring
    for why `d_lo`, `d_hi`, `n_max` are accepted but unused.
    """

    def __init__(self, z0):
        self.z0 = float(z0)

    def image_potential(self, source_position, source_charge, query_position, d_lo=None, d_hi=None, n_max=None):
        """Potential [V] due to the single image charge
        `image_charge_force` uses, evaluated at arbitrary query points --
        see `HemisphericalTipImageSolution.image_potential`'s docstring
        for the shared conventions (excludes the source's own self-
        potential; 0 inside the *real* conductor, z <= 0, not the
        recessed plane -z0 the image construction itself uses -- no real
        particle ever occupies that thin sliver).

        Parameters
        ----------
        source_position : array, shape (3,)
        source_charge : float
        query_position : array, shape (..., 3)
        d_lo, d_hi, n_max : unused (see class docstring).

        Returns
        -------
        V : array, shape query_position.shape[:-1]
        """
        source_position = np.asarray(source_position, dtype=float)
        query_position = np.asarray(query_position, dtype=float)

        image_position = source_position.copy()
        image_position[2] = -2.0 * self.z0 - source_position[2]

        diff = query_position - image_position
        r = np.linalg.norm(diff, axis=-1)
        V = COULOMB_CONSTANT * (-float(source_charge)) / r

        inside = query_position[..., 2] <= 0.0
        return np.where(inside, 0.0, V)
