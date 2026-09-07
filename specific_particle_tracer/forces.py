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


def hemispherical_tip_image_force(position, charge, active, sphere_radius, plummer_radius, xp=np):
    """Attraction of each particle toward the image charges induced in an
    infinite grounded plane (z=0) with a grounded hemispherical tip of
    radius `sphere_radius` at the origin, restricted (like Coulomb) to the
    particles within one group.

    This is the exact image system for that compound shape: 3 images per
    real charge, not just 1. For a real charge q at position r0 (distance
    d = |r0| > a from the origin, with a = `sphere_radius`):

      1. Sphere image: charge -q*a/d at a^2/d^2 * r0 (the classic
         single-image construction for a grounded sphere -- makes the
         sphere r=a an exact equipotential by itself).
      2. Plane image: charge -q at r0 reflected through z=0.
      3. Plane image of the sphere image: charge +q*a/d at the sphere
         image's position, also reflected through z=0.

    All three sit strictly inside the solid conductor (the sphere images
    at radius a^2/d < a, i.e. inside the tip; the plane images at z<0,
    inside the bulk cathode), which is what makes them valid images in the
    first place. Together, verified numerically, they make the potential
    exactly zero on both the sphere r=a (for z>=0) and the plane z=0 (for
    r>=a) simultaneously -- unlike using the sphere image alone, this
    reproduces the correct flat-mirror-like behavior for a particle
    anywhere on the flat part of the cathode, not just right at the tip.

    Call with `sphere_radius` = R - z0 (R the physical tip radius, z0 the
    same effective offset the flat cathode uses), so a particle sitting
    exactly on the real tip surface (r=R) sees a finite separation from
    its nearest image rather than a divergent one right at contact.

    Parameters
    ----------
    position : ndarray, shape (n_groups, n_emit, 3)
    charge : ndarray, shape (n_groups, n_emit)
    active : ndarray of bool, shape (n_groups, n_emit)
    sphere_radius : float
        Image-sphere radius [m] (R - z0 at the tracker level).
    plummer_radius : float
        Same softening length used for the real-real Coulomb sum.
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
    plane_image_position[..., 2] = -position[..., 2]

    plane_of_sphere_image_position = sphere_image_position.copy()
    plane_of_sphere_image_position[..., 2] = -sphere_image_position[..., 2]

    f_sphere = _pairwise_force(position, sphere_image_position, charge, sphere_image_charge, active, plummer_radius, xp)
    f_plane = _pairwise_force(position, plane_image_position, charge, -charge, active, plummer_radius, xp)
    f_plane_of_sphere = _pairwise_force(
        position, plane_of_sphere_image_position, charge, -sphere_image_charge, active, plummer_radius, xp,
    )

    return f_sphere + f_plane + f_plane_of_sphere
