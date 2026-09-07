"""Analytic external fields.

This module intentionally only knows about a couple of simple,
cheap-to-evaluate cathode shapes (a flat plane, and a flat plane with a
hemispherical tip) for which the field in a uniform asymptotic gun field
has a closed form. A Boundary Element Method solver for fields from
arbitrary cathode nanostructures (and their image charges) is planned as a
future addition with the same call signature, so that it can be dropped in
without changing the tracker -- see geometry.py for the interface every
shape (analytic or BEM) is expected to implement.
"""

import numpy as np


class GunField:
    """A uniform electric field that exists only in the z > 0 half-space.

    Parameters
    ----------
    Ez : float
        z-component of the electric field [V/m], in the ordinary physics
        sign convention (F = q E). For an electron (q = -e) to be
        accelerated away from a cathode sitting at z = 0 (i.e. in the +z
        direction), Ez must be negative -- exactly as for a real DC/RF gun,
        where the field points from the anode back to the cathode.
    xp : module, optional
        Array backend (numpy or cupy) used to build the returned field
        array. Default numpy.

    Notes
    -----
    The field is evaluated as nonzero for z >= 0 (not strictly z > 0), so
    that particles born exactly at the cathode surface z = 0 -- the normal
    case for a photoemission ParticleGroup -- are actually accelerated
    instead of sitting at zero field forever.
    """

    def __init__(self, Ez, xp=np):
        self.Ez = float(Ez)
        self.xp = xp

    def evaluate(self, position):
        """Evaluate the field at the given positions.

        Parameters
        ----------
        position : ndarray, shape (..., 3)

        Returns
        -------
        E : ndarray, shape (..., 3)
            Electric field [V/m] at each position.
        """
        xp = self.xp
        z = position[..., 2]
        E = xp.zeros_like(position)
        E[..., 2] = xp.where(z >= 0.0, self.Ez, 0.0)
        return E


class HemisphericalTipField:
    """Analytic field for an infinite grounded flat cathode (the half-space
    z <= 0 is solid conductor) with a grounded hemispherical tip of radius
    R protruding from it at the origin, sitting in an otherwise uniform
    asymptotic field Ez (the same "gun field" the flat cathode uses, far
    from the tip).

    This is the classic "grounded sphere in a uniform field" textbook
    solution, restricted to the exterior of the upper hemisphere (r >= R,
    z >= 0). By symmetry, that solution is also exactly zero everywhere on
    the z = 0 plane (cosΞΈ = 0 there), so it automatically satisfies the
    flat-plane boundary condition too -- no separate plane image is needed
    for the field itself (only for image-charge forces on particles; see
    `forces.hemispherical_tip_image_force`).

    Parameters
    ----------
    Ez : float
        Asymptotic field [V/m] far from the tip, same sign convention as
        `GunField.Ez`.
    R : float
        Tip radius [m].
    xp : module, optional

    Notes
    -----
    On the tip surface itself (r = R), the field is purely radial (as it
    must be for a conductor) and enhanced by a factor of 3 at the pole
    (x=y=0, z=R) relative to the asymptotic field -- the well-known
    field-enhancement factor for a hemispherical protrusion.
    """

    def __init__(self, Ez, R, xp=np):
        self.Ez = float(Ez)
        self.R = float(R)
        self.xp = xp

    def evaluate(self, position):
        """Evaluate the field at the given positions.

        Parameters
        ----------
        position : ndarray, shape (..., 3)

        Returns
        -------
        E : ndarray, shape (..., 3)
            Electric field [V/m] at each position. Zero inside the
            conductor (r < R, or z < 0 away from the tip).
        """
        xp = self.xp
        R = self.R
        x = position[..., 0]
        y = position[..., 1]
        z = position[..., 2]
        r2 = x * x + y * y + z * z
        r = xp.sqrt(r2)

        # A tolerance below R, not an exact r >= R, because particles are
        # typically placed on this surface via distributions.py's
        # sqrt(R^2 - x^2 - y^2)-based mapping -- recomputing r here from
        # (x, y, z) doesn't exactly invert that sqrt in floating point, so
        # a particle meant to sit exactly at r=R can land a bit below it.
        # Without this tolerance, such a particle sees zero field (rather
        # than the ~3x-enhanced field it should), leaving nothing to
        # oppose its own image-charge attraction back into the tip -- a
        # real bug this project hit, not a hypothetical one.
        valid = (r >= R * (1.0 - 1e-6)) & (z >= 0.0)
        r_safe = xp.where(valid, r, R)  # avoid 0/0 where invalid; discarded below
        r3 = r_safe * r_safe * r_safe
        r5 = r3 * r_safe * r_safe
        R3 = R * R * R

        Ex = 3.0 * self.Ez * R3 * x * z / r5
        Ey = 3.0 * self.Ez * R3 * y * z / r5
        Ezz = self.Ez * (1.0 - R3 / r3 + 3.0 * R3 * z * z / r5)

        E = xp.stack([Ex, Ey, Ezz], axis=-1)
        return xp.where(valid[..., None], E, 0.0)
