"""Initial-distribution helpers -- kept deliberately separate from
tracker.py, which only ever runs a ParticleGroup it's handed and has no
opinion about how that distribution was constructed.

Right now this has one helper: mapping a flat (z=0) distribution onto the
surface of a hemispherical tip, for use with
`SpecificParticleTracer(geometry=geometry.HemisphericalTip(E_gun, R), ...)`.
"""

import numpy as np

try:
    from beamphysics import ParticleGroup
except ImportError:  # pragma: no cover
    from pmd_beamphysics import ParticleGroup


def flat_distribution_to_hemisphere(particle_group, R):
    """Map a flat-cathode distribution (all particles at z=0) onto the
    surface of a hemispherical tip of radius R centered at the origin,
    for use as the initial distribution of a
    `SpecificParticleTracer(geometry=geometry.HemisphericalTip(E_gun, R), ...)`
    run.

    Each particle's (x, y) is left alone and used to place it directly
    "above" that point on the tip surface, at
    z = sqrt(R^2 - x^2 - y^2) -- i.e. the flat distribution is projected
    straight up onto the dome, the way a pattern would look projected onto
    a hemisphere from directly overhead. Its momentum vector is then
    rotated by whatever rotation takes +z (the flat cathode's normal) to
    that surface point's own outward normal (which, for a sphere centered
    at the origin, is just the radial direction) -- so a purely-normal
    emission direction on the flat cathode becomes a purely-radial
    (normal to the tip) emission direction, and any transverse/thermal
    momentum spread is carried along consistently rather than discarded.

    Parameters
    ----------
    particle_group : ParticleGroup
        Must have z == 0 for every particle, and x^2 + y^2 < R^2 (the
        transverse spread must fit within the tip's "shadow" -- this is
        meant for a distribution concentrated near the pole/tip, not one
        spread over the whole cathode).
    R : float
        Tip radius [m], matching the `R` the tracker's geometry will use.

    Returns
    -------
    ParticleGroup
        A new ParticleGroup with the same t, weight, status, id, species,
        but x, y, z, px, py, pz replaced as described above.
    """
    x = np.asarray(particle_group.x)
    y = np.asarray(particle_group.y)
    z = np.asarray(particle_group.z)

    if np.any(z != 0.0):
        raise ValueError("flat_distribution_to_hemisphere expects a distribution with z == 0 everywhere")

    rho2 = x * x + y * y
    if np.any(rho2 >= R * R):
        raise ValueError(
            "some particles' (x, y) offset from the axis is >= R; this mapping "
            "only makes sense for a distribution concentrated within the tip's "
            "footprint (x^2 + y^2 < R^2)"
        )

    z_surface = np.sqrt(R * R - rho2)
    normal = np.stack([x, y, z_surface], axis=-1) / R  # outward unit normal at each mapped point

    momentum = np.stack(
        [np.asarray(particle_group.px), np.asarray(particle_group.py), np.asarray(particle_group.pz)],
        axis=-1,
    )
    rotated_momentum = _rotate_zhat_to(normal, momentum)

    data = dict(
        x=x, y=y, z=z_surface,
        px=rotated_momentum[:, 0], py=rotated_momentum[:, 1], pz=rotated_momentum[:, 2],
        t=np.asarray(particle_group.t),
        weight=np.asarray(particle_group.weight),
        status=np.asarray(particle_group.status),
        id=np.asarray(particle_group.id),
        species=particle_group.species,
    )
    return ParticleGroup(data=data)


def _rotate_zhat_to(target, vectors):
    """Rotate each row of `vectors` (n, 3) by the rotation that takes +z to
    the corresponding (unit-length) row of `target` (n, 3), via Rodrigues'
    rotation formula applied directly to each vector (no explicit
    per-particle rotation matrix needed).
    """
    axis = np.cross(np.array([0.0, 0.0, 1.0]), target)  # (n, 3)
    sin_theta = np.linalg.norm(axis, axis=-1)  # (n,)
    cos_theta = target[:, 2]  # zhat . target

    degenerate = sin_theta < 1e-12  # target parallel (or antiparallel) to +z
    safe_sin = np.where(degenerate, 1.0, sin_theta)
    axis_unit = axis / safe_sin[:, None]

    dot = np.sum(axis_unit * vectors, axis=-1, keepdims=True)
    cross = np.cross(axis_unit, vectors)
    rotated = (
        vectors * cos_theta[:, None]
        + cross * sin_theta[:, None]
        + axis_unit * dot * (1.0 - cos_theta[:, None])
    )

    # Degenerate case: target == +z (identity) or -z (flip) -- cos_theta is
    # +1 or -1 there since it's zhat . target with target a unit vector.
    identity_or_flip = np.where(cos_theta[:, None] > 0, vectors, -vectors)
    return np.where(degenerate[:, None], identity_or_flip, rotated)
