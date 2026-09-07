"""Splitting an initial ParticleGroup into emission groups of n_emit particles."""

import numpy as np


def group_indices(particle_group, n_emit):
    """Return an (n_groups, n_emit) integer array of indices into
    ``particle_group``, sorted by ascending particle ID and chunked into
    groups of size ``n_emit``.

    Only particles within the same group interact with each other; groups
    are otherwise independent (and can be advanced together as a batch).

    Raises
    ------
    ValueError
        If the number of particles is not evenly divisible by n_emit, or if
        particle_group does not have unique IDs.
    """
    n_particles = len(particle_group)

    if n_emit < 1:
        raise ValueError(f"n_emit must be a positive integer, got {n_emit}")

    if n_particles % n_emit != 0:
        raise ValueError(
            f"Number of particles ({n_particles}) is not evenly divisible "
            f"by n_emit ({n_emit})."
        )

    ids = np.asarray(particle_group.id)
    if len(np.unique(ids)) != n_particles:
        raise ValueError("particle_group.id must contain unique values.")

    order = np.argsort(ids, kind="stable")

    n_groups = n_particles // n_emit
    return order.reshape(n_groups, n_emit)
