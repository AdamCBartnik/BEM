import numpy as np
import pytest

from specific_particle_tracer.emission import group_indices


def test_group_indices_sorts_by_id(particle_group_factory):
    pg = particle_group_factory(6, ids=[5, 3, 1, 6, 2, 4])
    idx = group_indices(pg, n_emit=2)
    assert idx.shape == (3, 2)
    ids_sorted = np.asarray(pg.id)[idx]
    assert np.array_equal(ids_sorted, [[1, 2], [3, 4], [5, 6]])


def test_group_indices_not_divisible_raises(particle_group_factory):
    pg = particle_group_factory(5)
    with pytest.raises(ValueError):
        group_indices(pg, n_emit=2)


def test_group_indices_duplicate_ids_raises(particle_group_factory):
    pg = particle_group_factory(4, ids=[1, 1, 2, 3])
    with pytest.raises(ValueError):
        group_indices(pg, n_emit=2)
