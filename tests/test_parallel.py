import numpy as np
import pytest

from specific_particle_tracer import SpecificParticleTracer, FlatCathode


def test_n_workers_gpu_combination_rejected(particle_group_factory):
    pg = particle_group_factory(1, t=[0.0])
    with pytest.raises(ValueError):
        SpecificParticleTracer(
            initial_particles=pg, n_emit=1, geometry=FlatCathode(-1e8), screens=[1e-6],
            backend="gpu", n_workers=2,
        )


def test_n_workers_more_than_groups_is_fine(particle_group_factory):
    """3 groups split across many more workers than groups should just use
    fewer effective workers, not error."""
    pg = particle_group_factory(3, t=[0.0, 0.0, 0.0], ids=[1, 2, 3])
    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(-1e8, z0=None), screens=[1e-6],
        n_workers=8,
    )
    (screen_pg,), _ = tracer.run()
    assert len(screen_pg) == 3


def test_n_workers_matches_serial_result(particle_group_factory):
    """Splitting groups across processes must not change the physics --
    same crossings (up to ordering) as running everything in one process."""
    rng = np.random.default_rng(0)
    n_groups, n_emit = 40, 3
    n = n_groups * n_emit
    t = np.sort(rng.uniform(0, 1e-13, n))
    x = rng.normal(0, 1e-6, n)
    y = rng.normal(0, 1e-6, n)
    pg = particle_group_factory(n, t=t, x=x, y=y)

    kwargs = dict(
        initial_particles=pg, n_emit=n_emit, geometry=FlatCathode(-1e8), screens=[1e-6],
        plummer_radius=1e-12,
    )

    serial = SpecificParticleTracer(n_workers=1, **kwargs).run()[0][0]
    parallel = SpecificParticleTracer(n_workers=4, **kwargs).run()[0][0]

    assert len(serial) == len(parallel)

    order_s = np.argsort(serial.id)
    order_p = np.argsort(parallel.id)

    assert np.array_equal(np.asarray(serial.id)[order_s], np.asarray(parallel.id)[order_p])
    assert np.allclose(np.asarray(serial.t)[order_s], np.asarray(parallel.t)[order_p], rtol=1e-9)
    assert np.allclose(np.asarray(serial.x)[order_s], np.asarray(parallel.x)[order_p], rtol=1e-9)
    assert np.allclose(np.asarray(serial.pz)[order_s], np.asarray(parallel.pz)[order_p], rtol=1e-9)
