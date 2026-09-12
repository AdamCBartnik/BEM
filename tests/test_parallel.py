import numpy as np
import pytest

from specific_particle_tracer import SpecificParticleTracer, FlatCathode
from specific_particle_tracer.bem.geometry import CylindricalWellBEMGeometry


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


def test_n_workers_with_bem_geometry_does_not_raise_keyerror(particle_group_factory):
    """Regression test: a BEM geometry (bem/geometry.py) self-registers
    into geometry._WORKER_REGISTRY as an import-time side effect, which
    only happens automatically in a worker process if *something* that
    process actually imports triggers it -- true for a plain script
    (Windows spawn re-executes the launching script's own top-level
    imports) but NOT for a geometry built from an interactive session
    (e.g. a Jupyter notebook), since a spawned worker reconstructs its
    state from `parallel.py` alone, which never imports bem/geometry.py.
    That gap produced a bare `KeyError: 'cylindrical_well_bem'` deep in a
    worker process the first time a BEM geometry was combined with
    n_workers>1 outside a plain script. Fixed via a lazy import inside
    `Geometry.from_worker_args`, verified directly (in a subprocess that
    imports only `specific_particle_tracer.geometry`, never
    `specific_particle_tracer.bem.geometry`) -- this test instead exercises
    it through the real multiprocess path, which is worth keeping too
    since spawned workers are always fresh interpreters regardless of
    what this test session itself has already imported."""
    R, H = 25e-9, 20e-9
    N = 4
    pg = particle_group_factory(N, t=[0.0] * N, x=[0.0] * N, y=[0.0] * N, z=[-H] * N)

    geometry = CylindricalWellBEMGeometry(-1e5, R, H, field_max_length=min(R, H) / 14)
    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=2, geometry=geometry, screens=[1e-6], z_max=1.1e-6,
        t_max=1e-13, backend="cpu", n_workers=2,
    )
    (screen_pg,), _ = tracer.run()
    assert screen_pg is not None  # didn't raise -- that's the actual regression check


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
