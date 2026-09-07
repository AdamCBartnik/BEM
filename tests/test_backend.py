import numpy as np
import pytest

from specific_particle_tracer import SpecificParticleTracer, FlatCathode
from specific_particle_tracer.backend import resolve_backend


def test_cpu_backend_is_numpy_float64():
    xp, dtype = resolve_backend("cpu")
    assert xp is np
    assert dtype == np.float64


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        resolve_backend("tpu")


def test_gpu_backend_requires_cupy_or_is_skipped():
    try:
        xp, dtype = resolve_backend("gpu")
    except ImportError:
        pytest.skip("cupy not available in this environment")
    else:
        assert dtype == np.float64


def test_cpu_run_matches_default_backend(particle_group_factory):
    """backend='cpu' (the default) should just work end-to-end."""
    pg = particle_group_factory(1, t=[0.0])
    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(-1e8), screens=[1e-6], backend="cpu",
    )
    (screen_pg,), _ = tracer.run()
    assert len(screen_pg) == 1
