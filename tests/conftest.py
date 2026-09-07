import numpy as np
import pytest

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup


def make_particle_group(n, t=None, x=None, y=None, z=None, px=None, py=None, pz=None, weight=None, ids=None):
    zero = np.zeros(n)
    data = dict(
        x=zero.copy() if x is None else np.asarray(x, dtype=float),
        y=zero.copy() if y is None else np.asarray(y, dtype=float),
        z=zero.copy() if z is None else np.asarray(z, dtype=float),
        px=zero.copy() if px is None else np.asarray(px, dtype=float),
        py=zero.copy() if py is None else np.asarray(py, dtype=float),
        pz=zero.copy() if pz is None else np.asarray(pz, dtype=float),
        t=zero.copy() if t is None else np.asarray(t, dtype=float),
        weight=np.full(n, 1.602176634e-19) if weight is None else np.asarray(weight, dtype=float),
        status=np.ones(n),
        species="electron",
        id=np.arange(1, n + 1) if ids is None else np.asarray(ids, dtype=int),
    )
    return ParticleGroup(data=data)


@pytest.fixture
def particle_group_factory():
    return make_particle_group
