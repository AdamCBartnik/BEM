import numpy as np

from specific_particle_tracer.forces import image_charge_force
from specific_particle_tracer.constants import COULOMB_CONSTANT
from specific_particle_tracer import SpecificParticleTracer, FlatCathode


def test_self_image_force_matches_analytic_formula():
    """A lone electron at height z feels an attractive force toward its own
    image, as if a charge of the opposite sign sat a distance 2*(z+z0) away."""
    z0 = 3e-9
    z = 5e-9
    position = np.array([[[0.0, 0.0, z]]])
    charge = np.array([[-1.602176634e-19]])
    active = np.array([[True]])

    force = image_charge_force(position, charge, active, z0, plummer_radius=1e-15)

    separation = 2 * (z + z0)
    expected_mag = COULOMB_CONSTANT * charge[0, 0] ** 2 / separation**2  # attractive (opposite-sign image)

    assert np.isclose(force[0, 0, 2], -expected_mag, rtol=1e-6)
    assert np.allclose(force[0, 0, :2], 0.0)


def test_z0_zero_is_the_bare_divergent_limit():
    """With z0=0, a particle sitting just off the surface only avoids a
    literal division by zero because of the (small) Plummer softening.

    (Exactly z=0 with z0=0 is a degenerate case where the real particle and
    its image literally coincide -- the separation *vector* used to get the
    force's direction is exactly zero there, not just small, so the force
    formula correctly -- if a little surprisingly -- gives exactly zero
    rather than a divergent one. That configuration needs z > 0 to have a
    well-defined direction, which is what's tested here.)"""
    z0 = 0.0
    z = 1e-13
    position = np.array([[[0.0, 0.0, z]]])
    charge = np.array([[-1.602176634e-19]])
    active = np.array([[True]])
    a = 1e-12

    force = image_charge_force(position, charge, active, z0, plummer_radius=a)

    separation = 2 * (z + z0)
    dist2 = separation**2 + a**2
    expected_force_z = -COULOMB_CONSTANT * charge[0, 0] ** 2 * separation / dist2**1.5
    assert np.isclose(force[0, 0, 2], expected_force_z, rtol=1e-6)


def test_larger_z0_weakens_the_attraction():
    position = np.array([[[0.0, 0.0, 1e-9]]])
    charge = np.array([[-1.602176634e-19]])
    active = np.array([[True]])

    f_small_z0 = image_charge_force(position, charge, active, z0=1e-9, plummer_radius=1e-15)
    f_large_z0 = image_charge_force(position, charge, active, z0=10e-9, plummer_radius=1e-15)

    assert abs(f_large_z0[0, 0, 2]) < abs(f_small_z0[0, 0, 2])


def test_image_charge_pulls_a_resting_particle_back_toward_the_cathode(particle_group_factory):
    """With no gun field, a particle sitting near the surface should be
    pulled toward -z by its own image, and cross a nearby screen moving
    inbound before it's killed on reaching the real surface (z<=0, the
    default kill boundary)."""
    pg = particle_group_factory(1, t=[0.0], z=[2e-9])

    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(0.0, z0=3e-9), screens=[1e-9],
        t_max=1e-13,
    )
    (screen_pg,), _ = tracer.run()

    assert len(screen_pg) >= 1
    first = np.argmin(screen_pg.t)
    assert screen_pg.pz[first] < 0  # moving in -z when it first crosses
