import numpy as np

from specific_particle_tracer.forces import coulomb_force
from specific_particle_tracer.constants import COULOMB_CONSTANT


def test_two_particle_repulsion_matches_plummer_formula():
    position = np.array([[[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0]]])  # (1 group, 2 particles, 3)
    charge = np.array([[1.0e-19, 1.0e-19]])
    active = np.array([[True, True]])
    a = 1e-12

    force = coulomb_force(position, charge, active, a)

    r = 1e-9
    expected_mag = COULOMB_CONSTANT * charge[0, 0] * charge[0, 1] * r / (r**2 + a**2) ** 1.5

    assert np.isclose(force[0, 0, 0], -expected_mag)
    assert np.isclose(force[0, 1, 0], expected_mag)
    assert np.allclose(force[0, :, 1:], 0.0)


def test_inactive_particles_exert_no_force():
    position = np.array([[[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0]]])
    charge = np.array([[1.0e-19, 1.0e-19]])
    active = np.array([[True, False]])

    force = coulomb_force(position, charge, active, plummer_radius=1e-12)

    assert np.allclose(force, 0.0)


def test_self_force_is_zero():
    position = np.zeros((1, 1, 3))
    charge = np.array([[1.0e-19]])
    active = np.array([[True]])

    force = coulomb_force(position, charge, active, plummer_radius=1e-12)
    assert np.allclose(force, 0.0)


def test_separate_groups_do_not_interact():
    position = np.array([
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
    ])  # 2 groups, 1 particle each, same position
    charge = np.array([[1.0e-19], [1.0e-19]])
    active = np.array([[True], [True]])

    force = coulomb_force(position, charge, active, plummer_radius=1e-12)
    assert np.allclose(force, 0.0)  # self-force only, still zero
