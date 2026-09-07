import numpy as np

from specific_particle_tracer import SpecificParticleTracer, FlatCathode
from specific_particle_tracer.constants import ELEMENTARY_CHARGE, ev_to_kg

ELECTRON_MASS_EV = 510998.95069
ELECTRON_MASS_KG = ev_to_kg(ELECTRON_MASS_EV)


def test_single_particle_ballistic_kinematics(particle_group_factory):
    """One electron born at rest at z=0 in a constant field should follow
    z(t) = 0.5*a*t^2 exactly (no space charge with n_emit=1)."""
    E_gun = -1e8  # V/m
    a = ELEMENTARY_CHARGE * abs(E_gun) / ELECTRON_MASS_KG

    z_screen = 1e-6  # 1 micron
    t_analytic = np.sqrt(2 * z_screen / a)
    vz_analytic = a * t_analytic

    pg = particle_group_factory(1, t=[0.0])

    tracer = SpecificParticleTracer(
        initial_particles=pg,
        n_emit=1,
        geometry=FlatCathode(E_gun, z0=None),  # isolate gun-field kinematics from the image-charge feature
        screens=[z_screen],
        t_max=t_analytic * 1.5,
    )
    (screen_pg,), _ = tracer.run()

    assert len(screen_pg) == 1
    assert np.isclose(screen_pg.t[0], t_analytic, rtol=1e-6)
    assert np.isclose(screen_pg.z[0], z_screen, rtol=1e-9)

    vz_sim = screen_pg.pz[0] * ELEMENTARY_CHARGE / 299792458.0 / ELECTRON_MASS_KG
    assert np.isclose(vz_sim, vz_analytic, rtol=1e-6)


def test_particle_emitted_late_starts_later(particle_group_factory):
    E_gun = -1e8
    a = ELEMENTARY_CHARGE * abs(E_gun) / ELECTRON_MASS_KG
    z_screen = 1e-6
    t0_analytic = np.sqrt(2 * z_screen / a)

    t_birth = 5e-13
    pg = particle_group_factory(1, t=[t_birth])

    tracer = SpecificParticleTracer(
        initial_particles=pg,
        n_emit=1,
        geometry=FlatCathode(E_gun, z0=None),
        screens=[z_screen],
        t_max=t_birth + t0_analytic * 1.5,
    )
    (screen_pg,), _ = tracer.run()

    assert np.isclose(screen_pg.t[0], t_birth + t0_analytic, rtol=1e-6)


def test_multiple_screens_recorded_in_flight(particle_group_factory):
    E_gun = -1e8
    a = ELEMENTARY_CHARGE * abs(E_gun) / ELECTRON_MASS_KG
    screens = [0.5e-6, 1.0e-6, 2.0e-6]

    pg = particle_group_factory(1, t=[0.0])
    t_max = np.sqrt(2 * max(screens) / a) * 1.5

    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(E_gun, z0=None), screens=screens,
        t_max=t_max,
    )
    results, _ = tracer.run()

    for z_screen, screen_pg in zip(screens, results):
        assert len(screen_pg) == 1
        t_analytic = np.sqrt(2 * z_screen / a)
        assert np.isclose(screen_pg.t[0], t_analytic, rtol=1e-6)


def test_groups_do_not_interact(particle_group_factory):
    """Two groups of 1 particle each, sitting on top of each other, should
    not repel each other (only particles within the same group interact)."""
    E_gun = -1e8
    a = ELEMENTARY_CHARGE * abs(E_gun) / ELECTRON_MASS_KG
    z_screen = 1e-6
    t_analytic = np.sqrt(2 * z_screen / a)

    pg = particle_group_factory(2, t=[0.0, 0.0], x=[0.0, 0.0])

    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(E_gun, z0=None), screens=[z_screen],
        t_max=t_analytic * 1.5,
    )
    (screen_pg,), _ = tracer.run()

    assert len(screen_pg) == 2
    assert np.allclose(screen_pg.t, t_analytic, rtol=1e-6)
    assert np.allclose(screen_pg.x, 0.0, atol=1e-9)
