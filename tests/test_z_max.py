import numpy as np

from specific_particle_tracer import SpecificParticleTracer, FlatCathode
from specific_particle_tracer.constants import ELEMENTARY_CHARGE, ev_to_kg

ELECTRON_MASS_KG = ev_to_kg(510998.95069)


def test_z_max_does_not_affect_a_particle_that_stays_below_it(particle_group_factory):
    """A screen well inside z_max should record the same crossing whether
    or not z_max is set."""
    E_gun = -1e8
    a = ELEMENTARY_CHARGE * abs(E_gun) / ELECTRON_MASS_KG
    z_screen = 1e-6
    t_analytic = np.sqrt(2 * z_screen / a)

    pg = particle_group_factory(1, t=[0.0])
    kwargs = dict(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(E_gun, z0=None), screens=[z_screen],
        t_max=t_analytic * 1.5,
    )
    (no_cutoff,), _ = SpecificParticleTracer(**kwargs).run()
    (with_cutoff,), _ = SpecificParticleTracer(z_max=z_screen * 10, **kwargs).run()

    assert np.isclose(no_cutoff.t[0], with_cutoff.t[0], rtol=1e-9)
    assert np.isclose(no_cutoff.pz[0], with_cutoff.pz[0], rtol=1e-9)


def test_z_max_still_records_a_screen_exactly_at_the_cutoff(particle_group_factory):
    """A screen placed at (or just inside) z_max should still be crossed
    and recorded before the kill takes effect."""
    E_gun = -1e8
    a = ELEMENTARY_CHARGE * abs(E_gun) / ELECTRON_MASS_KG
    z_screen = 1e-6
    t_analytic = np.sqrt(2 * z_screen / a)

    pg = particle_group_factory(1, t=[0.0])
    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(E_gun, z0=None), screens=[z_screen],
        z_max=z_screen, t_max=t_analytic * 1.5,
    )
    (screen_pg,), _ = tracer.run()
    assert len(screen_pg) == 1
    assert np.isclose(screen_pg.t[0], t_analytic, rtol=1e-6)
