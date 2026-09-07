import numpy as np

from specific_particle_tracer import SpecificParticleTracer, FlatCathode
from specific_particle_tracer.constants import ELEMENTARY_CHARGE, ev_to_kg

ELECTRON_MASS_KG = ev_to_kg(510998.95069)


def test_field_acceleration_independent_of_macro_charge_weight(particle_group_factory):
    """A particle's acceleration under the external field must depend only
    on its species (e/m for an electron), not on its statistical weight --
    a macroparticle carrying 1000 elementary charges of weight should
    accelerate under a field exactly like one carrying a single elementary
    charge. (Regression test: an earlier version divided the weighted
    force by the bare single-particle mass, inflating the field
    acceleration by the weight factor -- invisible whenever weight happens
    to equal exactly one elementary charge, as in every other test's
    fixture default, and wrong otherwise.)"""
    E_gun = -1e8
    a = ELEMENTARY_CHARGE * abs(E_gun) / ELECTRON_MASS_KG
    z_screen = 1e-6
    t_analytic = np.sqrt(2 * z_screen / a)

    pg_unit = particle_group_factory(1, t=[0.0], weight=[ELEMENTARY_CHARGE])
    pg_macro = particle_group_factory(1, t=[0.0], weight=[1000 * ELEMENTARY_CHARGE])

    kwargs = dict(
        n_emit=1, geometry=FlatCathode(E_gun, z0=None), screens=[z_screen],
        t_max=t_analytic * 1.5,
    )
    (unit_pg,), _ = SpecificParticleTracer(initial_particles=pg_unit, **kwargs).run()
    (macro_pg,), _ = SpecificParticleTracer(initial_particles=pg_macro, **kwargs).run()

    assert np.isclose(unit_pg.t[0], macro_pg.t[0], rtol=1e-9)
    assert np.isclose(unit_pg.pz[0], macro_pg.pz[0], rtol=1e-9)
    assert np.isclose(unit_pg.t[0], t_analytic, rtol=1e-6)
