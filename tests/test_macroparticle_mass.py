import warnings

import numpy as np
import pytest

from specific_particle_tracer import SpecificParticleTracer, FlatCathode
from specific_particle_tracer.constants import ELEMENTARY_CHARGE, ev_to_kg

ELECTRON_MASS_KG = ev_to_kg(510998.95069)


def test_nonstandard_weight_is_forced_to_elementary_charge_with_warning(particle_group_factory):
    """This tracker models individual real particles, not statistical
    macroparticles (Coulomb/image forces scale with each particle's own
    weight, which is only physically meaningful for aggregate real-real
    or real-image interactions if that weight really is one elementary
    charge -- see geometry.make_accel_fn). Rather than silently doing the
    wrong physics for some other total-charge convention, any weight that
    isn't already the species' elementary charge gets overridden, with a
    warning."""
    pg = particle_group_factory(3, weight=[5 * ELEMENTARY_CHARGE] * 3)

    with pytest.warns(UserWarning, match="elementary charge"):
        tracer = SpecificParticleTracer(
            initial_particles=pg, n_emit=1, geometry=FlatCathode(-1e8), screens=[1e-6],
        )

    assert np.allclose(np.asarray(tracer.weight), ELEMENTARY_CHARGE)


def test_standard_weight_triggers_no_warning(particle_group_factory):
    pg = particle_group_factory(3)  # fixture default weight is already one elementary charge
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here should fail the test
        SpecificParticleTracer(
            initial_particles=pg, n_emit=1, geometry=FlatCathode(-1e8), screens=[1e-6],
        )


def test_field_acceleration_independent_of_forced_weight(particle_group_factory):
    """Regression test for the underlying bug the weight-forcing above is a
    belt-and-suspenders guard against: a particle's acceleration under the
    external field must depend only on its species (e/m for an electron),
    not on its (macro-)charge weight. An earlier version divided the
    weighted force by the bare single-particle mass, inflating the field
    acceleration by the weight factor -- invisible whenever weight equals
    exactly one elementary charge (every other test's fixture default) and
    very wrong otherwise."""
    E_gun = -1e8
    a = ELEMENTARY_CHARGE * abs(E_gun) / ELECTRON_MASS_KG
    z_screen = 1e-6
    t_analytic = np.sqrt(2 * z_screen / a)

    pg = particle_group_factory(1, t=[0.0])
    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(E_gun, z0=None), screens=[z_screen],
        t_max=t_analytic * 1.5,
    )
    (screen_pg,), _ = tracer.run()

    assert np.isclose(screen_pg.t[0], t_analytic, rtol=1e-6)
