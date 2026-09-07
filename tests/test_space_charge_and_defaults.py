import numpy as np
import pytest

from specific_particle_tracer import SpecificParticleTracer, FlatCathode


def test_coulomb_repulsion_pushes_particles_apart_transversely(particle_group_factory):
    """Two co-emitted electrons offset only in x should repel and end up
    farther apart in x at the screen than two independent (n_emit=1)
    particles with the same initial conditions."""
    E_gun = -1e8
    z_screen = 1e-6
    dx0 = 1e-9

    pg_pair = particle_group_factory(2, t=[0.0, 0.0], x=[-dx0 / 2, dx0 / 2])
    tracer_pair = SpecificParticleTracer(
        initial_particles=pg_pair, n_emit=2, geometry=FlatCathode(E_gun), screens=[z_screen],
        plummer_radius=1e-12,
    )
    (screen_pair,), _ = tracer_pair.run()

    pg_solo = particle_group_factory(2, t=[0.0, 0.0], x=[-dx0 / 2, dx0 / 2], ids=[1, 2])
    tracer_solo = SpecificParticleTracer(
        initial_particles=pg_solo, n_emit=1, geometry=FlatCathode(E_gun), screens=[z_screen],
        plummer_radius=1e-12,
    )
    (screen_solo,), _ = tracer_solo.run()

    spread_pair = screen_pair.x.max() - screen_pair.x.min()
    spread_solo = screen_solo.x.max() - screen_solo.x.min()

    assert spread_pair > spread_solo


def test_default_dt_and_t_max_produce_a_result(particle_group_factory):
    pg = particle_group_factory(1, t=[0.0])
    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(-1e8), screens=[1e-6],
    )
    (screen_pg,), _ = tracer.run()
    assert len(screen_pg) == 1


def test_invalid_n_emit_raises(particle_group_factory):
    pg = particle_group_factory(5)
    with pytest.raises(ValueError):
        SpecificParticleTracer(
            initial_particles=pg, n_emit=2, geometry=FlatCathode(-1e8), screens=[1e-6],
        )
