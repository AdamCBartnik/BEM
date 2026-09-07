import numpy as np

from specific_particle_tracer import SpecificParticleTracer, FlatCathode
from specific_particle_tracer.constants import ELEMENTARY_CHARGE, ev_to_kg

ELECTRON_MASS_KG = ev_to_kg(510998.95069)


def test_trajectory_matches_analytic_kinematics(particle_group_factory):
    """A ballistic particle's recorded trajectory snapshots should match
    z(t) = 0.5*a*t^2, vz(t) = a*t exactly (RK45-tolerance exact, not
    interpolated), at every requested output time."""
    E_gun = -1e8
    a = ELEMENTARY_CHARGE * abs(E_gun) / ELECTRON_MASS_KG

    t_out = [1e-13, 2e-13, 3e-13]
    pg = particle_group_factory(1, t=[0.0])

    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(E_gun, z0=None),
        screens=[1e-6], t_out=t_out, t_max=4e-13,
    )
    _, trajectories = tracer.run()

    assert len(trajectories) == 3
    for t, traj in zip(t_out, trajectories):
        assert len(traj) == 1
        assert np.isclose(traj.t[0], t, rtol=0, atol=0)
        assert np.isclose(traj.z[0], 0.5 * a * t**2, rtol=1e-6)
        vz_sim = traj.pz[0] * ELEMENTARY_CHARGE / 299792458.0 / ELECTRON_MASS_KG
        assert np.isclose(vz_sim, a * t, rtol=1e-6)


def test_trajectory_excludes_particles_not_yet_born(particle_group_factory):
    E_gun = -1e8
    t_birth = 5e-13
    pg = particle_group_factory(1, t=[t_birth])

    t_out = [1e-13, t_birth + 1e-13]
    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(E_gun, z0=None),
        screens=[1e-6], t_out=t_out, t_max=t_birth + 2e-13,
    )
    _, (before, after) = tracer.run()

    assert len(before) == 0
    assert len(after) == 1
    assert np.isclose(after.t[0], t_birth + 1e-13)


def test_trajectory_frozen_after_kill(particle_group_factory):
    """A particle killed on re-entering the cathode should show up at the
    same (frozen) position for every output time after the kill."""
    pg = particle_group_factory(1, t=[0.0], z=[2e-9], pz=[-1000.0])  # heading into the surface

    t_out = [1e-14, 5e-14, 1e-13]
    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(0.0, z0=3e-9),
        screens=[], t_out=t_out, t_max=2e-13,
    )
    _, trajectories = tracer.run()

    # Should be killed well before the first requested output (falling
    # only 2nm at >>1e-9 m/s takes far less than 1e-14s).
    zs = [traj.z[0] for traj in trajectories]
    assert all(z <= 0.0 for z in zs)
    assert np.allclose(zs, zs[0])


def test_trajectory_matches_between_serial_and_parallel(particle_group_factory):
    rng = np.random.default_rng(0)
    n_groups, n_emit = 20, 3
    n = n_groups * n_emit
    t = np.sort(rng.uniform(0, 1e-13, n))
    x = rng.normal(0, 1e-6, n)
    y = rng.normal(0, 1e-6, n)
    pg = particle_group_factory(n, t=t, x=x, y=y)

    kwargs = dict(
        initial_particles=pg, n_emit=n_emit, geometry=FlatCathode(-1e8), screens=[],
        t_out=[1e-13, 2e-13], plummer_radius=1e-12,
    )

    _, serial_traj = SpecificParticleTracer(n_workers=1, **kwargs).run()
    _, parallel_traj = SpecificParticleTracer(n_workers=4, **kwargs).run()

    for s, p in zip(serial_traj, parallel_traj):
        assert len(s) == len(p)
        order_s = np.argsort(s.id)
        order_p = np.argsort(p.id)
        assert np.array_equal(np.asarray(s.id)[order_s], np.asarray(p.id)[order_p])
        assert np.allclose(np.asarray(s.x)[order_s], np.asarray(p.x)[order_p], rtol=1e-9)
        assert np.allclose(np.asarray(s.pz)[order_s], np.asarray(p.pz)[order_p], rtol=1e-9)


def test_default_t_max_extends_to_cover_requested_output_times(particle_group_factory):
    pg = particle_group_factory(1, t=[0.0])
    t_out = [5e-12]  # far beyond the default screen-based estimate
    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=FlatCathode(-1e8, z0=None), screens=[], t_out=t_out,
    )
    assert tracer.t_max >= 5e-12
    _, (traj,) = tracer.run()
    assert len(traj) == 1
    assert np.isclose(traj.t[0], 5e-12)
