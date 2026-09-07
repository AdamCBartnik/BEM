import numpy as np
import pytest

from specific_particle_tracer import SpecificParticleTracer, HemisphericalTipField, HemisphericalTip
from specific_particle_tracer.forces import hemispherical_tip_image_force
from specific_particle_tracer.constants import COULOMB_CONSTANT, ELEMENTARY_CHARGE, ev_to_kg
from specific_particle_tracer.distributions import flat_distribution_to_hemisphere

ELECTRON_MASS_KG = ev_to_kg(510998.95069)


# ----------------------------------------------------------------------
# Field
# ----------------------------------------------------------------------

def test_hemisphere_field_matches_finite_difference_gradient():
    E0, R = 1.7e7, 50e-9
    field = HemisphericalTipField(E0, R)

    def phi(pos):
        r = np.linalg.norm(pos, axis=-1)
        return -E0 * pos[..., 2] * (1 - R**3 / r**3)

    pts = np.array([
        [0.0, 0.0, R * 1.3],
        [R * 0.7, 0.0, R * 1.1],
        [0.0, R * 0.5, R * 1.2],
    ])
    h = 1e-15
    for p in pts:
        numeric = np.array([
            -(phi(p + [h, 0, 0]) - phi(p - [h, 0, 0])) / (2 * h),
            -(phi(p + [0, h, 0]) - phi(p - [0, h, 0])) / (2 * h),
            -(phi(p + [0, 0, h]) - phi(p - [0, 0, h])) / (2 * h),
        ])
        analytic = field.evaluate(p)
        assert np.allclose(numeric, analytic, rtol=1e-4)


def test_hemisphere_field_tip_enhancement_and_far_field_and_shoulder():
    E0, R = 1.7e7, 50e-9
    field = HemisphericalTipField(E0, R)

    tip = field.evaluate(np.array([0.0, 0.0, R]))
    assert np.isclose(tip[2], 3 * E0, rtol=1e-9)
    assert np.allclose(tip[:2], 0.0)

    far = field.evaluate(np.array([0.0, 0.0, 1e6 * R]))
    assert np.isclose(far[2], E0, rtol=1e-5)

    shoulder = field.evaluate(np.array([R, 0.0, 0.0]))
    assert np.allclose(shoulder, 0.0, atol=1e-6)


def test_hemisphere_field_zero_inside_conductor():
    E0, R = 1.7e7, 50e-9
    field = HemisphericalTipField(E0, R)
    assert np.allclose(field.evaluate(np.array([0.0, 0.0, 0.5 * R])), 0.0)
    assert np.allclose(field.evaluate(np.array([2 * R, 0.0, -1.0])), 0.0)


# ----------------------------------------------------------------------
# Image charge (exact 4-charge solution)
# ----------------------------------------------------------------------

def _four_charge_force(a, q0, r0, plummer_radius=0.0):
    """Reference implementation: direct pairwise Coulomb sum over the real
    charge's 3 images, used to check `hemispherical_tip_image_force`
    against an independent calculation of the same physics."""
    d = np.linalg.norm(r0)
    r1 = (a * a / d**2) * r0
    q1 = -q0 * a / d
    r2 = r0.copy(); r2[2] = -r0[2]
    q2 = -q0
    r3 = r1.copy(); r3[2] = -r1[2]
    q3 = -q1

    force = np.zeros(3)
    for q, r in [(q1, r1), (q2, r2), (q3, r3)]:
        diff = r0 - r
        dist2 = np.sum(diff**2) + plummer_radius**2
        force += COULOMB_CONSTANT * q0 * q * diff / dist2**1.5
    return force


def test_hemispherical_tip_image_force_matches_4charge_reference():
    a = 20e-9
    r0 = np.array([15e-9, 5e-9, 30e-9])
    q0 = -ELEMENTARY_CHARGE

    position = r0.reshape(1, 1, 3)
    charge = np.array([[q0]])
    active = np.array([[True]])

    force = hemispherical_tip_image_force(position, charge, active, a, plummer_radius=1e-15)
    expected = _four_charge_force(a, q0, r0, plummer_radius=1e-15)

    assert np.allclose(force[0, 0], expected, rtol=1e-6)


def test_hemispherical_tip_image_potential_zero_on_both_boundaries():
    """The 3 images should make the potential exactly zero on the sphere
    r=a (upper hemisphere) and on the plane z=0 (r>a) simultaneously --
    checking the potential directly (not just the force) is a stronger
    check that the exact solution (not just the single sphere image) is
    what's implemented."""
    a = 20e-9
    r0 = np.array([15e-9, 5e-9, 30e-9])
    q0 = -ELEMENTARY_CHARGE
    d = np.linalg.norm(r0)
    r1 = (a * a / d**2) * r0
    q1 = -q0 * a / d
    r2 = r0.copy(); r2[2] = -r0[2]
    r3 = r1.copy(); r3[2] = -r1[2]

    def phi(pt):
        total = COULOMB_CONSTANT * q0 / np.linalg.norm(pt - r0)
        for q, r in [(q1, r1), (-q0, r2), (-q1, r3)]:
            total += COULOMB_CONSTANT * q / np.linalg.norm(pt - r)
        return total

    rng = np.random.default_rng(2)
    for _ in range(5):
        v = rng.normal(size=3)
        v[2] = abs(v[2])
        v = v / np.linalg.norm(v) * a
        assert abs(phi(v)) < 1e-6

    for _ in range(5):
        ang = rng.uniform(0, 2 * np.pi)
        rad = a * (1 + rng.uniform(0.1, 5))
        v = np.array([rad * np.cos(ang), rad * np.sin(ang), 0.0])
        assert abs(phi(v)) < 1e-6


def test_hemispherical_tip_image_disabled_via_none_z0(particle_group_factory):
    pg = particle_group_factory(1, t=[0.0], z=[50e-9])
    tracer = SpecificParticleTracer(
        initial_particles=pg, n_emit=1, geometry=HemisphericalTip(-1e8, R=50e-9, z0=None),
        screens=[100e-9],
    )
    assert tracer.geometry.z0 is None  # image force skipped entirely in accel_fn


# ----------------------------------------------------------------------
# Distribution builder
# ----------------------------------------------------------------------

def test_flat_to_hemisphere_maps_onto_sphere_surface(particle_group_factory):
    R = 50e-9
    n = 5
    rng = np.random.default_rng(0)
    x = rng.normal(0, R * 0.1, n)
    y = rng.normal(0, R * 0.1, n)
    pg_flat = particle_group_factory(n, x=x, y=y, pz=np.full(n, 1.0))

    pg_sphere = flat_distribution_to_hemisphere(pg_flat, R)

    r = np.sqrt(np.asarray(pg_sphere.x)**2 + np.asarray(pg_sphere.y)**2 + np.asarray(pg_sphere.z)**2)
    assert np.allclose(r, R, rtol=1e-9)
    assert np.all(np.asarray(pg_sphere.z) > 0)


def test_flat_to_hemisphere_rotates_momentum_to_radial(particle_group_factory):
    R = 50e-9
    n = 4
    rng = np.random.default_rng(1)
    x = rng.normal(0, R * 0.2, n)
    y = rng.normal(0, R * 0.2, n)
    p0 = 1234.5
    pg_flat = particle_group_factory(n, x=x, y=y, pz=np.full(n, p0))

    pg_sphere = flat_distribution_to_hemisphere(pg_flat, R)

    pos = np.stack([pg_sphere.x, pg_sphere.y, pg_sphere.z], axis=-1)
    mom = np.stack([pg_sphere.px, pg_sphere.py, pg_sphere.pz], axis=-1)

    # A purely-normal (pz only) input momentum should become purely radial.
    normal = pos / R
    cos_angle = np.sum(mom * normal, axis=-1) / np.linalg.norm(mom, axis=-1)
    assert np.allclose(cos_angle, 1.0, atol=1e-9)
    assert np.allclose(np.linalg.norm(mom, axis=-1), p0, rtol=1e-9)


def test_flat_to_hemisphere_pole_particle_unrotated(particle_group_factory):
    R = 50e-9
    pg_flat = particle_group_factory(1, pz=[500.0])
    pg_sphere = flat_distribution_to_hemisphere(pg_flat, R)
    assert np.isclose(pg_sphere.z[0], R)
    assert np.isclose(pg_sphere.pz[0], 500.0)
    assert np.isclose(pg_sphere.px[0], 0.0, atol=1e-9)


def test_flat_to_hemisphere_rejects_transverse_spread_beyond_tip(particle_group_factory):
    R = 50e-9
    pg_flat = particle_group_factory(1, x=[R * 2], pz=[1.0])
    with pytest.raises(ValueError):
        flat_distribution_to_hemisphere(pg_flat, R)


def test_flat_to_hemisphere_rejects_nonzero_z(particle_group_factory):
    pg_flat = particle_group_factory(1, z=[1e-9], pz=[1.0])
    with pytest.raises(ValueError):
        flat_distribution_to_hemisphere(pg_flat, 50e-9)


# ----------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------

def test_tip_radius_must_exceed_z0():
    with pytest.raises(ValueError):
        HemisphericalTip(-1e8, R=2e-9, z0=3e-9)


def test_particle_emitted_from_tip_accelerates_radially_and_crosses_screen(particle_group_factory):
    R = 50e-9
    E_gun = -1e8
    a = ELEMENTARY_CHARGE * abs(E_gun) / ELECTRON_MASS_KG  # asymptotic-field acceleration scale

    pg_flat = particle_group_factory(1, t=[0.0], pz=[0.0])
    pg_sphere = flat_distribution_to_hemisphere(pg_flat, R)

    screen_r = R + 500e-9
    tracer = SpecificParticleTracer(
        initial_particles=pg_sphere, n_emit=1, geometry=HemisphericalTip(E_gun, R, z0=3e-9),
        screens=[screen_r], t_max=5e-13,
    )
    (screen_pg,), _ = tracer.run()

    assert len(screen_pg) == 1
    # Emitted along the pole (x=y=0), so it should still be on-axis at the screen.
    assert np.isclose(screen_pg.x[0], 0.0, atol=1e-12)
    assert np.isclose(screen_pg.y[0], 0.0, atol=1e-12)
    assert np.isclose(screen_pg.z[0], screen_r, rtol=1e-6)
    assert screen_pg.pz[0] > 0


def test_backscattered_particle_killed_inside_tip(particle_group_factory):
    """A particle given a momentum kick straight back into the tip (with
    no gun field to stop it) should get killed once it re-enters r <= R,
    not simulated forever."""
    R = 50e-9
    pg_flat = particle_group_factory(1, t=[0.0], pz=[-1000.0])  # into the surface
    pg_sphere = flat_distribution_to_hemisphere(pg_flat, R)

    tracer = SpecificParticleTracer(
        initial_particles=pg_sphere, n_emit=1, geometry=HemisphericalTip(0.0, R, z0=3e-9),
        screens=[R + 1e-9], t_max=1e-9,
    )
    (screen_pg,), _ = tracer.run()
    assert len(screen_pg) == 0  # never reaches the outward screen
