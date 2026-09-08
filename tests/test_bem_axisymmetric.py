import numpy as np
import pytest
from scipy import integrate

from specific_particle_tracer.bem.mesh import sphere_cap_profile
from specific_particle_tracer.bem.axisymmetric import (
    ring_potential,
    ring_field,
    AxisymmetricBEMSolution,
    _closest_point_on_profile,
    _segment_endpoints,
)
from specific_particle_tracer.fields import HemisphericalTipField
from specific_particle_tracer.bem.fields import HemisphericalTipBEMField


def test_ring_potential_matches_brute_force_azimuthal_integration():
    """ring_potential(rho, z, a) is meant to be the potential of a
    unit-*total*-charge ring -- check against direct numerical integration
    of the point kernel around the ring, with the charge normalized so the
    total really is 1 (a distinction this module's own development caught
    getting wrong: the "naive" azimuthal integral of the point kernel is
    the potential of a ring carrying total charge 2*pi, not 1)."""
    for rho, z, a in [(0.3, 0.5, 1.0), (2.0, 0.1, 1.0), (0.0, 1.0, 1.0), (1.5, 0.0, 1.0)]:
        closed = ring_potential(rho, z, a)

        def integrand(phi):
            return (1.0 / (2 * np.pi)) / (4 * np.pi * np.sqrt(rho**2 + a**2 - 2 * a * rho * np.cos(phi) + z**2))

        brute, _ = integrate.quad(integrand, 0, 2 * np.pi)
        assert abs(closed - brute) / abs(brute) < 1e-8


def test_ring_field_matches_finite_difference_of_ring_potential():
    """E = -grad(ring_potential), checked by central finite difference --
    catches sign and calculus errors in the closed-form elliptic-integral
    derivative independently of the potential/brute-force check above.

    Note rho is a radial coordinate (never negative): the correct way to
    difference through rho=0 is via cartesian x, using rho=|x|, not by
    evaluating ring_potential at a literal negative rho.
    """
    h = 1e-6

    def field_fd(rho, z, a):
        e_rho = -(ring_potential(rho + h, z, a) - ring_potential(abs(rho - h), z, a)) / (2 * h)
        e_z = -(ring_potential(rho, z + h, a) - ring_potential(rho, z - h, a)) / (2 * h)
        return e_rho, e_z

    for rho, z, a in [(0.3, 0.5, 1.0), (2.0, 0.1, 1.0), (0.0, 1.0, 1.0), (1.5, 0.0, 1.0), (0.01, 2.0, 1.0)]:
        fd = field_fd(rho, z, a)
        closed = ring_field(rho, z, a)
        assert abs(fd[0] - closed[0]) < 1e-8
        assert abs(fd[1] - closed[1]) < 1e-8


def test_uniform_sphere_field_matches_shell_theorem():
    """A uniformly-charged spherical shell's field just outside must equal
    its own surface charge density (the shell theorem's local form, and
    the standard single-layer jump condition) -- this is the physical
    sanity check that caught the missing-2*pi normalization bug that the
    point-by-point elliptic-integral checks above did not. Also exercises
    evaluate_axisymmetric_field's vectorized (fixed-quadrature) path
    directly, away from the near-surface regularization (points here are
    a comfortable 0.1% of R outside the profile)."""
    from specific_particle_tracer.bem.axisymmetric import evaluate_axisymmetric_field

    R = 50e-9
    profile = sphere_cap_profile(R, n_theta=40, theta_max=np.pi)
    sigma0 = 5.0
    sigma = np.full(len(profile), sigma0)

    eps = 1e-3 * R
    points = np.array([[0.0, 0.0, R + eps], [R + eps, 0.0, 0.0]])
    E = evaluate_axisymmetric_field(profile, sigma, points)

    e_z_pole = E[0, 2]
    e_rho_equator = E[1, 0]

    assert abs(e_z_pole - sigma0) / sigma0 < 0.1
    assert abs(e_rho_equator - sigma0) / sigma0 < 0.1


def test_axisymmetric_solve_matches_grounded_sphere_in_uniform_field():
    """Same closed-form case as test_bem_laplace's general-3D version, but
    via the axisymmetric solver."""
    a = 50e-9
    E0 = -1e8

    profile = sphere_cap_profile(a, n_theta=40, theta_max=np.pi)
    solution = AxisymmetricBEMSolution.solve(profile, dirichlet_fn=lambda rho, z: E0 * z)

    theta = np.array([0.0, 0.3, 0.9, 1.5, 2.5, np.pi])
    r = 3.0 * a
    points = np.column_stack([r * np.sin(theta), np.zeros_like(theta), r * np.cos(theta)])

    phi_bem = solution.potential(points)
    phi_analytic = E0 * a**3 * points[:, 2] / r**3

    assert np.max(np.abs(phi_bem - phi_analytic)) / np.abs(E0 * a) < 1e-3


def _valid_hemisphere_points(theta, r):
    """theta measured from the pole; only theta < pi/2 (z > 0) is physical
    for HemisphericalTipField, so keep tests off the mirrored lower half of
    the full-sphere profile HemisphericalTipBEMField actually solves on."""
    return np.column_stack([r * np.sin(theta), np.zeros_like(theta), r * np.cos(theta)])


def test_hemispherical_tip_bem_field_matches_analytic_field_off_surface():
    """Away from the tip surface, the BEM field (real geometry + uniform-
    field superposition trick + full-sphere mirror-symmetry trick, see
    bem.fields) should agree with the closed-form HemisphericalTipField --
    this is what all of that machinery is meant to reproduce."""
    R = 50e-9
    Ez = -1e8

    bem_field = HemisphericalTipBEMField(Ez, R, n_theta=30)
    analytic_field = HemisphericalTipField(Ez, R)

    theta = np.array([0.0, 0.3, 0.9, 1.4])
    points = _valid_hemisphere_points(theta, r=3.0 * R)

    E_bem = bem_field.evaluate(points)
    E_analytic = analytic_field.evaluate(points)

    rel_err = np.linalg.norm(E_bem - E_analytic, axis=-1) / np.abs(Ez)
    assert np.max(rel_err) < 1e-2


def test_hemispherical_tip_bem_field_matches_analytic_field_near_surface():
    """Accuracy should improve monotonically as r shrinks from 3R down
    toward the tip itself. (Within about one profile-segment size of the
    tip, accuracy stops improving with distance and instead depends on
    mesh resolution; that regime is covered by the exactly-on-the-surface
    test below.)"""
    R = 50e-9
    Ez = -1e8

    bem_field = HemisphericalTipBEMField(Ez, R, n_theta=30)
    analytic_field = HemisphericalTipField(Ez, R)

    theta = np.array([0.0, 0.3, 0.9, 1.4])

    def max_rel_err(r):
        points = _valid_hemisphere_points(theta, r=r)
        return np.max(np.linalg.norm(bem_field.evaluate(points) - analytic_field.evaluate(points), axis=-1)) / abs(Ez)

    assert max_rel_err(1.05 * R) < 1e-2


def test_hemispherical_tip_bem_field_matches_analytic_field_exactly_on_the_surface():
    """The whole point of the near-surface regularization in
    bem.axisymmetric is accuracy essentially *at* the tip's surface, where
    particles are actually emitted -- error should shrink monotonically
    with profile resolution (same regularization idea as bem.panel_field's
    3D version, and the same "check convergence, don't just pin a number"
    approach that caught that version's on-surface error not actually
    shrinking under refinement)."""
    R = 50e-9
    Ez = -1e8
    analytic_field = HemisphericalTipField(Ez, R)
    theta = np.array([0.15, 0.4, 0.65, 0.9, 1.1, 1.35])  # generic angles, not aligned to profile nodes
    points = _valid_hemisphere_points(theta, r=R)
    E_analytic = analytic_field.evaluate(points)

    def max_rel_err(n_theta):
        bem_field = HemisphericalTipBEMField(Ez, R, n_theta=n_theta)
        return np.max(np.linalg.norm(bem_field.evaluate(points) - E_analytic, axis=-1)) / np.abs(Ez)

    err_coarse = max_rel_err(10)
    err_fine = max_rel_err(40)

    assert err_coarse < 0.5
    assert err_fine < err_coarse


def test_hemispherical_tip_bem_field_is_zero_inside_the_conductor():
    R = 50e-9
    Ez = -1e8
    bem_field = HemisphericalTipBEMField(Ez, R, n_theta=15)

    inside_tip = np.array([[0.0, 0.0, 0.5 * R], [0.3 * R, 0.0, 0.0]])
    below_plane = np.array([[2 * R, 0.0, -1e-9]])

    E = bem_field.evaluate(np.concatenate([inside_tip, below_plane], axis=0))
    assert np.all(E == 0.0)


def test_closest_point_on_profile_gpu_kernel_matches_cpu():
    """The RawKernel path (dispatched automatically whenever xp is cupy)
    must reproduce the plain-Python-loop CPU path exactly. Worth checking
    independently, not just trusting a passing numpy test: a RawKernel
    reads its arguments as flat pointers with no stride information, so
    passing a non-contiguous array view (e.g. one column of a (n, 2)
    C-order array, which is a stride-2 view, not a contiguous buffer) reads
    silently-wrong-but-finite data with no exception at all -- caught here
    once already (segment/t/normal disagreed while dist2 coincidentally
    matched for a subset of points) before the columns passed to the
    kernel were made explicitly contiguous."""
    cp = pytest.importorskip("cupy")

    R = 50e-9
    profile = sphere_cap_profile(R, n_theta=30)
    p0, p1, tangent, normal = _segment_endpoints(profile)

    rng = np.random.default_rng(0)
    N = 200
    rho = rng.uniform(0, R, N)
    z = rng.uniform(-0.2 * R, 1.2 * R, N)

    bd2_cpu, bseg_cpu, bt_cpu, bn_cpu = _closest_point_on_profile(rho, z, p0, p1, tangent, normal, np)
    bd2_gpu, bseg_gpu, bt_gpu, bn_gpu = _closest_point_on_profile(
        cp.asarray(rho), cp.asarray(z), p0, p1, tangent, normal, cp
    )

    assert np.allclose(cp.asnumpy(bd2_gpu), bd2_cpu, atol=1e-12, rtol=1e-10)
    assert np.array_equal(cp.asnumpy(bseg_gpu), bseg_cpu)
    assert np.allclose(cp.asnumpy(bt_gpu), bt_cpu, atol=1e-12)
    assert np.allclose(cp.asnumpy(bn_gpu), bn_cpu, atol=1e-12)
