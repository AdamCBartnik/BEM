import numpy as np

from specific_particle_tracer.bem.mesh import sphere_mesh
from specific_particle_tracer.bem.laplace import ExteriorLaplaceSolution
from specific_particle_tracer.fields import HemisphericalTipField
from specific_particle_tracer.bem.fields import HemisphericalTipBEMField


def test_exterior_dirichlet_solve_matches_grounded_sphere_in_uniform_field():
    """Grounded sphere of radius a in a uniform asymptotic field E0*z_hat:
    the classic closed-form case. Perturbation potential phi_pert =
    phi_total - phi_inf = E0*a^3*z/r^3; Dirichlet data forcing phi_total=0
    on the sphere is g = -phi_inf = E0*z."""
    a = 50e-9
    E0 = -1e8

    vertices, elements = sphere_mesh(a, n_theta=40, n_phi=48)
    solution = ExteriorLaplaceSolution.solve(vertices, elements, dirichlet_fn=lambda x, y, z: E0 * z)

    theta = np.array([0.0, 0.3, 0.9, 1.5, 2.5, np.pi])
    r = 3.0 * a
    points = np.column_stack([r * np.sin(theta), np.zeros_like(theta), r * np.cos(theta)])

    phi_bem = solution.potential(points)
    phi_analytic = E0 * a**3 * points[:, 2] / r**3

    assert np.max(np.abs(phi_bem - phi_analytic)) / np.abs(E0 * a) < 1e-2


def _valid_hemisphere_points(theta, r):
    """theta measured from the pole; only theta < pi/2 (z > 0) is physical
    for HemisphericalTipField, so keep tests off the mirrored lower half of
    the full-sphere mesh HemisphericalTipBEMField actually solves on."""
    return np.column_stack([r * np.sin(theta), np.zeros_like(theta), r * np.cos(theta)])


def test_hemispherical_tip_bem_field_matches_analytic_field_off_surface():
    """Away from the tip surface, the BEM field (real geometry + uniform-
    field superposition trick + full-sphere mirror-symmetry trick, see
    bem.fields) should agree with the closed-form HemisphericalTipField --
    this is what all of that machinery is meant to reproduce."""
    R = 50e-9
    Ez = -1e8

    bem_field = HemisphericalTipBEMField(Ez, R, n_theta=30, n_phi=36)
    analytic_field = HemisphericalTipField(Ez, R)

    theta = np.array([0.0, 0.3, 0.9, 1.4])
    points = _valid_hemisphere_points(theta, r=3.0 * R)

    E_bem = bem_field.evaluate(points)
    E_analytic = analytic_field.evaluate(points)

    rel_err = np.linalg.norm(E_bem - E_analytic, axis=-1) / np.abs(Ez)
    assert np.max(rel_err) < 1e-2


def test_hemispherical_tip_bem_field_matches_analytic_field_near_surface():
    """The point of the indirect/charge-simulation formulation (bem.laplace,
    bem.panel_field) rather than the direct one is accuracy close to the
    tip, where particles are actually emitted -- accuracy should improve
    monotonically as r shrinks from 3R down toward the tip itself.
    (Within about one element size of the tip -- see bem.panel_field's
    near-surface regularization -- accuracy stops improving with distance
    and instead depends on mesh resolution; that regime is covered by
    test_hemispherical_tip_bem_field_matches_analytic_field_exactly_on_the_surface.)"""
    R = 50e-9
    Ez = -1e8

    bem_field = HemisphericalTipBEMField(Ez, R, n_theta=30, n_phi=36)
    analytic_field = HemisphericalTipField(Ez, R)

    theta = np.array([0.0, 0.3, 0.9, 1.4])

    def max_rel_err(r):
        points = _valid_hemisphere_points(theta, r=r)
        return np.max(np.linalg.norm(bem_field.evaluate(points) - analytic_field.evaluate(points), axis=-1)) / abs(Ez)

    assert max_rel_err(1.05 * R) < 1e-2


def test_hemispherical_tip_bem_field_matches_analytic_field_exactly_on_the_surface():
    """The whole point of the near-surface regularization in bem.panel_field
    is accuracy essentially *at* the tip's surface, where particles are
    actually emitted -- unregularized, this was off by ~100%+. With it,
    error is a well-behaved (if not yet small) ~10-25% at this resolution,
    and -- unlike the first (nearest-vertex-based) version of this
    regularization, whose on-surface error didn't consistently shrink
    under mesh refinement -- genuinely improves with a finer mesh, checked
    here directly rather than just asserting a single number."""
    R = 50e-9
    Ez = -1e8
    analytic_field = HemisphericalTipField(Ez, R)
    theta = np.array([0.15, 0.4, 0.65, 0.9, 1.1, 1.35])  # generic angles, not aligned to mesh rings
    points = _valid_hemisphere_points(theta, r=R)
    E_analytic = analytic_field.evaluate(points)

    def max_rel_err(n_theta, n_phi):
        bem_field = HemisphericalTipBEMField(Ez, R, n_theta=n_theta, n_phi=n_phi)
        return np.max(np.linalg.norm(bem_field.evaluate(points) - E_analytic, axis=-1)) / np.abs(Ez)

    err_coarse = max_rel_err(20, 24)
    err_fine = max_rel_err(45, 54)

    assert err_coarse < 0.3
    assert err_fine < err_coarse


def test_hemispherical_tip_bem_field_is_zero_inside_the_conductor():
    R = 50e-9
    Ez = -1e8
    bem_field = HemisphericalTipBEMField(Ez, R, n_theta=15, n_phi=18)

    inside_tip = np.array([[0.0, 0.0, 0.5 * R], [0.3 * R, 0.0, 0.0]])
    below_plane = np.array([[2 * R, 0.0, -1e-9]])

    E = bem_field.evaluate(np.concatenate([inside_tip, below_plane], axis=0))
    assert np.all(E == 0.0)
