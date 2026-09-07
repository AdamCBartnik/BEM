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
    tip, where particles are actually emitted -- this should hold up
    within a percent of R, not just far away. (Exactly at r=R itself is
    still inaccurate -- see bem.fields' module docstring -- because the
    flat-faceted mesh sits just barely outside the analytic sphere there,
    a standoff that shrinks rather than grows under mesh refinement.)"""
    R = 50e-9
    Ez = -1e8

    bem_field = HemisphericalTipBEMField(Ez, R, n_theta=30, n_phi=36)
    analytic_field = HemisphericalTipField(Ez, R)

    theta = np.array([0.0, 0.3, 0.9, 1.4])

    points_1p05R = _valid_hemisphere_points(theta, r=1.05 * R)
    rel_err_1p05R = np.linalg.norm(
        bem_field.evaluate(points_1p05R) - analytic_field.evaluate(points_1p05R), axis=-1
    ) / np.abs(Ez)
    assert np.max(rel_err_1p05R) < 1e-2

    points_1p01R = _valid_hemisphere_points(theta, r=1.01 * R)
    rel_err_1p01R = np.linalg.norm(
        bem_field.evaluate(points_1p01R) - analytic_field.evaluate(points_1p01R), axis=-1
    ) / np.abs(Ez)
    assert np.max(rel_err_1p01R) < 0.05


def test_hemispherical_tip_bem_field_is_zero_inside_the_conductor():
    R = 50e-9
    Ez = -1e8
    bem_field = HemisphericalTipBEMField(Ez, R, n_theta=15, n_phi=18)

    inside_tip = np.array([[0.0, 0.0, 0.5 * R], [0.3 * R, 0.0, 0.0]])
    below_plane = np.array([[2 * R, 0.0, -1e-9]])

    E = bem_field.evaluate(np.concatenate([inside_tip, below_plane], axis=0))
    assert np.all(E == 0.0)
