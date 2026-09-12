import numpy as np

from specific_particle_tracer.forces import (
    hemispherical_tip_image_force,
    image_charge_force,
    HemisphericalTipImageSolution,
    FlatCathodeImageSolution,
)
from specific_particle_tracer.geometry import HemisphericalTip, FlatCathode

R = 50e-9
Z0 = 3e-9
Q = -1.0


def _fd_gradient(potential_fn, point, h):
    grads = []
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = h
        grads.append(-(potential_fn(point + dp) - potential_fn(point - dp)) / (2.0 * h))
    return np.array(grads)


def test_hemispherical_tip_image_solution_matches_exact_force():
    a = R - Z0
    sol = HemisphericalTipImageSolution(a, Z0)

    theta = np.radians(25.0)
    source = (1.15 * R) * np.array([np.sin(theta), 0.0, np.cos(theta)])

    F_direct = hemispherical_tip_image_force(
        source.reshape(1, 1, 3), np.array([[Q]]), np.array([[True]]), a, plummer_radius=1e-12, plane_z0=Z0
    )[0, 0]

    E_fd = _fd_gradient(lambda p: sol.image_potential(source, Q, p), source, h=1e-4 * R)
    F_fd = Q * E_fd

    assert np.allclose(F_fd, F_direct, rtol=1e-6)


def test_hemispherical_tip_image_solution_zero_inside_real_conductor():
    a = R - Z0
    sol = HemisphericalTipImageSolution(a, Z0, real_radius=R)
    source = np.array([0.3 * R, 0.0, 1.1 * R])

    inside_points = np.array(
        [
            [0.0, 0.0, 0.5 * a],  # inside the image sphere
            [2.0 * R, 0.0, -0.5 * R],  # below the plane
        ]
    )
    V = sol.image_potential(source, Q, inside_points)
    assert np.allclose(V, 0.0)


def test_hemispherical_tip_image_solution_masks_against_real_not_reduced_radius():
    """The image construction uses the reduced radius R-z0, but a query
    point with reduced_radius < r < R is still physically *inside the
    real tip* (solid material, since the real tip occupies all r <= R) --
    masking against the reduced radius alone would incorrectly leave it
    unmasked (nonzero), as if it were vacuum."""
    a = R - Z0
    source = np.array([0.3 * R, 0.0, 1.1 * R])

    # Between the reduced and real radius, well above the base plane --
    # real material (r < R), should be masked to 0.
    theta = np.radians(40.0)
    point_in_shell = 0.99 * R * np.array([np.sin(theta), 0.0, np.cos(theta)])
    assert a < np.linalg.norm(point_in_shell) < R  # confirms the point is in the shell being tested

    sol = HemisphericalTipImageSolution(a, Z0, real_radius=R)
    assert sol.image_potential(source, Q, point_in_shell) == 0.0

    # Without real_radius, masking falls back to the reduced radius alone
    # -- the same point is incorrectly left unmasked (the old behavior,
    # kept only for callers that don't have a real profile to pass).
    sol_no_real = HemisphericalTipImageSolution(a, Z0)
    assert sol_no_real.image_potential(source, Q, point_in_shell) != 0.0


def test_flat_cathode_image_solution_matches_exact_force():
    sol = FlatCathodeImageSolution(Z0)
    source = np.array([0.3 * R, 0.2 * R, 0.5 * R])

    F_direct = image_charge_force(
        source.reshape(1, 1, 3), np.array([[Q]]), np.array([[True]]), Z0, plummer_radius=1e-12
    )[0, 0]

    E_fd = _fd_gradient(lambda p: sol.image_potential(source, Q, p), source, h=1e-4 * R)
    F_fd = Q * E_fd

    assert np.allclose(F_fd, F_direct, rtol=1e-6)


def test_flat_cathode_image_solution_zero_inside_conductor():
    sol = FlatCathodeImageSolution(Z0)
    source = np.array([0.3 * R, 0.2 * R, 0.5 * R])
    V = sol.image_potential(source, Q, np.array([[0.0, 0.0, -2.0 * Z0]]))
    assert np.allclose(V, 0.0)


def test_flat_cathode_image_solution_masks_at_real_plane_not_recessed_one():
    """A query point with -z0 < z < 0 is physically inside the real
    conductor (which occupies all z <= 0), even though the image
    construction itself treats the plane as recessed to z = -z0."""
    sol = FlatCathodeImageSolution(Z0)
    source = np.array([0.3 * R, 0.2 * R, 0.5 * R])
    point_in_sliver = np.array([0.3 * R, 0.2 * R, -0.5 * Z0])
    assert sol.image_potential(source, Q, point_in_sliver) == 0.0


def test_hemispherical_tip_geometry_image_solution_property():
    tip = HemisphericalTip(-1e8, R, z0=Z0)
    sol = tip.image_solution
    assert isinstance(sol, HemisphericalTipImageSolution)
    assert np.isclose(sol.sphere_radius, R - Z0)
    assert np.isclose(sol.real_radius, R)

    tip_no_image = HemisphericalTip(-1e8, R, z0=None)
    assert tip_no_image.image_solution is None


def test_flat_cathode_geometry_image_solution_property():
    flat = FlatCathode(-1e8, z0=Z0)
    sol = flat.image_solution
    assert isinstance(sol, FlatCathodeImageSolution)
    assert np.isclose(sol.z0, Z0)

    flat_no_image = FlatCathode(-1e8, z0=None)
    assert flat_no_image.image_solution is None


def test_hemispherical_tip_field_solver_matches_field():
    tip = HemisphericalTip(-1e8, R)
    position = np.array([[0.3 * R, 0.2 * R, 1.2 * R]])
    assert np.allclose(tip.field_solver.evaluate(position), tip.field(position))


def test_flat_cathode_field_solver_matches_field():
    flat = FlatCathode(-1e8)
    position = np.array([[0.3 * R, 0.2 * R, 1.2 * R]])
    assert np.allclose(flat.field_solver.evaluate(position), flat.field(position))
