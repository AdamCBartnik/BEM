import numpy as np

from specific_particle_tracer.fields import GunField, HemisphericalTipField
from specific_particle_tracer.bem.fields import (
    HemisphericalTipBEMField, CylindricalWellBEMField, _cylindrical_well_contains,
)

R = 50e-9
EZ = -1e8


def _fd_gradient(potential_fn, point, h):
    """-grad(potential_fn) at `point` (shape (3,)) via central differences."""
    grads = []
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = h
        v_plus = potential_fn((point + dp).reshape(1, 3))[0]
        v_minus = potential_fn((point - dp).reshape(1, 3))[0]
        grads.append(-(v_plus - v_minus) / (2.0 * h))
    return np.array(grads)


def test_gunfield_potential_matches_field_gradient():
    field = GunField(EZ)
    point = np.array([0.3 * R, 0.2 * R, 2.0 * R])
    E_direct = field.evaluate(point.reshape(1, 3))[0]
    E_fd = _fd_gradient(field.potential, point, h=1e-4 * R)
    assert np.allclose(E_fd, E_direct, rtol=1e-6)


def test_gunfield_potential_zero_below_z0():
    field = GunField(EZ)
    assert field.potential(np.array([[0.0, 0.0, -1e-9]])) == 0.0


def test_hemisphere_tip_potential_matches_field_gradient():
    field = HemisphericalTipField(EZ, R)
    for point in [
        np.array([0.3 * R, 0.2 * R, 1.2 * R]),
        np.array([0.05 * R, 0.0, 1.05 * R]),  # near the pole
    ]:
        E_direct = field.evaluate(point.reshape(1, 3))[0]
        E_fd = _fd_gradient(field.potential, point, h=1e-4 * R)
        assert np.allclose(E_fd, E_direct, rtol=1e-5)


def test_hemisphere_tip_potential_zero_on_and_inside_tip():
    field = HemisphericalTipField(EZ, R)
    on_surface = R * np.array([[np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))]])
    assert np.allclose(field.potential(on_surface), 0.0, atol=1e-6)
    inside = np.array([[0.0, 0.0, 0.5 * R]])
    assert field.potential(inside) == 0.0


def test_hemisphere_tip_bem_field_potential_matches_field_gradient():
    field = HemisphericalTipBEMField(EZ, R, max_length=R * np.pi / 39)
    point = np.array([0.3 * R, 0.2 * R, 1.2 * R])
    E_direct = field.evaluate(point.reshape(1, 3))[0]
    E_fd = _fd_gradient(field.potential, point, h=1e-4 * R)
    assert np.allclose(E_fd, E_direct, rtol=1e-5)


def test_hemisphere_tip_bem_field_potential_matches_analytic():
    """Cross-check against the closed-form solution, not just internal
    gradient consistency."""
    bem_field = HemisphericalTipBEMField(EZ, R, max_length=R * np.pi / 59)
    analytic_field = HemisphericalTipField(EZ, R)
    points = np.array(
        [
            [0.3 * R, 0.2 * R, 1.2 * R],
            [0.0, 0.0, 1.5 * R],
            [0.8 * R, 0.0, 2.0 * R],
        ]
    )
    V_bem = bem_field.potential(points)
    V_analytic = analytic_field.potential(points)
    assert np.allclose(V_bem, V_analytic, rtol=2e-2)


def test_cylindrical_well_bem_field_potential_matches_field_gradient():
    H = 5e-9
    field = CylindricalWellBEMField(EZ, R, H, max_length=min(R, H) / 24)
    for point in [
        np.array([0.2 * R, 0.1 * R, -0.5 * H]),  # inside the well
        np.array([0.3 * R, 0.0, 2.0 * R]),  # above
    ]:
        E_direct = field.evaluate(point.reshape(1, 3))[0]
        E_fd = _fd_gradient(field.potential, point, h=1e-4 * R)
        assert np.allclose(E_fd, E_direct, rtol=1e-4)


def test_cylindrical_well_bem_field_potential_zero_inside_conductor():
    H = 5e-9
    field = CylindricalWellBEMField(EZ, R, H, max_length=min(R, H) / 19)
    points = np.array(
        [
            [2.0 * R, 0.0, -0.5 * R],  # under the outer plane
            [0.5 * R, 0.0, -1.5 * H],  # below the well bottom
        ]
    )
    assert np.allclose(field.potential(points), 0.0)


def test_cylindrical_well_contains_matches_old_sharp_rule_by_default():
    """Regression check for a real bug: with corner rounding introduced,
    the old simple sharp-corner threshold (rho>=R, z>=0 or z>=-H) no
    longer matches the actual (rounded) conductor near a corner -- a
    query point can sit in a sliver that's actually vacuum (just outside
    the real rounded surface) but still gets zeroed as "inside the
    conductor" by the old rule. With both radii 0 (the default), the new
    helper must still match that old rule exactly everywhere."""
    R_, H_ = 50e-9, 30e-9
    rho = np.linspace(0.0, 2 * R_, 50)
    z = np.linspace(-2 * H_, 2 * H_, 50)
    RHO, Z = np.meshgrid(rho, z)
    old_rule = np.where(RHO >= R_, Z <= 0.0, Z <= -H_)  # True = inside conductor
    new_rule = _cylindrical_well_contains(RHO, Z, R_, H_, 0.0, 0.0, np)
    assert np.array_equal(old_rule, new_rule)


def test_cylindrical_well_contains_accounts_for_rim_rounding():
    """A point just outside a rounded rim's fillet arc, but inside where
    the old sharp-corner rule would have called it conductor, must be
    reported as vacuum (not conductor)."""
    R_, H_ = 50e-9, 30e-9
    rim_fillet_radius = 6e-9
    # Just outside the fillet arc, at 45 degrees from its center, in the
    # sliver the old rule (rho>=R -> z>=0) would wrongly call material
    # (rho > R and z < 0 there).
    center = np.array([R_ + rim_fillet_radius, -rim_fillet_radius])
    point = center + (rim_fillet_radius * 1.05) * np.array([-1.0, 1.0]) / np.sqrt(2)
    rho, z = point
    assert rho > R_ and z < 0.0  # confirms it's in the "old rule says material" sliver

    is_material = _cylindrical_well_contains(
        np.array([rho]), np.array([z]), R_, H_, 0.0, rim_fillet_radius, np
    )
    assert not is_material[0]
