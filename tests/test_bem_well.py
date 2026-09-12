import numpy as np
import pytest

from specific_particle_tracer.bem.fields import CylindricalWellBEMField
from specific_particle_tracer.bem.geometry import CylindricalWellBEMGeometry
from specific_particle_tracer.geometry import FlatCathode, Geometry
from specific_particle_tracer.constants import ELEMENTARY_CHARGE

R = 50e-9
H = 30e-9
E_GUN = -1e8
Z0 = 3e-9


# ----------------------------------------------------------------------
# CylindricalWellBEMField: no closed-form ground truth for this shape, so
# these check internal consistency instead -- see the module's own
# docstring correction in bem/fields.py for why (unlike the hemisphere
# tip) there's no way to avoid meshing the plane here.
# ----------------------------------------------------------------------


def test_field_matches_flat_cathode_far_above():
    """Far above the well (many well-radii up), the perturbation from the
    well should have decayed to where the field is indistinguishable from
    a plain flat cathode's -- this is the one regime with a genuine
    ground truth to compare against."""
    field = CylindricalWellBEMField(E_GUN, R, H, max_length=min(R, H) / 24)
    flat = FlatCathode(E_GUN, z0=None)

    points = np.array([[0.0, 0.0, 20.0 * R], [0.3 * R, 0.2 * R, 15.0 * R]])
    E_well = field.evaluate(points)
    E_flat = flat.field(points)
    # atol scaled to the field magnitude, not left at allclose's default
    # 1e-8 -- E_flat's transverse components are exactly zero, and a tiny
    # (~1e-6 relative) truncation residual there would otherwise fail an
    # exact-zero comparison despite being physically negligible (see
    # feedback_allclose_atol_small_scales in project memory).
    assert np.allclose(E_well, E_flat, rtol=1e-3, atol=1e-5 * abs(E_GUN))


def test_field_shallow_well_limit_matches_flat_cathode_at_bottom():
    """As H -> 0 the well degenerates to a flat plane -- the field at the
    (barely-recessed) bottom center should approach the flat-cathode
    value, same sign and magnitude, not some offset limit."""
    H_shallow = 0.01 * R
    # The global `max_length` has to be given explicitly for a well this
    # shallow. Its default is min(R, H)/19, which at H = R/100 means
    # R/1900 applied to *every* region -- including the full-radius bottom
    # disk and the plane out to 20R -- and a per-region override can only
    # refine past the global cap, never relax it (see
    # bem.mesh._region_max_length), so `bottom_max_length=R/29` alone did
    # nothing: this built a 1982-node profile and took 202s, 30% of the
    # whole suite's runtime, to check a rel=0.1 limit. 77 nodes gets the
    # same answer to 2% in 0.5s.
    field = CylindricalWellBEMField(
        E_GUN, R, H_shallow, max_length=R / 29, wall_max_length=H_shallow / 14,
    )
    E = field.evaluate(np.array([[0.0, 0.0, -H_shallow]]))
    assert E[0, 2] == pytest.approx(E_GUN, rel=0.1)


def test_field_at_bottom_converges_under_mesh_refinement():
    """No analytic ground truth for this shape -- so accuracy is checked
    via internal convergence instead (same diagnostic technique used to
    separate real geometric effects from discretization error on the
    hemisphere-tip case)."""
    point = np.array([[0.0, 0.0, -H]])
    max_lengths = [min(R, H) / 15, min(R, H) / 45, min(R, H) / 135]
    values = []
    for max_length in max_lengths:
        field = CylindricalWellBEMField(E_GUN, R, H, max_length=max_length)
        values.append(field.evaluate(point)[0, 2])

    # Each refinement should shrink the change from the previous one --
    # not asserting a specific rate, just that it's converging, not
    # drifting or oscillating.
    d1 = abs(values[1] - values[0])
    d2 = abs(values[2] - values[1])
    assert d2 < 0.5 * d1


def test_field_converges_as_plane_truncation_radius_grows():
    """This shape's field solve (unlike the hemisphere tip's) needs a
    genuinely truncated plane -- check the truncation error is actually
    shrinking as plane_radius grows. The plane region is graded (see
    bem.mesh.graded_annulus_profile), so a fixed max_length keeps its
    near-feature density fixed automatically regardless of plane_radius,
    unlike the uniform meshing this test used to have to compensate for
    by hand."""
    point = np.array([[0.3 * R, 0.0, -0.5 * H]])
    values = []
    for plane_radius in [5.0 * R, 10.0 * R, 20.0 * R]:
        field = CylindricalWellBEMField(E_GUN, R, H, plane_radius=plane_radius, max_length=min(R, H) / 24)
        values.append(field.evaluate(point)[0, 2])

    d1 = abs(values[1] - values[0])
    d2 = abs(values[2] - values[1])
    assert d2 < 0.5 * d1


def test_bottom_field_matches_potential_gradient_and_extracts_electrons():
    """Mesh stability alone formerly blessed a spurious field reversal.
    Check the independent potential derivative just outside the bottom.
    """
    for spacing in [R/25, R/50]:
        field = CylindricalWellBEMField(E_GUN, R, H, max_length=spacing,
            bottom_fillet_radius=1.5e-9, rim_fillet_radius=5e-9)
        point = np.array([.3*R, 0., -H+.002*R])
        h = 1e-5*R
        ez_fd = -(field.potential(point+[0,0,h])-field.potential(point-[0,0,h]))/(2*h)
        ez = field.evaluate(point)[2]
        assert ez < 0.
        assert ez == pytest.approx(ez_fd, rel=1e-4)
        assert field.evaluate(np.array([.3*R, 0., -H]))[2] < 0.


def test_field_zero_inside_conductor():
    field = CylindricalWellBEMField(E_GUN, R, H)
    points = np.array(
        [
            [2.0 * R, 0.0, -0.5 * R],  # under the outer plane
            [0.5 * R, 0.0, -1.5 * H],  # below the well bottom
        ]
    )
    E = field.evaluate(points)
    assert np.allclose(E, 0.0)


# ----------------------------------------------------------------------
# CylindricalWellBEMGeometry
# ----------------------------------------------------------------------


def _make_geometry(**kwargs):
    # Everything else (field_max_length, image_max_length,
    # image_fillet_max_length, rim_fillet_radius,
    # image_rim_fillet_max_length) is left at its built-in default, which
    # already auto-scales sensibly with R/H/z0 -- see
    # bem.geometry.CylindricalWellBEMGeometry's docstring.
    defaults = dict(z0=Z0, image_n_max=10)
    defaults.update(kwargs)
    return CylindricalWellBEMGeometry(E_GUN, R, H, **defaults)


@pytest.fixture(scope="module")
def geom():
    """The all-defaults geometry, built once for the whole module.

    Constructing it runs two BEM solves (the static field one, plus the
    image-charge operator assembly), ~12s -- and six tests below want the
    identical object, which was ~60s of the suite spent rebuilding the
    same thing. Module-scoped rather than function-scoped because none of
    those tests mutate it; they only ever evaluate fields/forces/masks.
    Tests that need *different* construction arguments still call
    `_make_geometry` directly.
    """
    return _make_geometry()


def test_kill_mask_two_piece_boundary(geom):
    positions = np.array(
        [
            [0.0, 0.0, -H],  # exactly at the emission surface: not killed
            [0.0, 0.0, -H * 1.01],  # fell through the bottom: killed
            [0.9 * R, 0.0, -0.5 * H],  # inside the well, off-axis: not killed
            [1.01 * R, 0.0, -0.5 * H],  # crossed into the wall: killed
            [0.0, 0.0, 0.01 * R],  # above the mouth: not killed
            [1.5 * R, 0.0, -0.01 * R],  # under the outer plane: killed
        ]
    )
    expected = np.array([False, True, False, True, False, True])
    assert np.array_equal(geom.kill_mask(positions), expected)


def test_image_force_matches_flat_cathode_far_above(geom):
    """Same ground-truth regime as the field check: far above the well,
    the image-charge force should match a plain flat cathode's."""
    from specific_particle_tracer.forces import image_charge_force

    position = np.array([[[0.0, 0.0, 20.0 * R]]])
    charge = np.array([[-ELEMENTARY_CHARGE]])
    active = np.array([[True]])

    F_well = geom.image_force(position, charge, active, plummer_radius=1e-12)
    F_flat = image_charge_force(position, charge, active, Z0, plummer_radius=1e-12)
    assert np.allclose(F_well, F_flat, rtol=1e-2)


def test_image_force_direction_is_attractive_toward_nearest_surface(geom):
    charge = np.array([[-ELEMENTARY_CHARGE]])
    active = np.array([[True]])

    # Off-axis, deep in the well: force should pull the (negative) charge
    # down toward the bottom (Fz < 0) since the bottom is the nearest
    # grounded surface.
    pos_deep = np.array([[[0.3 * R, 0.0, -H + 0.1 * R]]])
    F_deep = geom.image_force(pos_deep, charge, active, plummer_radius=1e-12)
    assert F_deep[0, 0, 2] < 0.0


def test_image_force_magnitude_decreases_with_height_above_well(geom):
    charge = np.array([[-ELEMENTARY_CHARGE]])
    active = np.array([[True]])

    heights = [-H + 0.1 * R, -0.05 * R, 3.0 * R]
    mags = []
    for z in heights:
        pos = np.array([[[0.3 * R, 0.0, z]]])
        F = geom.image_force(pos, charge, active, plummer_radius=1e-12)
        mags.append(np.linalg.norm(F))
    assert mags[0] > mags[1] > mags[2]


def test_field_and_image_force_handoff_blend_is_smooth(geom):
    """The far-field flat-cathode handoff (see bem/geometry.py's
    `_blend_with_flat_fallback`) should vary continuously through the
    blend zone -- no discontinuity a stepper could trip over."""
    z = np.linspace(0.3 * geom.z_handoff, 1.5 * geom.z_handoff, 40)
    points = np.column_stack([np.full_like(z, 0.3 * R), np.zeros_like(z), z])

    E = geom.field(points)
    Ez = E[:, 2]
    max_jump = np.max(np.abs(np.diff(Ez)))
    assert max_jump / np.max(np.abs(Ez)) < 0.01


def test_worker_args_round_trip(geom):
    rebuilt = Geometry.from_worker_args(geom.worker_args(), xp=np)

    position = np.array([[0.2 * R, 0.0, -0.8 * H]])
    assert np.allclose(geom.field(position), rebuilt.field(position))

    grouped = position.reshape(1, 1, 3)
    charge = np.array([[-ELEMENTARY_CHARGE]])
    active = np.array([[True]])
    assert np.allclose(
        geom.image_force(grouped, charge, active, plummer_radius=1e-12),
        rebuilt.image_force(grouped, charge, active, plummer_radius=1e-12),
    )
    assert np.array_equal(geom.kill_mask(position), rebuilt.kill_mask(position))


def test_image_force_off_when_z0_is_none():
    geom = _make_geometry(z0=None)
    position = np.array([[[0.3 * R, 0.0, -H + 0.1 * R]]])
    charge = np.array([[-ELEMENTARY_CHARGE]])
    active = np.array([[True]])
    F = geom.image_force(position, charge, active, plummer_radius=1e-12)
    assert np.allclose(F, 0.0)
