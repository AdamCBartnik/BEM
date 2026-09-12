import numpy as np
import pytest

from specific_particle_tracer.bem.mesh import (
    hemisphere_tip_real_profile,
    hemisphere_tip_image_profile,
    hemisphere_tip_image_doubled_profile,
    hemisphere_tip_real_mesh,
    hemisphere_tip_image_mesh,
    cylindrical_well_real_profile,
    cylindrical_well_image_profile,
    cylindrical_well_real_mesh,
    cylindrical_well_image_mesh,
    min_distance_to_profile,
    revolve_profile,
    sphere_mesh,
)


def test_image_profile_is_a_constant_offset_from_the_real_profile():
    """The recessed (image-charge) surface must sit exactly z0 away from the
    real surface everywhere -- across the sphere cap, the ridge fillet, and
    the plane alike -- not just on the two flat/spherical pieces."""
    R = 50e-9
    z0 = 3e-9
    plane_radius = 5 * R

    real_profile = hemisphere_tip_real_profile(R, plane_radius, max_length=R / 200)
    image_profile = hemisphere_tip_image_profile(R, z0, plane_radius, max_length=R / 200, fillet_max_length=z0 / 50)

    dist = min_distance_to_profile(image_profile, real_profile)

    assert np.max(np.abs(dist - z0)) / z0 < 1e-3


def test_image_doubled_profile_is_pole_to_pole_and_offset_from_the_real_profile():
    """The mirror-doubled profile should span pole to pole (two poles, no
    plane region), be exactly z0 away from the real surface everywhere
    (same guarantee as the truncated-plane image profile, since it's the
    same cap+fillet piece plus its own mirror image), and stay smooth
    (no large consecutive-point gaps) across the seam where the two
    halves meet."""
    R = 50e-9
    z0 = 3e-9
    max_length = R / 200

    doubled = hemisphere_tip_image_doubled_profile(R, z0, max_length=max_length, fillet_max_length=z0 / 50)
    assert np.allclose(doubled[0], [0.0, R - z0])
    assert np.allclose(doubled[-1], [0.0, -(R + z0)])

    diffs = np.linalg.norm(np.diff(doubled, axis=0), axis=1)
    assert diffs.max() <= max_length * (1.0 + 1e-6)

    real_profile = hemisphere_tip_real_profile(R, plane_radius=5 * R, max_length=max_length)
    dist = min_distance_to_profile(doubled, real_profile)
    # Only the cap+fillet half is meaningfully close to the real surface
    # (the mirrored half lives on the "wrong" side of it) -- restrict the
    # check to points at z >= -z0, i.e. the real (unmirrored) half.
    real_half = doubled[:, 1] >= -z0 - 1e-15
    assert np.max(np.abs(dist[real_half] - z0)) / z0 < 1e-3


def test_image_profile_requires_z0_between_zero_and_R():
    R = 50e-9
    with pytest.raises(ValueError):
        hemisphere_tip_image_profile(R, R, plane_radius=5 * R, max_length=R / 10)
    with pytest.raises(ValueError):
        hemisphere_tip_image_profile(R, -1e-9, plane_radius=5 * R, max_length=R / 10)


def test_revolve_profile_of_a_closed_profile_gives_a_closed_mesh():
    """A profile that starts and ends at a pole (e.g. a lens/sphere shape)
    should revolve into a topologically closed mesh: every edge shared
    between two triangles, with no holes or duplicated patches."""
    n_phi = 12
    profile = np.array([[0.0, 1.0], [0.5, 0.5], [0.0, 0.0]])  # pole -> ring -> pole
    vertices, elements = revolve_profile(profile, n_phi)

    assert vertices.shape[0] == 3
    assert elements.shape[0] == 3
    assert elements.min() >= 0
    assert elements.max() < vertices.shape[1]

    edge_counts = {}
    for tri in elements.T:
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    assert set(edge_counts.values()) == {2}


def test_revolve_profile_of_an_open_profile_has_one_free_boundary_ring():
    """A profile ending at a non-pole ring (like our truncated-plane meshes)
    should be open there: only that outer ring's own circumferential edges
    belong to a single triangle; every other edge is shared by two."""
    n_phi = 12
    profile = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])  # pole -> ring -> open ring
    vertices, elements = revolve_profile(profile, n_phi)

    edge_counts = {}
    for tri in elements.T:
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    outer_ring = set(range(vertices.shape[1] - n_phi, vertices.shape[1]))
    boundary_edges = {edge for edge, count in edge_counts.items() if count == 1}
    interior_edges = {edge for edge, count in edge_counts.items() if count == 2}

    assert set(edge_counts.values()) == {1, 2}
    assert len(boundary_edges) == n_phi
    assert all(a in outer_ring and b in outer_ring for a, b in boundary_edges)
    assert len(interior_edges) == len(edge_counts) - len(boundary_edges)


def test_hemisphere_tip_meshes_have_matching_azimuthal_resolution():
    R = 50e-9
    z0 = 3e-9
    plane_radius = 5 * R
    n_phi = 16

    real_v, real_e = hemisphere_tip_real_mesh(R, plane_radius, R / 8, n_phi=n_phi)
    image_v, image_e = hemisphere_tip_image_mesh(R, z0, plane_radius, R / 6, n_phi=n_phi)

    for vertices, elements in ((real_v, real_e), (image_v, image_e)):
        assert elements.min() >= 0
        assert elements.max() < vertices.shape[1]

    # Vertices with z>0 belong to the sphere-cap piece of the image mesh, and
    # should all sit exactly on a sphere of radius R-z0 about the origin.
    cap_mask = image_v[2] > 0
    r_cap = np.linalg.norm(image_v[:, cap_mask], axis=0)
    assert np.allclose(r_cap, R - z0, rtol=1e-9)


def test_well_real_profile_defaults_to_sharp_corners():
    """Corner rounding is opt-in for the real profile -- the default shape
    is unchanged from before this feature existed."""
    R, H = 50e-9, 30e-9
    profile = cylindrical_well_real_profile(R, H, plane_radius=5 * R, max_length=R / 10)
    assert np.any(np.all(np.isclose(profile, [R, -H], rtol=1e-9, atol=0.0), axis=1))
    assert np.any(np.all(np.isclose(profile, [R, 0.0], rtol=1e-9, atol=0.0), axis=1))


def test_well_real_profile_rounds_corners_in_place():
    """Rounding a corner shouldn't move the nominal disk radius R or wall
    span H -- the straight pieces just stop a bit short on either side of
    where the sharp corner would be, with a tangent arc bridging the gap."""
    R, H = 50e-9, 30e-9
    bottom_fillet_radius, rim_fillet_radius = 3e-9, 5e-9
    profile = cylindrical_well_real_profile(
        R, H, plane_radius=5 * R, max_length=R / 40,
        bottom_fillet_radius=bottom_fillet_radius, rim_fillet_radius=rim_fillet_radius,
    )

    # The old sharp corner points should no longer appear...
    assert not np.any(np.all(np.isclose(profile, [R, -H], rtol=1e-9, atol=0.0), axis=1))
    assert not np.any(np.all(np.isclose(profile, [R, 0.0], rtol=1e-9, atol=0.0), axis=1))
    # ...but R and H are still exactly where the flat/cylindrical pieces
    # reach: the bottom disk still spans out to (R - bottom_fillet_radius,
    # -H) and the wall is still at rho=R.
    assert np.any(np.all(np.isclose(profile, [R - bottom_fillet_radius, -H], rtol=1e-9, atol=0.0), axis=1))
    assert np.any(np.isclose(profile[:, 0], R, rtol=1e-9, atol=0.0))
    # Continuous (no gaps): consecutive points never jump by more than max_length.
    diffs = np.linalg.norm(np.diff(profile, axis=0), axis=1)
    # exclude the graded outer plane, which is allowed to grow past max_length
    non_plane = profile[:-1, 0] <= R + rim_fillet_radius + 1e-9
    assert np.all(diffs[non_plane] <= (R / 40) * (1.0 + 1e-6))


def test_well_real_profile_requires_bottom_plus_rim_fillet_radius_less_than_H():
    R, H = 50e-9, 30e-9
    with pytest.raises(ValueError):
        cylindrical_well_real_profile(
            R, H, plane_radius=5 * R, max_length=R / 10, bottom_fillet_radius=0.6 * H, rim_fillet_radius=0.5 * H
        )


def test_well_image_profile_is_a_constant_offset_from_the_real_profile():
    """Same guarantee as the hemisphere-tip case, across the well's
    bottom-of-well and rim corners alike, plus the flat bottom disk,
    wall, and outer plane pieces -- with rim_fillet_radius > z0, the
    image profile really is a uniform z0 offset of the (rounded) real
    profile everywhere, no localized deviation anywhere (unlike a
    from-scratch, independently-sized image-only rounding)."""
    R, H, z0 = 50e-9, 30e-9, 3e-9
    plane_radius = 5 * R
    bottom_fillet_radius, rim_fillet_radius = 1.5e-9, 2.0 * z0  # rim > z0, required

    real_profile = cylindrical_well_real_profile(
        R, H, plane_radius, max_length=min(R, H) / 300,
        bottom_fillet_radius=bottom_fillet_radius, rim_fillet_radius=rim_fillet_radius,
    )
    image_profile = cylindrical_well_image_profile(
        R, H, z0, plane_radius, max_length=min(R, H) / 300,
        bottom_fillet_radius=bottom_fillet_radius, rim_fillet_radius=rim_fillet_radius,
    )

    dist = min_distance_to_profile(image_profile, real_profile)

    assert np.max(np.abs(dist - z0)) / z0 < 1e-3


def test_well_image_profile_requires_z0_between_zero_and_min_R_H():
    R, H = 50e-9, 30e-9  # R > H: min(R, H) == H
    with pytest.raises(ValueError):
        # z0 == H, the binding constraint here
        cylindrical_well_image_profile(R, H, H, plane_radius=5 * R, max_length=R / 10)
    with pytest.raises(ValueError):
        cylindrical_well_image_profile(R, H, -1e-9, plane_radius=5 * R, max_length=R / 10)

    H2, R2 = 50e-9, 30e-9  # H > R: min(R, H) == R, the other constraint binds instead
    with pytest.raises(ValueError):
        cylindrical_well_image_profile(R2, H2, R2, plane_radius=5 * R2, max_length=R2 / 10)


def test_well_profile_requires_fillet_radii_leaving_a_positive_wall():
    R, H, z0 = 50e-9, 30e-9, 3e-9
    with pytest.raises(ValueError):
        cylindrical_well_real_profile(R, H, 5 * R, max_length=R / 10, rim_fillet_radius=H)
    with pytest.raises(ValueError):
        cylindrical_well_real_profile(R, H, 5 * R, max_length=R / 10, bottom_fillet_radius=-1e-9)
    with pytest.raises(ValueError):
        # Uses max(rim_fillet_radius, z0), not rim_fillet_radius directly --
        # a rim rounding too small to survive erosion still consumes z0's
        # worth of wall.
        cylindrical_well_image_profile(R, H, z0, 5 * R, max_length=R / 10, bottom_fillet_radius=H - z0)


def test_well_image_profile_bottom_corner_always_rounds_under_erosion():
    """The reflex bottom corner's eroded radius is bottom_fillet_radius +
    z0, which is always positive -- so it's rounded in the image profile
    even when the real corner is left perfectly sharp."""
    R, H, z0 = 50e-9, 30e-9, 3e-9
    profile = cylindrical_well_image_profile(R, H, z0, plane_radius=5 * R, max_length=R / 10)
    corner = np.array([R, -H])
    dist_from_corner = np.linalg.norm(profile - corner, axis=1)
    near = dist_from_corner < 2 * z0
    assert near.sum() >= 3  # real interior fillet points, not just tangent endpoints


def test_well_image_profile_rim_reverts_to_sharp_mitre_when_real_rounding_is_too_small():
    """A real rim rounding <= z0 doesn't survive erosion as a smaller
    rounded corner -- it's fully consumed, and the image profile reverts
    to exactly the same sharp mitre an unrounded (rim_fillet_radius=0)
    real corner would erode to."""
    R, H, z0 = 50e-9, 30e-9, 3e-9
    mitre = np.array([R + z0, -z0])

    sharp = cylindrical_well_image_profile(R, H, z0, plane_radius=5 * R, max_length=R / 10)
    insufficient = cylindrical_well_image_profile(
        R, H, z0, plane_radius=5 * R, max_length=R / 10, rim_fillet_radius=0.5 * z0
    )
    at_the_boundary = cylindrical_well_image_profile(
        R, H, z0, plane_radius=5 * R, max_length=R / 10, rim_fillet_radius=z0
    )
    for profile in (sharp, insufficient, at_the_boundary):
        hit = np.any(np.all(np.isclose(profile, mitre, rtol=1e-9, atol=0.0), axis=1))
        assert hit


def test_well_image_profile_rim_is_rounded_when_real_rounding_exceeds_z0():
    """Only once the real rim rounding is genuinely bigger than z0 does
    it survive erosion as a real (smaller) rounded corner in the image
    profile, replacing the sharp mitre."""
    R, H, z0 = 50e-9, 30e-9, 3e-9
    mitre = np.array([R + z0, -z0])
    profile = cylindrical_well_image_profile(
        R, H, z0, plane_radius=5 * R, max_length=R / 10, rim_fillet_radius=2.0 * z0, rim_fillet_max_length=z0 / 10,
    )

    hit = np.any(np.all(np.isclose(profile, mitre, rtol=1e-9, atol=0.0), axis=1))
    assert not hit

    dist_from_mitre = np.linalg.norm(profile - mitre, axis=1)
    near_corner = dist_from_mitre < 4.0 * z0
    assert near_corner.sum() >= 3  # real interior fillet points, not just tangent endpoints


def test_well_image_profile_rim_closest_point_normal_is_continuous():
    """Regression test for the bug this rounding fixes: sweeping past the
    (now-rounded) rim should never show the mirror-charge normal flipping
    discontinuously the way it did against the old sharp mitre -- see
    bem.image_charge.ImageChargeBEMSolution._closest_points_batch."""
    from specific_particle_tracer.bem.image_charge import ImageChargeBEMSolution

    R, H, z0 = 25e-9, 20e-9, 3e-9
    profile = cylindrical_well_image_profile(
        R, H, z0, plane_radius=5 * R, max_length=R / 40,
        rim_fillet_radius=2.0 * z0, rim_fillet_max_length=z0 / 50,
    )
    # Constructed directly rather than via `.solve`: this test only ever
    # asks for the closest-point search, which comes from the profile
    # geometry in __init__ and never touches the mode operators `A`.
    # Assembling them anyway (adaptive quadrature -- the expensive part of
    # this whole package's test suite) cost 52s here for nothing.
    sol = ImageChargeBEMSolution(profile, A=None, n_max=0)

    # Sweep rho across the rounded corner at a height just above it -- the
    # same neighborhood where a real particle escaping the well got stuck
    # forever against the old sharp mitre.
    rho = np.linspace(R + z0 - 5e-9, R + z0 + 5e-9, 400)
    z = np.full_like(rho, -z0 + 1e-9)
    _, _, normal, _ = sol._closest_points_batch(rho, z)

    # The angle the normal turns through between consecutive sweep points
    # should never jump by more than a fraction of a degree -- a sharp
    # corner would instead show one single point where it jumps by ~90 deg.
    cos_step = np.sum(normal[:-1] * normal[1:], axis=1)
    max_turn_deg = np.degrees(np.arccos(np.clip(cos_step, -1.0, 1.0))).max()
    assert max_turn_deg < 5.0


def test_well_meshes_have_matching_azimuthal_resolution():
    R, H, z0 = 50e-9, 30e-9, 3e-9
    plane_radius = 5 * R
    n_phi = 16

    real_v, real_e = cylindrical_well_real_mesh(R, H, plane_radius, min(R, H) / 8, n_phi=n_phi)
    image_v, image_e = cylindrical_well_image_mesh(R, H, z0, plane_radius, min(R, H) / 8, n_phi=n_phi)

    for vertices, elements in ((real_v, real_e), (image_v, image_e)):
        assert elements.min() >= 0
        assert elements.max() < vertices.shape[1]

    # Bottom-disk vertices of the real mesh should sit exactly at z=-H.
    bottom_mask = np.isclose(real_v[2], -H)
    assert bottom_mask.sum() > 0
    assert np.all(np.linalg.norm(real_v[:2, bottom_mask], axis=0) <= R * (1.0 + 1e-9))


def test_sphere_mesh_is_closed_and_all_vertices_lie_on_the_sphere():
    R = 50e-9
    vertices, elements = sphere_mesh(R, n_theta=20, n_phi=24)

    assert elements.min() >= 0
    assert elements.max() < vertices.shape[1]
    assert np.allclose(np.linalg.norm(vertices, axis=0), R, rtol=1e-9)

    edge_counts = {}
    for tri in elements.T:
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    assert set(edge_counts.values()) == {2}
