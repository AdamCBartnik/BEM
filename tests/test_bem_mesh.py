import numpy as np
import pytest

from specific_particle_tracer.bem.mesh import (
    hemisphere_tip_real_profile,
    hemisphere_tip_image_profile,
    hemisphere_tip_real_mesh,
    hemisphere_tip_image_mesh,
    min_distance_to_profile,
    revolve_profile,
    sphere_mesh,
    vertex_normals,
)


def test_image_profile_is_a_constant_offset_from_the_real_profile():
    """The recessed (image-charge) surface must sit exactly z0 away from the
    real surface everywhere -- across the sphere cap, the ridge fillet, and
    the plane alike -- not just on the two flat/spherical pieces."""
    R = 50e-9
    z0 = 3e-9
    plane_radius = 5 * R

    real_profile = hemisphere_tip_real_profile(R, plane_radius, n_theta=400, n_r=400)
    image_profile = hemisphere_tip_image_profile(R, z0, plane_radius, n_theta=200, n_fillet=100, n_r=200)

    dist = min_distance_to_profile(image_profile, real_profile)

    assert np.max(np.abs(dist - z0)) / z0 < 1e-3


def test_image_profile_requires_z0_between_zero_and_R():
    R = 50e-9
    with pytest.raises(ValueError):
        hemisphere_tip_image_profile(R, R, plane_radius=5 * R)
    with pytest.raises(ValueError):
        hemisphere_tip_image_profile(R, -1e-9, plane_radius=5 * R)


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

    real_v, real_e = hemisphere_tip_real_mesh(R, plane_radius, n_theta=10, n_r=10, n_phi=n_phi)
    image_v, image_e = hemisphere_tip_image_mesh(R, z0, plane_radius, n_theta=8, n_fillet=6, n_r=8, n_phi=n_phi)

    for vertices, elements in ((real_v, real_e), (image_v, image_e)):
        assert elements.min() >= 0
        assert elements.max() < vertices.shape[1]

    # Vertices with z>0 belong to the sphere-cap piece of the image mesh, and
    # should all sit exactly on a sphere of radius R-z0 about the origin.
    cap_mask = image_v[2] > 0
    r_cap = np.linalg.norm(image_v[:, cap_mask], axis=0)
    assert np.allclose(r_cap, R - z0, rtol=1e-9)


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


def test_vertex_normals_point_outward_on_a_sphere():
    """On a sphere the true outward normal at any point is just its own
    radial direction -- a strong, geometry-independent check that
    revolve_profile's triangle winding (and vertex_normals' use of it) is
    self-consistent. vertex_normals is only an area-weighted *approximation*
    to the true normal (exact only in the continuum limit), so check angular
    deviation with a tolerance appropriate to this mesh's resolution rather
    than expecting an exact match."""
    R = 50e-9
    vertices, elements = sphere_mesh(R, n_theta=20, n_phi=24)
    normals = vertex_normals(vertices, elements)

    assert np.allclose(np.linalg.norm(normals, axis=0), 1.0)

    radial = vertices / np.linalg.norm(vertices, axis=0)
    cos_angle = np.clip(np.sum(normals * radial, axis=0), -1.0, 1.0)
    assert np.degrees(np.arccos(cos_angle)).max() < 5.0
