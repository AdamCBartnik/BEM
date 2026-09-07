"""Axisymmetric surface meshing for BEM geometries.

A conductor's *real* surface sets the static-field boundary condition and
kills particles that cross it. Its *image* (or "recessed") surface is the
same shape offset inward by the image-plane depth ``z0``, and is what a BEM
solver should use as the source surface for the image-charge correction (the
same regularization idea as ``geometry.FlatCathode``'s and
``geometry.HemisphericalTip``'s analytic ``z0`` image-plane offset).

Offsetting a surface inward is a Minkowski erosion: convex pieces (a sphere,
a plane) simply shrink/translate, but a reflex ("concave", as seen from the
solid) edge -- like the ridge where a hemispherical tip meets the surrounding
plane -- gets rounded into a fillet, because a ball of radius z0 can't be
pushed into a sharp reflex corner without rounding it. For the hemisphere-
on-plane case that fillet is exactly a quarter-torus: tube radius z0, major
radius R, centered on the original ridge circle.

All geometries here are axisymmetric, so each is built as a 2D generating
profile (r, z) in the half-plane r >= 0, then revolved around the z-axis
into a triangulated 3D mesh. Mesh arrays follow BEMpp's convention:
``vertices`` has shape (3, n_vertices), ``elements`` has shape (3,
n_elements) of vertex indices per triangle.
"""

import numpy as np


def sphere_cap_profile(radius, n_theta, theta_max=np.pi / 2):
    """Profile of a sphere cap, from the pole (r=0) to polar angle theta_max.

    Returns an (n_theta, 2) array of (r, z) points, pole first.
    """
    theta = np.linspace(0.0, theta_max, n_theta)
    r = radius * np.sin(theta)
    z = radius * np.cos(theta)
    return np.column_stack([r, z])


def flat_annulus_profile(r_inner, r_outer, z, n_r):
    """Profile of a flat annulus at height z, from r_inner to r_outer."""
    r = np.linspace(r_inner, r_outer, n_r)
    return np.column_stack([r, np.full_like(r, z)])


def torus_fillet_profile(major_radius, tube_radius, phi_start, phi_end, n_phi):
    """Profile of a torus tube slice: (major_radius + tube_radius*cos(phi),
    tube_radius*sin(phi)) for phi in [phi_start, phi_end]."""
    phi = np.linspace(phi_start, phi_end, n_phi)
    r = major_radius + tube_radius * np.cos(phi)
    z = tube_radius * np.sin(phi)
    return np.column_stack([r, z])


def sphere_profile(radius, n_theta):
    """Profile of a full closed sphere, pole to pole (theta: 0 to pi)."""
    return sphere_cap_profile(radius, n_theta, theta_max=np.pi)


def sphere_mesh(radius, n_theta=40, n_phi=48):
    """Triangulated mesh of a full closed sphere."""
    return revolve_profile(sphere_profile(radius, n_theta), n_phi)


def hemisphere_tip_real_profile(R, plane_radius, n_theta=40, n_r=40):
    """Generating profile of the real (physical) hemisphere-on-plane
    conductor surface: sphere cap of radius R, then the surrounding plane
    (z=0) out to plane_radius."""
    cap = sphere_cap_profile(R, n_theta)
    plane = flat_annulus_profile(R, plane_radius, 0.0, n_r)[1:]
    return np.concatenate([cap, plane], axis=0)


def hemisphere_tip_image_profile(R, z0, plane_radius, n_theta=40, n_fillet=20, n_r=40):
    """Generating profile of the recessed image-charge surface for a
    hemisphere-on-plane conductor: sphere cap of radius R-z0, a quarter-torus
    fillet (tube radius z0, major radius R) rounding the ridge, then the
    plane at z=-z0 out to plane_radius.

    Requires 0 < z0 < R (the cap must not shrink to a point or invert).
    """
    if not (0.0 < z0 < R):
        raise ValueError(f"z0 must satisfy 0 < z0 < R (got z0={z0!r}, R={R!r})")
    cap = sphere_cap_profile(R - z0, n_theta)
    fillet = torus_fillet_profile(R, z0, np.pi, 1.5 * np.pi, n_fillet)[1:]
    plane = flat_annulus_profile(R, plane_radius, -z0, n_r)[1:]
    return np.concatenate([cap, fillet, plane], axis=0)


def revolve_profile(profile_rz, n_phi, pole_tol=1e-12):
    """Revolve a 2D generating profile (r, z), r>=0, ordered along the
    surface, around the z-axis into a triangulated 3D mesh.

    A profile point with r ~ 0 is treated as a pole (a single shared vertex
    rather than a full ring of n_phi vertices).

    Returns (vertices, elements): vertices has shape (3, n_vertices),
    elements has shape (3, n_elements) of vertex indices (BEMpp convention).
    """
    profile_rz = np.asarray(profile_rz, dtype=float)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    vertices = []
    ring_indices = []
    for r, z in profile_rz:
        if r < pole_tol:
            idx = len(vertices)
            vertices.append((0.0, 0.0, z))
            ring_indices.append(np.full(n_phi, idx, dtype=int))
        else:
            start = len(vertices)
            vertices.extend((r * c, r * s, z) for c, s in zip(cos_phi, sin_phi))
            ring_indices.append(np.arange(start, start + n_phi))

    elements = []
    for i in range(len(ring_indices) - 1):
        ring_a, ring_b = ring_indices[i], ring_indices[i + 1]
        for k in range(n_phi):
            a, b = ring_a[k], ring_a[(k + 1) % n_phi]
            c, d = ring_b[k], ring_b[(k + 1) % n_phi]
            if a == b:  # ring i is a pole
                elements.append((a, d, c))
            elif c == d:  # ring i+1 is a pole
                elements.append((a, b, c))
            else:
                elements.append((a, b, d))
                elements.append((a, d, c))

    vertices = np.asarray(vertices, dtype=float).T
    elements = np.asarray(elements, dtype=int).T
    return vertices, elements


def hemisphere_tip_real_mesh(R, plane_radius, n_theta=40, n_r=40, n_phi=48):
    """Triangulated mesh of the real hemisphere-on-plane conductor surface."""
    profile = hemisphere_tip_real_profile(R, plane_radius, n_theta=n_theta, n_r=n_r)
    return revolve_profile(profile, n_phi)


def hemisphere_tip_image_mesh(R, z0, plane_radius, n_theta=40, n_fillet=20, n_r=40, n_phi=48):
    """Triangulated mesh of the recessed image-charge surface for a
    hemisphere-on-plane conductor."""
    profile = hemisphere_tip_image_profile(
        R, z0, plane_radius, n_theta=n_theta, n_fillet=n_fillet, n_r=n_r
    )
    return revolve_profile(profile, n_phi)


def vertex_normals(vertices, elements):
    """Area-weighted outward unit normal at each mesh vertex: the average
    of the (outward) normals of every triangle touching that vertex,
    weighted by triangle area.

    Used by `panel_field`'s near-panel field regularization, which needs a
    local outward-normal direction near an evaluation point without
    assuming anything about the underlying analytic geometry (unlike, e.g.,
    just using x/|x| for a sphere).

    Parameters
    ----------
    vertices, elements : the mesh, BEMpp convention (see module docstring).

    Returns
    -------
    normals : ndarray, shape (3, n_vertices), unit vectors.
    """
    vertices = np.asarray(vertices, dtype=float)
    elements = np.asarray(elements, dtype=int)
    v = vertices.T

    normals = np.zeros_like(v)
    for i0, i1, i2 in elements.T:
        v0, v1, v2 = v[i0], v[i1], v[i2]
        # -cross(...): revolve_profile winds triangles so this is the
        # outward normal (checked against the sphere case).
        raw_normal = -np.cross(v1 - v0, v2 - v0)  # magnitude = 2*area
        for i in (i0, i1, i2):
            normals[i] += raw_normal

    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals.T


def min_distance_to_profile(points_rz, profile_rz):
    """For each (r, z) in points_rz, the minimum Euclidean distance to the
    polyline through profile_rz. Used to numerically check that an offset
    profile really does sit a constant distance from the original."""
    points_rz = np.asarray(points_rz, dtype=float)
    profile_rz = np.asarray(profile_rz, dtype=float)

    a = profile_rz[:-1][None, :, :]
    b = profile_rz[1:][None, :, :]
    p = points_rz[:, None, :]

    ab = b - a
    ab_len_sq = np.sum(ab**2, axis=-1)
    t = np.sum((p - a) * ab, axis=-1) / ab_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    dist = np.linalg.norm(p - closest, axis=-1)
    return dist.min(axis=-1)
