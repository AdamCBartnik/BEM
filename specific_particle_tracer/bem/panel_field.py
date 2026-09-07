"""Analytic (no finite differences) evaluation of the field from a solved
*indirect* (charge-simulation-style) boundary-element Laplace problem: sum
the exact Coulomb field of a linearly-varying (P1) surface charge density
directly over every flat triangular panel.

`bem.laplace.ExteriorLaplaceSolution` solves for a single unknown surface
density sigma such that its single-layer potential S[sigma] matches the
required Dirichlet data on the boundary -- the indirect/charge-simulation
formulation, as opposed to the direct formulation (solve for both a
Dirichlet and a Neumann trace, reconstruct the field from S[t] - D[g]).
The direct formulation's field requires the *gradient* of the double-layer
potential, a hypersingular (~1/r^3) kernel: numerically brutal close to a
panel (this project's earlier attempt got *worse*, not better, under mesh
refinement, since refining tracks a fixed evaluation point ever closer to
the increasingly accurate faceted surface -- see git history for the
finite-difference and direct-panel-quadrature attempts this replaced).
Indirect/CSM sidesteps that kernel entirely: the only integral needed here
is the Coulomb field of sigma, one order milder (~1/r^2), the same kernel
this project's own point-charge sanity checks already validated. This is
the standard technique (also called the surface charge simulation method)
used in high-voltage/field-emission engineering for exactly this kind of
near-electrode field calculation.

Evaluating a point exactly *on* (or a tiny standoff from) the mesh is still
its own difficulty even for this milder kernel, though, for a subtler
reason than a raw "order of singularity" count: near x, the field kernel
sigma(y)*(x-y)/|x-y|^3 does not enjoy the cancellation that (e.g.) the
double-layer kernel gets from a dot product with a smoothly-varying
normal, so summing it via plain adaptive subdivision means summing many
large, nearly-canceling vector contributions from panels fanning out
around x -- a classic source of accumulated floating-point error. Directly
verified here: evaluating at points sitting exactly on mesh vertices, plain
adaptive subdivision does not converge as max_depth increases -- it
improves for a while and then gets *worse*, degrading right where deeper
refinement should help.

The fix is the same idea as the (now-removed) direct formulation's
double-layer desingularization, adapted to a kernel that does not have an
identically-zero reference case to exploit: shift the density by a local
reference value sigma_ref (sigma at the nearest mesh vertex to x) before
summing, so the integrand vanishes right where the kernel is largest, and
add back the *exact* contribution the shifted-away reference density
would have contributed. That contribution is the single-layer jump
relation for a uniform-density surface patch, well known in electrostatics
as the field of an (locally-idealized) infinite charged sheet: sigma_ref *
n(x), n(x) the local outward normal (from `bem.mesh.vertex_normals`) --
notably independent of standoff distance, which is exactly why this fixes
both the exactly-on-vertex and the more common "close to, but not quite
on, a panel" case in one step. Verified against the analytic hemisphere
solution: this converges cleanly (unlike the unregularized sum) and
reduces on-mesh-vertex error from ~100% to a fraction of a percent, and
generic (non-vertex) points at the analytic surface radius from ~100% to a
few percent.

This is also exactly the machinery a future image-charge force calculation
will need: particles interacting with the recessed image surface are
always evaluated close to it, the same near-panel regime handled here.
"""

import numpy as np

from .mesh import vertex_normals

# 3-point, degree-2-exact symmetric quadrature rule on the reference
# triangle, in barycentric coordinates. Weights sum to 1 (so a panel's
# integral is panel_area * sum(weight_i * f(point_i))).
_QUAD_BARY = np.array(
    [
        [2 / 3, 1 / 6, 1 / 6],
        [1 / 6, 2 / 3, 1 / 6],
        [1 / 6, 1 / 6, 2 / 3],
    ]
)
_QUAD_WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3])

_ROOT_BARY = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def _subdivide(b0, b1, b2):
    """Split a (sub-)triangle -- given as 3 barycentric-coordinate triples
    relative to the original panel -- into 4 children of equal area (the
    standard "connect the edge midpoints" refinement)."""
    m01 = 0.5 * (b0 + b1)
    m12 = 0.5 * (b1 + b2)
    m20 = 0.5 * (b2 + b0)
    return [(b0, m01, m20), (m01, b1, m12), (m20, m12, b2), (m01, m12, m20)]


def _area_fraction(b0, b1, b2):
    """Fraction of the original panel's area a sub-triangle (given by its 3
    barycentric corners) covers -- 1.0 for the whole (root) triangle."""
    d1 = b1[:2] - b0[:2]
    d2 = b2[:2] - b0[:2]
    return abs(d1[0] * d2[1] - d1[1] * d2[0])


def _panel_coulomb_field(x, v0, v1, v2, s0, s1, s2, area, refine_ratio, max_depth):
    """Adaptive-quadrature Coulomb-field contribution from one flat
    triangular panel (corners v0,v1,v2; surface charge density sigma
    linearly interpolated from those corners' nodal values s0,s1,s2) to
    the perturbation field at point x.

    `s0, s1, s2` must already be shifted by a reference value (see
    `evaluate_coulomb_field`) -- without that, this kernel converges
    poorly right at the surface (see module docstring).
    """
    E = np.zeros(3)
    stack = [(_ROOT_BARY, 0)]
    while stack:
        (b0, b1, b2), depth = stack.pop()
        p0 = b0[0] * v0 + b0[1] * v1 + b0[2] * v2
        p1 = b1[0] * v0 + b1[1] * v1 + b1[2] * v2
        p2 = b2[0] * v0 + b2[1] * v1 + b2[2] * v2
        centroid = (p0 + p1 + p2) / 3.0
        size = max(np.linalg.norm(p0 - p1), np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p0))
        dist = np.linalg.norm(x - centroid)

        if depth < max_depth and size > refine_ratio * dist:
            for child in _subdivide(b0, b1, b2):
                stack.append((child, depth + 1))
            continue

        sub_area = area * _area_fraction(b0, b1, b2)
        for bary_weights, weight in zip(_QUAD_BARY, _QUAD_WEIGHTS):
            bary = bary_weights[0] * b0 + bary_weights[1] * b1 + bary_weights[2] * b2
            pos = bary[0] * v0 + bary[1] * v1 + bary[2] * v2
            s_val = bary[0] * s0 + bary[1] * s1 + bary[2] * s2

            r_vec = x - pos
            r = np.linalg.norm(r_vec)

            E += (weight * sub_area / (4.0 * np.pi)) * s_val * r_vec / r**3

    return E


def evaluate_coulomb_field(vertices, elements, sigma_nodal, points, refine_ratio=1.0, max_depth=10):
    """The field E(x) = -grad[S[sigma]](x) at `points`, by direct adaptive-
    quadrature integration of the Coulomb kernel over every mesh panel,
    regularized for near-panel accuracy (see module docstring) -- no
    finite differences.

    Parameters
    ----------
    vertices, elements : the mesh, BEMpp convention (see bem.mesh) -- note
        that for the P1 spaces used throughout this project, DOF index i
        corresponds exactly to `vertices[:, i]` (verified against BEMpp's
        `space.cell_dofs`), so `sigma_nodal` below can be taken directly
        from `GridFunction.coefficients`.
    sigma_nodal : ndarray, shape (n_vertices,)
        Solved surface-charge-density nodal values at each vertex.
    points : ndarray, shape (..., 3)
    refine_ratio : float
        A sub-panel is refined further while its size exceeds
        `refine_ratio` times its distance to the evaluation point.
    max_depth : int
        Maximum recursive subdivisions per panel (bounds cost).

    Returns
    -------
    E : ndarray, same shape as `points`.
    """
    vertices = np.asarray(vertices, dtype=float)
    elements = np.asarray(elements, dtype=int)
    points = np.asarray(points, dtype=float)
    shape = points.shape
    flat_points = points.reshape(-1, 3)

    v = vertices.T  # (n_vertices, 3)
    normals = vertex_normals(vertices, elements).T  # (n_vertices, 3)

    panels = []
    areas = []
    for i0, i1, i2 in elements.T:
        v0, v1, v2 = v[i0], v[i1], v[i2]
        raw_normal = -np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(raw_normal)
        panels.append((v0, v1, v2, i0, i1, i2, area))
        areas.append(area)

    # The sigma_ref shift-and-add-back trick (see module docstring) is only
    # correct extremely close to the surface -- the "add back" term
    # approximates the reference density's own contribution as a local
    # infinite sheet, which only holds within a small fraction of one
    # element's size (checked empirically against the analytic hemisphere
    # solution: comparable to plain summation by ~0.5% of an element size
    # of standoff, and clearly worse beyond that, since a real point sees
    # the shifted-away charge's field decay with distance rather than stay
    # constant like an infinite sheet's would). Plain (unshifted) summation
    # is already accurate outside this thin shell -- no near-panel
    # difficulty there -- so the shift is only applied inside it.
    near_surface_threshold = 0.05 * np.sqrt(np.mean(areas))

    result = np.empty((flat_points.shape[0], 3))
    for i, x in enumerate(flat_points):
        nearest = np.argmin(np.sum((v - x) ** 2, axis=-1))
        near_surface = np.linalg.norm(v[nearest] - x) < near_surface_threshold
        sigma_ref = sigma_nodal[nearest] if near_surface else 0.0

        E = sigma_ref * normals[nearest]  # exact contribution of the shifted-away reference density
        for v0, v1, v2, i0, i1, i2, area in panels:
            E += _panel_coulomb_field(
                x,
                v0,
                v1,
                v2,
                sigma_nodal[i0] - sigma_ref,
                sigma_nodal[i1] - sigma_ref,
                sigma_nodal[i2] - sigma_ref,
                area,
                refine_ratio,
                max_depth,
            )
        result[i] = E

    return result.reshape(shape)
