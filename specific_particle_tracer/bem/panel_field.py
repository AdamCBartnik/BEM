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
reference value sigma_ref before summing, so the integrand vanishes right
where the kernel is largest, and add back the *exact* contribution the
shifted-away reference density would have contributed. That contribution
is the single-layer jump relation for a uniform-density surface patch,
well known in electrostatics as the field of an (locally-idealized)
infinite charged sheet: sigma_ref * n(x) -- notably independent of
standoff distance, which is exactly why this fixes both the exactly-on-
vertex and the more common "close to, but not quite on, a panel" case in
one step.

sigma_ref and n(x) themselves are taken at the point closest to x on the
*whole mesh* (found by an exact point-to-triangle projection against
every panel, then interpolated with that panel's own barycentric
coordinates) rather than at the nearest vertex -- using the nearest vertex
instead was tried first and left a stubborn, non-shrinking ~10-25% error
at the analytic surface that didn't converge cleanly under mesh
refinement (a vertex can sit a nontrivial fraction of an element size away
from x's true closest point, and the resulting mismatch in both sigma_ref
and the reference patch's own normal doesn't vanish as the mesh refines,
since -- as bem.fields' module docstring notes -- the analytic surface's
standoff from the mesh shrinks right along with the element size). Using
the true closest point and that panel's own exact (not vertex-averaged)
normal fixed this: verified against the analytic hemisphere solution,
error at the analytic surface now shrinks monotonically with mesh
resolution (why the vertex approach didn't converge cleanly should have
been the first clue it wasn't using enough local information).

This is also exactly the machinery a future image-charge force calculation
will need: particles interacting with the recessed image surface are
always evaluated close to it, the same near-panel regime handled here.
"""

import numpy as np

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


def _closest_point_on_triangle(p, a, b, c):
    """The point on flat triangle abc closest to p, and its barycentric
    coordinates there (u, v, w), u+v+w=1, closest = u*a + v*b + w*c.

    Standard region-based algorithm (Ericson, "Real-Time Collision
    Detection", ch. 5): identify which of the triangle's 7 Voronoi regions
    (3 vertices, 3 edges, the face) p projects into, and place the answer
    there directly -- exact, and cheap (a handful of dot products).
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, (1.0, 0.0, 0.0)

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, (0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, (1.0 - v, v, 0.0)

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, (0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, (1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), (0.0, 1.0 - w, w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w, (1.0 - v - w, v, w)


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

    panels = []
    areas = []
    for i0, i1, i2 in elements.T:
        v0, v1, v2 = v[i0], v[i1], v[i2]
        raw_normal = -np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(raw_normal)
        n = raw_normal / (2.0 * area)  # exact flat-panel normal, not a vertex average
        panels.append((v0, v1, v2, i0, i1, i2, area, n))
        areas.append(area)

    # The sigma_ref shift-and-add-back trick (see module docstring) is only
    # correct extremely close to the surface -- the "add back" term
    # approximates the reference density's own contribution as a local
    # infinite sheet, which only holds within a small fraction of one
    # element's size. Plain (unshifted) summation is already accurate
    # outside this thin shell -- no near-panel difficulty there -- so the
    # shift is only applied inside it.
    near_surface_threshold = 0.5 * np.sqrt(np.mean(areas))

    result = np.empty((flat_points.shape[0], 3))
    for i, x in enumerate(flat_points):
        best_dist2 = np.inf
        closest = None
        for v0, v1, v2, i0, i1, i2, area, n in panels:
            point, bary = _closest_point_on_triangle(x, v0, v1, v2)
            dist2 = np.sum((x - point) ** 2)
            if dist2 < best_dist2:
                best_dist2 = dist2
                closest = (bary, i0, i1, i2, n)

        bary, i0, i1, i2, n_local = closest
        near_surface = best_dist2 < near_surface_threshold**2
        sigma_ref = (
            bary[0] * sigma_nodal[i0] + bary[1] * sigma_nodal[i1] + bary[2] * sigma_nodal[i2]
            if near_surface
            else 0.0
        )

        E = sigma_ref * n_local  # exact contribution of the shifted-away reference density
        for v0, v1, v2, j0, j1, j2, area, n in panels:
            E += _panel_coulomb_field(
                x,
                v0,
                v1,
                v2,
                sigma_nodal[j0] - sigma_ref,
                sigma_nodal[j1] - sigma_ref,
                sigma_nodal[j2] - sigma_ref,
                area,
                refine_ratio,
                max_depth,
            )
        result[i] = E

    return result.reshape(shape)
