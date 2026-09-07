"""Analytic (no finite differences) evaluation of the field from a solved
boundary-element Laplace problem, by directly integrating the exact field
kernel -- Coulomb's law for the single-layer (charge) term, the dipole
field for the double-layer (potential) term -- over each flat triangular
panel and its linearly-varying (P1) nodal data.

This replaces getting the field by finite-differencing the potential:
differencing right next to or on a panel is a classic hard case for BEM
(the potential itself is only mesh/quadrature-accurate, and differencing
amplifies that error close to the surface). Integrating the field kernel
directly avoids that, at the cost of the field kernel's own, stronger
(1/r^2 rather than potential's 1/r) near-panel singularity -- handled here
with simple adaptive quadrature: a panel is recursively subdivided (in
barycentric coordinates, so the original nodal data always interpolates
correctly on every sub-panel, no re-fitting needed) until each piece is
either far enough from the evaluation point relative to its own size, or a
maximum recursion depth is hit. The depth cap means a point sitting right
on a panel gets a large but bounded answer (reflecting the mesh's own
resolution limit) rather than a literal singularity -- a reasonable,
physically sensible regularization, and a deliberately simple alternative
to deriving closed-form linearly-varying-panel formulas.

The double-layer term needs one more trick, and adaptive subdivision alone
isn't it: D[g](x) = integral of g(y) K(x,y) dA(y) over a *closed* surface
is exactly a spatially constant function of x throughout the whole
exterior (0, by the standard solid-angle identity) whenever g is uniform,
so its gradient (the double-layer's contribution to the field) must be
*exactly* zero there, not just small. Numerically this identity only
emerges from a delicate cancellation across the *entire* surface, and
per-panel quadrature (however refined near x) converges to it far too
slowly to be usable -- confirmed here by testing a uniform g directly,
which should give an exactly-zero field outside any closed surface but
came back with an order-1 (relative to the real Ez*R data this module
actually sees) spurious value even with deep per-panel refinement. The
standard fix is to subtract a constant reference value g_ref from g before
integrating: since the integral of K(x,y) dA(y) over the whole surface is
exactly 0 (exterior) regardless of what g_ref is, D[g](x) = D[g - g_ref](x)
identically, but the integrand on the right is now small exactly where the
kernel is largest (near x's closest surface point) if g_ref is g's value
there -- that's what actually fixes the convergence, not a change in the
true answer. `evaluate_panel_field` takes g_ref, per evaluation point, as
the Dirichlet value at the nearest mesh vertex.

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


def _panel_field(x, v0, v1, v2, g0, g1, g2, t0, t1, t2, normal, area, refine_ratio, max_depth):
    """Adaptive-quadrature field contribution from one flat triangular
    panel (corners v0,v1,v2; Dirichlet data g and Neumann data t linearly
    interpolated from those corners' nodal values) to the perturbation
    field at point x.

    Representation formula phi_pert(x) = S[t](x) - D[g](x) gives
    E_pert(x) = -grad phi_pert(x), i.e. a Coulomb-law contribution from t
    and a dipole-field contribution from g (dipole moment density along
    the panel's outward normal), both summed here directly.

    `g0, g1, g2` must already be shifted by a reference value (see
    `evaluate_panel_field`) -- the double-layer kernel's gradient, unlike
    the single-layer's, needs this to converge at any reasonable
    resolution (see the module docstring).
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
            g_val = bary[0] * g0 + bary[1] * g1 + bary[2] * g2
            t_val = bary[0] * t0 + bary[1] * t1 + bary[2] * t2

            r_vec = x - pos
            r = np.linalg.norm(r_vec)
            r_hat = r_vec / r

            e_single = t_val * r_vec / r**3
            e_double = g_val * (3.0 * np.dot(normal, r_hat) * r_hat - normal) / r**3

            E += (weight * sub_area / (4.0 * np.pi)) * (e_single + e_double)

    return E


def evaluate_panel_field(vertices, elements, g_nodal, t_nodal, points, refine_ratio=1.0, max_depth=6):
    """The perturbation field E_pert(x) = -grad[S[t] - D[g]](x) at `points`,
    by direct adaptive-quadrature integration of the field kernels over
    every mesh panel (see module docstring) -- no finite differences.

    Parameters
    ----------
    vertices, elements : the mesh, BEMpp convention (see bem.mesh) -- note
        that for the P1 spaces used throughout this project, DOF index i
        corresponds exactly to `vertices[:, i]` (verified against BEMpp's
        `space.cell_dofs`), so `g_nodal`/`t_nodal` below can be taken
        directly from `GridFunction.coefficients`.
    g_nodal, t_nodal : ndarray, shape (n_vertices,)
        Dirichlet and Neumann nodal values at each vertex.
    points : ndarray, shape (..., 3)
    refine_ratio : float
        A sub-panel is refined further while its size exceeds
        `refine_ratio` times its distance to the evaluation point.
    max_depth : int
        Maximum recursive subdivisions per panel (bounds cost, and bounds
        how large the answer can get for a point sitting on or very near a
        panel).

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
    for i0, i1, i2 in elements.T:
        v0, v1, v2 = v[i0], v[i1], v[i2]
        # -cross(...): this project's revolve_profile winds triangles so
        # that -cross(v1-v0, v2-v0) is the outward normal (checked against
        # the sphere case, where "outward" unambiguously means away from
        # the center).
        raw_normal = -np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(raw_normal)
        normal = raw_normal / (2.0 * area)
        panels.append(
            (v0, v1, v2, g_nodal[i0], g_nodal[i1], g_nodal[i2], t_nodal[i0], t_nodal[i1], t_nodal[i2], normal, area)
        )

    result = np.empty((flat_points.shape[0], 3))
    for i, x in enumerate(flat_points):
        # Desingularize the double-layer term (see module docstring): shift
        # g by its value at the mesh vertex nearest x, which leaves the
        # true field unchanged but makes the near-panel integrand small
        # exactly where the kernel is largest.
        g_ref = g_nodal[np.argmin(np.sum((v - x) ** 2, axis=-1))]

        E = np.zeros(3)
        for v0, v1, v2, g0, g1, g2, t0, t1, t2, normal, area in panels:
            E += _panel_field(
                x, v0, v1, v2, g0 - g_ref, g1 - g_ref, g2 - g_ref, t0, t1, t2, normal, area, refine_ratio, max_depth
            )
        result[i] = E

    return result.reshape(shape)
