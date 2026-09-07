"""Axisymmetric (body-of-revolution) charge-simulation BEM solver.

This project's geometries so far (a hemispherical tip on an infinite flat
cathode, in a uniform field along the symmetry axis) are all axisymmetric:
the Dirichlet data g = Ez*z has no azimuthal (phi) dependence, so neither
does the solved surface charge density sigma. That means the azimuthal
integral in the 3D single-layer potential/field kernels can be done
*analytically* (via complete elliptic integrals) instead of numerically,
collapsing the problem from a full 2D surface mesh (bem.mesh, bem.laplace,
bem.panel_field) to a 1D mesh of the generating profile alone -- typically
tens of unknowns here instead of thousands, and (since the phi integral is
now exact rather than a discretized sum over n_phi panels) a much smoother
field as a function of position, which matters for an adaptive ODE
integrator stepping through it.

This only works because the excitation is axisymmetric; a shape or applied
field without that symmetry still needs the general bem.mesh/bem.laplace
machinery.

Physics
-------
The potential of a unit-total-charge ring of radius a (at height 0, field
point at cylindrical (rho, z)) is, in the G = 1/(4*pi*r) convention used
throughout this project's BEM work:

    V_ring(rho, z, a) = K(m) / (pi * sqrt(D)),   D = (rho+a)^2 + z^2,
                         m = 4*a*rho / D

with K the complete elliptic integral of the first kind (scipy's
`ellipk(m)` takes the parameter m = k^2, not the modulus k -- easy to get
backwards). This comes from the standard identity
integral_0^2pi dphi / sqrt(A - B*cos(phi)) = 4/sqrt(A+B) * K(2B/(A+B)).
Verified here against brute-force numerical azimuthal integration to
~1e-16 relative error.

The field (E = -grad V) follows by differentiating through K's and E's
(the complete elliptic integral of the second kind) dependence on m:
dK/dm = (E(m) - (1-m)*K(m)) / (2*m*(1-m)), dE/dm = (E(m) - K(m)) / (2*m),
with the m -> 0 (on-axis) limit of dK/dm handled as a removable
singularity (pi/8, from K's power series) since the general formula's
0/0 there is otherwise numerically fragile. Verified against central
finite differences of the (already-verified) potential, off axis and
on it; catching a pi/16-vs-pi/8 slip in that on-axis constant this way
is exactly why this project validates every closed form against an
independent check before trusting it.

For a surface with axisymmetric density sigma(s) (s = arc length along
the generating profile), a profile element contributes an effective ring
charge dQ = sigma(s) * 2*pi*rho(s) * ds -- so both the boundary integral
equation and the field evaluation are 1D integrals of sigma(s) * rho(s) *
ring_potential(...)/ring_field(...) along the profile, with sigma
represented piecewise-linearly (P1) between profile nodes.

As with bem.panel_field's Coulomb kernel, the *field* kernel here doesn't
enjoy the cancellation that keeps the *potential* kernel's self term
(a log singularity, confirmed by direct testing to converge cleanly under
adaptive subdivision) well-behaved: near the profile, it's the same
"summing large, nearly-canceling contributions" problem, confirmed here
too (adaptive subdivision improves then degrades, not unlike the 3D case).
Fixed the same way: shift the density by its value at the true closest
point on the profile (found by exact point-to-segment projection, not
just the nearest node) before summing, and add back that reference
density's exact local-infinite-sheet contribution, sigma_ref * n(x).
"""

import numpy as np
from scipy import special


def ring_potential(rho, z, a):
    """Potential at cylindrical (rho, z) of a unit-total-charge ring of
    radius a centered on the z-axis at height 0 (G = 1/(4*pi*r) convention).

    Note the 1/(2*pi) here beyond the "naive" azimuthal integral of the
    point kernel (K(m)/(pi*sqrt(D))): that naive integral is the potential
    of a ring where each unit of *angle* carries one unit of charge (total
    charge 2*pi, not 1) -- an easy mismatch to miss since the naive form
    still validates perfectly against brute-force azimuthal integration
    (they're computing the same, differently-normalized, thing). Caught
    only by the physical unit-charge sanity check (a uniformly charged
    sphere's field just outside must equal its own surface charge density,
    by the shell theorem) -- the point-by-point elliptic-integral checks
    below don't catch a missing overall constant like this one.
    """
    D = (rho + a) ** 2 + z**2
    m = np.clip(4.0 * a * rho / D, 0.0, 1.0 - 1e-15)
    return special.ellipk(m) / (2.0 * np.pi**2 * np.sqrt(D))


def ring_field(rho, z, a):
    """Field (E_rho, E_z) at cylindrical (rho, z) of a unit-total-charge
    ring of radius a centered on the z-axis at height 0."""
    D = (rho + a) ** 2 + z**2
    m = np.clip(4.0 * a * rho / D, 0.0, 1.0 - 1e-15)
    K = special.ellipk(m)
    E = special.ellipe(m)
    sqrtD = np.sqrt(D)

    dD_drho = 2.0 * (rho + a)
    dD_dz = 2.0 * z
    dm_drho = (4.0 * a * D - 4.0 * a * rho * dD_drho) / D**2
    dm_dz = (-4.0 * a * rho * dD_dz) / D**2

    dK_dm = np.pi / 8.0 if m < 1e-9 else (E - (1.0 - m) * K) / (2.0 * m * (1.0 - m))

    scale = 2.0 * np.pi**2  # matches ring_potential's normalization
    dV_drho = (dK_dm * dm_drho) / (scale * sqrtD) - K * dD_drho / (2.0 * scale * D**1.5)
    dV_dz = (dK_dm * dm_dz) / (scale * sqrtD) - K * dD_dz / (2.0 * scale * D**1.5)
    return -dV_drho, -dV_dz


def _closest_point_on_segment(p, a, b):
    """Closest point on segment ab (2D) to point p, and the interpolation
    parameter t in [0, 1] with closest = a + t*(b-a)."""
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    t = min(1.0, max(0.0, t))
    return a + t * ab, t


_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(4)


def _segment_potential(rho_f, z_f, p0, p1, s0, s1, refine_ratio, max_depth, depth=0):
    """Adaptive-quadrature potential contribution from one profile segment
    (endpoints p0, p1 = (rho, z); density linearly interpolated between
    s0, s1) to the field point (rho_f, z_f). The self-term (field point on
    the segment itself) is only a log singularity here -- verified to
    converge cleanly under this same recursive bisection, unlike the field
    kernel below."""
    mid = 0.5 * (p0 + p1)
    length = np.linalg.norm(p1 - p0)
    dist = np.linalg.norm(np.array([rho_f, z_f]) - mid)

    if depth < max_depth and length > refine_ratio * dist:
        pm = 0.5 * (p0 + p1)
        sm = 0.5 * (s0 + s1)
        return _segment_potential(rho_f, z_f, p0, pm, s0, sm, refine_ratio, max_depth, depth + 1) + _segment_potential(
            rho_f, z_f, pm, p1, sm, s1, refine_ratio, max_depth, depth + 1
        )

    total = 0.0
    for xi, wi in zip(_GAUSS_X, _GAUSS_W):
        t = 0.5 * (xi + 1.0)
        rho_s = p0[0] + t * (p1[0] - p0[0])
        z_s = p0[1] + t * (p1[1] - p0[1])
        sigma_s = s0 + t * (s1 - s0)
        seg_len = length * 0.5 * wi
        total += sigma_s * 2.0 * np.pi * rho_s * ring_potential(rho_f, z_f - z_s, rho_s) * seg_len
    return total


def _segment_field(rho_f, z_f, p0, p1, s0, s1, refine_ratio, max_depth, depth=0):
    """Adaptive-quadrature field contribution from one profile segment, as
    `_segment_potential` but returning (E_rho, E_z). `s0, s1` must already
    be shifted by a reference density for near-profile accuracy -- see
    `evaluate_axisymmetric_field`."""
    mid = 0.5 * (p0 + p1)
    length = np.linalg.norm(p1 - p0)
    dist = np.linalg.norm(np.array([rho_f, z_f]) - mid)

    if depth < max_depth and length > refine_ratio * dist:
        pm = 0.5 * (p0 + p1)
        sm = 0.5 * (s0 + s1)
        a = _segment_field(rho_f, z_f, p0, pm, s0, sm, refine_ratio, max_depth, depth + 1)
        b = _segment_field(rho_f, z_f, pm, p1, sm, s1, refine_ratio, max_depth, depth + 1)
        return a[0] + b[0], a[1] + b[1]

    e_rho_total = 0.0
    e_z_total = 0.0
    for xi, wi in zip(_GAUSS_X, _GAUSS_W):
        t = 0.5 * (xi + 1.0)
        rho_s = p0[0] + t * (p1[0] - p0[0])
        z_s = p0[1] + t * (p1[1] - p0[1])
        sigma_s = s0 + t * (s1 - s0)
        seg_len = length * 0.5 * wi
        e_rho, e_z = ring_field(rho_f, z_f - z_s, rho_s)
        weight = sigma_s * 2.0 * np.pi * rho_s * seg_len
        e_rho_total += weight * e_rho
        e_z_total += weight * e_z
    return e_rho_total, e_z_total


class AxisymmetricBEMSolution:
    """The solution (profile + solved nodal surface-charge density) of an
    axisymmetric exterior Laplace Dirichlet problem, via the indirect/
    charge-simulation method (see module docstring): a single unknown
    density sigma(s), piecewise-linear along the generating profile,
    solved by collocation so its potential matches given Dirichlet data at
    every profile node.

    Build with `AxisymmetricBEMSolution.solve(profile, dirichlet_fn)`.
    """

    def __init__(self, profile, sigma):
        self.profile = profile
        self.sigma = sigma

    @classmethod
    def solve(cls, profile, dirichlet_fn, refine_ratio=1.0, max_depth=20):
        """Solve for the nodal surface charge density by point collocation:
        require the solved density's potential to match `dirichlet_fn` at
        every profile node exactly.

        Parameters
        ----------
        profile : ndarray, shape (n, 2)
            Generating profile as (rho, z) points, in order along the
            surface (e.g. `bem.mesh.sphere_profile`, reshaped to columns).
        dirichlet_fn : callable(rho, z) -> float
            The Dirichlet (potential) data on the surface.
        refine_ratio, max_depth : passed to the per-segment adaptive
            quadrature (see `_segment_potential`).
        """
        profile = np.asarray(profile, dtype=float)
        n = len(profile)
        A = np.zeros((n, n))
        b = np.zeros(n)

        for i in range(n):
            rho_f, z_f = profile[i]
            b[i] = dirichlet_fn(rho_f, z_f)
            for j in range(n - 1):
                p0, p1 = profile[j], profile[j + 1]
                A[i, j] += _segment_potential(rho_f, z_f, p0, p1, 1.0, 0.0, refine_ratio, max_depth)
                A[i, j + 1] += _segment_potential(rho_f, z_f, p0, p1, 0.0, 1.0, refine_ratio, max_depth)

        sigma = np.linalg.solve(A, b)
        return cls(profile, sigma)

    def potential(self, points):
        """Potential at `points`, shape (..., 3). Returns shape points.shape[:-1]."""
        points = np.asarray(points, dtype=float)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        rho = np.linalg.norm(flat[:, :2], axis=-1)
        z = flat[:, 2]

        result = np.empty(flat.shape[0])
        for k in range(flat.shape[0]):
            total = 0.0
            for j in range(len(self.profile) - 1):
                p0, p1 = self.profile[j], self.profile[j + 1]
                total += _segment_potential(rho[k], z[k], p0, p1, self.sigma[j], self.sigma[j + 1], 1.0, 20)
            result[k] = total
        return result.reshape(shape)

    def field(self, points, refine_ratio=1.0, max_depth=20):
        """Field E = -grad(potential) at `points`, shape (..., 3), via the
        same near-profile regularization as `bem.panel_field`: shift the
        density by its value at the true closest point on the profile
        (not just the nearest node) before summing, and add back that
        reference density's exact local jump contribution.

        Returns
        -------
        E : ndarray, same shape as `points`.
        """
        return evaluate_axisymmetric_field(self.profile, self.sigma, points, refine_ratio=refine_ratio, max_depth=max_depth)


def evaluate_axisymmetric_field(profile, sigma, points, refine_ratio=1.0, max_depth=20):
    """The field E(x) = -grad[S[sigma]](x) at `points`, for an axisymmetric
    surface-charge density sigma given at the nodes of a generating
    profile -- see module docstring.

    Parameters
    ----------
    profile : ndarray, shape (n, 2)
        Generating profile as (rho, z) points, in order along the surface.
    sigma : ndarray, shape (n,)
        Nodal surface-charge-density values.
    points : ndarray, shape (..., 3)
    refine_ratio, max_depth : per-segment adaptive-quadrature parameters.

    Returns
    -------
    E : ndarray, same shape as `points`.
    """
    profile = np.asarray(profile, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    points = np.asarray(points, dtype=float)
    shape = points.shape
    flat = points.reshape(-1, 3)

    x, y, z = flat[:, 0], flat[:, 1], flat[:, 2]
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)

    # 0.1x (not, say, 0.5x): checked empirically against the analytic
    # hemisphere solution -- much beyond this the regularized branch is
    # *worse* than plain summation would be at that same distance (its
    # "local reference density" approximation is only good very close in),
    # so a generous threshold actively hurts otherwise-accurate points
    # near, but not that near, the surface.
    segment_lengths = np.linalg.norm(profile[1:] - profile[:-1], axis=-1)
    near_surface_threshold = 0.1 * np.sqrt(np.mean(segment_lengths**2))

    result = np.empty((flat.shape[0], 3))
    for k in range(flat.shape[0]):
        p = np.array([rho[k], z[k]])

        best_dist2 = np.inf
        best = None
        for j in range(len(profile) - 1):
            a, b = profile[j], profile[j + 1]
            cp, t = _closest_point_on_segment(p, a, b)
            d2 = np.sum((p - cp) ** 2)
            if d2 < best_dist2:
                best_dist2 = d2
                tangent = b - a
                normal_2d = np.array([-tangent[1], tangent[0]])
                normal_2d /= np.linalg.norm(normal_2d)
                best = (j, t, normal_2d)

        j, t, normal_2d = best
        near_surface = best_dist2 < near_surface_threshold**2
        sigma_ref = (sigma[j] + t * (sigma[j + 1] - sigma[j])) if near_surface else 0.0

        e_rho = sigma_ref * normal_2d[0]
        e_z = sigma_ref * normal_2d[1]
        for jj in range(len(profile) - 1):
            p0, p1 = profile[jj], profile[jj + 1]
            er, ez = _segment_field(
                p[0], p[1], p0, p1, sigma[jj] - sigma_ref, sigma[jj + 1] - sigma_ref, refine_ratio, max_depth
            )
            e_rho += er
            e_z += ez

        if rho[k] > 1e-15:
            result[k] = [e_rho * np.cos(phi[k]), e_rho * np.sin(phi[k]), e_z]
        else:
            result[k] = [0.0, 0.0, e_z]

    return result.reshape(shape)
