"""Axisymmetric (body-of-revolution) charge-simulation BEM solver.

This project's geometries so far (a hemispherical tip on an infinite flat
cathode, in a uniform field along the symmetry axis) are all axisymmetric:
the Dirichlet data g = Ez*z has no azimuthal (phi) dependence, so neither
does the solved surface charge density sigma. That means the azimuthal
integral in the 3D single-layer potential/field kernels can be done
*analytically* (via complete elliptic integrals) instead of numerically,
collapsing the problem from a full 2D surface mesh to a 1D mesh of the
generating profile alone -- typically tens of unknowns here instead of
thousands, and (since the phi integral is now exact rather than a
discretized sum over azimuthal panels) a much smoother field as a
function of position, which matters for an adaptive ODE integrator
stepping through it.

This only works because the excitation is axisymmetric, not merely because
the geometry is: a uniform field along the symmetry axis has Dirichlet
data with no phi-dependence, so the induced charge has none either. That
holds for any future axisymmetric real-cathode shape sitting in an
on-axis field, not just this hemisphere-tip (or cylindrical-well) case. It
does *not* hold for the image-charge job, even on the (still axisymmetric)
recessed surface: the excitation there is a real particle at some
generally off-axis 3D position, which breaks the rotational symmetry of
the induced image charge regardless of the surface's own symmetry -- see
`bem.image_charge`/`bem.toroidal` for the azimuthal-Fourier-mode
decomposition (one 1D problem per mode) that job actually uses instead.
An earlier general 3D triangulated-mesh path (via bempp-cl) was tried
first and dropped -- see git history -- once the 1D reductions above
covered every geometry this project actually needed.

Physics
-------
The potential of a unit-total-charge ring of radius a (at height 0, field
point at cylindrical (rho, z)) is, in the G = 1/(4*pi*r) convention used
throughout this project's BEM work:

    V_ring(rho, z, a) = K(m) / (2*pi^2 * sqrt(D)),   D = (rho+a)^2 + z^2,
                         m = 4*a*rho / D

with K the complete elliptic integral of the first kind. This comes from
the standard identity integral_0^2pi dphi / sqrt(A - B*cos(phi)) =
4/sqrt(A+B) * K(2B/(A+B)) -- the extra 1/(2*pi) beyond that identity's
naive K(m)/(pi*sqrt(D)) is because the naive azimuthal integral is the
potential of a ring where each unit of *angle* carries one unit of charge
(total charge 2*pi, not 1). That distinction still validates perfectly
against brute-force azimuthal integration of the naive form (both sides
compute the same, differently-normalized, thing) -- it was only caught by
a physical identity (a uniformly charged sphere's field just outside must
equal its own surface charge density, the shell theorem), not by that
numerical check or by the field-vs-potential consistency check below.

The field (E = -grad V) follows by differentiating through K's and E's
(the complete elliptic integral of the second kind) dependence on m:
dK/dm = (E(m) - (1-m)*K(m)) / (2*m*(1-m)), with the m -> 0 (on-axis) limit
handled as a removable singularity (pi/8, from K's power series) since the
general formula's 0/0 there is otherwise numerically fragile. Verified
against central finite differences of the (already-verified) potential,
off axis and on it.

K(m) and E(m) themselves come from scipy.special.ellipk/ellipe on numpy,
and from cupyx.scipy.special on cupy -- which provides ellipk directly,
but (as of this writing) not ellipe. E(m) there instead comes from the
*incomplete* elliptic integral of the second kind at its upper limit
pi/2, ellipeinc(pi/2, m), which is exactly the complete integral by
definition (confirmed to match scipy.special.ellipe to floating-point
roundoff, not merely approximately) -- see `_ellip_ke`.

For a surface with axisymmetric density sigma(s) (s = arc length along
the generating profile), a profile element contributes an effective ring
charge dQ = sigma(s) * 2*pi*rho(s) * ds -- so both the boundary integral
equation and the field evaluation are 1D integrals of sigma(s) * rho(s) *
ring_potential(...)/ring_field(...) along the profile, with sigma
represented piecewise-linearly (P1) between profile nodes.

Near-surface field evaluation
-----------------------------
Potential assembly uses adaptive segment quadrature. Field evaluation uses
cached fixed quadrature for distant segments and a sinh change of variables
for nearby ones, resolving the actual single-layer integral on the actual
surface. No global constant density is removed or replaced by a planar
field. Exactly-on-surface queries use an exterior displacement proportional
to the local element length; polygon corners have no unique finite trace
for arbitrary charge density. Refining the profile controls that geometric
approximation. Both quadrature paths are vectorized on NumPy and CuPy.

"""

import numpy as np

from ..backend import array_cache_key

from ._elliptic import ellip_ke as _ellip_ke, ellip_ke_complement

_NEAR_GAUSS_X, _NEAR_GAUSS_W = np.polynomial.legendre.leggauss(8)

# 3-point (not the solve's 4-point) fixed Gauss-Legendre rule used per
# quadrature sub-panel in the cached field-evaluation quadrature.
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(4)

_CLOSEST_POINT_KERNEL_SOURCE = r"""
extern "C" __global__
void closest_point_on_profile(
    const double* rho, const double* z,
    const double* p0x, const double* p0y, const double* abx, const double* aby,
    const double* nx, const double* ny,
    int n_seg, long long n_points,
    double* best_dist2, long long* best_seg, double* best_t, double* best_nx, double* best_ny)
{
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_points) return;

    double rq = rho[idx], zq = z[idx];
    double bd2 = 1.0e300;
    long long bseg = 0;
    double bt = 0.0, bnx = 0.0, bny = 0.0;

    for (int j = 0; j < n_seg; j++) {
        double ax = abx[j], ay = aby[j];
        double ab_dot_ab = ax * ax + ay * ay;
        double t = ((rq - p0x[j]) * ax + (zq - p0y[j]) * ay) / ab_dot_ab;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double crho = p0x[j] + t * ax;
        double cz = p0y[j] + t * ay;
        double drho = rq - crho, dz = zq - cz;
        double d2 = drho * drho + dz * dz;
        if (d2 < bd2) {
            bd2 = d2;
            bseg = j;
            bt = t;
            bnx = nx[j];
            bny = ny[j];
        }
    }
    best_dist2[idx] = bd2;
    best_seg[idx] = bseg;
    best_t[idx] = bt;
    best_nx[idx] = bnx;
    best_ny[idx] = bny;
}
"""

_closest_point_kernel = None


def _get_closest_point_kernel():
    global _closest_point_kernel
    if _closest_point_kernel is None:
        import cupy as cp

        _closest_point_kernel = cp.RawKernel(_CLOSEST_POINT_KERNEL_SOURCE, "closest_point_on_profile")
    return _closest_point_kernel


def _prepare_closest_point_gpu(p0, tangent, normal):
    import cupy as cp
    # RawKernel pointers carry no strides. Make the six columns contiguous
    # once, then keep them on the device for subsequent queries.
    return tuple(cp.ascontiguousarray(cp.asarray(a, dtype=cp.float64)[:, i])
                 for a in (p0, tangent, normal) for i in range(2))


def _closest_point_on_profile_gpu(rho, z, p0, p1, tangent, normal, _prepared=None):
    """Same contract as `_closest_point_on_profile`, but as a single
    cupy.RawKernel launch: one CUDA thread per query point, looping over
    all n_segments profile segments in device code, instead of the
    Python-loop version's O(n_segments) rounds of tiny elementwise kernel
    launches (clip, compare, `xp.where` x3) over the whole query-point
    array each time -- measured to dominate this function's GPU cost even
    at query-point counts as small as 10 (the per-segment loop count, not
    the query-point count, drives the number of kernel launches). No
    upper bound on n_segments here (unlike bem.toroidal's RawKernel, which
    needs a fixed-size per-thread local array for its mode count): each
    thread just reads the (small, read-only, effectively L1/L2-cached)
    profile-geometry arrays directly, no local storage sized by n_seg."""
    import cupy as cp

    n_points = rho.shape[0]
    n_seg = len(p0)
    p0x, p0y, abx, aby, nx, ny = (
        _prepare_closest_point_gpu(p0, tangent, normal) if _prepared is None else _prepared
    )
    rho_c = cp.ascontiguousarray(rho.astype(cp.float64, copy=False))
    z_c = cp.ascontiguousarray(z.astype(cp.float64, copy=False))

    best_dist2 = cp.empty(n_points, dtype=cp.float64)
    best_seg = cp.empty(n_points, dtype=cp.int64)
    best_t = cp.empty(n_points, dtype=cp.float64)
    best_nx = cp.empty(n_points, dtype=cp.float64)
    best_ny = cp.empty(n_points, dtype=cp.float64)

    if n_points > 0:
        threads = 256
        blocks = (n_points + threads - 1) // threads
        kernel = _get_closest_point_kernel()
        kernel(
            (blocks,),
            (threads,),
            (
                rho_c,
                z_c,
                p0x,
                p0y,
                abx,
                aby,
                nx,
                ny,
                np.int32(n_seg),
                np.int64(n_points),
                best_dist2,
                best_seg,
                best_t,
                best_nx,
                best_ny,
            ),
        )
    best_normal = cp.stack([best_nx, best_ny], axis=-1)
    return best_dist2, best_seg, best_t, best_normal


def ring_potential(rho, z, a, xp=np):
    """Potential at cylindrical (rho, z) of a unit-total-charge ring of
    radius a centered on the z-axis at height 0 (G = 1/(4*pi*r)
    convention) -- see module docstring. Arguments broadcast as ordinary
    array arithmetic; `xp` is numpy or cupy."""
    D = (rho + a) ** 2 + z**2
    p = ((rho - a)**2 + z**2) / D
    K, _ = ellip_ke_complement(p, xp)
    return K / (2.0 * xp.pi**2 * xp.sqrt(D))


_ring_field_fused = None


def ring_field(rho, z, a, xp=np):
    """Evaluate the same ring field with fused elementwise work on CUDA."""
    if xp is np:
        return _ring_field_array(rho, z, a, xp)
    global _ring_field_fused
    if _ring_field_fused is None:
        import cupy as cp
        _ring_field_fused = cp.fuse(kernel_name="bem_ring_field")(
            lambda rho, z, a: _ring_field_array(rho, z, a, cp)
        )
    return _ring_field_fused(rho, z, a)


def _ring_field_array(rho, z, a, xp):
    """Field of a unit-charge ring, with a stable near-ring complement.

    The small distance squared is computed directly, not as 1-m after
    rounding m almost to one. The axis uses the first terms of its regular
    Taylor expansion to avoid cancellation in the radial component.
    """
    D = (rho + a)**2 + z**2
    d2 = (rho - a)**2 + z**2
    K, E = ellip_ke_complement(d2 / D, xp)
    near_axis = rho**2 < 1e-10 * (a*a + z*z)
    rho_safe = xp.where(near_axis, 1.0, rho)
    er = (K - (a*a - rho*rho + z*z) * E / d2) / (4*xp.pi**2*rho_safe*xp.sqrt(D))
    ez = z * E / (2*xp.pi**2*xp.sqrt(D)*d2)
    axis_d2 = a*a + z*z
    er_axis = rho * (2*z*z - a*a) / (8*xp.pi*axis_d2**2.5)
    ez_axis = z/(4*xp.pi*axis_d2**1.5) + 3*rho*rho*z*(3*a*a-2*z*z)/(16*xp.pi*axis_d2**3.5)
    return xp.where(near_axis, er_axis, er), xp.where(near_axis, ez_axis, ez)


def _subdivide(b0, b1, b2):
    m01 = 0.5 * (b0 + b1)
    m12 = 0.5 * (b1 + b2)
    m20 = 0.5 * (b2 + b0)
    return [(b0, m01, m20), (m01, b1, m12), (m20, m12, b2), (m01, m12, m20)]


def _segment_potential(rho_f, z_f, p0, p1, s0, s1, refine_ratio, max_depth, depth=0):
    """Adaptive-quadrature potential contribution from one profile segment
    (endpoints p0, p1 = (rho, z); density linearly interpolated between
    s0, s1) to the field point (rho_f, z_f). Scalar/numpy only -- used by
    the (one-time) BEM solve, not by field evaluation (see module
    docstring's Vectorization section). The self-term (field point on the
    segment itself) is only a log singularity here, confirmed to converge
    cleanly under this recursive bisection."""
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


def _fixed_quadrature(profile, sigma, n_subdiv):
    """Fixed (non-adaptive) quadrature nodes over the whole profile, as
    flat numpy arrays (rho_q, z_q, area_q, sigma_q) -- one-time, cheap (a
    few hundred nodes for a typical profile), meant to be built once and
    reused across many field-evaluation calls."""
    rho_q, z_q, area_q, sigma_q = [], [], [], []
    for j in range(len(profile) - 1):
        p0, p1 = profile[j], profile[j + 1]
        s0, s1 = sigma[j], sigma[j + 1]
        length = np.linalg.norm(p1 - p0)
        for k in range(n_subdiv):
            ta, tb = k / n_subdiv, (k + 1) / n_subdiv
            for xi, wi in zip(_GAUSS_X, _GAUSS_W):
                t = ta + 0.5 * (tb - ta) * (xi + 1.0)
                rho_s = p0[0] + t * (p1[0] - p0[0])
                z_s = p0[1] + t * (p1[1] - p0[1])
                seg_len = (length / n_subdiv) * 0.5 * wi
                rho_q.append(rho_s)
                z_q.append(z_s)
                area_q.append(2.0 * np.pi * rho_s * seg_len)
                sigma_q.append(s0 + t * (s1 - s0))
    return np.array(rho_q), np.array(z_q), np.array(area_q), np.array(sigma_q)


def _segment_endpoints(profile):
    """Per-segment (p0, p1, tangent, outward normal) as numpy arrays,
    shape (n_segments, 2) each -- precomputed once, used by the vectorized
    closest-point search."""
    p0 = profile[:-1]
    p1 = profile[1:]
    tangent = p1 - p0
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    return p0, p1, tangent, normal


def _closest_point_on_profile(rho, z, p0, p1, tangent, normal, xp, _prepared=None):
    """Vectorized closest point on the whole profile to arrays of (rho, z)
    field points: loops over the (few tens of) profile segments, fully
    vectorized over the (possibly many) field points within each
    iteration -- so cost is O(n_segments) Python-level iterations
    regardless of how many field points there are. On GPU this Python-loop
    cost (each iteration launching several tiny elementwise kernels) is
    what actually dominates at typical (tens) query-point counts -- see
    `_closest_point_on_profile_gpu`'s docstring -- so this dispatches to
    that fused single-kernel version whenever xp is cupy.

    Parameters
    ----------
    rho, z : arrays, shape (M,)
    p0, p1, tangent, normal : arrays, shape (n_segments, 2) -- numpy (they
        describe the fixed profile geometry, tiny, converted to `xp` once
        by the caller if needed).
    xp : array module.

    Returns
    -------
    best_dist2, best_seg, best_t : arrays, shape (M,)
    best_normal : array, shape (M, 2)
    """
    if xp is not np:
        return _closest_point_on_profile_gpu(rho, z, p0, p1, tangent, normal, _prepared)

    M = rho.shape[0]
    best_dist2 = xp.full(M, xp.inf)
    best_seg = xp.zeros(M, dtype=xp.int64)
    best_t = xp.zeros(M)
    best_normal = xp.zeros((M, 2))

    for j in range(len(p0)):
        ab = tangent[j]
        ab_dot_ab = float(ab[0] * ab[0] + ab[1] * ab[1])
        t = ((rho - p0[j, 0]) * ab[0] + (z - p0[j, 1]) * ab[1]) / ab_dot_ab
        t = xp.clip(t, 0.0, 1.0)
        closest_rho = p0[j, 0] + t * ab[0]
        closest_z = p0[j, 1] + t * ab[1]
        dist2 = (rho - closest_rho) ** 2 + (z - closest_z) ** 2

        better = dist2 < best_dist2
        best_dist2 = xp.where(better, dist2, best_dist2)
        best_seg = xp.where(better, j, best_seg)
        best_t = xp.where(better, t, best_t)
        best_normal = xp.where(better[:, None], xp.asarray(normal[j])[None, :], best_normal)

    return best_dist2, best_seg, best_t, best_normal


class AxisymmetricBEMSolution:
    """The solution (profile + solved nodal surface-charge density) of an
    axisymmetric exterior Laplace Dirichlet problem, via the indirect/
    charge-simulation method (see module docstring): a single unknown
    density sigma(s), piecewise-linear along the generating profile,
    solved by collocation so its potential matches given Dirichlet data at
    every profile node.

    Build with `AxisymmetricBEMSolution.solve(profile, dirichlet_fn)`.
    Treat the profile and solved density as fixed: quadrature and backend
    arrays are cached. Construct a new solution when either changes.
    """

    def __init__(self, profile, sigma):
        self.profile = profile
        self.sigma = sigma
        self._quadrature_cache = {}
        self._field_cache = {}

    @classmethod
    def solve(cls, profile, dirichlet_fn, refine_ratio=1.0, max_depth=20):
        """Solve for the nodal surface charge density by point collocation:
        require the solved density's potential to match `dirichlet_fn` at
        every profile node exactly. Always numpy/CPU (see module
        docstring) -- a one-time cost regardless of what backend
        subsequent field evaluations use.

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
        """Potential at `points`, shape (..., 3). Returns shape points.shape[:-1].
        Numpy only, via the adaptive per-segment quadrature -- a reference/
        validation path, not the one the tracker uses for the field."""
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

    def quadrature(self, n_subdiv=8):
        """Cached fixed quadrature (see `_fixed_quadrature`) for this
        solution's profile/sigma, as numpy arrays. Field evaluation
        converts these to whatever backend it needs once and caches that
        too (see `field`) -- this method's
        own cache just avoids rebuilding the numpy version repeatedly."""
        if n_subdiv not in self._quadrature_cache:
            self._quadrature_cache[n_subdiv] = _fixed_quadrature(self.profile, self.sigma, n_subdiv)
        return self._quadrature_cache[n_subdiv]

    def field(self, points, n_subdiv=8, xp=np):
        """Field E = -grad(potential) at `points`, shape (..., 3), via the
        vectorized fixed/near-segment quadrature (see module docstring).

        Returns
        -------
        E : ndarray, same shape as `points`.
        """
        quadrature = self.quadrature(n_subdiv)
        key = (array_cache_key(xp), n_subdiv)
        if key not in self._field_cache:
            self._field_cache[key] = _prepare_field_data(self.profile, self.sigma, quadrature, xp)
        return evaluate_axisymmetric_field(self.profile, self.sigma, points,
            quadrature=quadrature, xp=xp, _prepared=self._field_cache[key])


def _prepare_field_data(profile, sigma, quadrature, xp):
    """Geometry and quadrature are fixed throughout a trajectory."""
    geom = _segment_endpoints(profile)
    lengths = xp.asarray(np.linalg.norm(geom[2], axis=-1))
    p0, tangent = xp.asarray(geom[0]), xp.asarray(geom[2])
    return dict(geom=geom, lengths=lengths, p0=p0, unit=tangent/lengths[:, None],
        quadrature=tuple(xp.asarray(a) for a in quadrature), sigma=xp.asarray(sigma),
        gauss=tuple(xp.asarray(a) for a in (_NEAR_GAUSS_X, _NEAR_GAUSS_W)),
        closest=None if xp is np else _prepare_closest_point_gpu(geom[0], geom[2], geom[3]))


def evaluate_axisymmetric_field(profile, sigma, points, quadrature=None, n_subdiv=8, xp=np, _prepared=None):
    """The field E(x) = -grad[S[sigma]](x) at `points`, for an axisymmetric
    surface-charge density sigma given at the nodes of a generating
    profile -- see module docstring. Fully vectorized: a single
    (n_points, n_quadrature_points) broadcast, no Python loop over points.

    Parameters
    ----------
    profile : ndarray, shape (n, 2)
        Generating profile as (rho, z) points, in order along the surface
        (plain numpy -- it's small, and only used here to build the
        quadrature/segment-geometry arrays if not already cached).
    sigma : ndarray, shape (n,)
        Nodal surface-charge-density values (plain numpy, same reason).
    points : array, shape (..., 3)
        On whatever backend `xp` is (numpy or cupy).
    quadrature : (rho_q, z_q, area_q, sigma_q) numpy arrays, optional
        Precomputed via `_fixed_quadrature`/`AxisymmetricBEMSolution.quadrature`;
        built fresh (and not cached) if omitted.
    n_subdiv : int
        Only used if `quadrature` is None.
    xp : array module
        numpy or cupy; `points` must already be an array of this module.

    Returns
    -------
    E : array (same module as `points`), same shape as `points`.
    """
    profile = np.asarray(profile, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if quadrature is None:
        quadrature = _fixed_quadrature(profile, sigma, n_subdiv)
    data = _prepare_field_data(profile, sigma, quadrature, xp) if _prepared is None else _prepared
    rho_q, z_q, area_q, sigma_q = data["quadrature"]
    p0_np, p1_np, tangent_np, normal_np = data["geom"]
    lengths, p0, unit = data["lengths"], data["p0"], data["unit"]
    sigma_device = data["sigma"]
    points = xp.asarray(points, dtype=float)
    shape = points.shape
    flat = points.reshape(-1, 3)
    rho = xp.hypot(flat[:, 0], flat[:, 1])
    z = flat[:, 2].copy()
    phi = xp.arctan2(flat[:, 1], flat[:, 0])
    if not flat.shape[0]:
        return xp.empty_like(points)

    # A single-layer field jumps across the surface. Exactly on it, take
    # the exterior trace using a small local-mesh-scaled displacement.
    # At polygon vertices this specifies a finite one-sided value; the
    # continuum value there is not defined for an arbitrary nodal density.
    dist2, seg, _, normals = _closest_point_on_profile(
        rho, z, p0_np, p1_np, tangent_np, normal_np, xp, _prepared=data["closest"]
    )
    offset = xp.where(dist2 < (1e-9*lengths[seg])**2, 1e-4*lengths[seg], 0.0)
    rho_eval = rho + offset * normals[:, 0]
    rho_eval = xp.where(rho == 0., 0., rho_eval)
    z = z + offset * normals[:, 1]

    # Keep cached fixed quadrature for well-separated point/segment pairs.
    er, ez = ring_field(rho_eval[:, None], z[:, None]-z_q[None, :], rho_q[None, :], xp)
    weight = area_q * sigma_q
    n_seg = len(p0_np)
    er_seg = (er*weight[None, :]).reshape(-1, n_seg, len(rho_q)//n_seg).sum(axis=-1)
    ez_seg = (ez*weight[None, :]).reshape(-1, n_seg, len(rho_q)//n_seg).sum(axis=-1)

    delta = xp.stack([rho_eval, z], axis=-1)[:, None, :] - p0[None, :, :]
    along = xp.sum(delta * unit[None, :, :], axis=-1)
    projection = xp.clip(along, 0., lengths[None, :])
    distance = xp.linalg.norm(delta - projection[..., None]*unit[None, :, :], axis=-1)
    point_idx, seg_idx = xp.nonzero(distance < lengths[None, :])
    if point_idx.size:
        # For each close pair, s = s_closest + d*sinh(v) resolves the
        # narrow peak with O(log(length/d)) quadrature nodes. All close
        # pairs are evaluated in one batch on either CPU or GPU.
        a = p0[seg_idx]
        t = unit[seg_idx]
        L = lengths[seg_idx]
        s0 = projection[point_idx, seg_idx]
        d = xp.maximum(distance[point_idx, seg_idx], 1e-15*L)
        lo = xp.arcsinh(-s0/d)
        hi = xp.arcsinh((L-s0)/d)
        # Gauss-8 per v interval of width at most 0.5.
        count = xp.maximum(1, xp.ceil((hi-lo)/.5).astype(xp.int64))
        n_intervals = int(xp.max(count))
        gx, gw = data["gauss"]
        v = lo[:, None, None] + (xp.arange(n_intervals)[None, :, None] +
            .5*(xp.asarray(gx)[None, None, :]+1.)) * ((hi-lo)/count)[:, None, None]
        valid = xp.arange(n_intervals)[None, :, None] < count[:, None, None]
        # Clamp padded intervals before sinh to avoid overflow in unused nodes.
        v = xp.where(valid, v, 0.)
        arclength = s0[:, None, None] + d[:, None, None]*xp.sinh(v)
        source = a[:, None, None, :] + arclength[..., None]*t[:, None, None, :]
        density = sigma_device[seg_idx, None, None] + (arclength/L[:, None, None]) * (
            sigma_device[seg_idx+1]-sigma_device[seg_idx])[:, None, None]
        w = density * (2*xp.pi*source[..., 0]) * d[:, None, None]*xp.cosh(v) * (
            .5*(hi-lo)/count)[:, None, None] * xp.asarray(gw)[None, None, :]
        er_close, ez_close = ring_field(rho_eval[point_idx, None, None],
            z[point_idx, None, None]-source[..., 1], source[..., 0], xp)
        er_seg[point_idx, seg_idx] = xp.sum(xp.where(valid, w*er_close, 0.), axis=(1, 2))
        ez_seg[point_idx, seg_idx] = xp.sum(xp.where(valid, w*ez_close, 0.), axis=(1, 2))
    e_rho = er_seg.sum(axis=-1)
    e_z = ez_seg.sum(axis=-1)
    ex = xp.where(rho == 0., 0., e_rho*xp.cos(phi))
    ey = xp.where(rho == 0., 0., e_rho*xp.sin(phi))
    return xp.stack([ex, ey, e_z], axis=-1).reshape(shape)
