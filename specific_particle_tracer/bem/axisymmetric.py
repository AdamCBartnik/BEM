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

This only works because the excitation is axisymmetric, not merely because
the geometry is: a uniform field along the symmetry axis has Dirichlet
data with no phi-dependence, so the induced charge has none either. That
holds for any future axisymmetric real-cathode shape sitting in an
on-axis field, not just this hemisphere-tip case. It will *not* hold for
the image-charge job, even on the (still axisymmetric) recessed surface:
the excitation there is a real particle at some generally off-axis 3D
position, which breaks the rotational symmetry of the induced image
charge regardless of the surface's own symmetry -- so that job should
default to the general 3D machinery (bem.mesh/bem.panel_field), not try
to force a 1D reduction here. (A middle ground exists if 3D turns out too
slow there too: decompose the off-axis excitation into azimuthal Fourier
modes and solve one 1D problem per mode -- more moving parts, so a
fallback optimization, not a starting point.)

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

As with bem.panel_field's Coulomb kernel, the *field* kernel here doesn't
enjoy the cancellation that keeps the *potential* kernel's self term
(a log singularity, confirmed to converge cleanly under adaptive
subdivision) well-behaved: near the profile, it's the same "summing
large, nearly-canceling contributions" problem hit in 3D. Fixed the same
way: shift the density by its value at the true closest point on the
profile (found by exact point-to-segment projection, not just the nearest
node) before summing, and add back that reference density's exact
local-infinite-sheet contribution, sigma_ref * n(x) -- applied only within
a small fraction of one profile-segment's size, since (checked
empirically, as in the 3D version) the local-reference approximation is
actively worse than plain summation beyond that.

Vectorization
-------------
The one-time BEM *solve* (collocation matrix assembly) keeps the original
adaptive-subdivision quadrature (`_segment_potential`): it runs once per
geometry construction, is not a bottleneck, and the potential's self-term
needs genuine adaptivity to converge quickly (checked directly: a fixed,
non-adaptive quadrature converges to the same self-term value only very
slowly, unlike the adaptive one).

Field *evaluation* -- called repeatedly by the tracker's ODE integrator,
potentially for many points at once -- instead uses a fixed (non-adaptive)
quadrature built once over the whole profile and cached, so every
subsequent call is a single batched (n_points, n_quadrature_points)
broadcast: no per-point, per-segment Python loop, and no data-dependent
branching (subdivide-or-not) that would defeat GPU execution. Evaluating
a point exactly on top of a quadrature node is never required for this
fixed quadrature (the near-surface case is handled by the reference-shift
trick instead), so the lack of adaptivity here doesn't cost accuracy.
"""

import numpy as np

from ._elliptic import ellip_ke as _ellip_ke

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


def _closest_point_on_profile_gpu(rho, z, p0, p1, tangent, normal):
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
    p0 = cp.asarray(p0, dtype=cp.float64)
    tangent = cp.asarray(tangent, dtype=cp.float64)
    normal = cp.asarray(normal, dtype=cp.float64)
    # Each column of a (n_seg, 2) C-order array is a stride-2 view, not a
    # contiguous buffer -- a RawKernel argument is just a raw pointer with
    # no stride information, so an implicitly-strided view here would read
    # silently-wrong (shuffled) data instead of raising. Every column
    # passed to the kernel below must be made contiguous explicitly.
    p0x, p0y = cp.ascontiguousarray(p0[:, 0]), cp.ascontiguousarray(p0[:, 1])
    abx, aby = cp.ascontiguousarray(tangent[:, 0]), cp.ascontiguousarray(tangent[:, 1])
    nx, ny = cp.ascontiguousarray(normal[:, 0]), cp.ascontiguousarray(normal[:, 1])
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
    m = xp.clip(4.0 * a * rho / D, 0.0, 1.0 - 1e-12)
    K, _ = _ellip_ke(m, xp)
    return K / (2.0 * xp.pi**2 * xp.sqrt(D))


def ring_field(rho, z, a, xp=np):
    """Field (E_rho, E_z) at cylindrical (rho, z) of a unit-total-charge
    ring of radius a centered on the z-axis at height 0. Arguments
    broadcast as ordinary array arithmetic."""
    D = (rho + a) ** 2 + z**2
    m = xp.clip(4.0 * a * rho / D, 0.0, 1.0 - 1e-12)
    K, E = _ellip_ke(m, xp)
    sqrtD = xp.sqrt(D)

    dD_drho = 2.0 * (rho + a)
    dD_dz = 2.0 * z
    dm_drho = (4.0 * a * D - 4.0 * a * rho * dD_drho) / D**2
    dm_dz = (-4.0 * a * rho * dD_dz) / D**2

    # dK/dm = (E - (1-m)K) / (2m(1-m)); removable 0/0 at m=0 (limit pi/8,
    # from K's power series). m_safe avoids a division by ~0 feeding NaN
    # into the branch xp.where discards (xp.where evaluates both sides).
    on_axis = m < 1e-9
    m_safe = xp.where(on_axis, 0.5, m)  # arbitrary safe value; result discarded by xp.where below
    dK_dm_general = (E - (1.0 - m_safe) * K) / (2.0 * m_safe * (1.0 - m_safe))
    dK_dm = xp.where(on_axis, xp.pi / 8.0, dK_dm_general)

    scale = 2.0 * xp.pi**2  # matches ring_potential's normalization
    dV_drho = (dK_dm * dm_drho) / (scale * sqrtD) - K * dD_drho / (2.0 * scale * D**1.5)
    dV_dz = (dK_dm * dm_dz) / (scale * sqrtD) - K * dD_dz / (2.0 * scale * D**1.5)
    return -dV_drho, -dV_dz


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


def _closest_point_on_profile(rho, z, p0, p1, tangent, normal, xp):
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
        return _closest_point_on_profile_gpu(rho, z, p0, p1, tangent, normal)

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
    """

    def __init__(self, profile, sigma):
        self.profile = profile
        self.sigma = sigma
        self._quadrature_cache = {}

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
        too (see `bem.fields.HemisphericalTipBEMField`) -- this method's
        own cache just avoids rebuilding the numpy version repeatedly."""
        if n_subdiv not in self._quadrature_cache:
            self._quadrature_cache[n_subdiv] = _fixed_quadrature(self.profile, self.sigma, n_subdiv)
        return self._quadrature_cache[n_subdiv]

    def field(self, points, n_subdiv=8, xp=np):
        """Field E = -grad(potential) at `points`, shape (..., 3), via the
        vectorized fixed-quadrature evaluator (see module docstring).

        Returns
        -------
        E : ndarray, same shape as `points`.
        """
        quadrature = self.quadrature(n_subdiv)
        return evaluate_axisymmetric_field(self.profile, self.sigma, points, quadrature=quadrature, xp=xp)


def evaluate_axisymmetric_field(profile, sigma, points, quadrature=None, n_subdiv=8, xp=np):
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
    rho_q, z_q, area_q, sigma_q = (xp.asarray(a) for a in quadrature)

    p0, p1, tangent, normal = _segment_endpoints(profile)
    seg_sigma0, seg_sigma1 = xp.asarray(sigma[:-1]), xp.asarray(sigma[1:])

    points = xp.asarray(points, dtype=float)
    shape = points.shape
    flat = points.reshape(-1, 3)
    x, y, z = flat[:, 0], flat[:, 1], flat[:, 2]
    rho = xp.hypot(x, y)
    phi = xp.arctan2(y, x)

    # 0.1x (not, say, 0.5x): checked empirically against the analytic
    # hemisphere solution -- much beyond this the regularized branch is
    # *worse* than plain summation would be at that same distance (its
    # "local reference density" approximation is only good very close in),
    # so a generous threshold actively hurts otherwise-accurate points
    # near, but not that near, the surface.
    segment_lengths = np.linalg.norm(p1 - p0, axis=-1)
    near_surface_threshold = 0.1 * np.sqrt(np.mean(segment_lengths**2))

    best_dist2, best_seg, best_t, best_normal = _closest_point_on_profile(
        rho, z, p0, p1, tangent, normal, xp
    )
    sigma_at_closest = seg_sigma0[best_seg] + best_t * (seg_sigma1[best_seg] - seg_sigma0[best_seg])
    near_surface = best_dist2 < near_surface_threshold**2
    sigma_ref = xp.where(near_surface, sigma_at_closest, 0.0)

    # (M, Q) broadcast: field from every quadrature node, density shifted
    # per-point by that point's own sigma_ref.
    e_rho_q, e_z_q = ring_field(rho[:, None], z[:, None] - z_q[None, :], rho_q[None, :], xp=xp)
    weight = (sigma_q[None, :] - sigma_ref[:, None]) * area_q[None, :]
    e_rho = xp.sum(weight * e_rho_q, axis=-1) + sigma_ref * best_normal[:, 0]
    e_z = xp.sum(weight * e_z_q, axis=-1) + sigma_ref * best_normal[:, 1]

    on_axis = rho <= 1e-15
    ex = xp.where(on_axis, 0.0, e_rho * xp.cos(phi))
    ey = xp.where(on_axis, 0.0, e_rho * xp.sin(phi))
    result = xp.stack([ex, ey, e_z], axis=-1)
    return result.reshape(shape)
