"""Toroidal harmonics Q_{m-1/2}(chi): the special function behind the
azimuthal-Fourier-mode decomposition of the free-space Green's function on
an axisymmetric surface (bem.ring_modes, bem.image_charge).

Why this function
------------------
For two points at cylindrical (rho, phi, z) and (a, phi', za), the
classic toroidal-coordinate expansion of 1/|r - r'| is

    1/|r-r'| = 1/(pi*sqrt(rho*a)) * sum_{n=-inf}^{inf} Q_{n-1/2}(chi) * e^{i*n*(phi-phi')}

    chi = (rho^2 + a^2 + (z-za)^2) / (2*rho*a)     [ = cosh(eta) in the
    usual toroidal-coordinate eta ]

with Q_{n-1/2} the Legendre function of the second kind at half-integer
degree ("ring function" / toroidal harmonic). Since Q_{n-1/2} = Q_{-n-1/2}
(confirmed numerically against mpmath.legenq), the n<0 and n>0 terms pair
up:

    1/|r-r'| = 1/(pi*sqrt(rho*a)) * [Q_{-1/2}(chi) + 2*sum_{m=1}^inf Q_{m-1/2}(chi)*cos(m*(phi-phi'))]

so the m-th azimuthal Fourier mode of the Green's function is entirely
carried by Q_{m-1/2}(chi) -- this is what makes a per-mode 1D (generating-
profile) BEM solve possible for a non-axisymmetric (e.g. off-axis point
charge) excitation on an axisymmetric surface, the same way bem.axisymmetric
already collapses the axisymmetric-excitation case via m=0 alone.

Closed forms for m=0, 1
------------------------
Matched against `bem.axisymmetric.ring_potential`'s existing (numerically
validated) normalization and against mpmath.legenq to double-precision:

    Q_{-1/2}(chi) = sqrt(2/(chi+1)) * K(k),                 k = 2/(chi+1)
    Q_{1/2}(chi)  = chi*sqrt(2/(chi+1))*K(k) - sqrt(2*(chi+1))*E(k)

(K, E the complete elliptic integrals of the first/second kind; note
`bem.axisymmetric.ring_potential(rho, z, a) = Q_{-1/2}(chi) / (4*pi^2*sqrt(rho*a))`,
i.e. m=0 here reproduces that module's existing kernel exactly.)

Hybrid recurrence
-----------------
With eta = arccosh(chi), the unwanted solution grows relative to Q roughly
as exp(2*m*eta). For M*eta <= 1, upward recurrence from the two elliptic
seeds is well conditioned and avoids the arbitrarily deep Miller descent
needed near coincidence. Elsewhere we retain downward ratio recurrence.
The split is per element, so one nearly coincident pair does not force an
entire batch to use a long descent. There is no accuracy-limiting padding
cap. The GPU kernel uses the same switch and a per-element descent depth.

K is evaluated from its complementary parameter (chi-1)/(chi+1), avoiding
loss of the small separation when forming 2/(chi+1). Derivatives use the
Legendre identity with (chi-1)*(chi+1) rather than chi**2-1; mode zero has
a direct elliptic expression which also avoids cancellation at large chi.
See https://dlmf.nist.gov/14.10 for the recurrence/derivative identities.

"""

import numpy as np

from ._elliptic import ellip_ke_complement

# Upper bound on M (= max(n_max, 1)) the RawKernel path below will handle;
# above this it falls back to the plain Python-loop recursion (works on
# any xp, just with the per-iteration kernel-launch overhead the RawKernel
# exists to avoid -- see _toroidal_ratio_cumprod_gpu's docstring). 64 is
# generous: this project's own image-charge mode counts top out in the
# 10s (see bem.image_charge's measured mode-count-vs-standoff table).
_TOROIDAL_MAX_KERNEL_M = 64

_TOROIDAL_RATIO_KERNEL_SOURCE = r"""
extern "C" __global__
void toroidal_ratio_cumprod(const double* chi, const double* q0,
    const double* q1, double* out, int M, int pad_min, double pad_const,
    long long n_elements) {
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_elements) return;
    double c = chi[idx], eta = acosh(c);
    out[idx] = q0[idx];
    if (M * eta <= 1.0) {
        double prev = q0[idx], curr = q1[idx];
        out[n_elements + idx] = curr;
        for (int n = 1; n < M; n++) {
            double next = (2.0*n*c*curr - (n-0.5)*prev)/(n+0.5);
            out[(long long)(n+1)*n_elements + idx] = next;
            prev = curr; curr = next;
        }
    } else {
        int pad = max(pad_min, (int)ceil(pad_const / eta));
        double r_next = 0.0, r[65];
        for (int n = M + pad; n >= 1; n--) {
            double rn = (n-0.5)/(2.0*n*c - (n+0.5)*r_next);
            if (n <= M) r[n] = rn;
            r_next = rn;
        }
        double value = q0[idx];
        for (int n = 1; n <= M; n++) {
            value *= r[n];
            out[(long long)n*n_elements + idx] = value;
        }
    }
}
"""

_toroidal_ratio_kernel = None


def _get_toroidal_ratio_kernel():
    global _toroidal_ratio_kernel
    if _toroidal_ratio_kernel is None:
        import cupy as cp

        _toroidal_ratio_kernel = cp.RawKernel(_TOROIDAL_RATIO_KERNEL_SOURCE, "toroidal_ratio_cumprod")
    return _toroidal_ratio_kernel


def _toroidal_ratio_cumprod_loop(chi, M, N, xp):
    """u[n] = r_1*r_2*...*r_n for n=0..M (u[0]=1), via the plain Python
    downward-then-upward loop (see module docstring) -- works on any xp,
    but each of the up-to-N downward iterations launches a handful of tiny
    elementwise kernels when xp is cupy, which `_toroidal_ratio_cumprod_gpu`
    exists to avoid by fusing the whole per-element recursion into one
    kernel launch instead."""
    r_next = xp.zeros(chi.shape)
    r = xp.ones((M + 1,) + chi.shape)
    for n in range(N, 0, -1):
        r_n = (n - 0.5) / (2.0 * n * chi - (n + 0.5) * r_next)
        if n <= M:
            r[n] = r_n
        r_next = r_n

    u = xp.empty((M + 1,) + chi.shape)
    u[0] = 1.0
    for n in range(1, M + 1):
        u[n] = u[n - 1] * r[n]
    return u


def _hybrid_values(chi, q0, q1, M, xp, pad_min, pad_const):
    if xp is not np and M <= _TOROIDAL_MAX_KERNEL_M:
        import cupy as cp
        arrays = [cp.ascontiguousarray(a.reshape(-1)) for a in (chi, q0, q1)]
        size = chi.size
        out = cp.empty((M + 1, size), dtype=cp.float64)
        if size:
            _get_toroidal_ratio_kernel()(
                ((size + 255) // 256,), (256,),
                (*arrays, out, np.int32(M), np.int32(pad_min),
                 np.float64(pad_const), np.int64(size)),
            )
        return out.reshape((M + 1,) + chi.shape)

    shape = chi.shape
    chi, q0, q1 = (a.reshape(-1) for a in (chi, q0, q1))
    eta = xp.arccosh(chi)
    near = M * eta <= 1.0
    out = xp.empty((M + 1, chi.size))
    cn = chi[near]
    forward = xp.empty((M + 1, cn.size))
    forward[0], forward[1] = q0[near], q1[near]
    for n in range(1, M):
        forward[n + 1] = (2*n*cn*forward[n] - (n-.5)*forward[n-1])/(n+.5)
    out[:, near] = forward
    far = ~near
    if chi[far].size:
        pad = max(pad_min, int(np.ceil(pad_const / float(xp.min(eta[far])))))
        out[:, far] = q0[far][None, :] * _toroidal_ratio_cumprod_loop(chi[far], M, M + pad, xp)
    return out.reshape((M + 1,) + shape)


def toroidal_Q(chi, n_max, xp=np, pad_min=50, pad_const=40.0):
    """Q_{n-1/2}(chi), dQ/dchi for n=0..n_max, on NumPy or CuPy.

    chi must be >= 1. Exact coincidence is regularized to the nearest
    representable number above 1 (the true kernel is singular there).
    No separation larger than roundoff is clipped. pad_min/pad_const
    control only the downward branch; the default switch M*eta <= 1
    keeps upward recurrence away from exponential error amplification.
    """
    if not isinstance(n_max, (int, np.integer)) or n_max < 0:
        raise ValueError("n_max must be a nonnegative integer")
    chi = xp.maximum(xp.asarray(chi, dtype=float), np.nextafter(1., 2.))
    M = max(n_max, 1)
    p = (chi - 1.0) / (chi + 1.0)
    K, E = ellip_ke_complement(p, xp)
    root = xp.sqrt(2.0 / (chi + 1.0))
    q0 = root * K
    # This seed is used only near coincidence; cancellation at large chi
    # is immaterial because that branch uses Miller ratios instead.
    q1 = chi * q0 - xp.sqrt(2.0 * (chi + 1.0)) * E
    q_all = _hybrid_values(chi, q0, q1, M, xp, pad_min, pad_const)
    q = q_all[:n_max + 1]
    n = xp.arange(n_max + 1, dtype=float).reshape((-1,) + (1,) * chi.ndim)
    q_prev = xp.concatenate([q_all[1:2], q[:-1]], axis=0)
    dq = (n - .5) * (chi[None, ...] * q - q_prev) / ((chi - 1.) * (chi + 1.))[None, ...]
    dq[0] = -root * E / (2.0 * (chi - 1.0))
    return q, dq
