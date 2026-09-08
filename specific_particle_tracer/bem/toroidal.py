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

Recursion for m >= 2: stability, and why by *ratio*
------------------------------------------------------
Q_{n-1/2} satisfies the standard 3-term Legendre recursion (nu = n-1/2):

    (n+1/2)*Q_{n+1/2} = 2*n*chi*Q_{n-1/2} - (n-1/2)*Q_{n-3/2}

Applied *upward* (increasing n) this is catastrophically unstable: Q decays
with degree while the recursion's other solution (P-like) grows, so
rounding error is amplified exponentially -- confirmed directly against
mpmath (relative error passes 100% within ~15-20 modes, then explodes by
20+ orders of magnitude by mode 30, for chi in the range 1.05-10 actually
relevant here). Applied *downward* from a high starting order it is
stable (Miller's algorithm -- the standard fix for exactly this kind of
decaying-solution recursion, as used for e.g. spherical Bessel functions
y_n or Legendre Q_n of high integer degree): the needed padding above the
highest mode actually wanted grows as chi -> 1 (Q_{n-1/2}(chi) decays like
exp(-n*arccosh(chi)), so padding ~ 1/arccosh(chi) is needed for the
starting values' arbitrariness to have decayed away by the time the
recursion reaches the modes that matter) -- chi close to 1 means the
field/source points are nearly coincident, which is also exactly the
regime bem.image_charge avoids feeding to this module directly (see its
module docstring: the local planar-image-charge subtraction keeps the
*residual* problem's chi comfortably away from 1).

Rather than track Q_{n-1/2} itself downward (which needs an arbitrary tiny
seed and can overflow float64 during the padding phase, before the final
rescale-to-a-known-value fixes the overall scale -- there IS no overall
scale yet while descending, so the running values grow essentially
unboundedly for large chi or deep padding), this module recurs the
*ratio* r_n = Q_{n-1/2}/Q_{n-3/2} instead. Dividing the recursion above by
Q_{n-1/2} and solving for r_n gives a downward *continued-fraction*
recursion:

    r_n = (n-1/2) / (2*n*chi - (n+1/2)*r_{n+1}),   r_{N+1} = 0

Every r_n is a ratio between two terms of a decaying sequence, so it stays
a bounded, well-scaled O(1) number throughout the descent -- no seed
arbitrariness, no overflow, and (unlike the value-tracking version) no
data-dependent rescaling check partway through, which matters on GPU: a
per-iteration `if array.any(...):` forces a device-to-host sync every
step. Once the full ratio sequence r_1..r_M is known, the values follow
directly from the one known scale, Q_{-1/2}(chi) (closed form below, via
K alone -- Q_{1/2} needed for the derivative identity comes out of the
recursion too, r_1 = Q_{1/2}/Q_{-1/2}, so E is no longer needed at all):

    Q_{-1/2} = q0,   Q_{n-1/2} = q0 * r_1 * r_2 * ... * r_n

Known limitation: the padding depth is chosen from the *smallest* chi in
the whole batch (the one needing the most padding), so one nearly-
coincident element forces every other element in the same call to pay for
that depth too. Fine for how this module is actually called (batches of
comparable field/source separations), but a real inefficiency for a batch
mixing very different chi -- fixable with a RawKernel giving each element
its own N, or by bucketing calls by chi range, neither implemented here.

Measured against the original value-tracking Miller implementation (same
padding heuristic, chi in [1.05, 5], n_max=20, 10-run average): the ratio
form is faster everywhere, and dramatically so on GPU, where the removed
`if array.any(...):` branch's per-iteration device-to-host sync was most
of the old cost:

    N        CPU old   CPU ratio   GPU old   GPU ratio
    1        0.44 ms    0.19 ms    19.1 ms     4.6 ms
    100      0.71 ms    0.30 ms    22.4 ms     8.1 ms
    10000    6.28 ms    4.89 ms    31.1 ms     8.3 ms
    1e6         --         --      38.4 ms    12.6 ms

The 3-term recursion, differentiated (using x = chi as the independent
variable), also gives Q_{n-1/2}'(chi) in terms of Q_{n-1/2} and Q_{n-3/2}
without a second recursion:

    (chi^2 - 1) * Q_{n-1/2}'(chi) = (n - 1/2) * [chi*Q_{n-1/2}(chi) - Q_{n-3/2}(chi)]

(confirmed against central finite differences of mpmath.legenq to ~1e-10,
limited by the finite-difference step, not the identity). This lets
`toroidal_Q` return the derivative for free from the same recursion pass,
needed by bem.ring_modes for the field (not just potential) kernels.

An FFT-based alternative (pulling all modes at once as Fourier
coefficients of 1/sqrt(chi-cos(phi)) via `xp.fft.rfft`) was tried and
dropped: its error hits a floor around 1e-4 to 1e-2 relative for modes
that have decayed a few orders below the dominant one (extracting an
exponentially-small Fourier coefficient from O(1)-magnitude samples is
its own form of catastrophic cancellation, more sample points don't fix
it) -- not accurate enough for the operator assembly step, which needs
every mode kept to full precision regardless of size.
"""

import numpy as np

from ._elliptic import ellip_ke

# Upper bound on M (= max(n_max, 1)) the RawKernel path below will handle;
# above this it falls back to the plain Python-loop recursion (works on
# any xp, just with the per-iteration kernel-launch overhead the RawKernel
# exists to avoid -- see _toroidal_ratio_cumprod_gpu's docstring). 64 is
# generous: this project's own image-charge mode counts top out in the
# 10s (see bem.image_charge's measured mode-count-vs-standoff table).
_TOROIDAL_MAX_KERNEL_M = 64

_TOROIDAL_RATIO_KERNEL_SOURCE = r"""
extern "C" __global__
void toroidal_ratio_cumprod(const double* chi, double* u, int M, int N, long long n_elements) {
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_elements) return;

    double c = chi[idx];
    double r_next = 0.0;
    double r[65];  // r[1..M], M <= 64 enforced by the Python wrapper

    for (int n = N; n >= 1; n--) {
        double rn = (n - 0.5) / (2.0 * n * c - (n + 0.5) * r_next);
        if (n <= M) {
            r[n] = rn;
        }
        r_next = rn;
    }

    double cumprod = 1.0;
    for (int n = 1; n <= M; n++) {
        cumprod *= r[n];
        u[(long long)(n - 1) * n_elements + idx] = cumprod;
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


def _toroidal_ratio_cumprod_gpu(chi, M, N):
    """Same as `_toroidal_ratio_cumprod_loop` (u[n] = r_1*...*r_n, u[0]=1),
    but as a single cupy.RawKernel launch: one CUDA thread per chi element,
    looping over its own N-step recursion (and the M-step cumulative
    product) entirely in device code. `_toroidal_ratio_cumprod_loop`'s
    Python loop instead launches ~N tiny elementwise kernels (one set of
    multiply/subtract/divide ops per downward step, over the *whole* chi
    array each time) -- fine for a handful of iterations, but N can reach
    the low thousands for near-coincident chi (see module docstring), and
    each launch carries fixed overhead regardless of how much work it
    does. Measured (chi in [1.05, 5], n_max=20, RTX 5070 Ti): the
    kernel-launch overhead this removes dominates `toroidal_Q`'s GPU cost
    at small-to-medium batch sizes -- see bem.image_charge's own
    CPU-vs-GPU image_force timing, which is what motivated writing this."""
    import cupy as cp

    chi_flat = cp.ascontiguousarray(chi.reshape(-1))
    n_elements = chi_flat.size
    out = cp.empty((M, n_elements), dtype=cp.float64)  # out[n-1] = u[n], n=1..M
    if n_elements > 0:
        threads = 256
        blocks = (n_elements + threads - 1) // threads
        kernel = _get_toroidal_ratio_kernel()
        kernel((blocks,), (threads,), (chi_flat, out, np.int32(M), np.int32(N), np.int64(n_elements)))
    u = cp.empty((M + 1, n_elements), dtype=cp.float64)
    u[0] = 1.0
    u[1:] = out
    return u.reshape((M + 1,) + chi.shape)


def toroidal_Q(chi, n_max, xp=np, pad_min=50, pad_const=40.0):
    """Q_{n-1/2}(chi) and its derivative w.r.t. chi, for n = 0..n_max.

    Parameters
    ----------
    chi : array, elementwise > 1 (strictly -- see module docstring: this
        is not meant to be called with chi close to 1).
    n_max : int
    xp : numpy or cupy.
    pad_min, pad_const : downward-recursion padding controls (see module
        docstring); the defaults were checked against mpmath to give
        double-precision agreement for chi >= 1.01, and were not tuned
        finer than that -- see tests/test_bem_toroidal.py.

    Returns
    -------
    q, dq_dchi : arrays, shape (n_max+1,) + chi.shape.
    """
    # chi is mathematically >= 1 always (it's (rho-a)^2+(z-za)^2 >= 0 in
    # disguise -- see ring_modes' docstring, which computes it via the
    # numerically stabler 1 + [(rho-a)^2+(z-za)^2]/(2*rho*a) form for
    # exactly this reason), but floating-point cancellation can still
    # round it very slightly below 1 right at coincidence, which would
    # otherwise feed a negative number into arccosh. Mirrors the m=0
    # solver's own `xp.clip(m, 0, 1-1e-12)` guard against the same
    # coincidence limit in a different parametrization
    # (bem.axisymmetric.ring_potential).
    chi = xp.clip(xp.asarray(chi, dtype=float), 1.0 + 1e-13, None)
    eta = xp.arccosh(chi)
    eta_min = float(xp.min(eta)) if eta.size else 1.0
    # Capped at 2000: beyond that, chi is close enough to 1 (nearly
    # coincident field/source points) that this module isn't the right
    # tool anyway -- see module docstring (bem.image_charge's planar-image
    # subtraction is what's meant to keep chi away from this regime). No
    # overflow risk drove this cap (the ratio recursion below has none);
    # it's purely to bound the Python loop length.
    pad = min(2000, max(pad_min, int(np.ceil(pad_const / max(eta_min, 1e-12)))))
    # Always resolve at least Q_{1/2} (M=1): the derivative identity at
    # n=0 needs Q_{-3/2} = Q_{1/2} (the m -> -m symmetry) even when the
    # caller only wants n_max=0.
    M = max(n_max, 1)
    N = M + pad

    K, _ = ellip_ke(2.0 / (chi + 1.0), xp)  # E discarded: not needed (see module docstring)
    q0 = xp.sqrt(2.0 / (chi + 1.0)) * K

    # u[n] = r_1*r_2*...*r_n (the downward continued-fraction recursion's
    # cumulative product, see module docstring); q_all = q0*u. On GPU with
    # a modest mode count, the fused RawKernel avoids the Python loop's
    # per-iteration launch overhead -- see _toroidal_ratio_cumprod_gpu.
    if xp is not np and M <= _TOROIDAL_MAX_KERNEL_M:
        u = _toroidal_ratio_cumprod_gpu(chi, M, N)
    else:
        u = _toroidal_ratio_cumprod_loop(chi, M, N, xp)

    q_all = q0[None, ...] * u
    q = q_all[: n_max + 1]
    q1 = q_all[1]  # Q_{1/2}, always available regardless of n_max (see M above)

    # Derivative via the same recursion identity, needing Q_{n-3/2}: for
    # n>=1 that's q[n-1]; for n=0 it's Q_{-3/2} = Q_{1/2}, i.e. q1.
    n = xp.arange(n_max + 1, dtype=float).reshape((-1,) + (1,) * chi.ndim)
    q_prev = xp.concatenate([q1[None, ...], q[:-1]], axis=0)
    dq_dchi = (n - 0.5) * (chi[None, ...] * q - q_prev) / (chi[None, ...] ** 2 - 1.0)

    return q, dq_dchi
