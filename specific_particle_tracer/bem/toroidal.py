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

Recursion for m >= 2: stability
--------------------------------
Q_{n-1/2} satisfies the standard 3-term Legendre recursion (nu = n-1/2):

    (n+1/2)*Q_{n+1/2} = 2*n*chi*Q_{n-1/2} - (n-1/2)*Q_{n-3/2}

Applied *upward* (increasing n) this is catastrophically unstable: Q decays
with degree while the recursion's other solution (P-like) grows, so
rounding error is amplified exponentially -- confirmed directly against
mpmath (relative error passes 100% within ~15-20 modes, then explodes by
20+ orders of magnitude by mode 30, for chi in the range 1.05-10 actually
relevant here). Applied *downward* from a high starting order with an
arbitrary seed, then rescaled so the recursion's own Q_{-1/2} matches the
closed form above, it is stable to double precision (Miller's algorithm --
the standard fix for exactly this kind of decaying-solution recursion, as
used for e.g. spherical Bessel functions y_n or Legendre Q_n of high
integer degree). The needed padding above the highest mode actually wanted
grows as chi -> 1 (Q_{n-1/2}(chi) decays like exp(-n*arccosh(chi)), so
padding ~ 1/arccosh(chi) is needed for the seed's arbitrariness to have
decayed away by the time the recursion reaches the modes that matter) --
chi close to 1 means the field/source points are nearly coincident, which
is also exactly the regime bem.image_charge avoids feeding to this module
directly (see its module docstring: the local planar-image-charge
subtraction keeps the *residual* problem's chi comfortably away from 1).

The same 3-term recursion, differentiated (and using x = chi as the
independent variable), also gives Q_{n-1/2}'(chi) in terms of Q_{n-1/2} and
Q_{n-3/2} without a second recursion:

    (chi^2 - 1) * Q_{n-1/2}'(chi) = (n - 1/2) * [chi*Q_{n-1/2}(chi) - Q_{n-3/2}(chi)]

(confirmed against central finite differences of mpmath.legenq to ~1e-10,
limited by the finite-difference step, not the identity). This lets
`toroidal_Q` return the derivative for free from the same recursion pass,
needed by bem.ring_modes for the field (not just potential) kernels.
"""

import numpy as np

from ._elliptic import ellip_ke


def _arccosh(chi, xp):
    return xp.log(chi + xp.sqrt(chi * chi - 1.0))


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
    # disguise -- see ring_modes' docstring), but floating-point
    # cancellation in that combination can round it very slightly below 1
    # right at coincidence (exactly the regime the adaptive self-term
    # quadrature in bem.image_charge approaches), which would otherwise
    # feed a negative number into arccosh's sqrt. Mirrors the m=0 solver's
    # own `xp.clip(m, 0, 1-1e-12)` guard against the same coincidence
    # limit in a different parametrization (bem.axisymmetric.ring_potential).
    chi = xp.clip(xp.asarray(chi, dtype=float), 1.0 + 1e-13, None)
    eta = _arccosh(chi, xp)
    eta_min = float(xp.min(eta)) if eta.size else 1.0
    # Capped at 2000: beyond that, chi is close enough to 1 (nearly
    # coincident field/source points) that this module isn't the right
    # tool anyway -- see module docstring (bem.image_charge's planar-image
    # subtraction is what's meant to keep chi away from this regime).
    pad = min(2000, max(pad_min, int(np.ceil(pad_const / max(eta_min, 1e-12)))))
    N = n_max + pad

    k = 2.0 / (chi + 1.0)
    K, E = ellip_ke(k, xp)
    sqrt_pref = xp.sqrt(2.0 / (chi + 1.0))
    q0 = sqrt_pref * K
    q1 = chi * sqrt_pref * K - xp.sqrt(2.0 * (chi + 1.0)) * E

    # Downward (Miller) recursion from an arbitrary tiny seed at n=N+1, N,
    # then rescale the whole sequence so q[0] matches the closed form.
    #
    # The padding region (n from N down to n_max+1) can be thousands of
    # steps deep when chi is very close to 1 (near-coincident field/source
    # points -- exactly what bem.image_charge's adaptive self-term
    # quadrature approaches), each multiplying the running values by
    # ~2*n*chi: left unchecked this overflows float64 long before reaching
    # n_max, even though the *ratios* between values stay perfectly
    # representable. Since the recursion is linear and homogeneous,
    # rescaling the running pair by any positive factor part-way through
    # doesn't change the final answer -- only the values actually stored
    # (n <= n_max, computed last, after any rescaling in the padding
    # region above them) matter, so it's enough to keep the *running*
    # values bounded during the padding phase and only start storing once
    # n reaches n_max+1.
    q_next = xp.zeros(chi.shape)
    q_curr = xp.full(chi.shape, 1.0e-280)
    q = xp.zeros((n_max + 2,) + chi.shape)
    overflow_guard = 1.0e250
    for n in range(N, 0, -1):
        q_prev_running = (2.0 * n * chi * q_curr - (n + 0.5) * q_next) / (n - 0.5)
        too_big = xp.abs(q_prev_running) > overflow_guard
        if xp.any(too_big):
            rescale = xp.where(too_big, xp.abs(q_prev_running), 1.0)
            q_prev_running = q_prev_running / rescale
            q_curr = q_curr / rescale
            # If we've already started storing (n small enough), the
            # entries stored in earlier iterations (index > n) were
            # written before this rescale and need the same correction
            # for the final array to stay internally consistent (only
            # the *ratios* the closing scale-to-q0 step relies on matter,
            # but they have to be ratios within one consistent scaling).
            if n <= n_max + 1:
                q[n:] = q[n:] / rescale[None, ...]
        q_next = q_curr
        q_curr = q_prev_running
        if n - 1 <= n_max + 1:
            q[n - 1] = q_curr

    scale = q0 / xp.where(q[0] == 0.0, 1.0, q[0])
    q = q[: n_max + 1] * scale[None, ...]
    # The recursion's own q[0] is Q_{-1/2}; overwrite the n=0,1 entries
    # with the closed forms directly (exact, not just rescaled-consistent)
    # since they're cheap and removes any residual rescaling error there.
    q[0] = q0
    if n_max >= 1:
        q[1] = q1

    # Derivative via the same recursion identity, needing Q_{n-3/2}: for
    # n>=1 that's q[n-1]; for n=0 it's Q_{-3/2} = Q_{1/2} (the m -> -m
    # symmetry) -- always available as the closed-form q1, regardless of
    # n_max, so this doesn't depend on q having a second entry.
    n = xp.arange(n_max + 1, dtype=float).reshape((-1,) + (1,) * chi.ndim)
    q_prev = xp.concatenate([q1[None, ...], q[:-1]], axis=0)
    dq_dchi = (n - 0.5) * (chi[None, ...] * q - q_prev) / (chi[None, ...] ** 2 - 1.0)

    return q, dq_dchi
