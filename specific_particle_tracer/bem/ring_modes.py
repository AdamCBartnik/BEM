"""Azimuthal-Fourier-mode kernels built on bem.toroidal's Q_{m-1/2}, for
BEM problems on an axisymmetric surface with a non-axisymmetric (generally
off-axis point-charge) excitation -- see bem.image_charge, which uses these
to build a per-mode 1D generating-profile solve the same way bem.axisymmetric
uses `ring_potential`/`ring_field` (its m=0-only special case) for the
axisymmetric-excitation problem.

Two kernels, both derived (and validated, see tests/test_bem_ring_modes.py)
from the same toroidal-coordinate expansion of 1/|r-r'| used in
bem.toroidal's module docstring:

`ring_potential_modes`/`ring_field_modes` -- the single-layer BIE operator
    kernel: the potential (and its (rho, z) gradient) of a *source ring* of
    radius a at height za carrying azimuthal density cos(m*phi'), evaluated
    at (rho, z), *not including* the cos(m*phi) angular factor at the field
    point (the caller applies that once, when reconstructing the physical
    field/potential by summing modes -- keeping the reusable per-mode
    building block free of unnecessary phi book-keeping). Reduces exactly
    to `bem.axisymmetric.ring_potential`/`ring_field` at m=0 (checked in
    tests). Given by

        Phi_m(rho, z; a, za) = sqrt(a/rho) * Q_{m-1/2}(chi) / (2*pi)

    with chi = (rho^2 + a^2 + (z-za)^2) / (2*rho*a) -- the same chi as
    bem.toroidal. A profile element at (a, za) carrying nodal density
    sigma_m contributes sigma_m * Phi_m(rho, z; a, za) * ds' to the mode-m
    potential (no extra 2*pi*a factor, unlike `ring_potential`'s unit-
    *total*-charge convention -- Phi_m already represents a literal
    cos(m*phi')-density ring, not a renormalized unit-charge one).

`point_charge_potential_modes` -- the excitation: the m-th cos(m*phi)
    Fourier coefficient of a unit point charge's own potential (G =
    1/(4*pi*r)), needed to build the right-hand side of the per-mode BIE
    at each surface profile node. Given by

        g_m(rho, z; rho0, z0) = C_m * Q_{m-1/2}(chi) / sqrt(rho*rho0),
        C_0 = 1/(4*pi^2),  C_m = 1/(2*pi^2) for m >= 1

    (a point charge, not a ring, so this is a different normalization from
    Phi_m above -- validated independently in tests by reconstructing
    sum_m g_m*cos(m*phi) and checking it converges to the exact point-
    charge potential as more modes are kept).

No `point_charge_field_modes` is needed: the point charge's *own* direct
field at any evaluation point is just ordinary Coulomb's law -- the mode
decomposition is only needed for the *induced* response, i.e. these two
kernels are enough to build bem.image_charge's per-mode BIE, whose *output*
(the induced sigma_m) is then evaluated back into real (rho,phi,z) space
via `ring_field_modes` (not this excitation kernel) summed over m.
"""

import numpy as np

from .toroidal import toroidal_Q

_mode_field_fused = None
_mode_coefficients_fused = None


def _chi(rho, z, a, za):
    """chi = (rho^2 + a^2 + (z-za)^2) / (2*rho*a), computed as
    1 + [(rho-a)^2 + (z-za)^2] / (2*rho*a) instead -- algebraically
    identical, but avoids the near-coincidence cancellation of the naive
    form (rho^2+a^2 nearly cancelling -2*rho*a when rho~a), which is
    exactly what pushed chi slightly below its true minimum of 1 and
    required toroidal_Q's clip guard in the first place."""
    return 1.0 + ((rho - a) ** 2 + (z - za) ** 2) / (2.0 * rho * a)


def _mode_prefactors(n_max, xp):
    c = xp.full(n_max + 1, 1.0 / (2.0 * xp.pi**2))
    c[0] = 1.0 / (4.0 * xp.pi**2)
    return c


def ring_potential_modes(rho, z, a, za, n_max, xp=np):
    """Phi_m(rho, z; a, za) for m=0..n_max (see module docstring) -- the
    potential of a source ring at (a, za) with azimuthal density
    cos(m*phi'), excluding the field point's own cos(m*phi) factor.

    rho, z, a, za broadcast as ordinary array arithmetic; returned array
    has shape (n_max+1,) + broadcast_shape.
    """
    rho, z, a, za = xp.broadcast_arrays(
        *(xp.asarray(v, dtype=float) for v in (rho, z, a, za))
    )
    chi = _chi(rho, z, a, za)
    q, _ = toroidal_Q(chi, n_max, xp=xp)
    return xp.sqrt(a / rho)[None, ...] * q / (2.0 * xp.pi)


def ring_field_modes(rho, z, a, za, n_max, xp=np):
    """(Phi_m, dPhi_m/drho, dPhi_m/dz) for m=0..n_max, at (rho, z) due to a
    source ring at (a, za) with azimuthal density cos(m*phi') -- see
    `ring_potential_modes`. The caller combines these with the field
    point's own phi to get the actual (E_rho, E_phi, E_z):

        E_rho = -cos(m*phi) * dPhi_m/drho
        E_z   = -cos(m*phi) * dPhi_m/dz
        E_phi = (m/rho) * Phi_m * sin(m*phi)

    (E_phi from -1/rho * d/dphi[Phi_m*cos(m*phi)]; zero at m=0, as it must
    be for an axisymmetric source.)
    """
    rho, z, a, za = xp.broadcast_arrays(
        *(xp.asarray(v, dtype=float) for v in (rho, z, a, za))
    )
    chi = _chi(rho, z, a, za)
    q, dq = toroidal_Q(chi, n_max, xp=xp)
    if xp is np:
        return _mode_field_arrays(q, dq, rho[None, ...], z[None, ...], a[None, ...], za[None, ...], xp)
    global _mode_field_fused, _mode_coefficients_fused
    if _mode_field_fused is None:
        import cupy as cp
        _mode_coefficients_fused = cp.fuse(kernel_name="bem_mode_coefficients")(
            lambda rho, z, a, za: _mode_field_coefficients(rho, z, a, za, cp))
        _mode_field_fused = cp.fuse(kernel_name="bem_mode_field")(
            lambda q, dq, pref, cr, cdr, cdz: (pref*q, cr*q+cdr*dq, cdz*dq))
    # Geometry is independent of harmonic order. Fusing it together with
    # Q would repeat expensive divisions/square roots for every mode.
    coefficients = _mode_coefficients_fused(rho, z, a, za)
    return _mode_field_fused(q, dq, *(c[None, ...] for c in coefficients))


def _mode_field_coefficients(rho, z, a, za, xp):
    pref = xp.sqrt(a / rho) / (2.0 * xp.pi)
    dchi_drho = (rho**2 - a**2 - (z-za)**2) / (2.0 * a * rho**2)
    dchi_dz = (z-za) / (a * rho)
    return pref, -pref/(2.0*rho), pref*dchi_drho, pref*dchi_dz


def _mode_field_arrays(q, dq, rho, z, a, za, xp):
    dchi_drho = (rho**2 - a**2 - (z - za) ** 2) / (2.0 * a * rho**2)
    dchi_dz = (z - za) / (a * rho)
    sqrt_pref = xp.sqrt(a / rho)

    Phi = sqrt_pref * q / (2.0 * xp.pi)
    dPhi_drho = (
        -sqrt_pref / (2.0 * rho) * q
        + sqrt_pref * dq * dchi_drho
    ) / (2.0 * xp.pi)
    dPhi_dz = (sqrt_pref * dq * dchi_dz) / (2.0 * xp.pi)
    return Phi, dPhi_drho, dPhi_dz


def point_charge_potential_modes(rho, z, rho0, z0, n_max, xp=np):
    """g_m(rho, z; rho0, z0) for m=0..n_max (see module docstring) -- the
    m-th cos(m*phi) Fourier coefficient of a unit point charge's potential
    at (rho0, phi'=0, z0), evaluated at (rho, z) (any phi -- the physical
    potential is sum_m g_m * cos(m*phi))."""
    rho, z, rho0, z0 = xp.broadcast_arrays(
        *(xp.asarray(v, dtype=float) for v in (rho, z, rho0, z0))
    )
    chi = _chi(rho, z, rho0, z0)
    q, _ = toroidal_Q(chi, n_max, xp=xp)
    c = _mode_prefactors(n_max, xp)
    shape = (n_max + 1,) + (1,) * rho.ndim
    return c.reshape(shape) * q / xp.sqrt(rho * rho0)[None, ...]
