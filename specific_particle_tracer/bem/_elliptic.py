"""Shared complete-elliptic-integral helper, used by both the m=0-only
axisymmetric solver (bem.axisymmetric) and the general per-mode toroidal-
harmonic machinery (bem.toroidal) built on top of it."""

import numpy as np


def ellip_ke(m, xp):
    """(K(m), E(m)), the complete elliptic integrals of the first and
    second kind, to full double precision on either backend.

    numpy: scipy.special.ellipk/ellipe directly. cupy: cupyx.scipy.special
    provides ellipk but not ellipe (as of this writing) -- E(m) instead
    comes from the incomplete elliptic integral of the second kind at its
    upper limit pi/2, ellipeinc(pi/2, m), which *is* the complete integral
    by definition (confirmed to match scipy.special.ellipe to within
    2e-16, i.e. floating-point roundoff, not an approximation)."""
    if xp is np:
        from scipy import special

        return special.ellipk(m), special.ellipe(m)

    from cupyx.scipy import special as cupy_special

    return cupy_special.ellipk(m), cupy_special.ellipeinc(xp.pi / 2, m)


def ellip_ke_complement(p, xp):
    """K(1-p), E(1-p), retaining the small complementary parameter.

    Forming 1-p first loses relative accuracy near a ring singularity.
    ellipkm1 accepts p directly on both supported backends. E is finite
    there, so rounding its argument does not produce the same problem.
    """
    if xp is np:
        from scipy import special
        return special.ellipkm1(p), special.ellipe(1.0 - p)
    from cupyx.scipy import special
    return special.ellipkm1(p), special.ellipeinc(xp.pi / 2, 1.0 - p)
