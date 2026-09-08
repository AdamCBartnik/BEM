import numpy as np
import pytest

from specific_particle_tracer.bem.toroidal import toroidal_Q

mp = pytest.importorskip("mpmath")


def _Q_mpmath(n, chi, dps=30):
    mp.mp.dps = dps
    return float(mp.legenq(n - 0.5, 0, chi, type=3).real)


def test_toroidal_Q_matches_mpmath_legenq():
    """Q_{n-1/2}(chi) via the module's downward-recursion implementation,
    checked against mpmath's own (independently implemented) general
    Legendre-Q -- a stronger check than internal self-consistency, since
    mpmath doesn't share this module's recursion or seed formulas at all."""
    n_max = 40
    for chi_val in [1.02, 1.1, 1.5, 3.0, 10.0]:
        q, _ = toroidal_Q(np.array([chi_val]), n_max)
        for n in [0, 1, 2, 5, 10, 20, 30, 40]:
            true = _Q_mpmath(n, chi_val)
            assert abs(q[n, 0] - true) / abs(true) < 1e-10


def test_toroidal_Q_derivative_matches_finite_difference_of_mpmath():
    """dQ/dchi via the same recursion's derivative identity, checked
    against central finite differences of the mpmath reference potential
    -- independent of this module's own (internally consistent, but
    possibly wrong) derivative formula."""
    n_max = 20
    h = 1e-6
    for chi_val in [1.1, 1.5, 3.0, 10.0]:
        q, dq = toroidal_Q(np.array([chi_val]), n_max)
        for n in [0, 1, 5, 10, 20]:
            fd = (_Q_mpmath(n, chi_val + h) - _Q_mpmath(n, chi_val - h)) / (2 * h)
            assert abs(dq[n, 0] - fd) < 1e-6


def test_toroidal_Q_matches_ring_potential_at_m_equals_zero():
    """Q_{-1/2}(chi) (mode 0) must exactly reproduce the already-validated
    m=0-only kernel in bem.axisymmetric.ring_potential, since that's the
    special case this whole per-mode generalization is built to contain."""
    from specific_particle_tracer.bem.axisymmetric import ring_potential

    rho, z, a = 1.3, 0.7, 1.0
    D = (rho + a) ** 2 + z**2
    chi = (rho**2 + a**2 + z**2) / (2 * rho * a)

    q, _ = toroidal_Q(np.array([chi]), n_max=1)
    ring_pot_from_q = q[0, 0] / (4 * np.pi**2 * np.sqrt(rho * a))

    assert abs(ring_pot_from_q - ring_potential(rho, z, a)) / ring_potential(rho, z, a) < 1e-12


def test_toroidal_Q_is_vectorized_over_chi():
    chi = np.array([1.1, 1.5, 3.0, 10.0])
    q, dq = toroidal_Q(chi, n_max=10)
    assert q.shape == (11, 4)
    assert dq.shape == (11, 4)
    for i, c in enumerate(chi):
        q_single, _ = toroidal_Q(np.array([c]), n_max=10)
        assert np.allclose(q[:, i], q_single[:, 0], rtol=1e-10)


def test_toroidal_Q_stays_finite_at_near_coincidence():
    """The ratio-based recursion should never overflow no matter how close
    chi gets to 1 (unlike a value-tracking downward recursion, which needs
    thousands of padding steps -- each multiplying by ~2*n*chi -- before
    the final rescale-to-a-known-value fixes the overall scale, and can
    overflow float64 well before reaching it). Exercises exactly the
    near-coincidence regime bem.image_charge's adaptive self-term
    quadrature approaches."""
    for eps in [1e-2, 1e-6, 1e-10, 1e-13]:
        q, dq = toroidal_Q(np.array([1.0 + eps]), n_max=30)
        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(dq))
        assert q[0, 0] > 0.0


def test_toroidal_Q_n_max_zero_still_gets_correct_derivative():
    """n_max=0 needs Q_{1/2} internally (Q_{-3/2} = Q_{1/2} by symmetry,
    for the n=0 derivative) even though it's never returned -- regression
    check for the M = max(n_max, 1) bookkeeping."""
    chi = np.array([1.3, 2.5])
    q0, dq0 = toroidal_Q(chi, n_max=0)
    q_full, dq_full = toroidal_Q(chi, n_max=3)
    assert np.allclose(q0[0], q_full[0])
    assert np.allclose(dq0[0], dq_full[0])
