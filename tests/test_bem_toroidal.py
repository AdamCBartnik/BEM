import numpy as np
import pytest

from specific_particle_tracer.bem.toroidal import toroidal_Q

mp = pytest.importorskip("mpmath")


@pytest.mark.parametrize("n_max", [0, 16, 64, 128])
def test_near_coincidence_and_recurrence_switch_match_high_precision(n_max):
    M = max(n_max, 1)
    chi = np.array([np.nextafter(1., 2.), 1.+1e-13, 1.+1e-10,
                    1.+1e-6, np.cosh(.999/M), np.cosh(1.001/M), 1.02, 2.])
    q, dq = toroidal_Q(chi, n_max)
    with mp.workdps(60):
        for i, c in enumerate(chi):
            x = mp.mpf(float(c))  # compare at the same representable input
            for n in sorted(set([0, n_max//2, n_max])):
                fn = lambda y: mp.legenq(mp.mpf(n)-.5, 0, y, type=3).real
                assert q[n, i] == pytest.approx(float(fn(x)), rel=1e-10, abs=0.)
                assert dq[n, i] == pytest.approx(float(mp.diff(fn, x)), rel=1e-10, abs=0.)


@pytest.mark.parametrize("n_max", [0, 32, 69])
def test_mixed_near_and_far_batch_on_gpu(n_max):
    cp = pytest.importorskip("cupy")
    chi = np.array([np.nextafter(1., 2.), 1.+1e-12, 1.+1e-6, 1.001, 1.1, 5.])
    expected = toroidal_Q(chi, n_max)
    actual = toroidal_Q(cp.asarray(chi), n_max, xp=cp)
    for cpu, gpu in zip(expected, actual):
        np.testing.assert_allclose(cp.asnumpy(gpu), cpu, rtol=1e-10, atol=0.)


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


def test_toroidal_Q_gpu_kernel_matches_cpu_and_mpmath():
    """The RawKernel path (dispatched automatically whenever xp is cupy,
    for mode counts up to _TOROIDAL_MAX_KERNEL_M) must reproduce the
    plain-Python-loop CPU path exactly, and mpmath to the same tolerance
    the CPU path already meets -- this is the whole recursion fused into
    one kernel launch per call instead of one Python-level (and, on GPU,
    kernel-launch) iteration per recursion step, so it's worth checking
    independently rather than trusting it just because the CPU path
    already passed (a RawKernel's argument marshalling -- e.g. an
    accidentally non-contiguous array view read as if it were a flat
    buffer -- can silently produce wrong-but-finite numbers with no
    exception at all, unlike a plain Python/numpy bug)."""
    cp = pytest.importorskip("cupy")

    for chi_val in [1.02, 1.1, 1.5, 3.0, 10.0]:
        chi_cpu = np.array([chi_val, chi_val * 1.3, chi_val + 2.0])
        q_cpu, dq_cpu = toroidal_Q(chi_cpu, n_max=40)

        chi_gpu = cp.asarray(chi_cpu)
        q_gpu, dq_gpu = toroidal_Q(chi_gpu, n_max=40, xp=cp)
        q_gpu_np, dq_gpu_np = cp.asnumpy(q_gpu), cp.asnumpy(dq_gpu)

        assert np.max(np.abs(q_gpu_np - q_cpu)) < 1e-12
        assert np.max(np.abs(dq_gpu_np - dq_cpu)) < 1e-6

        for n in [0, 1, 10, 40]:
            true = _Q_mpmath(n, chi_val)
            assert abs(q_gpu_np[n, 0] - true) / abs(true) < 1e-10


def test_toroidal_Q_gpu_kernel_stays_finite_at_near_coincidence():
    cp = pytest.importorskip("cupy")
    for eps in [1e-2, 1e-6, 1e-10, 1e-13]:
        q, dq = toroidal_Q(cp.array([1.0 + eps]), n_max=30, xp=cp)
        q_np, dq_np = cp.asnumpy(q), cp.asnumpy(dq)
        assert np.all(np.isfinite(q_np))
        assert np.all(np.isfinite(dq_np))
        assert q_np[0, 0] > 0.0


def test_toroidal_Q_falls_back_to_loop_above_kernel_mode_cap():
    """n_max above _TOROIDAL_MAX_KERNEL_M must still work correctly on
    GPU (via the plain-loop fallback), not error or silently truncate."""
    cp = pytest.importorskip("cupy")
    from specific_particle_tracer.bem.toroidal import _TOROIDAL_MAX_KERNEL_M

    n_max = _TOROIDAL_MAX_KERNEL_M + 5
    chi_cpu = np.array([1.3, 2.5])
    q_cpu, _ = toroidal_Q(chi_cpu, n_max=n_max)
    q_gpu, _ = toroidal_Q(cp.asarray(chi_cpu), n_max=n_max, xp=cp)
    assert np.max(np.abs(cp.asnumpy(q_gpu) - q_cpu)) < 1e-10


@pytest.mark.parametrize("n_max", [0, 16, 69])
def test_gpu_derivatives_across_recurrence_branches(n_max):
    cp = pytest.importorskip("cupy")
    chi = np.array([[1.+1e-10, 1.001, 1.1, 3., 10., 100.],
                    [1.+1e-8, 1.002, 1.2, 4., 20., 200.]])
    # Noncontiguous input and both recurrence branches in the same call.
    q, dq = toroidal_Q(cp.asarray(chi).T, n_max, xp=cp)
    q_cpu, dq_cpu = toroidal_Q(chi.T, n_max)
    np.testing.assert_allclose(cp.asnumpy(q), q_cpu, rtol=2e-11, atol=1e-30)
    np.testing.assert_allclose(cp.asnumpy(dq), dq_cpu, rtol=2e-11, atol=1e-30)
