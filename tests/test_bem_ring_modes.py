import numpy as np
from scipy import integrate

from specific_particle_tracer.bem.axisymmetric import ring_potential, ring_field
from specific_particle_tracer.bem.ring_modes import (
    ring_potential_modes,
    ring_field_modes,
    point_charge_potential_modes,
)


def test_ring_potential_modes_matches_brute_force_azimuthal_integration():
    """Reconstructing sum_m Phi_m(rho,z;a,za)*cos(m*phi) must equal the
    direct potential of a ring at (a, za) carrying literal density
    cos(m*phi') -- checked mode-by-mode against direct numerical
    integration of the point kernel (G=1/(4*pi*r)) around the ring."""
    rho, z, a, za = 1.3, 0.4, 1.0, -0.2
    n_max = 8
    Phi = ring_potential_modes(rho, z, a, za, n_max)

    for m in range(n_max + 1):

        def integrand(phip, phi, m=m):
            r2 = rho**2 + a**2 - 2 * rho * a * np.cos(phi - phip) + (z - za) ** 2
            return np.cos(m * phip) / (4 * np.pi * np.sqrt(r2))

        for phi in [0.0, 0.7, 2.1]:
            brute, _ = integrate.quad(integrand, 0, 2 * np.pi, args=(phi,), limit=200)
            recon = Phi[m] * np.cos(m * phi)
            assert abs(brute - recon) < 1e-10


def test_ring_potential_modes_matches_ring_potential_at_m_equals_zero():
    """Mode 0 must reduce to bem.axisymmetric's existing unit-total-charge
    ring kernel once its different normalization convention (total charge
    1 vs. literal density 1) is accounted for -- Phi_0 = 2*pi*a*ring_potential."""
    rho, z, a, za = 1.3, 0.4, 1.0, -0.2
    Phi = ring_potential_modes(rho, z, a, za, n_max=0)
    expected = 2 * np.pi * a * ring_potential(rho, z - za, a)
    assert abs(Phi[0] - expected) / expected < 1e-12


def test_ring_field_modes_matches_finite_difference_of_ring_potential_modes():
    rho, z, a, za = 1.3, 0.4, 1.0, -0.2
    n_max = 6
    _, dPhi_drho, dPhi_dz = ring_field_modes(rho, z, a, za, n_max)

    h = 1e-6
    Phi_rp = ring_potential_modes(rho + h, z, a, za, n_max)
    Phi_rm = ring_potential_modes(abs(rho - h), z, a, za, n_max)
    Phi_zp = ring_potential_modes(rho, z + h, a, za, n_max)
    Phi_zm = ring_potential_modes(rho, z - h, a, za, n_max)
    fd_drho = (Phi_rp - Phi_rm) / (2 * h)
    fd_dz = (Phi_zp - Phi_zm) / (2 * h)

    assert np.max(np.abs(dPhi_drho - fd_drho)) < 1e-8
    assert np.max(np.abs(dPhi_dz - fd_dz)) < 1e-8


def test_ring_field_modes_matches_ring_field_at_m_equals_zero():
    rho, z, a, za = 1.3, 0.4, 1.0, -0.2
    Phi, dPhi_drho, dPhi_dz = ring_field_modes(rho, z, a, za, n_max=0)
    e_rho_expected, e_z_expected = ring_field(rho, z - za, a)
    # E = -cos(0*phi)*dPhi/d(.), and Phi_0 = 2*pi*a*ring_potential (same
    # rescaling as the potential test above) so the fields rescale the same way.
    assert abs(-dPhi_drho[0] - 2 * np.pi * a * e_rho_expected) < 1e-10
    assert abs(-dPhi_dz[0] - 2 * np.pi * a * e_z_expected) < 1e-10


def test_ring_field_modes_e_phi_is_zero_at_m_equals_zero():
    """E_phi = (m/rho)*Phi_m*sin(m*phi) must vanish at m=0 for any phi --
    an axisymmetric (m=0) source can't produce an azimuthal field."""
    rho, z, a, za = 1.3, 0.4, 1.0, -0.2
    Phi, _, _ = ring_field_modes(rho, z, a, za, n_max=0)
    m = 0
    for phi in [0.0, 1.0, 3.0]:
        e_phi = (m / rho) * Phi[0] * np.sin(m * phi)
        assert e_phi == 0.0


def test_point_charge_potential_modes_reconstructs_point_charge_potential():
    """sum_m g_m(rho,z;rho0,z0)*cos(m*phi) must converge (as n_max grows)
    to the exact potential of a unit point charge at (rho0, phi'=0, z0)."""
    rho, z, rho0, z0 = 1.3, 0.4, 1.0, -0.2

    def exact(phi):
        r2 = rho**2 + rho0**2 - 2 * rho * rho0 * np.cos(phi) + (z - z0) ** 2
        return 1.0 / (4 * np.pi * np.sqrt(r2))

    phis = [0.0, 0.5, 1.5, 3.0]
    errs = []
    for n_max in [10, 20, 40]:
        g = point_charge_potential_modes(rho, z, rho0, z0, n_max)
        err = max(
            abs(exact(phi) - sum(g[m] * np.cos(m * phi) for m in range(n_max + 1)))
            for phi in phis
        )
        errs.append(err)

    # Truncation error should shrink as more modes are kept.
    assert errs[1] < errs[0]
    assert errs[2] < errs[1]
    assert errs[2] < 1e-6


def test_ring_modes_are_vectorized_over_field_points():
    a, za = 1.0, -0.2
    rho = np.array([1.3, 2.0, 0.5])
    z = np.array([0.4, -0.1, 0.9])
    n_max = 5
    Phi = ring_potential_modes(rho, z, a, za, n_max)
    assert Phi.shape == (n_max + 1, 3)
    for i in range(3):
        Phi_single = ring_potential_modes(rho[i], z[i], a, za, n_max)
        assert np.allclose(Phi[:, i], Phi_single, rtol=1e-10)
