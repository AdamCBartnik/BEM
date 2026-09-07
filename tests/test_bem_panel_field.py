import numpy as np

from specific_particle_tracer.bem.mesh import sphere_mesh
from specific_particle_tracer.bem.panel_field import evaluate_coulomb_field, _closest_point_on_triangle


def test_uniform_charge_density_gives_the_coulomb_field_of_its_total_charge():
    """A uniform surface charge density sigma0 over a closed sphere is,
    from outside, indistinguishable from a point charge equal to the
    total charge sigma0 * surface_area -- the basic sanity check for this
    module's Coulomb-law panel integration."""
    R = 50e-9
    sigma0 = 3.0
    vertices, elements = sphere_mesh(R, n_theta=30, n_phi=36)
    sigma_nodal = np.full(vertices.shape[1], sigma0)

    r = 3.0 * R
    point = np.array([[0.0, 0.0, r]])
    E = evaluate_coulomb_field(vertices, elements, sigma_nodal, point)

    total_charge = sigma0 * 4.0 * np.pi * R**2
    E_expected = total_charge / (4.0 * np.pi * r**2)
    assert abs(E[0, 2] - E_expected) / E_expected < 1e-2
    assert np.allclose(E[0, :2], 0.0, atol=1e-6 * E_expected)


def test_closest_point_on_triangle_matches_brute_force():
    """Check _closest_point_on_triangle (used by evaluate_coulomb_field's
    near-surface regularization to find the true reference point on the
    mesh, not just the nearest vertex) against a brute-force dense sample
    of the triangle's interior, for points on both sides and off to the
    side of the triangle's plane."""
    rng = np.random.default_rng(0)
    a, b, c = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    # dense brute-force sample of the triangle (barycentric grid)
    n = 60
    us, vs = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    mask = us + vs <= 1.0
    us, vs = us[mask], vs[mask]
    ws = 1.0 - us - vs
    sample = us[:, None] * a + vs[:, None] * b + ws[:, None] * c

    for p in [
        np.array([0.2, 0.2, 0.5]),  # above the interior
        np.array([-0.5, -0.5, 0.1]),  # off past vertex a
        np.array([0.5, 0.5, -0.3]),  # off past the hypotenuse, below the plane
        np.array([2.0, 0.1, 0.0]),  # off past edge a-b, in-plane
    ]:
        point, bary = _closest_point_on_triangle(p, a, b, c)

        assert np.allclose(bary[0] * a + bary[1] * b + bary[2] * c, point)
        assert abs(sum(bary) - 1.0) < 1e-12
        assert all(w >= -1e-12 for w in bary)

        brute_dist = np.min(np.linalg.norm(sample - p, axis=1))
        assert abs(np.linalg.norm(point - p) - brute_dist) < 1.0 / n
